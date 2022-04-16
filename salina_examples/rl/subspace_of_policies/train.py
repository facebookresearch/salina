#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import random
import hydra
import os
import copy
import torch
from salina import Workspace, instantiate_class
import salina.rl.functional as RLF
from salina.agents import Agents, TemporalAgent
from salina_examples.rl.subspace_of_policies.agents import AlphaAgent, Normalizer, CustomBraxAgent

def clip_grad(parameters, grad):
    return (torch.nn.utils.clip_grad_norm_(parameters, grad) if grad > 0 else torch.Tensor([0.0]))

def run_line_ppo(action_agent, critic_agent, logger, cfg):

    # Initializing acquisition agent
    normalizer_agent = Normalizer(cfg.env_name)
    env_agent = CustomBraxAgent(cfg.acquisition.n_envs, **cfg.acquisition.env)
    alpha_agent = AlphaAgent(cfg.device, cfg.algorithm.n_models, cfg.algorithm.geometry, cfg.algorithm.distribution)
    acquisition_agent = TemporalAgent(Agents(env_agent, normalizer_agent, alpha_agent, action_agent)).to(cfg.device)
    acquisition_agent.seed(cfg.acquisition.seed)

    # Initializing validation agent
    env_agent = CustomBraxAgent(cfg.validation.n_envs, **cfg.validation.env)
    validation_agent = TemporalAgent(Agents(env_agent, normalizer_agent, alpha_agent, action_agent)).to(cfg.device)
    validation_agent.seed(cfg.validation.seed)
    validation_workspace = Workspace()

    # Initializing train agent and optimizers
    train_agent = Agents(action_agent, critic_agent).to(cfg.device)
    optimizer_policy = torch.optim.Adam(action_agent.parameters(), lr=cfg.algorithm.lr_policy)
    optimizer_critic = torch.optim.Adam(critic_agent.parameters(), lr=cfg.algorithm.lr_critic)

    # Running algorithm
    epoch = 0
    iteration = 0
    n_interactions = 0
    workspace = Workspace()

    while (n_interactions < cfg.algorithm.max_interactions):
        # Evaluating the training policy
        if (epoch % cfg.validation.evaluate_every == 0) and (epoch > 0):
            validation_agent.eval()
            validation_agent(validation_workspace, t=0, stop_variable="env/done", replay=False, action_std=0.0, update_normalizer=False)
            creward, done = validation_workspace["env/cumulated_reward", "env/done"]
            creward = creward[done].reshape(cfg.validation.n_envs)
            logger.add_scalar("reward/mean", creward.mean().item(), n_interactions)
            logger.add_scalar("reward/max", creward.max().item(), n_interactions)
            print("reward at epoch", epoch, ":\t", round(creward.mean().item(), 0))

        # Acquisition of trajectories
        if epoch > 0: workspace.copy_n_last_steps(1)
        acquisition_agent.eval()
        acquisition_agent(workspace, 
                          t=1 if epoch > 0 else 0, 
                          n_steps = cfg.acquisition.n_timesteps - 1 if epoch > 0 else cfg.acquisition.n_timesteps,
                          replay = False,
                          action_std = cfg.algorithm.action_std)
        workspace.set_full("acquisition_action_logprobs",workspace["action_logprobs"].detach())
        #workspace.set_full("acquisition_action",workspace["action"].detach())
        #workspace.set_full("env/normalized_env_obs",workspace["env/normalized_env_obs"].detach())
        n_interactions+=(workspace.time_size()-1)*workspace.batch_size()
        logger.add_scalar("monitor/n_interactions", n_interactions, epoch)

        workspace.zero_grad()
        miniworkspaces=[]
        for _ in range(cfg.algorithm.n_minibatches):
            miniworkspace=workspace.sample_subworkspace(1,cfg.algorithm.minibatch_size,cfg.algorithm.n_timesteps_per_minibatch)
            miniworkspaces.append(miniworkspace)

        while len(miniworkspaces) > 0:
            miniworkspace = miniworkspaces.pop()
            old_action_lp = miniworkspace["acquisition_action_logprobs"]
            train_agent.train()
            train_agent(miniworkspace, t=None, replay = True, action_std=cfg.algorithm.action_std)
            critic, done, reward = miniworkspace["critic", "env/done", "env/reward"]
            reward = reward * cfg.algorithm.reward_scaling

            # === Update policy
            if (iteration % cfg.algorithm.policy_update_delay) == 0:
                gae = RLF.gae(critic,reward,done,cfg.algorithm.discount_factor,cfg.algorithm.gae).detach()
                action_lp = miniworkspace["action_logprobs"]
                ratio = action_lp - old_action_lp
                ratio = ratio.exp()
                ratio = ratio[:-1]
                clip_adv = torch.clamp(ratio,1 - cfg.algorithm.clip_ratio,1 + cfg.algorithm.clip_ratio) * gae
                j,k = random.sample(range(cfg.algorithm.n_models),2)
                penalty = action_agent.cosine_similarity(j,k)
                loss_policy = -(torch.min(ratio * gae, clip_adv)).mean()
                loss = loss_policy + penalty * cfg.algorithm.beta
                optimizer_policy.zero_grad()
                loss.backward()
                n = clip_grad(action_agent.parameters(), cfg.algorithm.clip_grad)
                optimizer_policy.step()
                logger.add_scalar("monitor/grad_norm_policy", n.item(), iteration)
                logger.add_scalar("loss/policy", loss_policy.item(), iteration)
                logger.add_scalar("loss/cosine_penalty", penalty.item(), iteration)

            # === Update critic
            td0 = RLF.temporal_difference(critic, reward, done, cfg.algorithm.discount_factor)
            loss_critic = (td0 ** 2).mean()
            optimizer_critic.zero_grad()
            loss_critic.backward()
            n = clip_grad(critic_agent.parameters(), cfg.algorithm.clip_grad)
            optimizer_critic.step()
            logger.add_scalar("loss/critic", loss_critic.item(), iteration)
            logger.add_scalar("monitor/grad_norm_critic", n.item(), iteration)
            iteration += 1
        epoch += 1

    # Saving model
    if cfg.save_model:
        os.makedirs(os.getcwd() +"/model")
        torch.save(action_agent.state_dict(),os.getcwd() +"/model/policy")
        torch.save(normalizer_agent.state_dict(),os.getcwd() +"/model/normalizer")
                
@hydra.main(config_path="configs/", config_name="halfcheetah.yaml")
def main(cfg):
    import torch.multiprocessing as mp
    # For initializing brax
    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        v = torch.ones(1, device=cfg.device)
    action_agent = instantiate_class(cfg.action_agent).to(cfg.device)
    critic_agent = instantiate_class(cfg.critic_agent).to(cfg.device)
    logger = instantiate_class(cfg.logger)
    mp.set_start_method("spawn")
    run_line_ppo(action_agent, critic_agent, logger, cfg)

if __name__ == "__main__":
    main()