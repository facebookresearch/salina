#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import copy
import time

import hydra
import torch

import salina.rl.functional as RLF
from salina import Workspace, instantiate_class
from salina.agents import Agents, TemporalAgent
from salina.agents.brax import BraxAgent
from salina.logger import TFLogger


def clip_grad(parameters, grad):
    return (
        torch.nn.utils.clip_grad_norm_(parameters, grad)
        if grad > 0
        else torch.Tensor([0.0])
    )


def run_ppo(policy_agent, critic_agent, cfg):
    logger = TFLogger(
        log_dir=cfg.logger.logdir,
        hps=cfg,
        cache_size=cfg.logger.cache_size,
        every_n_seconds=60,
        modulo=cfg.logger.modulo,
        verbose=cfg.logger.verbose,
    )

    # === Instantiate acquisition agent
    normalizer_agent = instantiate_class(cfg.normalizer_agent)
    env_acquisition_agent = BraxAgent(**cfg.acquisition.env)
    acquisition_agent = TemporalAgent(
        Agents(env_acquisition_agent, normalizer_agent, policy_agent)
    ).to(cfg.device)
    acquisition_agent.seed(cfg.acquisition.seed)
    workspace = Workspace().to(cfg.device)

    # === Instantiate training agent
    # train_policy_agent = TemporalAgent(policy_agent).to(cfg.device)
    # train_critic_agent = TemporalAgent(critic_agent).to(cfg.device)
    train_agent = Agents(policy_agent, critic_agent)
    optimizer_policy = torch.optim.Adam(
        policy_agent.parameters(), lr=cfg.algorithm.lr_policy
    )
    optimizer_critic = torch.optim.Adam(
        critic_agent.parameters(), lr=cfg.algorithm.lr_critic
    )

    # === Instantiate validation agent
    env_validation_agent = BraxAgent(**cfg.validation.env)
    validation_agent = TemporalAgent(
        Agents(env_validation_agent, normalizer_agent, policy_agent)
    )
    validation_agent.seed(cfg.validation.seed)
    validation_workspace = Workspace().to(cfg.device)

    # === Running algorithm
    epoch = 0
    iteration = 0
    nb_interactions = cfg.acquisition.env.n_envs * cfg.acquisition.n_timesteps
    print("[PPO] Learning")
    _epoch_start_time = time.time()
    while (time.time() - _epoch_start_time < cfg.algorithm.time_limit) and (
        epoch < cfg.algorithm.max_epochs
    ):

        # === Validation
        if (epoch % cfg.validation.evaluate_every == 0) and (epoch > 0):
            validation_agent(
                validation_workspace,
                t=0,
                n_steps=cfg.validation.env.episode_length + 1,
                replay=False,
                action_std=0.0,
            )
            creward, done = validation_workspace["env/cumulated_reward", "env/done"]
            creward = creward[done].mean().item()
            logger.add_scalar("validation/reward", creward, epoch)
            print("reward at epoch", epoch, ":\t", round(creward, 0))

        # === Acquisition
        if epoch > 0:
            workspace.copy_n_last_steps(1)
        acquisition_agent(
            workspace,
            t=1 if epoch > 0 else 0,
            n_steps=cfg.acquisition.n_timesteps - 1
            if epoch > 0
            else cfg.acquisition.n_timesteps,
            replay=False,
            update_normalizer=cfg.algorithm.normalize_obs,
            action_std=cfg.algorithm.action_std,
        )
        logger.add_scalar(
            "monitor/nb_interactions", (nb_interactions * (epoch + 1)), epoch
        )

        # === Update
        for _ in range(cfg.algorithm.update_epochs):
            minibatches_idx = (
                torch.randperm(workspace.batch_size())
                .to(cfg.device)
                .split(cfg.algorithm.minibatch_size)
            )

            for minibatch_idx in minibatches_idx:
                miniworkspace = workspace.select_batch(minibatch_idx)

                # === Update policy
                train_agent(
                    miniworkspace,
                    t=None,
                    replay=True,
                    action_std=cfg.algorithm.action_std,
                )
                critic, done, reward = miniworkspace["critic", "env/done", "env/reward"]
                old_action_lp = miniworkspace["old_action_logprobs"].detach()
                reward = reward * cfg.algorithm.reward_scaling
                gae = RLF.gae(
                    critic,
                    reward,
                    done,
                    cfg.algorithm.discount_factor,
                    cfg.algorithm.gae,
                ).detach()
                action_lp = miniworkspace["action_logprobs"]
                ratio = action_lp - old_action_lp
                ratio = ratio.exp()
                ratio = ratio[:-1]
                clip_adv = (
                    torch.clamp(
                        ratio,
                        1 - cfg.algorithm.clip_ratio,
                        1 + cfg.algorithm.clip_ratio,
                    )
                    * gae
                )
                loss_policy = -(torch.min(ratio * gae, clip_adv)).mean()
                optimizer_policy.zero_grad()
                loss_policy.backward()
                n = clip_grad(policy_agent.parameters(), cfg.algorithm.clip_grad)
                optimizer_policy.step()
                logger.add_scalar("monitor/grad_norm_policy", n.item(), iteration)
                logger.add_scalar("loss/policy", loss_policy.item(), iteration)

                # === Update critic
                td0 = RLF.temporal_difference(
                    critic, reward, done, cfg.algorithm.discount_factor
                )
                loss_critic = (td0 ** 2).mean()
                optimizer_critic.zero_grad()
                loss_critic.backward()
                n = clip_grad(critic_agent.parameters(), cfg.algorithm.clip_grad)
                optimizer_critic.step()
                logger.add_scalar("loss/critic", loss_critic.item(), iteration)
                logger.add_scalar("monitor/grad_norm_critic", n.item(), iteration)
                iteration += 1
        epoch += 1


@hydra.main(config_path=".", config_name="halfcheetah.yaml")
def main(cfg):
    import torch.multiprocessing as mp

    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        v = torch.ones(1, device="cuda:0")

    policy_agent = instantiate_class(cfg.policy_agent).to(cfg.device)
    critic_agent = instantiate_class(cfg.critic_agent).to(cfg.device)
    mp.set_start_method("spawn")
    run_ppo(policy_agent, critic_agent, cfg)


if __name__ == "__main__":
    main()
