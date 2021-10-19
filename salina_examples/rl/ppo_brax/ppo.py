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
from salina import Agent
from brax.envs import _envs, create_gym_env
from brax.envs.to_torch import JaxToTorchWrapper
import torch.nn as nn

def make_brax_env(
    env_name
):
    e=create_gym_env(env_name)
    return JaxToTorchWrapper(e)

class BatchNormalizer(Agent):
    def __init__(self, env):
        super().__init__()
        env = make_brax_env(env.env_name)
        input_size = env.observation_space.shape[0]
        self.bn=nn.BatchNorm1d(input_size)

    def forward(self, t, **args):
        input = self.get(("env/env_obs", t))
        self.set(("env/env_obs", t), self.bn(input))

def clip_grad(parameters, grad):
    return (
        torch.nn.utils.clip_grad_norm_(parameters, grad)
        if grad > 0
        else torch.Tensor([0.0])
    )


def run_ppo(action_agent, critic_agent, logger,cfg):
    if cfg.algorithm.use_observation_normalizer:
        norm_agent=BatchNormalizer(cfg.algorithm.env)
    else:
        norm_agent=NoAgent()

    env_acquisition_agent = BraxAgent(env_name=cfg.algorithm.env.env_name,n_envs=cfg.algorithm.n_envs)


    acquisition_agent = TemporalAgent(
        Agents(env_acquisition_agent, norm_agent, action_agent)
    ).to(cfg.device)
    acquisition_agent.seed(cfg.algorithm.env_seed)
    workspace = Workspace()

    train_agent = Agents(action_agent, critic_agent).to(cfg.device)
    optimizer_policy = torch.optim.Adam(
        action_agent.parameters(), lr=cfg.algorithm.lr_policy
    )
    optimizer_critic = torch.optim.Adam(
        critic_agent.parameters(), lr=cfg.algorithm.lr_critic
    )

    env_validation_agent = BraxAgent(env_name=cfg.algorithm.validation.env.env_name,n_envs=cfg.algorithm.validation.n_envs)
    validation_agent = TemporalAgent(
        Agents(env_validation_agent, action_agent)
    ).to(cfg.device)
    validation_agent.seed(cfg.algorithm.validation.env_seed)
    validation_workspace = Workspace()

    # === Running algorithm
    epoch = 0
    iteration = 0
    nb_interactions = cfg.algorithm.n_envs * cfg.algorithm.n_timesteps
    print("[PPO] Learning")
    _epoch_start_time = time.time()
    while epoch < cfg.algorithm.max_epochs:
        # === Validation
        if (epoch % cfg.algorithm.validation.evaluate_every == 0) and (epoch > 0):
            validation_agent.eval()
            validation_agent(
                validation_workspace,
                t=0,
                stop_variable="env/done",
                replay=False,
                action_std=0.0,
            )
            length=validation_workspace["env/done"].float().argmax(0)
            creward = validation_workspace["env/cumulated_reward"][length].mean().item()
            logger.add_scalar("validation/reward", creward, epoch)
            print("reward at epoch", epoch, ":\t", round(creward, 0))
            validation_agent.train()

        # === Acquisition
        workspace.zero_grad()
        if epoch > 0:
            workspace.copy_n_last_steps(1)
        acquisition_agent(
            workspace,
            t=1 if epoch > 0 else 0,
            n_steps=cfg.algorithm.n_timesteps - 1
            if epoch > 0
            else cfg.algorithm.n_timesteps,
            replay=False,
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
            all_actions_lp=workspace["action_logprobs"].detach()

            for minibatch_idx in minibatches_idx:
                workspace.zero_grad()
                miniworkspace = workspace.select_batch(minibatch_idx)

                # === Update policy
                train_agent(
                    miniworkspace,
                    t=None,
                    replay=True,
                    action_std=cfg.algorithm.action_std,
                )
                critic, done, reward = miniworkspace["critic", "env/done", "env/reward"]
                old_action_lp = all_actions_lp[:,minibatch_idx].detach()
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

                td0 = RLF.temporal_difference(
                    critic, reward, done, cfg.algorithm.discount_factor
                )
                loss_critic = (td0 ** 2).mean()
                optimizer_critic.zero_grad()
                optimizer_policy.zero_grad()
                (loss_policy+loss_critic).backward()
                n = clip_grad(action_agent.parameters(), cfg.algorithm.clip_grad)
                optimizer_policy.step()
                optimizer_critic.step()
                logger.add_scalar("monitor/grad_norm_policy", n.item(), iteration)
                logger.add_scalar("loss/policy", loss_policy.item(), iteration)
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

    action_agent = instantiate_class(cfg.action_agent)
    critic_agent = instantiate_class(cfg.critic_agent)
    mp.set_start_method("spawn")
    logger=instantiate_class(cfg.logger)
    run_ppo(action_agent, critic_agent, logger,cfg)


if __name__ == "__main__":
    main()
