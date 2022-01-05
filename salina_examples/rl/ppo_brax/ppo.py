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
import torch.nn as nn

import salina.rl.functional as RLF
from salina import Agent, Workspace, instantiate_class
from salina.agents import Agents, TemporalAgent
from salina.agents.brax import AutoResetBraxAgent,NoAutoResetBraxAgent
from salina.agents.gyma import AutoResetGymAgent,NoAutoResetGymAgent
from salina.logger import TFLogger
from salina_examples.rl.ppo_brax.agents import make_brax_env,make_gym_env,make_env
import numpy as np
import random

class Normalizer(Agent):
    def __init__(self, env):
        super().__init__()
        env = make_env(env)

        self.n_features = env.observation_space.shape[0]
        self.n = None

    def forward(self, t, update_normalizer=True, **kwargs):
        input = self.get(("env/env_obs", t))
        assert torch.isnan(input).sum() == 0.0, "problem"
        if update_normalizer:
            self.update(input)
        input = self.normalize(input)
        assert torch.isnan(input).sum() == 0.0, "problem"
        self.set(("env/env_obs", t), input)

    def update(self, x):
        if self.n is None:
            device = x.device
            self.n = torch.zeros(self.n_features).to(device)
            self.mean = torch.zeros(self.n_features).to(device)
            self.mean_diff = torch.zeros(self.n_features).to(device)
            self.var = torch.ones(self.n_features).to(device)
        self.n += 1.0
        last_mean = self.mean.clone()
        self.mean += (x - self.mean).mean(dim=0) / self.n
        self.mean_diff += (x - last_mean).mean(dim=0) * (x - self.mean).mean(dim=0)
        self.var = torch.clamp(self.mean_diff / self.n, min=1e-2)

    def normalize(self, inputs):
        obs_std = torch.sqrt(self.var)
        return (inputs - self.mean) / obs_std

    def seed(self, seed):
        torch.manual_seed(seed)


class NoAgent(Agent):
    def __init__(self):
        super().__init__()

    def forward(self, **kwargs):
        pass


def clip_grad(parameters, grad):
    return (
        torch.nn.utils.clip_grad_norm_(parameters, grad)
        if grad > 0
        else torch.Tensor([0.0])
    )


def run_ppo(action_agent, critic_agent, logger, cfg):
    if cfg.algorithm.use_observation_normalizer:
        # norm_agent=BatchNormalizer(cfg.algorithm.env,momentum=None)
        norm_agent = Normalizer(cfg.env)
    else:
        norm_agent = NoAgent()

    env_acquisition_agent=None
    env_validation_agent=None
    env_name=cfg.env.env_name
    if env_name.startswith("brax/"):
        env_name=env_name[5:]
        env_acquisition_agent = AutoResetBraxAgent(
            env_name=env_name, n_envs=cfg.algorithm.n_envs
        )
        env_validation_agent = NoAutoResetBraxAgent(
            env_name=env_name,
            n_envs=cfg.algorithm.validation.n_envs,
        )
    else:
        assert env_name.startswith("gym/")
        env_name=env_name[4:]
        env_acquisition_agent = AutoResetGymAgent(
            make_gym_env,
            {"env_name":env_name,"max_episode_steps":cfg.env.max_episode_steps},
            n_envs=cfg.algorithm.n_envs,
        )
        env_validation_agent = NoAutoResetGymAgent(
            make_gym_env,
            {"env_name":env_name,"max_episode_steps":cfg.env.max_episode_steps},
            n_envs=cfg.algorithm.validation.n_envs,
        )

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


    validation_agent = TemporalAgent(
        Agents(env_validation_agent, norm_agent, action_agent)
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
                update_normalizer=False,
            )
            length = validation_workspace["env/done"].float().argmax(0)
            arange = torch.arange(length.size()[0], device=length.device)
            creward = (
                validation_workspace["env/cumulated_reward"][length, arange]
                .mean()
                .item()
            )
            logger.add_scalar("validation/reward", creward, epoch)
            print("reward at epoch", epoch, ":\t", round(creward, 0))
            validation_agent.train()

        # === Acquisition
        workspace.zero_grad()
        if epoch > 0:
            workspace.copy_n_last_steps(1)
        acquisition_agent.train()
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

        workspace.zero_grad()

        #Saving acquisition action probabilities
        workspace.set_full("old_action_logprobs",workspace["action_logprobs"].detach())


        #Building mini workspaces
        #Learning for cfg.algorithm.update_epochs epochs
        for _ in range(cfg.algorithm.update_epochs):
            miniworkspaces=[]
            _stb=time.time()
            for _ in range(cfg.algorithm.n_mini_batches):
                miniworkspace=workspace.sample_subworkspace(cfg.algorithm.n_times_per_minibatch,cfg.algorithm.n_envs_per_minibatch,cfg.algorithm.n_timesteps_per_minibatch)
                miniworkspaces.append(miniworkspace)

            _etb=time.time()
            logger.add_scalar("monitor/minibatches_building_time",_etb-_stb,epoch)
            _B,_T=miniworkspaces[0].batch_size(),miniworkspaces[0].time_size()
            print("Resulting in ",len(miniworkspaces)," workspaces of size ",(_B,_T)," => ",_B*_T," time = ",_etb-_stb)

            random.shuffle(miniworkspaces)

            #Learning on batches
            for miniworkspace in miniworkspaces:
                # === Update policy
                train_agent(
                    miniworkspace,
                    t=None,
                    replay=True,
                    action_std=cfg.algorithm.action_std,
                )
                critic, done, reward = miniworkspace["critic", "env/done", "env/reward"]
                old_action_lp = miniworkspace["old_action_logprobs"]
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
                (loss_policy + loss_critic).backward()
                n = clip_grad(action_agent.parameters(), cfg.algorithm.clip_grad)
                optimizer_policy.step()
                optimizer_critic.step()
                logger.add_scalar("monitor/grad_norm_policy", n.item(), iteration)
                logger.add_scalar("loss/policy", loss_policy.item(), iteration)
                logger.add_scalar("loss/critic", loss_critic.item(), iteration)
                logger.add_scalar("monitor/grad_norm_critic", n.item(), iteration)
                iteration += 1
        epoch += 1


@hydra.main(config_path=".", config_name="pendulum.yaml")
def main(cfg):
    import torch.multiprocessing as mp

    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        v = torch.ones(1, device="cuda:0")

    action_agent = instantiate_class(cfg.action_agent)
    critic_agent = instantiate_class(cfg.critic_agent)
    mp.set_start_method("spawn")
    logger = instantiate_class(cfg.logger)
    run_ppo(action_agent, critic_agent, logger, cfg)


if __name__ == "__main__":
    main()
