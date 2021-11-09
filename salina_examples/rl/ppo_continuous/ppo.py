#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import copy
import time

import gym
import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym.wrappers import TimeLimit
from omegaconf import DictConfig, OmegaConf

import salina
import salina.rl.functional as RLF
from salina import TAgent, Workspace, get_arguments, get_class, instantiate_class
from salina.agents import Agents, NRemoteAgent, TemporalAgent
from salina.agents.gyma import AutoResetGymAgent, GymAgent
from salina.logger import TFLogger


def _index(tensor_3d, tensor_2d):
    """This function is used to index a 3d tensors using a 2d tensor"""
    x, y, z = tensor_3d.size()
    t = tensor_3d.reshape(x * y, z)
    tt = tensor_2d.reshape(x * y)
    v = t[torch.arange(x * y), tt]
    v = v.reshape(x, y)
    return v


def _state_dict(agent, device):
    sd = agent.state_dict()
    for k, v in sd.items():
        sd[k] = v.to(device)
    return sd


def run_ppo(ppo_action_agent, ppo_critic_agent, logger, cfg):
    ppo_action_agent.set_name("ppo_action")
    env_agent = AutoResetGymAgent(
        get_class(cfg.algorithm.env),
        get_arguments(cfg.algorithm.env),
        n_envs=int(cfg.algorithm.n_envs / cfg.algorithm.n_processes),
    )

    acq_ppo_action = copy.deepcopy(ppo_action_agent)
    acq_agent = Agents(env_agent, acq_ppo_action)
    acq_agent = TemporalAgent(acq_agent)
    acq_remote_agent, acq_workspace = NRemoteAgent.create(
        acq_agent,
        num_processes=cfg.algorithm.n_processes,
        t=0,
        n_steps=cfg.algorithm.n_timesteps,
        stochastic=True,
        action_variance=0.0,
        replay=False,
    )
    acq_remote_agent.seed(cfg.algorithm.env_seed)

    tppo_action_agent = TemporalAgent(ppo_action_agent).to(cfg.algorithm.device)
    tppo_critic_agent = TemporalAgent(ppo_critic_agent).to(cfg.algorithm.device)

    optimizer_args = get_arguments(cfg.algorithm.optimizer)
    parameters = ppo_action_agent.parameters()
    optimizer_action = get_class(cfg.algorithm.optimizer)(parameters, **optimizer_args)
    parameters = ppo_critic_agent.parameters()
    optimizer_critic = get_class(cfg.algorithm.optimizer)(parameters, **optimizer_args)

    epoch = 0
    iteration = 0
    for epoch in range(cfg.algorithm.max_epochs):
        for a in acq_remote_agent.get_by_name("ppo_action"):
            a.load_state_dict(_state_dict(ppo_action_agent, "cpu"))

        if epoch > 0:
            acq_workspace.copy_n_last_steps(1)
            acq_remote_agent(
                acq_workspace,
                t=1,
                replay=False,
                n_steps=cfg.algorithm.n_timesteps - 1,
                action_variance=cfg.algorithm.action_variance,
            )
        else:
            acq_remote_agent(
                acq_workspace,
                t=0,
                replay=False,
                n_steps=cfg.algorithm.n_timesteps,
                action_variance=cfg.algorithm.action_variance,
            )

        replay_workspace = Workspace(acq_workspace).to(cfg.algorithm.device)

        with torch.no_grad():
            tppo_critic_agent(replay_workspace, t=0, n_steps=cfg.algorithm.n_timesteps)
        critic, done, reward, action = replay_workspace[
            "critic", "env/done", "env/reward", "action"
        ]

        gae = RLF.gae(
            critic, reward, done, cfg.algorithm.discount_factor, cfg.algorithm.gae
        ).detach()

        old_action_lp = replay_workspace["action_logprobs"]

        for _ in range(cfg.algorithm.pi_epochs):
            replay_workspace.zero_grad()
            tppo_action_agent(
                replay_workspace,
                replay=True,
                action_variance=cfg.algorithm.action_variance,
                t=0,
                n_steps=cfg.algorithm.n_timesteps,
            )
            action_lp = replay_workspace["action_logprobs"]
            entropy = replay_workspace["entropy"]
            ratio = (action_lp - old_action_lp).exp()
            ratio = ratio[:-1]
            clip_adv = (
                torch.clamp(
                    ratio, 1 - cfg.algorithm.clip_ratio, 1 + cfg.algorithm.clip_ratio
                )
                * gae
            )
            loss_pi = -(torch.min(ratio * gae, clip_adv)).mean()
            loss = loss_pi - cfg.algorithm.entropy_coef * entropy.mean()
            optimizer_action.zero_grad()
            loss.backward()
            if cfg.algorithm.clip_grad > 0:
                n = torch.nn.utils.clip_grad_norm_(
                    tppo_action_agent.parameters(), cfg.algorithm.clip_grad
                )
                logger.add_scalar("monitor/grad_norm_action", n.item(), iteration)

            optimizer_action.step()
            logger.add_scalar("loss_pi", loss_pi.item(), iteration)
            logger.add_scalar("loss_entropy", entropy.mean().item(), iteration)
            iteration += 1

        for _ in range(cfg.algorithm.v_epochs):
            replay_workspace.zero_grad()
            tppo_critic_agent(replay_workspace, t=0, n_steps=cfg.algorithm.n_timesteps)
            critic = replay_workspace["critic"]
            gae = RLF.gae(
                critic, reward, done, cfg.algorithm.discount_factor, cfg.algorithm.gae
            )
            optimizer_critic.zero_grad()
            loss = (gae ** 2).mean() * cfg.algorithm.critic_coef
            logger.add_scalar("loss_v", loss.item(), iteration)
            loss.backward()
            if cfg.algorithm.clip_grad > 0:
                n = torch.nn.utils.clip_grad_norm_(
                    tppo_critic_agent.parameters(), cfg.algorithm.clip_grad
                )
                logger.add_scalar("monitor/grad_norm_critic", n.item(), iteration)

            optimizer_critic.step()
            iteration += 1

        # Compute the cumulated reward on final_state
        creward = replay_workspace["env/cumulated_reward"]
        creward = creward[done]
        if creward.size()[0] > 0:
            logger.add_scalar("reward", creward.mean().item(), epoch)


@hydra.main(config_path=".", config_name="main.yaml")
def main(cfg):
    import torch.multiprocessing as mp

    logger = instantiate_class(cfg.logger)
    logger.save_hps(cfg)
    action_agent = instantiate_class(cfg.action_agent)
    critic_agent = instantiate_class(cfg.critic_agent)
    mp.set_start_method("spawn")
    run_ppo(action_agent, critic_agent, logger, cfg)


if __name__ == "__main__":
    main()
