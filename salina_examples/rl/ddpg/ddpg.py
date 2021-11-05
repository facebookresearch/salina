#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import copy
import math
import time

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

import salina
import salina.rl.functional as RLF
from salina import get_arguments, get_class, instantiate_class
from salina.agents import Agents, NRemoteAgent, TemporalAgent
from salina.agents.gyma import AutoResetGymAgent, GymAgent
from salina.logger import TFLogger
from salina.rl.replay_buffer import ReplayBuffer
from salina_examples.rl.ddpg.agents import OUNoise


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def _state_dict(agent, device):
    sd = agent.state_dict()
    for k, v in sd.items():
        sd[k] = v.to(device)
    return sd


def run_ddpg(q_agent, action_agent, logger, cfg):
    q_agent.set_name("q_agent")
    action_agent.set_name("action_agent")
    modulo = 1

    env_agent = AutoResetGymAgent(
        get_class(cfg.algorithm.env),
        get_arguments(cfg.algorithm.env),
        n_envs=int(cfg.algorithm.n_envs / cfg.algorithm.n_processes),
    )

    q_target_agent = copy.deepcopy(q_agent)  # Create target agent
    action_target_agent = copy.deepcopy(action_agent)

    acq_agent = TemporalAgent(Agents(env_agent, copy.deepcopy(action_agent)))
    acq_remote_agent, acq_workspace = NRemoteAgent.create(
        acq_agent,
        num_processes=cfg.algorithm.n_processes,
        t=0,
        n_steps=cfg.algorithm.n_timesteps,
        epsilon=1.0,
    )
    acq_remote_agent.seed(cfg.algorithm.env_seed)

    # == Setting up the training agents
    train_temporal_q_agent = TemporalAgent(q_agent)
    train_temporal_action_agent = TemporalAgent(action_agent)
    train_temporal_q_target_agent = TemporalAgent(q_target_agent)
    train_temporal_action_target_agent = TemporalAgent(action_target_agent)

    train_temporal_q_agent.to(cfg.algorithm.loss_device)
    train_temporal_action_agent.to(cfg.algorithm.loss_device)
    train_temporal_q_target_agent.to(cfg.algorithm.loss_device)
    train_temporal_action_target_agent.to(cfg.algorithm.loss_device)

    replay_buffer = ReplayBuffer(cfg.algorithm.buffer_size)

    acq_remote_agent(
        acq_workspace,
        t=0,
        n_steps=cfg.algorithm.n_timesteps,
        epsilon=cfg.algorithm.action_noise,
    )
    replay_buffer.put(acq_workspace, time_size=cfg.algorithm.buffer_time_size)
    logger.message("[DDQN] Initializing replay buffer")
    while replay_buffer.size() < cfg.algorithm.initial_buffer_size:
        acq_workspace.copy_n_last_steps(cfg.algorithm.overlapping_timesteps)
        acq_remote_agent(
            acq_workspace,
            t=cfg.algorithm.overlapping_timesteps,
            n_steps=cfg.algorithm.n_timesteps - cfg.algorithm.overlapping_timesteps,
            epsilon=cfg.algorithm.action_noise,
        )
        replay_buffer.put(acq_workspace, time_size=cfg.algorithm.buffer_time_size)

    print("[DDPG] Learning")
    _epoch_start_time = time.time()
    optimizer_args = get_arguments(cfg.algorithm.optimizer)
    optimizer_q = get_class(cfg.algorithm.optimizer)(
        q_agent.parameters(), **optimizer_args
    )

    optimizer_action = get_class(cfg.algorithm.optimizer)(
        action_agent.parameters(), **optimizer_args
    )

    for epoch in range(cfg.algorithm.max_epoch):
        for a in acq_remote_agent.get_by_name("action_agent"):
            a.load_state_dict(_state_dict(action_agent, "cpu"))

        acq_workspace.copy_n_last_steps(cfg.algorithm.overlapping_timesteps)
        acq_remote_agent(
            acq_workspace,
            t=cfg.algorithm.overlapping_timesteps,
            n_steps=cfg.algorithm.n_timesteps - cfg.algorithm.overlapping_timesteps,
            epsilon=cfg.algorithm.action_noise,
        )
        replay_buffer.put(acq_workspace, time_size=cfg.algorithm.buffer_time_size)

        # Get cumulated reward over terminated episodes
        done, creward = acq_workspace["env/done", "env/cumulated_reward"]
        creward = creward[done]
        if creward.size()[0] > 0:
            logger.add_scalar("monitor/reward", creward.mean().item(), epoch)
        logger.add_scalar("monitor/replay_buffer_size", replay_buffer.size(), epoch)

        # Inner loop to minimize the TD
        for inner_epoch in range(cfg.algorithm.inner_epochs):
            batch_size = cfg.algorithm.batch_size
            replay_workspace = replay_buffer.get(batch_size).to(
                cfg.algorithm.loss_device
            )
            done, reward = replay_workspace["env/done", "env/reward"]

            train_temporal_q_agent(
                replay_workspace,
                t=0,
                n_steps=cfg.algorithm.buffer_time_size,
                detach_action=True,
            )
            q = replay_workspace["q"].squeeze(-1)

            with torch.no_grad():
                train_temporal_action_target_agent(
                    replay_workspace,
                    t=0,
                    n_steps=cfg.algorithm.buffer_time_size,
                    epsilon=0.0,
                )  # epsilon = cfg.algorithm.target_noise
                train_temporal_q_target_agent(
                    replay_workspace, t=0, n_steps=cfg.algorithm.buffer_time_size
                )
                q_target = replay_workspace["q"]

            q_target = q_target.squeeze(-1)
            target = (
                reward[1:]
                + cfg.algorithm.discount_factor
                * (1.0 - done[1:].float())
                * q_target[1:]
            )

            td = q[:-1] - target
            error = td ** 2

            # Add burning for the first timesteps in the trajectories => no gradient (commonly done in R2D2)
            burning = torch.zeros_like(error)
            burning[cfg.algorithm.burning_timesteps :] = 1.0
            error = error * burning
            loss = error.mean()
            logger.add_scalar("loss/td_loss", error.mean().item(), epoch)
            optimizer_q.zero_grad()
            loss.backward()

            if cfg.algorithm.clip_grad > 0:
                n = torch.nn.utils.clip_grad_norm_(
                    q_agent.parameters(), cfg.algorithm.clip_grad
                )
                logger.add_scalar("monitor/grad_norm_q", n.item(), epoch)

            optimizer_q.step()

            # Update policy
            train_temporal_action_agent(
                replay_workspace,
                epsilon=0.0,
                t=0,
                n_steps=cfg.algorithm.buffer_time_size,
            )
            train_temporal_q_agent(
                replay_workspace, t=0, n_steps=cfg.algorithm.buffer_time_size
            )
            q = replay_workspace["q"].squeeze(-1)
            burning = torch.zeros_like(q)
            burning[cfg.algorithm.burning_timesteps :] = 1.0
            q = q * burning
            q = q[:-1]
            optimizer_action.zero_grad()
            loss = -q.mean()
            loss.backward()

            if cfg.algorithm.clip_grad > 0:
                n = torch.nn.utils.clip_grad_norm_(
                    action_agent.parameters(), cfg.algorithm.clip_grad
                )
                logger.add_scalar("monitor/grad_norm_action", n.item(), epoch)

            logger.add_scalar("loss/q_loss", loss.item(), epoch)
            optimizer_action.step()

            # Soft update of target policy and q function
            tau = cfg.algorithm.update_target_tau
            soft_update_params(q_agent, q_target_agent, tau)
            soft_update_params(action_agent, action_target_agent, tau)


@hydra.main(config_path=".", config_name="gym.yaml")
def main(cfg):
    import torch.multiprocessing as mp

    mp.set_start_method("spawn")
    logger = instantiate_class(cfg.logger)
    logger.save_hps(cfg)
    q_agent = instantiate_class(cfg.q_agent)
    action_agent = instantiate_class(cfg.action_agent)
    run_ddpg(q_agent, action_agent, logger, cfg)


if __name__ == "__main__":
    main()
