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
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

import salina
import salina.rl.functional as RLF
from salina import get_arguments, get_class, instantiate_class
from salina.agents import Agents, RemoteAgent, TemporalAgent
from salina.agents.gym import AutoResetGymAgent, GymAgent
from salina.logger import TFLogger
from salina.rl.replay_buffer import ReplayBuffer
from salina.salina_examples.rl.ddpg.agents import OUNoise


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def _state_dict(agent, device):
    sd = agent.state_dict()
    for k, v in sd.items():
        sd[k] = v.to(device)
    return sd


def run_ddpg(q_agent, action_agent, logger, cfg):
    modulo = 1

    env_agent = AutoResetGymAgent(
        get_class(cfg.algorithm.env), get_arguments(cfg.algorithm.env)
    )

    q_target_agent = copy.deepcopy(q_agent)  # Create target agent
    action_target_agent = copy.deepcopy(action_agent)

    action_acquisition_agent = copy.deepcopy(action_agent)

    # == Setting up the training agents
    train_temporal_q_agent = TemporalAgent(q_agent)
    train_temporal_action_agent = TemporalAgent(action_agent)
    train_temporal_q_target_agent = TemporalAgent(q_target_agent)
    train_temporal_action_target_agent = TemporalAgent(action_target_agent)

    train_temporal_q_agent.to(cfg.algorithm.loss_device)
    train_temporal_action_agent.to(cfg.algorithm.loss_device)
    train_temporal_q_target_agent.to(cfg.algorithm.loss_device)
    train_temporal_action_target_agent.to(cfg.algorithm.loss_device)

    remote_train_agent = RemoteAgent(
        TemporalAgent(Agents(env_agent, action_acquisition_agent)),
        num_processes=cfg.algorithm.n_processes,
    )
    remote_train_agent.seed(cfg.algorithm.env_seed)

    workspace = salina.Workspace(
        batch_size=cfg.algorithm.n_envs,
        time_size=cfg.algorithm.n_timesteps,
    )
    workspace = remote_train_agent(workspace, epsilon=cfg.algorithm.action_noise)

    # == Setting up & initializing the replay buffer for DQN
    replay_buffer = ReplayBuffer(cfg.algorithm.buffer_size)
    # replay_buffer.put(workspace)

    print("[DDPG] Initializing replay buffer")
    while replay_buffer.size() < cfg.algorithm.initial_buffer_size:
        ts = workspace.time_size()
        for tt in range(cfg.algorithm.overlapping_timesteps):
            workspace.copy_time(
                from_time=ts - cfg.algorithm.overlapping_timesteps + tt,
                to_time=tt,
            )
        workspace = remote_train_agent(
            workspace,
            t=cfg.algorithm.overlapping_timesteps,
            epsilon=cfg.algorithm.action_noise,
        )
        replay_buffer.put(workspace, time_size=cfg.algorithm.buffer_time_size)

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
        ts = workspace.time_size()

        # For having sliding windows
        for tt in range(cfg.algorithm.overlapping_timesteps):
            workspace.copy_time(
                from_time=ts - cfg.algorithm.overlapping_timesteps + tt,
                to_time=tt,
            )

        workspace = remote_train_agent(
            workspace,
            t=cfg.algorithm.overlapping_timesteps,
            epsilon=cfg.algorithm.action_noise,
        )
        done, cumulated_reward, timestep = workspace[
            "env/done", "env/cumulated_reward", "env/timestep"
        ]
        end_reward = cumulated_reward[done]
        if end_reward.size()[0] > 0:
            logger.add_scalar(
                "monitor/training_cumulated_reward", end_reward.mean().item(), epoch
            )

        replay_buffer.put(workspace, time_size=cfg.algorithm.buffer_time_size)

        logger.add_scalar("monitor/replay_buffer_size", replay_buffer.size(), epoch)

        # Inner loop to minimize the TD
        for inner_epoch in range(cfg.algorithm.inner_epochs):
            batch_size = cfg.algorithm.batch_size
            replay_workspace = replay_buffer.get(batch_size).to(
                cfg.algorithm.loss_device
            )
            done, reward = replay_workspace["env/done", "env/reward"]

            replay_workspace = train_temporal_q_agent(
                replay_workspace, detach_action=True
            )
            q = replay_workspace["q"].squeeze(-1)

            with torch.no_grad():
                replay_workspace = train_temporal_action_target_agent(
                    replay_workspace,
                    epsilon=0.0,
                )  # epsilon = cfg.algorithm.target_noise
                replay_workspace = train_temporal_q_target_agent(replay_workspace)
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
            replay_workspace = train_temporal_action_agent(
                replay_workspace, epsilon=0.0
            )
            replay_workspace = train_temporal_q_agent(replay_workspace)
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

        action_acquisition_agent.load_state_dict(_state_dict(action_agent, "cpu"))


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
