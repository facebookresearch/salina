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
from salina import Workspace, get_arguments, get_class, instantiate_class
from salina.agents import Agents, NRemoteAgent, RemoteAgent, TemporalAgent
from salina.agents.gyma import AutoResetGymAgent
from salina.logger import TFLogger
from salina.rl.replay_buffer import ReplayBuffer


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def _state_dict(agent, device):
    sd = agent.state_dict()
    for k, v in sd.items():
        sd[k] = v.to(device)
    return sd


def run_dqn(q_agent, logger, cfg):
    q_agent.set_name("q_agent")
    env_agent = AutoResetGymAgent(
        get_class(cfg.algorithm.env),
        get_arguments(cfg.algorithm.env),
        n_envs=int(cfg.algorithm.n_envs / cfg.algorithm.n_processes),
    )

    q_target_agent = copy.deepcopy(q_agent)

    acq_agent = TemporalAgent(Agents(env_agent, copy.deepcopy(q_agent)))
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
    train_temporal_q_target_agent = TemporalAgent(q_target_agent)
    train_temporal_q_agent.to(cfg.algorithm.loss_device)
    train_temporal_q_target_agent.to(cfg.algorithm.loss_device)

    replay_buffer = ReplayBuffer(cfg.algorithm.buffer_size)
    acq_remote_agent(acq_workspace, t=0, n_steps=cfg.algorithm.n_timesteps, epsilon=1.0)
    replay_buffer.put(acq_workspace, time_size=cfg.algorithm.buffer_time_size)
    logger.message("[DDQN] Initializing replay buffer")
    while replay_buffer.size() < cfg.algorithm.initial_buffer_size:
        acq_workspace.copy_n_last_steps(cfg.algorithm.overlapping_timesteps)
        acq_remote_agent(
            acq_workspace,
            t=cfg.algorithm.overlapping_timesteps,
            n_steps=cfg.algorithm.n_timesteps - cfg.algorithm.overlapping_timesteps,
            epsilon=1.0,
        )
        replay_buffer.put(acq_workspace, time_size=cfg.algorithm.buffer_time_size)

    logger.message("[DDQN] Learning")
    epsilon_by_epoch = lambda epoch: cfg.algorithm.epsilon_final + (
        cfg.algorithm.epsilon_start - cfg.algorithm.epsilon_final
    ) * math.exp(-1.0 * epoch / cfg.algorithm.epsilon_exploration_decay)

    optimizer_args = get_arguments(cfg.algorithm.optimizer)
    optimizer = get_class(cfg.algorithm.optimizer)(
        q_agent.parameters(), **optimizer_args
    )
    iteration = 0
    for epoch in range(cfg.algorithm.max_epoch):
        epsilon = epsilon_by_epoch(epoch)
        logger.add_scalar("monitor/epsilon", epsilon, iteration)

        for a in acq_remote_agent.get_by_name("q_agent"):
            a.load_state_dict(_state_dict(q_agent, "cpu"))

        acq_workspace.copy_n_last_steps(cfg.algorithm.overlapping_timesteps)
        acq_remote_agent(
            acq_workspace,
            t=cfg.algorithm.overlapping_timesteps,
            n_steps=cfg.algorithm.n_timesteps - cfg.algorithm.overlapping_timesteps,
            epsilon=epsilon,
        )
        replay_buffer.put(acq_workspace, time_size=cfg.algorithm.buffer_time_size)

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
            # Batch size + Time_size
            action = replay_workspace["action"]
            train_temporal_q_agent(
                replay_workspace,
                t=0,
                n_steps=cfg.algorithm.buffer_time_size,
                replay=True,
                epsilon=0.0,
            )
            q, done, reward = replay_workspace["q", "env/done", "env/reward"]

            with torch.no_grad():
                train_temporal_q_target_agent(
                    replay_workspace,
                    t=0,
                    n_steps=cfg.algorithm.buffer_time_size,
                    replay=True,
                    epsilon=0.0,
                )
                q_target = replay_workspace["q"]

            td = RLF.doubleqlearning_temporal_difference(
                q,
                action,
                q_target,
                reward,
                done,
                cfg.algorithm.discount_factor,
            )
            error = td ** 2

            # Add burning steps for the first timesteps in the trajectories (for recurrent policies)
            burning = torch.zeros_like(td)
            burning[cfg.algorithm.burning_timesteps :] = 1.0
            error = error * burning
            loss = error.mean()
            logger.add_scalar("loss/q_loss", loss.item(), iteration)

            optimizer.zero_grad()
            loss.backward()

            if cfg.algorithm.clip_grad > 0:
                n = torch.nn.utils.clip_grad_norm_(
                    q_agent.parameters(), cfg.algorithm.clip_grad
                )
                logger.add_scalar("monitor/grad_norm", n.item(), iteration)
            optimizer.step()
            iteration += 1

        # Update of the target network
        if cfg.algorithm.hard_target_update:
            if epoch % cfg.algorithm.update_target_epochs == 0:
                q_target_agent.load_state_dict(q_agent.state_dict())
        else:
            tau = cfg.algorithm.update_target_tau
            soft_update_params(q_agent, q_target_agent, tau)


@hydra.main(config_path=".", config_name="gym.yaml")
def main(cfg):
    import torch.multiprocessing as mp

    mp.set_start_method("spawn")
    logger = instantiate_class(cfg.logger)
    logger.save_hps(cfg)

    q_agent = instantiate_class(cfg.q_agent)
    run_dqn(q_agent, logger, cfg)


if __name__ == "__main__":
    OmegaConf.register_new_resolver("plus", lambda x, y: x + y)
    OmegaConf.register_new_resolver("n_gpus", lambda x: 0 if x == "cpu" else 1)
    main()
