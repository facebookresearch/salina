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
from salina.agents import Agents, NRemoteAgent, TemporalAgent
from salina.agents.gyma import AutoResetGymAgent, GymAgent
from salina.logger import TFLogger
from salina.rl.replay_buffer import ReplayBuffer
from salina_examples import weight_init


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def _state_dict(agent, device):
    sd = agent.state_dict()
    for k, v in sd.items():
        sd[k] = v.to(device)
    return sd


def run_td3(q_agent_1, q_agent_2, action_agent, logger, cfg):
    action_agent.set_name("action_agent")
    env_agent = AutoResetGymAgent(
        get_class(cfg.algorithm.env),
        get_arguments(cfg.algorithm.env),
        n_envs=int(cfg.algorithm.n_envs / cfg.algorithm.n_processes),
    )
    q_target_agent_1 = copy.deepcopy(q_agent_1)
    q_target_agent_2 = copy.deepcopy(q_agent_2)
    action_target_agent = copy.deepcopy(action_agent)

    acq_action_agent = copy.deepcopy(action_agent)
    acq_agent = TemporalAgent(Agents(env_agent, acq_action_agent))
    acq_remote_agent, acq_workspace = NRemoteAgent.create(
        acq_agent,
        num_processes=cfg.algorithm.n_processes,
        t=0,
        n_steps=cfg.algorithm.n_timesteps,
        epsilon=1.0,
    )
    acq_remote_agent.seed(cfg.algorithm.env_seed)

    # == Setting up the training agents
    train_temporal_q_agent_1 = TemporalAgent(q_agent_1)
    train_temporal_q_agent_2 = TemporalAgent(q_agent_2)
    train_temporal_action_agent = TemporalAgent(action_agent)
    train_temporal_q_target_agent_1 = TemporalAgent(q_target_agent_1)
    train_temporal_q_target_agent_2 = TemporalAgent(q_target_agent_2)
    train_temporal_action_target_agent = TemporalAgent(action_target_agent)

    train_temporal_q_agent_1.to(cfg.algorithm.loss_device)
    train_temporal_q_agent_2.to(cfg.algorithm.loss_device)
    train_temporal_action_agent.to(cfg.algorithm.loss_device)
    train_temporal_q_target_agent_1.to(cfg.algorithm.loss_device)
    train_temporal_q_target_agent_2.to(cfg.algorithm.loss_device)
    train_temporal_action_target_agent.to(cfg.algorithm.loss_device)

    acq_remote_agent(
        acq_workspace,
        t=0,
        n_steps=cfg.algorithm.n_timesteps,
        epsilon=cfg.algorithm.action_noise,
    )

    # == Setting up & initializing the replay buffer for DQN
    replay_buffer = ReplayBuffer(cfg.algorithm.buffer_size)
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

    logger.message("[DDQN] Learning")
    n_interactions = 0
    optimizer_args = get_arguments(cfg.algorithm.optimizer)
    optimizer_q_1 = get_class(cfg.algorithm.optimizer)(
        q_agent_1.parameters(), **optimizer_args
    )
    optimizer_q_2 = get_class(cfg.algorithm.optimizer)(
        q_agent_2.parameters(), **optimizer_args
    )
    optimizer_action = get_class(cfg.algorithm.optimizer)(
        action_agent.parameters(), **optimizer_args
    )
    iteration = 0
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

        done, creward = acq_workspace["env/done", "env/cumulated_reward"]
        creward = creward[done]
        if creward.size()[0] > 0:
            logger.add_scalar("monitor/reward", creward.mean().item(), epoch)
        logger.add_scalar("monitor/replay_buffer_size", replay_buffer.size(), epoch)

        n_interactions += (
            acq_workspace.time_size() - cfg.algorithm.overlapping_timesteps
        ) * acq_workspace.batch_size()
        logger.add_scalar("monitor/n_interactions", n_interactions, epoch)

        for inner_epoch in range(cfg.algorithm.inner_epochs):
            batch_size = cfg.algorithm.batch_size
            replay_workspace = replay_buffer.get(batch_size).to(
                cfg.algorithm.loss_device
            )
            done, reward = replay_workspace["env/done", "env/reward"]

            train_temporal_q_agent_1(
                replay_workspace,
                t=0,
                n_steps=cfg.algorithm.buffer_time_size,
                detach_action=True,
            )
            q_1 = replay_workspace["q"].squeeze(-1)
            train_temporal_q_agent_2(
                replay_workspace,
                t=0,
                n_steps=cfg.algorithm.buffer_time_size,
                detach_action=True,
            )
            q_2 = replay_workspace["q"].squeeze(-1)

            with torch.no_grad():
                train_temporal_action_target_agent(
                    replay_workspace,
                    t=0,
                    n_steps=cfg.algorithm.buffer_time_size,
                    epsilon=cfg.algorithm.target_noise,
                    epsilon_clip=cfg.algorithm.noise_clip,
                )
                train_temporal_q_target_agent_1(
                    replay_workspace,
                    t=0,
                    n_steps=cfg.algorithm.buffer_time_size,
                )
                q_target_1 = replay_workspace["q"]
                train_temporal_q_target_agent_2(
                    replay_workspace,
                    t=0,
                    n_steps=cfg.algorithm.buffer_time_size,
                )
                q_target_2 = replay_workspace["q"]

            q_target = torch.min(q_target_1, q_target_2).squeeze(-1)
            target = (
                reward[1:]
                + cfg.algorithm.discount_factor
                * (1.0 - done[1:].float())
                * q_target[1:]
            )

            td_1 = q_1[:-1] - target
            td_2 = q_2[:-1] - target
            error_1 = td_1 ** 2
            error_2 = td_2 ** 2

            burning = torch.zeros_like(error_1)
            burning[cfg.algorithm.burning_timesteps :] = 1.0
            error_1 = error_1 * burning
            error_2 = error_2 * burning
            error = error_1 + error_2
            loss = error.mean()
            logger.add_scalar("loss/td_loss_1", error_1.mean().item(), iteration)
            logger.add_scalar("loss/td_loss_2", error_2.mean().item(), iteration)
            optimizer_q_1.zero_grad()
            optimizer_q_2.zero_grad()
            loss.backward()

            if cfg.algorithm.clip_grad > 0:
                n = torch.nn.utils.clip_grad_norm_(
                    q_agent_1.parameters(), cfg.algorithm.clip_grad
                )
                logger.add_scalar("monitor/grad_norm_q_1", n.item(), iteration)
                n = torch.nn.utils.clip_grad_norm_(
                    q_agent_2.parameters(), cfg.algorithm.clip_grad
                )
                logger.add_scalar("monitor/grad_norm_q_2", n.item(), iteration)

            optimizer_q_1.step()
            optimizer_q_2.step()

            if inner_epoch % cfg.algorithm.policy_delay:
                train_temporal_action_agent(
                    replay_workspace,
                    epsilon=0.0,
                    t=0,
                    n_steps=cfg.algorithm.buffer_time_size,
                )
                train_temporal_q_agent_1(
                    replay_workspace,
                    t=0,
                    n_steps=cfg.algorithm.buffer_time_size,
                )
                q = replay_workspace["q"].squeeze(-1)
                burning = torch.zeros_like(q)
                burning[cfg.algorithm.burning_timesteps :] = 1.0
                q = q * burning
                q = q * (1.0 - done.float())
                optimizer_action.zero_grad()
                loss = -q.mean()
                loss.backward()

                if cfg.algorithm.clip_grad > 0:
                    n = torch.nn.utils.clip_grad_norm_(
                        action_agent.parameters(), cfg.algorithm.clip_grad
                    )
                    logger.add_scalar("monitor/grad_norm_action", n.item(), iteration)

                logger.add_scalar("loss/q_loss", loss.item(), iteration)
                optimizer_action.step()

                tau = cfg.algorithm.update_target_tau
                soft_update_params(q_agent_1, q_target_agent_1, tau)
                soft_update_params(q_agent_2, q_target_agent_2, tau)
                soft_update_params(action_agent, action_target_agent, tau)

            iteration += 1


@hydra.main(config_path=".", config_name="gym.yaml")
def main(cfg):
    import torch.multiprocessing as mp

    mp.set_start_method("spawn")
    logger = instantiate_class(cfg.logger)
    logger.save_hps(cfg)
    q_agent_1 = instantiate_class(cfg.q_agent)
    q_agent_2 = instantiate_class(cfg.q_agent)
    q_agent_2.apply(weight_init)
    action_agent = instantiate_class(cfg.action_agent)
    run_td3(q_agent_1, q_agent_2, action_agent, logger, cfg)


if __name__ == "__main__":
    OmegaConf.register_new_resolver("plus", lambda x, y: x + y)
    OmegaConf.register_new_resolver("n_gpus", lambda x: 0 if x == "cpu" else 1)
    main()
