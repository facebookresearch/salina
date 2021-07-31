#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import copy
import math
import time

import d4rl
import gym
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
from salina_examples import weight_init
from salina_examples.offline_rl import d4rl_dataset_to_workspaces


def _state_dict(agent, device):
    sd = agent.state_dict()
    for k, v in sd.items():
        sd[k] = v.to(device)
    return sd


def run_bc(action_agent, logger, cfg):
    env_evaluation_agent = AutoResetGymAgent(
        get_class(cfg.algorithm.env), get_arguments(cfg.algorithm.env)
    )
    action_evaluation_agent = copy.deepcopy(action_agent)

    # == Setting up the training agents
    train_temporal_action_agent = TemporalAgent(action_agent)
    train_temporal_action_agent.to(cfg.algorithm.loss_device)

    evaluation_agent = RemoteAgent(
        TemporalAgent(Agents(env_evaluation_agent, action_evaluation_agent)),
        num_processes=cfg.algorithm.evaluation.n_processes,
    )
    evaluation_agent.seed(cfg.algorithm.evaluation.env_seed)
    evaluation_workspace = salina.Workspace(
        batch_size=cfg.algorithm.evaluation.n_envs,
        time_size=cfg.algorithm.evaluation.n_timesteps,
    )
    evaluation_workspace = evaluation_agent(
        evaluation_workspace,
        epsilon=0.0,
    )

    # == Setting up & initializing the replay buffer for DQN
    replay_buffer = ReplayBuffer(cfg.algorithm.buffer_size)
    logger.message("[TD3] Loading dataset into replay_buffer")
    env = instantiate_class(cfg.algorithm.env)
    dataset = env.get_dataset()
    logger.message("\t Filling replay buffer (may take a few minutes)")
    for w in d4rl_dataset_to_workspaces(
        dataset,
        cfg.algorithm.buffer_time_size,
        proportion=cfg.algorithm.dataset_proportion,
    ):
        replay_buffer.put(w)

    if replay_buffer.size() == cfg.algorithm.buffer_size:
        logger.message(
            "[WARNING]: The replay buffer seems too small to store the dataset"
        )

    logger.message("[DDQN] Learning")
    optimizer_args = get_arguments(cfg.algorithm.optimizer)
    optimizer_action = get_class(cfg.algorithm.optimizer)(
        action_agent.parameters(), **optimizer_args
    )
    for epoch in range(cfg.algorithm.max_epoch):
        if not evaluation_agent.is_running():
            creward, done = evaluation_workspace["env/cumulated_reward", "env/done"]
            creward = creward[done]
            if creward.size()[0] > 0:
                logger.add_scalar("evaluation/reward", creward.mean().item(), epoch)
            action_evaluation_agent.load_state_dict(_state_dict(action_agent, "cpu"))
            evaluation_workspace.copy_time(
                from_time=-1,
                to_time=0,
            )
            evaluation_agent.asynchronous_forward_(
                evaluation_workspace,
                epsilon=0.0,
                t=1,
            )

        logger.add_scalar("monitor/replay_buffer_size", replay_buffer.size(), epoch)

        batch_size = cfg.algorithm.batch_size
        replay_workspace = replay_buffer.get(batch_size).to(cfg.algorithm.loss_device)
        target_action = replay_workspace["action"].detach()
        replay_workspace = train_temporal_action_agent(replay_workspace)
        action = replay_workspace["action"]
        mse = ((target_action - action) ** 2).sum(-1)
        mse_loss = mse.mean()
        logger.add_scalar("loss/mse_1", mse_loss.item(), epoch)
        optimizer_action.zero_grad()
        mse_loss.backward()
        if cfg.algorithm.clip_grad > 0:
            n = torch.nn.utils.clip_grad_norm_(
                action_agent.parameters(), cfg.algorithm.clip_grad
            )
            logger.add_scalar("monitor/grad_norm", n.item(), epoch)

        optimizer_action.step()


@hydra.main(config_path=".", config_name="gym.yaml")
def main(cfg):

    logger = instantiate_class(cfg.logger)
    logger.save_hps(cfg)
    from importlib import import_module

    import_module("examples")
    import_module("salina_examples.offline_rl")

    action_agent = instantiate_class(cfg.action_agent)
    run_bc(action_agent, logger, cfg)


import os

if __name__ == "__main__":
    import torch.multiprocessing as mp

    mp.set_start_method("spawn")
    main()
