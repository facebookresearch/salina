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
import d4rl_atari
import gym
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import salina
import salina.rl.functional as RLF
from salina import Workspace, get_arguments, get_class, instantiate_class
from salina.agents import Agents, NRemoteAgent, TemporalAgent
from salina.agents.gyma import AutoResetGymAgent, GymAgent
from salina.logger import TFLogger
from salina.rl.replay_buffer import ReplayBuffer
from salina_examples import weight_init
from salina_examples.offline_rl.d4rl import *


def _state_dict(agent, device):
    sd = agent.state_dict()
    for k, v in sd.items():
        sd[k] = v.to(device)
    return sd


def run_bc(action_agent, logger, cfg):
    action_agent.set_name("action_agent")
    env_evaluation_agent = AutoResetGymAgent(
        get_class(cfg.algorithm.env),
        get_arguments(cfg.algorithm.env),
        n_envs=int(
            cfg.algorithm.evaluation.n_envs / cfg.algorithm.evaluation.n_processes
        ),
    )
    action_evaluation_agent = copy.deepcopy(action_agent)

    train_temporal_action_agent = TemporalAgent(action_agent)
    train_temporal_action_agent.to(cfg.algorithm.loss_device)

    logger.message("[BC] Creating replay_buffer")
    env = instantiate_class(cfg.algorithm.env)
    replay_buffer = d4rl_transition_buffer(env)

    evaluation_agent, evaluation_workspace = NRemoteAgent.create(
        TemporalAgent(Agents(env_evaluation_agent, action_evaluation_agent)),
        num_processes=cfg.algorithm.evaluation.n_processes,
        t=0,
        n_steps=cfg.algorithm.evaluation.n_timesteps,
        epsilon=0.0,
    )
    evaluation_agent.seed(cfg.algorithm.evaluation.env_seed)
    evaluation_agent._asynchronous_call(
        evaluation_workspace,
        t=0,
        n_steps=cfg.algorithm.evaluation.n_timesteps,
        epsilon=0.0,
    )

    logger.message("Learning")
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
            for a in evaluation_agent.get_by_name("action_agent"):
                a.load_state_dict(_state_dict(action_agent, "cpu"))
            evaluation_workspace.copy_n_last_steps(1)
            evaluation_agent._asynchronous_call(
                evaluation_workspace,
                t=1,
                n_steps=cfg.algorithm.evaluation.n_timesteps - 1,
                epsilon=0.0,
            )

        batch_size = cfg.algorithm.batch_size
        replay_workspace = replay_buffer.select_batch_n(batch_size).to(
            cfg.algorithm.loss_device
        )
        target_action = replay_workspace["action"].detach()
        train_temporal_action_agent(
            replay_workspace, t=0, n_steps=replay_workspace.time_size()
        )
        action, scores = replay_workspace["action", "action_scores"]
        s = scores.size()
        target_action = target_action.reshape(s[0] * s[1])
        scores = scores.reshape(s[0] * s[1], *s[2:])
        loss = F.cross_entropy(scores, target_action)
        logger.add_scalar("loss/loss", loss.item(), epoch)
        optimizer_action.zero_grad()
        loss.backward()
        if cfg.algorithm.clip_grad > 0:
            n = torch.nn.utils.clip_grad_norm_(
                action_agent.parameters(), cfg.algorithm.clip_grad
            )
            logger.add_scalar("monitor/grad_norm", n.item(), epoch)

        optimizer_action.step()


@hydra.main(config_path=".", config_name="pong.yaml")
def main(cfg):

    logger = instantiate_class(cfg.logger)
    logger.save_hps(cfg)
    action_agent = instantiate_class(cfg.action_agent)
    run_bc(action_agent, logger, cfg)


import os

if __name__ == "__main__":
    import torch.multiprocessing as mp

    mp.set_start_method("spawn")
    main()
