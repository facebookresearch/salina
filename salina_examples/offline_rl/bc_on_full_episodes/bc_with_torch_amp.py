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
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

import salina
import salina.rl.functional as RLF
import salina_examples.offline_rl.d4rl
from salina import Workspace, get_arguments, get_class, instantiate_class
from salina.agents import Agents, NRemoteAgent, TemporalAgent
from salina.agents.gyma import AutoResetGymAgent, GymAgent
from salina.logger import TFLogger
import torch.cuda.amp

def _state_dict(agent, device):
    sd = agent.state_dict()
    for k, v in sd.items():
        sd[k] = v.to(device)
    return sd


def run_bc(buffer, logger, action_agent, cfg_algorithm, cfg_env):
    action_agent.set_name("action_agent")

    env = instantiate_class(cfg_env)

    env_evaluation_agent = GymAgent(
        get_class(cfg_env),
        get_arguments(cfg_env),
        n_envs=int(
            cfg_algorithm.evaluation.n_envs / cfg_algorithm.evaluation.n_processes
        ),
    )
    action_evaluation_agent = copy.deepcopy(action_agent)
    action_agent.to(cfg_algorithm.loss_device)
    evaluation_agent, evaluation_workspace = NRemoteAgent.create(
        TemporalAgent(Agents(env_evaluation_agent, action_evaluation_agent)),
        num_processes=cfg_algorithm.evaluation.n_processes,
        t=0,
        n_steps=10,
        epsilon=0.0,
        time_size=cfg_env.max_episode_steps + 1,
    )
    evaluation_agent.eval()

    evaluation_agent.seed(cfg_algorithm.evaluation.env_seed)
    evaluation_agent._asynchronous_call(
        evaluation_workspace, t=0, stop_variable="env/done"
    )

    logger.message("Learning")
    optimizer_args = get_arguments(cfg_algorithm.optimizer)
    optimizer_action = get_class(cfg_algorithm.optimizer)(
        action_agent.parameters(), **optimizer_args
    )
    nsteps_ps_cache=[]
    for epoch in range(cfg_algorithm.max_epoch):
        if not evaluation_agent.is_running():
            length = evaluation_workspace["env/done"].float().argmax(0)
            creward = evaluation_workspace["env/cumulated_reward"]
            arange = torch.arange(length.size()[0], device=length.device)
            creward = creward[length, arange]
            if creward.size()[0] > 0:
                logger.add_scalar("evaluation/reward", creward.mean().item(), epoch)
                v = []
                for i in range(creward.size()[0]):
                    v.append(env.get_normalized_score(creward[i].item()))
                logger.add_scalar("evaluation/normalized_reward", np.mean(v), epoch)
            for a in evaluation_agent.get_by_name("action_agent"):
                a.load_state_dict(_state_dict(action_agent, "cpu"))
            evaluation_workspace.copy_n_last_steps(1)
            evaluation_agent._asynchronous_call(
                evaluation_workspace,
                t=0,
                stop_variable="env/done",
                epsilon=0.0,
            )

        batch_size = cfg_algorithm.batch_size
        replay_workspace = buffer.select_batch_n(batch_size).to(
            cfg_algorithm.loss_device
        )
        _st=time.time()
        with torch.cuda.amp.autocast():
            T = replay_workspace.time_size()
            length = replay_workspace["env/done"].float().argmax(0)
            mask = torch.arange(T).unsqueeze(-1).repeat(1, batch_size).to(length.device)
            length = length.unsqueeze(0).repeat(T, 1)
            mask = mask.le(length).float()
            target_action = replay_workspace["action"].detach()
            action_agent(replay_workspace)
            action = replay_workspace["action"]
            mse = (target_action - action) ** 2
            mse_loss = (mse.sum(2) * mask).sum() / mask.sum()
            logger.add_scalar("loss/mse", mse_loss.item(), epoch)

        optimizer_action.zero_grad()
        mse_loss.backward()
        if cfg_algorithm.clip_grad > 0:
            n = torch.nn.utils.clip_grad_norm_(
                action_agent.parameters(), cfg_algorithm.clip_grad
            )
            logger.add_scalar("monitor/grad_norm", n.item(), epoch)

        optimizer_action.step()
        _et=time.time()
        nsteps=batch_size*T
        nsteps_ps=nsteps/(_et-_st)
        nsteps_ps_cache.append(nsteps_ps)
        if len(nsteps_ps_cache)>1000: nsteps_ps_cache.pop(0)
        logger.add_scalar("monitor/steps_per_seconds", np.mean(nsteps_ps_cache), epoch)

@hydra.main(config_path=".", config_name="gym.yaml")
def main(cfg):

    logger = instantiate_class(cfg.logger)
    logger.save_hps(cfg)
    from importlib import import_module

    env = instantiate_class(cfg.env)
    workspace = salina_examples.offline_rl.d4rl.d4rl_episode_buffer(env)
    agent = instantiate_class(cfg.agent)
    run_bc(workspace, logger, agent, cfg.algorithm, cfg.env)


import os

if __name__ == "__main__":
    import torch.multiprocessing as mp

    mp.set_start_method("spawn")
    main()
