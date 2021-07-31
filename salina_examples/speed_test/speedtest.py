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
from tqdm import tqdm

import salina
import salina.rl.functional as RLF
from salina import get_arguments, get_class, instantiate_class
from salina.agents import Agents, RemoteAgent, TemporalAgent
from salina.agents.gym import AutoResetGymAgent, GymAgent
from salina.logger import TFLogger
from salina.rl.replay_buffer import ReplayBuffer
from salina_examples import weight_init


def run_speed_test(agent, cfg):
    remote_agent = RemoteAgent(
        TemporalAgent(agent),
        num_processes=cfg.speed_test.n_processes,
    )
    remote_agent.seed(0)

    workspace = salina.Workspace(
        batch_size=cfg.speed_test.batch_size,
        time_size=cfg.speed_test.n_timesteps,
    )
    workspace = remote_agent(workspace)
    _start_time = time.time()
    n_steps = 0
    for epoch in tqdm(range(cfg.speed_test.max_epoch)):
        workspace.copy_time(
            from_time=-1,
            to_time=0,
        )
        remote_agent(workspace, t=1)
        n_steps += cfg.speed_test.batch_size * (cfg.speed_test.n_timesteps - 1)
    _end_time = time.time()
    ttime = _end_time - _start_time
    n_per_seconds = n_steps / ttime
    print(n_per_seconds, " steps per seconds")


@hydra.main(config_path=".", config_name="gym.yaml")
def main(cfg):
    import torch.multiprocessing as mp

    mp.set_start_method("spawn")
    agent = instantiate_class(cfg.agent)
    run_speed_test(agent, cfg)


if __name__ == "__main__":
    main()
