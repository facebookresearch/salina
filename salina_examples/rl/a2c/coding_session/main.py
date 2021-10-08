#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

###### IMPORT
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
from salina import TAgent, get_arguments, get_class, instantiate_class
from salina.agents import Agents, RemoteAgent, TemporalAgent
from salina.agents.gym import AutoResetGymAgent, GymAgent
from salina.logger import TFLogger


def _index(tensor_3d, tensor_2d):
    """This function is used to index a 3d tensors using a 2d tensor"""
    x, y, z = tensor_3d.size()
    t = tensor_3d.reshape(x * y, z)
    tt = tensor_2d.reshape(x * y)
    v = t[torch.arange(x * y), tt]
    v = v.reshape(x, y)
    return v

#### DEFINE AGENTS




### DEFINE ENVIRONMENT
def make_cartpole(max_episode_steps):
    return TimeLimit(gym.make("CartPole-v0"), max_episode_steps=max_episode_steps)


def run_a2c(cfg):
    ##### BUILD THE LOGGER (Salina provides a CSV+Tensroboard Logger that one is free to use or not)
    logger = instantiate_class(cfg.logger)

    #### CREATE THE ENVIRONMENT Agent (Salina provides Gym Agents and Brax Agents)

    ### CREATE THE POLICY

    #### CREATE THE ACQUSITION AGENT

    ### DEFINE A WORKSPACE

    # CONFIGURE THE OPTIMIZER

    # TEST FORWARD

    # TRAINING LOOP
    epoch = 0
    for epoch in range(cfg.algorithm.max_epochs):

        ### CONTINUE ACQUISITION
        workspace.copy_n_last_steps(1)
        acquisition_agent(workspace, t=1, stochastic=True)


        #### COMPUTE LOSS

        ###### GET RELEVANT VARIABLES


        ### COMPUTE TEMPORAL DIFFERENCE

        ### COMPUTE CRITIC LOSS

        ### COMPUTE ENTROPY

        ### COMPUTE ACTION LOSS

        #LOG


        ### LOG env/cumulated_reward FOR TERMINAL STATES


@hydra.main(config_path=".", config_name="main.yaml")
def main(cfg):
    import torch.multiprocessing as mp

    mp.set_start_method("spawn")
    run_a2c(cfg)


if __name__ == "__main__":
    main()
