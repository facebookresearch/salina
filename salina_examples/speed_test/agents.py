#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import gym
import torch
import torch.nn as nn
from gym.wrappers import TimeLimit

from salina import TAgent, get_arguments, get_class, instantiate_class
from salina.agents import Agents
from salina.agents.gym import AutoResetGymAgent, GymAgent
from salina_examples.rl.atari_wrappers import (
    make_atari,
    wrap_deepmind,
    wrap_pytorch,
)


def make_atari_env(**env_args):
    e = make_atari(env_args["env_name"])
    e = wrap_deepmind(e)
    e = wrap_pytorch(e)
    e = TimeLimit(e, max_episode_steps=env_args["max_episode_steps"])
    return e


def make_gym_env(**env_args):
    e = gym.make(env_args["env_name"])
    e = TimeLimit(e, max_episode_steps=env_args["max_episode_steps"])
    return e


def GymRandomAgent(env):
    e = instantiate_class(env)
    action_space = e.action_space
    env_agent = AutoResetGymAgent(get_class(env), get_arguments(env))
    policy_agent = RandomAgent(action_space)
    return Agents(env_agent, policy_agent)


class RandomAgent(TAgent):
    def __init__(self, action_space):
        super().__init__()
        self.action_space = action_space

    def forward(self, t, **args):
        B = self.workspace.batch_size()
        actions = [
            torch.tensor(self.action_space.sample()).unsqueeze(0) for _ in range(B)
        ]
        action = torch.cat(actions, dim=0)
        self.set(("action", t), action)
