#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import math

import gym
import torch
import torch.nn as nn
from gym.wrappers import TimeLimit

from salina import TAgent, instantiate_class
from salina.agents import Agents
from salina_examples.rl.atari_wrappers import make_atari, wrap_deepmind, wrap_pytorch


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


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class ActionMLPAgent(TAgent):
    def __init__(self, env, n_layers, hidden_size):
        super().__init__()
        env = instantiate_class(env)
        input_size = env.observation_space.shape[0]
        output_size = env.action_space.shape[0]
        hidden_sizes = [hidden_size for _ in range(n_layers)]
        self.fc = mlp(
            [input_size] + list(hidden_sizes) + [output_size],
            activation=nn.ReLU,
        )

    def forward(self, t, epsilon, **kwargs):
        input = self.get(("env/env_obs", t))
        action = self.fc(input)
        action = torch.tanh(action)
        s = action.size()
        noise = torch.randn(*s, device=action.device) * epsilon
        action = action + noise
        action = torch.clip(action, min=-1.0, max=1.0)
        self.set(("action", t), action)


class QMLPAgent(TAgent):
    def __init__(self, env, n_layers, hidden_size):
        super().__init__()
        env = instantiate_class(env)
        input_size = env.observation_space.shape[0]
        output_size = env.action_space.shape[0]
        hidden_sizes = [hidden_size for _ in range(n_layers)]
        self.fc = mlp(
            [input_size + output_size] + list(hidden_sizes) + [1],
            activation=nn.ReLU,
        )

    def forward(self, t, detach_action=False, **kwargs):
        input = self.get(("env/env_obs", t))
        action = self.get(("action", t))
        if detach_action:
            action = action.detach()
        x = torch.cat([input, action], dim=1)
        q = self.fc(x)
        self.set(("q", t), q)


class OUNoise:
    """
    Ornstein Uhlenbeck process noise for actions as suggested by DDPG paper
    """

    def __init__(self, mean, std_dev, theta=0.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_dev
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        # Generating correlated gaussian noise
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * math.sqrt(self.dt) * torch.randn(self.mean.shape)
        )
        self.x_prev = x
        return x

    def reset(self):
        if self.x0 is not None:
            self.x_prev = self.x0
        else:
            self.x_prev = torch.zeros_like(self.mean)
