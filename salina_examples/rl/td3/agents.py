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
    m=nn.Sequential(*layers)
    #m=torch.jit.script(m)
    return m

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

    def forward(self, t, epsilon, epsilon_clip=100000, **kwargs):
        input = self.get(("env/env_obs", t))
        action = self.fc(input)
        action = torch.tanh(action)
        s = action.size()
        noise = torch.randn(*s, device=action.device) * epsilon
        noise = torch.clip(noise, min=-epsilon_clip, max=epsilon_clip)
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
