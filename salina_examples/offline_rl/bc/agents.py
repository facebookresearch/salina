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


def make_d4rl_env(**env_args):
    e = gym.make(env_args["env_name"], stack=True)
    e = TimeLimit(e, max_episode_steps=env_args["max_episode_steps"])
    return e


def make_d4rl_atari_env(**env_args):
    import d4rl_atari

    e = gym.make(env_args["env_name"], stack=True)
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

    def forward(self, t, **kwargs):
        input = self.get(("env/env_obs", t))
        action = self.fc(input)
        action = torch.tanh(action)
        action = torch.clip(action, min=-1.0, max=1.0)
        self.set(("action", t), action)


class DuelingCnnDQN(nn.Module):
    def __init__(self, input_shape, num_outputs):
        super(DuelingCnnDQN, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_outputs

        self.features = nn.Sequential(
            nn.Conv2d(input_shape[1], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.advantage = nn.Sequential(
            nn.Linear(self.feature_size(), 512), nn.ReLU(), nn.Linear(512, num_outputs)
        )

        self.value = nn.Sequential(
            nn.Linear(self.feature_size(), 512), nn.ReLU(), nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean()

    def feature_size(self):
        return self.features(torch.zeros(1, *self.input_shape[1:])).view(1, -1).size(1)


class AtariAgent(TAgent):
    def __init__(self, env):
        super().__init__()
        env = instantiate_class(env)
        input_shape = (1,) + env.observation_space.shape
        output_size = env.action_space.n
        self.fc = DuelingCnnDQN(input_shape, output_size)

    def forward(self, t, **kwargs):
        input = self.get(("env/env_obs", t))
        scores = self.fc(input)
        probs = torch.softmax(scores, dim=-1)
        action = torch.distributions.Categorical(probs).sample().long()
        self.set(("action", t), action)
        self.set(("action_scores", t), scores)
