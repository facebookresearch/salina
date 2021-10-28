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
from salina_examples.rl.atari_wrappers import make_atari, wrap_deepmind, wrap_pytorch


def make_gym_env(**env_args):
    e = gym.make(env_args["env_name"])
    e = TimeLimit(e, max_episode_steps=env_args["max_episode_steps"])
    return e


def make_atari_env(**env_args):
    e = make_atari(env_args["env_name"])
    e = wrap_deepmind(e)
    e = wrap_pytorch(e)
    e = TimeLimit(e, max_episode_steps=env_args["max_episode_steps"])
    return e


class MLP(nn.Module):
    def __init__(self, sizes, activation, output_activation=nn.Identity):
        super().__init__()
        layers = []
        for j in range(len(sizes) - 1):
            act = activation if j < len(sizes) - 2 else output_activation
            layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
        self.layers = layers
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)


class DQNMLPAgent(TAgent):
    def __init__(self, env, hidden_size, n_layers):
        super().__init__()
        env = instantiate_class(env)
        input_size = env.observation_space.shape[0]
        num_outputs = env.action_space.n
        hidden_sizes = [hidden_size for _ in range(n_layers)]
        self.model = MLP(
            [input_size] + list(hidden_sizes) + [num_outputs],
            activation=nn.ReLU,
        )

    def forward(self, t, replay=False, epsilon=0.0, **kwargs):
        input = self.get(("env/env_obs", t))
        q = self.model(input)

        if not replay:
            B = q.size()[0]
            n_actions = q.size()[1]
            is_random = torch.rand(B).lt(epsilon).float()
            random_action = torch.randint(low=0, high=n_actions, size=(B,))
            max_action = q.max(1)[1]
            action = is_random * random_action + (1 - is_random) * max_action
            action = action.long()

            self.set(("action", t), action)

        self.set(("q", t), q)


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


class DQNAtariAgent(TAgent):
    def __init__(self, **kwargs):
        super().__init__()
        env = instantiate_class(kwargs["env"])
        input_shape = (1,) + env.observation_space.shape
        num_outputs = env.action_space.n
        self.cnn = DuelingCnnDQN(input_shape, num_outputs)

    def _forward_nn(self, state):
        qvals = self.cnn(state)
        return qvals

    def forward(self, t, replay=False, epsilon=0.0, **kwargs):
        input = self.get(("env/env_obs", t)).float()
        q = self._forward_nn(input)

        if not replay:
            B = q.size()[0]
            n_actions = q.size()[1]
            is_random = torch.rand(B).lt(epsilon).float()
            random_action = torch.randint(low=0, high=n_actions, size=(B,))
            max_action = q.max(1)[1]
            action = is_random * random_action + (1 - is_random) * max_action
            action = action.long()

            self.set(("action", t), action)

        self.set(("q", t), q)
