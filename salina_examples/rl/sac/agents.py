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
from salina_examples import weight_init
import math

from salina import TAgent, instantiate_class
from salina.agents import Agents
from salina_examples.rl.atari_wrappers import make_atari, wrap_deepmind, wrap_pytorch
from salina_examples.rl.sac.continuouscartpole import CartPoleEnv
import numpy as np
import torch.nn.functional as F
from torch import distributions as pyd

class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu




LOG_SIG_MAX = 2
LOG_SIG_MIN = -5
epsilon = epsilon = 1e-6

def make_atari_env(**env_args):
    e = make_atari(env_args["env_name"])
    e = wrap_deepmind(e)
    e = wrap_pytorch(e)
    e = TimeLimit(e, max_episode_steps=env_args["max_episode_steps"])
    return e


def make_gym_env(**env_args):
    e = gym.make(env_args["env_name"])
    #e = CartPoleEnv()
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
        self.output_size=output_size
        hidden_sizes = [hidden_size for _ in range(n_layers)]
        self.fc = mlp(
            [input_size] + list(hidden_sizes) + [output_size],
            activation=nn.ReLU,
        )
        self.fc2 = mlp(
            [input_size] + list(hidden_sizes) + [output_size],
            activation=nn.ReLU,
        )
        self.fc.apply(weight_init)
        self.fc2.apply(weight_init)

    def forward(self, t, deterministic,**kwargs):
        input = self.get(("env/_env_obs", t))
        B=input.size()[0]

        mean=self.fc(input)
        log_std = self.fc2(input)

        log_std = torch.tanh(log_std)
        log_std = LOG_SIG_MIN + 0.5 * (LOG_SIG_MAX - LOG_SIG_MIN) * (log_std +
                                                                     1)
        self.set(("sac/mean",t),mean)
        self.set(("sac/log_std",t),log_std)
        std=log_std.exp()

        normal=SquashedNormal(mean, std)
        if not deterministic:
            action=normal.rsample()
        else:
            action=normal.mean
        #print("Action = ",action)
        self.set(("action", t), action)

        log_prob=normal.log_prob(action)
        #log_prob -= torch.log(1.0 * (1 - action.pow(2)) + epsilon)
        #print(torch.log(1.0 * (1 - action.pow(2)) + epsilon),log_prob)
        self.set(("sac/log_prob_action",t),log_prob.sum(-1))

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
        self.fc.apply(weight_init)

    def forward(self, t, detach_action=False, **kwargs):
        input = self.get(("env/_env_obs", t))
        action = self.get(("action", t))
        if detach_action:
            action = action.detach()
        x = torch.cat([input, action], dim=1)
        q = self.fc(x)
        self.set(("q", t), q)
