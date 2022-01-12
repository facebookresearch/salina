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
from salina_examples.rl.sac.continuouscartpole import CartPoleEnv
import numpy as np
import torch.nn.functional as F

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
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
        self.fc_mean = mlp(
            [input_size] + list(hidden_sizes) + [output_size],
            activation=nn.ReLU,
        )
        self.fc_log_std = mlp(
            [input_size] + list(hidden_sizes) + [output_size],
            activation=nn.ReLU,
        )

    def forward(self, t, deterministic,**kwargs):
        input = self.get(("env/env_obs", t))
        B=input.size()[0]

        mean=self.fc_mean(input)
        log_std=self.fc_log_std(input)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        self.set(("sac/mean",t),mean)
        self.set(("sac/log_std",t),log_std)
        std=log_std.exp()

        random_value=torch.normal(mean=torch.zeros(B,self.output_size),std=torch.ones(B,self.output_size))
        self.set(("sac/random_value",t),random_value)
        saction=None
        if not deterministic:
            saction=mean+std*random_value
        else:
            saction=mean
        action=torch.tanh(saction)
        print(saction[0])
        self.set(("saction", t), saction)
        self.set(("action", t), action)

        normal=torch.distributions.Normal(mean, std)
        log_prob=normal.log_prob(saction)
        log_prob -= (2 * (np.log(2) - saction - F.softplus(-2 * saction)))
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

    def forward(self, t, detach_action=False, **kwargs):
        input = self.get(("env/env_obs", t))
        action = self.get(("action", t))
        if detach_action:
            action = action.detach()
        x = torch.cat([input, action], dim=1)
        q = self.fc(x)
        self.set(("q", t), q)
