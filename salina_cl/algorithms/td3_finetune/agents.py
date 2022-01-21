#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import gym
import torch
import torch.nn as nn
import math
from salina_cl.algorithms.optimizers.distributions import SquashedNormal
from salina_cl.algorithms.optimizers.tools import weight_init

from salina import Agent, instantiate_class
from salina.agents import Agents
import numpy as np
import torch.nn.functional as F

LOG_SIG_MAX = 2
LOG_SIG_MIN = -5
epsilon = epsilon = 1e-6

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    m=nn.Sequential(*layers)
    return m


class ActionAgent(Agent):
    def __init__(self, n_layers, hidden_size,input_dimension,output_dimension):
        super().__init__()
        input_size = input_dimension[0]
        output_size = output_dimension[0]
        hidden_sizes = [hidden_size for _ in range(n_layers)]
        self.fc = mlp(
            [input_size] + list(hidden_sizes) + [output_size],
            activation=nn.ReLU,
        )

    def forward(self, t=None, epsilon=0.0, epsilon_clip=None, **kwargs):
        if not self.training:
            assert epsilon==0.0
            assert epsilon_clip is None

        if not t is None:
            input = self.get(("env/env_obs", t))
            action = self.fc(input)
            action = torch.tanh(action)
            s = action.size()
            noise = torch.randn(*s, device=action.device) * epsilon
            if not epsilon_clip is None:
                noise = torch.clip(noise, min=-epsilon_clip, max=epsilon_clip)
            action = action + noise
            action = torch.clip(action, min=-1.0, max=1.0)
            self.set(("action", t), action)
        else:
            input = self.get("env/env_obs")
            action = self.fc(input)
            action = torch.tanh(action)
            s = action.size()
            noise = torch.randn(*s, device=action.device) * epsilon
            if not epsilon_clip is None:
                noise = torch.clip(noise, min=-epsilon_clip, max=epsilon_clip)
            action = action + noise
            action = torch.clip(action, min=-1.0, max=1.0)
            self.set("action", action)            

class QAgent(Agent):
    def __init__(self, n_layers, hidden_size,input_dimension,output_dimension):
        super().__init__()
        input_size = input_dimension[0]
        output_size = output_dimension[0]
        hidden_sizes = [hidden_size for _ in range(n_layers)]
        self.fc = mlp(
            [input_size + output_size] + list(hidden_sizes) + [1],
            activation=nn.ReLU,
        )
        self.fc.apply(weight_init)

    def forward(self, **kwargs):
            input = self.get("env/env_obs")
            action = self.get("action")
            x = torch.cat([input, action], dim=-1)
            q = self.fc(x)
            self.set("q", q)
