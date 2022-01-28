#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.nn as nn
from salina_cl.algorithms.distributions import SquashedNormal
from salina_cl.algorithms.tools import weight_init
from salina import Agent

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

    def forward(self, t=None, **kwargs):
        if self.training:
            deterministic=False
        else:
            deterministic=True

        if not t is None:
            input = self.get(("env/env_obs", t))
            B=input.size()[0]

            mean=self.fc(input)
            log_std = self.fc2(input)

            log_std = torch.tanh(log_std)
            log_std = LOG_SIG_MIN + 0.5 * (LOG_SIG_MAX - LOG_SIG_MIN) * (log_std +1)
            self.set(("sac/mean",t),mean)
            self.set(("sac/log_std",t),log_std)
            self.set(("action_std",t),log_std.exp())
            std=log_std.exp()

            dist=SquashedNormal(mean, std)
            if not deterministic:
                action=dist.rsample()
            else:
                action=dist.mean
            #print("Action = ",action)
            self.set(("action", t), action)

            log_prob=dist.log_prob(action)
            #log_prob -= torch.log(1.0 * (1 - action.pow(2)) + epsilon)
            #print(torch.log(1.0 * (1 - action.pow(2)) + epsilon),log_prob)
            self.set(("action_logprobs",t),log_prob.sum(-1))
        else:
            input = self.get("env/env_obs")
            B=input.size()[0]

            mean=self.fc(input)
            log_std = self.fc2(input)

            log_std = torch.tanh(log_std)
            log_std = LOG_SIG_MIN + 0.5 * (LOG_SIG_MAX - LOG_SIG_MIN) * (log_std +1)
            self.set("sac/mean",mean)
            self.set("sac/log_std",log_std)
            self.set("action_std",log_std.exp())
            std=log_std.exp()

            dist=SquashedNormal(mean, std)
            if not deterministic:
                action=dist.rsample()
            else:
                action=dist.mean
            self.set("action", action)

            log_prob=dist.log_prob(action)
            self.set("action_logprobs",log_prob.sum(-1))

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
