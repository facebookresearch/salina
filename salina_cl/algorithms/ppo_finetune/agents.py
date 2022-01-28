#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions.normal import Normal
from salina import Agent
from salina.agents import Agents


class BatchNorm(Agent):
    def __init__(self,input_dimension):
        super().__init__()
        self.bn=nn.BatchNorm1d(num_features=input_dimension[0])
    
    def forward(self, t=None, **kwargs):
        if not t is None:
            input = self.get(("env/env_obs", t))
            input=self.bn(input)
            self.set(("env/normalized_env_obs", t), input)
        else:
            input = self.get("env/env_obs")
            T,B,s=input.size()
            input=input.reshape(T*B,s)
            input=self.bn(input)
            input=input.reshape(T,B,s)
            self.set("env/normalized_env_obs",input)
   
def clip_grad(parameters, grad):
    return (
        torch.nn.utils.clip_grad_norm_(parameters, grad)
        if grad > 0
        else torch.Tensor([0.0])
    )

def BatchNormActionAgent(input_dimension,output_dimension, n_layers, hidden_size):
    return Agents(BatchNorm(input_dimension),ActionAgent(input_dimension,output_dimension, n_layers, hidden_size,True))

class ActionAgent(Agent):
    def __init__(self, input_dimension,output_dimension, n_layers, hidden_size,use_normalized_obs=False):
        super().__init__()
        self.iname="env/env_obs"
        if use_normalized_obs:
            self.iname="env/normalized_env_obs"
            
        input_size = input_dimension[0]
        num_outputs = output_dimension[0]
        hs = hidden_size
        n_layers = n_layers
        hidden_layers = (
            [
                nn.Linear(hs, hs) if i % 2 == 0 else nn.ReLU()
                for i in range(2 * (n_layers - 1))
            ]
            if n_layers > 1
            else [nn.Identity()]
        )
        self.model = nn.Sequential(
            nn.Linear(input_size, hs),
            nn.ReLU(),
            *hidden_layers,
            nn.Linear(hs, num_outputs),
        )

    def forward(self, t=None, action_std=0.0, **kwargs):
        if not self.training: assert action_std==0.0

        if t is None:
            input = self.get(self.iname)
            mean = self.model(input)
            var = torch.ones_like(mean) * action_std + 0.000001
            dist = Normal(mean, var)
            action = self.get("action_before_tanh")
            logp_pi = dist.log_prob(action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(axis=-1)
            self.set("action_logprobs", logp_pi)
        else:
            input = self.get((self.iname, t))
            mean = self.model(input)
            var = torch.ones_like(mean) * action_std + 0.000001
            dist = Normal(mean, var)
            action = dist.sample() if action_std > 0 else dist.mean
            self.set(("action_before_tanh", t), action)
            logp_pi = dist.log_prob(action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(axis=-1)
            self.set(("action_logprobs", t), logp_pi)
            action = torch.tanh(action)
            self.set(("action", t), action)

def BatchNormCriticAgent(input_dimension, n_layers, hidden_size):
    return Agents(BatchNorm(input_dimension),CriticAgent(input_dimension, n_layers, hidden_size,True))


class CriticAgent(Agent):
    def __init__(self, input_dimension, n_layers, hidden_size,use_normalized_obs=False):
        super().__init__()
        self.iname="env/env_obs"
        if use_normalized_obs:
            self.iname="env/normalized_env_obs"
            
        input_size = input_dimension[0]
        hs = hidden_size
        n_layers = n_layers
        hidden_layers = (
            [
                nn.Linear(hs, hs) if i % 2 == 0 else nn.ReLU()
                for i in range(2 * (n_layers - 1))
            ]
            if n_layers > 1
            else [nn.Identity()]
        )
        self.model_critic = nn.Sequential(
            nn.Linear(input_size, hs),
            nn.ReLU(),
            *hidden_layers,
            nn.Linear(hs, 1),
        )

    def forward(self, **kwargs):
        input = self.get(self.iname)
        critic = self.model_critic(input).squeeze(-1)
        self.set("critic", critic)
