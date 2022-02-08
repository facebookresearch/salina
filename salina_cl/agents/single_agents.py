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
from salina_cl.core import CRLAgent, CRLAgents
from salina_cl.agents.tools import weight_init, SquashedNormal
import copy


def ActionAgent(input_dimension,output_dimension, n_layers, hidden_size):
    return CRLAgents(ActionAgent(input_dimension,output_dimension, n_layers, hidden_size,False))

def CriticAgent(input_dimension, n_layers, hidden_size):
    return CRLAgents(CriticAgent(input_dimension, n_layers, hidden_size,False))

def BatchNormActionAgent(input_dimension,output_dimension, n_layers, hidden_size):
    return CRLAgents(BatchNorm(input_dimension),ActionAgent(input_dimension,output_dimension, n_layers, hidden_size,True))

def BatchNormCriticAgent(input_dimension, n_layers, hidden_size):
    return CRLAgents(BatchNorm(input_dimension),CriticAgent(input_dimension, n_layers, hidden_size,True))

def IncrementalBatchNormActionAgent(input_dimension,output_dimension, n_layers, hidden_size):
    return CRLAgents(IncrementalBatchNorm(input_dimension),ActionAgent(input_dimension,output_dimension, n_layers, hidden_size,True))

def IncrementalBatchNormIncrementalActionAgent(input_dimension,output_dimension, n_layers, hidden_size):
    return CRLAgents(IncrementalBatchNorm(input_dimension),IncrementalActionAgent(input_dimension,output_dimension, n_layers, hidden_size,True))

class BatchNorm(CRLAgent):
    """
    Applying Batch Normalization to env_obs
    """
    def __init__(self,input_dimension):
        super().__init__()
        self.num_features = input_dimension[0]
        self.bn = nn.BatchNorm1d(num_features=self.num_features)
    
    def forward(self, t=None, **kwargs):
        if not t is None:
            input = self.get(("env/env_obs", t))
            input=self.bn(input)
            self.set(("env/normalized_env_obs", t), input)
        else:
            input = self.get("env/env_obs")
            T,B,s = input.size()
            input = input.reshape(T*B,s)
            input = self.bn(input)
            input = input.reshape(T,B,s)
            self.set("env/normalized_env_obs",input)

class IncrementalBatchNorm(CRLAgent):
    """
    Creating one BatchNorm module for each task
    """
    def __init__(self,input_dimension):
        super().__init__()
        self.num_features = input_dimension[0]
        self.bn = nn.ModuleList([nn.BatchNorm1d(num_features=self.num_features)])
        self.task_id = 0
    
    def forward(self, t=None, **kwargs):
        model_id = min(self.task_id, len(self.bn)-1)
        if not t is None:
            input = self.get(("env/env_obs", t))
            input = self.bn[model_id](input)
            self.set(("env/normalized_env_obs", t), input)
        else:
            input = self.get("env/env_obs")
            T,B,s = input.size()
            input = input.reshape(T*B,s)
            input = self.bn[model_id](input)
            input = input.reshape(T,B,s)
            self.set("env/normalized_env_obs",input)

    def set_task(self,task_id = None):
        if task_id is None:
            self.bn.append(nn.BatchNorm1d(num_features=self.num_features))
            self.task_id = len(self.bn) - 1
        else:
            self.task_id = task_id

class ActionAgent(CRLAgent):
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

class IncrementalActionAgent(CRLAgent):
    def __init__(self, input_dimension,output_dimension, n_layers, hidden_size,use_normalized_obs=False):
        super().__init__()
        self.iname="env/env_obs"
        if use_normalized_obs:
            self.iname="env/normalized_env_obs"
        self.task_id = 0
            
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
        self.model = nn.ModuleList([nn.Sequential(
            nn.Linear(input_size, hs),
            nn.ReLU(),
            *hidden_layers,
            nn.Linear(hs, num_outputs),
        )])

    def forward(self, t=None, action_std=0.0, **kwargs):
        if not self.training: assert action_std==0.0
        model_id = min(self.task_id,len(self.model) - 1)

        if t is None:
            input = self.get(self.iname)
            mean = self.model[model_id](input)
            var = torch.ones_like(mean) * action_std + 0.000001
            dist = Normal(mean, var)
            action = self.get("action_before_tanh")
            logp_pi = dist.log_prob(action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(axis=-1)
            self.set("action_logprobs", logp_pi)
        else:
            input = self.get((self.iname, t))
            mean = self.model[model_id](input)
            var = torch.ones_like(mean) * action_std + 0.000001
            dist = Normal(mean, var)
            action = dist.sample() if action_std > 0 else dist.mean
            self.set(("action_before_tanh", t), action)
            logp_pi = dist.log_prob(action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(axis=-1)
            self.set(("action_logprobs", t), logp_pi)
            action = torch.tanh(action)
            self.set(("action", t), action)

    def set_task(self,task_id = None):
        if task_id is None:
            self.model.append(copy.deepcopy(self.model[-1]))
            self.task_id = len(self.model) - 1
        else:
            self.task_id = task_id

class CriticAgent(CRLAgent):
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

class SACPolicyAgent(CRLAgent):
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

class QAgent(CRLAgent):
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
