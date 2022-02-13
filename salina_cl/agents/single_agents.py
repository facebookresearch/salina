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
import copy

def CriticAgent(input_dimension, n_layers, hidden_size):
    return CRLAgents(BatchNorm(input_dimension),Critic(input_dimension, n_layers, hidden_size))

def ActionAgent(input_dimension,output_dimension, n_layers, hidden_size):
    return CRLAgents(BatchNorm(input_dimension),Action(input_dimension,output_dimension, n_layers, hidden_size))

def MultiActionAgent(input_dimension,output_dimension, n_layers, hidden_size):
    return CRLAgents(BatchNorm(input_dimension),MultiAction(input_dimension,output_dimension, n_layers, hidden_size))

def FromscratchActionAgent(input_dimension,output_dimension, n_layers, hidden_size):
    return CRLAgents(BatchNorm(input_dimension),FromscratchAction(input_dimension,output_dimension, n_layers, hidden_size))

def MultiHeadAgent(input_dimension,output_dimension, n_layers, hidden_size):
    return CRLAgents(BatchNorm(input_dimension),NN(input_dimension, [hidden_size], n_layers - 1, hidden_size),MultiAction([hidden_size],output_dimension, 0, hidden_size, input_name = "env/transformed_env_obs"))

def FromscratchHeadAgent(input_dimension,output_dimension, n_layers, hidden_size):
    return CRLAgents(BatchNorm(input_dimension),NN(input_dimension, [hidden_size], n_layers - 1, hidden_size),FromscratchAction([hidden_size],output_dimension, 0, hidden_size, input_name = "env/transformed_env_obs"))

def FreezeNN_FromscratchHeadAgent(input_dimension,output_dimension, n_layers, hidden_size):
    return CRLAgents(BatchNorm(input_dimension),FreezeNN(input_dimension, [hidden_size], n_layers - 1, hidden_size),FromscratchAction([hidden_size],output_dimension, 0, hidden_size, input_name = "env/transformed_env_obs"))

def FreezeNN_MultiHeadAgent(input_dimension,output_dimension, n_layers, hidden_size):
    return CRLAgents(BatchNorm(input_dimension),FreezeNN(input_dimension, [hidden_size], n_layers - 1, hidden_size),MultiAction([hidden_size],output_dimension, 0, hidden_size, input_name = "env/transformed_env_obs"))

class BatchNorm(CRLAgent):
    """
    Apply batch normalization on "env/env_obs" variable and store it in "env/normalized_env_obs"
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

class NN(CRLAgent):
    """
    just a NN.
    """
    def __init__(self, input_dimension,output_dimension, n_layers, hidden_size, input_name = "env/normalized_env_obs", output_name = "env/transformed_env_obs"):
        super().__init__()
        self.iname = input_name
        self.oname = output_name
        self.task_id = 0
            
        input_size = input_dimension[0]
        num_outputs = output_dimension[0]
        hs = hidden_size
        n_layers = n_layers
        hidden_layers = ([nn.Linear(hs, hs) if i % 2 == 0 else nn.ReLU() for i in range(2 * (n_layers - 1))] if n_layers > 1 else [nn.Identity()])
        self.model = nn.ModuleList([nn.Sequential(nn.Linear(input_size, hs),nn.ReLU(),*hidden_layers,nn.Linear(hs, num_outputs))])

    def forward(self, t=None, **kwargs):
        model_id = min(self.task_id,len(self.model) - 1)
        if t is None:
            x = self.get(self.iname)
            x = self.model[model_id](x)
            self.set(self.oname, x)
        else:
            x = self.get((self.iname, t))
            x = self.model[model_id](x)
            self.set((self.oname, t), x)

class MultiNN(NN):
    def set_task(self,task_id = None):
        if task_id is None:
            self.model.append(copy.deepcopy(self.model[-1]))
            self.task_id = len(self.model) - 1
        else:
            self.task_id = task_id

class FromScratchNN(NN):
    def set_task(self,task_id = None):
        if task_id is None:
            self.model.append(copy.deepcopy(self.model[-1]).__init__())
            self.task_id = len(self.model) - 1
        else:
            self.task_id = task_id

class FreezeNN(NN):
    def set_task(self,task_id = None):
        if task_id is None:
            for param in self.model[-1].parameters():
                param.requires_grad = False


class Action(CRLAgent):
    def __init__(self, input_dimension,output_dimension, n_layers, hidden_size, input_name = "env/normalized_env_obs"):
        super().__init__()
        self.iname = input_name
        self.task_id = 0
            
        input_size = input_dimension[0]
        num_outputs = output_dimension[0]
        hs = hidden_size
        n_layers = n_layers
        if n_layers == 0:
            self.model = nn.ModuleList([nn.Sequential(nn.Linear(input_size, num_outputs))])
        else:
            hidden_layers = ([nn.Linear(hs, hs) if i % 2 == 0 else nn.ReLU() for i in range(2 * (n_layers - 1))] if n_layers > 1 else [nn.Identity()])
            self.model = nn.ModuleList([nn.Sequential(nn.Linear(input_size, hs),nn.ReLU(),*hidden_layers,nn.Linear(hs, num_outputs))])

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

class MultiAction(Action):
    def set_task(self,task_id = None):
        if task_id is None:
            self.model.append(copy.deepcopy(self.model[-1]))
            self.task_id = len(self.model) - 1
        else:
            self.task_id = task_id

class FromscratchAction(Action):
    def set_task(self,task_id = None):
        if task_id is None:
            self.model.append(copy.deepcopy(self.model[-1]).__init__())
            self.task_id = len(self.model) - 1
        else:
            self.task_id = task_id

class Critic(CRLAgent):
    def __init__(self, input_dimension, n_layers, hidden_size, input_name = "env/normalized_env_obs"):
        super().__init__()
        self.iname = input_name 
        input_size = input_dimension[0]
        hs = hidden_size
        n_layers = n_layers
        hidden_layers = ([nn.Linear(hs, hs) if i % 2 == 0 else nn.ReLU() for i in range(2 * (n_layers - 1))] if n_layers > 1 else [nn.Identity()])
        self.model_critic = nn.Sequential(nn.Linear(input_size, hs), nn.ReLU(), *hidden_layers, nn.Linear(hs, 1))

    def forward(self, **kwargs):
        input = self.get(self.iname)
        critic = self.model_critic(input).squeeze(-1)
        self.set("critic", critic)