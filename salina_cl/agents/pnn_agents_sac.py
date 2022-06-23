#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
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
from salina_cl.agents.single_agents import  Normalizer
import copy

def PNNActionAgent(input_dimension,output_dimension, hidden_size, start_steps,layer_norm):
    return CRLAgents(PNNAction(input_dimension,output_dimension, hidden_size, start_steps, input_name = "env/env_obs",layer_norm = layer_norm))
   
class PNNAction(CRLAgent):
    def __init__(self, input_dimension,output_dimension, hidden_size, start_steps = 0, input_name = "env/env_obs", activation = nn.LeakyReLU(negative_slope=0.2), layer_norm = True):
        super().__init__()
        self.iname = input_name
        self.task_id = 0
        self.activation = activation
        self.layer_norm = layer_norm
        self.activation_layer_norm = nn.Tanh()

        self.start_steps = start_steps
        self.counter = 0        

        self.output_dimension = output_dimension[0]
        self.hs = hidden_size
        self.input_size = input_dimension[0]
      
        self.columns=nn.ModuleList()
        self.laterals=nn.ModuleList()
        self.create_columns()
        
    def create_columns(self):
        print('Creating a new columns and its lateral')

        ## we create a column and its lateral connection
        column_model = nn.ModuleList([
            nn.Linear(self.input_size,self.hs),
            nn.LayerNorm(self.hs) if self.layer_norm else nn.Identity(),
            nn.Linear(self.hs,self.hs),
            nn.Linear(self.hs,self.hs),
            nn.Linear(self.hs,self.output_dimension * 2)])
        self.columns.append(column_model)

        lateral_model = nn.ModuleList([nn.ModuleList([
            nn.Linear(self.hs,self.hs),
            nn.Linear(self.hs,self.hs),
            nn.Linear(self.hs,self.hs),
            nn.Linear(self.hs,self.hs)]) for _ in range(len(self.columns)-1)])
        self.laterals.append(lateral_model)

    def _forward(self,x,column_id):
        h = [copy.deepcopy(x) for i in range(column_id + 1)]
        for j in range(len(self.columns[0])):
            activation = self.activation # if j!=1 else self.activation_layer_norm
            h[column_id] = self.columns[column_id][j](h[column_id])
            h[column_id] = activation(h[column_id])
            if j < len(self.columns[0]) - 1:
                # Adding laterals
                for i in range(column_id):
                    h[i] = self.columns[i][j](h[i])
                    h[i] = activation(h[i]) #if j!=1 else self.activation_layer_norm(h[i])
                    h[column_id] += self.laterals[column_id][i][j](h[i])
        return h[column_id]

    def forward(self, t = None, **kwargs):
        column_id = min(self.task_id,len(self.columns) - 1)
        if not self.training:
            input = self.get((self.iname, t))
            mu, _ = self._forward(input,column_id).chunk(2, dim=-1)
            action = torch.tanh(mu)
            self.set(("action", t), action)
        elif not (t is None):
            input = self.get((self.iname, t)).detach()
            if self.counter <= self.start_steps:
                action = torch.rand(input.shape[0],self.output_dimension).to(input.device) * 2 - 1
            else:
                mu, log_std = self._forward(input,column_id).chunk(2, dim=-1)
                log_std = torch.clip(log_std, min=-20., max=2.)
                std = log_std.exp()
                action = mu + torch.randn(*mu.shape).to(mu.device) * std
                action = torch.tanh(action)
            self.set(("action", t), action)
            self.counter += 1
        else:
            input = self.get(self.iname).detach()
            mu, log_std = self._forward(input,column_id).chunk(2, dim=-1)
            log_std = torch.clip(log_std, min=-20., max=2.)
            std = log_std.exp()
            action = mu + torch.randn(*mu.shape).to(mu.device) * std
            log_prob = (-0.5 * (((action - mu) / (std + 1e-8)) ** 2 + 2 * log_std + np.log(2 * np.pi))).sum(-1, keepdim=True)
            log_prob -= (2 * np.log(2) - action - F.softplus( - 2 * action)).sum(-1, keepdim=True)
            action = torch.tanh(action)
            self.set("action", action)
            self.set("action_logprobs", log_prob)

    def set_task(self,task_id = None):
        if task_id is None:
            for param in self.columns[-1].parameters():
                param.requires_grad = False
            self.create_columns()
            self.task_id = len(self.columns) - 1
        else:
            self.task_id = task_id
