#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
from torch import nn
from salina_cl.core import CRLAgent, CRLAgents
import torch.nn.functional as F
from torch import distributions as pyd
import math
import numpy as np
import copy

def ActionAgent(input_dimension,output_dimension, hidden_size, start_steps, layer_norm = False):
    """
    Baseline model: a single policy that is re-used and fine-tuned over the task sequences.
    """
    return CRLAgents(Action(input_dimension,output_dimension, hidden_size, start_steps, input_name = "env/env_obs", layer_norm = layer_norm))

def MultiActionAgent(input_dimension,output_dimension, hidden_size, start_steps, layer_norm = False):
    """
    Fine-tune and clone: the model is saved when the task is ended, and duplicated to be fine-tuned on the next task.
    """
    return CRLAgents(MultiAction(input_dimension,output_dimension, hidden_size, start_steps, input_name = "env/env_obs", layer_norm = layer_norm))

def FromscratchActionAgent(input_dimension,output_dimension, hidden_size, start_steps):
    """
    From scratch: the model is saved when the task is ended, and a new random one is created for the next task.
    """
    return CRLAgents(FromscratchAction(input_dimension,output_dimension, hidden_size, start_steps, input_name = "env/env_obs"))

def TwinCritics(obs_dimension, action_dimension, hidden_size):
    """
    Twin q value functions for SAC algorithm.
    """
    return CRLAgents(Critic(obs_dimension, action_dimension, hidden_size, input_name = "env/env_obs",output_name = "q1"),Critic(obs_dimension, action_dimension, hidden_size, input_name = "env/env_obs", output_name = "q2"))

class Action(CRLAgent):
    def __init__(self, input_dimension,output_dimension, hidden_size, start_steps = 0, input_name = "env/env_obs", layer_norm = False):
        super().__init__()
        self.start_steps = start_steps
        self.counter = 0
        self.iname = input_name
        self.task_id = 0
        self.output_dimension = output_dimension[0]
        hs = hidden_size
        self.input_size = input_dimension[0]
        
        self.model = nn.Sequential(
            nn.Linear(self.input_size,hs),
            nn.LayerNorm(hs) if layer_norm else nn.Identity(),
            nn.Tanh(),
            nn.Linear(hs,hs),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(hs,hs),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(hs,self.output_dimension * 2),
        )
        self.model = nn.ModuleList([self.model])

    def forward(self, t = None, **kwargs):
        model_id = min(self.task_id,len(self.model) - 1)
        if not self.training:
            input = self.get((self.iname, t))
            mu, _ = self.model[model_id](input).chunk(2, dim=-1)
            action = torch.tanh(mu)
            self.set(("action", t), action)
        elif not (t is None):
            input = self.get((self.iname, t)).detach()
            if self.counter <= self.start_steps:
                action = torch.rand(input.shape[0],self.output_dimension).to(input.device) * 2 - 1
            else:
                mu, log_std = self.model[model_id](input).chunk(2, dim=-1)
                log_std = torch.clip(log_std, min=-20., max=2.)
                std = log_std.exp()
                action = mu + torch.randn(*mu.shape).to(mu.device) * std
                action = torch.tanh(action)
            self.set(("action", t), action)
            self.counter += 1
        else:
            input = self.get(self.iname).detach()
            mu, log_std = self.model[model_id](input).chunk(2, dim=-1)
            log_std = torch.clip(log_std, min=-20., max=2.)
            std = log_std.exp()
            action = mu + torch.randn(*mu.shape).to(mu.device) * std
            log_prob = (-0.5 * (((action - mu) / (std + 1e-8)) ** 2 + 2 * log_std + np.log(2 * np.pi))).sum(-1, keepdim=True)
            log_prob -= (2 * np.log(2) - action - F.softplus( - 2 * action)).sum(-1, keepdim=True)
            action = torch.tanh(action)
            self.set("action", action)
            self.set("action_logprobs", log_prob)

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
    def __init__(self, obs_dimension, action_dimension, hidden_size, input_name = "env/env_obs", output_name = "q"):
        super().__init__()

        self.iname = input_name 
        input_size = obs_dimension[0] + action_dimension[0]
        hs = hidden_size
        self.output_name = output_name
        self.model = nn.Sequential(
            nn.Linear(input_size,hs),
            nn.LayerNorm(hs),
            nn.Tanh(),
            nn.Linear(hs,hs),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(hs,hs),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(hs,1),
        )

    def forward(self, detach_action=False,**kwargs):
        input = self.get(self.iname).detach()
        action = self.get(("action"))
        if detach_action:
            action = action.detach()
        input = torch.cat([input, action], dim=-1)
        critic = self.model(input).squeeze(-1)
        self.set(self.output_name, critic)