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
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.categorical import Categorical
from salina_cl.agents.tools import LinearSubspace, Sequential
from salina_cl.agents.single_agents import BatchNorm

def SubspaceActionAgent(n_initial_anchors, dist_type, input_dimension,output_dimension, n_layers, hidden_size):
    return CRLAgents(AlphaAgent(n_initial_anchors,dist_type),BatchNorm(input_dimension),SubspacePolicy(n_initial_anchors,input_dimension,output_dimension, n_layers, hidden_size,True))

def CriticAgent(n_anchors, input_dimension, n_layers, hidden_size):
    return CRLAgents(BatchNorm(input_dimension),Critic(n_anchors, input_dimension, n_layers, hidden_size,True))


class AlphaAgent(CRLAgent):
    def __init__(self, n_initial_anchors, dist_type = "flat"):
        super().__init__()
        self.n_anchors = n_initial_anchors
        self.dist_type = dist_type
        if self.dist_type == "flat":
            self.dist = Dirichlet(torch.ones(self.n_anchors))
        elif self.dist_type == "categorical":
            self.dist = Categorical(torch.ones(self.n_anchors))
        self.best_alpha = None
        self.id = nn.Parameter(torch.randn(1,1))

    def forward(self, t = None, replay = False, **args):
        device = self.id.device
        if self.best_alpha is None:
            if not t is None:
                B = self.workspace.batch_size()
                alphas =  self.dist.sample(torch.Size([B])).to(device)
                if isinstance(self.dist,Categorical):
                    alphas = F.one_hot(alphas,num_classes = self.n_anchors).float()
                if (t > 0) and (not replay):
                    done = self.get(("env/done", t)).float().unsqueeze(-1)
                    alphas_old = self.get(("alphas", t-1))
                    alphas =  alphas * done + alphas_old * (1 - done)
                self.set(("alphas", t), alphas)
        else:
            B = self.workspace.batch_size()
            alphas = self.best_alpha.unsqueeze(0).repeat(B,1).to(device)
            self.set(("alphas", t), alphas)

    def add_anchor(self):
        self.n_anchors += 1
        if self.dist_type == "flat":
            self.dist = Dirichlet(torch.ones(self.n_anchors))
        else:
            self.dist = Categorical(torch.ones(self.n_anchors))

    def set_task(self,task_id):
        if task_id >= self.n_anchors:
            self.best_alpha = torch.ones(self.n_anchors) / self.n_anchors
        else: 
            self.best_alpha = torch.eye(self.n_anchors)[task_id]

    def set_new_task(self,*args):
        self.best_alpha = None
        self.add_anchor()
        
class SubspacePolicy(CRLAgent):
    def __init__(self, n_initial_anchors, input_dimension,output_dimension, n_layers, hidden_size,use_normalized_obs=False):
        super().__init__()
        self.d_check = {}
        self.n_anchors = n_initial_anchors
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.iname="env/env_obs"
        if use_normalized_obs:
            self.iname="env/normalized_env_obs"
        input_size = input_dimension[0]
        num_outputs = output_dimension[0]
        hs = hidden_size
        hidden_layers = [LinearSubspace(n_initial_anchors,hs,hs) if i%2==0 else nn.ReLU() for i in range(2*(n_layers - 1))] if n_layers >1 else [nn.Identity()]
        self.model = Sequential(
            LinearSubspace(n_initial_anchors, input_size, hs),
            nn.ReLU(),
            *hidden_layers,
            LinearSubspace(n_initial_anchors, hs, num_outputs),
        )

    def forward(self, t = None, action_std = 0.0, **kwargs):
        if not self.training: 
            assert action_std==0.0

        if t is None:
            x = self.get(self.iname)
            alphas = self.get("alphas")
            mean = self.model(x,alphas)
            var = torch.ones_like(mean) * action_std + 0.000001
            dist = Normal(mean, var)
            action = self.get("action_before_tanh")
            logp_pi = dist.log_prob(action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(axis=-1)
            self.set("action_logprobs", logp_pi)
        else:
            x = self.get((self.iname, t))
            alphas = self.get(("alphas",t))
            mean = self.model(x,alphas)
            var = torch.ones_like(mean) * action_std + 0.000001
            dist = Normal(mean, var)
            action = dist.sample() if action_std > 0 else dist.mean
            self.set(("action_before_tanh", t), action)
            logp_pi = dist.log_prob(action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(axis=-1)
            self.set(("action_logprobs", t), logp_pi)
            action = torch.tanh(action)
            self.set(("action", t), action)

    def set_new_task(self,info = None):
        i = 0
        theta = [info] * (self.hidden_size + 2)
        for module in self.model:
            if isinstance(module,LinearSubspace):
                module.add_anchor(theta[i])
                i+=1
            self.n_anchors += 1

class Critic(CRLAgent):
    def __init__(self, n_anchors, input_dimension, n_layers, hidden_size,use_normalized_obs=False):
        super().__init__()
        self.iname="env/env_obs"
        self.n_anchors = n_anchors
        if use_normalized_obs:
            self.iname="env/normalized_env_obs"
            
        input_size = input_dimension[0] + self.n_anchors
        hs = hidden_size
        n_layers = n_layers
        hidden_layers = ([nn.Linear(hs, hs) if i % 2 == 0 else nn.ReLU() for i in range(2 * (n_layers - 1))] if n_layers > 1 else [nn.Identity()])
        self.model_critic = nn.Sequential(nn.Linear(input_size, hs),nn.ReLU(),*hidden_layers,nn.Linear(hs, 1))

    def forward(self, **kwargs):
        input = self.get(self.iname)
        alphas = self.get("alphas")
        x = torch.cat([input,alphas], dim=-1)
        critic = self.model_critic(x).squeeze(-1)
        self.set("critic", critic)