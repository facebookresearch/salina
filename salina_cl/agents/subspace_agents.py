#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np
import copy
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions.normal import Normal
from salina_cl.core import CRLAgent, CRLAgents
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.categorical import Categorical
from salina_cl.agents.single_agents import Normalizer
from salina_cl.agents.tools import LinearSubspace, Sequential
from salina_cl.agents.tools import weight_init

class SubspaceAgents(CRLAgents):
    def add_anchor(self, **kwargs):
        for agent in self:
            agent.add_anchor(**kwargs)
    def remove_anchor(self, **kwargs):
        for agent in self:
            agent.remove_anchor(**kwargs)
    def set_best_alpha(self, **kwargs):
        for agent in self:
            agent.set_best_alpha(**kwargs)

class SubspaceAgent(CRLAgent):
    def add_anchor(self,**kwargs):
        pass
    def remove_anchor(self,**kwargs):
        pass
    def set_best_alpha(self,**kwargs):
        pass

class Normalizer(Normalizer,SubspaceAgent): pass

def SubspaceActionAgent(n_initial_anchors, dist_type, refresh_rate, input_dimension,output_dimension, n_layers, hidden_size):
    return SubspaceAgents(AlphaAgent(n_initial_anchors,dist_type,refresh_rate),Normalizer(input_dimension),SubspacePolicy(n_initial_anchors,input_dimension,output_dimension, n_layers, hidden_size))

def CriticAgent(n_anchors, input_dimension, n_layers, hidden_size):
    return SubspaceAgents(Critic(n_anchors, input_dimension, n_layers, hidden_size))

class BatchNorm(SubspaceAgent):
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

    def set_task(self,task_id):
        self.task_id = task_id

    def add_anchor(self, logger = None, **kwargs):
        if not (logger is None):
            logger = logger.get_logger(type(self).__name__+str("/"))
            logger.message("Copying model and setting 'num_batches_tracked' to zero")
        self.bn.append(copy.deepcopy(self.bn[-1]))
        self.bn[-1].state_dict()["num_batches_tracked"] = 0
        self.task_id = len(self.bn) - 1

class FreezeBatchNorm(SubspaceAgent):
    """
    This batchnorm is frozen after task 1. More convenient to compare methods.
    """
    def __init__(self,input_dimension):
        super().__init__()
        self.num_features = input_dimension[0]
        self.bn = nn.BatchNorm1d(num_features=self.num_features)
        self.stop_update = False
    
    def forward(self, t=None, **kwargs):
        if self.stop_update:
            self.eval()
        if not t is None:
            input = self.get(("env/env_obs", t))
            input = self.bn(input)
            self.set(("env/normalized_env_obs", t), input)

    def add_anchor(self,task_id = None, **kwargs):
        for params in self.parameters():
            params.requires_grad = False
        self.stop_update = True

class AlphaAgent(SubspaceAgent):
    def __init__(self, n_initial_anchors, dist_type = "flat", refresh_rate = 1.):
        super().__init__()
        self.n_anchors = n_initial_anchors
        self.dist_type = dist_type
        self.refresh_rate = refresh_rate
        if self.dist_type == "flat":
            self.dist = Dirichlet(torch.ones(self.n_anchors))
        elif self.dist_type == "categorical":
            self.dist = Categorical(torch.ones(self.n_anchors))
        elif self.dist_type == "last_anchor":
            self.dist = Categorical(torch.Tensor([0] * (self.n_anchors-1) + [1]))
        self.best_alpha = None
        self.best_alphas = torch.Tensor([])
        self.id = nn.Parameter(torch.randn(1,1))

    def forward(self, t = None, k_shot = False, force_random = False, **args):
        device = self.id.device
        if k_shot:
            if t > 0:
                alphas = self.get(("alphas", t-1))
                self.set(("alphas", t), alphas)
        elif ((not t is None) and self.training) or force_random:
            B = self.workspace.batch_size()
            alphas =  self.dist.sample(torch.Size([B])).to(device)
            if isinstance(self.dist,Categorical):
                alphas = F.one_hot(alphas,num_classes = self.n_anchors).float()
            if t > 0:
                done = self.get(("env/done", t)).float().unsqueeze(-1)
                if (done.sum() > 0) and (self.refresh_rate<1.):
                    cr = self.get(("env/cumulated_reward", t-1))
                    k = max(int(len(cr) * (1 - self.refresh_rate)) - 1, 0)
                    threshold = sorted(cr,reverse = True)[k]
                    refresh_variable = (cr < threshold).float().unsqueeze(-1)
                    done *= refresh_variable
                alphas_old = self.get(("alphas", t-1))
                alphas =  alphas * done + alphas_old * (1 - done)
            self.set(("alphas", t), alphas)
        elif not self.training:
            B = self.workspace.batch_size()
            alphas = self.best_alpha.unsqueeze(0).repeat(B,1).to(device)
            self.set(("alphas", t), alphas)

    def set_best_alpha(self, alpha = None, logger = None,**kwargs):
        device = self.id.device

        if alpha is None:
            alpha = torch.Tensor([0.] * (self.n_anchors - 1) + [1.]).to(device)
        else:
            alpha = alpha.to(device)
        self.best_alphas = torch.cat([self.best_alphas.to(device),alpha.unsqueeze(0)],dim=0)

        if not (logger is None):
            logger = logger.get_logger(type(self).__name__+str("/"))
            if alpha is None:
                logger.message("Set best_alpha = None")
            else:
                logger.message("Set best_alpha = "+str(list(map(lambda x:round(x,2),alpha.tolist()))))
            
    def add_anchor(self, logger = None,**kwargs):
        device = self.id.device
        self.n_anchors += 1
        self.best_alphas = torch.cat([self.best_alphas.to(device),torch.zeros(self.best_alphas.shape[0],1).to(device)],dim=-1)
        if self.dist_type == "flat":
            self.dist = Dirichlet(torch.ones(self.n_anchors))
        else:
            self.dist = Categorical(torch.ones(self.n_anchors))
        logger.message("Increasing alpha size to "+str(self.n_anchors))

    def remove_anchor(self, logger = None,**kwargs):
        self.n_anchors -= 1
        self.best_alphas = self.best_alphas[:,:-1]
        if self.dist_type == "flat":
            self.dist = Dirichlet(torch.ones(self.n_anchors))
        else:
            self.dist = Categorical(torch.ones(self.n_anchors))
        logger.message("Decreasing alpha size to "+str(self.n_anchors))

    def set_task(self,task_id):
        if task_id >= self.best_alphas.shape[0]:
            self.best_alpha = torch.ones(self.n_anchors) / self.n_anchors
        else: 
            self.best_alpha = self.best_alphas[task_id]
        
class SubspacePolicy(SubspaceAgent):
    def __init__(self, n_initial_anchors, input_dimension,output_dimension, n_layers, hidden_size, input_name = "env/normalized_env_obs"):
        super().__init__()
        self.d_check = {}
        self.n_anchors = n_initial_anchors
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.iname = input_name
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
            done = self.get(("env/done", t)).float()
            if t == 0:
                self.done_monitor = 1 - done
            else:
                self.done_monitor *= (1 - done)
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

    def add_anchor(self,alpha = None, logger = None, **kwargs):
        i = 0
        alphas = [alpha] * (self.hidden_size + 2)
        if not (logger is None):
            logger = logger.get_logger(type(self).__name__+str("/"))
            if alpha is None:
                logger.message("Adding one anchor with alpha = None")
            else:
                logger.message("Adding one anchor with alpha = "+str(list(map(lambda x:round(x,2),alpha.tolist()))))
        for module in self.model:
            if isinstance(module,LinearSubspace):
                module.add_anchor(alphas[i])
                ### Sanity check
                #if i == 0:
                #    for j,anchor in enumerate(module.anchors):
                #        print("--- anchor",j,":",anchor.weight[0].data[:4])
                i+=1
        self.n_anchors += 1

    def remove_anchor(self, logger = None, **kwargs):
        if not (logger is None):
            logger = logger.get_logger(type(self).__name__+str("/"))
            logger.message("Removing last anchor")
        for module in self.model:
            if isinstance(module,LinearSubspace):
                module.anchors = module.anchors[:-1]
                module.n_anchors -= 1
        self.n_anchors -= 1

class Critic(SubspaceAgent):
    def __init__(self, n_anchors, input_dimension, n_layers, hidden_size, input_name = "env/normalized_env_obs"):
        super().__init__()
        self.iname = input_name 
        self.n_anchors = n_anchors
        self.input_size = input_dimension[0]
        self.hs = hidden_size
        self.n_layers = n_layers
        hidden_layers = ([nn.Linear(self.hs, self.hs) if i % 2 == 0 else nn.ReLU() for i in range(2 * (n_layers - 1))] if n_layers > 1 else [nn.Identity()])
        self.model_critic = nn.Sequential(nn.Linear(self.input_size + self.n_anchors, self.hs),nn.ReLU(),*hidden_layers,nn.Linear(self.hs, 1))
        self.model_critic.apply(weight_init)

    def forward(self, **kwargs):
        input = self.get(self.iname).detach()
        alphas = self.get("alphas").detach()
        x = torch.cat([input,alphas], dim=-1)
        critic = self.model_critic(x).squeeze(-1)
        self.set("critic", critic)

    def add_anchor(self, logger = None,**kwargs):
        self.__init__(self.n_anchors + 1, [self.input_size], self.n_layers, self.hs)
        if not (logger is None):
            logger = logger.get_logger(type(self).__name__+str("/"))
            logger.message("Setting input size to "+str(self.input_size + self.n_anchors)+" and reinitializing network")

    def remove_anchor(self, logger = None,**kwargs):
        self.__init__(self.n_anchors - 1, [self.input_size], self.n_layers, self.hs)
        if not (logger is None):
            logger = logger.get_logger(type(self).__name__+str("/"))
            logger.message("Setting input size to "+str(self.input_size + self.n_anchors)+" and reinitializing network")