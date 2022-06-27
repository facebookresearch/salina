#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import numpy as np
from torch import nn
from salina_cl.core import CRLAgent, CRLAgents
import torch.nn.functional as F
from salina_cl.agents.tools import LinearSubspace, Sequential, create_dist
from torch.distributions.categorical import Categorical

def SubspaceActionAgent(n_initial_anchors, dist_type, refresh_rate, input_dimension,output_dimension, hidden_size, start_steps, resampling_q, resampling_policy, repeat_alpha):
    """
    ActionAgent that is using "alphas" variable during forward to compute a convex combination of its anchor policies.
    """
    return SubspaceAgents(AlphaAgent(n_initial_anchors, dist_type, refresh_rate, resampling_q, resampling_policy, repeat_alpha),
                          SubspaceAction(n_initial_anchors,input_dimension,output_dimension, hidden_size, start_steps)
                          )

def SubspaceActionAgent_cw(n_initial_anchors, dist_type, refresh_rate, input_dimension,output_dimension, hidden_size, start_steps, resampling_q, resampling_policy, repeat_alpha):
    """
    Only the head is a subspace.
    """
    return SubspaceAgents(AlphaAgent(n_initial_anchors, dist_type, refresh_rate, resampling_q, resampling_policy, repeat_alpha),
                          BackBoneCW(n_initial_anchors,input_dimension,output_dimension, hidden_size, start_steps),
                          SubspaceAction(n_initial_anchors,input_dimension,output_dimension, hidden_size, start_steps)
                          )

def TwinCritics(n_anchors, obs_dimension, action_dimension, hidden_size):
    """
    Twin critics model used for SAC. In addition to the (obs,actions), they also take the convex combination alpha as as input.
    """
    return SubspaceAgents(Critic(n_anchors, obs_dimension, action_dimension, hidden_size, output_name = "q1"),
                          Critic(n_anchors, obs_dimension, action_dimension, hidden_size, output_name = "q2")
                          )

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
    def add_anchor(self, **kwargs):
        pass
    def remove_anchor(self, **kwargs):
        pass
    def set_best_alpha(self, **kwargs):
        pass

class AlphaAgent(SubspaceAgent):
    def __init__(self, n_initial_anchors, dist_type = "flat", refresh_rate = 1., resampling_q = True, resampling_policy = True, repeat_alpha = 1000):
        super().__init__()
        self.n_anchors = n_initial_anchors
        self.dist_type = dist_type
        self.refresh_rate = refresh_rate
        self.dist = create_dist(self.dist_type,self.n_anchors)
        self.dist2 = create_dist("flat",self.n_anchors - 1)
        self.best_alpha = None
        self.best_alphas = torch.Tensor([])
        self.id = nn.Parameter(torch.randn(1,1))
        self.resampling_q = resampling_q
        self.resampling_policy = resampling_policy
        self.repeat_alpha = repeat_alpha

    #reward tracking
    def track_reward(self,t = None):
        if not t is None:
            if t == 0:
                r = self.get(("env/reward", t))
                self.set(("tracking_reward",t),r)
            elif t > 0:
                r = self.get(("env/reward", t))
                old_tracking_reward = self.get(("tracking_reward", t - 1))
                refresh_timestep = ((self.get(("env/timestep", t - 1)) % self.repeat_alpha) == 0).float()
                tracking_reward = r + old_tracking_reward * (1 - refresh_timestep)
                self.set(("tracking_reward",t),tracking_reward)

    def forward(self, t = None, force_random = False, q_update = False, policy_update = False, mute_alpha = False,**args):
        device = self.id.device
        self.track_reward(t)
        if mute_alpha:
            self.set(("alphas", t), self.get(("alphas", max(t-1,0))))
        elif (not self.training) and (not force_random):
            B = self.workspace.batch_size()
            alphas = self.best_alpha.unsqueeze(0).repeat(B,1).to(device)
            self.set(("alphas", t), alphas)
        elif not (t is None):
            B = self.workspace.batch_size()
            # Sampling in the new subspace AND the former subspace
            alphas1 =  self.dist.sample(torch.Size([B // 2])).to(device)
            alphas2 =  self.dist2.sample(torch.Size([B - (B // 2)])).to(device)
            if alphas2.shape[-1] < alphas1.shape[-1]:
                alphas2 = torch.cat([alphas2,torch.zeros(*alphas2.shape[:-1],1).to(device)],dim=-1)
            alphas = torch.cat([alphas1,alphas2], dim = 0)
            if isinstance(self.dist,Categorical):
                alphas = F.one_hot(alphas,num_classes = self.n_anchors).float()
            if t > 0 and self.repeat_alpha > 1:
                done = self.get(("env/done", t)).float().unsqueeze(-1)
                refresh_timestep = ((self.get(("env/timestep", t)) % self.repeat_alpha) == 0).float().unsqueeze(-1)
                refresh = torch.max(done,refresh_timestep)
                if ((done.sum() > 0) or (refresh_timestep.sum() > 0) ) and (self.refresh_rate<1.):
                    cr = self.get(("tracking_reward", t))
                    k = max(int(len(cr) * (1 - self.refresh_rate)) - 1, 0)
                    threshold = sorted(cr,reverse = True)[k]
                    refresh_condition = (cr < threshold).float().unsqueeze(-1)
                    refresh *= refresh_condition
                alphas_old = self.get(("alphas", t-1))
                alphas =  alphas * refresh + alphas_old * (1 - refresh)
            self.set(("alphas", t), alphas)
        elif q_update:
            if self.resampling_q:
                T = self.workspace.time_size()
                B = self.workspace.batch_size()
                alphas1 =  self.dist.sample(torch.Size([T, B // 2])).to(device)
                alphas2 =  self.dist2.sample(torch.Size([T, B - (B // 2)])).to(device)
                if alphas2.shape[-1] < alphas1.shape[-1]:
                    alphas2 = torch.cat([alphas2,torch.zeros(*alphas2.shape[:-1],1).to(device)],dim=-1)
                alphas = torch.cat([alphas1,alphas2], dim = 1)
                if isinstance(self.dist,Categorical):
                    alphas = F.one_hot(alphas,num_classes = self.n_anchors).float()
                self.set("alphas_q_update", alphas)
            else:
                self.set("alphas_q_update", self.get("alphas"))
        elif policy_update:
            if self.resampling_policy:
                T = self.workspace.time_size()
                B = self.workspace.batch_size()
                alphas =  self.dist.sample(torch.Size([T,B])).to(device)
                if isinstance(self.dist,Categorical):
                    alphas = F.one_hot(alphas,num_classes = self.n_anchors).float()
                self.set("alphas_policy_update", alphas)
            else:
                self.set("alphas_policy_update", self.get("alphas"))

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
        self.dist = create_dist(self.dist_type,self.n_anchors)
        self.dist2 = create_dist("flat",self.n_anchors - 1)
        if not logger is None:
            logger = logger.get_logger(type(self).__name__+str("/"))
            logger.message("Increasing alpha size to "+str(self.n_anchors))

    def remove_anchor(self, logger = None,**kwargs):
        self.n_anchors -= 1
        self.best_alphas = self.best_alphas[:,:-1]
        self.dist = create_dist(self.dist_type,self.n_anchors)
        self.dist2 = create_dist("flat",self.n_anchors - 1)
        if not logger is None:
            logger = logger.get_logger(type(self).__name__+str("/"))
            logger.message("Decreasing alpha size to "+str(self.n_anchors))

    def set_task(self,task_id):
        if task_id >= self.best_alphas.shape[0]:
            self.best_alpha = torch.ones(self.n_anchors) / self.n_anchors
        else: 
            self.best_alpha = self.best_alphas[task_id]

class SubspaceAction(SubspaceAgent):
    def __init__(self, n_initial_anchors, input_dimension, output_dimension, hidden_size, start_steps = 0, input_name = "env/env_obs"):
        super().__init__()
        self.start_steps = start_steps
        self.counter = 0
        self.iname = input_name
        self.task_id = 0
        self.n_anchors = n_initial_anchors
        self.input_size = input_dimension
        self.output_dimension = output_dimension
        self.hidden_size = hidden_size
        
        self.model = Sequential(
            LinearSubspace(self.n_anchors, self.input_size, self.hidden_size),
            nn.LeakyReLU(negative_slope=0.2),
            LinearSubspace(self.n_anchors, self.hidden_size, self.hidden_size),
            nn.LeakyReLU(negative_slope=0.2),
            LinearSubspace(self.n_anchors, self.hidden_size, self.hidden_size),
            nn.LeakyReLU(negative_slope=0.2),
            LinearSubspace(self.n_anchors, self.hidden_size, self.output_dimension * 2),
        )

    def forward(self, t = None, q_update = False, policy_update = False, **kwargs):
        if not self.training:
            x = self.get((self.iname, t))
            alphas = self.get(("alphas",t))
            mu, _ = self.model(x,alphas).chunk(2, dim=-1)
            action = torch.tanh(mu)
            self.set(("action", t), action)
        elif not (t is None):
            x = self.get((self.iname, t)).detach()
            alphas = self.get(("alphas",t))
            if self.counter <= self.start_steps:
                action = torch.rand(x.shape[0],self.output_dimension).to(x.device) * 2 - 1
            else:
                mu, log_std = self.model(x,alphas).chunk(2, dim=-1)
                log_std = torch.clip(log_std, min=-20., max=2.)
                std = log_std.exp()
                action = mu + torch.randn(*mu.shape).to(mu.device) * std
                action = torch.tanh(action)
            self.set(("action", t), action)
            self.counter += 1
        else:
            input = self.get(self.iname).detach()
            if q_update:
                alphas = self.get("alphas_q_update")
            elif policy_update:
                alphas = self.get("alphas_policy_update")
            else:
                alphas = self.get("alphas")
            mu, log_std = self.model(input,alphas).chunk(2, dim=-1)
            log_std = torch.clip(log_std, min=-20., max=2.)
            std = log_std.exp()
            action = mu + torch.randn(*mu.shape).to(mu.device) * std
            log_prob = (-0.5 * (((action - mu) / (std + 1e-8)) ** 2 + 2 * log_std + np.log(2 * np.pi))).sum(-1, keepdim=True)
            log_prob -= (2 * np.log(2) - action - F.softplus( - 2 * action)).sum(-1, keepdim=True)
            action = torch.tanh(action)
            self.set("action", action)
            self.set("action_logprobs", log_prob)

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

    def L2_norms(self):
        L2_norms = {}
        i = 1
        for module in self.model:
            if isinstance(module,LinearSubspace) and len(module.anchors)>1:
                L2_norms["layer_"+str(i)] = module.L2_norms()
                i += 1
        return L2_norms

    def cosine_similarities(self):
        cosine_similarities = {}
        i = 1
        for module in self.model:
            if isinstance(module,LinearSubspace) and len(module.anchors)>1:
                cosine_similarities["layer_"+str(i)] = module.cosine_similarities()
                i += 1
        return cosine_similarities

class SubspaceAction(SubspaceAgent):
    def __init__(self, n_initial_anchors, input_dimension, output_dimension, hidden_size, start_steps = 0, input_name = "env/env_obs"):
        super().__init__()
        self.start_steps = start_steps
        self.counter = 0
        self.iname = input_name
        self.task_id = 0
        self.n_anchors = n_initial_anchors
        self.input_size = input_dimension
        self.output_dimension = output_dimension
        self.hidden_size = hidden_size
        
        
        self.backbone = nn.Sequential(
            nn.Linear(self.input_size,self.hs),
            nn.LayerNorm(self.hs),
            nn.Tanh(),
            nn.Linear(self.hs,self.hs),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(self.hs,self.hs),
            nn.LeakyReLU(negative_slope=0.2))
        self.model = Sequential(LinearSubspace(self.n_anchors, self.hidden_size, self.output_dimension * 2))

    def forward(self, t = None, q_update = False, policy_update = False, **kwargs):
        if not self.training:
            x = self.get((self.iname, t))
            alphas = self.get(("alphas",t))
            x = self.backbone(x)
            mu, _ = self.model(x,alphas).chunk(2, dim=-1)
            action = torch.tanh(mu)
            self.set(("action", t), action)
        elif not (t is None):
            x = self.get((self.iname, t)).detach()
            alphas = self.get(("alphas",t))
            if self.counter <= self.start_steps:
                action = torch.rand(x.shape[0],self.output_dimension).to(x.device) * 2 - 1
            else:
                mu, log_std = self.model(x,alphas).chunk(2, dim=-1)
                log_std = torch.clip(log_std, min=-20., max=2.)
                std = log_std.exp()
                action = mu + torch.randn(*mu.shape).to(mu.device) * std
                action = torch.tanh(action)
            self.set(("action", t), action)
            self.counter += 1
        else:
            input = self.get(self.iname).detach()
            if q_update:
                alphas = self.get("alphas_q_update")
            elif policy_update:
                alphas = self.get("alphas_policy_update")
            else:
                alphas = self.get("alphas")
            mu, log_std = self.model(input,alphas).chunk(2, dim=-1)
            log_std = torch.clip(log_std, min=-20., max=2.)
            std = log_std.exp()
            action = mu + torch.randn(*mu.shape).to(mu.device) * std
            log_prob = (-0.5 * (((action - mu) / (std + 1e-8)) ** 2 + 2 * log_std + np.log(2 * np.pi))).sum(-1, keepdim=True)
            log_prob -= (2 * np.log(2) - action - F.softplus( - 2 * action)).sum(-1, keepdim=True)
            action = torch.tanh(action)
            self.set("action", action)
            self.set("action_logprobs", log_prob)

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

    def L2_norms(self):
        L2_norms = {}
        i = 1
        for module in self.model:
            if isinstance(module,LinearSubspace) and len(module.anchors)>1:
                L2_norms["layer_"+str(i)] = module.L2_norms()
                i += 1
        return L2_norms

    def cosine_similarities(self):
        cosine_similarities = {}
        i = 1
        for module in self.model:
            if isinstance(module,LinearSubspace) and len(module.anchors)>1:
                cosine_similarities["layer_"+str(i)] = module.cosine_similarities()
                i += 1
        return cosine_similarities

class Critic(SubspaceAgent):
    def __init__(self, n_anchors, obs_dimension, action_dimension, hidden_size, input_name = "env/env_obs", output_name = "q"):
        super().__init__()
        self.iname = input_name
        self.n_anchors = n_anchors
        self.obs_dimension = obs_dimension
        self.action_dimension= action_dimension
        self.input_size = self.obs_dimension + self.action_dimension + self.n_anchors
        self.hs = hidden_size
        self.output_name = output_name
        self.model = nn.Sequential(
            nn.Linear(self.input_size,self.hs),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(self.hs,self.hs),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(self.hs,self.hs),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(self.hs,1),
        )

    def forward(self, q_update: bool = False, policy_update: bool = False, **kwargs)-> None:
        input = self.get(self.iname).detach()
        action = self.get(("action"))
        if q_update:
            alphas = self.get("alphas_q_update")
        elif policy_update:
            alphas = self.get("alphas_policy_update")
        else:
            alphas = self.get("alphas")
        if alphas.shape[-1] < self.n_anchors:
            alphas = torch.cat([alphas,torch.zeros(*alphas.shape[:-1],self.n_anchors - alphas.shape[-1]).to(alphas.device)], dim = -1)
        input = torch.cat([input, action, alphas], dim=-1)
        critic = self.model(input).squeeze(-1)
        self.set(self.output_name, critic)

    def add_anchor(self, n_anchors = None, logger = None,**kwargs)-> None:
        self.__init__(self.n_anchors if n_anchors is None else n_anchors, self.obs_dimension, self.action_dimension, self.hs, input_name = self.iname, output_name = self.output_name)
        if not (logger is None):
            logger = logger.get_logger(type(self).__name__+str("/"))
            logger.message("Setting input size to "+str(self.input_size)+" and reinitializing network")