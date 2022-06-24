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

def FromscratchActionAgent(input_dimension,output_dimension, hidden_size, start_steps, layer_norm = False):
    """
    From scratch: the model is saved when the task is ended, and a new random one is created for the next task.
    """
    return CRLAgents(FromscratchAction(input_dimension,output_dimension, hidden_size, start_steps, input_name = "env/env_obs", layer_norm = layer_norm))

def EWCActionAgent(input_dimension,output_dimension, hidden_size, fisher_coeff, start_steps, layer_norm = False):
    """
    EWC regularizer added on top of the ActionAgent model. (see )
    """
    return CRLAgents(EWCAction(input_dimension,output_dimension, hidden_size, fisher_coeff, start_steps, input_name = "env/env_obs", layer_norm = layer_norm))

def L2ActionAgent(input_dimension,output_dimension, hidden_size, l2_coeff, start_steps, layer_norm = False):
    """
    L2 regularizer added on top of the ActionAgent model.
    """
    return CRLAgents(L2Action(input_dimension,output_dimension, hidden_size, l2_coeff, start_steps, input_name = "env/env_obs", layer_norm = layer_norm))

def PNNActionAgent(input_dimension,output_dimension, hidden_size, start_steps,layer_norm):
    """
    PNN Agent 
    """
    return CRLAgents(PNNAction(input_dimension,output_dimension, hidden_size, start_steps, input_name = "env/env_obs",layer_norm = layer_norm))

def TwinCritics(obs_dimension, action_dimension, hidden_size):
    """
    Twin q value functions for SAC algorithm.
    """
    return CRLAgents(Critic(obs_dimension, action_dimension, hidden_size, input_name = "env/env_obs",output_name = "q1"),Critic(obs_dimension, action_dimension, hidden_size, input_name = "env/env_obs", output_name = "q2"))


class Normalizer(CRLAgent):
    """
    Pre-trained normalizer over Brax envs. Helps to compare models fairly.
    """
    def __init__(self,input_dimension):
        super().__init__()
        if input_dimension == 23: #halfcheetah
            self.running_mean = nn.Parameter(torch.Tensor([ 6.1431e-01,  4.5919e-01,  0.0000e+00, -6.2606e-03,  0.0000e+00,
                                                            1.1327e-01, -6.0021e-02, -1.5187e-01, -2.2399e-01, -4.0081e-01,
                                                            -2.8977e-01,  6.5863e+00,  0.0000e+00, -7.2588e-03,  0.0000e+00,
                                                            1.6598e-01,  0.0000e+00,  5.8859e-02, -6.8729e-02,  8.9783e-02,
                                                            1.1471e-01, -1.9337e-01,  1.9044e-01]),requires_grad = False)
            self.std = nn.Parameter(torch.sqrt(torch.Tensor([2.7747e-02, 7.0441e-01, 5.6052e-45, 5.3661e-02, 5.6052e-45, 4.2445e-01,
                                                            3.5026e-01, 1.3651e-01, 4.2359e-01, 5.5605e-01, 9.4230e-02, 5.3188e+00,
                                                            5.6052e-45, 1.9010e+00, 5.6052e-45, 1.0593e+01, 5.6052e-45, 1.5619e+02,
                                                            2.1769e+02, 7.1641e+02, 2.4682e+02, 1.0647e+03, 9.3556e+02])),requires_grad = False)
        elif input_dimension == 27: #ant
            self.running_mean = nn.Parameter(torch.Tensor([ 0.6112,  0.9109, -0.0210, -0.0481,  0.2029, -0.0707,  0.6687,  0.0399,
                                                            -0.6143,  0.0655, -0.6917, -0.1086,  0.6811,  4.4375, -0.2056, -0.0135,
                                                            0.0437,  0.0760,  0.0340, -0.0578, -0.1628,  0.0781,  0.2136,  0.0246,
                                                            0.1336, -0.0270, -0.0235]),requires_grad = False)
            self.std = nn.Parameter(torch.sqrt(torch.Tensor([1.4017e-02, 4.3541e-02, 2.2925e-02, 2.3364e-02, 3.6594e-02, 1.4881e-01,
                                                            4.0282e-02, 1.3600e-01, 2.6437e-02, 1.4056e-01, 4.4391e-02, 1.4609e-01,
                                                            5.3686e-02, 2.6051e+00, 1.0322e+00, 6.5472e-01, 2.7710e+00, 1.4680e+00,
                                                            6.2482e+00, 2.8523e+01, 1.3569e+01, 2.3991e+01, 1.1001e+01, 2.8217e+01,
                                                            1.6400e+01, 2.5816e+01, 1.4181e+01])),requires_grad = False)
        

    def forward(self, t = None, **kwargs):
        if not t is None:
            x = self.get(("env/env_obs", t))
            x = (x - self.running_mean) / self.std
            self.set(("env/normalized_env_obs", t), x)

class BatchNorm(CRLAgent):
    """
    Apply batch normalization on "env/env_obs" variable and store it in "env/normalized_env_obs".
    At each change of task, the last BN network is saved and cloned for the new task.
    """
    def __init__(self,input_dimension):
        super().__init__()
        self.num_features = input_dimension
        self.bn = nn.ModuleList([nn.BatchNorm1d(num_features=self.num_features)])
        self.task_id = 0
    
    def forward(self, t=None, **kwargs):
        model_id = min(self.task_id, len(self.bn)-1)
        if not t is None:
            input = self.get(("env/env_obs", t))
            input = self.bn[model_id](input)
            self.set(("env/normalized_env_obs", t), input)

    def set_task(self,task_id = None):
        if task_id is None:
            self.bn.append(copy.deepcopy(self.bn[-1]))
            self.bn[-1].state_dict()["num_batches_tracked"] = 0
            self.task_id = len(self.bn) - 1
        else:
            self.task_id = task_id

class Action(CRLAgent):
    def __init__(self, input_dimension,output_dimension, hidden_size, start_steps = 0, input_name = "env/env_obs", layer_norm = False):
        super().__init__()
        self.start_steps = start_steps
        self.counter = 0
        self.iname = input_name
        self.task_id = 0
        self.output_dimension = output_dimension
        self.hs = hidden_size
        self.input_size = input_dimension
        self.layer_norm = layer_norm
        
        self.model = nn.ModuleList([self.make_model()])

    def make_model(self):
        return nn.Sequential(
        nn.Linear(self.input_size,self.hs),
        nn.LeakyReLU(negative_slope=0.2),
        nn.Linear(self.hs,self.hs),
        nn.LeakyReLU(negative_slope=0.2),
        nn.Linear(self.hs,self.hs),
        nn.LeakyReLU(negative_slope=0.2),
        nn.Linear(self.hs,self.output_dimension * 2),
    )

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
            self.model.append(self.make_model())
            self.task_id = len(self.model) - 1
        else:
            self.task_id = task_id


class EWCAction(Action):
    def __init__(self, input_dimension,output_dimension, hidden_size, fisher_coeff, start_steps = 0, input_name = "env/env_obs", layer_norm = False):
        super().__init__(input_dimension,output_dimension, hidden_size, start_steps, input_name,layer_norm)
        self.fisher_coeff = fisher_coeff
        self.regularize = False

    def register_and_consolidate(self,fisher_diagonals):
        param_names = [n.replace('.', '_') for n, p in  self.model.named_parameters()]
        fisher_dict={n: f.detach() for n, f in zip(param_names, fisher_diagonals)}
        for name, param in self.model.named_parameters():
            name = name.replace('.', '_')
            self.model.register_buffer(f"{name}_mean", param.data.clone())
            if self.regularize:
                fisher = getattr(self.model, f"{name}_fisher") + fisher_dict[name].data.clone() ## add to the old fisher coeff
            else:
                fisher =  fisher_dict[name].data.clone()
            self.model.register_buffer(f"{name}_fisher", fisher)
        self.regularize = True
    
    def add_regularizer(self):
        if self.regularize:
            losses = []
            for name, param in self.model.named_parameters():
                name = name.replace('.', '_')
                mean = getattr(self.model, f"{name}_mean")
                fisher = getattr(self.model,f"{name}_fisher")
                losses.append((fisher * (param - mean)**2).sum())
           
          
            return (self.fisher_coeff)*sum(losses).view(1).to(list(self.parameters())[0].device)
        else:
            return torch.Tensor([0.]).to(list(self.parameters())[0].device)

class L2Action(Action):
    def __init__(self, input_dimension,output_dimension, hidden_size, l2_coeff, start_steps = 0, input_name = "env/env_obs", layer_norm = False):
        super().__init__(input_dimension,output_dimension, hidden_size, start_steps, input_name,layer_norm)
        self.l2_coeff = l2_coeff
        self.regularize = False

    def register_and_consolidate(self):
        for name, param in self.model.named_parameters():
            name = name.replace('.', '_')
            self.model.register_buffer(f"{name}_mean", param.data.clone())
        self.regularize = True
    
    def add_regularizer(self):
        if self.regularize:
            losses = []
            for name, param in self.model.named_parameters():
                name = name.replace('.', '_')
                mean = getattr(self.model, f"{name}_mean")
                losses.append(((param - mean.detach())**2).sum())
            return (self.l2_coeff)*sum(losses).view(1).to(list(self.parameters())[0].device)
        else:
            return torch.Tensor([0.]).to(list(self.parameters())[0].device)

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

        self.output_dimension = output_dimension
        self.hs = hidden_size
        self.input_size = input_dimension
      
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

class Critic(CRLAgent):
    def __init__(self, obs_dimension, action_dimension, hidden_size, input_name = "env/env_obs", output_name = "q"):
        super().__init__()

        self.iname = input_name 
        input_size = obs_dimension + action_dimension
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