#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
from torch import nn
from salina_cl.core import CRLAgent, CRLAgents

def BraxActionAgent(input_dimension,output_dimension, n_layers, hidden_size):
    return CRLAgents(Normalizer(input_dimension),Action(input_dimension,output_dimension, n_layers, hidden_size))

def BraxTwinCritics(obs_dimension, action_dimension, n_layers, hidden_size):
    return CRLAgents(Critic(obs_dimension, action_dimension, n_layers, hidden_size, output_name = "q1"),Critic(obs_dimension, action_dimension, n_layers, hidden_size, output_name = "q2"))

def CWActionAgent(input_dimension,output_dimension, n_layers, hidden_size):
    return CRLAgents(CWAction(input_dimension,output_dimension, n_layers, hidden_size, input_name = "env/env_obs"))

def CWTwinCritics(obs_dimension, action_dimension, n_layers, hidden_size):
    return CRLAgents(CWCritic(obs_dimension, action_dimension, n_layers, hidden_size, input_name = "env/env_obs",output_name = "q1"),CWCritic(obs_dimension, action_dimension, n_layers, hidden_size, input_name = "env/env_obs", output_name = "q2"))
   
class Normalizer(CRLAgent):
    """
    Pre-trained normalizer over Brax envs. Helps to compare models fairly.
    """
    def __init__(self,input_dimension):
        super().__init__()
        if input_dimension[0] == 23: #halfcheetah
            self.running_mean = nn.Parameter(torch.Tensor([ 6.1431e-01,  4.5919e-01,  0.0000e+00, -6.2606e-03,  0.0000e+00,
                                                            1.1327e-01, -6.0021e-02, -1.5187e-01, -2.2399e-01, -4.0081e-01,
                                                            -2.8977e-01,  6.5863e+00,  0.0000e+00, -7.2588e-03,  0.0000e+00,
                                                            1.6598e-01,  0.0000e+00,  5.8859e-02, -6.8729e-02,  8.9783e-02,
                                                            1.1471e-01, -1.9337e-01,  1.9044e-01]),requires_grad = False)
            self.std = nn.Parameter(torch.sqrt(torch.Tensor([2.7747e-02, 7.0441e-01, 5.6052e-45, 5.3661e-02, 5.6052e-45, 4.2445e-01,
                                                            3.5026e-01, 1.3651e-01, 4.2359e-01, 5.5605e-01, 9.4230e-02, 5.3188e+00,
                                                            5.6052e-45, 1.9010e+00, 5.6052e-45, 1.0593e+01, 5.6052e-45, 1.5619e+02,
                                                            2.1769e+02, 7.1641e+02, 2.4682e+02, 1.0647e+03, 9.3556e+02])),requires_grad = False)
        elif input_dimension[0] == 27: #ant
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


class Action(CRLAgent):
    def __init__(self, input_dimension,output_dimension, n_layers, hidden_size, input_name = "env/normalized_env_obs"):
        super().__init__()
        self.iname = input_name
        self.task_id = 0
            
        input_size = input_dimension[0]
        # import ipdb;ipdb.set_trace()
        num_outputs = output_dimension[0]
        hs = hidden_size
        n_layers = n_layers
        hidden_layers = ([nn.Linear(hs, hs) if i % 2 == 0 else nn.ReLU() for i in range(2 * (n_layers - 1))] if n_layers > 1 else [nn.Identity()])
        self.model = nn.Sequential(nn.Linear(input_size, hs),nn.ReLU(),*hidden_layers,nn.Linear(hs, num_outputs))

    def forward(self, t = None, epsilon = 0.000001, epsilon_clip=100000, **kwargs):
        input = self.get(self.iname if t is None else (self.iname, t)).detach()
        action = self.model(input)
        action = torch.tanh(action)
        s = action.size()
        noise = torch.randn(*s, device=action.device) * epsilon
        noise = torch.clip(noise, min=-epsilon_clip, max=epsilon_clip)
        action = action + noise
        action = torch.clip(action, min=-1.0, max=1.0)
        self.set("action" if t is None else ("action", t), action)         
           
class Critic(CRLAgent):
    def __init__(self, obs_dimension, action_dimension, n_layers, hidden_size, input_name = "env/normalized_env_obs", output_name = "q"):
        super().__init__()
        self.iname = input_name 
        input_size = obs_dimension[0] + action_dimension[0]
        hs = hidden_size
        n_layers = n_layers
        self.output_name = output_name
        hidden_layers = ([nn.Linear(hs, hs) if i % 2 == 0 else nn.ReLU() for i in range(2 * (n_layers - 1))] if n_layers > 1 else [nn.Identity()])
        self.model = nn.Sequential(nn.Linear(input_size, hs), nn.ReLU(), *hidden_layers, nn.Linear(hs, 1))

    def forward(self, detach_action=False,**kwargs):
        input = self.get(self.iname).detach()
        action = self.get(("action"))
        if detach_action:
            action = action.detach()
        input = torch.cat([input, action], dim=-1)
        critic = self.model(input).squeeze(-1)
        self.set(self.output_name, critic)

class CWAction(CRLAgent):
    def __init__(self, input_dimension,output_dimension, n_layers, hidden_size, input_name = "env/env_obs"):
        super().__init__()
        self.iname = input_name
        self.task_id = 0
        hs = hidden_size
        input_size = input_dimension[0]
        
        self.model = nn.Sequential(
            nn.Linear(input_size,hs),
            nn.LayerNorm(hs),
            nn.Tanh(),
            nn.Linear(hs,hs),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(hs,hs),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(hs,output_dimension[0]),
        )

    def forward(self, t = None, epsilon = 0.000001, epsilon_clip=100000, **kwargs):
        input = self.get(self.iname if t is None else (self.iname, t)).detach()
        action = self.model(input)
        action = torch.tanh(action)
        s = action.size()
        noise = torch.randn(*s, device=action.device) * epsilon
        noise = torch.clip(noise, min=-epsilon_clip, max=epsilon_clip)
        action = action + noise
        action = torch.clip(action, min=-1.0, max=1.0)
        self.set("action" if t is None else ("action", t), action)

class CWCritic(CRLAgent):
    def __init__(self, obs_dimension, action_dimension, n_layers, hidden_size, input_name = "env/env_obs", output_name = "q"):
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
