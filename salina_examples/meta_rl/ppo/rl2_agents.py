#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import gym
import numpy as np
import torch
import torch.nn.functional as F
from brax.envs import _envs, create_gym_env, wrappers
from gym.wrappers import TimeLimit
from torch import nn
from torch.distributions.normal import Normal

from salina import Agent, instantiate_class
from salina.agents import Agents
from salina_examples.meta_rl.env_tools import make_env

def masked_tensor(tensor0, tensor1, mask):
    """Compute a tensor by combining two tensors with a mask

    :param tensor0: a Bx(N) tensor
    :type tensor0: torch.Tensor
    :param tensor1: a Bx(N) tensor
    :type tensor1: torch.Tensor
    :param mask: a B tensor
    :type mask: torch.Tensor
    :return: (1-m) * tensor 0 + m *tensor1 (averafging is made ine by line)
    :rtype: tensor0.dtype
    """
    s = tensor0.size()
    assert s[0] == mask.size()[0]
    m = mask
    for i in range(len(s) - 1):
        m = mask.unsqueeze(-1)
    m = m.repeat(1, *s[1:])
    m = m.float()
    out = ((1.0 - m) * tensor0 + m * tensor1).type(tensor0.dtype)
    return out

class GRUAgent(Agent):
    def __init__(self, env, hidden_size,output_name):
        super().__init__()
        env = make_env(env)
        input_size = env.observation_space.shape[0]
        self.action_size = env.action_space.shape[0]
        self.output_name=output_name
        self.gru=nn.GRUCell(input_size+self.action_size,hidden_size)
        self.hidden_size=hidden_size

    def forward(self, t=None, **kwargs):
        if t is None:
            for t in range(1,self.workspace.time_size()):
                self.forward(t)
        else:
            ts=self.get(("env/meta/timestep",t))
            B=ts.size()[0]
            _initial_state=torch.zeros(B,self.hidden_size).to(ts.device)
            _empty_action=torch.zeros(B,self.action_size).to(ts.device)
            _previous_state=None
            if t==0:
                assert ts.eq(0).all()
                _previous_state=_initial_state
                _previous_action=_empty_action
            else:
                _previous_state=self.get((self.output_name,t-1))
                _previous_state=masked_tensor(_previous_state,_initial_state,ts.eq(0))
                _previous_action=self.get(("action",t-1))
                _previous_action=masked_tensor(_previous_action,_empty_action,ts.eq(0))

            obs=self.get(("env/env_obs",t))
            obs=torch.cat([obs,_previous_action],dim=1)
            new_z=self.gru(obs,_previous_state)
            self.set((self.output_name,t),new_z)

class ActionAgent(Agent):
    def __init__(self,env, n_layers, hidden_size,input_name):
        super().__init__()
        env = make_env(env)
        self.input_name=input_name
        num_outputs = env.action_space.shape[0]
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
            nn.Linear(hidden_size, hs),
            nn.ReLU(),
            *hidden_layers,
            nn.Linear(hs, num_outputs),
        )

    def forward(self, t=None, replay=False, action_std=0.1, **kwargs):
        if replay:
            assert t == None
            input = self.get(self.input_name)
            mean = self.model(input)
            var = torch.ones_like(mean) * action_std + 0.000001
            dist = Normal(mean, var)
            action = self.get("action_before_tanh")
            logp_pi = dist.log_prob(action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(axis=-1)
            self.set("action_logprobs", logp_pi)

        else:
            assert not t is None
            input = self.get((self.input_name, t))
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


class CriticAgent(Agent):
    def __init__(self, env, n_layers, hidden_size,input_name):
        super().__init__()
        env = make_env(env)
        self.input_name=input_name

        input_size = env.observation_space.shape[0]
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
            nn.Linear(hs, hs),
            nn.ReLU(),
            *hidden_layers,
            nn.Linear(hs, 1),
        )

    def forward(self, **kwargs):
        input = self.get(self.input_name)
        critic = self.model_critic(input).squeeze(-1)
        self.set("critic", critic)

def RL2_ActionAgent(env,n_layers,hidden_size):
    gru_a=GRUAgent(env,hidden_size,"z")
    gru_critic=GRUAgent(env,hidden_size,"z_critic")
    action=ActionAgent(env,n_layers,hidden_size,"z")
    return Agents(gru_a,gru_critic,action)

def RL2_CriticAgent(env,n_layers,hidden_size):
    critic=CriticAgent(env,n_layers,hidden_size,"z_critic")
    return critic
