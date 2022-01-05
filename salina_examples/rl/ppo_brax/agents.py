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
from brax.envs.to_torch import JaxToTorchWrapper
from gym.wrappers import TimeLimit
from torch import nn
from torch.distributions.normal import Normal

from salina import Agent, instantiate_class
from salina_examples.rl.atari_wrappers import make_atari, wrap_deepmind, wrap_pytorch

def make_env(args):
    if args["env_name"].startswith("brax/"):
        env_name=args["env_name"][5:]
        return make_brax_env(env_name)
    else:
        assert args["env_name"].startswith("gym/")
        env_name=args["env_name"][4:]
        return make_gym_env(env_name,args["max_episode_steps"])

def make_brax_env(env_name):
    e = create_gym_env(env_name)
    return JaxToTorchWrapper(e)

def make_gym_env(env_name,max_episode_steps):
    e = gym.make(env_name)
    e = TimeLimit(e, max_episode_steps=max_episode_steps)
    return e

class ActionAgent(Agent):
    def __init__(self, env, n_layers, hidden_size):
        super().__init__()
        env = make_env(env)
        input_size = env.observation_space.shape[0]
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
            nn.Linear(input_size, hs),
            nn.ReLU(),
            *hidden_layers,
            nn.Linear(hs, num_outputs),
        )

    def forward(self, t=None, replay=False, action_std=0.1, **kwargs):
        if replay:
            assert t == None
            input = self.get("env/env_obs")
            mean = self.model(input)
            var = torch.ones_like(mean) * action_std + 0.000001
            dist = Normal(mean, var)
            action = self.get("action_before_tanh")
            logp_pi = dist.log_prob(action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(axis=-1)
            self.set("action_logprobs", logp_pi)

        else:
            assert not t is None
            input = self.get(("env/env_obs", t))
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
    def __init__(self, env, n_layers, hidden_size):
        super().__init__()
        env = make_env(env)
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
            nn.Linear(input_size, hs),
            nn.ReLU(),
            *hidden_layers,
            nn.Linear(hs, 1),
        )

    def forward(self, **kwargs):
        input = self.get("env/env_obs")
        critic = self.model_critic(input).squeeze(-1)
        self.set("critic", critic)
