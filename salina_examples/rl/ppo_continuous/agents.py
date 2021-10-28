#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import math

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym.wrappers import TimeLimit
from torch import nn
from torch.distributions.normal import Normal

from salina import TAgent, instantiate_class
from salina.agents import Agents
from salina_examples.rl.atari_wrappers import make_atari, wrap_deepmind, wrap_pytorch


def make_atari_env(**env_args):
    e = make_atari(env_args["env_name"])
    e = wrap_deepmind(e)
    e = wrap_pytorch(e)
    e = TimeLimit(e, max_episode_steps=env_args["max_episode_steps"])
    return e


def make_gym_env(**env_args):
    e = gym.make(env_args["env_name"])
    e = TimeLimit(e, max_episode_steps=env_args["max_episode_steps"])
    return e


class PPOMLPActionVarianceAgent(TAgent):
    """torch.distributions implementation of an diagonal Gaussian policy."""

    def __init__(self, **kwargs):
        super().__init__()
        env = instantiate_class(kwargs["env"])
        input_size = env.observation_space.shape[0]
        self.num_outputs = env.action_space.shape[0]
        hs = kwargs["hidden_size"]
        n_layers = kwargs["n_layers"]
        self.log_std_bounds = kwargs["log_std_bounds"]
        hidden_layers = (
            [nn.Linear(hs, hs), nn.SiLU()] * (n_layers - 1)
            if n_layers > 1
            else nn.Identity()
        )
        self.model = nn.Sequential(
            nn.Linear(input_size, hs),
            nn.SiLU(),
            *hidden_layers,
            nn.Linear(hs, self.num_outputs * 2),
        )

    def forward(self, t, replay, action_variance, **kwargs):
        input = self.get(("env/env_obs", t))
        mu, log_std = self.model(input).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        mu = torch.tanh(mu)
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
        std = log_std.exp()
        dist = Normal(mu, std)

        # if replay:
        #    self.set(("entropy", t), torch.Tensor([0.]).to(mu.device))
        if not replay:
            action = dist.sample()  # if action_variance > 0 else dist.mean
            action = torch.tanh(action)
            logp_pi = dist.log_prob(action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(axis=-1)
            self.set(("action", t), action)
            self.set(("action_logprobs", t), logp_pi)
        else:
            action = self.get(("action", t))
            logp_pi = dist.log_prob(action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(axis=-1)
            self.set(("action_logprobs", t), logp_pi)


class PPOMLPActionAgent(TAgent):
    def __init__(self, **kwargs):
        super().__init__()
        env = instantiate_class(kwargs["env"])
        input_size = env.observation_space.shape[0]
        num_outputs = env.action_space.shape[0]
        hs = kwargs["hidden_size"]
        n_layers = kwargs["n_layers"]
        hidden_layers = (
            [nn.Linear(hs, hs), nn.SiLU()] * (n_layers - 1)
            if n_layers > 1
            else nn.Identity()
        )
        self.model = nn.Sequential(
            nn.Linear(input_size, hs),
            nn.SiLU(),
            *hidden_layers,
            nn.Linear(hs, num_outputs),
        )

    def forward(self, t, replay, action_variance, **kwargs):
        input = self.get(("env/env_obs", t))
        mean = torch.tanh(self.model(input))
        var = torch.ones_like(mean) * action_variance + 0.000001
        dist = Normal(mean, var)
        self.set(("entropy", t), dist.entropy())
        if not replay:
            action = dist.sample()  # if action_variance > 0 else dist.mean
            # action = torch.clip(action, min=-1.0, max=1.0)
            action = torch.tanh(action)
            logp_pi = dist.log_prob(action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(axis=-1)
            self.set(("action", t), action)
            self.set(("action_logprobs", t), logp_pi)
        else:
            action = self.get(("action", t))
            logp_pi = dist.log_prob(action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(axis=-1)
            self.set(("action_logprobs", t), logp_pi)


class PPOMLPCriticAgent(TAgent):
    def __init__(self, **kwargs):
        super().__init__()
        env = instantiate_class(kwargs["env"])
        input_size = env.observation_space.shape[0]
        hs = kwargs["hidden_size"]
        n_layers = kwargs["n_layers"]
        hidden_layers = (
            [nn.Linear(hs, hs), nn.SiLU()] * (n_layers - 1)
            if n_layers > 1
            else nn.Identity()
        )
        self.model_critic = nn.Sequential(
            nn.Linear(input_size, hs),
            nn.SiLU(),
            *hidden_layers,
            nn.Linear(hs, 1),
        )

    def forward(self, t, **kwargs):
        input = self.get(("env/env_obs", t))

        critic = self.model_critic(input).squeeze(-1)
        self.set(("critic", t), critic)
