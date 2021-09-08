#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import gym
import torch
import torch.nn as nn
from gym.wrappers import TimeLimit

from salina import TAgent, instantiate_class
from salina.agents import Agents
from salina_examples.rl.atari_wrappers import (
    make_atari,
    wrap_deepmind,
    wrap_pytorch,
)


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
    def __init__(self, **args):
        super().__init__()
        env = instantiate_class(args["env"])
        input_size = env.observation_space.shape[0]
        self.num_outputs = env.action_space.shape[0]
        hs = args["hidden_size"]
        n_layers = args["n_layers"]
        hidden_layers = [nn.Linear(hs,hs),nn.SiLU()] * (n_layers - 1) if n_layers >1 else nn.Identity()
        self.model = nn.Sequential(
            nn.Linear(input_size, hs), nn.SiLU(), *hidden_layers,nn.Linear(hs, self.num_outputs * 2)
        )

    def forward(self, t, replay, action_variance, **args):
        input = self.get(("env/env_obs", t))
        mean, var = torch.split( self.model(input), self.num_outputs, dim = -1)
        mean = torch.tanh(mean)
        var = nn.Softplus()(var) + 0.001
        print("\n\n----mean.abs().max():",mean.abs().max().item())
        print("\n\n----var.min():",var.min().item())
        dist = torch.distributions.Normal(mean, var)
        self.set(("entropy", t), dist.entropy().sum(-1))
        if not replay:
            action = dist.sample()
            action = torch.tanh(action)
            action = torch.clip(action, min=-1.0, max=1.0)
            self.set(("action", t), action)
        _action = self.get(("action", t))
        lp = dist.log_prob(_action)
        tlp = lp.sum(-1)
        self.set(("action_logprobs", t), tlp)


class PPOMLPActionAgent(TAgent):
    def __init__(self, **args):
        super().__init__()
        env = instantiate_class(args["env"])
        input_size = env.observation_space.shape[0]
        num_outputs = env.action_space.shape[0]
        hs = args["hidden_size"]
        n_layers = args["n_layers"]
        hidden_layers = [nn.Linear(hs,hs),nn.SiLU()] * (n_layers - 1) if n_layers >1 else nn.Identity()
        self.model = nn.Sequential(
            nn.Linear(input_size, hs),
            nn.SiLU(),
            *hidden_layers,
            nn.Linear(hs, num_outputs),
        )

    def forward(self, t, replay, action_variance, **args):
        input = self.get(("env/env_obs", t))
        mean = torch.tanh(self.model(input))
        var = torch.ones_like(mean) * action_variance + 0.000001
        dist = torch.distributions.Normal(mean, var)
        self.set(("entropy", t), dist.entropy().sum(-1))

        if not replay:
            action = dist.sample()
            action = torch.clip(action, min=-1.0, max=1.0)
            self.set(("action", t), action)
        _action = self.get(("action", t))
        lp = dist.log_prob(_action)
        tlp = lp.sum(-1)
        self.set(("action_logprobs", t), tlp)


class PPOMLPCriticAgent(TAgent):
    def __init__(self, **args):
        super().__init__()
        env = instantiate_class(args["env"])
        input_size = env.observation_space.shape[0]
        hs = args["hidden_size"]
        n_layers = args["n_layers"]
        hidden_layers = [nn.Linear(hs,hs),nn.SiLU()] * (n_layers - 1) if n_layers >1 else nn.Identity()
        self.model_critic = nn.Sequential(
            nn.Linear(input_size, hs),
            nn.SiLU(),
            *hidden_layers,
            nn.Linear(hs, 1),
        )

    def forward(self, t, **args):
        input = self.get(("env/env_obs", t))

        critic = self.model_critic(input).squeeze(-1)
        self.set(("critic", t), critic)
