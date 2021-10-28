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


class PPOMLPActionAgent(TAgent):
    def __init__(self, **kwargs):
        super().__init__()
        env = instantiate_class(kwargs["env"])
        input_size = env.observation_space.shape[0]
        num_outputs = env.action_space.n
        hs = kwargs["hidden_size"]
        self.model = nn.Sequential(
            nn.Linear(input_size, hs), nn.ReLU(), nn.Linear(hs, num_outputs)
        )

    def forward(self, t, replay, stochastic, **kwargs):
        input = self.get(("env/env_obs", t))
        scores = self.model(input)
        probs = torch.softmax(scores, dim=-1)
        self.set(("action_probs", t), probs)

        if not replay:
            if stochastic:
                action = torch.distributions.Categorical(probs).sample()
            else:
                action = probs.argmax(1)
            self.set(("action", t), action)


class PPOMLPCriticAgent(TAgent):
    def __init__(self, **kwargs):
        super().__init__()
        env = instantiate_class(kwargs["env"])
        input_size = env.observation_space.shape[0]
        num_outputs = env.action_space.n
        hs = kwargs["hidden_size"]
        self.model_critic = nn.Sequential(
            nn.Linear(input_size, hs), nn.ReLU(), nn.Linear(hs, 1)
        )

    def forward(self, t, **kwargs):
        input = self.get(("env/env_obs", t))

        critic = self.model_critic(input).squeeze(-1)
        self.set(("critic", t), critic)
