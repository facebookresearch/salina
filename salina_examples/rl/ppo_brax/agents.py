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
from brax.envs import _envs, wrappers
from gym.wrappers import TimeLimit
from torch import nn
from torch.distributions.normal import Normal

from salina import TAgent, instantiate_class
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


def make_brax_env(
    env_name,
    seed=0,
    batch_size=None,
    episode_length=1000,
    action_repeat=1,
    backend=None,
    auto_reset=True,
    **kwargs
):

    env = _envs[env_name](**kwargs)
    if episode_length is not None:
        env = wrappers.EpisodeWrapper(env, episode_length, action_repeat)
    if batch_size:
        env = wrappers.VectorWrapper(env, batch_size)
    if auto_reset:
        env = wrappers.AutoResetWrapper(env)
    if batch_size is None:
        return wrappers.GymWrapper(env, seed=seed, backend=backend)
    return wrappers.VectorGymWrapper(env, seed=seed, backend=backend)


class ActionAgent(TAgent):
    def __init__(self, **args):
        super().__init__()
        env = instantiate_class(args["env"])
        input_size = env.observation_space.shape[0]
        num_outputs = env.action_space.shape[0]
        hs = args["hidden_size"]
        n_layers = args["n_layers"]
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

    def forward(self, t, replay, action_std, **args):
        if replay:
            input = self.get("env/env_obs")
            mean = self.model(input)
            var = torch.ones_like(mean) * action_std + 0.000001
            dist = Normal(mean, var)
            action = self.get("real_action")
            logp_pi = dist.log_prob(action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(axis=-1)
            self.set("action_logprobs", logp_pi)

        else:
            input = self.get(("env/env_obs", t))
            mean = self.model(input)
            var = torch.ones_like(mean) * action_std + 0.000001
            dist = Normal(mean, var)
            action = dist.sample() if action_std > 0 else dist.mean
            self.set(("real_action", t), action)
            logp_pi = dist.log_prob(action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(axis=-1)
            self.set(("old_action_logprobs", t), logp_pi)
            action = torch.tanh(action)
            self.set(("action", t), action)


class CriticAgent(TAgent):
    def __init__(self, **args):
        super().__init__()
        env = instantiate_class(args["env"])
        input_size = env.observation_space.shape[0]
        hs = args["hidden_size"]
        n_layers = args["n_layers"]
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

    def forward(self, t, **args):
        input = self.get("env/env_obs")
        critic = self.model_critic(input).squeeze(-1)
        self.set("critic", critic)


class Normalizer(TAgent):
    def __init__(self, **args):
        super().__init__()
        env = instantiate_class(args["env"])
        n_features = env.observation_space.shape[0]
        self.n = torch.zeros(n_features).to(args["device"])
        self.mean = nn.Parameter(torch.zeros(n_features), requires_grad=False)
        self.mean_diff = torch.zeros(n_features).to(args["device"])
        self.var = nn.Parameter(torch.ones(n_features), requires_grad=False).to(
            args["device"]
        )

    def forward(self, t, update_normalizer=False, **args):
        input = self.get(("env/env_obs", t))
        assert torch.isnan(input).sum() == 0.0, "problem"
        if update_normalizer:
            self.update(input)
        input = self.normalize(input)
        assert torch.isnan(input).sum() == 0.0, "problem"
        self.set(("env/env_obs", t), input)

    def update(self, x):
        self.n += 1.0
        last_mean = self.mean.clone()
        self.mean += (x - self.mean).mean(dim=0) / self.n
        self.mean_diff += (x - last_mean).mean(dim=0) * (x - self.mean).mean(dim=0)
        self.var = nn.Parameter(
            torch.clamp(self.mean_diff / self.n, min=1e-2), requires_grad=False
        ).to(x.device)

    def normalize(self, inputs):
        obs_std = torch.sqrt(self.var)
        return (inputs - self.mean) / obs_std

    def seed(self, seed):
        torch.manual_seed(seed)
