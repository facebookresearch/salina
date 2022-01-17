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
from salina.agents.transformers import *
from salina_examples.rl.atari_wrappers import make_atari, wrap_deepmind, wrap_pytorch

def make_brax_env(env_name):
    e = create_gym_env(env_name)
    return JaxToTorchWrapper(e)

def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def _timestep(timestep):
    mask = timestep.lt(0).float()
    zeros = torch.zeros_like(timestep)
    if mask.any():
        return ((1.0 - mask) * timestep + mask * zeros).long()
    else:
        return timestep


class TransitionEncoder(Agent):
    # Transform a tuple (s,a,s') to an embedding vector by concatenating s,a annd s' representation + positionnal encoding
    def __init__(
        self,
        env,
        n_layers,
        hidden_size,
        embedding_size,
        max_episode_steps,
        output_name="attn_in/x",
    ):
        super().__init__()
        assert embedding_size % 2 == 0
        env = instantiate_class(env)
        input_size = env.observation_space.shape[0]
        output_size = env.action_space.shape[0]
        sizes = [hidden_size for k in range(n_layers)]
        self.model_obs = mlp([input_size] + sizes + [embedding_size])
        self.model_act = mlp([output_size] + sizes + [embedding_size])
        self.mix = mlp([embedding_size * 3] + [embedding_size] + [embedding_size // 2])
        self.max_episode_steps = max_episode_steps
        self.n_t_embeddings = max_episode_steps + 2
        self.positional_embeddings = nn.Embedding(
            self.n_t_embeddings, embedding_size // 2
        )
        self.output_name = output_name

    def forward(self, t=None, **kwargs):
        if not t is None:
            if t == 0:
                e_s = self.model_obs(self.get(("env/env_obs", t)))
                t_s = _timestep(self.get(("env/timestep", t)))
                B = e_s.size()[0]
                empty = torch.zeros_like(e_s)
                embedding = self.mix(torch.cat([empty, empty, e_s], dim=1))
                pe = self.positional_embeddings(t_s)
                embedding = torch.cat([embedding, pe], dim=1)
                self.set((self.output_name, t), embedding)
            else:
                e_s = self.model_obs(self.get(("env/env_obs", t - 1)))
                B = e_s.size()[0]
                e_ss = self.model_obs(self.get(("env/env_obs", t)))
                e_a = self.model_act(self.get(("action", t - 1)))
                t_s = _timestep(self.get(("env/timestep", t)))
                assert (
                    t_s.max().item() < self.n_t_embeddings
                ), "Episode too long coparing to time embeddings: " + str(
                    t_s.max().item()
                )
                v = torch.cat([e_s, e_a, e_ss], dim=1)
                embedding = self.mix(v)
                pe = self.positional_embeddings(t_s)
                embedding = torch.cat([embedding, pe], dim=1)
                self.set((self.output_name, t), embedding)
        else:
            e_s = self.model_obs(self.get("env/env_obs"))
            t_s = _timestep(self.get("env/timestep"))

            T = e_s.size()[0]
            B = e_s.size()[1]
            empty = torch.zeros_like(e_s[0].unsqueeze(0))
            e_ss = e_s
            e_s = torch.cat([empty, e_s[:-1]], dim=0)
            e_a = self.model_act(self.get("action"))
            e_a = torch.cat([empty, e_a[:-1]], dim=0)
            v = torch.cat([e_s, e_a, e_ss], dim=2)
            complete = self.mix(v)
            pe = self.positional_embeddings(t_s)
            complete = torch.cat([complete, pe], dim=2)
            self.set(self.output_name, complete)


class ActionAgent(Agent):
    def __init__(self, env, n_layers, hidden_size, embedding_size):
        super().__init__()
        env = make_brax_env(env.env_name)
        input_size = embedding_size
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
            input = self.get("action_attn_out/x")
            mean = self.model(input)
            var = torch.ones_like(mean) * action_std + 0.000001
            dist = Normal(mean, var)
            action = self.get("action_before_tanh")
            logp_pi = dist.log_prob(action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(axis=-1)
            self.set("action_logprobs", logp_pi)

        else:
            assert not t is None
            input = self.get(("action_attn_out/x", t))
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
    def __init__(self, env, n_layers, hidden_size, embedding_size):
        super().__init__()
        env = make_brax_env(env.env_name)
        input_size = embedding_size
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
        input = self.get("critic_attn_out/x")
        critic = self.model_critic(input).squeeze(-1)
        self.set("critic", critic)


def action_transformer(encoder, transformer, decoder):
    _encoder = TransitionEncoder(output_name="action_attn_in/x", **dict(encoder))
    mblock = TransformerMultiBlockAgent(
        transformer.n_layers,
        encoder.embedding_size,
        transformer.n_heads,
        n_steps=transformer.n_steps,
        prefix="action_attn_",
        use_layer_norm=transformer.use_layer_norm,
    )
    internal_action_agent = ActionAgent(
        decoder.env, decoder.n_layers, decoder.hidden_size, encoder.embedding_size
    )
    action_agent = Agents(_encoder, mblock, internal_action_agent)
    return action_agent


def critic_transformer(encoder, transformer, decoder):
    _encoder = TransitionEncoder(output_name="critic_attn_in/x", **dict(encoder))
    mblock = TransformerMultiBlockAgent(
        transformer.n_layers,
        encoder.embedding_size,
        transformer.n_heads,
        n_steps=transformer.n_steps,
        prefix="critic_attn_",
        use_layer_norm=transformer.use_layer_norm,
    )
    internal_critic_agent = CriticAgent(
        decoder.env, decoder.n_layers, decoder.hidden_size, encoder.embedding_size
    )
    critic_agent = Agents(_encoder, mblock, internal_critic_agent)
    return critic_agent
