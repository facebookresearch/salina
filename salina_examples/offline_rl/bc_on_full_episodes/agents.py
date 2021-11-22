#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import d4rl
import d4rl_atari
import gym
import torch
import torch.nn as nn
from gym.wrappers import TimeLimit

from salina import Agent, TAgent, instantiate_class
from salina.agents import Agents
from salina.agents.transformers import *


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

        self.positional_embeddings = nn.Embedding(
            max_episode_steps, embedding_size // 2
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


class ActionMLPAgentFromTransformer(Agent):
    def __init__(
        self, env, n_layers, hidden_size, embedding_size, input_name="attn_out/x"
    ):
        super().__init__()
        env = instantiate_class(env)
        input_size = embedding_size
        output_size = env.action_space.shape[0]
        hidden_sizes = [hidden_size for _ in range(n_layers)]
        self.fc = mlp(
            [input_size] + list(hidden_sizes) + [output_size],
            activation=nn.ReLU,
        )
        self.input_name = input_name

    def forward(self, t=None, **kwargs):
        if not t is None:
            input = self.get((self.input_name, t))
            action = self.fc(input)
            action = torch.tanh(action)
            action = torch.clip(action, min=-1.0, max=1.0)
            self.set(("action", t), action)
        else:
            input = self.get(self.input_name)
            action = self.fc(input)
            action = torch.tanh(action)
            action = torch.clip(action, min=-1.0, max=1.0)
            self.set("action", action)


class ActionMLPAgentFromObservation(Agent):
    def __init__(self, env, n_layers, hidden_size):
        super().__init__()
        env = instantiate_class(env)
        input_size = env.observation_space.shape[0]
        output_size = env.action_space.shape[0]
        hidden_sizes = [hidden_size for _ in range(n_layers)]
        self.fc = mlp(
            [input_size] + list(hidden_sizes) + [output_size],
            activation=nn.ReLU,
        )

    def forward(self, t=None, **kwargs):
        if not t is None:
            input = self.get(("env/env_obs", t))
            action = self.fc(input)
            action = torch.tanh(action)
            action = torch.clip(action, min=-1.0, max=1.0)
            self.set(("action", t), action)
        else:
            input = self.get("env/env_obs")
            action = self.fc(input)
            action = torch.tanh(action)
            action = torch.clip(action, min=-1.0, max=1.0)
            self.set("action", action)


def transition_transformers(encoder, transformer, decoder):
    ns=None
    if "n_steps" in transformer and transformer.n_steps>0:
        ns=transformer.n_steps

    _encoder = TransitionEncoder(**dict(encoder))
    mblock = TransformerMultiBlockAgent(
        transformer.n_layers,
        encoder.embedding_size,
        transformer.n_heads,
        use_layer_norm=transformer.use_layer_norm,
        n_steps=ns
    )
    internal_action_agent = ActionMLPAgentFromTransformer(
        decoder.env, decoder.n_layers, decoder.hidden_size, encoder.embedding_size
    )
    action_agent = Agents(_encoder, mblock, internal_action_agent)
    return action_agent


def observation_mlp(**kwargs):
    return ActionMLPAgentFromObservation(**kwargs)
