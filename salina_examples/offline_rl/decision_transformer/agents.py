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
        use_timestep,
        use_reward_to_go,
        output_name="attn_in/x",
    ):
        super().__init__()
        assert embedding_size % 2 == 0
        env = instantiate_class(env)
        input_size = env.observation_space.shape[0]
        output_size = env.action_space.shape[0]
        sizes = [hidden_size for k in range(n_layers)]
        self.model_obs = nn.Linear(input_size, embedding_size)
        self.model_act = nn.Linear(output_size, embedding_size)
        self.model_rtg = nn.Linear(1, embedding_size)
        self.mix = mlp(
            [embedding_size * 3]
            + [hidden_size for _ in range(n_layers)]
            + [hidden_size]
        )
        self.use_timestep = use_timestep
        self.use_reward_to_go = use_reward_to_go
        self.positional_embeddings = nn.Embedding(max_episode_steps + 1, embedding_size)
        self.output_name = output_name

    def forward(self, t=None, control_variable="reward_to_go", **kwargs):
        if not t is None:
            if t == 0:
                e_s = self.model_obs(self.get(("env/env_obs", t)))
                e_rtg = self.model_rtg(self.get((control_variable, t)).unsqueeze(-1))
                t_s = _timestep(self.get(("env/timestep", t)))
                pe = self.positional_embeddings(t_s)
                if not self.use_timestep:
                    pe.fill_(0.0)

                B = e_s.size()[0]
                empty = torch.zeros_like(e_s)
                if not self.use_reward_to_go:
                    e_rtg.fill_(0.0)
                embedding = self.mix(
                    torch.cat([empty + pe, e_s + pe, e_rtg + pe], dim=1)
                )
                self.set((self.output_name, t), embedding)
            else:
                e_rtg = self.model_rtg(self.get((control_variable, t)).unsqueeze(-1))
                B = e_rtg.size()[0]
                e_ss = self.model_obs(self.get(("env/env_obs", t)))
                e_a = self.model_act(self.get(("action", t - 1)))
                t_s = _timestep(self.get(("env/timestep", t)))
                pe = self.positional_embeddings(t_s)
                if not self.use_timestep:
                    pe.fill_(0.0)

                if not self.use_reward_to_go:
                    e_rtg.fill_(0.0)
                v = torch.cat([e_a + pe, e_ss + pe, e_rtg + pe], dim=1)
                embedding = self.mix(v)
                self.set((self.output_name, t), embedding)
        else:
            e_s = self.model_obs(self.get("env/env_obs"))
            e_rtg = self.model_rtg(self.get(control_variable).unsqueeze(-1))
            if not self.use_reward_to_go:
                e_rtg.fill_(0.0)
            t_s = _timestep(self.get("env/timestep"))
            pe = self.positional_embeddings(t_s)
            if not self.use_timestep:
                pe.fill_(0.0)
            T = e_s.size()[0]
            B = e_s.size()[1]
            empty = torch.zeros_like(e_s[0].unsqueeze(0))
            e_ss = e_s
            e_a = self.model_act(self.get("action"))
            e_a = torch.cat([empty, e_a[:-1]], dim=0)
            v = torch.cat([e_a + pe, e_ss + pe, e_rtg + pe], dim=2)
            complete = self.mix(v)
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


def transition_transformers(encoder, transformer, decoder):
    _encoder = TransitionEncoder(**dict(encoder))
    ns=None
    if "n_steps" in transformer:
        ns=transformer.n_steps

    mblock = TransformerMultiBlockAgent(
        n_layers=transformer.n_layers,
        n_steps=ns,
        embedding_size=encoder.hidden_size,
        n_heads=transformer.n_heads,
        use_layer_norm=transformer.use_layer_norm,
    )
    internal_action_agent = ActionMLPAgentFromTransformer(
        decoder.env, decoder.n_layers, decoder.hidden_size, encoder.hidden_size
    )
    action_agent = Agents(_encoder, mblock, internal_action_agent)
    return action_agent


class ActionMLPAgentFromObservation(Agent):
    def __init__(
        self,
        env,
        n_layers,
        hidden_size,
        embedding_size,
        maximum_episode_steps,
        use_timestep,
        use_reward_to_go,
    ):
        super().__init__()
        env = instantiate_class(env)
        self.l_obs = nn.Linear(env.observation_space.shape[0], embedding_size)
        self.l_timestep = nn.Embedding(maximum_episode_steps + 10, embedding_size)
        self.l_rtg = nn.Linear(1, embedding_size)
        self.use_timestep = use_timestep
        self.use_reward_to_go = use_reward_to_go
        output_size = env.action_space.shape[0]
        hidden_sizes = [hidden_size for _ in range(n_layers)]
        self.fc = mlp(
            [embedding_size * 2] + list(hidden_sizes) + [output_size],
            activation=nn.ReLU,
        )

    def forward(self, t=None, control_variable="reward_to_go", **args):
        if not t is None:
            input = self.get(("env/env_obs", t))
            input = self.l_obs(input)
            ts = self.get(("env/timestep", t))
            ts = self.l_timestep(ts)
            if not self.use_timestep:
                ts = torch.zeros_like(ts)
            rtg = self.get((control_variable, t)).unsqueeze(-1)
            rtg = self.l_rtg(rtg)
            if not self.use_reward_to_go:
                rtg = torch.zeros_like(rtg)
            input = torch.cat([input + ts, rtg + ts], dim=-1)

            action = self.fc(input)
            action = torch.tanh(action)
            action = torch.clip(action, min=-1.0, max=1.0)
            self.set(("action", t), action)
        else:
            input = self.get("env/env_obs")
            input = self.l_obs(input)
            ts = self.get("env/timestep")
            ts = self.l_timestep(ts)
            if not self.use_timestep:
                ts = torch.zeros_like(ts)

            rtg = self.get(control_variable).unsqueeze(-1)
            rtg = self.l_rtg(rtg)
            if not self.use_reward_to_go:
                rtg = torch.zeros_like(rtg)

            input = torch.cat([input + ts, rtg + ts], dim=-1)
            action = self.fc(input)
            action = torch.tanh(action)
            action = torch.clip(action, min=-1.0, max=1.0)
            self.set("action", action)
