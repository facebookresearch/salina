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


class MLP(nn.Module):
    def __init__(self, sizes, activation, output_activation=nn.Identity):
        super().__init__()
        layers = []
        for j in range(len(sizes) - 1):
            act = activation if j < len(sizes) - 2 else output_activation
            layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
        self.layers = layers
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)


class A2CMLPAgent(TAgent):
    def __init__(self, env, hidden_size, n_layers):
        super().__init__()
        env = instantiate_class(env)
        input_size = env.observation_space.shape[0]
        num_outputs = env.action_space.n
        hidden_sizes = [hidden_size for _ in range(n_layers)]
        self.model = MLP(
            [input_size] + list(hidden_sizes) + [num_outputs],
            activation=nn.ReLU,
        )
        self.model_critic = MLP(
            [input_size] + list(hidden_sizes) + [1],
            activation=nn.ReLU,
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
        else:
            critic = self.model_critic(input).squeeze(-1)
            self.set(("critic", t), critic)


class A2C_ObservationAgent(TAgent):
    """Compute a representation of the observation+action+reward"""

    def __init__(self, n_input, n_actions, n_output, output_name):
        super().__init__()
        self.linear = nn.Linear(n_input, n_output)
        self.emb = nn.Embedding(n_actions, n_output)
        self.linear_r = nn.Linear(1, n_output)
        self.linear_out = nn.Linear(n_output * 3, n_output)
        self.output_name = output_name

    def forward(self, t, **kwargs):
        obs = self.get(("env/env_obs", t))
        obs = self.linear(obs)
        obs = torch.relu(obs)
        B = obs.size()[0]
        if t == 0:
            action = torch.tensor([0]).repeat(B).to(obs.device)
        else:
            action = self.get(("action", t - 1))
        action = self.emb(action)
        reward = self.get(("env/reward", t))  #
        reward = reward.unsqueeze(-1)
        reward = self.linear_r(reward)
        reward = torch.relu(reward)
        z = torch.cat([obs, action, reward], dim=-1)
        out = torch.relu(self.linear_out(z))
        self.set((self.output_name, t), out)


class A2C_GRUAgent(TAgent):
    def __init__(self, n_input, n_output, input_name, output_name):
        super().__init__()
        self.gru = nn.GRUCell(n_input, n_output)
        self.n_output = n_output
        self.input_name = input_name
        self.output_name = output_name

    def forward(self, t, **kwargs):
        _input = self.get((self.input_name, t))
        B = _input.size()[0]
        if t == 0:
            _input_z = torch.zeros(B, self.n_output, device=_input.device)
        else:
            _input_z = self.get((self.output_name, t - 1))
        initial_state = self.get(("env/initial_state", t))

        # Re-initialize state if we are in an env initial_state
        _input_z = masked_tensor(
            _input_z, torch.zeros(B, self.n_output, device=_input.device), initial_state
        )

        out = self.gru(_input, _input_z)
        self.set((self.output_name, t), out)


class A2C_PolicyAgent(TAgent):
    def __init__(self, n_input, n_actions, input_name):
        super().__init__()
        self.linear = nn.Linear(n_input, n_actions)
        self.input_name = input_name

    def forward(self, t, replay, stochastic, **kwargs):
        _input = self.get((self.input_name, t))
        scores = self.linear(_input)
        probs = torch.softmax(scores, dim=-1)
        self.set(("action_probs", t), probs)

        if not replay:
            if stochastic:
                action = torch.distributions.Categorical(probs).sample()
            else:
                action = probs.argmax(1)
            self.set(("action", t), action)


class A2C_CriticAgent(TAgent):
    def __init__(self, n_input, input_name):
        super().__init__()
        self.linear = nn.Linear(n_input, 1)
        self.input_name = input_name

    def forward(self, t, replay, **kwargs):
        if replay:
            _input = self.get((self.input_name, t))
            c = self.linear(_input).squeeze(-1)
            self.set(("critic", t), c)


def a2c_recurrent(env, hidden_size):
    """One common recurrent NN for policy and critic"""
    e = instantiate_class(env)
    n_input = e.observation_space.shape[0]
    n_actions = e.action_space.n
    agent_observation = A2C_ObservationAgent(n_input, n_actions, hidden_size, "z_obs")
    recurrent_agent = A2C_GRUAgent(hidden_size, hidden_size, "z_obs", "z")
    policy = A2C_PolicyAgent(hidden_size, n_actions, "z")
    critic = A2C_CriticAgent(hidden_size, "z")
    return Agents(agent_observation, recurrent_agent, policy, critic)


def a2c_recurrent_sep(env, hidden_size):
    """Two seperate recurrent NN for policy and critic"""
    e = instantiate_class(env)
    n_input = e.observation_space.shape[0]
    n_actions = e.action_space.n
    agent_observation = A2C_ObservationAgent(n_input, n_actions, hidden_size, "z_obs")
    recurrent_agent = A2C_GRUAgent(hidden_size, hidden_size, "z_obs", "z")
    policy = A2C_PolicyAgent(hidden_size, n_actions, "z")

    agent_observation_c = A2C_ObservationAgent(
        n_input, n_actions, hidden_size, "z_obs_critic"
    )
    recurrent_agent_c = A2C_GRUAgent(
        hidden_size, hidden_size, "z_obs_critic", "z_critic"
    )
    critic = A2C_CriticAgent(hidden_size, "z_critic")
    return Agents(
        agent_observation,
        recurrent_agent,
        policy,
        agent_observation_c,
        recurrent_agent_c,
        critic,
    )


class DuelingCnnDQN(nn.Module):
    def __init__(self, input_shape, num_outputs):
        super(DuelingCnnDQN, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_outputs

        self.features = nn.Sequential(
            nn.Conv2d(input_shape[1], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.advantage = nn.Sequential(
            nn.Linear(self.feature_size(), 512), nn.ReLU(), nn.Linear(512, num_outputs)
        )

        self.value = nn.Sequential(
            nn.Linear(self.feature_size(), 512), nn.ReLU(), nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean()

    def feature_size(self):
        return self.features(torch.zeros(1, *self.input_shape[1:])).view(1, -1).size(1)


class A2CAtariAgent(TAgent):
    def __init__(self, env, hidden_size):
        super().__init__()
        e = instantiate_class(env)
        input_shape = (1,) + e.observation_space.shape
        num_outputs = e.action_space.n
        self.cnn = DuelingCnnDQN(input_shape, hidden_size)
        self.linear = nn.Linear(hidden_size, num_outputs)
        self.linear_critic = nn.Linear(hidden_size, 1)

    def _forward_nn(self, state):
        qvals = self.cnn(state)
        return qvals

    def forward(self, t, stochastic, replay, **kwargs):
        input = self.get(("env/env_obs", t))
        z = self._forward_nn(input)
        scores = self.linear(z)
        probs = torch.softmax(scores, dim=-1)
        self.set(("action_probs", t), probs)

        if not replay:
            if stochastic:
                action = torch.distributions.Categorical(probs).sample()
            else:
                action = probs.argmax(1)
            self.set(("action", t), action)
        else:
            critic = self.linear_critic(z).squeeze(-1)
            self.set(("critic", t), critic)
