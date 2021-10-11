#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import copy
import time

import gym
import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym.wrappers import TimeLimit
from omegaconf import DictConfig, OmegaConf

import salina
import salina.rl.functional as RLF
from salina import Agent, TAgent, Workspace, get_arguments, get_class, instantiate_class
from salina.agents import Agents, NRemoteAgent, TemporalAgent
from salina.agents.gym import AutoResetGymAgent, GymAgent
from salina.logger import TFLogger

# from salina_examples.rl.a2c.coding_session.tools import _index


def _index(tensor_3d, tensor_2d):
    """This function is used to index a 3d tensors using a 2d tensor"""
    x, y, z = tensor_3d.size()
    t = tensor_3d.reshape(x * y, z)
    tt = tensor_2d.reshape(x * y)
    v = t[torch.arange(x * y), tt]
    v = v.reshape(x, y)
    return v


class PAgent(Agent):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(4, 32), nn.Tanh(), nn.Linear(32, 2))

    def forward(self, t, **args):
        observation = self.get(("env/env_obs", t))
        scores = self.model(observation)
        probs = torch.softmax(scores, dim=-1)
        self.set(("probs", t), probs)


class ActionAgent(Agent):
    def __init__(self):
        super().__init__()

    def forward(self, t, stochastic, **args):
        probs = self.get(("probs", t))
        if stochastic:
            action = torch.distributions.Categorical(probs).sample()
            self.set(("action", t), action)
        else:
            action = probs.max(1)[1]
            self.set(("action", t), action)


class CriticAgent(Agent):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(4, 32), nn.Tanh(), nn.Linear(32, 1))

    def forward(self, t, **args):
        observation = self.get(("env/env_obs", t))
        critic = self.model(observation).squeeze(-1)
        self.set(("critic", t), critic)


def make_env(max_episode_steps):
    return TimeLimit(gym.make("CartPole-v0"), max_episode_steps=max_episode_steps)


def run_a2c(cfg):

    # Read: "action" at time t-1
    # Write: "env/env_obs","env/reward","env/done","env/initial_state","env/timestep","env/cumulated_reward"
    env_agent = AutoResetGymAgent(make_env, {"max_episode_steps": 100}, n_envs=4)

    pagent = PAgent()
    pagent.set_name("p_agent")

    acq_pagent = copy.deepcopy(pagent)

    action_agent = ActionAgent()
    critic_agent = CriticAgent()
    tcritic_agent = TemporalAgent(critic_agent)
    acquisition_agent = Agents(env_agent, acq_pagent, action_agent)
    acquisition_agent = TemporalAgent(acquisition_agent)
    acquisition_agent, acquisition_workspace = NRemoteAgent.create(
        acquisition_agent,
        num_processes=4,
        t=0,
        n_steps=cfg.n_timesteps,
        stochastic=True,
    )
    acquisition_agent.seed(0)

    optimizer_prob_agent = torch.optim.Adam(pagent.parameters(), lr=cfg.lr)
    optimizer_critic_agent = torch.optim.Adam(critic_agent.parameters(), lr=cfg.lr)
    workspace = Workspace()

    for epoch in range(cfg.max_epochs):
        # 1: Get a new trajectory
        for a in acquisition_agent.get_by_name("p_agent"):
            a.load_state_dict(pagent.state_dict())

        if epoch == 0:
            acquisition_agent(
                acquisition_workspace, t=0, n_steps=cfg.n_timesteps, stochastic=True
            )
        else:
            acquisition_workspace.copy_n_last_steps(1)
            acquisition_workspace.zero_grad()
            acquisition_agent(
                acquisition_workspace, t=1, n_steps=cfg.n_timesteps - 1, stochastic=True
            )

        # Print the reward
        done = acquisition_workspace["env/done"]
        cumulated_reward = acquisition_workspace["env/cumulated_reward"]
        cr = cumulated_reward[done]
        if cr.size()[0]:
            print("Reward : ", cr.mean().item())

        # 1':
        workspace = Workspace(acquisition_workspace)
        pagent(workspace, t=0, n_steps=cfg.n_timesteps, stochastic=True)

        # 1bis: Computing critic over the trajectories
        tcritic_agent(workspace, t=0, n_steps=cfg.n_timesteps)

        # 2: Compute the loss
        entropy_loss = (
            torch.distributions.Categorical(workspace["probs"]).entropy().mean()
        )

        critic = workspace["critic"]
        done = workspace["env/done"].float()
        reward = workspace["env/reward"]
        td = (
            cfg.discount_factor * critic[1:] * (1.0 - done[1:]) + reward[1:]
        ) - critic[:-1]
        critic_loss = (td ** 2).mean()

        paction = _index(workspace["probs"], workspace["action"])
        lpaction = paction.log()
        a2c_loss = lpaction[:-1] * td.detach()
        a2c_loss = a2c_loss.mean()

        loss = (
            -cfg.a2c_coef * a2c_loss
            - cfg.entropy_coef * entropy_loss
            + cfg.critic_coef * critic_loss
        )
        # print("Losses: ",a2c_loss,entropy_loss,critic_loss)

        optimizer_prob_agent.zero_grad()
        optimizer_critic_agent.zero_grad()
        loss.backward()
        optimizer_prob_agent.step()
        optimizer_critic_agent.step()


@hydra.main(config_path=".", config_name="main.yaml")
def main(cfg):
    import torch.multiprocessing as mp

    mp.set_start_method("spawn")
    run_a2c(cfg)


if __name__ == "__main__":
    main()
