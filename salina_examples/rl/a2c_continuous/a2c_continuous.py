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
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym.wrappers import TimeLimit
from omegaconf import DictConfig, OmegaConf

import salina
import salina.rl.functional as RLF
from salina import TAgent, get_arguments, get_class, instantiate_class
from salina.agents import Agents, RemoteAgent, TemporalAgent
from salina.agents.gyma import AutoResetGymAgent, GymAgent
from salina.logger import TFLogger
from torch.distributions.normal import Normal
import continuous_cartpole

def _index(tensor_3d, tensor_2d):
    """This function is used to index a 3d tensors using a 2d tensor"""
    x, y, z = tensor_3d.size()
    t = tensor_3d.reshape(x * y, z)
    tt = tensor_2d.reshape(x * y)
    v = t[torch.arange(x * y), tt]
    v = v.reshape(x, y)
    return v


class ContinuousActionAgent(TAgent):
    def __init__(self, **kwargs):
        super().__init__()
        env = instantiate_class(kwargs["env"])
        input_size = env.observation_space.shape[0]
        num_outputs = env.action_space.shape[0]
        hs = kwargs["hidden_size"]
        n_layers = kwargs["n_layers"]
        hidden_layers = (
            [nn.Linear(hs, hs), nn.ReLU()] * (n_layers - 1)
            if n_layers > 1
            else nn.Identity()
        )
        self.model = nn.Sequential(
            nn.Linear(input_size, hs),
            nn.ReLU(),
            *hidden_layers,
            nn.Linear(hs, num_outputs),
        )
        self.std_param = nn.parameter.Parameter(torch.randn(num_outputs,1))
        self.soft_plus = torch.nn.Softplus()

    def forward(self, t,stochastic, **kwargs):
        input = self.get(("env/env_obs", t))
        mean = self.model(input)
        dist = Normal(mean, self.soft_plus(self.std_param)) # std must be positive
        self.set(("entropy", t), dist.entropy())
        if stochastic:
            action = torch.tanh(dist.sample()) # valid actions are supposed to be in [-1,1] range
        else : 
            action = torch.tanh(mean) # valid actions are supposed to be in [-1,1] range
        logp_pi = dist.log_prob(action).sum(axis=-1)
        self.set(("action", t), action)
        self.set(("action_logprobs", t), logp_pi)


class ContinuousCriticAgent(TAgent):
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




def make_cartpole(max_episode_steps):
    return TimeLimit(gym.make("CartPole-v0"), max_episode_steps=max_episode_steps)


def run_a2c(cfg):
    # 1)  Build the  logger
    logger = instantiate_class(cfg.logger)

    # 2) Create the environment agent
    # This agent implements N gym environments with auto-reset
    env_agent = AutoResetGymAgent(
        get_class(cfg.algorithm.env),
        get_arguments(cfg.algorithm.env),
        n_envs=cfg.algorithm.n_envs,
    )

    # 3) Create the Agents

    action_agent = instantiate_class(cfg.action_agent)
    critic_agent = instantiate_class(cfg.critic_agent)

    # 4) Combine env and policy agents
    agent = Agents(env_agent, action_agent)

    # 5) Get an agent that is executed on a complete workspace
    agent = TemporalAgent(agent)
    agent.seed(cfg.algorithm.env_seed)

    # 5 bis) Create the temporal critic agent to compute critic values over the workspace
    tcritic_agent = TemporalAgent(critic_agent)

    # 6) Configure the workspace to the right dimension
    workspace = salina.Workspace()

    # 7) Confgure the optimizer over the a2c agent
    optimizer_args = get_arguments(cfg.algorithm.optimizer)
    parameters = nn.Sequential(action_agent, critic_agent).parameters()
    optimizer = get_class(cfg.algorithm.optimizer)(parameters, **optimizer_args)

    # 8) Training loop
    epoch = 0
    for epoch in range(cfg.algorithm.max_epochs):
        workspace.zero_grad()
        # Execute the agent on the workspace
        if epoch > 0:
            workspace.copy_n_last_steps(1)
            agent(
                workspace, t=1, n_steps=cfg.algorithm.n_timesteps - 1,
                stochastic = True)
        else:
            agent(workspace, t=0, n_steps=cfg.algorithm.n_timesteps,
                  stochastic = True)
        # Compute the critic value over the whole workspace
        tcritic_agent(workspace, n_steps=cfg.algorithm.n_timesteps)

        # Get relevant tensors (size are timestep x n_envs x ....)
        critic, done, action_logp, reward = workspace[
            "critic", "env/done", "action_logprobs", "env/reward"
        ]

        # Compute temporal difference
        target = reward[1:] + cfg.algorithm.discount_factor * critic[1:].detach() * (
            1 - done[1:].float()
        )
        td = target - critic[:-1]

        # Compute critic loss
        td_error = td ** 2
        critic_loss = td_error.mean()

        # Compute entropy loss
        entropy_loss = torch.mean(workspace['entropy'])

        # Compute A2C loss
        a2c_loss = action_logp[:-1] * td.detach()
        a2c_loss = a2c_loss.mean()

        # Log losses
        logger.add_scalar("learner/critic_loss", critic_loss.item(), epoch)
        logger.add_scalar("learner/entropy_loss", entropy_loss.item(), epoch)
        logger.add_scalar("learner/std", action_agent.std_param.item(), epoch)
        logger.add_scalar("learner/a2c_loss", a2c_loss.item(), epoch)

        loss = (
            -cfg.algorithm.entropy_coef * entropy_loss
            + cfg.algorithm.critic_coef * critic_loss
            - cfg.algorithm.a2c_coef * a2c_loss
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute the cumulated reward on final_state
        creward = workspace["env/cumulated_reward"]
        creward = creward[done]
        if creward.size()[0] > 0:
            logger.add_scalar("monitor/reward", creward.mean().item(), epoch)


@hydra.main(config_path=".", config_name="a2c_continuous.yaml")
def main(cfg):
    import torch.multiprocessing as mp

    mp.set_start_method("spawn")
    run_a2c(cfg)


if __name__ == "__main__":
    # import sys, os
    # sys.path.append(os.getcwd())
    main()
