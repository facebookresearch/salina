

#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

###### IMPORT
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
from salina import TAgent, get_arguments, get_class, instantiate_class
from salina.agents import Agents, RemoteAgent, TemporalAgent
from salina.agents.gym import AutoResetGymAgent, GymAgent
from salina.logger import TFLogger


def _index(tensor_3d, tensor_2d):
    """This function is used to index a 3d tensors using a 2d tensor"""
    x, y, z = tensor_3d.size()
    t = tensor_3d.reshape(x * y, z)
    tt = tensor_2d.reshape(x * y)
    v = t[torch.arange(x * y), tt]
    v = v.reshape(x, y)
    return v


class MyAgent(TAgent):
    def __init__(self):
        super().__init__()
        self.model=nn.Sequential(nn.Linear(4,16),nn.Tanh(),nn.Linear(16,2))
        self.critic=nn.Sequential(nn.Linear(4,16),nn.Tanh(),nn.Linear(16,1))

    def forward(self, t, **args):
        observation = self.get(("env/env_obs", t))
        scores = self.model(observation)
        probs = torch.softmax(scores, dim=-1)
        critic = self.critic(observation).squeeze(-1)
        self.set(("action_probs", t), probs)
        self.set(("critic", t), critic)

class ActionAgent(TAgent):
    def __init__(self):
        super().__init__()

    def forward(self,t,stochastic,**args):
        probs=self.get(("action_probs",t))
        if stochastic:
            action=torch.distributions.Categorical(probs).sample()
        else:
            action=probs.max(1)[1]
        self.set(("action",t),action)

def make_cartpole(max_episode_steps):
    return TimeLimit(gym.make("CartPole-v0"), max_episode_steps=max_episode_steps)


def run_a2c(cfg):
    ##### BUILD THE LOGGER (Salina provides a CSV+Tensroboard Logger that one is free to use or not)
    logger = instantiate_class(cfg.logger)

    #### CREATE THE ENVIRONMENT Agent (Salina provides Gym Agents and Brax Agents)
    env_agent = AutoResetGymAgent(make_cartpole,{"max_episode_steps":100})

    ### CREATE THE POLICY
    a2c_agent=MyAgent()
    ta2c_agent=TemporalAgent(a2c_agent)
    acquisition_a2c_agent=copy.deepcopy(a2c_agent) # Copy the model for other processes

    action_agent=ActionAgent()

    #### CREATE THE ACQUSITION AGENT
    acquisition_agent=Agents(env_agent,acquisition_a2c_agent,action_agent)
    acquisition_agent=TemporalAgent(acquisition_agent)
    acquisition_agent=RemoteAgent(acquisition_agent,num_processes=2)
    acquisition_agent.seed(123)

    ### DEFINE A WORKSPACE
    workspace = salina.Workspace(
        batch_size=cfg.algorithm.n_envs,
        time_size=cfg.algorithm.n_timesteps,
    )

    # CONFIGURE THE OPTIMIZER
    optimizer = torch.optim.Adam(a2c_agent.parameters(),lr=cfg.algorithm.lr)

    # TEST FORWARD
    workspace=acquisition_agent(workspace,stochastic=True)
    # 8) Training loop
    epoch = 0
    for epoch in range(cfg.algorithm.max_epochs):

        acquisition_a2c_agent.load_state_dict(a2c_agent.state_dict())
        workspace.copy_n_last_steps(1)
        acquisition_agent(workspace, t=1, stochastic=True)

        replay_workspace=workspace.convert_to_workspace()
        ta2c_agent(replay_workspace,stochastic=True)

        #### COMPUTE LOSS

        ###### GET RELEVANT VARIABLES
        critic, done, action_probs, reward, action = replay_workspace[
            "critic", "env/done", "action_probs", "env/reward", "action"
        ]

        ### COMPUTE TEMPORAL DIFFERENCE
        target = reward[1:] + cfg.algorithm.discount_factor * critic[1:].detach() * (
            1 - done[1:].float()
        )
        td = target - critic[:-1]

        ### COMPUTE CRITIC LOSS
        td_error = td ** 2
        critic_loss = td_error.mean()

        ### COMPUTE ENTROPY
        entropy_loss = torch.distributions.Categorical(action_probs).entropy().mean()

        ### COMPUTE ACTION LOSS
        action_logp = _index(action_probs, action).log()
        a2c_loss = action_logp[:-1] * td.detach()
        a2c_loss = a2c_loss.mean()

        #LOG
        logger.add_scalar("critic_loss", critic_loss.item(), epoch)
        logger.add_scalar("entropy_loss", entropy_loss.item(), epoch)
        logger.add_scalar("a2c_loss", a2c_loss.item(), epoch)

        loss = (
            -cfg.algorithm.entropy_coef * entropy_loss
            + cfg.algorithm.critic_coef * critic_loss
            - cfg.algorithm.a2c_coef * a2c_loss
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ### LOG env/cumulated_reward FOR TERMINAL STATES
        creward = workspace["env/cumulated_reward"]
        creward = creward[done]
        if creward.size()[0] > 0:
            logger.add_scalar("reward", creward.mean().item(), epoch)


@hydra.main(config_path=".", config_name="main.yaml")
def main(cfg):
    import torch.multiprocessing as mp

    mp.set_start_method("spawn")
    run_a2c(cfg)


if __name__ == "__main__":
    main()
