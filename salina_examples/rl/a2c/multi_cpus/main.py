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
from salina import TAgent, AgentArray,get_arguments, get_class, instantiate_class, create_shared_workspaces_array,Workspace
from salina.agents import Agents, TemporalAgent, RemoteAgentArray
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


class ProbAgent(TAgent):
    """This agent outputs:
    - action_probs: the lob probabilities of each action

    """

    def __init__(self, observation_size, hidden_size, n_actions):
        super().__init__(name="prob_agent")
        self.model = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, t, **args):
        observation = self.get(("env/env_obs", t))
        scores = self.model(observation)
        probs = torch.softmax(scores, dim=-1)
        self.set(("action_probs", t), probs)


class ActionAgent(TAgent):
    """This agent chooses which action to output depending on the probabilities"""

    def __init__(self):
        super().__init__()

    def forward(self, t, stochastic, **args):
        probs = self.get(("action_probs", t))
        if stochastic:
            action = torch.distributions.Categorical(probs).sample()
        else:
            action = probs.argmax(1)

        self.set(("action", t), action)


class CriticAgent(TAgent):
    """This agent outputs the critic value"""

    def __init__(self, observation_size, hidden_size, n_actions):
        super().__init__()
        self.critic_model = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, t, **args):
        observation = self.get(("env/env_obs", t))
        critic = self.critic_model(observation).squeeze(-1)
        self.set(("critic", t), critic)


def make_cartpole(max_episode_steps):
    return TimeLimit(gym.make("CartPole-v0"), max_episode_steps=max_episode_steps)


def run_a2c(cfg):
    # Build the  logger
    logger = instantiate_class(cfg.logger)

    # Get info on the environment
    env = instantiate_class(cfg.algorithm.env)
    observation_size = env.observation_space.shape[0]
    n_actions = env.action_space.n
    del env

    assert cfg.algorithm.n_envs%cfg.algorithm.n_processes==0

    # Create the agents
    acq_env_agent = AutoResetGymAgent(
        get_class(cfg.algorithm.env), get_arguments(cfg.algorithm.env),n_envs=int(cfg.algorithm.n_envs/cfg.algorithm.n_processes)
    )
    prob_agent= ProbAgent(observation_size, cfg.algorithm.architecture.hidden_size, n_actions)
    acq_prob_agent = copy.deepcopy(prob_agent)
    acq_action_agent = ActionAgent()
    acq_agent=TemporalAgent(Agents(acq_env_agent,acq_prob_agent,acq_action_agent))
    acq_remote_agents=[copy.deepcopy(acq_agent) for k in range(cfg.algorithm.n_processes)]
    critic_agent = CriticAgent(observation_size, cfg.algorithm.architecture.hidden_size, n_actions)

    acq_agents=RemoteAgentArray(acq_remote_agents)
    #acq_agents=RemoteAgentArray(acq_remote_agents)
    acq_agents.seed(cfg.algorithm.env_seed)
    acq_agent.seed(0)
    acq_workspaces=create_shared_workspaces_array(acq_agent,t=0,n_steps=cfg.algorithm.n_timesteps,stochastic=True,n_workspaces=cfg.algorithm.n_processes)

    # 5 bis) Create the temporal critic agent to compute critic values over the workspace
    # Create the agent to recompute action probabilities
    tprob_agent = TemporalAgent(prob_agent)
    tcritic_agent = TemporalAgent(critic_agent)


    # 7) Confgure the optimizer over the a2c agent
    optimizer_args = get_arguments(cfg.algorithm.optimizer)
    parameters = nn.Sequential(prob_agent, critic_agent).parameters()
    optimizer = get_class(cfg.algorithm.optimizer)(parameters, **optimizer_args)

    # 8) Training loop
    epoch = 0
    for epoch in range(cfg.algorithm.max_epochs):
        for a in acq_remote_agents:
            pagent=a.get_by_name("prob_agent")
            pagent[0].load_state_dict(prob_agent.state_dict())
        # Execute the agent on the workspace
        if epoch > 0:
            # To avoid to loose a transition, the last element of the workspace is copied at the first timestep (see README)
            acq_workspaces.copy_n_last_steps(1)
            acq_agents(acq_workspaces, t=1, n_steps=cfg.algorithm.n_timesteps-1,stochastic=True)
        else:
            acq_agents(acq_workspaces, t=0, n_steps=cfg.algorithm.n_timesteps,stochastic=True)

        # Since agent is a RemoteAgent, it computes a SharedWorkspace without gradient
        # First this sharedworkspace has to be converted to a classical workspace
        # print("ICI")
        # print(acq_workspaces["env/timestep"])
        # time.sleep(1.0)
        replay_workspace = acq_workspaces.to_workspace()
        # Then probabilities and critic have to be (re) computed to get gradient
        tprob_agent(replay_workspace,t=0, n_steps=cfg.algorithm.n_timesteps)
        tcritic_agent(replay_workspace,t=0, n_steps=cfg.algorithm.n_timesteps)

        # Remaining code is exactly the same
        # Get relevant tensors (size are timestep x n_envs x ....)
        critic, done, action_probs, reward, action = replay_workspace[
            "critic", "env/done", "action_probs", "env/reward", "action"
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
        entropy_loss = torch.distributions.Categorical(action_probs).entropy().mean()

        # Compute A2C loss
        action_logp = _index(action_probs, action).log()
        a2c_loss = action_logp[:-1] * td.detach()
        a2c_loss = a2c_loss.mean()

        # Log losses
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

        # Compute the cumulated reward on final_state
        creward = replay_workspace["env/cumulated_reward"]
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
