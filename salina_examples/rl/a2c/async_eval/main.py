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


class ProbAgent(TAgent):
    """This agent outputs:
    - action_probs: the lob probabilities of each action

    """

    def __init__(self, observation_size, hidden_size, n_actions):
        super().__init__()
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
    # 1)  Build the  logger
    logger = instantiate_class(cfg.logger)

    # 2) Create the environment agent
    # This agent implements N gym environments with auto-reset
    env_agent = AutoResetGymAgent(
        get_class(cfg.algorithm.env), get_arguments(cfg.algorithm.env)
    )

    # 3) Create the A2C Agent
    env = instantiate_class(cfg.algorithm.env)
    observation_size = env.observation_space.shape[0]
    n_actions = env.action_space.n
    del env
    prob_agent = ProbAgent(
        observation_size, cfg.algorithm.architecture.hidden_size, n_actions
    )
    action_agent = ActionAgent()
    acquisition_agent = Agents(prob_agent, action_agent)
    critic_agent = CriticAgent(
        observation_size, cfg.algorithm.architecture.hidden_size, n_actions
    )

    # 4) Combine env and policy agents
    agent = Agents(env_agent, acquisition_agent)

    # 5) Get an agent that is executed on a complete workspace
    agent = TemporalAgent(agent)

    # 5 bis) The agent is transformed to a remoteagent working on multiple cpus
    agent = RemoteAgent(agent, num_processes=cfg.algorithm.n_processes)
    agent.seed(cfg.algorithm.env_seed)

    # --- EVALUATION
    evaluation_env_agent = AutoResetGymAgent(
        get_class(cfg.algorithm.env), get_arguments(cfg.algorithm.env)
    )
    evaluation_acquisition_agent = copy.deepcopy(acquisition_agent)
    evaluation_agent = TemporalAgent(
        Agents(evaluation_env_agent, evaluation_acquisition_agent)
    )
    evaluation_agent = RemoteAgent(
        evaluation_agent, num_processes=cfg.algorithm.evaluation.n_processes
    )
    evaluation_agent.seed(cfg.algorithm.evaluation.env_seed)

    # 5 bis) Create the temporal critic agent to compute critic values over the workspace
    # Create the agent to recompute action probabilities
    tprob_agent = TemporalAgent(prob_agent)
    tcritic_agent = TemporalAgent(critic_agent)

    # ---- Now, we need to also have an evaluation agent

    # 6) Configure the workspace to the right dimension
    workspace = salina.Workspace(
        batch_size=cfg.algorithm.n_envs,
        time_size=cfg.algorithm.n_timesteps,
    )

    # Creation of the evaluation workspace
    evaluation_workspace = salina.Workspace(
        batch_size=cfg.algorithm.evaluation.n_envs,
        time_size=cfg.algorithm.evaluation.n_timesteps,
    )

    # 7) Confgure the optimizer over the a2c agent
    optimizer_args = get_arguments(cfg.algorithm.optimizer)
    parameters = nn.Sequential(prob_agent, critic_agent).parameters()
    optimizer = get_class(cfg.algorithm.optimizer)(parameters, **optimizer_args)

    # We launch the evaluation agent
    evaluation_workspace = evaluation_agent.asynchronous_forward_(
        evaluation_workspace, stochastic=False
    )

    # 8) Training loop
    epoch = 0
    for epoch in range(cfg.algorithm.max_epochs):
        # At each epoch, if the evaluation_agent is not running, then we log the reward
        if not evaluation_agent.is_running():
            done, creward = evaluation_workspace["env/done", "env/cumulated_reward"]
            creward = creward[done]
            if creward.size()[0] > 0:
                creward = creward.mean().item()
                logger.add_scalar("evaluation/reward", creward, epoch)

            # We relaunch the evalution
            evaluation_acquisition_agent.load_state_dict(acquisition_agent.state_dict())
            evaluation_workspace.copy_time(from_time=-1, to_time=0)
            evaluation_agent.asynchronous_forward_(
                evaluation_workspace, t=1, stochastic=False
            )

        # Execute the agent on the workspace
        if epoch > 0:
            # To avoid to loose a transition, the last element of the workspace is copied at the first timestep (see README)
            workspace.copy_time(from_time=-1, to_time=0)
            agent(workspace, t=1, stochastic=True)
        else:
            workspace = agent(workspace, stochastic=True)

        # Since agent is a RemoteAgent, it computes a SharedWorkspace without gradient
        # First this sharedworkspace has to be converted to a classical workspace
        replay_workspace = workspace.convert_to_workspace()

        # Then probabilities and critic have to be (re) computed to get gradient
        tprob_agent(replay_workspace)
        tcritic_agent(replay_workspace)

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
