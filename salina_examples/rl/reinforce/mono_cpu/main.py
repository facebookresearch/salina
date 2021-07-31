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
from salina.agents.gym import GymAgent
from salina.logger import TFLogger


def _index(tensor_3d, tensor_2d):
    """This function is used to index a 3d tensors using a 2d tensor"""
    x, y, z = tensor_3d.size()
    t = tensor_3d.reshape(x * y, z)
    tt = tensor_2d.reshape(x * y)
    v = t[torch.arange(x * y), tt]
    v = v.reshape(x, y)
    return v


class REINFORCEAgent(TAgent):
    """This agent outputs:
        - action_probs: the lob probabilities of each action
        - action: a randomly sampled action
        - baseline: the baseline value

    It works in two models: the stochastic mode where action are sampled, and the non-stochastic mode where action are taken using the argmax of the probability
    It is a simple MLP
    Args:
        TAgent ([type]): [description]
    """

    def __init__(self, observation_size, hidden_size, n_actions):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )
        self.critic_model = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, t, stochastic, **args):
        observation = self.get(("env/env_obs", t))
        scores = self.model(observation)
        probs = torch.softmax(scores, dim=-1)
        critic = self.critic_model(observation).squeeze(-1)
        if stochastic:
            action = torch.distributions.Categorical(probs).sample()
        else:
            action = probs.argmax(1)

        self.set(("action", t), action)
        self.set(("action_probs", t), probs)
        self.set(("baseline", t), critic)


def make_cartpole(max_episode_steps):
    return TimeLimit(gym.make("CartPole-v0"), max_episode_steps=max_episode_steps)


def run_reinforce(cfg):
    # 1)  Build the  logger
    logger = instantiate_class(cfg.logger)

    # 2) Create the environment agent
    # This agent implements N gym environments
    env_agent = GymAgent(get_class(cfg.algorithm.env), get_arguments(cfg.algorithm.env))

    # 3) Create the A2C Agent
    env = instantiate_class(cfg.algorithm.env)
    observation_size = env.observation_space.shape[0]
    n_actions = env.action_space.n
    del env
    a2c_agent = REINFORCEAgent(
        observation_size, cfg.algorithm.architecture.hidden_size, n_actions
    )

    # 4) Combine env and a2c agents
    agent = Agents(env_agent, a2c_agent)

    # 5) Get an agent that is executed on a complete workspace
    # When env/done is True, then the agent will stop the forward and not compute the next timesteps
    agent = TemporalAgent(agent, stop_variable="env/done")
    agent.seed(cfg.algorithm.env_seed)

    # 6) Configure the workspace to the right dimension. The time size is greater than the naximum episode size to be able to store all episode states
    workspace = salina.Workspace(
        batch_size=cfg.algorithm.n_envs,
        time_size=cfg.algorithm.env.max_episode_steps + 1,
    )

    # 7) Confgure the optimizer over the a2c agent
    optimizer_args = get_arguments(cfg.algorithm.optimizer)
    optimizer = get_class(cfg.algorithm.optimizer)(
        a2c_agent.parameters(), **optimizer_args
    )

    # 8) Training loop
    epoch = 0
    for epoch in range(cfg.algorithm.max_epochs):

        # Execute the agent on the workspace to sample complete episodes
        # Since not all the variables of workspace will be overwritten, it is better to clear the workspace
        workspace.clear()
        agent(workspace, stochastic=True)

        # Get relevant tensors (size are timestep x n_envs x ....)
        baseline, done, action_probs, reward, action = workspace[
            "baseline", "env/done", "action_probs", "env/reward", "action"
        ]
        r_loss = compute_reinforce_loss(
            reward, action_probs, baseline, action, done, cfg.algorithm.discount_factor
        )

        # Log losses
        [logger.add_scalar(k, v.item(), epoch) for k, v in r_loss.items()]

        loss = (
            -cfg.algorithm.entropy_coef * r_loss["entropy_loss"]
            + cfg.algorithm.baseline_coef * r_loss["baseline_loss"]
            - cfg.algorithm.reinforce_coef * r_loss["reinforce_loss"]
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute the cumulated reward on final_state
        creward = workspace["env/cumulated_reward"]
        tl = done.float().argmax(0)
        creward = creward[tl, torch.arange(creward.size()[1])]
        logger.add_scalar("reward", creward.mean().item(), epoch)


def compute_reinforce_loss(
    reward, action_probabilities, baseline, action, done, discount_factor
):
    """This function computes the reinforce loss, considering that episodes may have different lengths."""
    batch_size = reward.size()[1]

    # Find the first done occurence for each episode
    v_done, trajectories_length = done.float().max(0)
    trajectories_length += 1
    assert v_done.eq(1.0).all()
    max_trajectories_length = trajectories_length.max().item()

    # Shorten trajectories for accelerate computation
    reward = reward[:max_trajectories_length]
    action_probabilities = action_probabilities[:max_trajectories_length]
    baseline = baseline[:max_trajectories_length]
    action = action[:max_trajectories_length]

    # Create a binary mask to mask useless values (of size max_trajectories_length x batch_size)
    arange = (
        torch.arange(max_trajectories_length, device=done.device)
        .unsqueeze(-1)
        .repeat(1, batch_size)
    )
    mask = arange.lt(
        trajectories_length.unsqueeze(0).repeat(max_trajectories_length, 1)
    )
    reward = reward * mask

    # Compute discounted cumulated reward
    cumulated_reward = [torch.zeros_like(reward[-1])]
    for t in range(max_trajectories_length - 1, 0, -1):
        cumulated_reward.append(discount_factor + cumulated_reward[-1] + reward[t])
    cumulated_reward.reverse()
    cumulated_reward = torch.cat([c.unsqueeze(0) for c in cumulated_reward])

    # baseline loss
    g = baseline - cumulated_reward
    baseline_loss = (g) ** 2
    baseline_loss = (baseline_loss * mask).mean()

    # policy loss
    log_probabilities = _index(action_probabilities, action).log()
    policy_loss = log_probabilities * -g.detach()
    policy_loss = policy_loss * mask
    policy_loss = policy_loss.mean()

    # entropy loss
    entropy = torch.distributions.Categorical(action_probabilities).entropy() * mask
    entropy_loss = entropy.mean()

    return {
        "baseline_loss": baseline_loss,
        "reinforce_loss": policy_loss,
        "entropy_loss": entropy_loss,
    }


@hydra.main(config_path=".", config_name="main.yaml")
def main(cfg):
    import torch.multiprocessing as mp

    mp.set_start_method("spawn")
    run_reinforce(cfg)


if __name__ == "__main__":
    main()
