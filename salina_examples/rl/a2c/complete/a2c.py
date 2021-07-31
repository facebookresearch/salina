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


def _state_dict(agent, device):
    sd = agent.state_dict()
    for k, v in sd.items():
        sd[k] = v.to(device)
    return sd


def run_a2c(a2c_agent, logger, cfg):
    # 2) Create the environment agent
    # This agent implements N gym environments with auto-reset
    env_agent = AutoResetGymAgent(
        get_class(cfg.algorithm.env), get_arguments(cfg.algorithm.env)
    )

    # 3) Create the A2C Agent
    a2c_agent_cuda = copy.deepcopy(a2c_agent)

    # 4) Combine env and policy agents
    agent = Agents(env_agent, a2c_agent)

    # 5) Get an agent that is executed on a complete workspace
    agent = TemporalAgent(agent)

    # 5 bis) The agent is transformed to a remoteagent working on multiple cpus
    agent = RemoteAgent(agent, num_processes=cfg.algorithm.n_processes)
    agent.seed(cfg.algorithm.env_seed)

    # 5 bis) Create the temporal critic agent to compute critic values over the workspace
    # Create the agent to recompute action probabilities
    ta2c_agent = TemporalAgent(a2c_agent_cuda).to(cfg.algorithm.device)

    # 6) Configure the workspace to the right dimension
    workspace = salina.Workspace(
        batch_size=cfg.algorithm.n_envs,
        time_size=cfg.algorithm.n_timesteps,
    )

    # 7) Confgure the optimizer over the a2c agent
    optimizer_args = get_arguments(cfg.algorithm.optimizer)
    parameters = a2c_agent_cuda.parameters()
    optimizer = get_class(cfg.algorithm.optimizer)(parameters, **optimizer_args)

    # 8) Training loop
    epoch = 0
    for epoch in range(cfg.algorithm.max_epochs):
        # Before smapling, the acquisition agent has to be sync with the loss agent
        a2c_agent.load_state_dict(_state_dict(a2c_agent_cuda, "cpu"))

        # Execute the agent on the workspace
        if epoch > 0:
            ts = workspace.time_size()
            for tt in range(cfg.algorithm.overlapping_timesteps):
                workspace.copy_time(
                    from_time=ts - cfg.algorithm.overlapping_timesteps + tt,
                    to_time=tt,
                )

            agent(
                workspace,
                t=cfg.algorithm.overlapping_timesteps,
                replay=False,
                stochastic=True,
            )
        else:
            workspace = agent(workspace, replay=False, stochastic=True)

        # Since agent is a RemoteAgent, it computes a SharedWorkspace without gradient
        # First this sharedworkspace has to be converted to a classical workspace
        replay_workspace = workspace.convert_to_workspace().to(cfg.algorithm.device)

        # Then probabilities and critic have to be (re) computed to get gradient
        ta2c_agent(replay_workspace, replay=True, stochastic=True)

        # Remaining code is exactly the same
        # Get relevant tensors (size are timestep x n_envs x ....)
        critic, done, action_probs, reward, action = replay_workspace[
            "critic", "env/done", "action_probs", "env/reward", "action"
        ]

        burning = torch.zeros_like(reward)
        burning[cfg.algorithm.burning_timesteps :] = 1.0

        td = RLF.gae(
            critic, reward, done, cfg.algorithm.discount_factor, cfg.algorithm.gae
        )

        # Compute critic loss
        td_error = td ** 2

        critic_loss = (td_error * burning[:-1]).mean()

        # Compute entropy loss
        entropy_loss = (
            torch.distributions.Categorical(action_probs).entropy() * burning
        ).mean()

        # Compute A2C loss
        action_logp = _index(action_probs, action).log()
        a2c_loss = action_logp[:-1] * td.detach()
        a2c_loss = (a2c_loss * burning[:-1]).mean()

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

    logger = instantiate_class(cfg.logger)
    logger.save_hps(cfg)
    a2c_agent = instantiate_class(cfg.a2c_agent)
    mp.set_start_method("spawn")
    run_a2c(a2c_agent, logger, cfg)


if __name__ == "__main__":
    main()
