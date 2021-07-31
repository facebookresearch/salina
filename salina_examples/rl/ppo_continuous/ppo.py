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


def run_ppo(ppo_action_agent, ppo_critic_agent, logger, cfg):
    # 2) Create the environment agent
    # This agent implements N gym environments with auto-reset
    env_agent = AutoResetGymAgent(
        get_class(cfg.algorithm.env), get_arguments(cfg.algorithm.env)
    )

    # 3) Create the A2C Agent
    ppo_action_agent_learning = copy.deepcopy(ppo_action_agent)
    ppo_evaluation = copy.deepcopy(ppo_action_agent)
    # 4) Combine env and policy agents
    agent = Agents(env_agent, ppo_action_agent)

    # 5) Get an agent that is executed on a complete workspace
    agent = TemporalAgent(agent)

    # 5 bis) The agent is transformed to a remoteagent working on multiple cpus
    agent = RemoteAgent(agent, num_processes=cfg.algorithm.n_processes)
    agent.seed(cfg.algorithm.env_seed)

    env_evaluation_agent = AutoResetGymAgent(
        get_class(cfg.algorithm.env), get_arguments(cfg.algorithm.env)
    )
    t_evaluation_agent = TemporalAgent(Agents(env_evaluation_agent, ppo_evaluation))
    evaluation_agent = RemoteAgent(
        t_evaluation_agent, num_processes=cfg.algorithm.evaluation.n_processes
    )
    evaluation_agent.seed(cfg.algorithm.evaluation.env_seed)

    # 5 bis) Create the temporal critic agent to compute critic values over the workspace
    # Create the agent to recompute action probabilities
    tppo_action_agent = TemporalAgent(ppo_action_agent_learning).to(
        cfg.algorithm.device
    )
    tppo_critic_agent = TemporalAgent(ppo_critic_agent).to(cfg.algorithm.device)

    # 6) Configure the workspace to the right dimension
    workspace = salina.Workspace(
        batch_size=cfg.algorithm.n_envs,
        time_size=cfg.algorithm.n_timesteps,
    )

    evaluation_workspace = salina.Workspace(
        batch_size=cfg.algorithm.evaluation.n_envs,
        time_size=cfg.algorithm.evaluation.n_timesteps,
    )

    evaluation_workspace = evaluation_agent(
        evaluation_workspace, replay=False, action_variance=0.0
    )

    # 7) Confgure the optimizer over the ppo agent
    optimizer_args = get_arguments(cfg.algorithm.optimizer)
    parameters = tppo_action_agent.parameters()
    optimizer_action = get_class(cfg.algorithm.optimizer)(parameters, **optimizer_args)
    parameters = tppo_critic_agent.parameters()
    optimizer_critic = get_class(cfg.algorithm.optimizer)(parameters, **optimizer_args)

    # 8) Training loop
    epoch = 0
    iteration = 0
    for epoch in range(cfg.algorithm.max_epochs):
        if not evaluation_agent.is_running():
            creward, done = evaluation_workspace["env/cumulated_reward", "env/done"]
            creward = creward[done]
            if creward.size()[0] > 0:
                logger.add_scalar("evaluation/reward", creward.mean().item(), epoch)
            ppo_evaluation.load_state_dict(
                _state_dict(ppo_action_agent_learning, "cpu")
            )
            evaluation_workspace.copy_time(
                from_time=-1,
                to_time=0,
            )
            evaluation_agent.asynchronous_forward_(
                evaluation_workspace, t=1, replay=False, action_variance=0.0
            )

        # Before smapling, the acquisition agent has to be sync with the loss agent
        ppo_action_agent.load_state_dict(_state_dict(ppo_action_agent_learning, "cpu"))

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
                action_variance=cfg.algorithm.action_variance,
            )
        else:
            workspace = agent(
                workspace,
                replay=False,
                action_variance=cfg.algorithm.action_variance,
            )

        replay_workspace = workspace.convert_to_workspace().to(cfg.algorithm.device)

        with torch.no_grad():
            tppo_critic_agent(replay_workspace)
        critic, done, reward, action = replay_workspace[
            "critic", "env/done", "env/reward", "action"
        ]

        gae = RLF.gae(
            critic, reward, done, cfg.algorithm.discount_factor, cfg.algorithm.gae
        ).detach()

        old_action_lp = replay_workspace["action_logprobs"]

        for _ in range(cfg.algorithm.pi_epochs):
            tppo_action_agent(
                replay_workspace,
                replay=True,
                action_variance=cfg.algorithm.action_variance,
            )
            action_lp = replay_workspace["action_logprobs"]
            entropy = replay_workspace["entropy"]
            ratio = (action_lp - old_action_lp).exp()
            ratio = ratio[:-1]
            clip_adv = (
                torch.clamp(
                    ratio, 1 - cfg.algorithm.clip_ratio, 1 + cfg.algorithm.clip_ratio
                )
                * gae
            )
            loss_pi = -(torch.min(ratio * gae, clip_adv)).mean()
            loss = loss_pi - cfg.algorithm.entropy_coef * entropy.mean()
            optimizer_action.zero_grad()
            loss_pi.backward()
            if cfg.algorithm.clip_grad > 0:
                n = torch.nn.utils.clip_grad_norm_(
                    tppo_action_agent.parameters(), cfg.algorithm.clip_grad
                )
                logger.add_scalar("monitor/grad_norm_action", n.item(), iteration)

            optimizer_action.step()
            logger.add_scalar("loss_pi", loss_pi.item(), iteration)
            logger.add_scalar("loss_entropy", entropy.mean().item(), iteration)
            iteration += 1

        for _ in range(cfg.algorithm.v_epochs):
            tppo_critic_agent(replay_workspace)
            critic = replay_workspace["critic"]
            gae = RLF.gae(
                critic, reward, done, cfg.algorithm.discount_factor, cfg.algorithm.gae
            )
            optimizer_critic.zero_grad()
            loss = (gae ** 2).mean() * cfg.algorithm.critic_coef
            logger.add_scalar("loss_v", loss.item(), iteration)
            loss.backward()
            if cfg.algorithm.clip_grad > 0:
                n = torch.nn.utils.clip_grad_norm_(
                    tppo_critic_agent.parameters(), cfg.algorithm.clip_grad
                )
                logger.add_scalar("monitor/grad_norm_critic", n.item(), iteration)

            optimizer_critic.step()
            iteration += 1

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
    action_agent = instantiate_class(cfg.action_agent)
    critic_agent = instantiate_class(cfg.critic_agent)
    mp.set_start_method("spawn")
    run_ppo(action_agent, critic_agent, logger, cfg)


if __name__ == "__main__":
    main()
