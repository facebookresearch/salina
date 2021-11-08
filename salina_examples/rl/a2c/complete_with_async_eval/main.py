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
from salina import TAgent, Workspace, get_arguments, get_class, instantiate_class
from salina.agents import Agents, NRemoteAgent, RemoteAgent, TemporalAgent
from salina.agents.gyma import AutoResetGymAgent, GymAgent
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
    logger = instantiate_class(cfg.logger)
    a2c_agent.set_name("a2c_agent")

    assert cfg.algorithm.n_envs % cfg.algorithm.n_processes == 0

    acq_env_agent = AutoResetGymAgent(
        get_class(cfg.algorithm.env),
        get_arguments(cfg.algorithm.env),
        n_envs=int(cfg.algorithm.n_envs / cfg.algorithm.n_processes),
    )
    eval_env_agent = AutoResetGymAgent(
        get_class(cfg.algorithm.env),
        get_arguments(cfg.algorithm.env),
        n_envs=int(
            cfg.algorithm.evaluation.n_envs / cfg.algorithm.evaluation.n_processes
        ),
    )

    acq_a2c_agent = copy.deepcopy(a2c_agent)
    acq_agent = TemporalAgent(Agents(acq_env_agent, acq_a2c_agent))
    acq_remote_agent, acq_workspace = NRemoteAgent.create(
        acq_agent,
        num_processes=cfg.algorithm.n_processes,
        t=0,
        n_steps=cfg.algorithm.n_timesteps,
        stochastic=True,
        replay=False,
    )
    acq_remote_agent.seed(cfg.algorithm.env_seed)

    eval_a2c_agent = copy.deepcopy(a2c_agent)
    eval_agent = TemporalAgent(Agents(eval_env_agent, eval_a2c_agent))
    eval_remote_agent, eval_workspace = NRemoteAgent.create(
        eval_agent,
        num_processes=cfg.algorithm.evaluation.n_processes,
        t=0,
        n_steps=cfg.algorithm.evaluation.n_timesteps,
        stochastic=False,
        replay=False,
    )
    eval_remote_agent.seed(cfg.algorithm.evaluation.env_seed)

    ta2c_agent = TemporalAgent(a2c_agent).to(device=cfg.algorithm.device)

    optimizer_args = get_arguments(cfg.algorithm.optimizer)
    parameters = a2c_agent.parameters()
    optimizer = get_class(cfg.algorithm.optimizer)(parameters, **optimizer_args)

    epoch = 0

    for a in eval_remote_agent.get_by_name("a2c_agent"):
        a.load_state_dict(_state_dict(a2c_agent, "cpu"))
    eval_remote_agent._asynchronous_call(
        eval_workspace,
        t=0,
        n_steps=cfg.algorithm.evaluation.n_timesteps,
        replay=False,
        stochastic=False,
    )

    for epoch in range(cfg.algorithm.max_epochs):
        # Evaluation
        if not eval_remote_agent.is_running():
            done, creward = eval_workspace["env/done", "env/cumulated_reward"]
            creward = creward[done]
            if not creward.size()[0] == 0:
                logger.add_scalar("evaluation/reward", creward.mean().item(), epoch)
            for a in eval_remote_agent.get_by_name("a2c_agent"):
                a.load_state_dict(_state_dict(a2c_agent, "cpu"))
            eval_workspace.copy_n_last_steps(1)
            eval_remote_agent._asynchronous_call(
                eval_workspace,
                t=1,
                n_steps=cfg.algorithm.evaluation.n_timesteps - 1,
                replay=False,
                stochastic=False,
            )

        for a in acq_remote_agent.get_by_name("a2c_agent"):
            a.load_state_dict(_state_dict(a2c_agent, "cpu"))

        if epoch > 0:
            acq_workspace.copy_n_last_steps(1)
            acq_remote_agent(
                acq_workspace,
                t=1,
                replay=False,
                n_steps=cfg.algorithm.n_timesteps - 1,
                stochastic=True,
            )
        else:
            acq_remote_agent(
                acq_workspace,
                t=0,
                replay=False,
                n_steps=cfg.algorithm.n_timesteps,
                stochastic=True,
            )

        replay_workspace = Workspace(acq_workspace).to(cfg.algorithm.device)

        ta2c_agent(
            replay_workspace,
            t=0,
            n_steps=cfg.algorithm.n_timesteps,
            replay=True,
            stochastic=True,
        )

        critic, done, action_probs, reward, action = replay_workspace[
            "critic", "env/done", "action_probs", "env/reward", "action"
        ]

        burning = torch.zeros_like(reward)
        burning[cfg.algorithm.burning_timesteps :] = 1.0

        td = RLF.gae(
            critic, reward, done, cfg.algorithm.discount_factor, cfg.algorithm.gae
        )

        td_error = td ** 2
        critic_loss = (td_error * burning[:-1]).mean()

        entropy_loss = (
            torch.distributions.Categorical(action_probs).entropy() * burning
        ).mean()

        action_logp = _index(action_probs, action).log()
        a2c_loss = action_logp[:-1] * td.detach()
        a2c_loss = (a2c_loss * burning[:-1]).mean()

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
