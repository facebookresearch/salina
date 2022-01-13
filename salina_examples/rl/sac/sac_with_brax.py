#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import copy
import math
import time

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

import salina
import salina.rl.functional as RLF
from salina import Workspace, get_arguments, get_class, instantiate_class
from salina.agents import Agents, TemporalAgent
from salina.agents.brax import AutoResetBraxAgent,NoAutoResetBraxAgent
from salina.logger import TFLogger
from salina.rl.replay_buffer import ReplayBuffer
from salina_examples import weight_init
from salina import Agent
from brax.envs import create_gym_env
from brax.envs.to_torch import JaxToTorchWrapper

def make_brax_env(env_name):
    e = create_gym_env(env_name)
    return JaxToTorchWrapper(e)

class Normalizer(Agent):
    def __init__(self, env_name):
        super().__init__()
        env = make_brax_env(env_name)

        self.n_features = env.observation_space.shape[0]
        self.n = None

    def forward(self, t, update_normalizer=True, **kwargs):
        input = self.get(("env/env_obs", t))
        assert torch.isnan(input).sum() == 0.0, "problem"
        if update_normalizer:
            self.update(input)
        input = self.normalize(input)
        assert torch.isnan(input).sum() == 0.0, "problem"
        self.set(("env/env_obs", t), input)

    def update(self, x):
        if self.n is None:
            device = x.device
            self.n = torch.zeros(self.n_features).to(device)
            self.mean = torch.zeros(self.n_features).to(device)
            self.mean_diff = torch.zeros(self.n_features).to(device)
            self.var = torch.ones(self.n_features).to(device)
        self.n += 1.0
        last_mean = self.mean.clone()
        self.mean += (x - self.mean).mean(dim=0) / self.n
        self.mean_diff += (x - last_mean).mean(dim=0) * (x - self.mean).mean(dim=0)
        self.var = torch.clamp(self.mean_diff / self.n, min=1e-2)

    def normalize(self, inputs):
        obs_std = torch.sqrt(self.var)
        return (inputs - self.mean) / obs_std

    def seed(self, seed):
        torch.manual_seed(seed)


class NoAgent(Agent):
    def __init__(self):
        super().__init__()

    def forward(self, **kwargs):
        pass


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def _state_dict(agent, device):
    sd = agent.state_dict()
    for k, v in sd.items():
        sd[k] = v.to(device)
    return sd


def run_sac(q_agent_1, q_agent_2, action_agent, logger, cfg):
    if cfg.algorithm.use_observation_normalizer:
        norm_agent = Normalizer(cfg.algorithm.brax_env.env_name)
    else:
        norm_agent = NoAgent()


    action_agent.set_name("action_agent")
    env_agent = AutoResetBraxAgent(
        env_name=cfg.algorithm.brax_env.env_name,
        n_envs=cfg.algorithm.n_envs,
    )
    env_validation_agent = NoAutoResetBraxAgent(
            env_name=cfg.algorithm.brax_env.env_name,
            n_envs=cfg.algorithm.validation.n_envs,
        )
    q_target_agent_1 = copy.deepcopy(q_agent_1)
    q_target_agent_2 = copy.deepcopy(q_agent_2)

    acq_agent = TemporalAgent(Agents(env_agent, norm_agent,action_agent)).to(cfg.algorithm.device)
    acq_agent.seed(cfg.algorithm.env_seed)

    taction_agent=TemporalAgent(action_agent).to(cfg.algorithm.device)

    validation_agent = TemporalAgent(
        Agents(env_validation_agent, norm_agent, action_agent)
    ).to(cfg.algorithm.device)
    validation_agent.seed(cfg.algorithm.validation.env_seed)

    # == Setting up the training agents
    train_temporal_q_agent_1 = TemporalAgent(q_agent_1)
    train_temporal_q_agent_2 = TemporalAgent(q_agent_2)
    train_temporal_action_agent = TemporalAgent(action_agent)
    train_temporal_q_target_agent_1 = TemporalAgent(q_target_agent_1)
    train_temporal_q_target_agent_2 = TemporalAgent(q_target_agent_2)
    train_temporal_q_agent_1.to(cfg.algorithm.device)
    train_temporal_q_agent_2.to(cfg.algorithm.device)
    train_temporal_q_target_agent_1.to(cfg.algorithm.device)
    train_temporal_q_target_agent_2.to(cfg.algorithm.device)

    acq_workspace=Workspace()
    acq_agent(
        acq_workspace,
        deterministic=False,
        t=0,
        n_steps=cfg.algorithm.n_timesteps,
    )

    # == Setting up & initializing the replay buffer for DQN
    replay_buffer = ReplayBuffer(cfg.algorithm.buffer_size,device=torch.device("cpu"))
    replay_buffer.put(acq_workspace, time_size=cfg.algorithm.buffer_time_size)

    logger.message("[DDQN] Initializing replay buffer")
    while replay_buffer.size() < cfg.algorithm.initial_buffer_size:
        acq_workspace.copy_n_last_steps(cfg.algorithm.overlapping_timesteps)
        acq_agent(acq_workspace,t=cfg.algorithm.overlapping_timesteps,deterministic=False,n_steps=cfg.algorithm.n_timesteps - cfg.algorithm.overlapping_timesteps)
        acq_workspace.zero_grad()
        replay_buffer.put(acq_workspace, time_size=cfg.algorithm.buffer_time_size)


    action_shape=acq_workspace["action"].size()[2:]

    _alpha = cfg.algorithm.alpha
    _target_entropy = - 0.5 * torch.prod(torch.Tensor(action_shape).to(cfg.algorithm.device)).item()
    _log_alpha = torch.tensor([math.log(cfg.algorithm.alpha)], requires_grad=True, device=cfg.algorithm.device)
    logger.message("[DDQN] Learning")
    n_interactions = 0
    optimizer_args = get_arguments(cfg.algorithm.optimizer)
    optimizer_q_1 = get_class(cfg.algorithm.optimizer)(
        q_agent_1.parameters(), **optimizer_args
    )
    optimizer_q_2 = get_class(cfg.algorithm.optimizer)(
        q_agent_2.parameters(), **optimizer_args
    )
    optimizer_action = get_class(cfg.algorithm.optimizer)(
        action_agent.parameters(), **optimizer_args
    )
    optimizer_alpha = get_class(cfg.algorithm.optimizer)(
        [_log_alpha], **optimizer_args
    )


    iteration = 0

    for epoch in range(cfg.algorithm.max_epoch):
        # === Validation
        if (epoch % cfg.algorithm.validation.evaluate_every == 0) and epoch>0:
            validation_workspace=Workspace()
            print("Starting evaluation...")
            validation_agent.eval()
            validation_agent(
                validation_workspace,
                t=0,
                stop_variable="env/done",
                deterministic=True,
                update_normalizer=False,
            )
            length = validation_workspace["env/done"].float().argmax(0)
            arange = torch.arange(length.size()[0], device=length.device)
            creward = (
                validation_workspace["env/cumulated_reward"][length, arange]
                .mean()
                .item()
            )
            logger.add_scalar("validation/reward", creward, epoch)
            print("\treward at epoch", epoch, ":\t", round(creward, 0))
            validation_agent.train()

        acq_workspace.copy_n_last_steps(cfg.algorithm.overlapping_timesteps)
        acq_agent(
            acq_workspace,
            deterministic=False,
            t=cfg.algorithm.overlapping_timesteps,
            n_steps=cfg.algorithm.n_timesteps - cfg.algorithm.overlapping_timesteps,
        )
        acq_workspace.zero_grad()
        replay_buffer.put(acq_workspace, time_size=cfg.algorithm.buffer_time_size)

        done, creward = acq_workspace["env/done", "env/cumulated_reward"]
        creward = creward[done]
        if creward.size()[0] > 0:
            logger.add_scalar("monitor/reward", creward.mean().item(), epoch)
        logger.add_scalar("monitor/replay_buffer_size", replay_buffer.size(), epoch)

        n_interactions += (
            acq_workspace.time_size() - cfg.algorithm.overlapping_timesteps
        ) * acq_workspace.batch_size()
        logger.add_scalar("monitor/n_interactions", n_interactions, epoch)

        for inner_epoch in range(cfg.algorithm.inner_epochs):
            batch_size = cfg.algorithm.batch_size
            replay_workspace = replay_buffer.get(batch_size).to(
                cfg.algorithm.device
            )
            done, reward = replay_workspace["env/done", "env/reward"]
            reward=reward*cfg.algorithm.reward_scaling

            train_temporal_q_agent_1(
                replay_workspace,
                t=0,
                n_steps=cfg.algorithm.buffer_time_size,
                detach_action=True,
            )
            q_1 = replay_workspace["q"].squeeze(-1)
            train_temporal_q_agent_2(
                replay_workspace,
                t=0,
                n_steps=cfg.algorithm.buffer_time_size,
                detach_action=True,
            )
            q_2 = replay_workspace["q"].squeeze(-1)
            assert not q_1.eq(q_2).all()
            with torch.no_grad():
                taction_agent(
                    replay_workspace,
                    deterministic=False,
                    t=0,
                    n_steps=cfg.algorithm.buffer_time_size,
                )
                train_temporal_q_target_agent_1(
                    replay_workspace,
                    t=0,
                    detach_action=True,
                    n_steps=cfg.algorithm.buffer_time_size,
                )
                q_target_1 = replay_workspace["q"]
                train_temporal_q_target_agent_2(
                    replay_workspace,
                    t=0,
                    detach_action=True,
                    n_steps=cfg.algorithm.buffer_time_size,
                )
                q_target_2 = replay_workspace["q"]
            assert not q_target_1.eq(q_target_2).all()

            q_target = torch.min(q_target_1, q_target_2).squeeze(-1)
            _logp=replay_workspace["sac/log_prob_action"].detach()
            target = (
                reward[1:]
                + cfg.algorithm.discount_factor
                * (1.0 - done[1:].float())
                * (q_target[1:]-_alpha*_logp[1:])
            )

            td_1 = q_1[:-1] - target
            td_2 = q_2[:-1] - target
            error_1 = td_1 ** 2
            error_2 = td_2 ** 2

            error = error_1 + error_2
            loss = error.mean()
            logger.add_scalar("loss/td_loss_1", error_1.mean().item(), iteration)
            logger.add_scalar("loss/td_loss_2", error_2.mean().item(), iteration)
            optimizer_q_1.zero_grad()
            optimizer_q_2.zero_grad()
            loss.backward()

            if cfg.algorithm.clip_grad > 0:
                n = torch.nn.utils.clip_grad_norm_(
                    q_agent_1.parameters(), cfg.algorithm.clip_grad
                )
                logger.add_scalar("monitor/grad_norm_q_1", n.item(), iteration)
                n = torch.nn.utils.clip_grad_norm_(
                    q_agent_2.parameters(), cfg.algorithm.clip_grad
                )
                logger.add_scalar("monitor/grad_norm_q_2", n.item(), iteration)

            optimizer_q_1.step()
            optimizer_q_2.step()

            if ((inner_epoch+1) % cfg.algorithm.policy_delay==0):
                optimizer_action.zero_grad()
                optimizer_q_1.zero_grad()
                optimizer_q_2.zero_grad()

                batch_size = cfg.algorithm.batch_size
                replay_workspace = Workspace(replay_buffer.get(batch_size)).to(
                    cfg.algorithm.device
                )
                replay_workspace.zero_grad()
                done = replay_workspace["env/done"]

                taction_agent(
                    replay_workspace,
                    deterministic=False,
                    t=0,
                    n_steps=cfg.algorithm.buffer_time_size,
                )
                train_temporal_q_agent_1(
                    replay_workspace,
                    t=0,
                    detach_action=False,
                    n_steps=cfg.algorithm.buffer_time_size,
                )
                q1 = replay_workspace["q"].squeeze(-1)
                train_temporal_q_agent_2(
                    replay_workspace,
                    t=0,
                    detach_action=False,
                    n_steps=cfg.algorithm.buffer_time_size,
                )
                q2 = replay_workspace["q"].squeeze(-1)
                assert not q1.eq(q2).all()
                q = torch.min(q1, q2)


                logp=replay_workspace["sac/log_prob_action"]
                loss =(_alpha*logp-q).mean()
                loss.backward()

                log_std=replay_workspace["sac/log_std"]
                logger.add_scalar("monitor/action_std",log_std.exp().mean().item(),iteration)

                if cfg.algorithm.clip_grad > 0:
                    n = torch.nn.utils.clip_grad_norm_(
                        action_agent.parameters(), cfg.algorithm.clip_grad
                    )
                    logger.add_scalar("monitor/grad_norm_action", n.item(), iteration)

                logger.add_scalar("loss/q_loss", loss.item(), iteration)
                optimizer_action.step()

                T,B=logp.size()
                _e_alpha=_log_alpha.unsqueeze(0).repeat(T,B).exp()
                alpha_loss = (_e_alpha *
                          (-logp - _target_entropy).detach()).mean()
                logger.add_scalar("loss/alpha_loss", alpha_loss.item(), iteration)

                if (cfg.algorithm.learning_alpha):
                    optimizer_alpha.zero_grad()
                    alpha_loss.backward()
                    optimizer_alpha.step()
                    _alpha=_log_alpha.exp().item()
                logger.add_scalar("monitor/alpha", _alpha, iteration)


            tau = cfg.algorithm.update_target_tau
            soft_update_params(q_agent_1, q_target_agent_1, tau)
            soft_update_params(q_agent_2, q_target_agent_2, tau)

            iteration += 1


@hydra.main(config_path=".", config_name="brax.yaml")
def main(cfg):
    import torch.multiprocessing as mp

    # BARX Stuffs
    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        v = torch.ones(1, device="cuda:0")

    mp.set_start_method("spawn")
    logger = instantiate_class(cfg.logger)
    logger.save_hps(cfg)
    q_agent_1 = instantiate_class(cfg.q_agent)
    q_agent_2 = instantiate_class(cfg.q_agent)
    q_agent_2.apply(weight_init)
    action_agent = instantiate_class(cfg.action_agent)
    run_sac(q_agent_1, q_agent_2, action_agent, logger, cfg)


if __name__ == "__main__":
    OmegaConf.register_new_resolver("plus", lambda x, y: x + y)
    OmegaConf.register_new_resolver("n_gpus", lambda x: 0 if x == "cpu" else 1)
    main()
