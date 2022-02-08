#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import copy
import math
import time
import torch
from salina import Workspace, get_arguments, get_class
from salina.agents import Agents, TemporalAgent, EpisodesDone
from salina.agents.remote import NRemoteAgent
from salina.rl.replay_buffer import ReplayBuffer
from salina_cl.algorithms.tools import compute_time_unit, _state_dict, soft_update_params


def sac_train(q_agent_1, q_agent_2, action_agent, env_agent,logger, cfg_sac, seed,n_max_interactions,control_env_agent=None):
    time_unit=None
    if cfg_sac.time_limit>0:
        time_unit = compute_time_unit(cfg_sac.device)
        logger.message("Time unit is "+str(time_unit)+" seconds.")

    action_agent.set_name("action")


    acq_action_agent=copy.deepcopy(action_agent)
    acq_agent = TemporalAgent(Agents(env_agent, acq_action_agent)).to(cfg_sac.acquisition_device)
    acquisition_workspace=Workspace()
    if cfg_sac.n_processes>1:
        acq_agent,acquisition_workspace = NRemoteAgent.create(acq_agent, num_processes=cfg_sac.n_processes, time_size=cfg_sac.n_timesteps, n_steps=1)
    acq_agent.seed(seed)

    if not control_env_agent is None:
        control_action_agent=copy.deepcopy(action_agent)
        control_agent=TemporalAgent(Agents(control_env_agent, EpisodesDone(), control_action_agent)).to(cfg_sac.acquisition_device)
        control_agent.seed(seed)

    # == Setting up the training agents
    q_target_agent_1 = copy.deepcopy(q_agent_1)
    q_target_agent_2 = copy.deepcopy(q_agent_2)
    q_agent_1.to(cfg_sac.learning_device)
    q_agent_2.to(cfg_sac.learning_device)
    q_target_agent_1.to(cfg_sac.learning_device)
    q_target_agent_2.to(cfg_sac.learning_device)
    action_agent.to(cfg_sac.learning_device)

    # == Setting up & initializing the replay buffer for DQN
    replay_buffer = ReplayBuffer(cfg_sac.buffer_size,device=cfg_sac.buffer_device)
    acq_agent.train()
    action_agent.train()

    logger.message("[SAC] Initializing replay buffer")
    acq_agent(
        acquisition_workspace,
        t=0,
        n_steps=cfg_sac.n_timesteps,
    )
    replay_buffer.put(acquisition_workspace, time_size=cfg_sac.buffer_time_size)

    while replay_buffer.size() < cfg_sac.initial_buffer_size:
        acquisition_workspace.copy_n_last_steps(1)
        acq_agent(acquisition_workspace,t=1,n_steps=cfg_sac.n_timesteps - 1)
        acquisition_workspace.zero_grad()
        replay_buffer.put(acquisition_workspace, time_size=cfg_sac.buffer_time_size)

    action_shape=acquisition_workspace["action"].size()[2:]
    _target_entropy = -0.5 * torch.prod(torch.Tensor(action_shape).to(cfg_sac.learning_device)).item()
    _log_alpha = torch.tensor(math.log(cfg_sac.alpha), requires_grad=True, device=cfg_sac.learning_device)

    logger.message("[SAC] Learning")
    n_interactions = 0

    optimizer_args = get_arguments(cfg_sac.optimizer_q)
    optimizer_q_1 = get_class(cfg_sac.optimizer_q)(
        q_agent_1.parameters(), **optimizer_args
    )
    optimizer_q_2 = get_class(cfg_sac.optimizer_q)(
        q_agent_2.parameters(), **optimizer_args
    )

    optimizer_args = get_arguments(cfg_sac.optimizer_policy)
    optimizer_action = get_class(cfg_sac.optimizer_policy)(
        action_agent.parameters(), **optimizer_args
    )

    optimizer_args = get_arguments(cfg_sac.optimizer_alpha)
    optimizer_alpha = get_class(cfg_sac.optimizer_alpha)(
        [_log_alpha], **optimizer_args
    )
    iteration = 0

    epoch=0
    is_training=True
    _training_start_time=time.time()
    while is_training:

        if not control_env_agent is None and epoch%cfg_sac.control_every_n_epochs==0:
            for a in control_agent.get_by_name("action"):
                a.load_state_dict(_state_dict(action_agent, cfg_sac.acquisition_device))

            control_agent.eval()
            w=Workspace()
            control_agent(
                w,
                t=0,
                stop_variable="env/done"
            )
            length=w["env/done"].max(0)[1]
            arange = torch.arange(length.size()[0], device=length.device)
            creward = (
                w["env/cumulated_reward"][length, arange]
                .mean()
                .item()
            )
            logger.add_scalar("validation/reward", creward, epoch)
            print("reward at ",epoch," = ",creward)

        for a in acq_agent.get_by_name("action"):
            a.load_state_dict(_state_dict(action_agent, cfg_sac.acquisition_device))

        acquisition_workspace.copy_n_last_steps(1)
        acquisition_workspace.zero_grad()
        acq_agent(
            acquisition_workspace,
            t=1,
            n_steps=cfg_sac.n_timesteps - 1,
        )
        replay_buffer.put(acquisition_workspace, time_size=cfg_sac.buffer_time_size)
        done, creward = acquisition_workspace["env/done", "env/cumulated_reward"]

        creward = creward[done]
        if creward.size()[0] > 0:
            logger.add_scalar("monitor/reward", creward.mean().item(), epoch)
        logger.add_scalar("monitor/replay_buffer_size", replay_buffer.size(), epoch)

        n_interactions += (
            acquisition_workspace.time_size() - 1
        ) * acquisition_workspace.batch_size()
        logger.add_scalar("monitor/n_interactions", n_interactions, epoch)

        _st_inner_epoch=time.time()
        for inner_epoch in range(cfg_sac.inner_epochs):
            _alpha=_log_alpha.exp().detach()
            __e=time.time()
            batch_size = cfg_sac.batch_size
            _workspace=replay_buffer.get(batch_size)
            replay_workspace = _workspace.to(
                cfg_sac.learning_device
            )
            done, reward = replay_workspace["env/done", "env/reward"]
            not_done=1.0-done.float()
            reward=reward*cfg_sac.reward_scaling

            q_agent_1(replay_workspace)
            q_1 = replay_workspace["q"].squeeze(-1)
            q_agent_2(replay_workspace)
            q_2 = replay_workspace["q"].squeeze(-1)
            replay_workspace.clear("q")

            assert not q_1.eq(q_2).all()
            with torch.no_grad():
                action_agent(replay_workspace)

                q_target_agent_1(replay_workspace)
                q_target_1 = replay_workspace["q"]

                q_target_agent_2(replay_workspace)
                q_target_2 = replay_workspace["q"]

            assert not q_target_1.eq(q_target_2).all()

            q_target = torch.min(q_target_1, q_target_2).squeeze(-1)
            _logp=replay_workspace["action_logprobs"].detach()
            target = (
                reward[1:]
                + cfg_sac.discount_factor
                * not_done[1:]
                * (q_target[1:]-_alpha*_logp[1:].detach())
            )

            td_1 = (q_1[:-1] - target)*not_done[:-1]+0.000001
            td_2 = (q_2[:-1] - target)*not_done[:-1]+0.000001
            error_1 = (td_1 ** 2).sqrt()
            error_2 = (td_2 ** 2).sqrt()

            optimizer_q_1.zero_grad()
            optimizer_q_2.zero_grad()
            error = error_1 + error_2
            loss = error.mean()
            logger.add_scalar("loss/td_loss_1", error_1.mean().item(), iteration)
            logger.add_scalar("loss/td_loss_2", error_2.mean().item(), iteration)

            loss.backward()

            if cfg_sac.clip_grad > 0:
                n = torch.nn.utils.clip_grad_norm_(
                    q_agent_1.parameters(), cfg_sac.clip_grad
                )
                logger.add_scalar("monitor/grad_norm_q_1", n.item(), iteration)
                n = torch.nn.utils.clip_grad_norm_(
                    q_agent_2.parameters(), cfg_sac.clip_grad
                )
                logger.add_scalar("monitor/grad_norm_q_2", n.item(), iteration)

            optimizer_q_1.step()
            optimizer_q_2.step()

            #Actor loss
            done = replay_workspace["env/done"]
            not_done = (1.0-done.float())

            action_agent(replay_workspace,deterministic=False,)

            q_agent_1(replay_workspace)
            q1 = replay_workspace["q"].squeeze(-1)

            q_agent_2(replay_workspace)
            q2 = replay_workspace["q"].squeeze(-1)

            assert not q1.eq(q2).all()
            q = torch.min(q1, q2)

            optimizer_action.zero_grad()
            logp=replay_workspace["action_logprobs"]
            loss_1=(not_done*(_alpha*logp)).mean()
            loss_2=(not_done*(-q)).mean()
            loss =loss_1+loss_2
            loss.backward()

            if "action_std" in list(replay_workspace.keys()):
                _std=replay_workspace["action_std"]
                logger.add_scalar("monitor/action_std",_std.exp().mean().item(),iteration)

            if cfg_sac.clip_grad > 0:
                n = torch.nn.utils.clip_grad_norm_(
                    action_agent.parameters(), cfg_sac.clip_grad
                )
                logger.add_scalar("monitor/grad_norm_action", n.item(), iteration)

            logger.add_scalar("loss/q_loss", loss.item(), iteration)
            logger.add_scalar("loss/q_loss/alpha_term", loss_1.item(), iteration)
            logger.add_scalar("loss/q_loss/q_term", loss_2.item(), iteration)
            optimizer_action.step()

            #Alpha loss
            if (cfg_sac.learning_alpha):
                action_agent(replay_workspace,deterministic=False,)
                logp=replay_workspace["action_logprobs"]
                _alpha=_log_alpha.exp()
                alpha_loss = (_alpha *
                            (-logp - _target_entropy).detach()*not_done).mean()
                logger.add_scalar("loss/alpha_loss", alpha_loss.item(), iteration)


                optimizer_alpha.zero_grad()
                alpha_loss.backward()
                n = torch.nn.utils.clip_grad_norm_(
                    [_log_alpha], cfg_sac.clip_grad
                )
                logger.add_scalar("monitor/grad_norm_alpha", n.item(), iteration)

                optimizer_alpha.step()
            _alpha=_log_alpha.exp().item()
            logger.add_scalar("monitor/alpha", _alpha, iteration)

            tau = cfg_sac.update_target_tau
            soft_update_params(q_agent_1, q_target_agent_1, tau)
            soft_update_params(q_agent_2, q_target_agent_2, tau)

            iteration += 1
        _et_inner_epoch=time.time()
        logger.add_scalar("monitor/epoch_time",_et_inner_epoch-_st_inner_epoch,epoch)
        epoch+=1
        if n_interactions>n_max_interactions:
            logger.message("== Maximum interactions reached")
            is_training=False
        else:
            if cfg_sac.time_limit>0:
                is_training=time.time()-_training_start_time<cfg_sac.time_limit*time_unit

    r={"n_epochs":epoch,"training_time":time.time()-_training_start_time,"n_interactions":n_interactions}
    action_agent.to("cpu")
    q_agent_1.to("cpu")
    q_agent_2.to("cpu")
    if cfg_sac.n_processes>1: acq_agent.close()
    return r,action_agent,q_agent_1,q_agent_2,replay_buffer.to("cpu")
