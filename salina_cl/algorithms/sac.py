#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import copy
import time
import torch
import numpy as np
from salina import Workspace, get_arguments, get_class
from salina.agents import Agents, TemporalAgent, EpisodesDone
from salina.agents.remote import NRemoteAgent
from salina.rl.replay_buffer import ReplayBuffer
from salina_cl.algorithms.tools import compute_time_unit, _state_dict, soft_update_params
from salina_cl.agents.subspace_agents_sac import SubspaceAgents

class sac:
    def __init__(self,params):
        self.cfg = params
    
    def run(self, action_agent, q_agent, env_agent,logger, seed, n_max_interactions):
        time_unit=None
        logger = logger.get_logger(type(self).__name__+str("/"))
        
        # import ipdb;ipdb.set_trace()
        cfg=self.cfg
        if cfg.time_limit>0:
            time_unit=compute_time_unit(cfg.device)
            logger.message("Time unit is "+str(time_unit)+" seconds.")
        inner_epochs = int(cfg.inner_epochs * cfg.grad_updates_per_step)
        logger.message("Nb of updates per epoch: "+str(inner_epochs))
    
        action_agent.set_name("action")
        acq_agent = TemporalAgent(Agents(env_agent, copy.deepcopy(action_agent))).to(cfg.acquisition_device)
        acquisition_workspace=Workspace()
        if cfg.n_processes>1:
            acq_agent,acquisition_workspace = NRemoteAgent.create(acq_agent, num_processes=cfg.n_processes, time_size=cfg.n_timesteps, n_steps=1)
        acq_agent.seed(seed)
    
        control_agent=TemporalAgent(Agents(copy.deepcopy(env_agent), EpisodesDone(), copy.deepcopy(action_agent))).to(cfg.acquisition_device)
        control_agent.seed(seed)
        control_agent.eval()
    
        # == Setting up the training agents
        action_agent.to(cfg.learning_device)
        q_target_agent = copy.deepcopy(q_agent)
        q_target_agent.to(cfg.learning_device)
        q_agent.to(cfg.learning_device)
    
        # == Setting up & initializing the replay buffer for DQN
        replay_buffer = ReplayBuffer(cfg.buffer_size,device=cfg.buffer_device)
        acq_agent.train()
        action_agent.train()
        logger.message("Initializing replay buffer")
        acq_agent(acquisition_workspace, t=0, n_steps=cfg.n_timesteps)
        n_interactions = acquisition_workspace.time_size() * acquisition_workspace.batch_size()
        replay_buffer.put(acquisition_workspace, time_size=cfg.buffer_time_size)
        while replay_buffer.size() < cfg.initial_buffer_size:
            acquisition_workspace.copy_n_last_steps(1)
            with torch.no_grad():
                acq_agent(acquisition_workspace, t=1, n_steps=cfg.n_timesteps - 1)
            replay_buffer.put(acquisition_workspace, time_size=cfg.buffer_time_size)
            n_interactions += (acquisition_workspace.time_size() - 1) * acquisition_workspace.batch_size()

        # == configuring SAC entropy
        optimizer_args = get_arguments(cfg.optimizer_entropy)
        action_shape = acquisition_workspace["action"].size()[2:]
        target_entropy = -0.5 * np.prod(np.array(action_shape)) * cfg.target_multiplier
        log_entropy = torch.tensor(np.log(cfg.init_temperature), requires_grad=True, device=cfg.learning_device)
        optimizer_entropy = get_class(cfg.optimizer_entropy)([log_entropy], **optimizer_args)

        optimizer_args = get_arguments(cfg.optimizer_q)
        optimizer_q = get_class(cfg.optimizer_q)(q_agent.parameters(), **optimizer_args)
    
        optimizer_args = get_arguments(cfg.optimizer_policy)
        optimizer_action = get_class(cfg.optimizer_policy)(action_agent.parameters(), **optimizer_args)
        iteration = 0
        epoch=0
        is_training=True
        _training_start_time=time.time()
        best_model=None
        best_performance=None
        logger.message("Start training")
        ## there is a warmup of 1000 steps before training
        while is_training:
            # Compute average performance of multiple rollouts
            if epoch%cfg.control_every_n_epochs==0:
                for a in control_agent.get_by_name("action"):
                    a.load_state_dict(_state_dict(action_agent, cfg.acquisition_device))
                control_agent.eval()
                rewards=[]
                for _ in range(cfg.n_control_rollouts):
                    w=Workspace()
                    control_agent(w, t=0, force_random = True, stop_variable="env/done")
                    length=w["env/done"].max(0)[1]
                    arange = torch.arange(length.size()[0], device=length.device)
                    creward = w["env/cumulated_reward"][length, arange]
                    rewards=rewards+creward.to("cpu").tolist()
                    if "env/success" in w.variables:
                        success_rate = w["env/success"][length, arange].mean().item()
                        logger.add_scalar("validation/success_rate", success_rate, epoch)
                    if "env/goalDist" in w.variables:
                        goalDist = w["env/goalDist"][length, arange].mean().item()
                        logger.add_scalar("monitor/goalDist", goalDist, epoch)
    
                mean_reward=np.mean(rewards)
                logger.add_scalar("validation/reward", mean_reward, epoch)
                print("reward at ",epoch," = ",mean_reward," vs ",best_performance)
    
                logger.add_scalar("validation/best_reward", best_performance, epoch)

            for a in acq_agent.get_by_name("action"):
                a.load_state_dict(_state_dict(action_agent, cfg.acquisition_device))

            acquisition_workspace.copy_n_last_steps(1)
            with torch.no_grad():
                acq_agent(acquisition_workspace,t=1,n_steps=cfg.n_timesteps - 1)
            replay_buffer.put(acquisition_workspace, time_size=cfg.buffer_time_size)
            done, creward = acquisition_workspace["env/done", "env/cumulated_reward"]
    
            creward = creward[done]
            if creward.size()[0] > 0:
                logger.add_scalar("monitor/reward", creward.mean().item(), epoch)
                if "env/success" in acquisition_workspace.variables:
                    success_rate = acquisition_workspace["env/success"][done].mean().item()
                    logger.add_scalar("monitor/success_rate", success_rate, epoch)
                if "env/goalDist" in acquisition_workspace.variables:
                    goalDist = acquisition_workspace["env/goalDist"][done].mean().item()
                    logger.add_scalar("monitor/goalDist", goalDist, epoch)
            #logger.add_scalar("monitor/replay_buffer_size", replay_buffer.size(), epoch)
    
            n_interactions += (acquisition_workspace.time_size() - 1) * acquisition_workspace.batch_size()
            logger.add_scalar("monitor/n_interactions", n_interactions, epoch)
    
            _st_inner_epoch=time.time()
            for inner_epoch in range(inner_epochs):
                entropy = log_entropy.exp()
                replay_workspace = replay_buffer.get(cfg.batch_size).to(cfg.learning_device)
                done, reward = replay_workspace["env/done", "env/reward"]
                not_done = 1.0 - done.float()
                reward = reward * cfg.reward_scaling

                # == q1 and q2 losses
                q_agent(replay_workspace)
                q_1 = replay_workspace["q1"]
                q_2 = replay_workspace["q2"]
                with torch.no_grad():
                    action_agent(replay_workspace, q_update = True)
                    q_target_agent(replay_workspace, q_update = True)
                    q_target_1 = replay_workspace["q1"]
                    q_target_2 = replay_workspace["q2"]
                    _logp = replay_workspace["action_logprobs"]
                    q_target = torch.min(q_target_1, q_target_2)
                    target = (reward[1:]+ cfg.discount_factor * not_done[1:] * (q_target[1:] - (entropy * _logp[1:]).detach()))
                td_1 = ((q_1[:-1] - target) ** 2).mean()
                td_2 = ((q_2[:-1] - target) ** 2).mean()
                optimizer_q.zero_grad()
                loss = td_1 + td_2
                #logger.add_scalar("loss/td_loss_1", td_1.item(), iteration)
                #logger.add_scalar("loss/td_loss_2", td_2.item(), iteration)
                loss.backward()
                if cfg.clip_grad > 0:
                    n = torch.nn.utils.clip_grad_norm_(q_agent.parameters(), cfg.clip_grad)
                    #logger.add_scalar("monitor/grad_norm_q", n.item(), iteration)
                optimizer_q.step()

                # == Actor and entropy losses
                if iteration % cfg.policy_update_delay == 0:
                    action_agent(replay_workspace, policy_update = True)
                    q_agent(replay_workspace, policy_update = True)
                    logp = replay_workspace["action_logprobs"]
                    q1 = replay_workspace["q1"]
                    q2 = replay_workspace["q2"]
                    qloss = torch.min(q1,q2).mean()
                    entropy_loss = (entropy.detach() * logp).mean()
                    optimizer_action.zero_grad()
                    loss = - qloss + entropy_loss
                    loss.backward()
                    if cfg.clip_grad > 0:
                        n = torch.nn.utils.clip_grad_norm_(action_agent.parameters(), cfg.clip_grad)
                        #logger.add_scalar("monitor/grad_norm_action", n.item(), iteration)
                    #logger.add_scalar("loss/q_loss", qloss.item(), iteration)
                    optimizer_action.step()

                    optimizer_entropy.zero_grad()
                    entropy_loss = (log_entropy.exp() * (-logp - target_entropy).detach()).mean()
                    entropy_loss.backward()
                    if cfg.clip_grad > 0:
                        n = torch.nn.utils.clip_grad_norm_(log_entropy, cfg.clip_grad)
                    optimizer_entropy.step()
                    #logger.add_scalar("loss/entropy_loss", entropy_loss.item(), iteration)
                    #logger.add_scalar("loss/entropy_value", entropy.item(), iteration)
    
                # == Target network update
                if iteration % cfg.target_update_delay == 0:
                    tau = cfg.update_target_tau
                    soft_update_params(q_agent[0], q_target_agent[0], tau)
                    soft_update_params(q_agent[1], q_target_agent[1], tau)


                iteration += 1
            _et_inner_epoch=time.time()
            #logger.add_scalar("monitor/epoch_time", _et_inner_epoch - _st_inner_epoch,epoch)
            epoch += 1

            if isinstance(action_agent,SubspaceAgents):
                L2_norms = action_agent[-1].L2_norms()
                cosine_similarities = action_agent[-1].cosine_similarities()
                for layer in L2_norms:
                    for penalty in L2_norms[layer]:
                        logger.add_scalar("L2_norm/"+layer+"/"+penalty,L2_norms[layer][penalty],epoch)
                for layer in cosine_similarities:
                    for penalty in cosine_similarities[layer]:
                        logger.add_scalar("Cos_sim/"+layer+"/"+penalty,cosine_similarities[layer][penalty],epoch)




            if n_interactions > n_max_interactions:
                logger.message("== Maximum interactions reached")
                is_training = False
            else:
                if cfg.time_limit>0:
                    is_training = time.time() - _training_start_time < cfg.time_limit*time_unit

        r = {"n_epochs":epoch, "training_time":time.time() - _training_start_time, "n_interactions":n_interactions}
        if cfg.n_processes>1: acq_agent.close()
        #action_agent, q_agent = best_model
        infos = {"replay_buffer":replay_buffer}
        return r, action_agent.to("cpu"), q_agent.to("cpu"), infos