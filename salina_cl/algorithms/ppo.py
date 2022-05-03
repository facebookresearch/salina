#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import copy
import time
import torch
import salina.rl.functional as RLF
from salina import Workspace, get_arguments, get_class
from salina.agents import Agents, TemporalAgent, EpisodesDone
import numpy as np
from salina_cl.algorithms.tools import compute_time_unit, _state_dict, clip_grad
from salina_cl.algorithms.tools import display_kshot
from salina.agents.remote import NRemoteAgent

class ppo:
    def __init__(self,params):
        self.cfg_ppo = params
    
    def run(self,action_agent, critic_agent, env_agent, logger, seed, n_max_interactions):
        logger = logger.get_logger(type(self).__name__+str("/"))
        action_agent.train()
        critic_agent.train()
        time_unit = None
        best_alpha = None
        best_reward = - float("inf")
        if self.cfg_ppo.time_limit>0:
            time_unit=compute_time_unit(self.cfg_ppo.device)
            logger.message("Time unit is "+str(time_unit)+" seconds.")

        action_agent.set_name("action")
        acq_action_agent = copy.deepcopy(action_agent)
        acquisition_agent = TemporalAgent(Agents(env_agent, acq_action_agent)).to(self.cfg_ppo.acquisition_device)
        acquisition_workspace = Workspace()
        if self.cfg_ppo.n_processes > 1:
            acquisition_agent,acquisition_workspace = NRemoteAgent.create(acquisition_agent, num_processes=self.cfg_ppo.n_processes, time_size=self.cfg_ppo.n_timesteps, n_steps=1)
        acquisition_agent.seed(seed)

        if self.cfg_ppo.n_control_rollouts > 0:
            control_env_agent = copy.deepcopy(env_agent)
            control_action_agent = copy.deepcopy(action_agent).to(self.cfg_ppo.acquisition_device) 
            control_agent = TemporalAgent(Agents(control_env_agent, EpisodesDone(), control_action_agent)).to(self.cfg_ppo.acquisition_device)  
            control_env_agent.to(self.cfg_ppo.acquisition_device)
            control_agent.seed(seed)
            control_agent.eval()

        train_agent = Agents(action_agent, critic_agent).to(self.cfg_ppo.learning_device)

        optimizer_args = get_arguments(self.cfg_ppo.optimizer_policy)
        optimizer_policy = get_class(self.cfg_ppo.optimizer_policy)(action_agent.parameters(), **optimizer_args)
        
        optimizer_args = get_arguments(self.cfg_ppo.optimizer_critic)
        optimizer_critic = get_class(self.cfg_ppo.optimizer_critic)(critic_agent.parameters(), **optimizer_args)

        # === Running algorithm
        epoch = 0
        iteration = 0
        n_interactions = 0

        _training_start_time = time.time()
        best_model = None
        best_performance = None
        normalized_env_obs = []
        alphas_for_ve = []
        j = 0
        while True:
        # Compute average performance of multiple rollouts
            if (self.cfg_ppo.n_control_rollouts > 0) and (epoch%self.cfg_ppo.control_every_n_epochs==0):
                for a in control_agent.get_by_name("action"):
                    a.load_state_dict(_state_dict(action_agent, self.cfg_ppo.acquisition_device))
                control_agent.train()
                rewards=[]
                for _ in range(self.cfg_ppo.n_control_rollouts):
                    w=Workspace()
                    control_agent(w, t = 0, stop_variable = "env/done", force_random = True)
                    length=w["env/done"].max(0)[1]
                    #n_interactions+=length.sum().item()
                    arange = torch.arange(length.size()[0], device=length.device)
                    creward = w["env/cumulated_reward"][length, arange]
                    rewards=rewards+creward.to("cpu").tolist()

                mean_reward=np.mean(rewards)
                max_reward=np.max(rewards)
                logger.add_scalar("validation/reward", mean_reward, epoch)
                logger.add_scalar("validation/best_reward",max_reward, epoch)
                print("reward at ",epoch," = ",mean_reward," vs ",best_performance)
            
                if best_performance is None or mean_reward >= best_performance:
                    best_performance = mean_reward
                    best_model = copy.deepcopy(control_agent.get_by_name("action")[0]),copy.deepcopy(critic_agent)

            # Acquisition of trajectories
            for a in acquisition_agent.get_by_name("action"):
                a.load_state_dict(_state_dict(action_agent, self.cfg_ppo.acquisition_device))

            acquisition_workspace.zero_grad()
            if epoch > 0: acquisition_workspace.copy_n_last_steps(1)
            acquisition_agent.train()
            acquisition_agent( acquisition_workspace, t=1 if epoch > 0 else 0, n_steps=self.cfg_ppo.n_timesteps - 1 if epoch > 0 else self.cfg_ppo.n_timesteps, action_std=self.cfg_ppo.action_std)
            
            
            workspace = Workspace(acquisition_workspace).to(self.cfg_ppo.learning_device)
            workspace.set_full("acquisition_action_logprobs",workspace["action_logprobs"].detach())
            workspace.set_full("acquisition_action",workspace["action"].detach())
            workspace.set_full("env/normalized_env_obs",workspace["env/normalized_env_obs"].detach())
            normalized_env_obs.append(workspace["env/normalized_env_obs"])
            alphas_for_ve.append(workspace["alphas"])


            if n_interactions+(workspace.time_size()-1)*workspace.batch_size() > n_max_interactions:
                logger.message("== Maximum interactions reached")
                break
            n_interactions += (workspace.time_size()-1)*workspace.batch_size()
            logger.add_scalar("monitor/n_interactions", n_interactions, epoch)

            # Log cumulated reward of training trajectories
            d=workspace["env/done"]
            if d.any():
                rewards = workspace["env/cumulated_reward"][d].round().cpu()
                if "alphas" in list(workspace.keys()) and rewards.max() > best_reward:
                    normalized_env_obs = []
                    alphas_for_ve = []
                    #alphas = workspace["alphas"][d].cpu()
                    best_alpha = workspace["alphas"][d].cpu()[rewards.argmax()]
                    #image = display_kshot(alphas,rewards)
                    #logger.add_figure("alphas_drawn",image,epoch)
                    best_reward = max(best_reward,rewards.max())
                    logger.message("Found new best reward: "+str(int(best_reward.item())))
                r = rewards.mean().item()
                logger.add_scalar("monitor/avg_training_reward",r,epoch)
                if "env/success" in list(workspace.keys()):
                    r=workspace["env/success"][d].mean().item()
                    logger.add_scalar("monitor/success",r,epoch)
            workspace.zero_grad()

            #Building mini workspaces
            #Learning for cfg.algorithm.update_epochs epochs
            miniworkspaces=[]
            _stb = time.time()
            for _ in range(self.cfg_ppo.n_minibatches):
                miniworkspace=workspace.sample_subworkspace(self.cfg_ppo.n_times_per_minibatch,self.cfg_ppo.n_envs_per_minibatch,self.cfg_ppo.n_timesteps_per_minibatch)
                miniworkspaces.append(miniworkspace)
            del workspace
            _etb = time.time()
            logger.add_scalar("monitor/minibatches_building_time",_etb-_stb,epoch)

            #Learning on batches
            train_agent.train()
            while len(miniworkspaces) > 0:
                miniworkspace = miniworkspaces.pop()
                old_action_lp = miniworkspace["acquisition_action_logprobs"]
                
                train_agent(miniworkspace, t=None, action_std=self.cfg_ppo.action_std)
                critic, done, reward = miniworkspace["critic", "env/done", "env/reward"]
                reward = reward * self.cfg_ppo.reward_scaling

                # === Update policy
                if (iteration % self.cfg_ppo.policy_update_delay) == 0:
                    gae = RLF.gae(critic,reward,done,self.cfg_ppo.discount_factor,self.cfg_ppo.gae).detach()
                    action_lp = miniworkspace["action_logprobs"]
                    ratio = action_lp - old_action_lp
                    ratio = ratio.exp()
                    ratio = ratio[:-1]
                    clip_adv = torch.clamp(ratio,1 - self.cfg_ppo.clip_ratio,1 + self.cfg_ppo.clip_ratio) * gae
                    loss_regularizer = action_agent.add_regularizer()
                    loss_policy = -(torch.min(ratio * gae, clip_adv)).mean()
                    loss = loss_policy + loss_regularizer
                    optimizer_policy.zero_grad()
                    loss.backward()
                    n = clip_grad(action_agent.parameters(), self.cfg_ppo.clip_grad)
                    optimizer_policy.step()
                    logger.add_scalar("monitor/grad_norm_policy", n.item(), iteration)
                    logger.add_scalar("loss/policy", loss_policy.item(), iteration)
                    logger.add_scalar("loss/regularizer", loss_regularizer.item(), iteration)
                
                # === Update critic
                td0 = RLF.temporal_difference(critic, reward, done, self.cfg_ppo.discount_factor)
                loss_critic = (td0 ** 2).mean()
                optimizer_critic.zero_grad()
                loss_critic.backward()
                n = clip_grad(critic_agent.parameters(), self.cfg_ppo.clip_grad)
                optimizer_critic.step()
                logger.add_scalar("loss/critic", loss_critic.item(), iteration)
                logger.add_scalar("monitor/grad_norm_critic", n.item(), iteration)
                iteration += 1
            epoch += 1

        r = {"n_epochs":epoch,"training_time":time.time()-_training_start_time,"n_interactions":n_interactions}
        if self.cfg_ppo.n_control_rollouts == 0:
            action_agent, critic_agent = copy.deepcopy(acquisition_agent.get_by_name("action")[0]),copy.deepcopy(critic_agent)
        else:
            action_agent, critic_agent = best_model
        
        action_agent.to("cpu")
        critic_agent.to("cpu")
        if self.cfg_ppo.n_processes>1: acquisition_agent.close()
        normalized_env_obs = torch.cat(normalized_env_obs,dim=0)
        logger.message("Number of states collected for value estimation: "+str(np.prod(normalized_env_obs.shape[:-1])))
        alphas_for_ve = torch.cat(alphas_for_ve,dim=0)
        infos = {"normalized_env_obs":normalized_env_obs, "best_alpha":best_alpha, "alphas_for_ve":alphas_for_ve}
        return r,action_agent,critic_agent, infos
