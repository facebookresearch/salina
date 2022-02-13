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
from salina.agents.remote import NRemoteAgent

class k_shot:
    def __init__(self,params):
        self.cfg = params
    
    def run(self,action_agent, critic_agent, env_agent, logger, seed, n_max_interactions):
        action_agent.eval()
        acquisition_agent = TemporalAgent(Agents(env_agent, action_agent)).to(self.cfg.acquisition_device)
        acquisition_agent.seed(seed)
        if self.cfg.n_processes > 1:
            acquisition_agent, workspace = NRemoteAgent.create(acquisition_agent, num_processes=self.cfg.n_processes, time_size=self.cfg.n_timesteps, n_steps=1)
        n_interactions = 0
        rewards = []
        _training_start_time = time.time()

        n_epochs = int(n_max_interactions // (env_agent.n_envs * env_agent.make_env_args['max_episode_steps']))
        logger.message("Starting k-shot procedure on "+str(int(n_epochs * env_agent.n_envs))+" episodes")
        if n_epochs > 0:
            for i in range(n_epochs):
                w = Workspace()
                with torch.no_grad():
                    acquisition_agent(w, t = 0, stop_variable = "env/done")
                length=w["env/done"].max(0)[1]
                n_interactions += length.sum().item()
                arange = torch.arange(length.size()[0], device=length.device)
                rewards.append(w["env/cumulated_reward"][length, arange])
            rewards = torch.stack(rewards, dim = 0).mean(0)
            best_alpha = w["alphas"][0,rewards.argmax()].reshape(-1)
            logger.add_scalar("k_shot/mean_reward", rewards.mean().item(), 0)
            logger.add_scalar("k_shot/max_reward", rewards.max().item(), 0)
            logger.message("mean reward:"+str(round(rewards.mean().item(),2)))
            logger.message("max reward:"+str(round(rewards.max().item(),2)))
            logger.message("best alpha:"+str(best_alpha))
            r = {"n_epochs":0,"training_time":time.time()-_training_start_time,"n_interactions":n_interactions}
            return r, action_agent, critic_agent, best_alpha
        else:
            r = {"n_epochs":0,"training_time":time.time()-_training_start_time,"n_interactions":n_interactions}
            return r, action_agent, critic_agent, None
