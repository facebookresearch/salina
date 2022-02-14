#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import time
import torch
from salina import Workspace
from salina.agents import Agents, TemporalAgent
from salina.agents.remote import NRemoteAgent

class k_shot:
    def __init__(self,params):
        self.cfg = params
    
    def run(self,action_agent, critic_agent, env_agent, logger, seed, n_max_interactions, add_anchor = True):
        n_epochs = int(n_max_interactions // (env_agent.n_envs * env_agent.make_env_args['max_episode_steps']))
        if (action_agent[0].n_anchors > 1) and (n_epochs > 0):
            action_agent.eval()
            acquisition_agent = TemporalAgent(Agents(env_agent, action_agent)).to(self.cfg.acquisition_device)
            acquisition_agent.seed(seed)
            if self.cfg.n_processes > 1:
                acquisition_agent, workspace = NRemoteAgent.create(acquisition_agent, num_processes=self.cfg.n_processes, time_size=self.cfg.n_timesteps, n_steps=1)
            n_interactions = 0
            rewards = []
            _training_start_time = time.time()
            logger.message("Starting k-shot procedure on "+str(int(n_epochs * env_agent.n_envs))+" episodes")
            for i in range(n_epochs):
                w = Workspace()
                with torch.no_grad():
                    acquisition_agent(w, t = 0, stop_variable = "env/done", force_random_alpha = True)
                length=w["env/done"].max(0)[1]
                n_interactions += length.sum().item()
                arange = torch.arange(length.size()[0], device=length.device)
                rewards.append(w["env/cumulated_reward"][length, arange])
            rewards = torch.stack(rewards, dim = 0).mean(0)
            best_alpha = w["alphas"][0,rewards.argmax()].reshape(-1)
            logger = logger.get_logger(type(self).__name__)
            logger.add_scalar("mean_reward", rewards.mean().item(), 0)
            logger.add_scalar("max_reward", rewards.max().item(), 0)
            logger.message("mean reward:"+str(round(rewards.mean().item(),0)))
            logger.message("max reward:"+str(round(rewards.max().item(),0)))
            logger.message("best alpha:"+str(best_alpha))
            r = {"n_epochs":0,"training_time":time.time()-_training_start_time,"n_interactions":n_interactions}
        else:
            best_alpha = None
            r = {"n_epochs":0,"training_time":0,"n_interactions":0}
        if add_anchor:
            action_agent.add_anchor(alpha = best_alpha,logger=logger)
            critic_agent.add_anchor(logger = logger)
        return r, action_agent, critic_agent