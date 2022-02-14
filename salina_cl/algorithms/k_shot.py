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
        logger = logger.get_logger(type(self).__name__+str("/"))
        if (action_agent[0].n_anchors > 1) and (n_max_interactions > 0):
            action_agent.eval()
            acquisition_agent = TemporalAgent(Agents(env_agent, action_agent)).to(self.cfg.acquisition_device)
            acquisition_agent.seed(seed)
            if self.cfg.n_processes > 1:
                acquisition_agent, workspace = NRemoteAgent.create(acquisition_agent, num_processes=self.cfg.n_processes, time_size=self.cfg.n_timesteps, n_steps=1)
            n_interactions = 0
            rewards = []
            _training_start_time = time.time()
            logger.message("Starting k-shot procedure")
            alphas = action_agent[0].dist.sample(torch.Size([self.cfg.n_envs])).to(action_agent[0].id.device)
            while True:
                w = Workspace()
                w.set("alphas",0,alphas)
                with torch.no_grad():
                    acquisition_agent(w, t = 0, stop_variable = "env/done", k_shot = True)
                w = w.select_batch(torch.LongTensor(list(range(self.cfg.k))))
                length=w["env/done"].max(0)[1]
                alphas_print = w["alphas"][0]
                n_interactions += length.sum().item()
                arange = torch.arange(length.size()[0], device=length.device)
                rewards.append(w["env/cumulated_reward"][length, arange])
                if (n_interactions + length.sum().item()) > n_max_interactions:
                    logger.message("k-shot ends with "+str(n_interactions)+" interactions used.")
                    break
            rewards = torch.stack(rewards, dim = 0).mean(0)
            best_alpha = w["alphas"][0,rewards.argmax()].reshape(-1)
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