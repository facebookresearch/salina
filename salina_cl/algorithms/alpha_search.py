#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import time
import torch
from salina_cl.agents.tools import LinearSubspace
from torch.distributions.dirichlet import Dirichlet
from salina.agents import Agents, TemporalAgent
from salina import Workspace
from salina.agents.remote import NRemoteAgent
from ternary.helpers import simplex_iterator

def remove_anchor(model):
    model.agents[1].n_anchors -= 1
    for nn_module in model[1].model:
        if isinstance(nn_module,LinearSubspace):
            nn_module.anchors = nn_module.anchors[:-1]
            nn_module.n_anchors -= 1
    return model

def draw_alphas(n_anchors, steps, scale, batch_size = None):
    midpoint = torch.ones(n_anchors).unsqueeze(0) / n_anchors
    if n_anchors == 1:
        alphas = torch.Tensor([[1.]]* steps)
    if n_anchors == 2:
        alphas = torch.stack([torch.linspace(0.,1.,steps = steps - 1),1 - torch.linspace(0.,1.,steps = steps - 1)],dim=1)
        alphas = torch.cat([midpoint,alphas],dim = 0)
    if n_anchors == 3:
        alphas = torch.Tensor([[i/scale,j/scale,k/scale] for i,j,k in simplex_iterator(scale)])
        alphas = torch.cat([midpoint,alphas],dim = 0)
    if n_anchors > 3:
        dist = Dirichlet(torch.ones(n_anchors))
        last_anchor = torch.Tensor([0] * (n_anchors - 1) + [1]).unsqueeze(0)
        alphas = torch.cat([last_anchor,midpoint,dist.sample(torch.Size([steps - 2]))], dim = 0)
    #alphas = torch.split(alphas, alphas.shape[0] if batch_size is None else batch_size, dim = 0)
    return alphas

class value_estimation:
    def __init__(self,params):
        self.cfg = params

    def run(self,action_agent, critic_agent, task, logger, seed, infos = {}):
        logger = logger.get_logger(type(self).__name__+str("/"))
        if (action_agent[0].n_anchors > 1):

            # Estimating best alpha
            critic_agent.to(self.cfg.device)
            replay_buffer = infos["replay_buffer"]
            alphas = draw_alphas(action_agent[-1].n_anchors,self.cfg.steps, self.cfg.scale).to(self.cfg.device)
            alphas = torch.stack([alphas for _ in range(self.cfg.time_size)], dim=0)
            values = []
            logger.message("Starting value estimation")
            _training_start_time = time.time()
            for _ in range(self.cfg.n_estimations):
                replay_workspace = replay_buffer.get(alphas.shape[1]).to(self.cfg.device)
                replay_workspace.set_full("alphas",alphas)
                with torch.no_grad():
                    critic_agent(replay_workspace)
                values.append(replay_workspace["q1"].mean(0))
            values = torch.stack(values,dim = 0).mean(0)
            best_alpha = alphas[0,values.argmax()].reshape(-1)
            action_agent.set_best_alpha(alpha = best_alpha, logger=logger)
            action_agent.set_task(task.task_id())
            logger.message("best alpha is : "+str(list(map(lambda x:round(x,2),best_alpha.tolist()))))
            logger.message("Time elapsed: "+str(round(time.time() - _training_start_time,0))+" sec")

            del replay_workspace
            del alphas
            del replay_buffer
            

            # Validating best alpha through rollout 
            n_interactions = 0
            action_agent.eval()
            task._env_agent_cfg["n_envs"] = self.cfg.n_rollouts
            env_agent = task.make()
            acquisition_agent = TemporalAgent(Agents(env_agent, action_agent)).to(self.cfg.device)
            acquisition_agent.seed(seed)
            if self.cfg.n_processes > 1:
                acquisition_agent, w = NRemoteAgent.create(acquisition_agent, num_processes=self.cfg.n_processes, time_size=self.cfg.n_timesteps, n_steps=1)
            else:
                w = Workspace()
            with torch.no_grad():
                acquisition_agent(w, t = 0, stop_variable = "env/done")
            length = w["env/done"].max(0)[1]
            n_interactions += length.sum().item()
            arange = torch.arange(length.size()[0], device=length.device)
            rewards = w["env/cumulated_reward"][length, arange].mean()

            # Deciding to keep the anchor or not
            if rewards < infos["best_reward_before_training"] * (1 + self.cfg.improvement_threshold):
                action_agent.remove_anchor(logger=logger)
                action_agent.set_best_alpha(alpha = infos["best_alpha_before_training"], logger=logger)

            r = {"n_epochs":0,"training_time":time.time()-_training_start_time,"n_interactions":n_interactions}
            del w
        else:
            best_alpha = None
            r = {"n_epochs":0,"training_time":0,"n_interactions":0}
            action_agent.set_best_alpha(alpha = best_alpha, logger=logger)
        infos["best_alpha"] = best_alpha
        return r, action_agent, critic_agent, infos

class dual_subspace_estimation:
    def __init__(self,params):
        self.cfg = params

    def run(self,action_agent, critic_agent, task, logger, seed, infos = {}):
        logger = logger.get_logger(type(self).__name__+str("/"))
        if (action_agent[0].n_anchors > 1):

            # Estimating best alpha
            critic_agent.to(self.cfg.device)
            replay_buffer = infos["replay_buffer"]
            alphas = draw_alphas(action_agent[-1].n_anchors,self.cfg.steps, self.cfg.scale).to(self.cfg.device)
            alphas = torch.stack([alphas for _ in range(self.cfg.time_size)], dim=0)
            values = []
            logger.message("Starting value estimation")
            _training_start_time = time.time()
            for _ in range(self.cfg.n_estimations):
                replay_workspace = replay_buffer.get(alphas.shape[1]).to(self.cfg.device)
                replay_workspace.set_full("alphas",alphas)
                with torch.no_grad():
                    critic_agent(replay_workspace)
                values.append(replay_workspace["q1"].mean(0))
            values = torch.stack(values,dim = 0).mean(0)
            best_alpha = alphas[0,values.argmax()].reshape(-1)
            logger.message("best alpha is : "+str(list(map(lambda x:round(x,2),best_alpha.tolist()))))
            logger.message("Time elapsed: "+str(round(time.time() - _training_start_time,0))+" sec")
            
            alphas = draw_alphas(action_agent[-1].n_anchors - 1, self.cfg.steps, self.cfg.scale).to(self.cfg.device)
            if alphas.shape[-1] < action_agent[-1].n_anchors:
                alphas = torch.cat([alphas,torch.zeros(*alphas.shape[:-1],1).to(self.cfg.device)], dim = -1)
            alphas = torch.stack([alphas for _ in range(self.cfg.time_size)], dim=0)
            values = []
            logger.message("Starting value estimation")
            _training_start_time = time.time()
            for _ in range(self.cfg.n_estimations):
                replay_workspace = replay_buffer.get(alphas.shape[1]).to(self.cfg.device)
                replay_workspace.set_full("alphas",alphas)
                with torch.no_grad():
                    critic_agent(replay_workspace)
                values.append(replay_workspace["q1"].mean(0))
            values = torch.stack(values,dim = 0).mean(0)
            best_alpha_before_training = alphas[0,values.argmax()].reshape(-1)
            logger.message("best alpha before training is : "+str(list(map(lambda x:round(x,2),best_alpha_before_training.tolist()))))
            logger.message("Time elapsed: "+str(round(time.time() - _training_start_time,0))+" sec")            
            
            del replay_workspace
            del alphas
            del replay_buffer
            

            # Validating best alpha through rollout
            logger.message("Evaluating the two best alphas...")  
            n_interactions = 0
            B = self.cfg.n_rollouts
            task._env_agent_cfg["n_envs"] = B
            env_agent = task.make()
            alphas = torch.cat([torch.stack([best_alpha for _ in range(B // 2)],dim=0),torch.stack([best_alpha_before_training for _ in range(B - (B // 2))],dim=0)],dim = 0)
            action_agent.eval()
            acquisition_agent = TemporalAgent(Agents(env_agent, action_agent)).to(self.cfg.device)
            acquisition_agent.seed(seed)
            if self.cfg.n_processes > 1:
                acquisition_agent, w = NRemoteAgent.create(acquisition_agent, num_processes=self.cfg.n_processes, time_size=self.cfg.n_timesteps, n_steps=1)
            else:
                w = Workspace()
            with torch.no_grad():
                w.set("alphas",0,alphas)
                acquisition_agent(w, t = 0, stop_variable = "env/done", mute_alpha = True)
            length = w["env/done"].max(0)[1]
            
            n_interactions += length.sum().item()
            arange = torch.arange(length.size()[0], device=length.device)
            best_reward = w["env/cumulated_reward"][length, arange][: B // 2].mean()
            best_reward_before_training = w["env/cumulated_reward"][length, arange][B - (B // 2):].mean()

            # Deciding to keep the anchor or not
            logger.message("best_reward = "+str(round(best_reward.item(),0))) 
            logger.message("best_reward_before_training = "+str(round(best_reward_before_training.item(),0))) 
            logger.message("threshold = "+str(round(best_reward_before_training.item() * (1 + self.cfg.improvement_threshold),0)))
            if best_reward < best_reward_before_training * (1 + self.cfg.improvement_threshold):
                action_agent.set_best_alpha(alpha = best_alpha_before_training, logger=logger)
                action_agent.remove_anchor(logger=logger)
            else:
                action_agent.set_best_alpha(alpha = best_alpha, logger=logger)

            r = {"n_epochs":0,"training_time":time.time()-_training_start_time,"n_interactions":n_interactions}
            del w
        else:
            best_alpha = None
            r = {"n_epochs":0,"training_time":0,"n_interactions":0}
            action_agent.set_best_alpha(alpha = best_alpha, logger=logger)
        infos["best_alpha"] = best_alpha
        return r, action_agent, critic_agent, infos

class dual_subspace_estimation_cw:
    def __init__(self,params):
        self.cfg = params

    def run(self,action_agent, critic_agent, task, logger, seed, infos = {}):
        logger = logger.get_logger(type(self).__name__+str("/"))
        if (action_agent[0].n_anchors > 1):

            # Estimating best alpha
            critic_agent.to(self.cfg.device)
            replay_buffer = infos["replay_buffer"]
            alphas = draw_alphas(action_agent[-1].n_anchors,self.cfg.steps, self.cfg.scale).to(self.cfg.device)
            alphas = torch.stack([alphas for _ in range(self.cfg.time_size)], dim=0)
            values = []
            logger.message("Starting value estimation")
            _training_start_time = time.time()
            for _ in range(self.cfg.n_estimations):
                replay_workspace = replay_buffer.get(alphas.shape[1]).to(self.cfg.device)
                replay_workspace.set_full("alphas",alphas)
                with torch.no_grad():
                    critic_agent(replay_workspace)
                values.append(replay_workspace["q1"].mean(0))
            values = torch.stack(values,dim = 0).mean(0)
            best_alpha = alphas[0,values.argmax()].reshape(-1)
            logger.message("best alpha is : "+str(list(map(lambda x:round(x,2),best_alpha.tolist()))))
            logger.message("Time elapsed: "+str(round(time.time() - _training_start_time,0))+" sec")
            
            alphas = draw_alphas(action_agent[-1].n_anchors - 1, self.cfg.steps, self.cfg.scale).to(self.cfg.device)
            if alphas.shape[-1] < action_agent[-1].n_anchors:
                alphas = torch.cat([alphas,torch.zeros(*alphas.shape[:-1],1).to(self.cfg.device)], dim = -1)
            alphas = torch.stack([alphas for _ in range(self.cfg.time_size)], dim=0)
            values = []
            logger.message("Starting value estimation")
            _training_start_time = time.time()
            for _ in range(self.cfg.n_estimations):
                replay_workspace = replay_buffer.get(alphas.shape[1]).to(self.cfg.device)
                replay_workspace.set_full("alphas",alphas)
                with torch.no_grad():
                    critic_agent(replay_workspace)
                values.append(replay_workspace["q1"].mean(0))
            values = torch.stack(values,dim = 0).mean(0)
            best_alpha_before_training = alphas[0,values.argmax()].reshape(-1)
            logger.message("best alpha before training is : "+str(list(map(lambda x:round(x,2),best_alpha_before_training.tolist()))))
            logger.message("Time elapsed: "+str(round(time.time() - _training_start_time,0))+" sec")            
            
            del replay_workspace
            del alphas
            del replay_buffer
            

            # Validating best alpha through rollout
            logger.message("Evaluating the two best alphas...")  
            n_interactions = 0
            task._env_agent_cfg["n_envs"] = 2
            env_agent = task.make()
            alphas = torch.cat([best_alpha.unsqueeze(0),best_alpha_before_training.unsqueeze(0)],dim = 0)
            action_agent.eval()
            acquisition_agent = TemporalAgent(Agents(env_agent, action_agent)).to(self.cfg.device)
            acquisition_agent.seed(seed)
            if self.cfg.n_processes > 1:
                acquisition_agent, w = NRemoteAgent.create(acquisition_agent, num_processes=self.cfg.n_processes, time_size = 201, n_steps = 201 )
            else:
                w = Workspace()
            best_reward = []
            best_reward_before_training = []
            for i in range(self.cfg.n_rollouts):
                with torch.no_grad():
                    w.set("alphas",0,alphas)
                    acquisition_agent(w, t = 0, stop_variable = "env/done", mute_alpha = True)
                length = w["env/done"].max(0)[1]
                
                n_interactions += length.sum().item()
                arange = torch.arange(length.size()[0], device=length.device)
                best_reward.append(w["env/cumulated_reward"][length, arange][0].unsqueeze(0))
                best_reward_before_training.append(w["env/cumulated_reward"][length, arange][1].unsqueeze(0))
            best_reward = torch.cat(best_reward).mean()
            best_reward_before_training = torch.cat(best_reward_before_training).mean()

            # Deciding to keep the anchor or not
            logger.message("best_reward = "+str(round(best_reward.item(),0))) 
            logger.message("best_reward_before_training = "+str(round(best_reward_before_training.item(),0))) 
            logger.message("threshold = "+str(round(best_reward_before_training.item() * (1 + self.cfg.improvement_threshold),0)))
            if best_reward < best_reward_before_training * (1 + self.cfg.improvement_threshold):
                action_agent.set_best_alpha(alpha = best_alpha_before_training, logger=logger)
                action_agent.remove_anchor(logger=logger)
            else:
                action_agent.set_best_alpha(alpha = best_alpha, logger=logger)

            r = {"n_epochs":0,"training_time":time.time()-_training_start_time,"n_interactions":n_interactions}
            del w
        else:
            best_alpha = None
            r = {"n_epochs":0,"training_time":0,"n_interactions":0}
            action_agent.set_best_alpha(alpha = best_alpha, logger=logger)
        infos["best_alpha"] = best_alpha
        return r, action_agent, critic_agent, infos