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
from salina import Workspace
from salina_cl.algorithms.tools import display_kshot
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

    def run(self,action_agent, critic_agent, logger, infos, task_id):
        logger = logger.get_logger(type(self).__name__+str("/"))
        if (action_agent[0].n_anchors > 1):
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
            #image = display_kshot(alphas[0].cpu(),values.round().cpu())
            #logger.add_figure(str(task_id)+"/values_distribution",image,0)
            best_alpha = alphas[0,values.argmax()].reshape(-1)
            logger.message("Time elapsed: "+str(round(time.time() - _training_start_time,0))+" sec")
            r = {"n_epochs":0,"training_time":time.time()-_training_start_time,"n_interactions":0}
        else:
            best_alpha = None
            r = {"n_epochs":0,"training_time":0,"n_interactions":0}
        action_agent.set_best_alpha(alpha = best_alpha, logger=logger)
        infos["best_alpha"] = best_alpha
        return r, action_agent, critic_agent, infos