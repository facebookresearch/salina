#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from salina_cl.core import Model
from salina.agents import Agents,TemporalAgent
from salina.agents.brax import EpisodesDone
from salina import Workspace
from torch.distributions.dirichlet import Dirichlet
from ternary.helpers import simplex_iterator
from salina import instantiate_class
import torch
import os
import copy
import time
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon

def draw_alphas(n_anchors, steps, scale):
    midpoint = torch.ones(n_anchors).unsqueeze(0) / n_anchors
    if n_anchors == 1:
        alphas = torch.Tensor([[1.]])
    if n_anchors == 2:
        alphas = torch.stack([torch.linspace(0.,1.,steps = steps - 1),1 - torch.linspace(0.,1.,steps = steps - 1)],dim=1)
        alphas = torch.cat([midpoint,alphas],dim = 0)
    if n_anchors == 3:
        alphas = torch.Tensor([[i/scale,j/scale,k/scale] for i,j,k in simplex_iterator(scale)])
        alphas = torch.cat([midpoint,alphas],dim = 0)
    if n_anchors > 3:
        dist = Dirichlet(torch.ones(n_anchors))
        last_anchor = torch.Tensor([0] * (n_anchors - 1) + [1]).unsqueeze(0)
        alphas = torch.cat([midpoint,dist.sample(torch.Size([steps - 2])),last_anchor], dim = 0)
    return alphas

def display_kshot(alphas,rewards):
    fig, ax = plt.subplots(figsize = (10,8))
    plt.axis('off')
    n_anchors = alphas.shape[1]
    radius = 0.5
    center = (0.5,0.5)

    subspace = RegularPolygon((0.5,0.5),n_anchors,radius = radius, fc=(1,1,1,0), edgecolor="black")
    anchors = subspace.get_path().vertices[:-1] * radius + center

    for i,anchor in enumerate(anchors):
        x = anchor[0] + (anchor[0]-center[0]) * 0.1
        y = anchor[1] + (anchor[1]-center[1]) * 0.1
        ax.text(x,y,"θ"+str(i+1),fontsize="x-large")

    coordinates = (alphas @ anchors).T
    ax.add_artist(subspace)
    points = ax.scatter(coordinates[0],coordinates[1],c=rewards, cmap="RdYlGn", s=5)
    ax.scatter(coordinates[0][rewards.argmax()],coordinates[1][rewards.argmax()], s=300, color="darkgreen", marker="x")
    ax.set_xlim(0.,1.)
    ax.set_ylim(0.,1.)

    cbar = fig.colorbar(points, ax=ax, pad=0.1)
    minVal = int(rewards.min().item())
    maxVal = int(rewards.max().item())
    cbar.set_ticks([minVal, maxVal])
    cbar.set_ticklabels([minVal, maxVal])

    return fig


def display_time(last_time, line):
    new_time = time.time()
    print(line,": ",round(new_time-last_time,2),"sec")
    return new_time


class Subspace(Model):
    """

    """
    def __init__(self,seed,params):
        super().__init__(seed,params)
        self.k_shot = instantiate_class(self.cfg.k_shot)
        self.train_algorithm = instantiate_class(self.cfg.algorithm)
        self.alpha_search = instantiate_class(self.cfg.alpha_search)
        self.policy_agent = None
        self.critic_agent = None
        self.best_alpha_train = []
        self.lr_policy = self.cfg.algorithm.params.optimizer_policy.lr

    def _create_policy_agent(self,task,logger):
        logger.message("Creating policy Agent")
        assert self.policy_agent is None
        input_dimension = task.input_dimension()
        output_dimension = task.output_dimension()
        policy_agent_cfg = self.cfg.policy_agent
        policy_agent_cfg.input_dimension = input_dimension
        policy_agent_cfg.output_dimension = output_dimension
        self.policy_agent = instantiate_class(policy_agent_cfg)

    def _create_critic_agent(self,task,logger):
        logger.message("Creating Critic Agent")
        obs_dimension = task.input_dimension()
        action_dimension = task.output_dimension()
        critic_agent_cfg = self.cfg.critic_agent
        critic_agent_cfg.obs_dimension = obs_dimension
        critic_agent_cfg.action_dimension = action_dimension
        critic_agent_cfg.n_anchors = self.policy_agent[0].n_anchors
        self.critic_agent = instantiate_class(critic_agent_cfg)

    def _train(self,task,logger):
        if self.policy_agent is None:
            self._create_policy_agent(task,logger)
            self._create_critic_agent(task,logger)

        env_agent = task.make()
        self.train_algorithm.cfg.optimizer_policy.lr = self.lr_policy * (1 + task._task_id * self.cfg.lr_scaling)
        logger.message("Setting policy_lr to "+str(self.train_algorithm.cfg.optimizer_policy.lr))
        infos = {}
        r0 = {"n_interactions":0}
        if task._task_id > 0:
            #r0, self.policy_agent, self.critic_agent, infos = self.k_shot.run(self.policy_agent, self.critic_agent, env_agent,logger, self.seed)
            self.policy_agent.add_anchor(logger = logger)
            self.critic_agent.add_anchor(n_anchors = self.policy_agent[0].n_anchors,logger = logger)
        r1, self.policy_agent, self.critic_agent, infos = self.train_algorithm.run(self.policy_agent, self.critic_agent, env_agent,logger, self.seed, n_max_interactions = task.n_interactions() - r0["n_interactions"], infos = infos)
        r2, self.policy_agent, self.critic_agent, infos = self.alpha_search.run(self.policy_agent, self.critic_agent, task, logger, self.seed, infos = infos)

        if self.cfg.checkpoint:
            torch.save(self.critic_agent,os.getcwd()+"/critic_"+str(task._task_id)+".dat")
            torch.save(self.policy_agent,os.getcwd()+"/policy_"+str(task._task_id)+".dat")
            os.makedirs(os.getcwd()+"/replay_buffer_"+str(task._task_id))
            for variable in infos["replay_buffer"].variables:
                v = variable.replace("/","_")
                torch.save(infos["replay_buffer"].variables[variable].cpu(),os.getcwd()+"/replay_buffer_"+str(task._task_id)+"/"+v+".dat")
            del infos
        return r1

    def memory_size(self):
        pytorch_total_params = sum(p.numel() for p in self.policy_agent.parameters())
        return {"n_parameters":pytorch_total_params}

    def get_evaluation_agent(self,task_id):
        self.policy_agent.set_task(task_id)
        return copy.deepcopy(self.policy_agent),self.critic_agent


    def evaluate(self,test_tasks,logger):
        """ Evaluate a model over a set of test tasks
        Args:
            test_tasks: The set of tasks to evaluate on
            logger
        Returns:
            evaluation: Some statistics about the evaluation (i.e metrics)
        """
        logger.message("Starting evaluation...")
        with torch.no_grad():
            evaluation={}
            for k,task in enumerate(test_tasks):
                metrics=self._evaluate_single_task(task,logger)
                evaluation[task.task_id()]=metrics
                logger.message("Evaluation over task "+str(k)+":"+str(metrics))

        logger.message("-- End evaluation...")
        return evaluation


    def _evaluate_single_task(self,task,logger):
        
        env_agent = task.make()
        policy_agent, _ =self.get_evaluation_agent(task.task_id())
        policy_agent.eval()
        env_agent = task.make()
        acquisition_agent = TemporalAgent(Agents(env_agent, EpisodesDone(), policy_agent)).to(self.cfg.evaluation.device)
        acquisition_agent.seed(self.seed*13+self._stage*100)
        rewards = []
        successes = []
        goalDists = []
        w = Workspace()
        for i in range(self.cfg.evaluation.n_rollouts):
            with torch.no_grad():
                acquisition_agent(w, t = 0, stop_variable = "env/done")
            ep_lengths= w["env/done"].max(0)[1]+1
            B = ep_lengths.size()[0]
            arange = torch.arange(B).to(ep_lengths.device)
            cr = w["env/cumulated_reward"][ep_lengths-1,arange]
            if self.cfg.evaluation.evaluate_success:
                success = w["env/success"][ep_lengths-1,arange]
                goalDist = w["env/goalDist"][ep_lengths-1,arange]
            rewards.append(cr)
            successes.append(success)
            goalDists.append(goalDist)
        rewards = torch.stack(rewards, dim = 0).mean()
        successes = torch.stack(successes, dim = 0).mean()
        goalDists = torch.stack(goalDists, dim = 0).mean()
        metrics={ "best_alpha/avg_reward" : rewards.item(),
                  "best_alpha/avg_success" : successes.item(),
                  "best_alpha/avg_goalDist" : goalDists.item()
                 }
        del w
        return metrics