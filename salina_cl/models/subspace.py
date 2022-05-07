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
        if task._task_id > 0:
            r0, self.policy_agent, self.critic_agent, infos = self.k_shot.run(self.policy_agent, self.critic_agent, env_agent,logger, self.seed)
            self.policy_agent.add_anchor(logger = logger)
            self.critic_agent.add_anchor(n_anchors = self.policy_agent[0].n_anchors,logger = logger)
        r1, self.policy_agent, self.critic_agent, infos = self.train_algorithm.run(self.policy_agent, self.critic_agent, env_agent,logger, self.seed, n_max_interactions = task.n_interactions(), infos = infos)
        r2, self.policy_agent, self.critic_agent, infos = self.alpha_search.run(self.policy_agent, self.critic_agent, env_agent, logger, self.seed, task.task_id(), infos = infos)

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
        metrics={}
        env_agent = task.make()
        policy_agent, critic_agent =self.get_evaluation_agent(task.task_id())
#
        #best_alpha = policy_agent[0].best_alpha.to(self.cfg.evaluation.device)
        #logger.message("Best alpha for task "+str(task._task_id)+":"+str(best_alpha.tolist()))
        #best_alpha = torch.cat([best_alpha,torch.zeros(policy_agent[0].n_anchors - best_alpha.shape[-1]).to(self.cfg.evaluation.device)], dim = 0)
        #alphas = torch.stack([best_alpha for _ in range(self.cfg.evaluation.n_evaluation_envs)], dim = 0)
#
        #policy_agent.agents = policy_agent.agents[1:] #deleting alpha agent
        #policy_agent.eval()
        #no_autoreset=EpisodesDone()
        #acquisition_agent=TemporalAgent(Agents(env_agent,no_autoreset,policy_agent))
        #acquisition_agent.seed(self.seed*13+self._stage*100)
        #acquisition_agent.to(self.cfg.evaluation.device)
        #avg_reward=0.0
        #n=0
        #avg_success=0.0
        #w = Workspace()
        #for r in range(self.cfg.evaluation.n_rollouts):
        #    w.set_full("alphas",torch.stack([alphas for _ in range(1001)],dim=0))
        #    acquisition_agent(w, t=0, stop_variable="env/done")
        #    ep_lengths = w["env/done"].max(0)[1]+1
        #    B = ep_lengths.size()[0]
        #    arange=torch.arange(B).to(ep_lengths.device)
        #    cr=w["env/cumulated_reward"][ep_lengths-1,arange]
        #    avg_reward+=cr.sum().item()
        #    if self.cfg.evaluation.evaluate_success:
        #        cr=w["env/success"][ep_lengths-1,arange]
        #        avg_success+=cr.sum().item()
        #    n += B
        #avg_reward /= n
        #metrics["best_alpha/avg_reward"] = avg_reward
        #if self.cfg.evaluation.evaluate_success:
        #    avg_success /= n
        #    metrics["success_rate"] = avg_success

        # Oracle and full value estimation
        alphas = draw_alphas(policy_agent[0].n_anchors,self.cfg.evaluation.steps, self.cfg.evaluation.scale).to(self.cfg.evaluation.device)
        best_alpha_train = policy_agent[0].best_alpha.to(self.cfg.evaluation.device)
        best_alpha_train = torch.cat([best_alpha_train,torch.zeros(alphas.shape[-1] - best_alpha_train.shape[-1]).to(self.cfg.evaluation.device)], dim = 0)
        alphas = torch.cat([best_alpha_train.unsqueeze(0),alphas],dim = 0)
        policy_agent.agents = policy_agent.agents[1:] #deleting alpha agent
        policy_agent.eval()
        oracle_task = copy.deepcopy(task)
        oracle_task._env_agent_cfg["n_envs"] = alphas.shape[0]
        env_agent = oracle_task.make()
        no_autoreset = EpisodesDone()
        acquisition_agent = TemporalAgent(Agents(env_agent, no_autoreset, policy_agent)).to(self.cfg.evaluation.device)
        critic_agent.eval().to(self.cfg.evaluation.device)
        acquisition_agent.seed(self.seed*13+self._stage*100)
        rewards = []
        values = []
        w = Workspace()
        w.set_full("alphas",torch.stack([alphas for _ in range(1001)],dim=0))
        for i in range(self.cfg.evaluation.n_rollouts):
            with torch.no_grad():
                acquisition_agent(w, t = 0, stop_variable = "env/done")
                critic_agent(w)
            ep_lengths= w["env/done"].max(0)[1]+1
            B = ep_lengths.size()[0]
            arange = torch.arange(B).to(ep_lengths.device)
            cr = w["env/cumulated_reward"][ep_lengths-1,arange]
            rewards.append(cr)
            values.append(w["q1"].mean(0))
        rewards = torch.stack(rewards, dim = 0).mean(0)
        values = torch.stack(values, dim = 0).mean(0)
        #if alphas.shape[-1] == 2:
        #    logger_images = logger.get_logger("")
        #    logger_images.prefix = "stage_"+str(policy_agent[0].n_anchors - 1)+"/task_"+str(task.task_id())
        #    image = display_kshot_2anchors(alphas.cpu(),values.round().cpu(),"task_"+str(task.task_id()))
        #    logger_images.add_figure("/value_distribution",image,0)
        #    image = display_kshot_2anchors(alphas.cpu(),rewards.round().cpu(),"task_"+str(task.task_id()))
        #    logger_images.add_figure("/reward_distribution",image,0)
        #elif alphas.shape[-1] == 3:
        #    logger_images = logger.get_logger("")
        #    logger_images.prefix = "stage_"+str(policy_agent[0].n_anchors - 1)+"/task_"+str(task.task_id())
        #    image = display_kshot_3anchors(alphas.cpu(),values.round().cpu(),"task_"+str(task.task_id()))
        #    logger_images.add_figure("/value_distribution",image,0)
        #    image = display_kshot_3anchors(alphas.cpu(),rewards.round().cpu(),"task_"+str(task.task_id()))
        #    logger_images.add_figure("/reward_distribution",image,0)
        metrics["best_alpha/avg_reward"] = rewards[0].item()
        metrics["avg_reward"] = rewards.mean().item()
        metrics["oracle/avg_reward"] = rewards.max().item()
        metrics["value/avg_reward"] = rewards[values.argmax()].item()
        metrics["midpoint/avg_reward"] = rewards[1].item()
        metrics["last_anchor/avg_reward"] = rewards[-1].item()
        del w
        return metrics