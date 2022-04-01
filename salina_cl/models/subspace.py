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
from salina_cl.algorithms.tools import display_kshot
from ternary.helpers import simplex_iterator
from salina import instantiate_class
import torch
import os
import copy

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


class TwoSteps(Model):
    """
    A model that is using 2 algorithms. 
    """
    def __init__(self,seed,params):
        super().__init__(seed,params)
        self.algorithm0 = instantiate_class(self.cfg.algorithm0)
        self.algorithm1 = instantiate_class(self.cfg.algorithm1)
        self.algorithm2 = instantiate_class(self.cfg.algorithm2)
        self.policy_agent = None
        self.critic_agent = None
        self.best_alpha_train = []
        self.lr_policy = self.cfg.algorithm1.params.optimizer_policy.lr

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
        input_dimension = task.input_dimension()
        critic_agent_cfg = self.cfg.critic_agent
        critic_agent_cfg.input_dimension = input_dimension
        critic_agent_cfg.n_anchors = self.policy_agent[0].n_anchors
        self.critic_agent = instantiate_class(critic_agent_cfg)

    def _train(self,task,logger):
        if self.policy_agent is None:
            self._create_policy_agent(task,logger)
            self._create_critic_agent(task,logger)

        env_agent = task.make()
        self.algorithm1.cfg_ppo.optimizer_policy.lr = self.lr_policy * (1 + task._task_id * self.cfg.lr_scaling)
        logger.message("Setting policy_lr to "+str(self.algorithm1.cfg_ppo.optimizer_policy.lr))

        
        if task._task_id >0:
            budget0 = task.n_interactions() * self.cfg.algorithm0.params.budget
            r0, self.policy_agent, self.critic_agent = self.algorithm0.run(self.policy_agent, self.critic_agent, env_agent,logger, self.seed, n_max_interactions = budget0)
        #budget1 = task.n_interactions() * (1 - self.cfg.algorithm2.params.budget) if (task._task_id>0) else task.n_interactions()
        r1, self.policy_agent, self.critic_agent, infos = self.algorithm1.run(self.policy_agent, self.critic_agent, env_agent,logger, self.seed, n_max_interactions = task.n_interactions())
        torch.save(self.critic_agent,os.getcwd()+"/critic_"+str(task._task_id)+".dat")
        self.best_alpha_train.append(infos["best_alpha"])

        #budget2 = task.n_interactions() #- r1["n_interactions"]
        r2, self.policy_agent, self.critic_agent = self.algorithm2.run(self.policy_agent, self.critic_agent, logger, infos, task._task_id)
    
        return {kv1[0]:kv1[1]+kv2[1]  for kv1,kv2 in zip(r1.items(),r2.items())}

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
        env_agent=task.make()
        policy_agent, critic_agent =self.get_evaluation_agent(task.task_id())

        if not policy_agent is None:
            policy_agent.eval()
            no_autoreset=EpisodesDone()
            acquisition_agent=TemporalAgent(Agents(env_agent,no_autoreset,policy_agent))
            acquisition_agent.seed(self.seed*13+self._stage*100)
            acquisition_agent.to(self.cfg.evaluation.device)

            avg_reward=0.0
            n=0
            avg_success=0.0
            for r in range(self.cfg.evaluation.n_rollouts):
                workspace=Workspace()
                acquisition_agent(workspace,t=0,stop_variable="env/done")
                ep_lengths=workspace["env/done"].max(0)[1]+1
                B=ep_lengths.size()[0]
                arange=torch.arange(B).to(ep_lengths.device)
                cr=workspace["env/cumulated_reward"][ep_lengths-1,arange]
                avg_reward+=cr.sum().item()
                if self.cfg.evaluation.evaluate_success:
                    cr=workspace["env/success"][ep_lengths-1,arange]
                    avg_success+=cr.sum().item()
                n+=B
            avg_reward/=n
            metrics["avg_reward"]=avg_reward
            if self.cfg.evaluation.evaluate_success:
                avg_success/=n
                metrics["success_rate"]=avg_success

        # Oracle and full value estimation
        alphas = draw_alphas(policy_agent[0].n_anchors,self.cfg.evaluation.steps, self.cfg.evaluation.scale).to(self.cfg.evaluation.device)
        best_alpha_train = self.best_alpha_train[min(task._task_id,len(self.best_alpha_train)-1)].to(self.cfg.evaluation.device)
        best_alpha_train = torch.cat([best_alpha_train,torch.zeros(alphas.shape[-1] - best_alpha_train.shape[-1]).to(self.cfg.evaluation.device)], dim = 0)
        alphas = torch.cat([best_alpha_train.unsqueeze(0),alphas],dim = 0)
        oracle_task = copy.deepcopy(task)
        oracle_task._env_agent_cfg["n_envs"] = alphas.shape[0]
        env_agent = oracle_task.make()
        policy_agent.agents = policy_agent.agents[1:] #deleting alpha agent
        acquisition_agent = TemporalAgent(Agents(env_agent, policy_agent)).to(self.cfg.evaluation.device)
        critic_agent.eval().to(self.cfg.evaluation.device)
        acquisition_agent.seed(self.seed*13+self._stage*100)
        w = Workspace()
        rewards = []
        values = []
        for i in range(self.cfg.evaluation.n_rollouts):
            w = Workspace()
            w.set_full("alphas",torch.stack([alphas for _ in range(1001)],dim=0))
            with torch.no_grad():
                acquisition_agent(w, t = 0, stop_variable = "env/done", action_std=0.)
                critic_agent(w)
            ep_lengths= w["env/done"].max(0)[1]+1
            B=ep_lengths.size()[0]
            arange=torch.arange(B).to(ep_lengths.device)
            cr = w["env/cumulated_reward"][ep_lengths-1,arange]
            rewards.append(cr)
            values.append(w["critic"].mean(0))
        rewards = torch.stack(rewards, dim = 0).mean(0)
        values = torch.stack(values, dim = 0).mean(0)
        image = display_kshot(alphas.cpu(),values.round().cpu())
        logger.add_figure("evaluation/"+str(task.task_id())+"/values_distribution",image,0)
        image = display_kshot(alphas.cpu(),rewards.round().cpu())
        logger.add_figure("evaluation/"+str(task.task_id())+"/reward_distribution",image,0)
        metrics["oracle/avg_reward"] = rewards.max().item()
        metrics["value/avg_reward"] = rewards[values.argmax()].item()
        metrics["best_alpha_train/avg_reward"] = rewards[0].item()
        metrics["midpoint/avg_reward"] = rewards[1].item()
        metrics["last_anchor/avg_reward"] = rewards[-1].item()
        return metrics