#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from salina_cl.core import Model
from salina import instantiate_class
import torch
import os

class TwoSteps(Model):
    """
    A model that is using 2 algorithms. 
    """
    def __init__(self,seed,params):
        super().__init__(seed,params)
        self.algorithm1 = instantiate_class(self.cfg.algorithm1)
        self.algorithm2 = instantiate_class(self.cfg.algorithm2)
        self.policy_agent = None
        self.critic_agent = None
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

        #budget1 = task.n_interactions() * (1 - self.cfg.algorithm2.params.budget) if (task._task_id>0) else task.n_interactions()
        r1, self.policy_agent, self.critic_agent = self.algorithm1.run(self.policy_agent, self.critic_agent, env_agent,logger, self.seed, n_max_interactions = task.n_interactions())
        torch.save(self.critic_agent,os.getcwd()+"/critic_"+str(task._task_id)+".dat")
        #budget2 = task.n_interactions() #- r1["n_interactions"]
        r2, self.policy_agent, self.critic_agent = self.algorithm2.run(self.policy_agent, self.critic_agent, env_agent,logger, self.seed, n_max_interactions = 0)
    
        return {kv1[0]:kv1[1]+kv2[1]  for kv1,kv2 in zip(r1.items(),r2.items())}

    def memory_size(self):
        pytorch_total_params = sum(p.numel() for p in self.policy_agent.parameters())
        return {"n_parameters":pytorch_total_params}

    def get_evaluation_agent(self,task_id):
        self.policy_agent.set_task(task_id)
        return self.policy_agent