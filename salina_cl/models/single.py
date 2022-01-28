#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from salina_cl.core import Model
from salina import instantiate_class
from salina_cl.agents.tools import weight_init
import numpy as np

class FromScratch(Model):
    """
    Learn one policy per task and store it in a dict with the task id as a key.
    """
    def __init__(self,seed,params):
        super().__init__(seed,params)
        self.algorithm = instantiate_class(self.cfg.algorithm)
        self.policy_agent = None
        self.critic_agent = None
        self.policy_agents = {}

    def _create_agent(self,task,logger):
        logger.message("Creating Policy and Critic Agents")
        input_dimension = task.input_dimension()
        output_dimension = task.output_dimension()
        policy_agent_cfg = self.cfg.policy_agent
        policy_agent_cfg.input_dimension=input_dimension
        policy_agent_cfg.output_dimension=output_dimension
        self.policy_agent = instantiate_class(policy_agent_cfg)

        critic_agent_cfg = self.cfg.critic_agent
        critic_agent_cfg.input_dimension=input_dimension
        self.critic_agent = instantiate_class(critic_agent_cfg)

    def _train(self,task,logger):        
        self._create_agent(task,logger)
        self.critic_agent.apply(weight_init)
        self.policy_agent.apply(weight_init)
        env_agent = task.make()
        r,self.policy_agent,self.critic_agent = self.algorithm.run(self.policy_agent, self.critic_agent, env_agent,logger, self.seed,n_max_interactions=task.n_interactions())
        self.policy_agents[task.task_id()] = self.policy_agent
        return r

    def get_evaluation_agent(self,task_id):
        if task_id in self.policy_agents:
            return self.policy_agents[task_id]
        else:
            return None

    def memory_size(self):        
        pytorch_total_params = [sum(p.numel() for p in v.parameters()) for _,v in self.policy_agents.items()]
        return {"n_parameters":np.sum(pytorch_total_params)}

class FineTune(Model):
    """
    Simply Finetune the existing policy each time a new task appears. No policy storage
    """
    def __init__(self,seed,params):
        super().__init__(seed,params)
        self.algorithm = instantiate_class(self.cfg.algorithm)
        self.policy_agent = None
        self.critic_agent = None

    def _create_agent(self,task,logger):
        logger.message("Creating Policy and Critic Agents")
        assert self.policy_agent is None
        input_dimension = task.input_dimension()
        output_dimension = task.output_dimension()
        policy_agent_cfg = self.cfg.policy_agent
        policy_agent_cfg.input_dimension = input_dimension
        policy_agent_cfg.output_dimension = output_dimension
        self.policy_agent = instantiate_class(policy_agent_cfg)

        critic_agent_cfg = self.cfg.critic_agent
        critic_agent_cfg.input_dimension=input_dimension
        self.critic_agent = instantiate_class(critic_agent_cfg)

    def _train(self,task,logger):
        if self.policy_agent is None:
            self._create_agent(task,logger)
        self.critic_agent.apply(weight_init)
        env_agent = task.make()
        r,self.policy_agent,self.critic_agent = self.algorithm.run(self.policy_agent, self.critic_agent, env_agent,logger, self.seed, n_max_interactions=task.n_interactions())
        return r

    def memory_size(self):
        pytorch_total_params = sum(p.numel() for p in self.policy_agent.parameters())
        return {"n_parameters":pytorch_total_params}

    def get_evaluation_agent(self,task_id):
        return self.policy_agent
