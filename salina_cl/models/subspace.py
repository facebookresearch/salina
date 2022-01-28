#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from salina_cl.core import Model
from salina import instantiate_class
from salina_cl.agents.tools import weight_init

class Incremental(Model):
    """
    Add one anchor per task. 
    """
    def __init__(self,seed,params):
        super().__init__(seed,params)
        self.algorithm = instantiate_class(self.cfg.algorithm)
        self.policy_agent=None
        self.critic_agent=None

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
        critic_agent_cfg.n_anchors = self.policy_agent.n_anchors()
        self.critic_agent = instantiate_class(critic_agent_cfg)
        self.critic_agent.apply(weight_init)

    def _train(self,task,logger):
        if self.policy_agent is None:
            self._create_policy_agent(task,logger)
        else:
            logger.message("Adding an anchor to the policy subspace")
            self.policy_agent.add_anchor()
        self._create_critic_agent(task,logger)
        env_agent = task.make()
        self.policy_agent.set_task_id(task.task_id())
        r,self.policy_agent,self.critic_agent = self.algorithm.run(self.policy_agent, self.critic_agent, env_agent,logger, self.seed,n_max_interactions=task.n_interactions())
        return r

    def memory_size(self):
        pytorch_total_params = sum(p.numel() for p in self.policy_agent.parameters())
        return {"n_parameters":pytorch_total_params}

    def get_evaluation_agent(self,task_id):
        self.policy_agent.set_task_id(task_id)
        return self.policy_agent