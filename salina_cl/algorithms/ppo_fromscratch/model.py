#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


from salina_cl.core import RLModel
from salina import instantiate_class
from salina_cl.algorithms.tools import weight_init
from salina_cl.algorithms.ppo_finetune.ppo import ppo_train
import time
import numpy as np

class PPOFromScratch(RLModel):
    def __init__(self,seed,params):
        super().__init__(seed,params)
        self.ppo_agent = None
        self.critic_agent = None
        self.ppo_agents = {}

    def _create_agent(self,task,logger):
        input_dimension=task.input_dimension()
        output_dimension=task.output_dimension()
        ppo_agent_cfg=self.cfg.ppo_agent
        ppo_agent_cfg.input_dimension=input_dimension
        ppo_agent_cfg.output_dimension=output_dimension
        self.ppo_agent=instantiate_class(ppo_agent_cfg)

        critic_agent_cfg=self.cfg.critic_agent
        critic_agent_cfg.input_dimension=input_dimension
        self.critic_agent=instantiate_class(critic_agent_cfg)

    def _train(self,task,logger):        
        self._create_agent(task,logger)
        self.critic_agent.apply(weight_init)
        self.ppo_agent.apply(weight_init)
        env_agent=task.make()
        r,self.ppo_agent,self.critic_agent=ppo_train(self.ppo_agent, self.critic_agent, env_agent,logger, self.cfg.ppo,self.seed,n_max_interactions=task.n_interactions())
        self.ppo_agents[task.task_id()]=self.ppo_agent
        return r

    def get_evaluation_agent(self,task_id):
        if task_id in self.ppo_agents:
            return self.ppo_agents[task_id]
        else:
            return None

    def memory_size(self):        
        pytorch_total_params = [sum(p.numel() for p in v.parameters()) for _,v in self.ppo_agents.items()]
        return {"n_parameters":np.sum(pytorch_total_params)}        
