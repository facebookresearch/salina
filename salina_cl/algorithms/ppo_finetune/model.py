#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from salina_cl.core import RLModel
from salina import instantiate_class
from salina_cl.algorithms.ppo_finetune.ppo import ppo_train
from salina_cl.algorithms.tools import weight_init
import time

class PPOFineTune(RLModel):
    def __init__(self,seed,params):
        super().__init__(seed,params)
        self.ppo_agent=None
        self.critic_agent=None

    def _create_agent(self,task,logger):
        logger.message("Creating PPO and CriticAgent")
        assert self.ppo_agent is None
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
        if self.ppo_agent is None:
            self._create_agent(task,logger)
        self.critic_agent.apply(weight_init)
        env_agent=task.make()
        r,self.ppo_agent,self.critic_agent=ppo_train(self.ppo_agent, self.critic_agent, env_agent,logger, self.cfg.ppo,self.seed,n_max_interactions=task.n_interactions())

        return r

    def memory_size(self):
        pytorch_total_params = sum(p.numel() for p in self.ppo_agent.parameters())
        return {"n_parameters":pytorch_total_params}

    def get_evaluation_agent(self,task_id):
        return self.ppo_agent
