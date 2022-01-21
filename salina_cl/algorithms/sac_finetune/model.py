#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


from salina_cl.core import RLModel
from salina import instantiate_class
from salina_cl.algorithms.sac_finetune.sac import sac_train
from salina_cl.algorithms.tools import weight_init
import time

class SACFineTune(RLModel):
    def __init__(self,seed,params):
        super().__init__(params)
        self.sac_agent=None
        self.q_agent=None

    def _create_agent(self,task,logger):
        logger.message("Creating SAC and CriticAgent")
        assert self.sac_agent is None
        input_dimension=task.input_dimension()
        output_dimension=task.output_dimension()
        sac_agent_cfg=self.cfg.sac_agent
        sac_agent_cfg.input_dimension=input_dimension
        sac_agent_cfg.output_dimension=output_dimension
        self.sac_agent=instantiate_class(sac_agent_cfg)
        q_agent_cfg=self.cfg.q_agent
        q_agent_cfg.input_dimension=input_dimension
        q_agent_cfg.output_dimension=output_dimension
        self.q1_agent=instantiate_class(q_agent_cfg)
        self.q2_agent=instantiate_class(q_agent_cfg)

    def _train(self,task,logger):
        if self.sac_agent is None:
            self._create_agent(task,logger)
        self.q1_agent.apply(weight_init)
        self.q2_agent.apply(weight_init)
        env_agent=task.make()
        control_env_agent=task.make()
        r,self.sac_agent,self.q1_agent,self.q2_agent,replay_buffer=sac_train(self.q1_agent, self.q2_agent, self.sac_agent, env_agent,logger, self.cfg.sac,self.seed,n_max_interactions=task.n_interactions())
        return r

    def get_evaluation_agent(self,task_id):
        return self.sac_agent

    def memory_size(self):
        pytorch_total_params = sum(p.numel() for p in self.sac_agent.parameters())
        return {"n_parameters":pytorch_total_params}