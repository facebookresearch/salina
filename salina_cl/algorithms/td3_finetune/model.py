from salina_cl.core import RLModel
from salina import instantiate_class
from salina_cl.algorithms.optimizers.td3 import td3_train
from salina_cl.algorithms.optimizers.tools import weight_init
import time

class FineTune(RLModel):
    def __init__(self,params):
        super().__init__(params)
        self.td3_agent=None
        self.q_agent=None

    def _create_agent(self,task,logger):
        logger.message("Creating td3 and CriticAgent")
        assert self.td3_agent is None
        input_dimension=task.input_dimension()
        output_dimension=task.output_dimension()
        td3_agent_cfg=self.cfg.td3_agent
        td3_agent_cfg.input_dimension=input_dimension
        td3_agent_cfg.output_dimension=output_dimension
        self.td3_agent=instantiate_class(td3_agent_cfg)
        q_agent_cfg=self.cfg.q_agent
        q_agent_cfg.input_dimension=input_dimension
        q_agent_cfg.output_dimension=output_dimension
        self.q1_agent=instantiate_class(q_agent_cfg)
        self.q2_agent=instantiate_class(q_agent_cfg)

    def _train(self,task,logger):
        if self.td3_agent is None:
            self._create_agent(task,logger)
        self.q1_agent.apply(weight_init)
        self.q2_agent.apply(weight_init)
        env_agent=task.make()
        control_env_agent=task.make()
        r,self.td3_agent,self.q1_agent,self.q2_agent,replay_buffer=td3_train(self.q1_agent, self.q2_agent, self.td3_agent, env_agent,logger, self.cfg.td3,n_max_interactions=task.n_interactions(),control_env_agent=control_env_agent)
        return r

    def get_evaluation_agent(self,task_id):
        return self.td3_agent
