from salina_cl.core import RLModel
from salina import instantiate_class
from salina_cl.algorithms.optimizers.sac import sac_train
from salina_cl.algorithms.optimizers.tools import weight_init
import time

class FineTune(RLModel):
    def __init__(self,params):
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
        r,self.sac_agent,self.q1_agent,self.q2_agent,replay_buffer=sac_train(self.q1_agent, self.q2_agent, self.sac_agent, env_agent,logger, self.cfg.sac,n_max_interactions=task.n_interactions())
        return r

    def get_evaluation_agent(self,task_id):
        return self.sac_agent
