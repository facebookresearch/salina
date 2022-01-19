from salina_cl.core import RLModel
from salina import instantiate_class
from salina_cl.algorithms.optimizers.ppo import ppo_train
from salina_cl.algorithms.optimizers.tools import weight_init
import time

class FineTune(RLModel):
    def __init__(self,params):
        super().__init__(params)
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
        r,self.ppo_agent,self.critic_agent=ppo_train(self.ppo_agent, self.critic_agent, env_agent,logger, self.cfg.ppo,n_max_interactions=task.n_interactions())

        return r

    def get_evaluation_agent(self,task_id):
        return self.ppo_agent
