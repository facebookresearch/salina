import time
from salina import instantiate_class
import torch
import torch.utils.data
import numpy as np
from salina.agents import Agents,TemporalAgent
from salina.agents.brax import EpisodesDone
from salina import Workspace

# CL Generic Classes
class Task:
    def make(self):
        raise NotImplementedError

    def task_id(self):
        raise NotImplementedError

class Scenario:
    def train_tasks(self):
        raise NotImplementedError

    def test_tasks(self):
        raise NotImplementedError

class Model:
    def train(self,task,*extra_args):
        raise NotImplementedError

    def evaluate(self,test_tasks):
        raise NotImplementedError

# RL Specific Classes

class RLTask(Task):
    def __init__(self,env_agent_cfg,input_dimension,output_dimension,task_id):
        self._env_agent_cfg=env_agent_cfg
        self._task_id=task_id
        self._input_dimension=input_dimension
        self._output_dimension=output_dimension

    def output_dimension(self):
        return self._output_dimension

    def input_dimension(self):
        return self._input_dimension

    def task_id(self):
        return self._task_id

    def make(self):
        return instantiate_class(self._env_agent_cfg)

class RLModel(Model):
    def __init__(self,params):
        self.cfg=params
        self._stage=0

    def train(self,task,logger):
        logger.message("-- Train stage "+str(self._stage))
        output=self._train(task,logger.get_logger("stage_"+str(self._stage)+"/"))
        [logger.add_scalar("monitor_per_stage/"+k,output[k],self._stage) for k in output]
        self._stage+=1

    def _train(self,task,logger):
        raise NotImplementedError

    def get_evaluation_agent(self,task_id):
        raise NotImplementedError

    def evaluate(self,test_tasks,logger):
        logger.message("Starting evaluation...")
        with torch.no_grad():
            evaluation={}
            for k,task in enumerate(test_tasks):
                tid=task.task_id()
                evaluation[tid]={}

                env_agent=task.make()
                policy_agent=self.get_evaluation_agent(task.task_id())
                if not policy_agent is None:
                    policy_agent.eval()
                    no_autoreset=EpisodesDone()
                    acquisition_agent=TemporalAgent(Agents(env_agent,no_autoreset,policy_agent))
                    acquisition_agent.seed(self.cfg.evaluation.seed+self._stage*100+k)
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
                    evaluation[tid]["avg_reward"]=avg_reward

                    if self.cfg.evaluation.evaluate_success:
                        avg_success/=n
                        evaluation[tid]["success_rate"]=avg_success

        logger.message("-- End evaluation...")
        return evaluation
