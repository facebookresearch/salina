#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import salina
from salina import instantiate_class
import torch
import torch.utils.data
from salina.agents import Agents,TemporalAgent
from salina.agents.brax import EpisodesDone
from salina import Workspace

# CRL Generic Classes
class Task:
    """ A Reinforcement Learning task defined as a SaLinA agent
    """
    def __init__(self,env_agent_cfg,input_dimension,output_dimension,task_id,n_interactions=None):
        """ Create a new RL task
        Args:
            env_agent_cfg   : The OmegaConf (or dict) that allows to configure the SaLinA agent
            input_dimension : The input dimension of the observations
            output_dimension: The output dimension of the actions (i.e size of the output tensor, or number of actions if discrete actions)
            task_id         : An identifier of the task
            n_interactions  : [description]. Defaults to None. Number of env interactions allowed for training
        """
        self._env_agent_cfg=env_agent_cfg
        self._task_id=task_id
        self._input_dimension=input_dimension
        self._output_dimension=output_dimension
        self._n_interactions=n_interactions

    def output_dimension(self):
        return self._output_dimension

    def input_dimension(self):
        return self._input_dimension

    def task_id(self):
        return self._task_id

    def make(self)-> salina.Agent:
        """ Return the environment agent corresponding to this task
        Returns:
            salina.Agent: The env agent
        """
        return instantiate_class(self._env_agent_cfg)

    def n_interactions(self):
        return self._n_interactions

class Scenario:
    """ A scenario is a sequence of train tasks and a sequence of test tasks
    """

    def train_tasks(self):
        raise NotImplementedError

    def test_tasks(self):
        raise NotImplementedError



class Model:
    """ A (CRL) Model can be updated over one new task, and evaluated over any task
        Args:
            seed 
            params : The OmegaConf (or dict) that allows to configure the model
    """
    def __init__(self,seed,params):
        self.seed=seed
        self.cfg=params
        self._stage=0

    def memory_size(self) -> dict:
        """ Returns a dict containing different infos about the memory size of the model
        Returns:
            dict: a dict containing different infos about the memory size of the model
        """        
        raise NotImplementedError

    def train(self,task,logger,**extra_args):
        """ Update a model over a particular task
        Args:
            task: The task to train on
            logger
        """
        logger.message("-- Train stage "+str(self._stage))
        output=self._train(task,logger.get_logger("stage_"+str(self._stage)+"/"))
        [logger.add_scalar("monitor_per_stage/"+k,output[k],self._stage) for k in output]
        self._stage+=1

    def evaluate(self,test_tasks,logger):
        """ Evaluate a model over a set of test tasks
        Args:
            test_tasks: The set of tasks to evaluate on
            logger
        Returns:
            evaluation: Some statistics about the evaluation (i.e metrics)
        """
        logger.message("Starting evaluation...")
        with torch.no_grad():
            evaluation={}
            for k,task in enumerate(test_tasks):
                metrics=self._evaluate_single_task(task)
                evaluation[task.task_id()]=metrics
                logger.message("Evaluation over task "+str(k)+":"+str(metrics))

        logger.message("-- End evaluation...")
        return evaluation

    def _train(self,task,logger):
        raise NotImplementedError

    def get_evaluation_agent(self,task_id):
        raise NotImplementedError

    def _evaluate_single_task(self,task):
        metrics={}
        tid=task.task_id()
        env_agent=task.make()
        policy_agent=self.get_evaluation_agent(task.task_id())

        if not policy_agent is None:
            policy_agent.eval()
            no_autoreset=EpisodesDone()
            acquisition_agent=TemporalAgent(Agents(env_agent,no_autoreset,policy_agent))
            acquisition_agent.seed(self.seed*13+self._stage*100)
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
            metrics["avg_reward"]=avg_reward

            if self.cfg.evaluation.evaluate_success:
                avg_success/=n
                metrics["success_rate"]=avg_success
        return metrics


