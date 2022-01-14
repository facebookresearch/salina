from salina_cl.core import RLTask
from salina_cl.core import Scenario
import numpy
import torch
import torch.utils.data
import torchvision.datasets
import gym
from salina.agents.gyma import AutoResetGymAgent
from gym.wrappers import TimeLimit

def make_pendulum(max_episode_steps):
    e=gym.make("Pendulum-v1")
    e=TimeLimit(e,max_episode_steps=max_episode_steps)
    return e

class SimplePendulum(Scenario):
    def __init__(self,n_train_tasks,n_train_envs,n_evaluation_envs,max_episode_steps):
        env = make_pendulum(10)
        input_dimension = [env.observation_space.shape[0]]
        output_dimension = [env.action_space.shape[0]]

        self._train_tasks=[]
        for k in range(n_train_tasks):
            agent_cfg={
                "classname":"salina.agents.gyma.AutoResetGymAgent",
                "make_env_fn":make_pendulum,
                "make_env_args":{"max_episode_steps":max_episode_steps},
                "n_envs":n_train_envs
            }
            self._train_tasks.append(RLTask(agent_cfg,input_dimension,output_dimension,k))

        self._test_tasks=[]
        for k in range(n_train_tasks):
            agent_cfg={
                "classname":"salina.agents.gyma.AutoResetGymAgent",
                "make_env_fn":make_pendulum,
                "make_env_args":{"max_episode_steps":max_episode_steps},
                "n_envs":n_evaluation_envs
            }
            self._test_tasks.append(RLTask(agent_cfg,input_dimension,output_dimension,k))

    def train_tasks(self):
        return self._train_tasks

    def test_tasks(self):
        return self._test_tasks
