#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


from salina_cl.core import RLTask
from salina_cl.core import Scenario
import gym
import metaworld
from salina.agents.gyma import AutoResetGymAgent
import random
from gym.wrappers import TimeLimit

class MetaWorldWrapper(gym.Wrapper):
    def __init__(self,e):
        super().__init__(e)

    def reset(self):
        return {"env_obs":self.env.reset(),"success":0.0}

    def step(self,a):
        o,r,d,i=self.env.step(a)
        o={"env_obs":o,"success":i["success"]}
        return o,r,d,i


def make_mt(name,seed):
    print("Building environment ",name)
    random.seed(seed)
    ml1 = metaworld.ML1(name)
    env = ml1.train_classes[name]()  # Create an environment with task `pick_place`
    task = random.choice(ml1.train_tasks)
    env.set_task(task)
    env=MetaWorldWrapper(env)
    env=TimeLimit(env,max_episode_steps=500)
    return env

class MT10(Scenario):
    def __init__(self,n_train_envs,n_evaluation_envs,seed):
        ml10 = metaworld.MT10()
        train_names=[n for n in ml10.train_classes]
        test_names=[n for n in ml10.train_classes]
        input_dimension = [39]
        output_dimension = [4]
        seed=seed

        self._train_tasks=[]
        self._test_tasks=[]
        for k,n in enumerate(train_names):
            agent_cfg={
                "classname":"salina.agents.gyma.AutoResetGymAgent",
                "make_env_fn":make_mt,
                "make_env_args":{"name":n,"seed":seed},
                "n_envs":n_train_envs
            }
            self._train_tasks.append(RLTask(agent_cfg,input_dimension,output_dimension,k))
            self._test_tasks.append(RLTask(agent_cfg,input_dimension,output_dimension,k))

    def train_tasks(self):
        return self._train_tasks

    def test_tasks(self):
        return self._test_tasks
