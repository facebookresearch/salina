#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from salina_cl.core import Task
from salina_cl.core import Scenario
import gym
import metaworld
import numpy as np
from salina.agents.gyma import AutoResetGymAgent
import random
from gym.wrappers import TimeLimit
from continualworld.utils.wrappers import OneHotAdder, RandomizationWrapper, SuccessCounter
from typing import Any, Dict, List, Tuple, Union
TASK_SEQS = {
    "CW2": [
        "drawer-close-v1",
        "push-wall-v1",
    ],
    "CW3": [
        "push-v1",
        "window-close-v1",
        "hammer-v1"
    ],
    "CW10": [
        "hammer-v1",
        "push-wall-v1",
        "faucet-close-v1",
        "push-back-v1",
        "stick-pull-v1",
        "handle-press-side-v1",
        "push-v1",
        "shelf-place-v1",
        "window-close-v1",
        "peg-unplug-side-v1",
    ],
}
TASK_SEQS["CW20"] = TASK_SEQS["CW10"] + TASK_SEQS["CW10"]
META_WORLD_TIME_HORIZON = 200
def get_mt50() -> metaworld.MT50:
    saved_random_state = np.random.get_state()
    np.random.seed(1)
    MT50 = metaworld.MT50()
    np.random.set_state(saved_random_state)
    return MT50

MT50 = get_mt50()
def get_subtasks(name: str) -> List[metaworld.Task]:
    return [s for s in MT50.train_tasks if s.env_name == name]

class MetaWorldWrapper(gym.Wrapper):
    def __init__(self,e):
        super().__init__(e)
        # self.env=e

    def reset(self):
        return {"env_obs":self.env.reset(),"success":0.0}

    def step(self,a):
        o,r,d,i=self.env.step(a)
        o={"env_obs":o,"success":i["success"],"goalDist":i["goalDist"]}
        return o,r,d,i


def make_mt(name,seed,randomization="random_init_all"):
    print("Building environment ",name)
    random.seed(seed)
    # ml1 = metaworld.ML1(name)
    # env = metaworld.MT50() #.train_classes[name]()  # Create an environment with task `pick_place`
    # task = random.choice(ml1.train_tasks)
    
    env = metaworld.MT50().train_classes[name]()
    env = RandomizationWrapper(env, get_subtasks(name), randomization)
    
    
    # env.set_task(name)
    env=MetaWorldWrapper(env)
    env=TimeLimit(env,max_episode_steps=META_WORLD_TIME_HORIZON)
    # import ipdb;ipdb.set_trace()
    return env

# env = MT50.train_classes[task_name]()
# env = RandomizationWrapper(env, get_subtasks(task_name), randomization)
# env = OneHotAdder(env, one_hot_idx=i, one_hot_len=num_tasks)
# env.name = task_name
# env = TimeLimit(env, META_WORLD_TIME_HORIZON)
# env = SuccessCounter(env)
# envs.append(env)


class CW_scenarios(Scenario):
    def __init__(self,name,n_train_envs,n_evaluation_envs,n_steps,seed=0):
        # ml10 = metaworld.MT10()
        # MT50 = metaworld.MT50()
        task_list=TASK_SEQS[name]
        train_names=[n for n in task_list]
        
        
        
        test_names=[n for n in TASK_SEQS["CW10"]]
        
        input_dimension = [12]
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
            self._train_tasks.append(Task(agent_cfg,input_dimension,output_dimension,k,n_steps))
            
        for k,n in enumerate(test_names):
           
            agent_cfg={
                "classname":"salina.agents.gyma.AutoResetGymAgent",
                "make_env_fn":make_mt,
                "make_env_args":{"name":n,"seed":seed},
                "n_envs":n_train_envs
            }
            self._test_tasks.append(Task(agent_cfg,input_dimension,output_dimension,k,n_steps))
        # import ipdb;ipdb.set_trace()

    def train_tasks(self):
        return self._train_tasks

    def test_tasks(self):
        return self._test_tasks