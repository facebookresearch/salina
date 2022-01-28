#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

from salina_cl.core import Task
from salina_cl.core import Scenario
from brax.envs import wrappers
import brax
from brax.envs.halfcheetah import Halfcheetah
from google.protobuf import text_format
from brax.envs.halfcheetah import _SYSTEM_CONFIG as halfcheetah_config

def halfcheetah_debug(n_train_envs,n_evaluation_envs,n_steps,**kwargs):
    return MultiHalfcheetah(n_train_envs,n_evaluation_envs,n_steps,["normal"])

def halfcheetah_hard(n_train_envs,n_evaluation_envs,n_steps,**kwargs):
    return MultiHalfcheetah(n_train_envs,n_evaluation_envs,n_steps,["normal","disproportionate_feet","modified_physics"])

def halfcheetah_gravity(n_train_envs,n_evaluation_envs,n_steps,**kwargs):
    return MultiHalfcheetah(n_train_envs,n_evaluation_envs,n_steps,["gravity_"+str(2*x/10) for x in range(1,11)])

env_cfgs = {
    "normal":{},
    "disproportionate_feet":{
      "torso": 0.75,
      "thigh": 0.75,
      "shin": 0.75,
      "foot": 1.25
      },
    "modified_physics":{
      "gravity": 1.5,
      "friction": 1.25,
      },
}
env_gravity_cfgs = {"gravity_"+str(2*x/10):{"gravity":2*x/10} for x in range(1,11)}
env_cfgs = dict(**env_cfgs,**env_gravity_cfgs)

class CustomHalfcheetah(Halfcheetah):
    def __init__(self, env_cfg, **kwargs):
        config = text_format.Parse(halfcheetah_config, brax.Config())
        env_specs = env_cfgs[env_cfg]
        for spec,coeff in env_specs.items():
            if spec == "gravity":
                config.gravity.z *= coeff
            elif spec == "friction":
                config.friction *= coeff
            else:
                for body in config.bodies:
                    if spec in body.name:
                        body.mass *= coeff
                        body.colliders[0].capsule.radius *= coeff
        self.sys = brax.System(config)


def make_halfcheetah(seed = 0,
                   batch_size = None,
                   max_episode_steps = 1000,
                   action_repeat = 1,
                   backend = None,
                   auto_reset = True,
                   env_cfg = "normal",
                   **kwargs):

    env = CustomHalfcheetah(env_cfg, **kwargs)
    if max_episode_steps is not None:
        env = wrappers.EpisodeWrapper(env, max_episode_steps, action_repeat)
    if batch_size:
        env = wrappers.VectorWrapper(env, batch_size)
    if auto_reset:
        env = wrappers.AutoResetWrapper(env)
    if batch_size is None:
        return wrappers.GymWrapper(env, seed=seed, backend=backend)
    return wrappers.VectorGymWrapper(env, seed=seed, backend=backend)

class MultiHalfcheetah(Scenario):
    def __init__(self,n_train_envs,n_evaluation_envs,n_steps,cfgs):
        print("Scenario is ",cfgs)
        env = make_halfcheetah(10)
        input_dimension = [env.observation_space.shape[0]]
        output_dimension = [env.action_space.shape[0]]

        self._train_tasks=[]
        for k,cfg in enumerate(cfgs):
            agent_cfg={
                "classname":"salina_cl.scenarios.brax.tools.AutoResetBraxAgent",
                "make_env_fn":make_halfcheetah,
                "make_env_args":{
                                "max_episode_steps":1000,
                                 "env_cfg":cfg},
                "n_envs":n_train_envs
            }
            self._train_tasks.append(Task(agent_cfg,input_dimension,output_dimension,k,n_steps))

        self._test_tasks=[]
        for k,cfg in enumerate(cfgs):
            agent_cfg={
                "classname":"salina_cl.scenarios.brax.tools.AutoResetBraxAgent",
                "make_env_fn":make_halfcheetah,
                "make_env_args":{"max_episode_steps":1000,
                                 "env_cfg":cfg},
                "n_envs":n_evaluation_envs
            }
            self._test_tasks.append(Task(agent_cfg,input_dimension,output_dimension,k))

    def train_tasks(self):
        return self._train_tasks

    def test_tasks(self):
        return self._test_tasks
