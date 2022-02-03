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
from brax import jumpy as jp
import numpy as np
from brax.envs import env

def halfcheetah_debug(n_train_envs,n_evaluation_envs,n_steps,**kwargs):
    return MultiHalfcheetah(n_train_envs,n_evaluation_envs,n_steps,["normal","disproportionate_feet","modified_physics"])

def halfcheetah_simple(n_train_envs,n_evaluation_envs,n_steps,**kwargs):
    return MultiHalfcheetah(n_train_envs,n_evaluation_envs,n_steps,["tinyfoot","hugetorso","tinygravity","hugethigh","tinyfriction","hugefoot","tinyshin","hugegravity","tinytorso","hugefriction","tinythigh","hugeshin"])

def halfcheetah_hard(n_train_envs,n_evaluation_envs,n_steps,**kwargs):
    return MultiHalfcheetah(n_train_envs,n_evaluation_envs,n_steps,["normal","rainfall","defective_module","overweight","moon"])

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
     "tinyfoot":{"foot":0.5},
     "hugefoot":{"foot":1.5},
     "tinythigh":{"thigh":0.5},
     "hugethigh":{"thigh":1.5},
     "tinyshin":{"shin":0.5},
     "hugeshin":{"shin":1.5},
     "tinytorso":{"torso":0.5},
     "hugetorso":{"torso":1.5},
     "tinygravity":{"gravity":0.5},
     "hugegravity":{"gravity":1.5},
     "tinyfriction":{"friction":0.5},
     "hugefriction":{"friction":1.5},
     "rainfall":{"friction":0.4},
     "moon":{"friction":0.15,
             "gravity":0.15},
     "overweight":{
      "torso": 1.5,
      "thigh": 1.5,
      "shin": 1.5,
      "foot": 1.5
      },
      "defective_module":{"mask":0.5}
}
env_gravity_cfgs = {"gravity_"+str(2*x/10):{"gravity":2*x/10} for x in range(1,11)}
env_cfgs = dict(**env_cfgs,**env_gravity_cfgs)

class CustomHalfcheetah(Halfcheetah):
    def __init__(self, env_cfg, **kwargs):
        config = text_format.Parse(halfcheetah_config, brax.Config())
        env_specs = env_cfgs[env_cfg]
        self.mask = jp.concatenate(np.ones((1,23)))
        for spec,coeff in env_specs.items():
            if spec == "gravity":
                config.gravity.z *= coeff
            elif spec == "friction":
                config.friction *= coeff
            elif spec == "mask":
                zeros = int(coeff*23)
                ones = 23-zeros
                np.random.seed(0)
                self.mask = jp.concatenate(np.random.permutation(([0]*zeros)+([1]*ones)).reshape(1,-1))
            else:
                for body in config.bodies:
                    if spec in body.name:
                        body.mass *= coeff
                        body.colliders[0].capsule.radius *= coeff
        self.sys = brax.System(config)

    def _get_obs(self, qp: brax.QP, info: brax.Info) -> jp.ndarray:
        """Observe halfcheetah body position and velocities."""
        # some pre-processing to pull joint angles and velocities
        (joint_angle,), (joint_vel,) = self.sys.joints[0].angle_vel(qp)

        # qpos:
        # Z of the torso (1,)
        # orientation of the torso as quaternion (4,)
        # joint angles (8,)
        qpos = [qp.pos[0, 2:], qp.rot[0], joint_angle]

        # qvel:
        # velcotiy of the torso (3,)
        # angular velocity of the torso (3,)
        # joint angle velocities (8,)
        qvel = [qp.vel[0], qp.ang[0], joint_vel]
        #print(jp.concatenate(qpos + qvel))
        #print(self.mask)
        #print(jp.concatenate(qpos + qvel) * self.mask)
        return jp.concatenate(qpos + qvel) * self.mask


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
