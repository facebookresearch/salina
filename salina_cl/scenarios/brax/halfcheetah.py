#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import brax
from brax.envs.halfcheetah import Halfcheetah
from google.protobuf import text_format
from brax.envs.halfcheetah import _SYSTEM_CONFIG as halfcheetah_config
from brax import jumpy as jp
import numpy as np
from brax.envs import env as _env

class Halfcheetah(Halfcheetah):
    def __init__(self, env_task: str, **kwargs):
        config = text_format.Parse(halfcheetah_config, brax.Config())
        env_specs = env_tasks[env_task]
        self.obs_mask = jp.concatenate(np.ones((1,23)))
        self.action_mask = jp.concatenate(np.ones((1,6)))
        for spec,coeff in env_specs.items():
            if spec == "gravity":
                config.gravity.z *= coeff
            elif spec == "friction":
                config.friction *= coeff
            elif spec == "obs_mask":
                zeros = int(coeff * 23)
                ones = 23 - zeros
                np.random.seed(0)
                self.obs_mask = jp.concatenate(np.random.permutation(([0]*zeros)+([1]*ones)).reshape(1,-1))
            elif spec == "action_mask":
                self.action_mask[coeff] = 0.
            elif spec == "action_swap":
                self.action_mask[coeff] = -1.
            else:
                for body in config.bodies:
                    if spec in body.name:
                        body.mass *= coeff
                        body.colliders[0].capsule.radius *= coeff
        self.sys = brax.System(config)

    def reset(self, rng: jp.ndarray) -> _env.State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jp.random_split(rng, 3)
        qpos = self.sys.default_angle() + jp.random_uniform(
            rng1, (self.sys.num_joint_dof,), -.1, .1)
        qvel = jp.random_uniform(rng2, (self.sys.num_joint_dof,), -.1, .1)
        qp = self.sys.default_qp(joint_angle=qpos, joint_velocity=qvel)
        info = self.sys.info(qp)
        obs = self._get_obs(qp, info)
        reward, done, zero = jp.zeros(3)
        metrics = {
            'reward_ctrl_cost': zero,
            'reward_forward': zero,
        }
        return _env.State(qp, obs, reward, done, metrics)

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
        #print(self.obs_mask)
        #print(jp.concatenate(qpos + qvel) * self.obs_mask)
        return jp.concatenate(qpos + qvel) * self.obs_mask

    def step(self, state: _env.State, action: jp.ndarray) -> _env.State:
      """Run one timestep of the environment's dynamics."""
      action = action * self.action_mask
      qp, info = self.sys.step(state.qp, action)
      obs = self._get_obs(qp, info)
      x_before = state.qp.pos[0, 0]
      x_after = qp.pos[0, 0]
      forward_reward = (x_after - x_before) / self.sys.config.dt
      ctrl_cost = -.1 * jp.sum(jp.square(action))
      reward = forward_reward + ctrl_cost
      state.metrics.update(
          reward_ctrl_cost=ctrl_cost, reward_forward=forward_reward)
      return state.replace(qp=qp, obs=obs, reward=reward)

env_tasks = {
    "normal":{},
    "jumpcheetah":{},
    
    # Morphological changes
    "disproportionate_feet":{"torso": 0.75,"thigh": 0.75,"shin": 0.75,"foot": 1.25},
    "tinythigh":{"thigh":0.5},
    "hugethigh":{"thigh":1.5},
    "tinyshin":{"shin":0.5},
    "hugeshin":{"shin":1.5},
    "tinytorso":{"torso":0.5},
    "hugetorso":{"torso":1.5},
    "overweight":{"torso": 1.5,"thigh": 1.5,"shin": 1.5,"foot": 1.5},
    "underweight":{"torso": 0.75,"thigh": 0.75,"shin": 0.75,"foot": 0.75},
    "carry_stuff":{"torso": 4.,"thigh": 1.,"shin": 1.,"foot": 1.},
    "defective_module":{"obs_mask":0.5},
    "tinyfoot":{"foot":0.5},
    "hugefoot":{"foot":1.5},
      
    # Environment changes
    "modified_physics":{"gravity": 1.5,"friction": 1.25},
    "tinygravity":{"gravity":0.5},
    "hugegravity":{"gravity":1.5},
    "tinyfriction":{"friction":0.5},
    "hugefriction":{"friction":1.5},
    "rainfall":{"friction":0.4},
    "moon":{"gravity":0.15},
     
    # Combinations
    "tinyfoot_moon": {'foot': 0.5, 'gravity': 0.15},
    "hugefoot_moon": {'foot': 1.5, 'gravity': 0.15},
    "tinyfoot_rainfall": {'foot': 0.5, 'friction': 0.4},
    "hugefoot_rainfall": {'foot': 1.5, 'friction': 0.4},
    "tinyfoot_hugegravity'": {'foot': 0.5, 'gravity': 1.5},
    "hugefoot_hugegravity": {'foot': 1.5, 'gravity': 1.5},
    "carry_stuff_moon": {'torso': 4.0,'thigh': 1.0,'shin': 1.0,'foot': 1.0,'gravity': 0.15},
    'carry_stuff_rainfall': {'torso': 4.0,'thigh': 1.0,'shin': 1.0,'foot': 1.0,'friction': 0.4},
    'carry_stuff_hugegravity': {'torso': 4.0,'thigh': 1.0,'shin': 1.0,'foot': 1.0,'gravity': 1.5},
    "defective_module_moon":{"obs_mask":0.5,'gravity': 0.15},
    "defective_module_rainfall":{"obs_mask":0.5,"friction":0.4},
    "crippled_backlegs":{"action_mask":[0,1,2]},
    "crippled_forelegs":{"action_mask":[3,4,5]},
    "inverted_actions":{"action_swap":[0,1,2,3,4,5]},
}
env_gravity_tasks = {"gravity_"+str(2*x/10):{"gravity":2*x/10} for x in range(1,11)}
env_tasks = dict(**env_tasks,**env_gravity_tasks)