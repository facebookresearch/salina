#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import brax
from brax.envs.ant import Ant
from google.protobuf import text_format
from brax.envs.ant import _SYSTEM_CONFIG as ant_config
from brax import jumpy as jp
from brax.envs import env as _env
import numpy as np

class Ant(Ant):
    def __init__(self, env_task, **kwargs):
        config = text_format.Parse(ant_config, brax.Config())
        env_specs = env_tasks[env_task]
        self.obs_mask = jp.concatenate(np.ones((1,27)))
        self.action_mask = jp.concatenate(np.ones((1,8)))
        for spec,coeff in env_specs.items():
            if spec == "gravity":
                config.gravity.z *= coeff
            elif spec == "friction":
                config.friction *= coeff
            elif spec == "mask":
                zeros = int(coeff*27)
                ones = 27-zeros
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
        return jp.concatenate(qpos + qvel) * self.obs_mask

    def step(self, state: _env.State, action: jp.ndarray) -> _env.State:
        """Run one timestep of the environment's dynamics."""
        action = action * self.action_mask
        qp, info = self.sys.step(state.qp, action)
        obs = self._get_obs(qp, info)

        x_before = state.qp.pos[0, 0]
        x_after = qp.pos[0, 0]
        forward_reward = (x_after - x_before) / self.sys.config.dt
        ctrl_cost = .5 * jp.sum(jp.square(action))
        contact_cost = (0.5 * 1e-3 *
                        jp.sum(jp.square(jp.clip(info.contact.vel, -1, 1))))
        survive_reward = jp.float32(1)
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward

        done = jp.where(qp.pos[0, 2] < 0.2, x=jp.float32(1), y=jp.float32(0))
        done = jp.where(qp.pos[0, 2] > 1.0, x=jp.float32(1), y=done)
        state.metrics.update(
            reward_ctrl_cost=ctrl_cost,
            reward_contact_cost=contact_cost,
            reward_forward=forward_reward,
            reward_survive=survive_reward)

        return state.replace(qp=qp, obs=obs, reward=reward, done=done)

env_tasks = {
    ## main insight:  hugefoot : bcp de stabilité + rapidité
    ##                tiny foot : peu stable
    ##                huge torso donne de la stabilité aussi
    ##                idée : faire des scenario style gros pied (stabilité ++ )  puis direct petit pied avec rainfall (perte de stabilité)
    
    
    ###########" TOTAL OF 17 (MATRIX OF 17 X 17) ##################################################################
    ## reward from scratch obviously et sur 10M timesteps
    "normal":{},
    "hugefoot":{"Body 4": 3,"Body 2": 3,"Body 1": 3,"Body 7": 3}, ## reward 4800
    "tinyfoot":{"Body 4": 0.75,"Body 2": 0.75,"Body 1": 0.75,"Body 7": 0.75}, ## reward 3200
    "small_gravity":{'gravity':0.8}, ## reward 3491
    "high_gravity":{'gravity':1.25}, ## reward 
    "moon":{'gravity':0.7}, ## reward 2250
    "hugetorso":{'Torso':1.5},  ## reward  5200 (good spirng = bon effet ressort)
    "rainfall":{'friction':0.375}, ## reward  3521
    "spring":{"$ Torso_Aux 1":5,"$ Torso_Aux 2":5,"$ Torso_Aux 3":5,"$ Torso_Aux 4":5},  ## reward  3500
    "defective_module":{"obs_mask":0.5}, ## reward  1000
    "damping":{"spring_damping": 400}, ## reward 3200 le coefficient de ressort
     
    ## fancy combination
    "hugefoot_moon":{"Body 4": 3,"Body 2": 3,"Body 1": 3,"Body 7": 3,"gravity":0.70}, ## reward 4300
    'hugefoot_rainfall':{"Body 4": 3,"Body 2": 3,"Body 1": 3,"Body 7": 3,'friction':0.375},
     "hugetorso_moon": {'Torso':1.5,"gravity":0.7}, ## reward 3300
    "hugetorso_rainfall": {'Torso':1.5,"friction":0.375}, ## reward 3200,
    'tinyfoot_moon':{"Body 4": 0.75,"Body 2": 0.75,"Body 1": 0.75,"Body 7": 0.75,'gravity':0.7}, ## reward 3000 
    'tinyfoot_rainfall':{"Body 4": 0.75,"Body 2": 0.75,"Body 1": 0.75,"Body 7": 0.75,'friction':0.375}, ## reward 3200 --> on lui savonne la planche lol      
    "spring_moon":{"$ Torso_Aux 1":5,"$ Torso_Aux 2":5,"$ Torso_Aux 3":5,"$ Torso_Aux 4":5, "gravity":0.7}, ## reward 2245
    "spring_rainfall":{"$ Torso_Aux 1":5,"$ Torso_Aux 2":5,"$ Torso_Aux 3":5,"$ Torso_Aux 4":5,'friction':0.375},

    # mixing actions
    "disabled_hard1":{"action_mask":[2,3,4,5,6,7]},
    "disabled_hard2":{"action_mask":[0,1,4,5,6,7]},
    "disabled_hard3":{"action_mask":[0,1,2,3,6,7]},
    "disabled_hard4":{"action_mask":[0,1,2,3,4,5]},
    "disabled_forefeet":{"action_mask":[0,1,2,3]},
    "disabled_backfeet":{"action_mask":[4,5,6,7]},
    "disabled_first_diagonal":{"action_mask":[0,1,4,5]},
    "disabled_second_diagonal":{"action_mask":[2,3,6,7]},
    "inverted_actions":{"action_swap":[0,1,2,3,4,5,6,7]},
    ###################################################################################################################################################################
     
     
     # "defective_module_moon":{"obs_mask":0.5,'gravity':0.7}, ## reward  1000
     
     ## train                  ---> eval
     ## defective_module       --->  defective_module_moon= 1791
     ## defective_module       --->  defective_module_rainfall = 1227
     ## defective_module       --->  damping = -93
     ## defective_module       --->  tinyfoot = -231
     
     
     ## train                  ---> eval
     ## defective_module_moon       --->  defective_module= 2100
     ## defective_module_moon       --->  defective_module_rainfall = 1227
     ## defective_module_moon       --->  damping = -93
     ## defective_module_moon       --->  tinyfoot = -231
}
env_gravity_tasks = {"gravity_"+str(2*x/10):{"gravity":2*x/10} for x in range(1,11)}
env_tasks = dict(**env_tasks,**env_gravity_tasks)