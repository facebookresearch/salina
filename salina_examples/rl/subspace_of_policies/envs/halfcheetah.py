#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from brax.envs.halfcheetah import Halfcheetah, _SYSTEM_CONFIG
from google.protobuf import text_format
import brax
from brax import jumpy as jp
import numpy as np

env_cfgs = {
    "Normal":{},
    "BigFoot":{"foot": 1.25},
    "SmallFoot":{"foot": 0.75},
    "BigThig":{"thig": 1.25},
    "SmallThig":{"thig": 0.75},
    "BigShin":{"shin": 1.25},
    "SmallShin":{"shin": 0.75},
    "BigTorso":{"torso": 1.25},
    "SmallTorso":{"torso": 0.75},
    "SmallGravity":{"gravity": 0.75},
    "BigGravity":{"gravity": 1.25},
    "SmallFriction":{"friction": 0.75},
    "BigFriction":{"friction": 1.25},
    "TinyGravity":{"gravity": 0.5},
    "HugeGravity":{"gravity": 1.5},
    "TinyFriction":{"friction": 0.5},
    "HugeFriction":{"friction": 1.5}
}

class CustomHalfcheetah(Halfcheetah):
    """Modified Halfcheetah environment (see test_cfgs)"""
    def __init__(self, env_cfg = "Normal", **kwargs):
        config = text_format.Parse(_SYSTEM_CONFIG, brax.Config())
        env_specs = env_cfgs[env_cfg]
        self.obs_mask = jp.concatenate(np.ones((1,23)))
        for spec,coeff in env_specs.items():
            if spec == "gravity":
                config.gravity.z *= coeff
            elif spec == "friction":
                config.friction *= coeff
            elif spec == "obs_mask":
                zeros = int(coeff*23)
                ones = 23-zeros
                np.random.seed(0)
                self.obs_mask = jp.concatenate(np.random.permutation(([0]*zeros)+([1]*ones)).reshape(1,-1))
            else:
                for body in config.bodies:
                    if spec in body.name:
                        body.mass *= coeff
                        body.colliders[0].capsule.radius *= coeff
        self.sys = brax.System(config)

    def _get_obs(self, qp: brax.QP, info: brax.Info) -> jp.ndarray:
        """Observe halfcheetah body position and velocities.
            obs_mask applied to simulate defective modules"""
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