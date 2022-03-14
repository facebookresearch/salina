from brax.envs import wrappers
from brax.envs.ant import Ant
from brax.envs.halfcheetah import Halfcheetah
from google.protobuf import text_format
from brax.envs.halfcheetah import _SYSTEM_CONFIG as halfcheetah_config
from brax.envs.ant import _SYSTEM_CONFIG as ant_config
import brax
from brax import jumpy as jp
import numpy as np

class CustomHalfcheetah(Halfcheetah):
    """Modified Halfcheetah environment (see test_cfgs)"""
    def __init__(self, **kwargs):
        config = text_format.Parse(halfcheetah_config, brax.Config())
        self.obs_mask = jp.concatenate(np.ones((1,23)))
        if "env_specs" in kwargs:
            env_specs = kwargs["env_spec"]
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

class CustomAnt(Ant):
    def __init__(self, **kwargs):
        config = text_format.Parse(ant_config, brax.Config())
        #if "env_spec" in kwargs:
        #    config = modify_cheetah(config,kwargs["env_spec"])
        self.sys = brax.System(config)

__envs__ = {
    'CustomHalfcheetah': CustomHalfcheetah,
    'CustomAnt': CustomAnt
}

def create_brax_env(env_name,
                   seed = 0,
                   batch_size = None,
                   episode_length = 1000,
                   action_repeat = 1,
                   backend = None,
                   auto_reset = True,
                   **kwargs):
    env = __envs__[env_name](**kwargs)
    if episode_length is not None:
        env = wrappers.EpisodeWrapper(env, episode_length, action_repeat)
    if batch_size:
        env = wrappers.VectorWrapper(env, batch_size)
    if auto_reset:
        env = wrappers.AutoResetWrapper(env)
    if batch_size is None:
        return wrappers.GymWrapper(env, seed=seed, backend=backend)
    return wrappers.VectorGymWrapper(env, seed=seed, backend=backend)

test_cfgs = {
    "Normal":{"env_spec":{
      "torso": 1.,
      "thig": 1.,
      "shin": 1.,
      "foot": 1.,
      "gravity": 1.,
      "friction": 1.,
    }},
    "BigFoot":{"env_spec":{
      "torso": 1.,
      "thig": 1.,
      "shin": 1.,
      "foot": 1.25,
      "gravity": 1.,
      "friction": 1.,
    }},
    "SmallFoot":{"env_spec":{
      "torso": 1.,
      "thig": 1.,
      "shin": 1.,
      "foot": 0.75,
      "gravity": 1.,
      "friction": 1.,
    }},
    "BigThig":{"env_spec":{
      "torso": 1.,
      "thig": 1.25,
      "shin": 1.,
      "foot": 1.,
      "gravity": 1.,
      "friction": 1.,
    }},
    "SmallThig":{"env_spec":{
      "torso": 1.,
      "thig": 0.75,
      "shin": 1.,
      "foot": 1.,
      "gravity": 1.,
      "friction": 1.,
    }},
    "BigShin":{"env_spec":{
      "torso": 1.,
      "thig": 1.,
      "shin": 1.25,
      "foot": 1.,
      "gravity": 1.,
      "friction": 1.,
    }},
    "SmallShin":{"env_spec":{
      "torso": 1.,
      "thig": 1.,
      "shin": 0.75,
      "foot": 1.,
      "gravity": 1.,
      "friction": 1.,
    }},
    "BigTorso":{"env_spec":{
      "torso": 1.25,
      "thig": 1.,
      "shin": 1,
      "foot": 1.,
      "gravity": 1.,
      "friction": 1.,
    }},
    "SmallTorso":{"env_spec":{
      "torso": 0.75,
      "thig": 1.,
      "shin": 1.,
      "foot": 1.,
      "gravity": 1.,
      "friction": 1.,
    }},
    "SmallGravity":{"env_spec":{
      "torso": 1.,
      "thig": 1.,
      "shin": 1,
      "foot": 1.,
      "gravity": 0.75,
      "friction": 1.,
    }},
    "BigGravity":{"env_spec":{
      "torso": 1.,
      "thig": 1.,
      "shin": 1,
      "foot": 1.,
      "gravity": 1.25,
      "friction": 1.,
    }},
    "SmallFriction":{"env_spec":{
      "torso": 1.,
      "thig": 1.,
      "shin": 1,
      "foot": 1.,
      "gravity": 1.,
      "friction": 0.75,
    }},
    "BigFriction":{"env_spec":{
      "torso": 1.,
      "thig": 1.,
      "shin": 1,
      "foot": 1.,
      "gravity": 1.,
      "friction": 1.25,
    }},
    "TinyGravity":{"env_spec":{
      "torso": 1.,
      "thig": 1.,
      "shin": 1,
      "foot": 1.,
      "gravity": 0.5,
      "friction": 1.,
    }},
    "HugeGravity":{"env_spec":{
      "torso": 1.,
      "thig": 1.,
      "shin": 1,
      "foot": 1.,
      "gravity": 1.5,
      "friction": 1.,
    }},
    "TinyFriction":{"env_spec":{
      "torso": 1.,
      "thig": 1.,
      "shin": 1,
      "foot": 1.,
      "gravity": 1.,
      "friction": 0.5,
    }},
    "HugeFriction":{"env_spec":{
      "torso": 1.,
      "thig": 1.,
      "shin": 1,
      "foot": 1.,
      "gravity": 1.,
      "friction": 1.5,
    }},
}