#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from brax.envs import wrappers
from salina_examples.rl.subspace_of_policies.envs.halfcheetah import CustomHalfcheetah

__envs__ = {
    'CustomHalfcheetah': CustomHalfcheetah,
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

