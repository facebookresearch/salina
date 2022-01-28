#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

""" Code is greatly inspired by https://github.com/sunblaze-ucb/rl-generalization """


import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from salina_cl.core import Scenario,Task
from salina import instantiate_class
from gym.wrappers import TimeLimit

def make_env(env,max_episode_steps):
    e=instantiate_class(env)
    e=TimeLimit(e,max_episode_steps=max_episode_steps)
    return e

def cartpole_7tasks(n_train_envs,n_evaluation_envs,n_tasks,n_steps,**kwargs):
    classes=[
       ("ContinuousCartPoleEnv",n_steps),
       ("StrongPushCartPole",n_steps),
       ("ShortPoleCartPole",n_steps),
       ("HeavyPoleCartPole",n_steps),
       ("WeakPushCartPole",n_steps),
       ("LongPoleCartPole",n_steps),
       ("LightPoleCartPole",n_steps),
    ]
       
    assert n_tasks<=len(classes)
    classes=classes[:n_tasks]

    return CartPoleScenario(n_train_envs,n_evaluation_envs,200,classes)

class CartPoleScenario(Scenario):
    def __init__(self,n_train_envs,n_evaluation_envs,max_episode_steps,classes=None):
        input_dimension = [4]
        output_dimension = [1]

        self._train_tasks=[]
        for k,c in enumerate(classes):
                agent_cfg={
                        "classname":"salina.agents.gyma.AutoResetGymAgent",
                        "make_env_fn":make_env,
                        "make_env_args":{"env":{"classname":"salina_cl.scenarios.classic_control.cartpole."+c[0]},"max_episode_steps":max_episode_steps},
                        "n_envs":n_train_envs,
                }
                self._train_tasks.append(Task(env_agent_cfg=agent_cfg,input_dimension=input_dimension,output_dimension=output_dimension,task_id=k,n_interactions=c[1]))

        self._test_tasks=[]
        for k,c in enumerate(classes):
            agent_cfg={
                    "classname":"salina.agents.gyma.AutoResetGymAgent",
                    "make_env_fn":make_env,
                    "make_env_args":{"env":{"classname":"salina_cl.scenarios.classic_control.cartpole."+c[0]},"max_episode_steps":max_episode_steps},
                "n_envs":n_evaluation_envs
            }
            self._test_tasks.append(Task(agent_cfg,input_dimension,output_dimension,k))

    def train_tasks(self):
        return self._train_tasks

    def test_tasks(self):
        return self._test_tasks


class ContinuousCartPoleEnv(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along
        a frictionless track. The pendulum starts upright, and the goal is to
        prevent it from falling over by increasing and reducing the cart's
        velocity.

    Source:
        This environment corresponds to the version of the cart-pole problem
        described by Barto, Sutton, and Anderson

    Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       Cart Position             -2.4                    2.4
        1       Cart Velocity             -Inf                    Inf
        2       Pole Angle                -0.209 rad (-12 deg)    0.209 rad (12 deg)
        3       Pole Angular Velocity     -Inf                    Inf

    Actions:
        Type: Discrete(2)
        Num   Action
        0     Push cart to the left
        1     Push cart to the right

        Note: The amount the velocity that is reduced or increased is not
        fixed; it depends on the angle the pole is pointing. This is because
        the center of gravity of the pole increases the amount of energy needed
        to move the cart underneath it

    Reward:
        Reward is 1 for every step taken, including the termination step

    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]

    Episode Termination:
        Pole Angle is more than 12 degrees.
        Cart Position is more than 2.4 (center of the cart reaches the edge of
        the display).
        Episode length is greater than 200.
        Solved Requirements:
        Considered solved when the average return is greater than or equal to
        195.0 over 100 consecutive trials.
    """

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )

        self.action_space = spaces.Box(np.array([-1.0]), np.array([1.0]), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def step(self, action):
        action=np.clip(action,-1.0,1.0)
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag*action[0]
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot ** 2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)

        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state, dtype=np.float32), reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state, dtype=np.float32)

    def seed(self,seed):
        self.np_random,_=seeding.np_random(seed)

    def render(self, mode="human"):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.utils import pyglet_rendering

            self.viewer = pyglet_rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = pyglet_rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = pyglet_rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = (
                -polewidth / 2,
                polewidth / 2,
                polelen - polewidth / 2,
                -polewidth / 2,
            )
            pole = pyglet_rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(0.8, 0.6, 0.4)
            self.poletrans = pyglet_rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = pyglet_rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(0.5, 0.5, 0.8)
            self.viewer.add_geom(self.axle)
            self.track = pyglet_rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole

        if self.state is None:
            return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

def uniform_exclude_inner(np_uniform, a, b, a_i, b_i):
    """Draw sample from uniform distribution, excluding an inner range"""
    if not (a < a_i and b_i < b):
        raise ValueError(
            "Bad range, inner: ({},{}), outer: ({},{})".format(a, b, a_i, b_i))
    while True:
        # Resample until value is in-range
        result = np_uniform(a, b)
        if (a <= result and result < a_i) or (b_i <= result and result < b):
            return result

# Cart pole environment variants.

class ModifiableCartPoleEnv(ContinuousCartPoleEnv):

    RANDOM_LOWER_FORCE_MAG = 5.0
    RANDOM_UPPER_FORCE_MAG = 15.0
    EXTREME_LOWER_FORCE_MAG = 3.5
    EXTREME_UPPER_FORCE_MAG = 40.0

    RANDOM_LOWER_LENGTH = 0.25
    RANDOM_UPPER_LENGTH = 0.75
    EXTREME_LOWER_LENGTH = 0.05
    EXTREME_UPPER_LENGTH = 2.0

    RANDOM_LOWER_MASSPOLE = 0.05
    RANDOM_UPPER_MASSPOLE = 0.5
    EXTREME_LOWER_MASSPOLE = 0.01
    EXTREME_UPPER_MASSPOLE = 2.0

    def _followup(self):
        """Cascade values of new (variable) parameters"""
        self.total_mass = (self.masspole + self.masscart)
        self.polemass_length = (self.masspole * self.length)

    def reset(self, new=True):
        """new is a boolean variable telling whether to regenerate the environment parameters"""
        """Default is to just ignore it"""
        self.nsteps = 0
        return super(ModifiableCartPoleEnv, self).reset()

    @property
    def parameters(self):
        return {'id': self.spec.id, }

    def step(self, *args, **kwargs):
        """Wrapper to increment new variable nsteps"""
        self.nsteps += 1
        return super().step(*args, **kwargs)

    def is_success(self):
         """Returns True is current state indicates success, False otherwise
         Balance for at least 195 time steps ("definition" of success in Gym:
         https://github.com/openai/gym/wiki/CartPole-v0#solved-requirements)
         """
         target = 195
         if self.nsteps >= target:
             #print("[SUCCESS]: nsteps is {}, reached target {}".format(
             #      self.nsteps, target))
             return True
         else:
             #print("[NO SUCCESS]: nsteps is {}, target {}".format(
             #      self.nsteps, target))
             return False


class StrongPushCartPole(ModifiableCartPoleEnv):
    def __init__(self):
        super(StrongPushCartPole, self).__init__()
        self.force_mag = self.EXTREME_UPPER_FORCE_MAG

    @property
    def parameters(self):
        parameters = super(StrongPushCartPole, self).parameters
        parameters.update({'force': self.force_mag, })
        return parameters


class WeakPushCartPole(ModifiableCartPoleEnv):
    def __init__(self):
        super(WeakPushCartPole, self).__init__()
        self.force_mag = self.EXTREME_LOWER_FORCE_MAG

    @property
    def parameters(self):
        parameters = super(WeakPushCartPole, self).parameters
        parameters.update({'force': self.force_mag, })
        return parameters


class RandomStrongPushCartPole(ModifiableCartPoleEnv):
    def __init__(self):
        super(RandomStrongPushCartPole, self).__init__()
        self.force_mag = self.np_random.uniform(self.RANDOM_LOWER_FORCE_MAG, self.RANDOM_UPPER_FORCE_MAG)

    def reset(self, new=True):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        if new:
            self.force_mag = self.np_random.uniform(self.RANDOM_LOWER_FORCE_MAG, self.RANDOM_UPPER_FORCE_MAG)
        return np.array(self.state)

    @property
    def parameters(self):
        parameters = super(RandomStrongPushCartPole, self).parameters
        parameters.update({'force': self.force_mag, })
        return parameters


class RandomWeakPushCartPole(ModifiableCartPoleEnv):
    def __init__(self):
        super(RandomWeakPushCartPole, self).__init__()
        self.force_mag = uniform_exclude_inner(self.np_random.uniform, self.EXTREME_LOWER_FORCE_MAG, self.EXTREME_UPPER_FORCE_MAG,
            self.RANDOM_LOWER_FORCE_MAG, self.RANDOM_UPPER_FORCE_MAG)

    def reset(self, new=True):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        if new:
            self.force_mag = uniform_exclude_inner(self.np_random.uniform, self.EXTREME_LOWER_FORCE_MAG, self.EXTREME_UPPER_FORCE_MAG,
                self.RANDOM_LOWER_FORCE_MAG, self.RANDOM_UPPER_FORCE_MAG)
        return np.array(self.state)

    @property
    def parameters(self):
        parameters = super(RandomWeakPushCartPole, self).parameters
        parameters.update({'force': self.force_mag, })
        return parameters


class ShortPoleCartPole(ModifiableCartPoleEnv):
    def __init__(self):
        super(ShortPoleCartPole, self).__init__()
        self.length = self.EXTREME_LOWER_LENGTH
        self._followup()

    @property
    def parameters(self):
        parameters = super(ShortPoleCartPole, self).parameters
        parameters.update({'length': self.length, })
        return parameters


class LongPoleCartPole(ModifiableCartPoleEnv):
    def __init__(self):
        super(LongPoleCartPole, self).__init__()
        self.length = self.EXTREME_UPPER_LENGTH
        self._followup()

    @property
    def parameters(self):
        parameters = super(LongPoleCartPole, self).parameters
        parameters.update({'length': self.length, })
        return parameters


class RandomLongPoleCartPole(ModifiableCartPoleEnv):
    def __init__(self):
        super(RandomLongPoleCartPole, self).__init__()
        self.length = self.np_random.uniform(self.RANDOM_LOWER_LENGTH, self.RANDOM_UPPER_LENGTH)
        self._followup()

    def reset(self, new=True):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        if new:
            self.length = self.np_random.uniform(self.RANDOM_LOWER_LENGTH, self.RANDOM_UPPER_LENGTH)
            self._followup()
        return np.array(self.state)

    @property
    def parameters(self):
        parameters = super(RandomLongPoleCartPole, self).parameters
        parameters.update({'length': self.length, })
        return parameters


class RandomShortPoleCartPole(ModifiableCartPoleEnv):
    def __init__(self):
        super(RandomShortPoleCartPole, self).__init__()
        self.length = uniform_exclude_inner(self.np_random.uniform, self.EXTREME_LOWER_LENGTH, self.EXTREME_UPPER_LENGTH,
            self.RANDOM_LOWER_LENGTH, self.RANDOM_UPPER_LENGTH)
        self._followup()

    def reset(self, new=True):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        if new:
            self.length = uniform_exclude_inner(self.np_random.uniform, self.EXTREME_LOWER_LENGTH, self.EXTREME_UPPER_LENGTH,
                self.RANDOM_LOWER_LENGTH, self.RANDOM_UPPER_LENGTH)
            self._followup()
        return np.array(self.state)

    @property
    def parameters(self):
        parameters = super(RandomShortPoleCartPole, self).parameters
        parameters.update({'length': self.length, })
        return parameters


class LightPoleCartPole(ModifiableCartPoleEnv):
    def __init__(self):
        super(LightPoleCartPole, self).__init__()
        self.masspole = self.EXTREME_LOWER_MASSPOLE
        self._followup()

    @property
    def parameters(self):
        parameters = super(LightPoleCartPole, self).parameters
        parameters.update({'mass': self.masspole, })
        return parameters


class HeavyPoleCartPole(ModifiableCartPoleEnv):
    def __init__(self):
        super(HeavyPoleCartPole, self).__init__()
        self.masspole = self.EXTREME_UPPER_MASSPOLE
        self._followup()

    @property
    def parameters(self):
        parameters = super(HeavyPoleCartPole, self).parameters
        parameters.update({'mass': self.masspole, })
        return parameters


class RandomHeavyPoleCartPole(ModifiableCartPoleEnv):
    def __init__(self):
        super(RandomHeavyPoleCartPole, self).__init__()
        self.masspole = self.np_random.uniform(self.RANDOM_LOWER_MASSPOLE, self.RANDOM_UPPER_MASSPOLE)
        self._followup()

    def reset(self, new=True):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        if new:
            self.masspole = self.np_random.uniform(self.RANDOM_LOWER_MASSPOLE, self.RANDOM_UPPER_MASSPOLE)
            self._followup()
        return np.array(self.state)

    @property
    def parameters(self):
        parameters = super(RandomHeavyPoleCartPole, self).parameters
        parameters.update({'mass': self.masspole, })
        return parameters


class RandomLightPoleCartPole(ModifiableCartPoleEnv):
    def __init__(self):
        super(RandomLightPoleCartPole, self).__init__()
        self.masspole = uniform_exclude_inner(self.np_random.uniform, self.EXTREME_LOWER_MASSPOLE, self.EXTREME_UPPER_MASSPOLE,
            self.RANDOM_LOWER_MASSPOLE, self.RANDOM_UPPER_MASSPOLE)
        self._followup()

    def reset(self, new=True):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        if new:
            self.masspole = uniform_exclude_inner(self.np_random.uniform, self.EXTREME_LOWER_MASSPOLE, self.EXTREME_UPPER_MASSPOLE,
                self.RANDOM_LOWER_MASSPOLE, self.RANDOM_UPPER_MASSPOLE)
            self._followup()
        return np.array(self.state)

    @property
    def parameters(self):
        parameters = super(RandomLightPoleCartPole, self).parameters
        parameters.update({'mass': self.masspole, })
        return parameters


class RandomNormalCartPole(ModifiableCartPoleEnv):

    def __init__(self):
        super(RandomNormalCartPole, self).__init__()
        self.force_mag = self.np_random.uniform(self.RANDOM_LOWER_FORCE_MAG, self.RANDOM_UPPER_FORCE_MAG)
        self.length = self.np_random.uniform(self.RANDOM_LOWER_LENGTH, self.RANDOM_UPPER_LENGTH)
        self.masspole = self.np_random.uniform(self.RANDOM_LOWER_MASSPOLE, self.RANDOM_UPPER_MASSPOLE)
        self._followup()

    def reset(self, new=True):
        self.nsteps = 0  # for super.is_success()
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        if new:
            self.force_mag = self.np_random.uniform(self.RANDOM_LOWER_FORCE_MAG, self.RANDOM_UPPER_FORCE_MAG)
            self.length = self.np_random.uniform(self.RANDOM_LOWER_LENGTH, self.RANDOM_UPPER_LENGTH)
            self.masspole = self.np_random.uniform(self.RANDOM_LOWER_MASSPOLE, self.RANDOM_UPPER_MASSPOLE)
            self._followup()
        return np.array(self.state)

    @property
    def parameters(self):
        parameters = super(RandomNormalCartPole, self).parameters
        #parameters.update({'force_mag': self.force_mag, 'length': self.length, 'masspole': self.masspole, })
        parameters.update({'force_mag': self.force_mag, 'length': self.length, 'masspole': self.masspole, 'total_mass': self.total_mass, 'polemass_length': self.polemass_length, })
        return parameters


class RandomExtremeCartPole(ModifiableCartPoleEnv):

    def __init__(self):
        super(RandomExtremeCartPole, self).__init__()
        '''
        self.force_mag = self.np_random.uniform(self.LOWER_FORCE_MAG, self.UPPER_FORCE_MAG)
        self.length = self.np_random.uniform(self.LOWER_LENGTH, self.UPPER_LENGTH)
        self.masspole = self.np_random.uniform(self.LOWER_MASSPOLE, self.UPPER_MASSPOLE)
        '''
        self.force_mag = uniform_exclude_inner(self.np_random.uniform,
            self.EXTREME_LOWER_FORCE_MAG, self.EXTREME_UPPER_FORCE_MAG,
            self.RANDOM_LOWER_FORCE_MAG, self.RANDOM_UPPER_FORCE_MAG)
        self.length = uniform_exclude_inner(self.np_random.uniform,
            self.EXTREME_LOWER_LENGTH, self.EXTREME_UPPER_LENGTH,
            self.RANDOM_LOWER_LENGTH, self.RANDOM_UPPER_LENGTH)
        self.masspole = uniform_exclude_inner(self.np_random.uniform,
            self.EXTREME_LOWER_MASSPOLE, self.EXTREME_UPPER_MASSPOLE,
            self.RANDOM_LOWER_MASSPOLE, self.RANDOM_UPPER_MASSPOLE)

        self._followup()
        # NOTE(cpacker): even though we're just changing the above params,
        # we still need to regen the other var dependencies
        # We need to scan through the other methods to make sure the same
        # mistake isn't being made

        #self.gravity = 9.8
        #self.masscart = 1.0
        #self.masspole = 0.1
        #self.total_mass = (self.masspole + self.masscart)
        #self.length = 0.5 # actually half the pole's length
        #self.polemass_length = (self.masspole * self.length)
        #self.force_mag = 10.0
        #self.tau = 0.02  # seconds between state updates

    def reset(self, new=True):
        self.nsteps = 0  # for super.is_success()
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        '''
        self.force_mag = self.np_random.uniform(self.LOWER_FORCE_MAG, self.UPPER_FORCE_MAG)
        self.length = self.np_random.uniform(self.LOWER_LENGTH, self.UPPER_LENGTH)
        self.masspole = self.np_random.uniform(self.LOWER_MASSPOLE, self.UPPER_MASSPOLE)
        '''
        if new:
            self.force_mag = uniform_exclude_inner(self.np_random.uniform,
                self.EXTREME_LOWER_FORCE_MAG, self.EXTREME_UPPER_FORCE_MAG,
                self.RANDOM_LOWER_FORCE_MAG, self.RANDOM_UPPER_FORCE_MAG)
            self.length = uniform_exclude_inner(self.np_random.uniform,
                self.EXTREME_LOWER_LENGTH, self.EXTREME_UPPER_LENGTH,
                self.RANDOM_LOWER_LENGTH, self.RANDOM_UPPER_LENGTH)
            self.masspole = uniform_exclude_inner(self.np_random.uniform,
                self.EXTREME_LOWER_MASSPOLE, self.EXTREME_UPPER_MASSPOLE,
                self.RANDOM_LOWER_MASSPOLE, self.RANDOM_UPPER_MASSPOLE)
            self._followup()
        return np.array(self.state)

    @property
    def parameters(self):
        parameters = super(RandomExtremeCartPole, self).parameters
        parameters.update({'force_mag': self.force_mag, 'length': self.length, 'masspole': self.masspole, 'total_mass': self.total_mass, 'polemass_length': self.polemass_length, })
        return parameters
