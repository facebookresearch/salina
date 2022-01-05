

import numpy as np
import gym

from gym import spaces
from gym.utils import seeding
import math

class Navigation2DEnv(gym.Env):
    def __init__(self, task , low=-0.5, high=0.5,sparse=True,goal_size=0.2,goal_distance=None,**extra_args):
        super(Navigation2DEnv, self).__init__()
        self.low = low
        self.high = high
        self.sparse=sparse
        self.goal_size=goal_size
        self.goal_distance=goal_distance

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
            shape=(3,), dtype=np.float32)
        self.action_space = spaces.Box(low=-0.1, high=0.1,
            shape=(2,), dtype=np.float32)

        self._task = task
        np_random, seed = seeding.np_random(task)

        if goal_distance is None:
            self.goal = np_random.uniform(self.low, self.high, size=(2,))
        else:
            angle=np_random.rand()*2*math.pi-math.pi
            self.goal=math.cos(angle)*goal_distance,math.sin(angle)*goal_distance,
            print(self.goal)
        self._state = np.zeros(2, dtype=np.float32)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def on_goal(self):
        x = self._state[0] - self.goal[0]
        y = self._state[1] - self.goal[1]
        distance = np.sqrt(x ** 2 + y ** 2)
        return distance<self.goal_size

    def reset(self, env=True):
        self._state = np.zeros(2, dtype=np.float32)

        og=self.on_goal()
        if og:
            og=1.0
        else:
            og=-1.0
        obs=list(self._state)+[og]
        return obs

    def step(self, action):
        action=action/10.0
        action = np.clip(action, -0.1, 0.1)
        ax,ay=action
        assert ax>=-0.1 and ax<=0.1
        assert ay>=-0.1 and ay<=0.1
        self._state = self._state + action

        x = self._state[0] - self.goal[0]
        y = self._state[1] - self.goal[1]
        distance = np.sqrt(x ** 2 + y ** 2)
        #print(x,y, " -> ",distance)


        if self.sparse:
            if distance < self.goal_size: #(np.abs(x) < 1.) and (np.abs(y) < 1.):
                reward = +1.    # / (distance + 1e-8)
                success = True
            else:
                success = False
                reward = + 0.
            info = {'task': self._task, 'success': float(success)}
        else:
            reward = -distance
            if distance < self.goal_size: #(np.abs(x) < 1.) and (np.abs(y) < 1.):
                success = True
            else:
                success = False
            info = {'task': self._task, 'success': float(success)}

        done = success

        og=self.on_goal()
        if og:
            og=1.0
        else:
            og=-1.0
        obs=list(self._state)+[og]
        return obs, reward, done, info
