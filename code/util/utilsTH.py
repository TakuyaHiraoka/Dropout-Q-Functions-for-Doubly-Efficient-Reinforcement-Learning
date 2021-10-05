from gym import Env

import inspect
import sys
import math

class ProxyEnv(Env):
    """
    From openai/rllab. We don't take a dependency on rllab itself because it is
    being deprecated and additionally it is difficult to install!!
    This wraps an OpenAI environment generically
    """
    def __init__(self, wrapped_env):
        self._wrapped_env = wrapped_env

    @property
    def spec(self):
        return self._wrapped_env.spec

    @property
    def wrapped_env(self):
        return self._wrapped_env

    def reset(self, **kwargs):
        return self._wrapped_env.reset(**kwargs)

    @property
    def action_space(self):
        return self._wrapped_env.action_space

    @property
    def observation_space(self):
        return self._wrapped_env.observation_space

    def step(self, action):
        return self._wrapped_env.step(action)

    def render(self, *args, **kwargs):
        return self._wrapped_env.render(*args, **kwargs)

    def log_diagnostics(self, paths, *args, **kwargs):
        self._wrapped_env.log_diagnostics(paths, *args, **kwargs)

    @property
    def horizon(self):
        return self._wrapped_env.horizon

    def terminate(self):
        self._wrapped_env.terminate()

    def get_param_values(self):
        return self._wrapped_env.get_param_values()

    def set_param_values(self, params):
        self._wrapped_env.set_param_values(params)





import numpy as np

from gym import spaces
from gym.spaces.box import Box


class SparseRewardEnv(ProxyEnv):
    def __init__(self, env, rew_thresh):
        self.rew_thresh = rew_thresh
        self.initial_x = None # 0.0
        ProxyEnv.__init__(self, env)


    def reset(self):
        obs = self.wrapped_env.reset()
        self.initial_x = self.wrapped_env.env.sim.data.qpos[0]
        return obs


    def step(self, action):
        next_obs, reward, done, info = self.wrapped_env.step(action)

        current_x_pos = self.wrapped_env.env.sim.data.qpos[0]
        if ( (current_x_pos - self.initial_x) >= self.rew_thresh):
            reward = 1.0 + 0.0 * reward # use previous reward value to keep resulting value type as float 64
            # unlike PolyRL paper setting, base x position is updated like Ant nav.
            # self.initial_x = current_x_pos
            # step reward 20210624
            reward = 0.0 * reward + math.floor((math.fabs(current_x_pos - self.initial_x ))/ self.rew_thresh)
        else:
            reward = 0.0 + 0.0 * reward # ditto

        return next_obs, reward, done, info

