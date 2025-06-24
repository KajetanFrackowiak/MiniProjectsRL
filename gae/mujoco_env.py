import gymnasium as gym
import numpy as np
import mujoco
from mujoco import MjModel, MjData, viewer
import os


class MuJoCoCartPoleEnv(gym.Env):
    def __init__(self, model_path="cartpole.xml"):
        full_path = os.path.abspath(model_path)
        self.model = MjModel.from_xml_path(full_path)
        self.data = MjData(self.model)

        self.obs_dim = self.model.nq + self.model.nv
        self.act_dim = self.model.nu
        self.action_space = gym.spaces.Box(low=-10, high=10.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        self.viewer = None
        self.step_count = 0

        self.np_random = None

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
            
        self.data.qpos[:] = 0
        self.data.qvel[:] = 0
        self.step_count = 0
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action):
        self.data.ctrl[:] = np.clip(action, -10.0, 10.0)
        mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()

        x, x_dot, theta, theta_dot = obs
        self.step_count += 1

        done = abs(x) > 2.4 or abs(theta) > 0.2 or self.step_count >= 500
        reward = 1.0 if not done else 0.0
        return obs, reward, done, False, {}

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel])

    def render(self, mode="human"):
        if self.viewer is None:
            self.viewer = viewer.launch(self.model, self.data)
        else:
            pass
