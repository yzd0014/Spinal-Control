import gym
import numpy as np
from gym import spaces
import mujoco as mj
import os
import random
import time

ctrl0 = 0
ctrl1 = 0
xml_path = 'muscle_control_narrow.xml'

def my_controller(model, data):
    data.ctrl[1] = ctrl0
    data.ctrl[2] = ctrl1

class PendulumEnv(gym.Env):
    """Custom Environment that follows gym interface."""
    def __init__(self):
        super(PendulumEnv, self).__init__()
        self.init_mujoco()
        self.pos_t_candidate = -0.8
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Box(low=0, high=1.0,shape=(2,), dtype=np.float32)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=-50.0, high=50.0,shape=(4,), dtype=np.float32)

    def step(self, action):
        global ctrl0
        global ctrl1
        ctrl0 = action[0]
        ctrl1 = action[1]
        mj.mj_step(self.model, self.data)

        pos_diff_new = np.absolute(self.data.qpos[0] - self.pos_t)
        vel_diff_new = np.absolute(self.data.qvel[0] - self.vel_t)
        reward = 10 * np.exp(-30 * pos_diff_new) * 10 * np.exp(-70 * vel_diff_new)
        if reward > 90:
            self.done = True
        if pos_diff_new >= self.pos_diff:
            self.done = True

        if pos_diff_new < self.pos_diff:
            reward += 5
        self.pos_diff = pos_diff_new

        observation = [self.data.qpos[0], self.data.qvel[0], self.pos_t, self.vel_t]
        observation = np.array(observation, dtype=np.float32)
        info = {}

        return observation, reward, self.done, info

    def reset(self):
        self.pos_t = self.pos_t_candidate
        self.pos_t_candidate += 0.05
        if self.pos_t_candidate > 0.8:
            self.pos_t_candidate = -0.8
        self.vel_t = 0
        self.done = False
        self.pos_diff = np.pi
        self.old_time = time.time()
        mj.mj_resetData(self.model, self.data)
        mj.mj_forward(self.model, self.data)

        observation = [self.data.qpos[0], self.data.qvel[0], self.pos_t, self.vel_t]
        observation = np.array(observation, dtype=np.float32)
        return observation

    # def render(self):
    #     pass
    #
    # def close(self):
    #     pass

    def init_mujoco(self):
        global xml_path
        dirname = os.path.dirname(__file__)
        abspath = os.path.join(dirname + "/" + xml_path)
        xml_path = abspath
        self.model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
        self.data = mj.MjData(self.model)  # MuJoCo data
        cam = mj.MjvCamera()  # Abstract camera
        opt = mj.MjvOption()

        mj.mjv_defaultCamera(cam)
        mj.mjv_defaultOption(opt)

        cam.azimuth = 90
        cam.elevation = -20
        cam.distance = 2
        cam.lookat = np.array([0.0, -1, 2])

        mj.set_mjcb_control(my_controller)
