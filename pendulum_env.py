import gym
import numpy as np
from gym import spaces
import mujoco as mj
from mujoco.glfw import glfw
import os
import random
import time

class PendulumEnv(gym.Env):
    """Custom Environment that follows gym interface."""
    def __init__(self):
        super(PendulumEnv, self).__init__()
        self.rendering = True
        self.init_mujoco()
        if self.rendering == True:
            self.init_window()
        self.pos_t_candidate = -0.8
        self.ctrl0 = 0
        self.ctrl1 = 0
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Box(low=0, high=1.0,shape=(2,), dtype=np.float32)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=-50.0, high=50.0,shape=(4,), dtype=np.float32)

    def step(self, action):
        self.ctrl0 = action[0]
        self.ctrl1 = action[1]

        viewport_width = 0
        viewport_height = 0
        viewport = 0
        if self.rendering == True:
            viewport_width, viewport_height = glfw.get_framebuffer_size(self.window)
            viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

        mj.mj_step(self.model, self.data)

        if self.rendering == True:
            mj.mjv_updateScene(self.model, self.data, self.opt, None, self.cam, mj.mjtCatBit.mjCAT_ALL.value, self.scene)
            mj.mjr_render(viewport, self.scene, self.context)
            glfw.swap_buffers(self.window)
            glfw.poll_events()

        pos_diff_new = np.absolute(self.data.qpos[0] - self.pos_t)
        vel_diff_new = np.absolute(self.data.qvel[0] - self.vel_t)
        self.ticks += 1
        reward = -pos_diff_new
        #reward = 10 * np.exp(-10 * pos_diff_new) * np.exp(-20 * vel_diff_new)
        if self.ticks >= 10000:
            self.done = True
        # reward = reward - self.ticks/5000

        observation = [self.data.qpos[0], self.data.qvel[0], self.pos_t, self.vel_t]
        observation = np.array(observation, dtype=np.float32)
        info = {}

        return observation, reward, self.done, info

    def reset(self):
        #self.pos_t = self.pos_t_candidate
        self.pos_t = 0.4
        self.pos_t_candidate += 0.05
        if self.pos_t_candidate > 0.8:
            self.pos_t_candidate = -0.8
        self.vel_t = 0
        self.done = False
        self.ticks = 0
        mj.mj_resetData(self.model, self.data)
        mj.mj_forward(self.model, self.data)

        observation = [self.data.qpos[0], self.data.qvel[0], self.pos_t, self.vel_t]
        observation = np.array(observation, dtype=np.float32)
        return observation

    # def render(self):
    #   pass
    def close(self):
        if self.rendering == True:
            glfw.set_window_should_close(self.window, True)
            glfw.terminate()

    def init_mujoco(self):
        xml_path = 'muscle_control_narrow.xml'
        dirname = os.path.dirname(__file__)
        abspath = os.path.join(dirname + "/" + xml_path)
        xml_path = abspath
        self.model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
        self.data = mj.MjData(self.model)  # MuJoCo data
        self.cam = mj.MjvCamera()  # Abstract camera
        self.opt = mj.MjvOption()

        mj.mjv_defaultCamera(self.cam)
        mj.mjv_defaultOption(self.opt)

        self.cam.azimuth = 90
        self.cam.elevation = -20
        self.cam.distance = 2
        self.cam.lookat = np.array([0.0, -1, 2])

        mj.set_mjcb_control(self.spinal_controller)

    def init_window(self):
        glfw.init()
        self.window = glfw.create_window(1200, 900, "Demo", None, None)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        self.scene = mj.MjvScene(self.model, maxgeom=10000)
        self.context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150.value)

    def my_controller(self, model, data):
        data.ctrl[1] = self.ctrl0
        data.ctrl[2] = self.ctrl1

    def spinal_controller(self, model, data):
        r_spindle = data.actuator_length[1] + 0.05 * data.actuator_velocity[1]
        l_spindle = data.actuator_length[2] + 0.05 * data.actuator_velocity[2]
        data.ctrl[1] = r_spindle - self.ctrl0
        data.ctrl[2] = l_spindle - self.ctrl1