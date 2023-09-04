import gym
import numpy as np
from gym import spaces
import mujoco as mj
from mujoco.glfw import glfw
import os
from spinal_controllers import *

max_pos = 0.4
stride = 0.05
class PendulumEnv(gym.Env):
    """Custom Environment that follows gym interface."""
    def __init__(self, control_type = Control_Type.BASELINE):
        super(PendulumEnv, self).__init__()
        self.control_type = control_type
        self.rendering = False
        self.init_mujoco()
        if self.rendering == True:
            self.init_window()

        self.episode_length = 500
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        if self.control_type == Control_Type.NEURON:
            self.action_space = spaces.Box(low=0, high=1.0,shape=(4,), dtype=np.float32)
            self.m_ctrl = np.zeros(4)
            self.m_ctrl_old = np.zeros(4)
            self.loop_counter = 4
        else:
            self.action_space = spaces.Box(low=0, high=1.0,shape=(2,), dtype=np.float32)
            self.m_ctrl = np.zeros(2)
            self.m_ctrl_old = np.zeros(2)
            self.loop_counter = 2
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=-50.0, high=50.0,shape=(1,), dtype=np.float32)

    def step(self, action):
        for i in range(self.loop_counter):
            self.m_ctrl_old[i] = self.m_ctrl[i]
        for i in range(self.loop_counter):
            self.m_ctrl[i] = action[i]

        mj.mj_step(self.model, self.data)

        if self.rendering == True:
            mj.mjv_updateScene(self.model, self.data, self.opt, None, self.cam, mj.mjtCatBit.mjCAT_ALL.value, self.scene)
            mj.mjr_render(self.viewport, self.scene, self.context)
            glfw.swap_buffers(self.window)
            glfw.poll_events()

        ctrl_grad_penalty = -np.linalg.norm(self.m_ctrl - self.m_ctrl_old)
        pos_error_penalty = -np.linalg.norm(0 - self.data.qpos[1])
        reward = ctrl_grad_penalty + pos_error_penalty

        self.ticks += 1
        if self.ticks >= self.episode_length:
            self.done = True

        observation = np.array([0])
        info = {}

        return observation, reward, self.done, info

    def reset(self):
        self.done = False
        self.ticks = 0
        for i in range(self.loop_counter):
            self.m_ctrl_old[i] = 0
            self.m_ctrl[i] = 0

        mj.mj_resetData(self.model, self.data)
        self.data.qvel[0] = 2
        mj.mj_forward(self.model, self.data)

        observation =  np.array([0])
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

        if self.control_type == Control_Type.BASELINE:
            mj.set_mjcb_control(self.my_baseline)
        elif self.control_type == Control_Type.REFLEX:
            mj.set_mjcb_control(self.my_stretch_reflex)
        elif self.control_type == Control_Type.RI:
            mj.set_mjcb_control(self.my_RI)
        elif self.control_type == Control_Type.NEURON:
            mj.set_mjcb_control(self.my_neuron_controller)

    def init_window(self):
        glfw.init()
        self.window = glfw.create_window(1200, 900, "Demo", None, None)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        self.scene = mj.MjvScene(self.model, maxgeom=10000)
        self.context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150.value)

        viewport_width, viewport_height = glfw.get_framebuffer_size(self.window)
        self.viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    def my_baseline(self, model, data):
        action = np.array([self.ctrl0, self.ctrl1])
        baseline_controller(action, data)
        # joint0_controller(model, data)

    def my_RI(self, model, data):
        action = np.array([self.ctrl0, self.ctrl1])
        RI_controller(action, data)
        # joint0_controller(model, data)

    def my_stretch_reflex(self, model, data):
        action = np.array([self.ctrl0, self.ctrl1])
        stretch_reflex_controller(action, data)
        # joint0_controller(model, data)

    def my_neuron_controller(self, model, data):
        action = np.array([self.ctrl0, self.ctrl1, self.ctrl2, self.ctrl3])
        neuron_controller(action, data)
        # joint0_controller(model, data)