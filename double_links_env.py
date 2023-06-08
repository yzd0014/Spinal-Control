import gym
import numpy as np
from gym import spaces
import mujoco as mj
from mujoco.glfw import glfw
import os
from double_link_controllers import *
from enum import Enum

class Control_Type(Enum):
    BASELINE = 1
    RI = 2
    REFLEX = 3
    RI_AND_REFLEX = 4
    NEURON = 5

control_typle_dic = {Control_Type.BASELINE: "baseline",
                     Control_Type.RI: "RI",
                     Control_Type.REFLEX: "strech reflex",
                     Control_Type.RI_AND_REFLEX: "RI + stretch refelx",
                     Control_Type.NEURON: "neuron model"}

max_pos = 0.2
max_length = 2
min_length = 1.766
pos_stride = 0.01
length_stride = 0.1
class DoubleLinkEnv(gym.Env):
    """Custom Environment that follows gym interface."""
    def __init__(self, control_type = Control_Type.BASELINE):
        super(DoubleLinkEnv, self).__init__()

        self.control_type = control_type
        self.rendering = True
        self.init_mujoco()
        if self.rendering == True:
            self.init_window()

        self.pos_t = -max_pos
        self.length_t = min_length
        self.target_pos = np.zeros(3)
        self.m_ctrl = np.zeros(4)
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Box(low=0, high=1.0,shape=(4,), dtype=np.float32)
        # Example for using image as input (channel-first; channel-last also works):
        #current endfactor pos
        self.observation_space = spaces.Box(low=-50.0, high=50.0,shape=(5,), dtype=np.float32)

    def step(self, action):
        for i in range(4):
            self.m_ctrl[i] = action[i]

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

        pos_diff_new = np.linalg.norm(self.data.xpos[2] - self.target_pos)
        reward = -pos_diff_new

        self.ticks += 1
        if self.ticks >= 20000:
            self.done = True

        observation = np.concatenate((self.data.xpos[2], np.array([self.data.qpos[0], self.data.qpos[1]])))
        info = {}

        return observation, reward, self.done, info

    def reset(self):
        self.length_t += length_stride
        if self.length_t > max_length:
            self.length_t = min_length
            self.pos_t += pos_stride
        if self.pos_t > max_pos:
            self.pos_t = -max_pos
        print(f"target angle: {self.pos_t}")
        print(f"target length: {self.length_t}")
        self.compute_target_pos()

        self.done = False
        self.ticks = 0
        mj.mj_resetData(self.model, self.data)
        self.data.site_xpos[0] = self.target_pos
        mj.mj_forward(self.model, self.data)

        observation = np.concatenate((self.data.xpos[2], np.array([self.data.qpos[0], self.data.qpos[1]])))
        return observation

    # def render(self):
    #   pass
    def close(self):
        if self.rendering == True:
            glfw.set_window_should_close(self.window, True)
            glfw.terminate()

    def init_mujoco(self):
        xml_path = 'double_links.xml'
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
        elif self.control_type == Control_Type.RI_AND_REFLEX:
            mj.set_mjcb_control(self.my_RI_and_stretch_reflex_controller)
        elif self.control_type == Control_Type.NEURON:
            mj.set_mjcb_control(self.my_neuron_controller)

    def init_window(self):
        glfw.init()
        self.window = glfw.create_window(1200, 900, "Demo", None, None)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        self.scene = mj.MjvScene(self.model, maxgeom=10000)
        self.context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150.value)

    def compute_target_pos(self):
        x = np.cos(-0.5 * np.pi + self.pos_t)
        z = np.sin(-0.5 * np.pi + self.pos_t) + 2.5
        self.target_pos = np.array([x, 0, z])

    def my_baseline(self, model, data):
        baseline_controller(self.m_ctrl, data)

    def my_RI(self, model, data):
        RI_controller(self.m_ctrl, data)

    def my_stretch_reflex(self, model, data):
        pass


    def my_RI_and_stretch_reflex_controller(self, model, data):
        pass

    def my_neuron_controller(self, model, data):
        pass