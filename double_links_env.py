import gym
import numpy as np
from gym import spaces
import mujoco as mj
from mujoco.glfw import glfw
import os
from double_link_controllers import *

class DoubleLinkEnv(gym.Env):
    """Custom Environment that follows gym interface."""
    def __init__(self, control_type = Control_Type.BASELINE):
        super(DoubleLinkEnv, self).__init__()
        global num_of_targets

        self.control_type = control_type
        self.rendering = False
        self.init_mujoco()
        if self.rendering == True:
            self.init_window()

        self.target_qs = []
        for i in np.arange(-0.6, 0.6, 0.05):
            for j in np.arange(-0.6, 0.6, 0.05):
                self.target_qs.append(np.array([i, j]))
                num_of_targets += 1
        # self.target_qs = [np.array([0.195, -0.792])]

        self.target_iter = 0
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        if self.control_type == Control_Type.NEURON:
            self.m_ctrl = np.zeros(8)
            self.action_space = spaces.Box(low=0, high=1.0,shape=(8,), dtype=np.float32)
        else:
            self.m_ctrl = np.zeros(4)
            self.action_space = spaces.Box(low=0, high=1.0, shape=(4,), dtype=np.float32)
        # Example for using image as input (channel-first; channel-last also works):
        #current endfactor pos
        self.observation_space = spaces.Box(low=-50.0, high=50.0,shape=(13,), dtype=np.float32)
        # self.observation_space = spaces.Box(low=-50.0, high=50.0,shape=(6,), dtype=np.float32)

    def step(self, action):
        if self.control_type == Control_Type.NEURON:
            for i in range(8):
                self.m_ctrl[i] = action[i]
        else:
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
        # current_state = np.array([self.data.qpos[0], self.data.qpos[1]])
        # m_target = self.target_qs[self.target_iter]
        # pos_diff_new = np.linalg.norm(current_state - m_target)
        reward = -pos_diff_new

        self.ticks += 1
        if self.ticks >= 10000:
            self.done = True


        observation = np.concatenate((self.target_pos, self.data.xpos[1], self.data.xpos[2], np.array([self.data.qpos[0], self.data.qpos[1], self.data.qvel[0], self.data.qvel[1]])))
        # observation = np.array([m_target[0], m_target[1], self.data.qpos[0], self.data.qpos[1], self.data.qvel[0], self.data.qvel[1]])
        info = {}

        return observation, reward, self.done, info

    def reset(self):
        self.target_iter += 1
        if self.target_iter >= num_of_targets:
            self.target_iter = 0

        # forward kinematics to get the target endfactor position
        self.data.qpos[0] = self.target_qs[self.target_iter][0]
        self.data.qpos[1] = self.target_qs[self.target_iter][1]
        mj.mj_forward(self.model, self.data)
        self.target_pos = self.data.xpos[2].copy()
        print(f"{self.target_iter} {self.target_qs[self.target_iter]} {self.target_pos}")

        self.done = False
        self.ticks = 0
        mj.mj_resetData(self.model, self.data)
        mj.mj_forward(self.model, self.data)

        observation = np.concatenate((self.target_pos, self.data.xpos[1], self.data.xpos[2], np.array([self.data.qpos[0], self.data.qpos[1], self.data.qvel[0], self.data.qvel[1]])))
        # m_target = self.target_qs[self.target_iter]
        # observation = np.array([m_target[0], m_target[1], self.data.qpos[0], self.data.qpos[1], self.data.qvel[0], self.data.qvel[1]])

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
        elif self.control_type == Control_Type.NEURON:
            mj.set_mjcb_control(self.my_neuron_controller)

    def init_window(self):
        glfw.init()
        self.window = glfw.create_window(1200, 900, "Demo", None, None)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        self.scene = mj.MjvScene(self.model, maxgeom=10000)
        self.context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150.value)

    def my_baseline(self, model, data):
        baseline_controller(self.m_ctrl, data)

    def my_RI(self, model, data):
        RI_controller(self.m_ctrl, data)

    def my_stretch_reflex(self, model, data):
        stretch_reflex_controller(self.m_ctrl, data)

    def my_neuron_controller(self, model, data):
        neuron_controller(self.m_ctrl, data)