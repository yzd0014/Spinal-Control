import gym
import numpy as np
from gym import spaces
import mujoco as mj
from mujoco.glfw import glfw
import os
from double_link_controllers import *
SLOW = 0
FAST = 1

class MyDoublePendulumEnv:
    """Custom Environment that follows gym interface."""
    def __init__(self, i_episode_length=500):
        self.rendering = False
        self.init_mujoco()
        if self.rendering == True:
            self.init_window()

        self.episode_length = i_episode_length

        self.num_of_targets = 0
        self.target_qs = []
        for i in np.arange(-0.6, 0.6, 0.2):
            for j in np.arange(-0.6, 0.6, 0.2):
                self.target_qs.append(np.array([i, j]))
                self.num_of_targets += 1

        self.max_muscle_length = 0.677
        self.target_iter = 0
        self.spinal_weights =  np.zeros((4, 8))
        self.spinal_inputs = np.zeros(8)
        self.spinal_outputs = np.zeros(4)

    def step(self):
        for i in range(4):
            self.spinal_inputs[4 + i] = self.data.actuator_length[i] / self.max_muscle_length

        self.spinal_outputs = np.matmul(self.spinal_weights, self.spinal_inputs)
        mj.mj_step(self.model, self.data)

        if self.rendering == True:
            mj.mjv_updateScene(self.model, self.data, self.opt, None, self.cam, mj.mjtCatBit.mjCAT_ALL.value, self.scene)
            mj.mjr_render(self.viewport, self.scene, self.context)
            glfw.swap_buffers(self.window)
            glfw.poll_events()

        current_state = np.array([self.data.qpos[0], self.data.qpos[1]])
        m_target = self.target_qs[self.target_iter]
        pos_penalty = np.linalg.norm(current_state - m_target)

        self.ticks += 1
        if self.ticks >= self.episode_length:
            self.done = True

        return pos_penalty

    def reset(self):
        self.done = False
        self.ticks = 0
        self.target_iter += 1
        if self.target_iter >= self.num_of_targets:
            self.target_iter = 0

        self.spinal_outputs = np.zeros(4)
        m_target = self.target_qs[self.target_iter]
        mj.mj_resetData(self.model, self.data)
        self.data.qpos[0] = m_target[0]
        self.data.qpos[1] = m_target[1]
        mj.mj_forward(self.model, self.data)
        for i in range(4):
            self.spinal_inputs[i] = self.data.actuator_length[i] / self.max_muscle_length

        mj.mj_resetData(self.model, self.data)
        mj.mj_forward(self.model, self.data)

    def close(self):
        if self.rendering == True:
            glfw.set_window_should_close(self.window, True)
            glfw.terminate()

    def init_mujoco(self):
        xml_path = 'double_links_fast.xml'
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

        mj.set_mjcb_control(self.my_baseline)

    def init_window(self):
        glfw.init()
        self.window = glfw.create_window(1200, 900, "Demo", None, None)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        self.scene = mj.MjvScene(self.model, maxgeom=10000)
        self.context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150.value)

        viewport_width, viewport_height = glfw.get_framebuffer_size(self.window)
        self.viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)


    def get_num_of_targets(self):
        return self.num_of_targets;

    def my_baseline(self, model, data):
        baseline_controller(self.spinal_outputs, data)

    def set_spinal_weights(self, i_A):
        self.spinal_weights = i_A
