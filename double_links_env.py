import gym
import numpy as np
from gym import spaces
import mujoco as mj
from mujoco.glfw import glfw
import os

from control import *


class DoubleLinkEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    def __init__(self, \
                 control_type=Control_Type.NEURON, \
                 env_id = 0, \
                 instance_id=0, \
                 c_params=None):
        super(DoubleLinkEnv, self).__init__()

        self.control_type = control_type
        self.c_params = c_params
        self.env_id = env_id

        self.init_mujoco()
        # Neuron Controller
        if self.control_type == Control_Type.NEURON:
            self.controller = NeuronController(c_params)
        elif self.control_type == Control_Type.NEURON_SIMPLE:
            self.controller = NeuronSimpleController(c_params)
        # Baseline Controller
        elif self.control_type == Control_Type.BASELINE:
            self.controller = BaselineController(c_params)
        # Optimal neuron Controller
        elif self.control_type == Control_Type.NEURON_OPTIMAL:
            self.controller = SpinalOptimalController(c_params)
        elif self.control_type == Control_Type.PID:
            self.controller = PIDController()

        mj.set_mjcb_control(self.controller.callback)

        # Other stuff
        self.dt_brain = c_params.brain_dt
        self.instance_id = instance_id
        self.cartesian = False
        self.cartesian_target = np.zeros(3)
        self.rendering = False
        if self.rendering == True:
            self.init_window()

        self.episode_length = c_params.episode_length_in_ticks

        self.num_of_targets = 0
        self.target_qs = []
        for i in np.arange(-0.5, 0.5, 0.25):
            for j in np.arange(-0.5, 0.5, 0.25):
                self.target_qs.append(np.array([i, j]))
                self.num_of_targets += 1

        self.target_iter = 0

        if self.control_type == Control_Type.NEURON:
            self.action_space = spaces.Box(low=0, high=1.0, shape=(8,), dtype=np.float32)

        elif self.control_type == Control_Type.NEURON_OPTIMAL or self.control_type == Control_Type.PID:
            self.action_space = spaces.Box(low=0, high=0.85, shape=(2,), dtype=np.float32)
        else:
            self.action_space = spaces.Box(low=0, high=1.0, shape=(4,), dtype=np.float32)

        self.observation_space = spaces.Box(low=-50, high=50, shape=(6,), dtype=np.float32)

    def step(self, action):
        self.controller.set_action(action)
        time_prev = self.data.time

        while self.data.time - time_prev < self.dt_brain:
            mj.mj_step(self.model, self.data)

        if self.rendering == True:
            mj.mjv_updateScene(self.model, self.data, self.opt, None, \
                               self.cam, mj.mjtCatBit.mjCAT_ALL.value, \
                               self.scene)
            mj.mjr_render(self.viewport, self.scene, self.context)
            glfw.swap_buffers(self.window)
            glfw.poll_events()

        self.ticks += 1
        if self.ticks >= self.episode_length:
            self.done = True

        if self.env_id == 0:
            if self.cartesian == True:
                position_error = self.data.xpos[2] - self.cartesian_target
            else:
                position_error = self.data.qpos - self.target_qs[self.target_iter]

            reward = -np.linalg.norm(position_error)
            if self.control_type == Control_Type.NEURON_OPTIMAL or self.control_type == Control_Type.PID:
                observation = np.array(self.target_qs[self.target_iter])
            else:
                observation = np.concatenate((self.target_qs[self.target_iter], self.controller.obs))
        elif self.env_id == 1:
            reward = 1
            current_q = abs(self.data.qpos[0] + self.data.qpos[1] + self.data.qpos[2]) % (2 * np.pi)
            position_penalty = abs(current_q - np.pi)
            if position_penalty > 0.25 * np.pi:
                self.done = True
            observation = np.array([self.data.qpos[0], self.data.qpos[1], self.data.qpos[2], self.data.qvel[0], self.data.qvel[1], self.data.qvel[2]])

        info = {}
        return observation, reward, self.done, info

    def reset(self):
        self.done = False
        self.ticks = 0

        if self.env_id == 0:
            self.target_iter += 1
            if self.target_iter >= self.num_of_targets:
                self.target_iter = 0
            m_target = self.target_qs[self.target_iter]
            mj.mj_resetData(self.model, self.data)
            self.data.qpos[0] = m_target[0]
            self.data.qpos[1] = m_target[1]
            mj.mj_forward(self.model, self.data)
            self.cartesian_target = self.data.xpos[2]

            mj.mj_resetData(self.model, self.data)
            mj.mj_forward(self.model, self.data)

            # print(m_target)
            if self.control_type == Control_Type.NEURON_OPTIMAL or self.control_type == Control_Type.PID:
                observation = np.array(self.target_qs[self.target_iter])
            else:
                observation = np.array([m_target[0], m_target[1], \
                                    self.data.qpos[0], self.data.qvel[0], self.data.qpos[1], \
                                    self.data.qvel[1]])

        elif self.env_id == 1:
            mj.mj_resetData(self.model, self.data)
            self.data.qpos[0] = 0.4
            self.data.qpos[1] = -0.87
            self.data.qpos[2] = -2.32
            mj.mj_forward(self.model, self.data)
            observation = np.array([self.data.qpos[0], self.data.qpos[1], self.data.qpos[2], self.data.qvel[0], self.data.qvel[1], self.data.qvel[2]])

        return observation

    def close(self):
        if self.rendering == True:
            glfw.set_window_should_close(self.window, True)
            glfw.terminate()

    def init_mujoco(self):

        xml_path = self.c_params.model_dir
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

        self.c_params.fs = 1 / self.model.opt.timestep

    def init_window(self):
        glfw.init()
        self.window = glfw.create_window(1200, 900, "Demo", None, None)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        self.scene = mj.MjvScene(self.model, maxgeom=10000)
        self.context = mj.MjrContext(self.model, \
                                     mj.mjtFontScale.mjFONTSCALE_150.value)

        viewport_width, viewport_height = glfw.get_framebuffer_size(self.window)
        self.viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    def get_num_of_targets(self):
        return self.num_of_targets;
