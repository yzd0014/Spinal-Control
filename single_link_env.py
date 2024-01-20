import gym
import numpy as np
from gym import spaces
import mujoco as mj
from mujoco.glfw import glfw
import os

from control import *


def get_obs(controller, data, env_id):
    if env_id == -1:
        obs = np.array([controller.target_pos[0], data.qpos[0], data.qvel[0]])
    return obs


class SingleLinkEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    def __init__(self, \
                 control_type=Control_Type.NEURON, \
                 env_id=0, \
                 instance_id=0, \
                 c_params=None):
        super(SingleLinkEnv, self).__init__()

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
            self.controller = BaselineController(c_params, env_id)
        # Optimal neuron Controller
        elif self.control_type == Control_Type.NEURON_OPTIMAL:
            self.controller = SpinalOptimalController()
        elif self.control_type == Control_Type.PID:
            self.controller = PIDController()
        elif self.control_type == Control_Type.EP:
            self.controller = EPController(env_id)
        elif self.control_type == Control_Type.FF:
            self.controller = FeedForwardController(env_id)
        elif self.control_type == Control_Type.EP_GENERAL:
            self.controller = GeneralEPController()
        elif self.control_type == Control_Type.FF_GENERAL:
            self.controller = FeedForwardGeneralController(env_id)
        elif self.control_type == Control_Type.FF_OPTIMAL:
            self.controller = AngleStiffnessController(env_id, enable_cocontraction=True)
        elif self.control_type == Control_Type.PPO:
            self.controller = PPOController(env_id)

        # Set callback
        mj.set_mjcb_control(self.env_callback)

        # Other stuff
        self.dt_brain = c_params.brain_dt
        self.instance_id = instance_id
        self.rendering = False

        if self.rendering == True:
            self.init_window()

        self.episode_length = c_params.episode_length_in_ticks
        self.action_space = self.controller.get_action_space()
        self.observation_space = self.get_obs_space()

    def step(self, action):
        if self.ticks >= self.episode_length - 1:
            self.done = True

        self.controller.set_action(action)
        time_prev = self.data.time
        while self.data.time - time_prev < self.dt_brain:
            mj.mj_step(self.model, self.data)

        if self.env_id == -1:
            target = 0.7 * np.sin(self.data.time * 2 * np.pi * self.w)
            self.controller.target_pos[0] = target
            position_error = self.data.qpos[0] - target
            reward = -0.1 * np.absolute(position_error)

        observation = get_obs(self.controller, self.data, self.env_id)
        info = {}

        if self.rendering == True:
            self.render()
        self.ticks += 1

        return observation, reward, self.done, info

    def reset(self):
        self.done = False
        self.ticks = 0
        if self.control_type == Control_Type.PID:
            self.controller.q_error = np.zeros(2)

        if self.env_id == -1:
            mj.mj_resetData(self.model, self.data)
            mj.mj_forward(self.model, self.data)
            self.w = 1
            target = 0.7 * np.sin(self.data.time * 2 * np.pi * self.w)
            self.controller.target_pos[0] = target

        observation = get_obs(self.controller, self.data, self.env_id)
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

    def render(self):
        mj.mjv_updateScene(self.model, self.data, self.opt, None, \
                           self.cam, mj.mjtCatBit.mjCAT_ALL.value, \
                           self.scene)
        mj.mjr_render(self.viewport, self.scene, self.context)
        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def env_callback(self, model, data):
        self.controller.callback(model, data)
        evn_controller(self.env_id, model, data)

    def get_obs_space(self):
        if self.env_id == -1:
            return spaces.Box(low=-100, high=100, shape=(3,), dtype=np.float32)