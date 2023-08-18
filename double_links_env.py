import gym
import numpy as np
from gym import spaces
import mujoco as mj
from mujoco.glfw import glfw
import os
from double_link_controllers import *
SLOW = 0
FAST = 1

class DoubleLinkEnv(gym.Env):
    """Custom Environment that follows gym interface."""
    def __init__(self, control_type = Control_Type.BASELINE, env_id = 1, instance_id = 0, speed_mode = SLOW):
        super(DoubleLinkEnv, self).__init__()

        self.qpos0_history = []
        self.qvel0_history = []
        self.qpos1_history = []
        self.qvel1_history = []
        self.qpos2_history = []
        self.qvel2_history = []
        self.dt_brain = 1.0/10.0

        self.alpha =  0.4691358024691358
        self.beta = 0.9
        self.speed_mode = speed_mode
        self.env_id = env_id
        self.instance_id = instance_id
        self.control_type = control_type
        self.rendering = False
        self.init_mujoco()
        if self.rendering == True:
            self.init_window()

        if self.env_id == 1:
            if self.speed_mode == SLOW:
                self.episode_length = 5000
            elif self.speed_mode == FAST:
                self.episode_length = 50
        elif self.env_id == 2:
            if self.speed_mode == SLOW:
                self.episode_length = 150000
            elif self.speed_mode == FAST:
                self.episode_length = 50000

        self.ep_position_reward = 0
        self.ep_ctrl_reward = 0
        self.ep_smoothing_reward = 0
        self.ep_ctrl = [[], [], [], []]

        self.num_of_targets = 0
        if self.env_id == 1:
            self.target_qs = []
            for i in np.arange(-0.2, 0.2, 0.1):
                for j in np.arange(-0.2, 0.2, 0.1):
            # for i in np.arange(-0.6, 0.6, 0.2):
            #     for j in np.arange(-0.6, 0.6, 0.2):
                    self.target_qs.append(np.array([i, j]))
                    self.num_of_targets += 1
            # self.target_qs = [np.array([0.195, -0.792])]

        elif self.env_id == 2:
            self.num_of_targets = 16
            self.episode_reward = 0

        self.target_iter = 0
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        if self.control_type == Control_Type.NEURON or self.control_type == Control_Type.NEURON_TEST or self.control_type == Control_Type.NEURON_FILTER:
            self.m_ctrl = np.zeros(8)
            self.action_space = spaces.Box(low=0, high=1.0,shape=(8,), dtype=np.float32)
        else:
            self.m_ctrl = np.zeros(4)
            self.action_space = spaces.Box(low=0, high=1.0, shape=(4,), dtype=np.float32)
        # Example for using image as input (channel-first; channel-last also works):
        #current endfactor pos
        if self.env_id == 1:
            # self.observation_space = spaces.Box(low=-50.0, high=50.0,shape=(13,), dtype=np.float32)
            self.observation_space = spaces.Box(low=-50.0, high=50.0, shape=(8,), dtype=np.float32)
        elif self.env_id == 2:
            self.observation_space = spaces.Box(low=-50.0, high=50.0, shape=(6,), dtype=np.float32)

    def step(self, action):
        if self.control_type == Control_Type.NEURON or self.control_type == Control_Type.NEURON_TEST or self.control_type == Control_Type.NEURON_FILTER:
            for i in range(8):
                self.m_ctrl[i] = action[i]
        else:
            for i in range(4):
                self.m_ctrl[i] = action[i]

        time_prev = self.data.time
        while self.data.time - time_prev < self.dt_brain:
            mj.mj_step(self.model, self.data)
            if len(self.qpos0_history) * self.model.opt.timestep >= self.dt_brain:
                self.qpos0_history.pop(0)
                self.qvel0_history.pop(0)
                self.qpos1_history.pop(0)
                self.qvel1_history.pop(0)
                if self.env_id == 2:
                    self.qpos2_history.pop(0)
                    self.qvel2_history.pop(0)
            self.qpos0_history.append(self.data.qpos[0])
            self.qvel0_history.append(self.data.qvel[0])
            self.qpos1_history.append(self.data.qpos[1])
            self.qvel1_history.append(self.data.qvel[1])
            if self.env_id == 2:
                self.qpos2_history.append(self.data.qpos[2])
                self.qvel2_history.append(self.data.qvel[2])

        if self.rendering == True:
            mj.mjv_updateScene(self.model, self.data, self.opt, None, self.cam, mj.mjtCatBit.mjCAT_ALL.value, self.scene)
            mj.mjr_render(self.viewport, self.scene, self.context)
            glfw.swap_buffers(self.window)
            glfw.poll_events()

        ctrl_reward = 0
        for i in range(4):
            ctrl_reward += -np.linalg.norm(self.data.ctrl[i])
        self.ep_ctrl_reward += ctrl_reward

        for i in range(4):
            self.ep_ctrl[i].append(self.data.ctrl[i])

        self.ticks += 1
        Sm = [0, 0, 0, 0]
        if self.ticks >= self.episode_length:
            self.done = True
            for i in range(4):
                x = np.array(self.ep_ctrl[i])
                X = np.fft.fft(x)
                M = np.absolute(X)
                for freq in range(len(X)):
                    Sm[i] += freq * M[freq]
                Sm[i] = 2 * Sm[i] / (len(X) * self.episode_length)
            self.ep_smoothing_reward = -np.sum(Sm)

        if self.env_id == 1:
            # pos_diff_new = np.linalg.norm(self.data.xpos[2] - self.target_pos)
            current_state = np.array([self.data.qpos[0], self.data.qpos[1]])
            m_target = self.target_qs[self.target_iter]
            pos_diff_new = np.linalg.norm(current_state - m_target)
            position_reward = -pos_diff_new
            self.ep_position_reward += position_reward
            reward = position_reward + 0.001 * self.ep_smoothing_reward

            # observation = np.concatenate((self.target_pos, self.data.xpos[1], self.data.xpos[2], np.array([self.data.qpos[0], self.data.qpos[1], self.data.qvel[0], self.data.qvel[1]])))
            # observation = np.array([m_target[0], m_target[1], self.data.qpos[0], self.data.qpos[1], self.data.qvel[0], self.data.qvel[1]])
            history_length = len(self.qpos0_history)
            # observation = np.array( [m_target[0], m_target[1], self.data.qpos[0], self.data.qvel[0], self.data.qpos[1], self.data.qvel[1], 0, 0])
            observation = np.array(
                [m_target[0], m_target[1],
                 sum(self.qpos0_history)/history_length, sum(self.qvel0_history)/history_length,
                 sum(self.qpos1_history)/history_length, sum(self.qvel1_history)/history_length, 0, 0])
        elif self.env_id == 2:
            reward = 1
            current_q = abs(self.data.qpos[0] + self.data.qpos[1] + self.data.qpos[2]) % (2 * np.pi)
            position_penalty = abs(current_q - np.pi)
            if position_penalty > 0.25 * np.pi:
                self.done = True
            # observation = np.array([self.data.qpos[0], self.data.qpos[1], self.data.qpos[2], self.data.qvel[0], self.data.qvel[1], self.data.qvel[2]])
            # observation = np.array( [0, 0, self.data.qpos[0], self.data.qvel[0], self.data.qpos[1], self.data.qvel[1], self.data.qpos[2], self.data.qvel[2]])
            history_length = len(self.qpos0_history)
            observation = np.array(
                [sum(self.qpos0_history)/history_length, sum(self.qpos1_history)/history_length, sum(self.qpos2_history)/history_length,
                 sum(self.qvel0_history)/history_length, sum(self.qvel1_history)/history_length, sum(self.qvel2_history)/history_length])

        info = {}
        return observation, reward, self.done, info

    def reset(self):
        self.done = False
        self.ticks = 0

        self.qpos0_history.clear()
        self.qpos1_history.clear()
        self.qpos2_history.clear()
        self.qvel0_history.clear()
        self.qvel1_history.clear()
        self.qvel2_history.clear()

        self.target_iter += 1
        if self.target_iter >= self.num_of_targets:
            self.target_iter = 0

        if self.env_id == 1:
            # print(f"{self.target_iter} position reward: {self.ep_position_reward}, smooth reward: {self.ep_smoothing_reward}")
            m_target = self.target_qs[self.target_iter]
            # observation = np.array([m_target[0], m_target[1], self.data.qpos[0], self.data.qpos[1], self.data.qvel[0], self.data.qvel[1]])
            # observation = np.concatenate((self.target_pos, self.data.xpos[1], self.data.xpos[2], np.array([self.data.qpos[0], self.data.qpos[1], self.data.qvel[0], self.data.qvel[1]])))
            observation = np.array([m_target[0], m_target[1], self.data.qpos[0], self.data.qvel[0], self.data.qpos[1], self.data.qvel[1], 0, 0])

            mj.mj_resetData(self.model, self.data)
            mj.mj_forward(self.model, self.data)
        elif self.env_id == 2:
            self.episode_reward = 0
            mj.mj_resetData(self.model, self.data)
            self.data.qpos[0] = 0.4
            self.data.qpos[1] = -0.87
            self.data.qpos[2] = -2.32
            mj.mj_forward(self.model, self.data)
            observation = np.array([self.data.qpos[0], self.data.qpos[1], self.data.qpos[2], self.data.qvel[0], self.data.qvel[1],self.data.qvel[2]])
            # observation = np.array([0, 0, self.data.qpos[0], self.data.qvel[0], self.data.qpos[1], self.data.qvel[1], self.data.qpos[2], self.data.qvel[2]])

        self.ep_position_reward = 0
        self.ep_ctrl_reward = 0
        self.ep_smoothing_reward = 0

        return observation

    # def render(self):
    #   pass
    def close(self):
        if self.rendering == True:
            glfw.set_window_should_close(self.window, True)
            glfw.terminate()

    def init_mujoco(self):
        if self.env_id == 1:
            if self.speed_mode == SLOW:
                xml_path = 'double_links.xml'
            elif self.speed_mode == FAST:
                xml_path = 'double_links_fast.xml'
        elif self.env_id == 2:
            if self.speed_mode == SLOW:
                xml_path = 'inverted_pendulum.xml'
            elif self.speed_mode == FAST:
                xml_path = 'inverted_pendulum_fast.xml'
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
        elif self.control_type == Control_Type.NEURON_FILTER:
            mj.set_mjcb_control(self.my_neuron_filter_controller)
        elif self.control_type == Control_Type.NEURON:
            mj.set_mjcb_control(self.my_neuron_controller)
        elif self.control_type == Control_Type.NEURON_TEST:
            mj.set_mjcb_control(self.my_neuron_test_controller)
        elif self.control_type == Control_Type.NEURON_SIMPLE:
            mj.set_mjcb_control(self.my_neuron_simple_controller)

    def init_window(self):
        glfw.init()
        self.window = glfw.create_window(1200, 900, "Demo", None, None)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        self.scene = mj.MjvScene(self.model, maxgeom=10000)
        self.context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150.value)

        viewport_width, viewport_height = glfw.get_framebuffer_size(self.window)
        self.viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)


    def update_neuron_weights(self, x):
        self.alpha = x[0]
        self.beta = x[1]

    def get_num_of_targets(self):
        return self.num_of_targets;

    def my_baseline(self, model, data):
        baseline_controller(self.m_ctrl, data)
        joints_controller(data)#add noise

    def my_neuron_filter_controller(self, model, data):
        neuron_filter_controller(self.m_ctrl, data)

    def my_stretch_reflex(self, model, data):
        stretch_reflex_controller(self.m_ctrl, data)
        joints_controller(data)  # add noise

    def my_neuron_controller(self, model, data):
        neuron_controller(self.m_ctrl, data)
        joints_controller(data)#add noise

    def my_neuron_simple_controller(self, model, data):
        neuron_simple_controller(self.m_ctrl, data)

    def my_neuron_test_controller(self, model, data):
        neuron_test_controller(self.m_ctrl, data, self.alpha, self.beta)
        # joints_controller(data)