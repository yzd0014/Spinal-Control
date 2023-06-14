import gym
import numpy as np
from gym import spaces
import mujoco as mj
from mujoco.glfw import glfw
import os
from double_link_controllers import *

max_angle = 0.2
max_length = 2
min_length = 1.766
pos_stride = 0.1
length_stride = 0.1

all_targets = np.zeros((15, 3))
length_candidate = [1.776, 1.8776, 2]
angle_candidate = [-0.2, -0.1, 0, 0.1, 0.2]
iter = 0
for length in length_candidate:
    for angle in angle_candidate:
        x = length * np.cos(-0.5 * np.pi + angle)
        z = length * np.sin(-0.5 * np.pi + angle) + 2.5
        all_targets[iter] = np.array([x, 0, z])
        iter += 1

num_training_each_target = 100

class DoubleLinkEnv(gym.Env):
    """Custom Environment that follows gym interface."""
    def __init__(self, control_type = Control_Type.BASELINE):
        super(DoubleLinkEnv, self).__init__()

        self.control_type = control_type
        self.rendering = False
        self.init_mujoco()
        if self.rendering == True:
            self.init_window()

        self.angle_t = -max_angle
        self.length_t = min_length
        self.target_pos = np.zeros(3)
        self.target_iter = 0
        self.m_ctrl = np.zeros(4)
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Box(low=0, high=1.0,shape=(4,), dtype=np.float32)
        # Example for using image as input (channel-first; channel-last also works):
        #current endfactor pos
        self.observation_space = spaces.Box(low=-50.0, high=50.0,shape=(7,), dtype=np.float32)

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
        if self.ticks >= 10000:
            self.done = True

        observation = np.concatenate((self.target_pos, np.array([self.data.qpos[0], self.data.qpos[1], self.data.qvel[0], self.data.qvel[1]])))
        info = {}

        return observation, reward, self.done, info

    def reset(self):
        global num_training_each_target

        self.target_pos = all_targets[self.target_iter]
        num_training_each_target -= 1
        if num_training_each_target < 1:
            self.target_iter += 1
            num_training_each_target = 100
        if self.target_iter == 15:
            self.target_iter = 0
        print(f"{self.target_iter}({num_training_each_target})")

        self.done = False
        self.ticks = 0
        mj.mj_resetData(self.model, self.data)
        mj.mj_forward(self.model, self.data)

        observation = np.concatenate((self.target_pos, np.array([self.data.qpos[0], self.data.qpos[1], self.data.qvel[0], self.data.qvel[1]])))
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
        x = self.length_t * np.cos(-0.5 * np.pi + self.angle_t)
        z = self.length_t * np.sin(-0.5 * np.pi + self.angle_t) + 2.5
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