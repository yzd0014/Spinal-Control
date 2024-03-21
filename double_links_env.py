import gym
import numpy as np
from gym import spaces
import mujoco as mj
from mujoco.glfw import glfw
import os
import random
from control import *


def get_obs(controller, data, env_id):
    if env_id == 0:
        # obs = np.array([controller.target_pos[0], controller.target_pos[1], data.qpos[0], data.qvel[0], data.qpos[1], data.qvel[1]])
        obs = np.array([controller.target_pos[0], controller.target_pos[1], controller.target_pos[2], controller.target_pos[3],
                        data.actuator_length[0], data.actuator_length[1],
                        data.actuator_length[2], data.actuator_length[3],
                        data.actuator_velocity[0], data.actuator_velocity[1],
                        data.actuator_velocity[2], data.actuator_velocity[3]])
        # obs = np.array([controller.target_pos[0], controller.target_pos[1],
        #                 data.actuator_length[0], data.actuator_length[1],
        #                 data.actuator_length[2], data.actuator_length[3],
        #                 data.actuator_velocity[0], data.actuator_velocity[1],
        #                 data.actuator_velocity[2], data.actuator_velocity[3]])
    elif env_id == 1:
        # obs = np.array([data.qpos[0], data.qpos[1], data.qpos[2], data.qvel[0], data.qvel[1], data.qvel[2]])
        # obs = np.array([data.actuator_length[0], data.actuator_length[1],
        #                 data.actuator_length[2], data.actuator_length[3],
        #                 data.actuator_velocity[0], data.actuator_velocity[1],
        #                 data.actuator_velocity[2], data.actuator_velocity[3],
        #                 data.xpos[3][0], data.xpos[3][1], data.xpos[3][2]])
        obs = np.array([data.actuator_length[0], data.actuator_length[1],
                        data.actuator_length[2], data.actuator_length[3],
                        data.actuator_velocity[0], data.actuator_velocity[1],
                        data.actuator_velocity[2], data.actuator_velocity[3],
                        data.qpos[0] + data.qpos[1] + data.qpos[2], data.qvel[0] + data.qvel[1] + data.qvel[2]])
    elif env_id == 2:
        obs = np.array([controller.target_pos[0], controller.target_pos[1], data.time])
    elif env_id == 3:
        obs = np.array([controller.target_pos[0], data.xpos[3][0], data.xpos[3][1], data.xpos[3][2], \
                        data.xpos[2][0], data.xpos[2][1], data.xpos[2][2], \
                        data.qpos[0], data.qvel[0], data.qpos[1], data.qvel[1]])
    elif env_id == 4:
        obs = np.array([data.qpos[0], data.qpos[1], data.qpos[2], data.qvel[0], data.qvel[1], data.qvel[2], data.time])
    return obs

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
        elif self.control_type == Control_Type.SAC:
            self.controller = SACController(env_id)

        # Set callback
        mj.set_mjcb_control(self.env_callback)

        # Other stuff
        self.dt_brain = c_params.brain_dt
        self.instance_id = instance_id
        self.cartesian = False
        self.cartesian_target = np.zeros(3)
        self.rendering = False

        if self.rendering == True:
            self.init_window()

        self.episode_length = c_params.episode_length_in_ticks

        self.training_data = []
        theta = [0.1, 0.6, 1.5, 1.7, 2.08]
        for i in range(5):
            for j in range(5):
                for k in np.arange(0, 1.0001, 0.5):
                    for w in np.arange(0, 1.0001, 0.5):
                        self.training_data.append(np.array([theta[i], theta[j], k, w]))
        self.target_iter = 0

        self.action_space = self.controller.get_action_space()
        self.observation_space = self.get_obs_space()

    def step(self, action):
        if self.ticks >= self.episode_length - 1:
            self.done = True

        self.controller.set_action(action)
        time_prev = self.data.time
        while self.data.time - time_prev < self.dt_brain:
            mj.mj_step(self.model, self.data)

            if self.env_id == 2:
                if self.data.ncon > 0:
                    if self.data.contact[0].geom1 == 0 and self.data.contact[0].geom2 == 3:
                        self.done = True
                    elif self.data.contact[0].geom1 == 3 and self.data.contact[0].geom2 == 0:
                        self.done = True

                if self.done == False:
                    reward = 0
                else:
                    dist = np.linalg.norm(
                        np.array([self.data.xpos[3][0], self.data.xpos[3][1]]) - self.controller.target_pos)
                    reward = -dist
                    # print(self.data.xpos[4][0])
                    break
          
        if self.env_id == 0:
            if self.cartesian == True:
                curr_pos = np.array([self.data.xpos[2][0], self.data.xpos[2][2]])
                position_error = 100 * (curr_pos - self.controller.target_pos)
            else:
                total_crtl = self.data.ctrl[0] + self.data.ctrl[1] + self.data.ctrl[2] + self.data.ctrl[3] + self.data.ctrl[4] + self.data.ctrl[5]
                target = np.array([self.controller.target_pos[0], self.controller.target_pos[1]])
                position_error = np.linalg.norm(self.data.qpos -  target)
            reward = -position_error - 0.005 * total_crtl
            # reward = -position_error

        elif self.env_id == 1:
            reward = self.dt_brain
            current_q = abs(self.data.qpos[0] + self.data.qpos[1] + self.data.qpos[2]) % (2 * np.pi)
            position_penalty = abs(current_q - np.pi)
            if position_penalty > 0.25 * np.pi:
                self.done = True
        elif self.env_id == 3:
            if self.done == False:
                reward = 0
            else:
                pos_err = np.linalg.norm(
                    np.array([self.data.xpos[3][0], self.data.xpos[3][1]]) - self.controller.target_pos)
                vel_err = np.linalg.norm(self.data.cvel[3])
                reward = -pos_err - vel_err
        elif self.env_id == 4:
            if self.done == False:
                reward = 0
            else:
                current_pos = self.data.qpos[0] + self.data.qpos[1] + self.data.qpos[2]
                pos_err = abs(current_pos - np.pi)
                vel_err = abs(self.data.qvel[0] + self.data.qvel[1] + self.data.qvel[2])
                reward = -3 * pos_err - vel_err

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

        if self.env_id == 0:
            self.controller.target_pos = np.array([self.training_data[self.target_iter][0], self.training_data[self.target_iter][1],
                                                   self.training_data[self.target_iter][2], self.training_data[self.target_iter][3]])
            # current_external_force = self.training_data[self.target_iter][2]
            mj.mj_resetData(self.model, self.data)
            # self.data.qpos[0] = self.controller.target_pos[0]
            # self.data.qpos[1] = self.controller.target_pos[1]
            mj.mj_forward(self.model, self.data)
            self.target_iter += 1

            if self.target_iter >= len(self.training_data):
                self.target_iter = 0

        elif self.env_id == 1:
            mj.mj_resetData(self.model, self.data)
            # self.data.qpos[0] = 0.4
            # self.data.qpos[1] = -0.87
            # self.data.qpos[2] = -2.32
            self.data.qpos[0] = random.uniform(0.1, 0.5)
            self.data.qpos[1] = random.uniform(0.1, 0.5)
            # last_link_angle = random.uniform(2.45, 3)
            last_link_angle = random.uniform(2.9, 3)
            self.data.qpos[2] = last_link_angle - self.data.qpos[0] - self.data.qpos[1]
            mj.mj_forward(self.model, self.data)

        elif self.env_id == 2:
            mj.mj_resetData(self.model, self.data)
            self.model.eq_active[0] = 1
            mj.mj_forward(self.model, self.data)
            self.controller.target_pos = np.array([-10, 0])

        elif self.env_id == 3:
            mj.mj_resetData(self.model, self.data)
            mj.mj_forward(self.model, self.data)
            self.controller.target_pos = np.array([-7, 0])

        elif self.env_id == 4:
            mj.mj_resetData(self.model, self.data)
            mj.mj_forward(self.model, self.data)

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

        if self.env_id == 0 or self.env_id == 1:
            self.cam.azimuth = 90
            self.cam.elevation = -20
            self.cam.distance = 2
            self.cam.lookat = np.array([0.0, 0, -0.4])
        elif self.env_id == 2:
            self.cam.azimuth = 160
            self.cam.elevation = -10
            self.cam.distance = 5
            self.cam.lookat = np.array([0.0, 0.0, 1])
        else:
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
        return len(self.training_data);

    def render(self):
        mj.mjv_updateScene(self.model, self.data, self.opt, None, \
                           self.cam, mj.mjtCatBit.mjCAT_ALL.value, \
                           self.scene)
        mj.mjr_render(self.viewport, self.scene, self.context)
        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def env_callback(self, model, data):
        self.controller.callback(model, data)
        env_controller(self.controller, model, data)

    def get_obs_space(self):
        if self.env_id == 0:
            # return spaces.Box(low=-100, high=100, shape=(6,), dtype=np.float32)
            return spaces.Box(low=-100, high=100, shape=(12,), dtype=np.float32)
        elif self.env_id == 1:
            # return spaces.Box(low=-100, high=100, shape=(6,), dtype=np.float32)
            return spaces.Box(low=-100, high=100, shape=(10,), dtype=np.float32)
        elif self.env_id == 2:
            return spaces.Box(low=-100, high=100, shape=(3,), dtype=np.float32)
        elif self.env_id == 3:
            return spaces.Box(low=-100, high=100, shape=(11,), dtype=np.float32)
        elif self.env_id == 4:
            return spaces.Box(low=-100, high=100, shape=(7,), dtype=np.float32)

