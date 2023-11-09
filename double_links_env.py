import gym
import numpy as np
from gym import spaces
import mujoco as mj
from mujoco.glfw import glfw
import os

from control import *

targetMax = 0.85
targetMin = -0.85


class DoubleLinkEnv(gym.Env):
    """Custom Environment that follows gym interface."""
    def __init__(self,  \
                  control_type = Control_Type.NEURON, \
                  instance_id = 0, \
                  episode_sec = 1, \
                  fs_brain_factor = 20, \
                  c_params = None):
      super(DoubleLinkEnv, self).__init__()

      # Env parameters
      self.control_type = control_type
      self.episode_sec = episode_sec
      self.dt_brain = (1.0/c_params.fs) * fs_brain_factor
      self.c_params = c_params


      self.controller = InitController(self.control_type,c_params)

      # Other stuff
      self.instance_id = instance_id
      self.rendering = False
      self.init_mujoco()
      if self.rendering == True:
       self.init_window()

      #self.target = self.gen_random_target();
      self.target = np.array([0.45, -0.45])
      self.action_space = self.controller.get_action_space()
      self.observation_space = self.controller.get_obs_space()

    def step(self, action):
      self.controller.set_action(action)
      time_prev = self.data.time

      loop_reward = 0
      while self.data.time - time_prev < self.dt_brain:
        mj.mj_step(self.model, self.data)
        position_error = np.linalg.norm(self.data.qpos - self.target)
        #loop_reward += -position_error + (position_error<0.001)/position_error
        loop_reward += -position_error

      reward = loop_reward
      observation = np.concatenate((self.target, \
                                    self.controller.obs, \
                                    np.array([0,0])))

      if self.rendering == True:
        mj.mjv_updateScene(self.model, self.data, self.opt, None, \
                           self.cam, mj.mjtCatBit.mjCAT_ALL.value, \
                           self.scene)
        mj.mjr_render(self.viewport, self.scene, self.context)
        glfw.swap_buffers(self.window)
        glfw.poll_events()

      if self.data.time >= self.episode_sec:
        self.done = True

      info = {}
      return observation, reward, self.done, info

    def reset(self):
      self.done = False
      self.ticks = 0
      #self.target = self.gen_random_target()
      self.target = np.array([0.45, -0.45])
      mj.mj_resetData(self.model, self.data)
      mj.mj_forward(self.model, self.data)
      observation = np.concatenate([self.target, \
                                    self.data.qpos, self.data.qvel, \
                                    np.array([0,0])])
      return observation


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

      mj.set_mjcb_control(self.controller.callback)

    def init_window(self):
      glfw.init()
      self.window = glfw.create_window(1200, 900, "Demo", None, None)
      glfw.make_context_current(self.window)
      glfw.swap_interval(1)

      self.scene = mj.MjvScene(self.model, maxgeom=10000)
      self.context = mj.MjrContext(self.model,\
                                  mj.mjtFontScale.mjFONTSCALE_150.value)

      viewport_width, viewport_height = glfw.get_framebuffer_size(self.window)
      self.viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    def get_num_of_targets(self):
      return self.num_of_targets;

    def gen_random_target(self):
      return np.array([np.random.uniform(targetMin,targetMax), \
                        np.random.uniform(targetMin,targetMax)])

