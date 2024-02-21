from stable_baselines3 import PPO
from stable_baselines3 import SAC
import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import sys, getopt

from control import *

global_timer = 0

def run_sim(modelid):

  # Load Params
  print("\n\n")
  print("loading env and control parameters " + "./models/" + modelid + "\n")

  control_type, \
  episode_length, \
  num_episodes, \
  fs_brain_factor, \
  controller_params  = pickle.load(open("./models/" + modelid + "/" \
                                      + "env_contr_params.p", "rb"))

  print("Simulating " + control_type_dic[control_type] + " model")
  print("\n\n")

  dt_brain = (1.0/controller_params.fs) * fs_brain_factor
  # For saving data
  fdata = open("./datalog/" + modelid,'w')

  # Find most recent model
  models_dir = "./models/" + modelid + "/"
  allmodels = sorted(os.listdir(models_dir))
  allmodels.sort(key=lambda fn: \
                 os.path.getmtime(os.path.join(models_dir, fn)))

  runid = allmodels[-1].split(".")
  runid = runid[0]

  xml_path = 'inverted_pendulum_fast.xml'

  simend = 5 #simulation time
  print_camera_config = 0 #set to 1 to print camera config
                          #this is useful for initializing view of the model)
  # For callback functions
  button_left = False
  button_middle = False
  button_right = False
  lastx = 0
  lasty = 0

  controller = InitController(control_type,controller_params)

  def callback(model, data):
    global global_timer
    if data.time - global_timer >= dt_brain:
      # observation = np.concatenate((controller.obs,
      #                               np.array([data.qpos[-1],
      #                                         data.qvel[-1],
      #                                         0, 0])))
      observation = np.array([data.qpos[0], data.qpos[1], data.qpos[2], data.qvel[0], data.qvel[1], data.qvel[2]])
      action, _states = rl_model.predict(observation)
      controller.set_action(action)
      global_timer = data.time
      print(str(data.time) + '\n')

    controller.callback(model,data)
    #data2write = np.concatenate(([target[0],target[1]], \
    #                            data.qpos, \
    #                            controller.obs, \
    #                            data.actuator_length, \
    #                            data.ctrl, \
    #                            controller.action))
    #datastr = ','.join(str(x) for x in data2write)
    #fdata.write(datastr + '\n')


  #get the full path
  dirname = os.path.dirname(__file__)
  abspath = os.path.join(dirname + "/" + xml_path)
  xml_path = abspath

  # MuJoCo data structures
  model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
  data = mj.MjData(model)                # MuJoCo data
  cam = mj.MjvCamera()                        # Abstract camera
  opt = mj.MjvOption()                        # visualization options

  data.qpos[2] = np.pi

  # Init GLFW, create window, make OpenGL context current, request v-sync
  glfw.init()
  window = glfw.create_window(1200, 900, "Demo", None, None)
  glfw.make_context_current(window)
  glfw.swap_interval(1)

  # get framebuffer viewport
  viewport_width, viewport_height = glfw.get_framebuffer_size(window)
  viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

  # initialize visualization data structures
  mj.mjv_defaultCamera(cam)
  mj.mjv_defaultOption(opt)
  scene = mj.MjvScene(model, maxgeom=10000)
  context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

  # Example on how to set camera configuration
  cam.azimuth = 90
  cam.elevation = -20
  cam.distance = 2
  cam.lookat = np.array([0.0, -1, 2])

  #load modes for each controller
  w = -0.48
  rl_model_path = "./models/" + modelid + "/" + runid

  if controller_params.RL_type == "PPO":
    rl_model = PPO.load(rl_model_path)
  elif controller_params.RL_type == "SAC":
    rl_model = SAC.load(rl_model_path)


  #set the controller
  mj.set_mjcb_control(callback)

  global_timer = data.time
  while not glfw.window_should_close(window):
    time_prev = data.time
    mj.mj_step(model, data)

    # Update scene and render
    mj.mjv_updateScene(model, data, opt, None, cam,
                       mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)

    # swap OpenGL buffers (blocking call due to v-sync)
    glfw.swap_buffers(window)

    # process pending GUI events, call GLFW callbacks
    glfw.poll_events()

  glfw.terminate()
  fdata.close()

def main(argv):
  modelid = '1708030061'
  opts, args = getopt.getopt(argv,"m:t")
  for opt, arg in opts:
    if opt == '-m':
      modelid = arg
    elif opt == '-t':
      target = arg

  run_sim(modelid)

if __name__=="__main__":
  main(sys.argv[1:])
