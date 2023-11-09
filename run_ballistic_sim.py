from stable_baselines3 import PPO
from stable_baselines3 import TD3
import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import sys, getopt
from scipy.optimize import minimize
from scipy.optimize import Bounds

from control import *

global_timer = 0

def run_sim(modelid,target):

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




  xml_path = 'double_links_fast.xml'

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

  def generate_trajectory(n, min_time=1, max_time=5, min_x=-0.8, max_x=0.8):

    # Generate t unique time points between min_time and max_time
    dt = 1/200

    # Randomly select n values for the trajectory
    x1 = np.random.uniform(min_x,max_x,n)
    x2 = np.random.uniform(min_x,max_x,n)

    X = np.vstack((np.concatenate((np.array([0]),x1)),
                  np.concatenate((np.array([0]),x2))))

    dX = X[:,0:-1] - X[:,1:]
    dXnorm = np.linalg.norm(dX,axis=0)
    Cmax = 5
    cost = lambda dt: np.absolute(Cmax - np.sum(np.dot(dXnorm,dt)))
    x0 = np.ones(n)*1
    bounds = Bounds(np.ones(n)*min_time,np.ones(n)*max_time,keep_feasible=True)
    res = minimize(cost, x0, bounds=bounds, method = 'Nelder-Mead',
                    options={'disp': False,
                            'maxiter': 1e3,
                            'maxfev' : 1e3,
                            'xatol' : 1e-5})

    Dt = res.x
    t = np.zeros((n,1))
    for i in range(len(Dt)-1):
      t[i+1] = t[i] + Dt[i]

    x1 = np.vstack((x1,x1))
    x1 = x1.transpose()
    x1 = x1.reshape(1,2*n)

    x2 = np.vstack((x2,x2))
    x2 = x2.transpose()
    x2 = x2.reshape(1,2*n)

    t = t.squeeze()
    t = np.vstack((np.concatenate((np.array([0]),t[0:-1]+dt)),t))
    t = t.transpose()
    t = t.reshape(1,2*n)

    t = t.squeeze()
    x1 = x1.squeeze()
    x2 = x2.squeeze()

    # Create the trajectory
    trajectory = lambda tq : np.array([np.interp(tq,t,x1),np.interp(tq,t,x2)])

    return trajectory

  trajectory = generate_trajectory(10,min_time=1,max_time=5)

  def callback(model, data):
    global global_timer
    target = trajectory(data.time)
    if data.time - global_timer >= dt_brain:
      observation = np.concatenate(([target[0], target[1]], \
                                    controller.obs, \
                                    np.array([0,0])))
      action, _states = PPO_model.predict(observation)
      controller.set_action(action)
      global_timer = data.time

    controller.callback(model,data)
    data2write = np.concatenate(([target[0],target[1]], \
                                data.qpos, \
                                controller.obs, \
                                data.actuator_length, \
                                data.ctrl, \
                                controller.action))
    datastr = ','.join(str(x) for x in data2write)
    fdata.write(datastr + '\n')


  #get the full path
  dirname = os.path.dirname(__file__)
  abspath = os.path.join(dirname + "/" + xml_path)
  xml_path = abspath

  # MuJoCo data structures
  model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
  data = mj.MjData(model)                # MuJoCo data
  cam = mj.MjvCamera()                        # Abstract camera
  opt = mj.MjvOption()                        # visualization options

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
  PPO_model_path0 = "./models/" + modelid + "/" + runid
  PPO_model = PPO.load(PPO_model_path0)

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
  target = np.array([0.85, -0.85])
  modelid = ''
  opts, args = getopt.getopt(argv,"m:t")
  for opt, arg in opts:
    if opt == '-m':
      modelid = arg
    elif opt == '-t':
      target = arg

  run_sim(modelid,target)

if __name__=="__main__":
  main(sys.argv[1:])
