from stable_baselines3 import PPO
from stable_baselines3 import TD3
import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle

from control import *

modelid = "1694208977"

# Load Params
print("\n\n")
print("loading env and control parameters " + "./models/" + modelid + "\n")

control_type, \
episode_length, \
fs_brain_factor, \
controller_params  = pickle.load(open("./models/" + modelid + "/" \
                                    + "env_contr_params.p", "rb"))

dt_brain = 1.0/controller_params.fs * fs_brain_factor
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

# Neuron Controller
if control_type == Control_Type.NEURON:
  controller = NeuronController(controller_params)
# Neuron Controller
elif control_type == Control_Type.NEURON_SIMPLE:
  controller = NeuronSimpleController(controller_params)
# Baseline Controller
elif control_type == Control_Type.BASELINE:
  controller = BaselineController(controller_params)

def keyboard(window, key, scancode, act, mods):
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)
        init_controller(model, data)

def mouse_button(window, button, act, mods):
    # update button state
    global button_left
    global button_middle
    global button_right

    button_left = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

    # update mouse position
    glfw.get_cursor_pos(window)

def mouse_move(window, xpos, ypos):
    # compute mouse displacement, save
    global lastx
    global lasty
    global button_left
    global button_middle
    global button_right

    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos
    lasty = ypos

    # no buttons down: nothing to do
    if (not button_left) and (not button_middle) and (not button_right):
        return

    # get current window size
    width, height = glfw.get_window_size(window)

    # get shift key state
    PRESS_LEFT_SHIFT = glfw.get_key(
        window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
    PRESS_RIGHT_SHIFT = glfw.get_key(
        window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

    # determine action based on mouse button
    if button_right:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_MOVE_H
        else:
            action = mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_ROTATE_H
        else:
            action = mj.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM

    mj.mjv_moveCamera(model, action, dx/height,
                      dy/height, scene, cam)

def scroll(window, xoffset, yoffset):
    action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, 0.0, -0.05 *
                      yoffset, scene, cam)

def init_controller(model,data):
    if env_id == 2:
        mj.mj_resetData(model, data)
        data.qpos[0] = 0.4
        data.qpos[1] = -0.87
        data.qpos[2] = -2.32
        mj.mj_forward(model, data)

def callback(model, data):
  global global_timer
  if data.time - global_timer >= dt_brain:
    observation = np.concatenate(([m_target[0], m_target[1]], \
                                  controller.obs, \
                                  np.array([0,0])))
    action, _states = PPO_model.predict(observation)
    controller.set_action(action)
    global_timer = data.time

  controller.callback(model,data)
  data2write = np.concatenate(([m_target[0],m_target[1]], \
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

# install GLFW mouse and keyboard callbacks
glfw.set_key_callback(window, keyboard)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)

# Example on how to set camera configuration
cam.azimuth = 90
cam.elevation = -20
cam.distance = 2
cam.lookat = np.array([0.0, -1, 2])

def compute_target_pos(w, r):
    x = r * np.cos(-0.5 * np.pi + w)
    z = r * np.sin(-0.5 * np.pi + w) + 2
    output = np.array([x, 0, z])
    return output
#load modes for each controller
w = -0.48
m_target = np.array([0.2, -0.15])
PPO_model_path0 = "./models/" + modelid + "/" + runid
PPO_model = PPO.load(PPO_model_path0)

#initialize the controller
#init_controller(model,data)
#set the controller
mj.set_mjcb_control(callback)
#if control_type == spinal_controllers.Control_Type.BASELINE:
#    mj.set_mjcb_control(baseline_callback)
#elif control_type == spinal_controllers.Control_Type.NEURON:
#    mj.set_mjcb_control(callback)

global_timer = data.time
while not glfw.window_should_close(window):
    time_prev = data.time

    while (data.time - time_prev < 1.0/60.0):
        mj.mj_step(model, data)
    # if (data.time>=simend):
    #     break;

    #print camera configuration (help to initialize the view)
    if print_camera_config==1:
        print('cam.azimuth =',cam.azimuth, \
              ';','cam.elevation =',cam.elevation, \
              ';','cam.distance = ',cam.distance)
        print('cam.lookat =np.array([',cam.lookat[0], \
              ',',cam.lookat[1],',',cam.lookat[2],'])')

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