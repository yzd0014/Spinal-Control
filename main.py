from stable_baselines3 import PPO
from stable_baselines3 import TD3
import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
import matplotlib.pyplot as plt
import spinal_controllers
import double_link_controllers

PPO_MODE = 0
TD3_MODE = 1

control_type = spinal_controllers.Control_Type.BASELINE
env_id = 2
RL_mode = PPO_MODE

if env_id == 0:
    xml_path = 'muscle_control_narrow.xml'  # xml file (assumes this is in the same folder as this file)
elif env_id == 1:
    xml_path = 'double_links.xml'
elif env_id == 2:
    xml_path = 'inverted_pendulum.xml'

simend = 5 #simulation time
print_camera_config = 0 #set to 1 to print camera config
                        #this is useful for initializing view of the model)

# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

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

h = 0.6
w = 0.06
S = 0.1
L = np.sqrt(h*h+w*w)
theta_offset = np.arctan(w/h)

def get_length_from_angle(theta):
    lsqr0 = S*S+L*L-2*S*L*np.cos(np.pi*0.5-theta-theta_offset)
    lsqr1 = S*S+L*L-2*S*L*np.cos(np.pi*0.5+theta-theta_offset)
    l_right = np.sqrt(lsqr0)
    l_left = np.sqrt(lsqr1)
    return l_left, l_right

def compute_target_pos(w, r):
    x = r * np.cos(-0.5 * np.pi + w)
    z = r * np.sin(-0.5 * np.pi + w) + 2
    output = np.array([x, 0, z])
    return output

e0_r = 0
e0_l = 0
def pid_controller(model, data):
    global e0_r
    global e0_l
    kp = 40
    ki = 2
    kd = 5
    l1, l0 = get_length_from_angle(0.7)
    e0_r += (data.actuator_length[1] - l0) * model.opt.timestep
    e0_l += (data.actuator_length[2] - l1) * model.opt.timestep
    r_spindle = 0.05 * data.actuator_velocity[1] + data.actuator_length[1]
    l_spindle = 0.05 * data.actuator_velocity[2] + data.actuator_length[2]
    data.ctrl[1] = kp * (r_spindle - l0) + kd * data.actuator_velocity[1] + ki * e0_r
    data.ctrl[2] = kp * (l_spindle - l1) + kd * data.actuator_velocity[2] + ki * e0_l
    print(data.qpos[0])

e1_r = 0
e1_l = 0
# def spinal_controller(model, data):
#     # global e1_r
#     # global e1_l
#     # kp = 50
#     # kd = 8
#     # ki = 8
#     # gl1, gl0 = get_length_from_angle(-0.45)
#     # e1_r += (data.actuator_length[1] - gl0) * model.opt.timestep
#     # e1_l += (data.actuator_length[2] - gl1) * model.opt.timestep
#     # mb = 0.1
#     # l0 = max(mb - (kp * (data.actuator_length[1] - gl0) + kd * data.actuator_velocity[1] + ki * e1_r), 0)
#     # l1 = max(mb - (kp * (data.actuator_length[2] - gl1) + kd * data.actuator_velocity[2] + ki * e1_l), 0)

def baseline_callback(model, data):
    if env_id == 0:
        obs = np.concatenate((target_pos, data.xpos[1], np.array([data.qvel[0]])))
        action, _states = PPO_model0.predict(obs)
        spinal_controllers.baseline_controller(input_action=action, data=data)
        spinal_controllers.joint0_controller(model, data)
        print(data.qpos[0], data.ctrl[1], data.ctrl[2])
    elif env_id == 1:
        # obs = np.concatenate((target_pos, data.xpos[1], data.xpos[2], np.array([data.qpos[0], data.qpos[1], data.qvel[0], data.qvel[1]])))
        # obs = np.array([m_target[0], m_target[1], data.qpos[0], data.qpos[1], data.qvel[0], data.qvel[1]])
        obs = np.array([m_target[0], m_target[1], data.qpos[0], data.qvel[0], data.qpos[1], data.qvel[1], 0, 0])
        if RL_mode == PPO_MODE:
            action, _states = PPO_model0.predict(obs)
        elif RL_mode == TD3_MODE:
            action, _states = TD3_model0.predict(obs)
        double_link_controllers.baseline_controller(input_action=action, data=data)
        # double_link_controllers.joints_controller(data)
        # print(data.xpos[2])
        print(data.qpos[0], data.qpos[1])

    elif env_id == 2:
        obs = np.array([data.qpos[0], data.qpos[1], data.qpos[2], data.qvel[0], data.qvel[1], data.qvel[2]])
        action, _states = PPO_model0.predict(obs)
        double_link_controllers.baseline_controller(input_action=action, data=data)

def neuron_filter_callback(model, data):
    obs = np.concatenate((target_pos, data.xpos[1], np.array([data.qvel[0]])))
    action, _states = PPO_model2.predict(obs)
    spinal_controllers.stretch_reflex_controller(input_action=action, data=data)
    spinal_controllers.joint0_controller(model, data)
    print(data.qpos[0])

def neuron_callback(model, data):
    if env_id == 0:
        obs = np.concatenate((target_pos, data.xpos[1], np.array([data.qvel[0]])))
        action, _states = PPO_model4.predict(obs)
        spinal_controllers.neuron_controller(input_action=action, data=data)
        spinal_controllers.joint0_controller(model, data)
        print(data.qpos[0], action[0], action[1], action[2], action[3])
    elif env_id == 1:
        # obs = np.array([m_target[0], m_target[1], data.qpos[0], data.qpos[1], data.qvel[0], data.qvel[1]])
        obs = np.array([m_target[0], m_target[1], data.qpos[0], data.qvel[0], data.qpos[1], data.qvel[1], 0, 0])
        #obs =  np.concatenate((target_pos, data.xpos[1], data.xpos[2], np.array([data.qpos[0], data.qpos[1], data.qvel[0], data.qvel[1]])))
        action, _states = PPO_model4.predict(obs)
        double_link_controllers.neuron_controller(input_action=action, data=data)
        print(data.qpos[0], data.qpos[1])

    elif env_id == 2:
        obs = np.array([data.qpos[0], data.qpos[1], data.qpos[2], data.qvel[0], data.qvel[1], data.qvel[2]])
        action, _states = PPO_model4.predict(obs)
        double_link_controllers.neuron_controller(input_action=action, data=data)

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

#load modes for each controller
w = -0.48
# m_target = np.array([0, 0])
m_target = np.array([-0.82, 0.65])
if control_type == spinal_controllers.Control_Type.BASELINE:
    if env_id == 0:
        target_pos = compute_target_pos(w, 1)
        PPO_model_path0 = "models/1687332383/10650000.zip"
        PPO_model0 = PPO.load(PPO_model_path0)
    elif env_id == 1:
        if RL_mode == PPO_MODE:
            # PPO_model_path0 = "..\\RL_data\\neuron-training-stable\\models\\1687820950\\39520000.zip"
            PPO_model_path0 =  "models\\1690832298\\1936000.zip"
            PPO_model0 = PPO.load(PPO_model_path0)
        elif RL_mode == TD3_MODE:
            TD3_model_path0 = "models\\1690085218\\4760000.zip"
            TD3_model0 = TD3.load(TD3_model_path0)
    elif env_id == 2:
        # PPO_model_path0 = "..\\RL_data\\first_working_inverted_pendulum\\models\\1690272718\\2590000.zip"
        PPO_model_path0 = "models\\1690921461\\2400000.zip"
        PPO_model0 = PPO.load(PPO_model_path0)


if control_type == spinal_controllers.Control_Type.REFLEX:
    PPO_model_path2="models/1686530946/3980000.zip"
    PPO_model2=PPO.load(PPO_model_path2)

if control_type == spinal_controllers.Control_Type.NEURON:
    if env_id == 0:
        PPO_model_path4 = "models/1687587323/19360000.zip"
        PPO_model4 = PPO.load(PPO_model_path4)
        target_pos = compute_target_pos(w, 1)
    elif env_id == 1:
        PPO_model_path4 = "models/1690581473/9040000.zip"
        # PPO_model_path4 = "..\\RL_data\\neuron-training-stable\\models\\1687913383\\56960000.zip"
        PPO_model4 = PPO.load(PPO_model_path4)
        data.qpos[0] = m_target[0]
        data.qpos[1] = m_target[1]
        mj.mj_forward(model, data)
        target_pos = data.xpos[2].copy()
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)
    elif env_id == 2:
        PPO_model_path4 = "..\\RL_data\\first_working_inverted_pendulum\\models\\1690274397\\2130000.zip"
        PPO_model4 = PPO.load(PPO_model_path4)

#initialize the controller
init_controller(model,data)
#set the controller
if control_type == spinal_controllers.Control_Type.BASELINE:
    mj.set_mjcb_control(baseline_callback)
elif control_type == spinal_controllers.Control_Type.NEURON_FILTER:
    mj.set_mjcb_control(neuron_filter_callback())
elif control_type == spinal_controllers.Control_Type.NEURON:
    mj.set_mjcb_control(neuron_callback)

while not glfw.window_should_close(window):
    time_prev = data.time

    while (data.time - time_prev < 1.0/60.0):
        mj.mj_step(model, data)
    # if (data.time>=simend):
    #     break;

    #print camera configuration (help to initialize the view)
    if print_camera_config==1:
        print('cam.azimuth =',cam.azimuth,';','cam.elevation =',cam.elevation,';','cam.distance = ',cam.distance)
        print('cam.lookat =np.array([',cam.lookat[0],',',cam.lookat[1],',',cam.lookat[2],'])')

    # Update scene and render
    mj.mjv_updateScene(model, data, opt, None, cam,
                       mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)

    # swap OpenGL buffers (blocking call due to v-sync)
    glfw.swap_buffers(window)

    # process pending GUI events, call GLFW callbacks
    glfw.poll_events()

glfw.terminate()
