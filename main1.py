import numpy as np
from stable_baselines3 import PPO
import torch
import torch_net
from mujoco.glfw import glfw
import pickle
from control import *

m_target = np.array([-0.25, -0.9])
# m_target = np.array([-10, 0])
cocontraction = 0
modelid = "1702363073"
#######################################################################
# Load Params
print("\n\n")
print("loading env and control parameters " + "./models/" + modelid + "\n")

training_type, control_type, env_id, controller_params = pickle.load(open("./models/" + modelid + "/" \
                                         + "env_contr_params.p", "rb"))
episode_length = controller_params.episode_length_in_ticks
dt_brain = controller_params.brain_dt

# For saving data
data_dir = "datalog"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
fdata = open(f"{data_dir}/{modelid}", 'w')
# fdata = open("./datalog/" + modelid, 'w')

# Find most recent model
models_dir = "./models/" + modelid + "/"
allmodels = sorted(os.listdir(models_dir))
allmodels.sort(key=lambda fn: \
    os.path.getmtime(os.path.join(models_dir, fn)))

#load feedforward model
feedforward_model_path0 = f"./models/{modelid}/{allmodels[-1]}"
ff_net = torch_net.FeedForwardNN(controller_params.input_size, controller_params.hidden_size,
                                 controller_params.output_size, control_type)
ff_net.load_state_dict(torch.load(feedforward_model_path0))
ff_net.eval()
xml_path = controller_params.model_dir

sim_pause = True
next_frame = False
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
controller = TemplateController(controller_params, env_id)

def keyboard(window, key, scancode, act, mods):
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)
        # init_controller(model, data)

    if act == glfw.PRESS and key == glfw.KEY_SPACE:
        global sim_pause
        sim_pause = not sim_pause

    if act == glfw.PRESS and key == glfw.KEY_RIGHT:
        global next_frame
        next_frame = True

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
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)
    controller.target_pos = np.array([m_target[0], m_target[1]])

ep_error = 0
def callback(model, data):
    global global_timer, ep_error
    if data.time - global_timer >= dt_brain or data.time < 0.000101:
        observation = controller.get_obs(data)
        observation_tensor = torch.tensor(observation, requires_grad=False, dtype=torch.float32)
        u_tensor = ff_net(observation_tensor.view(1, controller_params.input_size))
        u = np.zeros(controller_params.output_size)
        for i in range(controller_params.output_size):
            u[i] = u_tensor[0][i].item()
        controller.set_action(u)
        for i in range(2):
            if controller.target_pos[i] >= 0:
                data.ctrl[i * 2 + 1] = cocontraction
            else:
                data.ctrl[i * 2] = cocontraction
        global_timer = data.time

    controller.callback(model, data)
    # print(data.qpos)
    print(data.ctrl)


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

#initialize the controller
init_controller(model,data)
#set the controller
mj.set_mjcb_control(callback)

global_timer = data.time
while not glfw.window_should_close(window):
    if sim_pause == False or next_frame == True:
        time_prev = data.time
        while (data.time - time_prev < 1.0/60.0):
            mj.mj_step(model, data)
        next_frame = False
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