from stable_baselines3 import PPO
import torch
import torch_net
from mujoco.glfw import glfw
import pickle
from control import *

m_target = np.array([0.4, 0.4])
# modelid = "1696546724"
modelid = "1696546010"
#######################################################################
# Load Params
print("\n\n")
print("loading env and control parameters " + "./models/" + modelid + "\n")

training_type, control_type, controller_params = pickle.load(open("./models/" + modelid + "/" \
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

if training_type == "PPO":
    runid = allmodels[-1].split(".")
    runid = runid[0]
    PPO_model_path0 = "./models/" + modelid + "/" + runid
    PPO_model = PPO.load(PPO_model_path0)
elif training_type == "feedforward":
    feedforward_model_path0 = f"./models/{modelid}/{allmodels[-1]}"
    ff_net = torch_net.FeedForwardNN(controller_params.input_size, controller_params.hidden_size, controller_params.output_size)
    ff_net.load_state_dict(torch.load(feedforward_model_path0))
    ff_net.eval()
#######################################################################
# dt_brain = 0.05
# PPO_model = None
# fdata = None
# control_type = Control_Type.PID
# controller_params = ControllerParams(alpha=0.4691358024691358, \
#                                     beta=0.9, \
#                                     gamma=1, \
#                                     fc=10, \
#                                     episode_length_in_seconds=2.5,\
#                                     brain_dt=0.05)

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
    if training_type == "PPO":
        controller = BaselineController(controller_params)
# Optimal neuron Controller
elif control_type == Control_Type.NEURON_OPTIMAL:
    controller = SpinalOptimalController(controller_params)
elif control_type == Control_Type.PID:
    controller = PIDController()

def keyboard(window, key, scancode, act, mods):
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)
        # init_controller(model, data)

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
    pass

ep_error = 0
def callback(model, data):
    global global_timer, ep_error
    if data.time - global_timer >= dt_brain:
        if control_type == Control_Type.NEURON_OPTIMAL:
            controller.set_action(np.array([m_target[0], m_target[1], 0.5, 0.5]))
        elif control_type == Control_Type.PID:
            controller.set_action(m_target)
        else:
            if training_type == "PPO":
                observation = np.concatenate(([m_target[0], m_target[1]], \
                                              controller.obs, \
                                              np.array([0, 0])))
                action, _states = PPO_model.predict(observation)
                controller.set_action(action)
            elif training_type == "feedforward":
                observation = np.array([m_target[0], m_target[1], data.qpos[0], data.qvel[0], data.qpos[1], data.qvel[1]])
                observation_tensor = torch.tensor(observation, requires_grad=False, dtype=torch.float32)
                u_tensor = ff_net(observation_tensor.view(1, 6))
                for i in range(4):
                    data.ctrl[i] = u_tensor[0][i].item()

        # if data.time <= 2.5:
        #     position_error = -np.linalg.norm(data.qpos - m_target)
        #     ep_error += position_error
        #     print(ep_error)
        # else:
        #     print(ep_error)
        global_timer = data.time

    if training_type == "PPO":
        controller.callback(model, data)
    # print(controller.l_desired)
    # print(data.ctrl[0], data.ctrl[1])
    # print(data.qpos[0], data.qpos[1])


    # if control_type != Control_Type.NEURON_OPTIMAL and control_type != Control_Type.PID:
    #     data2write = np.concatenate(([m_target[0], m_target[1]], \
    #                                  data.qpos, \
    #                                  controller.obs, \
    #                                  data.actuator_length, \
    #                                  data.ctrl, \
    #                                  controller.action))
    #     datastr = ','.join(str(x) for x in data2write)
    #     fdata.write(datastr + '\n')


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
