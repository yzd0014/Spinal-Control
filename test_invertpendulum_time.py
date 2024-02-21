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

dt_brain = 0
controller = None
rl_model = None
global_timer = 0
initial_state = []

def generate_initial_state():
    global initial_state
    for i in np.arange(-0.3, 0.3, 0.1):
        for j in np.arange(-0.3, 0.3, 0.1):
            for k in np.arange(np.pi - 0.7, np.pi + 0.7, 0.2):
                initial_state.append([i, j, k])

def callback(model, data):
    global global_timer
    if data.time - global_timer >= dt_brain:
        observation = np.array([data.qpos[0], data.qpos[1], data.qpos[2], data.qvel[0], data.qvel[1], data.qvel[2]])
        action, _states = rl_model.predict(observation)
        controller.set_action(action)
        global_timer = data.time

    controller.callback(model, data)

def main():
    global dt_brain, controller, rl_model, global_timer

    modelid = '1708030061'

    control_type, \
        episode_length, \
        num_episodes, \
        fs_brain_factor, \
        controller_params = pickle.load(open("./models/" + modelid + "/" \
                                             + "env_contr_params.p", "rb"))

    dt_brain = (1.0 / controller_params.fs) * fs_brain_factor

    models_dir = "./models/" + modelid + "/"
    allmodels = sorted(os.listdir(models_dir))
    allmodels.sort(key=lambda fn: \
        os.path.getmtime(os.path.join(models_dir, fn)))

    runid = allmodels[-1].split(".")
    runid = runid[0]
    rl_model_path = "./models/" + modelid + "/" + runid

    if controller_params.RL_type == "PPO":
        rl_model = PPO.load(rl_model_path)
    elif controller_params.RL_type == "SAC":
        rl_model = SAC.load(rl_model_path)

    xml_path = 'inverted_pendulum_fast.xml'
    dirname = os.path.dirname(__file__)
    abspath = os.path.join(dirname + "/" + xml_path)
    xml_path = abspath
    model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
    data = mj.MjData(model)  # MuJoCo data
    cam = mj.MjvCamera()  # Abstract camera
    opt = mj.MjvOption()
    mj.mjv_defaultCamera(cam)
    mj.mjv_defaultOption(opt)

    # glfw.init()
    # window = glfw.create_window(1200, 900, "Demo", None, None)
    # glfw.make_context_current(window)
    # glfw.swap_interval(1)
    # # get framebuffer viewport
    # viewport_width, viewport_height = glfw.get_framebuffer_size(window)
    # viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)
    # scene = mj.MjvScene(model, maxgeom=10000)
    # context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

    cam.azimuth = 90
    cam.elevation = -20
    cam.distance = 2
    cam.lookat = np.array([0.0, -1, 2])


    controller = InitController(control_type, controller_params)
    mj.set_mjcb_control(callback)

    generate_initial_state()
    avg_balance_time = 0
    states_count = 0
    for state in initial_state:

        mj.mj_resetData(model, data)
        data.qpos[0] = state[0]
        data.qpos[1] = state[1]
        data.qpos[2] = state[2] - state[0] - state[1]
        mj.mj_forward(model, data)
        global_timer = data.time

        while True:
            mj.mj_step(model, data)
            if data.time > 20 or abs(abs(sum(data.qpos)) - np.pi) > 0.5*np.pi:
                avg_balance_time += data.time
                break
        print(f"State {states_count} done, time: {data.time}")
        states_count += 1

    total_states = len(initial_state)
    avg_balance_time /= total_states
    print(f"Average balance time: {avg_balance_time}")

    # mj.mj_resetData(model, data)
    # data.qpos[0] = initial_state[0][0]
    # data.qpos[1] = initial_state[0][1]
    # data.qpos[2] = initial_state[0][2] - initial_state[0][0] - initial_state[0][1]
    # mj.mj_forward(model, data)
    # global_timer = data.time
    # total_ctrl = 0
    # while data.time < 50:
    #     mj.mj_step(model, data)
    #     for i in range(4):
    #         ctrl = 0
    #         if data.ctrl[i] > 1:
    #             ctrl = 1
    #         elif data.ctrl[i] < 0:
    #             ctrl = 0
    #         else:
    #             ctrl = data.ctrl[i]
    #         total_ctrl += ctrl
    #
    # print(total_ctrl)


    # mj.mj_resetData(model, data)
    # data.qpos[2] = np.pi
    # mj.mj_forward(model, data)
    # global_timer = data.time
    # while not glfw.window_should_close(window):
    #     time_prev = data.time
    #     while (data.time - time_prev < 1.0/60.0):
    #         mj.mj_step(model, data)
    #
    #     # Update scene and render
    #     mj.mjv_updateScene(model, data, opt, None, cam,
    #                        mj.mjtCatBit.mjCAT_ALL.value, scene)
    #     mj.mjr_render(viewport, scene, context)
    #
    #     # swap OpenGL buffers (blocking call due to v-sync)
    #     glfw.swap_buffers(window)
    #
    #     # process pending GUI events, call GLFW callbacks
    #     glfw.poll_events()
    #
    # glfw.terminate()

if __name__ == "__main__":
    main()