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
model_ctrl = None
time_tracking = None

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

    # modelid = '1708034438'

    # baseline
    modelid = '1708068818'
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
    # cam = mj.MjvCamera()  # Abstract camera
    # opt = mj.MjvOption()
    # mj.mjv_defaultCamera(cam)
    # mj.mjv_defaultOption(opt)
    #
    # cam.azimuth = 90
    # cam.elevation = -20
    # cam.distance = 2
    # cam.lookat = np.array([0.0, -1, 2])

    controller = InitController(control_type, controller_params)
    mj.set_mjcb_control(callback)

    generate_initial_state()
    mj.mj_resetData(model, data)
    data.qpos[0] = initial_state[0][0]
    data.qpos[1] = initial_state[0][1]
    data.qpos[2] = initial_state[0][2] - initial_state[0][0] - initial_state[0][1]
    mj.mj_forward(model, data)

    sim_time = 50
    sz = (int)(sim_time / model.opt.timestep)
    model_ctrl = np.zeros((4, sz))
    time_tracking = np.zeros(sz)
    global_timer = data.time
    total_ctrl = 0
    iter = 0
    while data.time < sim_time:
        mj.mj_step(model, data)
        if abs(abs(sum(data.qpos)) - np.pi) > 0.5*np.pi:
            print("Fell down")
            print(data.time)
            break

        for i in range(4):
            ctrl = 0
            if data.ctrl[i] > 1:
                ctrl = 1
            elif data.ctrl[i] < 0:
                ctrl = 0
            else:
                ctrl = data.ctrl[i]
            total_ctrl += ctrl
            if iter < sz:
                model_ctrl[i, iter] = ctrl

        if iter < sz:
            time_tracking[iter] = data.time
        iter = iter + 1

    plt.plot(time_tracking, model_ctrl[0, :], label="ctrl 1")
    plt.plot(time_tracking, model_ctrl[1, :], label="ctrl 2")
    plt.plot(time_tracking, model_ctrl[2, :], label="ctrl 3")
    plt.plot(time_tracking, model_ctrl[3, :], label="ctrl 4")
    plt.title(f"{control_type_dic[control_type]}, total control: {total_ctrl}")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()