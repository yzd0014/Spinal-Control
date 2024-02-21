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
time_data = []

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

    modelid = '1708486874'

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

    controller = InitController(control_type, controller_params)
    mj.set_mjcb_control(callback)

    generate_initial_state()
    states_tracking = []
    avg_balance_time = 0
    loop_count = 5
    for i in range(loop_count):
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
                if data.time > 10 or abs(abs(sum(data.qpos)) - np.pi) > 0.5*np.pi:
                    if data.time >= 10:
                        states_tracking.append(states_count)
                    avg_balance_time += data.time
                    break
            print(f"iter: {i}, state {states_count} done, time: {data.time}")
            time_data.append(data.time)
            states_count += 1

    total_states = len(initial_state) * loop_count
    avg_balance_time /= total_states
    plt.figure(1)
    plt.hist(time_data, bins=10, color='skyblue', edgecolor='black')
    plt.xlabel('time to balance (s)')
    plt.ylabel('Frequency')
    plt.title(f"{control_type_dic[control_type]}, avg time: {avg_balance_time}")
    plt.figure(2)
    plt.hist(states_tracking, bins=200, color='skyblue', edgecolor='black')
    plt.xlabel('initial configuration #')
    plt.show()

if __name__ == "__main__":
    main()