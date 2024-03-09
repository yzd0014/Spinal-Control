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
    for i in np.arange(0.2, 0.7, 0.03):
        for j in np.arange(i, 0.6, 0.02):
                initial_state.append([j, i])

def callback(model, data):
    pass

def init_controller(model, data):
    pass

def main():
    logdir = "training_data/"
    # file = open(logdir + "training_data.txt", "w")
    file = open(logdir + "testing_data.txt", "w")

    xml_path = 'single_link.xml'
    dirname = os.path.dirname(__file__)
    abspath = os.path.join(dirname + "/" + xml_path)
    xml_path = abspath
    model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
    data = mj.MjData(model)  # MuJoCo data

    # mj.set_mjcb_control(callback)
    generate_initial_state()

    print(f"total number: {len(initial_state)}")
    file.write(f"{len(initial_state)}\n")
    i = 0
    for state in initial_state:
        mj.mj_resetData(model, data)
        data.ctrl[0] = initial_state[i][0]
        data.ctrl[1] = initial_state[i][1]
        mj.mj_forward(model, data)
        while data.time < 10:
            mj.mj_step(model, data)
        file.write(f"{data.ctrl[0]} { data.ctrl[1]} {data.qpos[0]}\n")
        print(f"write {i} to file")
        i += 1
    file.close()
if __name__ == "__main__":
    main()