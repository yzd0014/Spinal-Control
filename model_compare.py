from stable_baselines3 import PPO
import mujoco as mj
import os
from spinal_controllers import *
import matplotlib.pyplot as plt

model_is_avaialble = [True, True, True, True, True]
try:
    PPO_model_path0="models/1687332383/10650000.zip"
    PPO_model0=PPO.load(PPO_model_path0)
except FileNotFoundError:
    print("path is not correct")
    model_is_avaialble[0] = False
else:
    def baseline_callback(model, data):
        obs = np.concatenate((target_pos, data.xpos[1], np.array([data.qvel[0]])))
        action, _states = PPO_model0.predict(obs)
        baseline_controller(input_action=action, data=data)
        # print(data.qpos[0])

try:
    PPO_model_path1="na"
    PPO_model1=PPO.load(PPO_model_path1)
except FileNotFoundError:
    print("path is not correct")
    model_is_avaialble[1] = False
else:
    def RI_callback(model, data):
        obs = np.concatenate((target_pos, data.xpos[1], np.array([data.qvel[0]])))
        action, _states = PPO_model1.predict(obs)
        RI_controller(input_action=action, data=data)
        # print(data.qpos[0])
try:
    PPO_model_path2="models/1685057485/1850000.zip"
    PPO_model2=PPO.load(PPO_model_path2)
except FileNotFoundError:
    print("path is not correct")
    model_is_avaialble[2] = False
else:
    def stretch_reflex_callback(model, data):
        obs = np.concatenate((target_pos, data.xpos[1], np.array([data.qvel[0]])))
        action, _states = PPO_model2.predict(obs)
        stretch_reflex_controller(input_action=action, data=data)
        # print(data.qpos[0])

try:
    PPO_model_path3="models/1685079447/1850000.zip"
    PPO_model3=PPO.load(PPO_model_path3)
except FileNotFoundError:
    print("path is not correct")
    model_is_avaialble[3] = False
else:
    def RI_and_stretch_reflex_callback(model, data):
        obs = np.concatenate((target_pos, data.xpos[1], np.array([data.qvel[0]])))
        action, _states = PPO_model3.predict(obs)
        RI_and_stretch_reflex_controller(input_action=action, data=data)
        # print(data.qpos[0])

try:
    PPO_model_path4="models/1687587323/8050000.zip"
    PPO_model4=PPO.load(PPO_model_path4)
except FileNotFoundError:
    print("path is not correct")
    model_is_avaialble[4] = False
else:
    def neuron_callback(mode, data):
        obs = np.concatenate((target_pos, data.xpos[1], np.array([data.qvel[0]])))
        action, _states = PPO_model4.predict(obs)
        for i in range(4):
            RI_cmd[i].append(action[i]) #save the RI command
        neuron_controller(input_action=action, data=data)

#get the full path
xml_path = 'muscle_control_narrow.xml' #xml file (assumes this is in the same folder as this file)
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)
xml_path = abspath

model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
data = mj.MjData(model)                # MuJoCo data

w = -0.32
#forward kinamtics to get the target position
data.qpos[0] = w
mj.mj_forward(model, data)
target_pos = data.xpos[1].copy()

target_sim_time = 5
x_time = []
y0 = []
y1 = []
y2 = []
y3 = []
y4 = []
RI_cmd = [[],[],[],[]]

if model_is_avaialble[0]:
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)
    mj.set_mjcb_control(baseline_callback)
    while data.time < target_sim_time:
        mj.mj_step(model, data)
        x_time.append(data.time)
        y0.append(data.qpos[0])
    print("y0 done!")

if model_is_avaialble[1]:
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)
    mj.set_mjcb_control(RI_callback)
    while data.time < target_sim_time:
        mj.mj_step(model, data)
        y1.append(data.qpos[0])
    print("y1 done!")

if model_is_avaialble[2]:
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)
    mj.set_mjcb_control(stretch_reflex_callback)
    while data.time < target_sim_time:
        mj.mj_step(model, data)
        y2.append(data.qpos[0])
    print("y2 done!")

if model_is_avaialble[3]:
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)
    mj.set_mjcb_control(RI_and_stretch_reflex_callback)
    while data.time < target_sim_time:
        mj.mj_step(model, data)
        y3.append(data.qpos[0])
    print("y3 done!")

if model_is_avaialble[4]:
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)
    mj.set_mjcb_control(neuron_callback)
    while data.time < target_sim_time:
        mj.mj_step(model, data)
        y4.append(data.qpos[0])
    print("y4 done!")

# plt.subplot(1, 2, 1)
if model_is_avaialble[0]:
    plt.plot(x_time, y0, label = "baseline")
if model_is_avaialble[1]:
    plt.plot(x_time, y1, label = "RI")
if model_is_avaialble[2]:
    plt.plot(x_time, y2, label = "stretch reflex")
if model_is_avaialble[3]:
    plt.plot(x_time, y3, label = "RI + stretch reflex")
if model_is_avaialble[4]:
    plt.plot(x_time, y4, label = "RI + stretch reflex with neurons")
plt.axhline(y = w, color = 'r', linestyle = '-', label = "target position", linewidth = 0.2)
plt.xlabel('time')
plt.ylabel('position')

# plt.subplot(1, 2, 2)
# if model_is_avaialble[4]:
    # plt.plot(x_time, RI_cmd[0], label="alpha-r")
    # plt.plot(x_time, RI_cmd[1], label="alpha-l")
    # plt.plot(x_time, RI_cmd[2], label="internueron-r")
    # plt.plot(x_time, RI_cmd[3], label="internueron-l")

plt.legend()
plt.show()
