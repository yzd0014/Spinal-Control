from stable_baselines3 import PPO
import mujoco as mj
import os
import spinal_controllers
import double_link_controllers
import matplotlib.pyplot as plt
import numpy as np

env_id = 0
model_is_avaialble = [True, True, True, False, True]
try:
    if env_id == 0:
        PPO_model_path0 = "models/1693873419/1370000.zip"
    else:
        # PPO_model_path0="..\\RL_data\\neuron-training-stable\\models\\1687820950\\39520000.zip"
        PPO_model_path0 =  "models/1693531574/4328000.zip"
    PPO_model0=PPO.load(PPO_model_path0)
except FileNotFoundError:
    print("path 0 is not correct")
    model_is_avaialble[0] = False
else:
    def baseline_callback(model, data):
        if env_id == 0:
            obs = np.array([0, 0.1, 0.6, 0.06])
            action, _states = PPO_model0.predict(obs)
            spinal_controllers.baseline_controller(input_action=action, data=data)
            # print(data.qpos[0])
        elif env_id == 1:
            # obs = np.array([m_target[0], m_target[1], data.qpos[0], data.qvel[0], data.qpos[1], data.qvel[1], 0, 0])
            obs = np.array([m_target[0], m_target[1], data.qpos[0], data.qpos[1], data.qvel[0], data.qvel[1]])
            action, _states = PPO_model0.predict(obs)
            double_link_controllers.baseline_controller(input_action=action, data=data)

try:
    PPO_model_path1= "na"
    PPO_model1=PPO.load(PPO_model_path1)
except FileNotFoundError:
    print("path 1 is not correct")
    model_is_avaialble[1] = False
else:
    def RI_callback(model, data):
        if env_id == 0:
            obs = np.concatenate((target_pos, data.xpos[1], np.array([data.qvel[0]])))
            action, _states = PPO_model1.predict(obs)
            spinal_controllers.RI_controller(input_action=action, data=data)
            # print(data.qpos[0])
        elif env_id == 1:
            obs = np.array([m_target[0], m_target[1], data.qpos[0], data.qvel[0], data.qpos[1], data.qvel[1], 0, 0])
            action, _states = PPO_model1.predict(obs)
            double_link_controllers.baseline_controller(input_action=action, data=data)
try:
    if env_id == 0:
        PPO_model_path2="models/1693874233/607500.zip"
    else:
        PPO_model_path2="models/1693584791/880000.zip"
    PPO_model2=PPO.load(PPO_model_path2)
except FileNotFoundError:
    print("path 2 is not correct")
    model_is_avaialble[2] = False
else:
    def stretch_reflex_callback(model, data):
        if env_id == 0:
            obs = np.array([0, 0.1, 0.6, 0.06])
            action, _states = PPO_model2.predict(obs)
            spinal_controllers.stretch_reflex_controller(input_action=action, data=data)
        elif env_id == 1:
            # obs = np.concatenate((target_pos, data.xpos[1], np.array([data.qvel[0]])))
            obs = np.array([m_target[0], m_target[1], data.qpos[0], data.qpos[1], data.qvel[0], data.qvel[1]])
            action, _states = PPO_model2.predict(obs)
            spinal_controllers.stretch_reflex_controller(input_action=action, data=data)
            # print(data.qpos[0])

try:
    # PPO_model_path4="models/1693531592/4048000.zip"
    PPO_model_path4 = "na"
    PPO_model4=PPO.load(PPO_model_path4)
except FileNotFoundError:
    print("path 3 is not correct")
    model_is_avaialble[4] = False
else:
    def neuron_callback(mode, data):
        if env_id == 0:
            obs = np.concatenate((target_pos, data.xpos[1], np.array([data.qvel[0]])))
            action, _states = PPO_model4.predict(obs)
            for i in range(4):
                RI_cmd[i].append(action[i]) #save the RI command
            spinal_controllers.neuron_controller(input_action=action, data=data)
        elif env_id == 1:
            # obs = np.array([m_target[0], m_target[1], data.qpos[0], data.qvel[0], data.qpos[1], data.qvel[1], 0, 0])
            obs = np.array([m_target[0], m_target[1], data.qpos[0], data.qpos[1], data.qvel[0], data.qvel[1]])
            action, _states = PPO_model4.predict(obs)
            double_link_controllers.neuron_controller(input_action=action, data=data)
def init_controller(model,data):
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)

    if env_id == 0:
        data.qvel[0] = 2
    elif env_id == 2:
        data.qpos[0] = 0.4
        data.qpos[1] = -0.87
        data.qpos[2] = -2.32

    mj.mj_forward(model, data)

#get the full path
if env_id == 0:
    xml_path = 'muscle_control_narrow.xml'
elif env_id == 1:
    xml_path = 'double_links_fast.xml' #xml file (assumes this is in the same folder as this file)
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)
xml_path = abspath

model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
data = mj.MjData(model)                # MuJoCo data

m_target = np.array([0.55, -0.62])
w = -0.32
#forward kinamtics to get the target position
data.qpos[0] = w
mj.mj_forward(model, data)
target_pos = data.xpos[1].copy()

target_sim_time = 5
x_time = []
baseline_result = [[],[]]
y1 = []
reflex_result = [[],[]]
y3 = []
neuron_result = [[],[]]

RI_cmd = [[],[],[],[]]
baseline_ctrl = [[],[],[],[]]
neuron_ctrl = [[],[],[],[]]
reflex_ctrl = [[],[],[],[]]

baseline_total_energy = 0
neuron_total_energy = 0
reflex_total_energy = 0

if model_is_avaialble[0]:
    init_controller(model, data)
    mj.set_mjcb_control(baseline_callback)
    while data.time < target_sim_time:
        mj.mj_step(model, data)
        x_time.append(data.time)

        if env_id == 0:
            baseline_result[0].append(0 - data.qpos[0])
            for i in range(2):
                baseline_ctrl[i].append(data.ctrl[i+1])
                baseline_total_energy += data.ctrl[i+1]
        else:
            baseline_result[0].append(m_target[0] - data.qpos[0])
            baseline_result[1].append(m_target[1] - data.qpos[1])
            for i in range(4):
                baseline_ctrl[i].append(data.ctrl[i])
                baseline_total_energy += data.ctrl[i]

    print("y0 done!")

if model_is_avaialble[1]:
    init_controller(model, data)
    mj.set_mjcb_control(RI_callback)
    while data.time < target_sim_time:
        mj.mj_step(model, data)
        neuron_result[0].append(m_target[0] - data.qpos[0])
        neuron_result[1].append(m_target[1] - data.qpos[1])
        for i in range(4):
            neuron_ctrl[i].append(data.ctrl[i])  # save the neuron command
            neuron_total_energy += data.ctrl[i]
    print("y1 done!")

if model_is_avaialble[2]:
    init_controller(model, data)
    mj.set_mjcb_control(stretch_reflex_callback)
    while data.time < target_sim_time:
        mj.mj_step(model, data)

        if env_id == 0:
            reflex_result[0].append(0 - data.qpos[0])
            for i in range(2):
                reflex_ctrl[i].append(data.ctrl[i + 1])
                reflex_total_energy += data.ctrl[i + 1]
        else:
            reflex_result[0].append(m_target[0] - data.qpos[0])
            reflex_result[1].append(m_target[1] - data.qpos[1])
            for i in range(4):
                reflex_ctrl[i].append(data.ctrl[i])
                reflex_total_energy += data.ctrl[i]

    print("y2 done!")

if model_is_avaialble[4]:
    init_controller(model, data)
    mj.set_mjcb_control(neuron_callback)
    while data.time < target_sim_time:
        mj.mj_step(model, data)
        neuron_result[0].append(m_target[0] - data.qpos[0])
        neuron_result[1].append(m_target[1] - data.qpos[1])
        for i in range(4):
            neuron_ctrl[i].append(data.ctrl[i])  # save the neuron command
            neuron_total_energy += data.ctrl[i]
    print("y4 done!")

plt.figure(1)
if env_id == 0:
    if model_is_avaialble[0]:
        plt.plot(x_time, baseline_result[0], label = "baseline-0.1")
    if model_is_avaialble[1]:
        plt.plot(x_time, neuron_result[0], label = "baseline-1")
    if model_is_avaialble[2]:
        plt.plot(x_time, reflex_result[0], label = "stretch reflex")
    if model_is_avaialble[3]:
        plt.plot(x_time, y3, label = "RI + stretch reflex")
    if model_is_avaialble[4]:
        plt.plot(x_time, neuron_result[0], label="neuron-0.1")
    plt.xlabel('time')
    plt.ylabel('position error')
    plt.legend()
else:
    plt.subplot(1, 2, 1)
    if model_is_avaialble[0]:
        plt.plot(x_time, baseline_result[0], label = "baseline-0.1")
    if model_is_avaialble[1]:
        plt.plot(x_time, neuron_result[0], label = "baseline-1")
    if model_is_avaialble[2]:
        plt.plot(x_time, reflex_result[0], label = "stretch reflex")
    if model_is_avaialble[3]:
        plt.plot(x_time, y3, label = "RI + stretch reflex")
    if model_is_avaialble[4]:
        plt.plot(x_time, neuron_result[0], label = "neuron-0.1")
    # plt.axhline(y = w, color = 'r', linestyle = '-', label = "target position", linewidth = 0.2)
    plt.xlabel('time')
    plt.ylabel('joint 0 position')
    plt.legend()

    plt.subplot(1, 2, 2)
    if model_is_avaialble[0]:
        plt.plot(x_time, baseline_result[1], label = "baseline-0.1")
    if model_is_avaialble[1]:
        plt.plot(x_time, neuron_result[1], label = "baseline-1")
    if model_is_avaialble[2]:
        plt.plot(x_time, reflex_result[1], label = "stretch reflex")
    if model_is_avaialble[3]:
        plt.plot(x_time, y3, label = "RI + stretch reflex")
    if model_is_avaialble[4]:
        plt.plot(x_time, neuron_result[1], label = "neuron-0.1")
    # plt.axhline(y = w, color = 'r', linestyle = '-', label = "target position", linewidth = 0.2)
    plt.xlabel('time')
    plt.ylabel('joint 1 position')
    plt.legend()

print(f"baseline total energy: {baseline_total_energy}")
print(f"neuron total energy: {reflex_total_energy}")
plt.figure(2)
if env_id == 0:
    for i in range(2):
        plt.subplot(2, 1, i+1)
        if model_is_avaialble[0]:
            plt.plot(x_time, baseline_ctrl[i], label="baseline")
        if model_is_avaialble[2]:
            plt.plot(x_time, reflex_ctrl[i], label="stretch-reflex")
        if model_is_avaialble[4]:
            plt.plot(x_time, neuron_ctrl[i], label="neuron-0.1")
        plt.xlabel('time')
        plt.ylabel('u')
else:
    for i in range(4):
        plt.subplot(4, 1, i+1)
        if model_is_avaialble[0]:
            plt.plot(x_time, baseline_ctrl[i], label="baseline-0.1")
        if model_is_avaialble[2]:
            plt.plot(x_time, reflex_ctrl[i], label="neuron-0.1")
        plt.xlabel('time')
        plt.ylabel('u')
plt.legend()

plt.show()



