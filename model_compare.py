from stable_baselines3 import PPO
import mujoco as mj
import os
from spinal_controllers import *
import matplotlib.pyplot as plt

target_pos = -0.32

PPO_model_path0="models/1685125100/1850000.zip"
PPO_model0=PPO.load(PPO_model_path0)
def baseline_callback(model, data):
    obs = np.array([data.qpos[0], data.qvel[0], target_pos, 0])
    action, _states = PPO_model0.predict(obs)
    baseline_controller(input_action=action, data=data)
    # print(data.qpos[0])

PPO_model_path1="models/1685030922/1850000.zip"
PPO_model1=PPO.load(PPO_model_path1)
def RI_callback(model, data):
    obs = np.array([data.qpos[0], data.qvel[0], target_pos, 0])
    action, _states = PPO_model1.predict(obs)
    RI_controller(input_action=action, data=data)
    # print(data.qpos[0])

PPO_model_path2="models/1685057485/1850000.zip"
PPO_model2=PPO.load(PPO_model_path2)
def stretch_reflex_callback(model, data):
    obs = np.array([data.qpos[0], data.qvel[0], target_pos, 0])
    action, _states = PPO_model2.predict(obs)
    stretch_reflex_controller(input_action=action, data=data)
    # print(data.qpos[0])

PPO_model_path3="models/1685079447/1850000.zip"
PPO_model3=PPO.load(PPO_model_path3)
def RI_and_stretch_reflex_callback(model, data):
    obs = np.array([data.qpos[0], data.qvel[0], target_pos, 0])
    action, _states = PPO_model3.predict(obs)
    RI_and_stretch_reflex_controller(input_action=action, data=data)
    # print(data.qpos[0])

PPO_model_path4="models/1684970010/7960000.zip"
PPO_model4=PPO.load(PPO_model_path4)
def neuron_callback(mode, data):
    obs = np.array([data.qpos[0], data.qvel[0], target_pos, 0])
    action, _states = PPO_model4.predict(obs)
    neuron_controller(input_action=action, data=data)

#get the full path
xml_path = 'muscle_control_narrow.xml' #xml file (assumes this is in the same folder as this file)
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)
xml_path = abspath

model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
data = mj.MjData(model)                # MuJoCo data

target_sim_time = 10
x_time = []
y0 = []
y1 = []
y2 = []
y3 = []

mj.set_mjcb_control(baseline_callback)
while data.time < target_sim_time:
    mj.mj_step(model, data)
    x_time.append(data.time)
    y0.append(data.qpos[0])
print("y0 done!")

mj.mj_resetData(model, data)
mj.set_mjcb_control(RI_callback)
while data.time < target_sim_time:
    mj.mj_step(model, data)
    y1.append(data.qpos[0])
print("y1 done!")

mj.mj_resetData(model, data)
mj.set_mjcb_control(stretch_reflex_callback)
while data.time < target_sim_time:
    mj.mj_step(model, data)
    y2.append(data.qpos[0])
print("y2 done!")

mj.mj_resetData(model, data)
mj.set_mjcb_control(RI_and_stretch_reflex_callback)
while data.time < target_sim_time:
    mj.mj_step(model, data)
    y3.append(data.qpos[0])
print("y3 done!")

plt.plot(x_time, y0, label = "baseline")
plt.plot(x_time, y1, label = "RI")
plt.plot(x_time, y2, label = "stretch reflex")
plt.plot(x_time, y3, label = "RI + stretch reflex")
plt.axhline(y = target_pos, color = 'r', linestyle = '-', label = "target position", linewidth = 0.2)
plt.xlabel('time')
plt.ylabel('position')
plt.legend()
plt.show()
