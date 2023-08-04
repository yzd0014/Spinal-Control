from stable_baselines3 import PPO
import mujoco as mj
import numpy as np
import os
import matplotlib.pyplot as plt

def neuron_callback(mode, data):
    obs =np.array([m_target[0], m_target[1], data.qpos[0], data.qvel[0], data.qpos[1], data.qvel[1], 0, 0])
    action, _states = PPO_model.predict(obs)

    normalize_factor = 0.677
    for i in range(2):
        length_r = action[i * 4] * normalize_factor
        length_l = action[i * 4 + 1] * normalize_factor

        r_spindle = 0.05 * data.actuator_velocity[i * 2] + data.actuator_length[i * 2]
        l_spindle = 0.05 * data.actuator_velocity[i * 2 + 1] + data.actuator_length[i * 2 + 1]
        # inhibition_coeff = 0.4691358024691358
        # beta = 0.9
        l_diff = inhibition_coeff / (1 - beta * beta) * max((l_spindle - beta * r_spindle + beta * action[i * 4 + 2] - action[i * 4 + 3]), 0)
        r_diff = inhibition_coeff / (1 - beta * beta) * max((r_spindle - beta * l_spindle + beta * action[i * 4 + 3] - action[i * 4 + 2]), 0)

        ctrl_coeff = 1
        data.ctrl[i * 2] = max(ctrl_coeff * (r_spindle - length_r - l_diff), 0)
        data.ctrl[i * 2 + 1] = max(ctrl_coeff * (l_spindle - length_l - r_diff), 0)

m_target = np.array([-0.82, 0.65])
PPO_model_path = "models/1690837225/4112000.zip"
PPO_model = PPO.load(PPO_model_path)

xml_path = 'double_links.xml' #xml file (assumes this is in the same folder as this file)
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)
xml_path = abspath

model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
data = mj.MjData(model)                # MuJoCo data

mj.mj_resetData(model, data)
mj.mj_forward(model, data)
mj.set_mjcb_control(neuron_callback)

beta_list = [0, 0.5, 0.9]
alpha_list = [0, 0.4691358024691358, 2]
joint_positions = []
x_time = []

target_sim_time = 5
i = 1
for beta in beta_list:
    for inhibition_coeff in alpha_list:
        while data.time < target_sim_time:
            mj.mj_step(model, data)
            joint_positions.append(data.qpos[0])
            x_time.append(data.time)

        plt.subplot(9, 1, i)
        plt.plot(x_time, joint_positions, label="beta: " + str(beta) + " alpha: " + str(inhibition_coeff))
        plt.legend()

        i = i + 1
        joint_positions.clear()
        x_time.clear()

        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)
        print("finished!")

plt.show()

# while data.time < target_sim_time:
#     mj.mj_step(model, data)
#     joint_positions.append(data.qpos[0])
#     x_time.append(data.time)
#
# plt.plot(x_time, joint_positions)
# # plt.xlabel('time')
# # plt.ylabel('joint 1 position')
# plt.show()
# print("finished!")