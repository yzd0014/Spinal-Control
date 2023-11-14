import numpy as np
import mujoco as mj
import torch
import torch_net
import os
import copy
import time
import parameters
import pickle
from control import *
from torch.utils.tensorboard import SummaryWriter
import physics_grad
import sys

def reset_env(model, data, initial_state):
    mj.mj_resetData(model, data)
    # data.qpos[0] = 0.4
    # data.qpos[1] = -0.87
    # data.qpos[2] = -2.32
    data.qpos[0] = initial_state[0]
    data.qpos[1] = initial_state[1]
    data.qpos[2] = -np.pi + (initial_state[2] - initial_state[0] - initial_state[1])
    data.qvel[0] = initial_state[3]
    data.qvel[1] = initial_state[4]
    data.qvel[2] = initial_state[5]
    mj.mj_forward(model, data)

def random_inital_state(intial_state):
    for i in range(0, 3):
        intial_state[i] = np.random.uniform(-0.2, 0.2)
    for i in range(3, 6):
        intial_state[i] = np.random.uniform(-2, 2)

#tensorboard
logdir = f"logs/{int(time.time())}-{control_type_dic[parameters.control_type]}/"
if not os.path.exists(logdir):
    os.makedirs(logdir)
writer = SummaryWriter(logdir)

#training data dir
training_data_dir = f"training_data/"
if not os.path.exists(training_data_dir):
    os.makedirs(training_data_dir)

#generate training data
generate_training_data = 1
initial_states = []
pos_invertal = 0.1
vol_invertal = 1
pos_bound = 0.2
vel_bound = 1
training_size = 0
if generate_training_data == 0:
    for x_i in np.arange(-pos_bound, pos_bound, pos_invertal):
        for y_i in np.arange(-pos_bound, pos_bound, pos_invertal):
            for z_i in np.arange(-pos_bound, pos_bound, pos_invertal):
                for xdot_i in np.arange(-vel_bound, vel_bound, vol_invertal):
                    for ydot_i in np.arange(-vel_bound, vel_bound, vol_invertal):
                        for zdot_i in np.arange(-vel_bound, vel_bound, vol_invertal):
                            initial_states.append(np.array([x_i, y_i, z_i, xdot_i, ydot_i, zdot_i]))
                            training_size += 1
    pickle.dump(initial_states, open(f"{training_data_dir}/{int(time.time())}.pkl", "wb"))
    print("training data generated")
    sys.exit()
elif generate_training_data == 1:
    training_data = 1698739542
    initial_states = pickle.load(open(f"{training_data_dir}{training_data}.pkl", "rb"))
    training_size = len(initial_states)
elif generate_training_data == 2:
    training_size = 128

# create a feedforward neural network
input_size = parameters.controller_params.input_size
hidden_size = parameters.controller_params.hidden_size
output_size = parameters.controller_params.output_size
net = torch_net.FeedForwardNN(input_size, hidden_size, output_size, parameters.control_type)

# traning configuration
num_epochs = 2000000000
learning_rate =0.0001
batch_size = 5
# batch_size = 5
ep_id = 0

# initialize mujoco
model = parameters.model
data = parameters.data

# initialize controller
if parameters.control_type == Control_Type.BASELINE:
    mj.set_mjcb_control(parameters.controller.callback)
elif parameters.control_type == Control_Type.PID:
    mj.set_mjcb_control(parameters.controller.callback)

# intialize simutlation parameters
dt_brain = parameters.controller_params.brain_dt
episode_length = parameters.controller_params.episode_length_in_ticks
parameters.training_type = "feedforward"
parameters.env_id = 1

optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

print(f"total number of training samples: {training_size}")
epoch = 0
while True:
# for epoch in range(num_epochs):
    mean_batch_loss = 0.0
    for states_i in range(training_size):
        initial_state = np.zeros(6)
        if generate_training_data == 2:
            random_inital_state(initial_state)
        else:
            initial_state = initial_states[states_i]
        reset_env(model, data, initial_state)
        optimizer.zero_grad()

        init_obs = np.array([data.qpos[0], data.qpos[1], data.qpos[2], data.qvel[0], data.qvel[1], data.qvel[2]])
        obs = torch.tensor(init_obs, requires_grad=False, dtype=torch.float32).view(1, input_size)
        total_loss = torch.tensor(0.0, dtype=torch.float32)
        for i in range(batch_size):
            # feedforward to generate action
            u_tensor = net(obs)  # 1x4

            # simulation with action to genearsate new state
            physics_op = physics_grad.inverted_pendulum_physics.apply
            new_state_tensor = physics_op(u_tensor)
            obs = new_state_tensor

            # loss
            sum = torch.dot(new_state_tensor.view(6), torch.tensor(np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]), requires_grad=False, dtype=torch.float32))
            loss = torch.norm(-np.pi - sum, p=2)
            total_loss = torch.add(total_loss, loss)

            #check if end of episode is reached
            # if abs(-np.pi - sum) > 0.25 * np.pi:
            #     writer.add_scalar("Loss/ep_length", data.time, ep_id)
            #     reset_env(model, data)
            #     ep_id += 1
            #     break
        mean_batch_loss += loss.item()
        total_loss.backward()
        optimizer.step()

        # print_interval = 50
        # if states_i % print_interval == 0:
        #     print(f"states_i-{states_i}, loss: {total_loss.item()}")

    mean_batch_loss /= training_size
    writer.add_scalar("Loss/3rd_link_pos", mean_batch_loss, epoch)
    print_interval = 1
    if epoch % print_interval == 0:
        print(f"epoch-{epoch}, loss: {mean_batch_loss}")
    if mean_batch_loss < 0.23:
        break
    epoch += 1

writer.flush()

models_dir = f"models/{int(time.time())}/"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

pickle.dump([parameters.training_type, parameters.control_type, parameters.env_id, parameters.controller_params], open(models_dir + "env_contr_params.p", "wb"))
torch.save(net.state_dict(), f'{models_dir}/{int(time.time())}.pth')