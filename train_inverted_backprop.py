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

def reset_env(model, data):
    mj.mj_resetData(model, data)
    data.qpos[0] = 0.4
    data.qpos[1] = -0.87
    data.qpos[2] = -2.32
    mj.mj_forward(model, data)

#tensorboard
logdir = f"logs/{int(time.time())}-{control_type_dic[parameters.control_type]}/"
if not os.path.exists(logdir):
    os.makedirs(logdir)
writer = SummaryWriter(logdir)

# create a feedforward neural network
input_size = parameters.controller_params.input_size
hidden_size = parameters.controller_params.hidden_size
output_size = parameters.controller_params.output_size
net = torch_net.FeedForwardNN(input_size, hidden_size, output_size, parameters.control_type)

# traning configuration
num_epochs = 2000000000
learning_rate =0.0001
batch_size = parameters.controller_params.episode_length_in_ticks
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

epoch = 0
# while True:
for epoch in range(num_epochs):
    reset_env(model, data)
    optimizer.zero_grad()

    init_obs = np.array([data.qpos[0], data.qpos[1], data.qpos[2], data.qvel[0], data.qvel[1], data.qvel[2]])
    obs = torch.tensor(init_obs, requires_grad=False, dtype=torch.float32).view(1, input_size)
    total_loss = torch.tensor(0.0, dtype=torch.float32)
    for i in range(5):
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

    total_loss.backward()
    optimizer.step()

    writer.add_scalar("Loss/3rd_link_pos", total_loss.item(), epoch)
    interval = 10
    if epoch % interval == 0:
        print(f"epoch-{epoch}, loss: {total_loss.item()}")

    if total_loss.item() < 0.5:
        break

writer.flush()

models_dir = f"models/{int(time.time())}/"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

pickle.dump([parameters.training_type, parameters.control_type, parameters.env_id, parameters.controller_params], open(models_dir + "env_contr_params.p", "wb"))
torch.save(net.state_dict(), f'{models_dir}/{int(time.time())}.pth')