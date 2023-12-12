import numpy as np
import mujoco as mj
import torch

import physics_grad
import torch_net
import os
import copy
import time
import parameters as pa
import pickle
from control import *
from torch.utils.tensorboard import SummaryWriter

#tensorboard
logdir = f"logs/{int(time.time())}-{control_type_dic[pa.control_type]}/"
if not os.path.exists(logdir):
    os.makedirs(logdir)
writer = SummaryWriter(logdir)

models_dir = f"models/{int(time.time())}/"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
pickle.dump([pa.training_type, pa.control_type, pa.env_id, pa.controller_params], open(models_dir + "env_contr_params.p", "wb"))

# create a feedforward neural network
input_size = pa.controller_params.input_size
hidden_size = pa.controller_params.hidden_size
output_size = pa.controller_params.output_size
net = torch_net.FeedForwardNN(input_size, hidden_size, output_size, pa.control_type)

# create training data
num_of_targets = 0
max_training_angle = 0.9
angle_interval = 0.45
max_cocontraction = 0.5
cocontraction_interval = 0.25
traning_samples = []
for i in np.arange(-max_training_angle, max_training_angle+0.1, angle_interval):
    for j in np.arange(-max_training_angle, max_training_angle+0.1, angle_interval):
        for k in np.arange(0, max_cocontraction+0.1, cocontraction_interval):
            traning_samples.append(np.array([i, j, k]))
            num_of_targets += 1
print(f"total number of training samples: {num_of_targets}")

# traning configuration
num_epochs = 5000
learning_rate = 0.0001

# intialize simutlation parameters
dt_brain = pa.controller_params.brain_dt
episode_length = pa.controller_params.episode_length_in_ticks
pa.training_type = "feedforward"
pa.env_id = 0

optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
for epoch in range(num_epochs):
    mean_ep_loss = 0
    old_mean_ep_loss = 0
    for batch_id in range(num_of_targets):
        #reset simulation
        mj.mj_resetData(pa.model, pa.data)
        mj.mj_forward(pa.model, pa.data)

        # gnerate target state and rest
        pa.controller.target_pos = np.array([traning_samples[batch_id][0], traning_samples[batch_id][1]])
        cocontraction = traning_samples[batch_id][2]

        # set cocontraction
        for i in range(2):
            if pa.controller.target_pos[i] >= 0:
                pa.data.ctrl[i * 2 + 1] = cocontraction
            else:
                pa.data.ctrl[i * 2] = cocontraction

        # reset at the beginning of each episode
        batch_size = episode_length
        # set optimizer
        optimizer.zero_grad()
        #track loss
        batch_loss = torch.tensor(0.0, dtype=torch.float32)
        for i in range(batch_size):
            # generate actiavtion
            obs = pa.controller.get_obs(pa.data)
            observation_tensor = torch.tensor(obs, requires_grad=False, dtype=torch.float32)
            u_tensor = net(observation_tensor.view(1, input_size))  # 1xinput_size

            # simulation with action to genearsate new state
            physics_op = physics_grad.physics.apply
            new_state_tensor = physics_op(u_tensor)

            #loss
            target_state_tensor = torch.tensor(pa.controller.target_pos, requires_grad=False)
            loss = torch.norm(target_state_tensor - new_state_tensor, p=2)
            batch_loss = torch.add(batch_loss, loss)

        mean_ep_loss += batch_loss.item()
        batch_loss.backward()
        #update network
        optimizer.step()

    old_mean_ep_loss = mean_ep_loss
    mean_ep_loss /= num_of_targets
    print(f"epoch: {epoch}, mean_ep_loss: {mean_ep_loss}")
    writer.add_scalar("Loss/mean_ep_loss", mean_ep_loss, epoch)
    if epoch % 50 == 0:
        torch.save(net.state_dict(), f'{models_dir}/{int(time.time())}.pth')
    if mean_ep_loss < 0.45:
        break

writer.flush()
torch.save(net.state_dict(), f'{models_dir}/{int(time.time())}.pth')







