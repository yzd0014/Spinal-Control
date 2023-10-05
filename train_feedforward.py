import os
import copy
import numpy as np
import mujoco as mj
import torch
import torch.nn as nn
import torch.nn.init as init

def compute_physics_gradient(model, data_before_simulation, data_after_simulation, u, eps, num_of_steps, grad):
    for i in range(4):
        data_copy = copy.deepcopy(data_before_simulation)
        data_copy.ctrl[i] = u[i] + eps
        for k in range(num_of_steps):
            mj.mj_step(model, data_copy)
        for j in range(2):
            grad[j][i] = (data_copy.qpos[j] - data_after_simulation.qpos[j]) / eps

class FeedForwardNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        init.xavier_uniform_(self.fc1.weight)
        init.zeros_(self.fc1.bias)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        init.xavier_uniform_(self.fc2.weight)
        init.zeros_(self.fc2.bias)
        self.fc3 = nn.Linear(hidden_size, output_size)
        init.xavier_uniform_(self.fc3.weight)
        init.zeros_(self.fc3.bias)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

# create a feedforward neural network
input_size = 6
hidden_size = 8
output_size = 4
net = FeedForwardNN(input_size, hidden_size, output_size)

# create training data
episdoe_length = 200
num_of_targets = 0
max_training_angle = 0.5
angle_interval = 0.25
traning_samples = []
for i in np.arange(-max_training_angle, max_training_angle, angle_interval):
    for j in np.arange(-max_training_angle, max_training_angle, angle_interval):
        traning_samples.append(np.array([i, j]))
        num_of_targets += 1
print(f"total number of training samples: {num_of_targets}")

# traning configuration
num_epochs = 100
learning_rate = 0.0003

# initialize mujoco
xml_path = 'double_links_fast.xml'
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)
xml_path = abspath
model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
data = mj.MjData(model)  # MuJoCo data

# intialize simutlation parameters
dt_brain = 0.1

optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
for epoch in range(num_epochs):
    mean_ep_loss = 0
    for batch_id in range(num_of_targets):
        # reset at the beginning of each episode
        batch_size = episdoe_length
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)
        target_pos = np.array(traning_samples[batch_id])

        # set optimizer
        optimizer.zero_grad()

        #track loss
        ep_loss = 0
        for i in range(batch_size):
            #feedforward to generate action
            observation = np.array([target_pos[0], target_pos[1], data.qpos[0], data.qvel[0], data.qpos[1], data.qvel[1]])
            observation_tensor = torch.tensor(observation, requires_grad=True, dtype=torch.float32)
            u_tensor = net(observation_tensor.view(1, 6)) #1x4
            u = np.zeros(4)
            for i in range(4):
                u[i] = u_tensor[0][i].item()
            data.ctrl[0:4] = u[0:4]

            #simulation with action to genearsate new state
            data_before_simluation = copy.deepcopy(data)
            time_pre = data.time
            steps_simulated = 0
            while data.time - time_pre < dt_brain:
                mj.mj_step(model, data)
                steps_simulated += 1
            new_state_tensor = torch.tensor(np.array([data.qpos[0], data.qpos[1]]), requires_grad=True, dtype=torch.float32)
            target_state_tensor = torch.tensor(target_pos, requires_grad=False)

            #calculate loss
            loss = torch.norm(target_state_tensor - new_state_tensor, p=2)
            ep_loss += loss
            loss.backward()

            #compute gradient of loss wrt u
            grad_physics = np.zeros((2, 4))
            compute_physics_gradient(model, data_before_simluation, data, u, 0.0001, steps_simulated, grad_physics)
            grad_physics_tensor = torch.tensor(grad_physics, requires_grad=False, dtype=torch.float32) #2x4
            grad_loss_wrt_u_tensor = torch.matmul(new_state_tensor.grad.view(1, 2), grad_physics_tensor) #1x4

            #compute overall graident
            u_tensor.backward(grad_loss_wrt_u_tensor)
            # for param in net.parameters():
            #     if param.requires_grad:
            #         print(param.grad)

        mean_ep_loss += ep_loss
        #update network
        optimizer.step()

    mean_ep_loss /= num_of_targets
    print(f"epoch: {epoch}, mean_ep_loss: {mean_ep_loss}")





