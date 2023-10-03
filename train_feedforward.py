import os
import numpy as np
import mujoco as mj
import torch
import torch.nn as nn

class FeedForwardNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# create a feedforward neural network
input_size = 6
hidden_size = 64
output_size = 4
net = FeedForwardNN(input_size, hidden_size, output_size)
clamped_net = torch.clamp(net, 0, 1)

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

# traning configuration
num_epochs = 100
learning_rate = 0.001

# initialize mujoco
xml_path = 'double_links_fast.xml'
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)
xml_path = abspath
model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
data = mj.MjData(model)  # MuJoCo data

for epoch in range(num_epochs):
    for batch_id in range(num_of_targets):
        batch_size = episdoe_length
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)
        target_pos = np.array(traning_samples[batch_id])
        for i in range(batch_size):
            #feedforward to generate action
            observation = np.array([target_pos[0], target_pos[1], data.qpos[0], data.qvel[0], data.qpos[1], data.qvel[1]])
            observation_tensor = torch.from_numpy(observation).float()
            u_tensor = clamped_net(observation_tensor)
            u = u_tensor.numpy()
            data.ctrl[0:4] = u[0:4]

            #simulation with action to genearsate new state
            mj.mj_step(model, data)
            B = np.zeros((2*model.nv + model.na, model.nu), dtype=np.float64)
            mj.mjd_transitionFD(m=model, d=data, eps=0.0001, flag_centered=0, B=B)
            new_state_tensor = torch.from_numpy(np.array(data.qpos[0], data.qpos[1])).float()
            target_state_tensor = torch.from_numpy(target_pos).float()

            #calculate loss
            loss = torch.norm(target_state_tensor - new_state_tensor, p=2)
            loss.backward()

            #compute gradient of loss wrt u
            grad_physics = np.array([[B[0][0], B[0][1], B[0][2], B[0][3]],
                                    [B[1][0], B[1][1], B[1][2], B[1][3]]])
            grad_physics_tensor = torch.from_numpy(grad_physics).float()
            grad_loss_wrt_u = torch.matmul(grad_physics_tensor, new_state_tensor.grad)

            #compute overall graident
            I = np.identity(4)
            for act_i in range(4):
                dummy_loss = I[act_i][0] * u_tensor[0] + I[act_i][1] * u_tensor[1] + I[act_i][2] * u_tensor[2] + I[act_i][3] * u_tensor[3]
                dummy_loss.backward()
                for param in net.parameters():
                    if param.requires_grad:
                        param.data.grad = param.data.grad * grad_loss_wrt_u[act_i] #this part is currently wrong






