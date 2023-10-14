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
num_epochs = 5000
learning_rate = 0.0003
batch_size = 32
ep_id = 0

# initialize mujoco
xml_path = parameters.controller_params.model_dir
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)
xml_path = abspath
model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
data = mj.MjData(model)  # MuJoCo data

# initialize controller
if parameters.control_type == Control_Type.BASELINE:
    controller = BaselineController(parameters.controller_params)
    mj.set_mjcb_control(controller.callback)
elif parameters.control_type == Control_Type.PID:
    controller = PIDController()
    mj.set_mjcb_control(controller.callback)

# intialize simutlation parameters
dt_brain = parameters.controller_params.brain_dt
episode_length = parameters.controller_params.episode_length_in_ticks
parameters.training_type = "feedforward"
parameters.env_id = 1

optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

reset_env(model, data)
epoch = 0
while True:
# for epoch in range(num_epochs):
    # set optimizer
    optimizer.zero_grad()
    batch_loss = 0
    for i in range(batch_size):
        # feedforward to generate action
        observation = controller.get_obs(data, parameters.env_id)
        observation_tensor = torch.tensor(observation, requires_grad=True, dtype=torch.float32)
        u_tensor = net(observation_tensor.view(1, input_size))  # 1x4
        u = np.zeros(output_size)
        for i in range(output_size):
            u[i] = u_tensor[0][i].item()
        controller.set_action(u)
        # data.ctrl[0:4] = u[0:4]

        # simulation with action to genearsate new state
        data_before_simluation = copy.deepcopy(data)
        time_pre = data.time
        steps_simulated = 0
        while data.time - time_pre < dt_brain:
            mj.mj_step(model, data)
            steps_simulated += 1
        new_state_tensor = torch.tensor(np.array([data.qpos[0], data.qpos[1], data.qpos[2]]), requires_grad=True, dtype=torch.float32)
        vec_one = torch.tensor(np.array([1.0, 1.0, 1.0]), requires_grad=False, dtype=torch.float32)
        sum = torch.dot(new_state_tensor, vec_one)

        # calculate loss
        loss = torch.norm(-np.pi - sum, p=2)
        batch_loss += loss
        loss.backward()

        # compute gradient of loss wrt u
        grad_physics = np.zeros((3, output_size))
        controller.compute_physics_gradient(model, data_before_simluation, data, 0.0001, steps_simulated, 1, grad_physics)
        grad_physics_tensor = torch.tensor(grad_physics, requires_grad=False, dtype=torch.float32)  # 3x4
        grad_loss_wrt_u_tensor = torch.matmul(new_state_tensor.grad.view(1, 3), grad_physics_tensor)  # 1x4

        # compute overall graident
        u_tensor.backward(grad_loss_wrt_u_tensor)

        #check if end of episode is reached
        if abs(-np.pi - sum) > 0.25 * np.pi:
            writer.add_scalar("Loss/ep_length", data.time, ep_id)
            reset_env(model, data)
            ep_id += 1
            break

    optimizer.step()
    if epoch % 100 == 0:
        print(f"mean_ep_loss: {batch_loss}")
    epoch += 1
    if data.time > 50 or epoch > 10000:
        break

writer.flush()

models_dir = f"models/{int(time.time())}/"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

pickle.dump([parameters.training_type, parameters.control_type, parameters.env_id, parameters.controller_params], open(models_dir + "env_contr_params.p", "wb"))
torch.save(net.state_dict(), f'{models_dir}/{int(time.time())}.pth')