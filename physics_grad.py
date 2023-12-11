import copy
import torch
import math
import parameters as pa
import numpy as np
import mujoco as mj

u_dim = pa.controller_params.output_size
class double_pendulum_physics(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        joint_number = 2
        steps_simulated = 0

        u = np.zeros(u_dim)
        for i in range(u_dim):
            u[i] = input[0][i].item()
        pa.controller.set_action(u)

        ctx.save_for_backward(input)
        ctx.data_before_simulation = copy.deepcopy(pa.data)

        time_pre = pa.data.time
        while pa.data.time - time_pre < pa.controller_params.brain_dt:
            pa.mj.mj_step(pa.model, pa.data)
            steps_simulated += 1

        ctx.steps_simulated = steps_simulated
        ctx.data_after_simulation = copy.deepcopy(pa.data)

        new_state = np.array([pa.data.qpos[0], pa.data.qpos[1]])
        new_state_tensor = torch.tensor(new_state, requires_grad=True, dtype=torch.float32).view(1, joint_number)

        return new_state_tensor

    @staticmethod
    def backward(ctx, grad_output):
        joint_number = 2
        input, = ctx.saved_tensors
        data_before_simulation = ctx.data_before_simulation
        data_after_simulation = ctx.data_after_simulation
        steps_simulated = ctx.steps_simulated

        eps = 0.000001
        grad = np.zeros((joint_number, u_dim))

        u = np.zeros(u_dim)
        for i in range(u_dim):
            u[i] = input[0][i].item()

        old_action = pa.controller.action.copy()
        for i in range(u_dim):
            data_copy = copy.deepcopy(data_before_simulation)
            action_temp = u.copy()
            action_temp[i] += eps
            pa.controller.set_action(action_temp)
            for k in range(steps_simulated):
                mj.mj_step(pa.model, data_copy)
            for j in range(joint_number):
                grad[j][i] = (data_copy.qpos[j] - data_after_simulation.qpos[j]) / eps
        pa.controller.set_action(old_action)

        grad_physics_tensor = torch.tensor(grad, requires_grad=False, dtype=torch.float32)  # joint_numberxu_dim
        grad_loss_wrt_u_tensor = torch.matmul(grad_output.view(1, joint_number), grad_physics_tensor)  # 1xjoint_number * joint_numberxu_dim = 1xu_dim

        return grad_loss_wrt_u_tensor



