import numpy as np
from enum import Enum
from scipy import signal
import mujoco as mj
import os
import copy

import iir
import fir
from gym import spaces
import torch
import torch_net

class Control_Type(Enum):
    DEFAULT = -1
    BASELINE = 1
    NEURON = 2
    NEURON_SIMPLE = 3
    NEURON_OPTIMAL = 4
    PID = 5
    EP = 6
    FF = 7
    EP_GENERAL = 8
    FF_GENERAL = 9
    FF_OPTIMAL = 10


control_type_dic = {Control_Type.BASELINE: "baseline",
                    Control_Type.NEURON: "neuron",
                    Control_Type.NEURON_SIMPLE: "neuron-simple",
                    Control_Type.NEURON_OPTIMAL: "neuron-optimal",
                    Control_Type.PID: "pid",
                    Control_Type.EP: "ep",
                    Control_Type.FF: "feedforward",
                    Control_Type.EP_GENERAL: "ep-general",
                    Control_Type.FF_GENERAL: "feedforward-general",
                    Control_Type.DEFAULT: "default",
                    Control_Type.FF_OPTIMAL: "feedforward-optimal"
                    }


class ControllerParams:
    def __init__(self, alpha, beta, gamma, fc, model_dir, input_size, hidden_size, output_size, brain_dt, episode_length_in_seconds):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.fc = fc
        self.fs = 0
        self.model_dir = model_dir
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.brain_dt = brain_dt
        self.episode_length_in_ticks = int(episode_length_in_seconds / brain_dt)
# -----------------------------------------------------------------------------
# EP Controller
# -----------------------------------------------------------------------------
class EPController(object):
    def __init__(self, env_id):
        self.C = 1
        self.action = np.zeros(2)
        self.target_pos = np.zeros(2)
        self.env_id = env_id

    def set_action(self, newaction):
        for i in range(2):
            self.action[i] = newaction[i]

    def callback(self, model, data):
        for i in range(2):
            data.ctrl[i * 2] = self.C + self.action[i]
            data.ctrl[i * 2 + 1] = self.C - self.action[i]

        if self.env_id == 2:
            if data.time > 3:
                model.eq_active[0] = 0

    def get_obs(self, data):
        if self.env_id == 0:
            obs = np.array([self.target_pos[0], self.target_pos[1], data.qpos[0], data.qvel[0], data.qpos[1], data.qvel[1]])
        elif self.env_id == 1:
            obs = np.array([data.qpos[0], data.qpos[1], data.qpos[2], data.qvel[0], data.qvel[1], data.qvel[2]])
        elif self.env_id == 2:
            obs = np.array([self.target_pos[0], self.target_pos[1],  data.time])
        elif self.env_id == 3:
            obs = np.array([self.target_pos[0], data.xpos[3][0], data.xpos[3][1], data.xpos[3][2], \
                            data.xpos[2][0], data.xpos[2][1], data.xpos[2][2], \
                            data.qpos[0], data.qvel[0], data.qpos[1], data.qvel[1]])
        elif self.env_id == 4:
            obs = np.array([data.qpos[0], data.qpos[1], data.qpos[2], data.qvel[0], data.qvel[1], data.qvel[2]])

        return obs

    def get_action_space(self):
        return spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

    def get_obs_space(self):
        if self.env_id == 0:
            return spaces.Box(low=-100, high=100, shape=(6,), dtype=np.float32)
        elif self.env_id == 1:
            return spaces.Box(low=-100, high=100, shape=(6,), dtype=np.float32)
        elif self.env_id == 2:
            return spaces.Box(low=-100, high=100, shape=(3,), dtype=np.float32)
        elif self.env_id == 3:
            return spaces.Box(low=-100, high=100, shape=(11,), dtype=np.float32)
        elif self.env_id == 4:
            return spaces.Box(low=-100, high=100, shape=(6,), dtype=np.float32)

# -----------------------------------------------------------------------------
# General EP Controller
# -----------------------------------------------------------------------------
class GeneralEPController(object):
    def __init__(self):
        self.action = np.zeros(4)
        self.target_pos = np.zeros(2)

    def set_action(self, newaction):
        for i in range(4):
            self.action[i] = newaction[i]

    def callback(self, model, data):
        for i in range(2):
            data.ctrl[i * 2] = self.action[i*2] - self.action[i*2+1]
            data.ctrl[i * 2 + 1] = self.action[i*2] + self.action[i*2+1]

    def get_obs(self, data, env_id):
        if env_id == 0:
            obs = np.array([self.target_pos[0], self.target_pos[1], data.qpos[0], data.qvel[0], data.qpos[1], data.qvel[1]])
        elif env_id == 1:
            obs = np.array([data.qpos[0], data.qpos[1], data.qpos[2], data.qvel[0], data.qvel[1], data.qvel[2]])
        return obs

    def get_action_space(self):
        return spaces.Box(low=np.array([0, -1, 0, -1]), high=np.array([1, 1, 1, 1]), dtype=np.float32)
    def get_obs_space(self, env_id):
        return spaces.Box(low=-100, high=100, shape=(6,), dtype=np.float32)

# -----------------------------------------------------------------------------
# Feedforward Controller
# -----------------------------------------------------------------------------
class FeedForwardController(object):
    def __init__(self, env_id):
        self.action = np.zeros(2)
        self.target_pos = np.zeros(2)
        weights_path = "./ff_weights.pth"
        # weights_path = "./1702020394.pth"
        self.ff_net = torch_net.FeedForwardNN(2, 64, 4, Control_Type.BASELINE)

        self.ff_net.load_state_dict(torch.load(weights_path))
        self.ff_net.eval()
        self.env_id = env_id

    def set_action(self, newaction):
        for i in range(2):
            self.action[i] = newaction[i]

    def callback(self, model, data):
        action_tensor = torch.tensor(self.action, dtype=torch.float32)
        u_tensor = self.ff_net(action_tensor.view(1, 2))
        for i in range(4):
            data.ctrl[i] = u_tensor[0][i].item()

        if self.env_id == 2:
            if data.time > 3:
                model.eq_active[0] = 0

    def get_action_space(self):
        return spaces.Box(low=-5, high=5, shape=(2,), dtype=np.float32)

    def get_obs_space(self):
        if self.env_id == 0:
            return spaces.Box(low=-100, high=100, shape=(6,), dtype=np.float32)
        elif self.env_id == 1:
            return spaces.Box(low=-100, high=100, shape=(6,), dtype=np.float32)
        elif self.env_id == 2:
            return spaces.Box(low=-100, high=100, shape=(3,), dtype=np.float32)
        elif self.env_id == 3:
            return spaces.Box(low=-100, high=100, shape=(11,), dtype=np.float32)
        elif self.env_id == 4:
            return spaces.Box(low=-100, high=100, shape=(6,), dtype=np.float32)

    def get_obs(self, data):
        if self.env_id == 0:
            obs = np.array(
                [self.target_pos[0], self.target_pos[1], data.qpos[0], data.qvel[0], data.qpos[1], data.qvel[1]])
        elif self.env_id == 1:
            obs = np.array([data.qpos[0], data.qpos[1], data.qpos[2], data.qvel[0], data.qvel[1], data.qvel[2]])
        elif self.env_id == 2:
            obs = np.array([self.target_pos[0], self.target_pos[1], data.time])
        elif self.env_id == 3:
            obs = np.array([self.target_pos[0], data.xpos[3][0], data.xpos[3][1], data.xpos[3][2], \
                            data.xpos[2][0], data.xpos[2][1], data.xpos[2][2], \
                            data.qpos[0], data.qvel[0], data.qpos[1], data.qvel[1]])
        elif self.env_id == 4:
            obs = np.array([data.qpos[0], data.qpos[1], data.qpos[2], data.qvel[0], data.qvel[1], data.qvel[2]])
        return obs
# -----------------------------------------------------------------------------
# Feedforward Controller
# -----------------------------------------------------------------------------
class FeedForwardGeneralController(object):
    def __init__(self, env_id):
        self.action = np.zeros(4)
        self.target_pos = np.zeros(2)
        weights_path = "./ff4.pth"
        self.ff_net = torch_net.FeedForwardNN(4, 32, 4, Control_Type.BASELINE)

        self.ff_net.load_state_dict(torch.load(weights_path))
        self.ff_net.eval()
        self.env_id = env_id

    def set_action(self, newaction):
        for i in range(4):
            self.action[i] = newaction[i]

    def callback(self, model, data):
        action_tensor = torch.tensor(self.action, dtype=torch.float32)
        u_tensor = self.ff_net(action_tensor.view(1, 4))
        for i in range(4):
            data.ctrl[i] = u_tensor[0][i].item()

        if self.env_id == 2:
            if data.time > 3:
                model.eq_active[0] = 0

    def get_action_space(self):
        return spaces.Box(low=0, high=5, shape=(4,), dtype=np.float32)

    def get_obs_space(self):
        if self.env_id == 0:
            return spaces.Box(low=-100, high=100, shape=(6,), dtype=np.float32)
        elif self.env_id == 1:
            return spaces.Box(low=-100, high=100, shape=(6,), dtype=np.float32)
        elif self.env_id == 2:
            return spaces.Box(low=-100, high=100, shape=(3,), dtype=np.float32)
        elif self.env_id == 3:
            return spaces.Box(low=-100, high=100, shape=(11,), dtype=np.float32)
        elif self.env_id == 4:
            return spaces.Box(low=-100, high=100, shape=(6,), dtype=np.float32)

    def get_obs(self, data):
        if self.env_id == 0:
            obs = np.array(
                [self.target_pos[0], self.target_pos[1], data.qpos[0], data.qvel[0], data.qpos[1], data.qvel[1]])
        elif self.env_id == 1:
            obs = np.array([data.qpos[0], data.qpos[1], data.qpos[2], data.qvel[0], data.qvel[1], data.qvel[2]])
        elif self.env_id == 2:
            obs = np.array([self.target_pos[0], self.target_pos[1], data.time])
        elif self.env_id == 3:
            obs = np.array([self.target_pos[0], data.xpos[3][0], data.xpos[3][1], data.xpos[3][2], \
                            data.xpos[2][0], data.xpos[2][1], data.xpos[2][2], \
                            data.qpos[0], data.qvel[0], data.qpos[1], data.qvel[1]])
        elif self.env_id == 4:
            obs = np.array([data.qpos[0], data.qpos[1], data.qpos[2], data.qvel[0], data.qvel[1], data.qvel[2]])
        return obs
# -----------------------------------------------------------------------------
# Neural Controller
# -----------------------------------------------------------------------------
class NeuronController(object):
    def __init__(self, p):
        self.alpha = p.alpha;
        self.beta = p.beta;
        self.gamma = p.gamma
        self.action = np.zeros(8)
        self.ctrl_coeff = 1
        self.obs = np.zeros(4)

        b, a = signal.butter(1, p.fc, 'low', fs=p.fs)
        self.fq0 = iir.IirFilt(b, a)
        self.fq1 = iir.IirFilt(b, a)
        self.fv0 = iir.IirFilt(b, a)
        self.fv1 = iir.IirFilt(b, a)


def callback(self, model, data):
    self.get_obs(data)

    normalize_factor = 0.677
    for i in range(2):
        length_r = self.action[i * 2] * normalize_factor
        length_l = self.action[i * 2 + 1] * normalize_factor

        r_spindle = self.gamma * data.actuator_velocity[i * 2] \
                    + data.actuator_length[i * 2]
        l_spindle = self.gamma * data.actuator_velocity[i * 2 + 1] \
                    + data.actuator_length[i * 2 + 1]

        l_diff = (1 / (1 - self.beta * self.beta)) \
                 * max((l_spindle - self.beta * r_spindle \
                        + self.beta * self.action[i * 4 + 2] - self.action[i * 4 + 3]), 0)

        r_diff = (1 / (1 - self.beta * self.beta)) \
                 * max((r_spindle - self.beta * l_spindle \
                        + self.beta * self.action[i * 4 + 3] - self.action[i * 4 + 2]), 0)
        data.ctrl[i * 2] = max(self.ctrl_coeff * (r_spindle \
                                                  - length_r - self.alpha * l_diff), 0)
        data.ctrl[i * 2 + 1] = max(self.ctrl_coeff * (l_spindle \
                                                      - length_l - self.alpha * r_diff), 0)


    def set_action(self, newaction):
        self.action = newaction


    def get_obs(self, data):
        q0_est = self.fq0.filter(data.qpos[0])
        q1_est = self.fq1.filter(data.qpos[1])
        v0_est = self.fv0.filter(data.qvel[0])
        v1_est = self.fv1.filter(data.qvel[1])
        self.obs = np.array([q0_est, q1_est, v0_est, v1_est])


    def get_action(self):
        return self.action


# -----------------------------------------------------------------------------
# Neural Controller
# -----------------------------------------------------------------------------

class NeuronSimpleController(object):
    def __init__(self, p):
        self.alpha = p.alpha;
        self.beta = p.beta;
        self.gamma = p.gamma
        self.action = np.zeros(4)
        self.obs = np.zeros(4)

        b, a = signal.butter(1, p.fc, 'low', fs=p.fs)
        self.fq0 = iir.IirFilt(b, a)
        self.fq1 = iir.IirFilt(b, a)
        self.fv0 = iir.IirFilt(b, a)
        self.fv1 = iir.IirFilt(b, a)

    def callback(self, model, data):
        self.get_obs(data)

        for i in range(2):
            # desired lengths
            ldr = self.action[i * 2]
            ldl = self.action[i * 2 + 1]

            # spindles
            sr = self.gamma * data.actuator_velocity[i * 2] \
                 + data.actuator_length[i * 2]
            sl = self.gamma * data.actuator_velocity[i * 2 + 1] \
                 + data.actuator_length[i * 2 + 1]

            # IaRecIn
            zl = (1 / (1 - self.beta ** 2)) * max(sl - self.beta * sr + self.beta * ldl - ldr, 0)
            zr = (1 / (1 - self.beta ** 2)) * max(sr - self.beta * sl + self.beta * ldr - ldl, 0)

            # Motor Neuron
            data.ctrl[i * 2] = max(sr - ldr - self.alpha * zl, 0)
            data.ctrl[i * 2 + 1] = max(sl - ldl - self.alpha * zr, 0)

    def set_action(self, newaction):
        self.action = newaction

    def get_obs(self, data):
        q0_est = self.fq0.filter(data.qpos[0])
        q1_est = self.fq1.filter(data.qpos[1])
        v0_est = self.fv0.filter(data.qvel[0])
        v1_est = self.fv1.filter(data.qvel[1])
        self.obs = np.array([q0_est, q1_est, v0_est, v1_est])

    def get_action(self):
        return self.action


# -----------------------------------------------------------------------------
# Baseline Controller
# -----------------------------------------------------------------------------
class BaselineParams:
    def __init__(self, fc, fs):
        self.fc = fc
        self.fs = fs


class BaselineController(object):
    def __init__(self, p, env_id):
        # b, a = signal.butter(1, p.fc, 'low', fs=p.fs)
        # self.fq0 = iir.IirFilt(b, a)
        # self.fq1 = iir.IirFilt(b, a)
        # self.fv0 = iir.IirFilt(b, a)
        # self.fv1 = iir.IirFilt(b, a)
        self.action = np.zeros(4)
        self.target_pos = np.zeros(2)
        self.env_id = env_id

    def callback(self, model, data):
        # self.get_obs(data)
        data.ctrl[0:4] = self.action

        if self.env_id == 2:
            if data.time > 3:
                model.eq_active[0] = 0

    def set_action(self, newaction):
        for i in range(4):
            self.action[i] = newaction[i]

    def get_obs(self, data):
        if self.env_id == 0:
            # q0_est = self.fq0.filter(data.qpos[0])
            # q1_est = self.fq1.filter(data.qpos[1])
            # v0_est = self.fv0.filter(data.qvel[0])
            # v1_est = self.fv1.filter(data.qvel[1])
            # obs = np.array([self.target_pos[0], self.target_pos[1], q0_est, q1_est, v0_est, v1_est])
            obs = np.array([self.target_pos[0], self.target_pos[1], data.qpos[0], data.qvel[0], data.qpos[1], data.qvel[1]])
        elif self.env_id == 1:
            obs = np.array([data.qpos[0], data.qpos[1], data.qpos[2], data.qvel[0], data.qvel[1], data.qvel[2]])
        elif self.env_id == 2:
            obs = np.array([self.target_pos[0], self.target_pos[1],  data.time])
        elif self.env_id == 3:
            obs = np.array([self.target_pos[0], data.xpos[3][0], data.xpos[3][1], data.xpos[3][2], \
                            data.xpos[2][0], data.xpos[2][1], data.xpos[2][2], \
                           data.qpos[0], data.qvel[0], data.qpos[1], data.qvel[1]])
        elif self.env_id == 4:
            obs = np.array([data.qpos[0], data.qpos[1], data.qpos[2], data.qvel[0], data.qvel[1], data.qvel[2]])

        return obs

    def reset_filter(self):
        self.fq0.reset()
        self.fq1.reset()
        self.fv0.reset()
        self.fv1.reset()

    def get_action_space(self):
        return spaces.Box(low=0, high=1.0, shape=(4,), dtype=np.float32)

    def get_obs_space(self):
        if self.env_id == 0:
            return spaces.Box(low=-100, high=100, shape=(6,), dtype=np.float32)
        elif self.env_id == 1:
            return spaces.Box(low=-100, high=100, shape=(6,), dtype=np.float32)
        elif self.env_id == 2:
            return spaces.Box(low=-100, high=100, shape=(3,), dtype=np.float32)
        elif self.env_id == 3:
            return spaces.Box(low=-100, high=100, shape=(11,), dtype=np.float32)
        elif self.env_id == 4:
            return spaces.Box(low=-100, high=100, shape=(6,), dtype=np.float32)

# -----------------------------------------------------------------------------
# PID Controller
# -----------------------------------------------------------------------------
class PIDController():
    def __init__(self):
        self.target_pos = np.zeros(2)
        self.action = np.zeros(2)
        self.q_error = np.zeros(2)

    def set_action(self, inputs):
        for i in range(2):
            self.action[i] = inputs[i]

    def callback(self, model, data):
        Kp = 20
        Ki = 0
        Kd = 0

        for i in range(2):
            self.q_error[i] += (self.action[i] - data.qpos[i]) * model.opt.timestep
            tao = Kp * (self.action[i] - data.qpos[i]) + Ki * self.q_error[i] * data.time - Kd * data.qvel[i]

            # if tao > 0:
            #     data.ctrl[2*i] = tao
            # else:
            #     data.ctrl[2 * i + 1] = -tao
            if self.action[i] > 0:
                data.ctrl[2 * i] = self.action[i]
                data.ctrl[2 * i + 1] = 0
            else:
                data.ctrl[2 * i] = 0
                data.ctrl[2 * i + 1] = -self.action[i]

    def get_obs(self, data, env_id):
        if env_id == 0:
            obs = np.array([self.target_pos[0], self.target_pos[1], data.qpos[0], data.qvel[0], data.qpos[1], data.qvel[1]])
        elif env_id == 1:
            obs = np.array([data.qpos[0], data.qpos[1], data.qpos[2], data.qvel[0], data.qvel[1], data.qvel[2]])
        return obs

    def compute_physics_gradient(self, model, data_before_simulation, data_after_simulation, eps, num_of_steps, env_id, grad):
        if env_id == 0:
            num_joints = 2
        elif env_id == 1:
            num_joints = 3

        old_action = self.q_bar.copy()
        for i in range(2):
            data_copy = copy.deepcopy(data_before_simulation)
            action_temp = old_action.copy()
            action_temp[i] += eps
            self.set_action(action_temp)
            for k in range(num_of_steps):
                mj.mj_step(model, data_copy)
            for j in range(num_joints):
                grad[j][i] = (data_copy.qpos[j] - data_after_simulation.qpos[j]) / eps
        self.set_action(old_action)

    def get_action_space(self):
        return spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

    def get_obs_space(self, env_id):
        return spaces.Box(low=-100, high=100, shape=(6,), dtype=np.float32)

# -----------------------------------------------------------------------------
# Spinal Optimal Controller
# -----------------------------------------------------------------------------
class SpinalOptimalController():
    def __init__(self):
        xml_path = 'double_links_nopassive.xml'
        dirname = os.path.dirname(__file__)
        abspath = os.path.join(dirname + "/" + xml_path)
        xml_path = abspath
        self.m_model = mj.MjModel.from_xml_path(xml_path)
        self.m_data = mj.MjData(self.m_model)

        self.actions = np.zeros(4)
        self.lmax = 0.677

        # b, a = signal.butter(1, p.fc, 'low', fs=p.fs)
        # self.fq0 = iir.IirFilt(b, a)
        # self.fq1 = iir.IirFilt(b, a)
        # self.fv0 = iir.IirFilt(b, a)
        # self.fv1 = iir.IirFilt(b, a)

        self.obs = np.zeros(4)
        self.target_pos = np.zeros(2)
    def set_action(self, inputs):
        #cache original callback
        old_callback = mj.get_mjcb_control()
        mj.set_mjcb_control(None) #disable callback

        max_qpos = 0.977
        act_tuning_factor = 0.43
        base_act = 0.5
        for i in range(2):
            mj.mj_resetData(self.m_model, self.m_data)
            self.m_data.qpos[0] = inputs[i]
            mj.mj_forward(self.m_model, self.m_data)

            l_desired = np.zeros(2)
            l_desired[0:2] = self.m_data.actuator_length[0:2]

            if inputs[i] > 0:
                # u0_temp = inputs[2 + i]
                u0_temp = base_act
                u1_temp = u0_temp - act_tuning_factor * u0_temp * (inputs[i] / max_qpos)
            elif inputs[i] < 0:
                # u1_temp = inputs[2 + i]
                u1_temp = base_act
                u0_temp = u1_temp - act_tuning_factor * u1_temp * (-inputs[i] / max_qpos)
            else:
                # u0_temp = inputs[2 + i]
                # u1_temp = inputs[2 + i]
                u0_temp = base_act
                u1_temp = base_act

            # self.actions[i * 2] = l_desired[0] - self.lmax * u0_temp
            # self.actions[i * 2 + 1] = l_desired[1] - self.lmax * u1_temp
            self.actions[i * 2] = u0_temp
            self.actions[i * 2 + 1] = u1_temp

        #restore original callback
        mj.set_mjcb_control(old_callback)

    def callback(self, model, data):
        # self.get_obs(data)
        for i in range(4):
            # data.ctrl[i] = (data.actuator_length[i] + 0.1 * data.actuator_velocity[i] - self.actions[i]) / self.lmax
            data.ctrl[i] = self.actions[i]

    def get_obs(self, data, env_id):
        # q0_est = self.fq0.filter(data.qpos[0])
        # q1_est = self.fq1.filter(data.qpos[1])
        # v0_est = self.fv0.filter(data.qvel[0])
        # v1_est = self.fv1.filter(data.qvel[1])
        # self.obs = np.array([q0_est, q1_est, v0_est, v1_est])
        if env_id == 0:
            obs = np.array([self.target_pos[0], self.target_pos[1], data.qpos[0], data.qvel[0], data.qpos[1], data.qvel[1]])
        elif env_id == 1:
            obs = np.array([data.qpos[0], data.qpos[1], data.qpos[2], data.qvel[0], data.qvel[1], data.qvel[2]])
        return obs
# -----------------------------------------------------------------------------
# Template Controller
# -----------------------------------------------------------------------------
class TemplateController(object):
    def __init__(self, p, env_id):
        self.action = np.zeros(2)
        self.target_pos = np.zeros(2)
        self.env_id = env_id
        self.cocontraction = 0

    def callback(self, model, data):
        for i in range(1):
            if self.target_pos[i] >= 0:
                data.ctrl[i * 2] = self.action[i] + self.cocontraction
            else:
                data.ctrl[i * 2 + 1] = self.action[i] + self.cocontraction

    def set_action(self, newaction):
        for i in range(1):
            self.action[i] = newaction[i]

    def get_obs(self, data):
        if self.env_id == 0 or self.env_id == -1:
            obs = np.array([self.target_pos[0], self.target_pos[1], data.qpos[0], data.qvel[0], data.qpos[1], data.qvel[1]])
        elif self.env_id == 1:
            obs = np.array([data.qpos[0], data.qpos[1], data.qpos[2], data.qvel[0], data.qvel[1], data.qvel[2]])
        elif self.env_id == 2:
            obs = np.array([self.target_pos[0], self.target_pos[1],  data.time])
        elif self.env_id == 3:
            obs = np.array([self.target_pos[0], data.xpos[3][0], data.xpos[3][1], data.xpos[3][2], \
                            data.xpos[2][0], data.xpos[2][1], data.xpos[2][2], \
                           data.qpos[0], data.qvel[0], data.qpos[1], data.qvel[1]])
        elif self.env_id == 4:
            obs = np.array([data.qpos[0], data.qpos[1], data.qpos[2], data.qvel[0], data.qvel[1], data.qvel[2]])

        return obs

    def get_action_space(self):
        return spaces.Box(low=0, high=1.0, shape=(4,), dtype=np.float32)

    def get_obs_space(self):
        if self.env_id == 0:
            return spaces.Box(low=-100, high=100, shape=(6,), dtype=np.float32)
        elif self.env_id == 1:
            return spaces.Box(low=-100, high=100, shape=(6,), dtype=np.float32)
        elif self.env_id == 2:
            return spaces.Box(low=-100, high=100, shape=(3,), dtype=np.float32)
        elif self.env_id == 3:
            return spaces.Box(low=-100, high=100, shape=(11,), dtype=np.float32)
        elif self.env_id == 4:
            return spaces.Box(low=-100, high=100, shape=(6,), dtype=np.float32)

# -----------------------------------------------------------------------------
# Feedforward Controller
# -----------------------------------------------------------------------------
class AngleStiffnessController(object):
    def __init__(self, env_id, enable_cocontraction=False):
        self.enable_cocontraction = enable_cocontraction
        if self.enable_cocontraction:
            self.action_dim = 4
        else:
            self.action_dim = 2

        self.action = np.zeros(self.action_dim)
        self.target_pos = np.zeros(2)
        self.cocontraction = [0.3, 0.3] #range from 0 to 1
        weights_path = "./ff_optimal_1702622336.pth"
        self.ff_net = torch_net.FeedForwardNN(2, 32, 1, Control_Type.BASELINE)

        self.ff_net.load_state_dict(torch.load(weights_path))
        self.ff_net.eval()
        self.env_id = env_id

    def set_action(self, newaction):

        for i in range(self.action_dim):
            self.action[i] = newaction[i]

    def callback(self, model, data):
        for i in range(2):
            if self.enable_cocontraction:
                self.cocontraction[i] = self.action[i + 2]
            if self.action[i] >= 0:
                action_tensor = torch.tensor(np.array([self.action[i], self.cocontraction[i]]), dtype=torch.float32)
                u_tensor = self.ff_net(action_tensor.view(1, 2))
                data.ctrl[i * 2] = u_tensor[0][0].item() + self.cocontraction[i]
                data.ctrl[i * 2 + 1] = self.cocontraction[i]
            else:
                action_tensor = torch.tensor(np.array([-self.action[i], self.cocontraction[i]]), dtype=torch.float32)
                u_tensor = self.ff_net(action_tensor.view(1, 2))
                data.ctrl[i * 2] = self.cocontraction[i]
                data.ctrl[i * 2 + 1] = u_tensor[0][0].item() + self.cocontraction[i]

        if self.env_id == 2:
            if data.time > 3:
                model.eq_active[0] = 0

    def get_action_space(self):
        if self.enable_cocontraction:
            return spaces.Box(low=np.array([-5, -5, 0, 0]), high=np.array([5, 5, 1, 1]),  dtype=np.float32)
        else:
            return spaces.Box(low=-5, high=5, shape=(2,), dtype=np.float32)

    def get_obs_space(self):
        if self.env_id == 0:
            return spaces.Box(low=-100, high=100, shape=(6,), dtype=np.float32)
        elif self.env_id == 1:
            return spaces.Box(low=-100, high=100, shape=(6,), dtype=np.float32)
        elif self.env_id == 2:
            return spaces.Box(low=-100, high=100, shape=(3,), dtype=np.float32)
        elif self.env_id == 3:
            return spaces.Box(low=-100, high=100, shape=(11,), dtype=np.float32)
        elif self.env_id == 4:
            return spaces.Box(low=-100, high=100, shape=(6,), dtype=np.float32)

    def get_obs(self, data):
        if self.env_id == 0:
            obs = np.array(
                [self.target_pos[0], self.target_pos[1], data.qpos[0], data.qvel[0], data.qpos[1], data.qvel[1]])
        elif self.env_id == 1:
            obs = np.array([data.qpos[0], data.qpos[1], data.qpos[2], data.qvel[0], data.qvel[1], data.qvel[2]])
        elif self.env_id == 2:
            obs = np.array([self.target_pos[0], self.target_pos[1], data.time])
        elif self.env_id == 3:
            obs = np.array([self.target_pos[0], data.xpos[3][0], data.xpos[3][1], data.xpos[3][2], \
                            data.xpos[2][0], data.xpos[2][1], data.xpos[2][2], \
                            data.qpos[0], data.qvel[0], data.qpos[1], data.qvel[1]])
        elif self.env_id == 4:
            obs = np.array([data.qpos[0], data.qpos[1], data.qpos[2], data.qvel[0], data.qvel[1], data.qvel[2]])
        return obs