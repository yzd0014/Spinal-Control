import numpy as np
from enum import Enum
from scipy import signal
import mujoco as mj
import os

import iir
import fir


class Control_Type(Enum):
    BASELINE = 1
    NEURON = 2
    NEURON_SIMPLE = 3
    NEURON_OPTIMAL = 4
    PID = 5


control_type_dic = {Control_Type.BASELINE: "baseline",
                    Control_Type.NEURON: "neuron",
                    Control_Type.NEURON_SIMPLE: "neuron-simple",
                    Control_Type.NEURON_OPTIMAL: "neuron-optimal",
                    Control_Type.PID: "pid"}


class ControllerParams:
    def __init__(self, alpha, beta, gamma, fc, input_size, hidden_size, output_size, brain_dt, episode_length_in_seconds):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.fc = fc
        self.fs = 0
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.brain_dt = brain_dt
        self.episode_length_in_ticks = int(episode_length_in_seconds / brain_dt)

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
    def __init__(self, p):
        self.obs = np.zeros(4)

        b, a = signal.butter(1, p.fc, 'low', fs=p.fs)
        self.fq0 = iir.IirFilt(b, a)
        self.fq1 = iir.IirFilt(b, a)
        self.fv0 = iir.IirFilt(b, a)
        self.fv1 = iir.IirFilt(b, a)

        self.action = np.zeros(4)

    def callback(self, model, data):
        # self.get_obs(data)
        data.ctrl[0:4] = self.action

    def set_action(self, newaction):
        self.action = newaction

    def get_obs(self, data):
        q0_est = self.fq0.filter(data.qpos[0])
        q1_est = self.fq1.filter(data.qpos[1])
        v0_est = self.fv0.filter(data.qvel[0])
        v1_est = self.fv1.filter(data.qvel[1])
        self.obs = np.array([q0_est, q1_est, v0_est, v1_est])

    def reset_filter(self):
        self.fq0.reset()
        self.fq1.reset()
        self.fv0.reset()
        self.fv1.reset()
# -----------------------------------------------------------------------------
# PID Controller
# -----------------------------------------------------------------------------
class PIDController():
    def __init__(self):
        self.actions = np.zeros(4)
        self.q_bar = np.zeros(2)
        self.q_error = np.zeros(2)

    def set_action(self, inputs):
        self.q_bar = inputs[0:2]

    def callback(self, model, data):
        Kp = 20
        Ki = 0.01
        Kd = 1

        for i in range(2):
            data.ctrl[2*i] = 0
            data.ctrl[2 * i + 1] = 0

            self.q_error[i] += (self.q_bar[i] - data.qpos[i]) * model.opt.timestep
            tao = Kp * (self.q_bar[i] - data.qpos[i]) + Ki * self.q_error[i] * data.time - Kd * data.qvel[i]

            if tao > 0:
                data.ctrl[2*i] = tao
            else:
                data.ctrl[2 * i + 1] = -tao

# -----------------------------------------------------------------------------
# Baseline Controller
# -----------------------------------------------------------------------------
class SpinalOptimalController():
    def __init__(self, p):
        xml_path = 'double_links_nopassive.xml'
        dirname = os.path.dirname(__file__)
        abspath = os.path.join(dirname + "/" + xml_path)
        xml_path = abspath
        self.m_model = mj.MjModel.from_xml_path(xml_path)
        self.m_data = mj.MjData(self.m_model)

        self.actions = np.zeros(4)
        self.lmax = 0.677

        b, a = signal.butter(1, p.fc, 'low', fs=p.fs)
        self.fq0 = iir.IirFilt(b, a)
        self.fq1 = iir.IirFilt(b, a)
        self.fv0 = iir.IirFilt(b, a)
        self.fv1 = iir.IirFilt(b, a)

        self.obs = np.zeros(4)

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

            self.actions[i * 2] = l_desired[0] - self.lmax * u0_temp
            self.actions[i * 2 + 1] = l_desired[1] - self.lmax * u1_temp

        #restore original callback
        mj.set_mjcb_control(old_callback)

    def callback(self, model, data):
        # self.get_obs(data)
        for i in range(4):
            data.ctrl[i] = (data.actuator_length[i] + 0.1 * data.actuator_velocity[i] - self.actions[i]) / self.lmax

    def get_obs(self, data):
        q0_est = self.fq0.filter(data.qpos[0])
        q1_est = self.fq1.filter(data.qpos[1])
        v0_est = self.fv0.filter(data.qvel[0])
        v1_est = self.fv1.filter(data.qvel[1])
        self.obs = np.array([q0_est, q1_est, v0_est, v1_est])
