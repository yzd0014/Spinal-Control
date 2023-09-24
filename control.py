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

control_type_dic = {Control_Type.BASELINE: "baseline",
                    Control_Type.NEURON: "neuron",
                    Control_Type.NEURON_SIMPLE: "neuron-simple",
                    Control_Type.NEURON_SIMPLE: "neuron-optimal"}


# -----------------------------------------------------------------------------
# Neural Controller
# -----------------------------------------------------------------------------

class NeuronParams:
  def __init__(self,alpha,beta,gamma,fc,fs):
    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma
    self.fc = fc
    self.fs = fs


class NeuronController(object):
  def __init__(self,p):
    self.alpha = p.alpha;
    self.beta = p.beta;
    self.gamma = p.gamma
    self.action = np.zeros(8)
    self.ctrl_coeff = 1
    self.obs = np.zeros(4)

    b, a = signal.butter(1,p.fc,'low',fs=p.fs)
    self.fq0 = iir.IirFilt(b,a)
    self.fq1 = iir.IirFilt(b,a)
    self.fv0 = iir.IirFilt(b,a)
    self.fv1 = iir.IirFilt(b,a)

  def callback(self,model,data):

    self.get_obs(data)

    normalize_factor = 0.677
    for i in range(2):
        length_r = self.action[i*2] * normalize_factor
        length_l = self.action[i*2+1] * normalize_factor

        r_spindle = self.gamma*data.actuator_velocity[i*2]  \
                                + data.actuator_length[i*2]
        l_spindle = self.gamma*data.actuator_velocity[i*2+1] \
                                + data.actuator_length[i*2+1]

        l_diff = (1 /(1-self.beta*self.beta)) \
                  *max((l_spindle - self.beta*r_spindle \
                        + self.beta*self.action[i*4+2] - self.action[i*4+3]),0)

        r_diff = (1/(1-self.beta*self.beta)) \
                  *max((r_spindle - self.beta*l_spindle \
                      + self.beta*self.action[i*4+3] - self.action[i*4+2]),0)
        data.ctrl[i*2] = max(self.ctrl_coeff*(r_spindle \
                                      - length_r - self.alpha*l_diff),0)
        data.ctrl[i*2+1] = max(self.ctrl_coeff * (l_spindle \
                                      - length_l - self.alpha*r_diff),0)
  def set_action(self,newaction):
    self.action = newaction

  def get_obs(self,data):
    q0_est = self.fq0.filter(data.qpos[0])
    q1_est = self.fq1.filter(data.qpos[1])
    v0_est = self.fv0.filter(data.qvel[0])
    v1_est = self.fv1.filter(data.qvel[1])
    self.obs = np.array([q0_est,q1_est,v0_est,v1_est])

  def get_action(self):
    return self.action

# -----------------------------------------------------------------------------
# Neural Controller
# -----------------------------------------------------------------------------

class NeuronSimpleController(object):
  def __init__(self,p):
    self.alpha = p.alpha;
    self.beta = p.beta;
    self.gamma = p.gamma
    self.action = np.zeros(4)
    self.obs = np.zeros(4)

    b, a = signal.butter(1,p.fc,'low',fs=p.fs)
    self.fq0 = iir.IirFilt(b,a)
    self.fq1 = iir.IirFilt(b,a)
    self.fv0 = iir.IirFilt(b,a)
    self.fv1 = iir.IirFilt(b,a)

  def callback(self,model,data):

    self.get_obs(data)

    for i in range(2):

      # desired lengths
      ldr = self.action[i*2]
      ldl = self.action[i*2+1]

      # spindles
      sr = self.gamma*data.actuator_velocity[i*2] \
                      + data.actuator_length[i*2]
      sl = self.gamma*data.actuator_velocity[i*2+1] \
                      + data.actuator_length[i*2+1]

      # IaRecIn
      zl = (1/(1-self.beta**2))*max(sl - self.beta*sr + self.beta*ldl - ldr,0)
      zr = (1/(1-self.beta**2))*max(sr - self.beta*sl + self.beta*ldr - ldl,0)

      # Motor Neuron
      data.ctrl[i*2] = max(sr - ldr - self.alpha*zl,0)
      data.ctrl[i*2+1] = max(sl - ldl - self.alpha*zr,0)

  def set_action(self,newaction):
    self.action = newaction

  def get_obs(self,data):
    q0_est = self.fq0.filter(data.qpos[0])
    q1_est = self.fq1.filter(data.qpos[1])
    v0_est = self.fv0.filter(data.qvel[0])
    v1_est = self.fv1.filter(data.qvel[1])
    self.obs = np.array([q0_est,q1_est,v0_est,v1_est])

  def get_action(self):
    return self.action



# -----------------------------------------------------------------------------
# Baseline Controller
# -----------------------------------------------------------------------------
class BaselineParams:
  def __init__(self,fc,fs):
    self.fc = fc
    self.fs = fs

class BaselineController(object):
  def __init__(self,p):
    self.obs = np.zeros(4)

    b, a = signal.butter(1,p.fc,'low',fs=p.fs)
    self.fq0 = iir.IirFilt(b,a)
    self.fq1 = iir.IirFilt(b,a)
    self.fv0 = iir.IirFilt(b,a)
    self.fv1 = iir.IirFilt(b,a)

    self.action = np.zeros(4)

  def callback(self,model,data):
    self.get_obs(data)
    data.ctrl[0:4] = self.action

  def set_action(self,newaction):
    self.action = newaction

  def get_obs(self,data):
    q0_est = self.fq0.filter(data.qpos[0])
    q1_est = self.fq1.filter(data.qpos[1])
    v0_est = self.fv0.filter(data.qvel[0])
    v1_est = self.fv1.filter(data.qvel[1])
    self.obs = np.array([q0_est,q1_est,v0_est,v1_est])

# -----------------------------------------------------------------------------
# Baseline Controller
# -----------------------------------------------------------------------------
class SpinalOptimalController():
  def __init__(self):
    xml_path = 'double_links_fast.xml'
    dirname = os.path.dirname(__file__)
    abspath = os.path.join(dirname + "/" + xml_path)
    xml_path = abspath
    self.m_model = mj.MjModel.from_xml_path(xml_path)
    self.m_data = mj.MjData(self.m_model)

    self.actions = np.zeros(4)
    self.lmax = 0.677

  def set_inputs(self, data, inputs):
    mj.mj_resetData(self.m_model, self.m_data)
    self.m_data.qpos[0:2] = inputs[0:2]
    mj.mj_forward(self.m_model, self.m_data)

    l_desired = np.zeros(4)
    l_desired[0:4] = self.m_data.actuator_length[0:4]

    # Kp = 1
    # Kd = 0.1
    # for i in range(2):
    #   if np.absolute(data.qpos[i] - inputs[i]) > 0.1:
    #     tao = Kp * (inputs[i] - data.qpos[i]) - Kd * data.qvel[i]
    #     if tao > 0:
    #       u0_temp = tao
    #       u1_temp = 0
    #     else:
    #       u0_temp = 0
    #       u1_temp = -tao
    #
    #     self.actions[i * 2] = data.actuator_length[i * 2] + 0.1 * data.actuator_velocity[2 * i]- self.lmax * u0_temp
    #     self.actions[i * 2 + 1] = data.actuator_length[i * 2 + 1] + 0.1 * data.actuator_velocity[2 * i + 1]- self.lmax * u1_temp
    #   else:
    #     moment_ratio = abs(self.m_data.actuator_moment[i * 2 + 1][i]) / abs(self.m_data.actuator_moment[i * 2][i])
    #     co_contraction_level = inputs[2 + i]
    #     if moment_ratio < 1:
    #       u0_temp = co_contraction_level
    #       u1_temp = co_contraction_level * moment_ratio
    #     else:
    #       moment_ratio = 1 / moment_ratio
    #       u0_temp = co_contraction_level * moment_ratio
    #       u1_temp = co_contraction_level
    #
    #     self.actions[i * 2] = l_desired[i * 2] - self.lmax * u0_temp
    #     self.actions[i * 2 + 1] = l_desired[i * 2 + 1] - self.lmax * u1_temp

  def callback(self, model, data):
    # for i in range(4):
    #   data.ctrl[i] = (data.actuator_length[i] + 0.1 * data.actuator_velocity[i] - self.actions[i])/self.lmax

    Kp = 1
    Kd = 0.1
    for i in range(2):
      tao = Kp * (self.m_data.qpos[i] - data.qpos[i]) - Kd * data.qvel[i]
      print(tao)
      if tao > 0:
        u0_temp = tao
        u1_temp = 0
      else:
        u0_temp = 0
        u1_temp = -tao

      data.ctrl[i * 2] = u0_temp
      data.ctrl[i * 2 + 1] = u1_temp
