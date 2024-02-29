import numpy as np
from enum import Enum
from scipy import signal
from gym import spaces
import torch
import torch_net

import iir
import fir

class Control_Type(Enum):
    BASELINE = 1
    NEURON = 2
    NEURON_SIMPLE = 3
    NEURON_FULLCON = 4
    NEURON_FC_EP = 5
    NEURON_EP = 6
    NEURON_EP2 = 7
    FF = 8

control_type_dic = {Control_Type.BASELINE: "baseline",
                    Control_Type.NEURON: "neuron",
                    Control_Type.NEURON_SIMPLE: "neuron-simple",
                    Control_Type.NEURON_FULLCON: "neuron-fully-con",
                    Control_Type.NEURON_FC_EP: "neuron-EP-FC",
                    Control_Type.NEURON_EP: "neuron-EP",
                    Control_Type.NEURON_EP2: "neuron-EP2",
                    Control_Type.FF: "ff"
                    }


def InitController(control_type,p):
  print(control_type)
  if control_type == Control_Type.NEURON:
    controller = NeuronController(p)
  elif control_type == Control_Type.NEURON_SIMPLE:
    controller = NeuronSimpleController(p)
  elif control_type == Control_Type.NEURON_FULLCON:
    controller = NeuronFullConnectController(p)
  elif control_type == Control_Type.NEURON_FC_EP:
    controller = NeuronFuCoEpController(p)
  elif control_type == Control_Type.NEURON_EP:
    controller = NeuronEPController(p)
  elif control_type == Control_Type.NEURON_EP2:
    controller = NeuronEP2Controller(p)
  elif control_type == Control_Type.BASELINE:
    controller = BaselineController(p)
  elif control_type == Control_Type.FF:
    controller = FeedforwardParamsController(p)

  return controller

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


    #b, a = signal.butter(1,p.fc/2.0,'low',fs=p.fs)
    #self.fa = [None]*8
    #self.fa[0] = iir.IirFilt(b,a)
    #self.fa[1] = iir.IirFilt(b,a)
    #self.fa[2] = iir.IirFilt(b,a)
    #self.fa[3] = iir.IirFilt(b,a)
    #self.fa[4] = iir.IirFilt(b,a)
    #self.fa[5] = iir.IirFilt(b,a)
    #self.fa[6] = iir.IirFilt(b,a)
    #self.fa[7] = iir.IirFilt(b,a)



  def callback(self,model,data):
    self.get_obs(data)

    #normalize_factor = 1
    normalize_factor = 0.6
    for i in range(2):

      # desired neuron states
      #ldr = self.fa[i*4].filter(self.action[i*4])
      #ldl = self.fa[i*4+1].filter(self.action[i*4+1])
      #rdr = self.fa[i*4+2].filter(self.action[i*4+2])
      #rdl = self.fa[i*4+3].filter(self.action[i*4+3])

      ldr = self.action[i*4]
      ldl = self.action[i*4+1]
      rdr = self.action[i*4+2]
      rdl = self.action[i*4+3]

      # spindle
      sr = self.gamma*data.actuator_velocity[i*2] \
            + data.actuator_length[i*2]/normalize_factor
      sl = self.gamma*data.actuator_velocity[i*2+1] \
            + data.actuator_length[i*2+1]/normalize_factor

      # Ia
      factor = 1/(self.beta**2)
      zr = factor*max(sr - self.beta*sl + self.beta*rdl - rdr, 0)
      zl = factor*max(sl - self.beta*sr + self.beta*rdr - rdl, 0)

      # Motor Neuron
      data.ctrl[i*2] = max(sr - ldr - self.alpha*zl, 0)
      data.ctrl[i*2+1] = max(sl - ldl - self.alpha*zr, 0)

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

  def get_action_space(self):
    return spaces.Box(low=0, high=1.0,shape=(8,),dtype=np.float32)

  def get_obs_space(self):
    return spaces.Box(low=-50, high=50, shape=(8,), dtype=np.float32)

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

    normalize_factor = 0.6
    for i in range(2):

      # desired lengths
      ldr = self.action[i*2]
      ldl = self.action[i*2+1]

      # spindles
      sr = self.gamma*data.actuator_velocity[i*2] \
                      + data.actuator_length[i*2]/normalize_factor
      sl = self.gamma*data.actuator_velocity[i*2+1] \
                      + data.actuator_length[i*2+1]/normalize_factor

      # IaRecIn
      factor = 1/(self.beta**2)
      zr = factor*max(sr - self.beta*sl + self.beta*ldl - ldr,0)
      zl = factor*max(sl - self.beta*sr + self.beta*ldr - ldl,0)

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

  def get_action_space(self):
    return  spaces.Box(low=0, high=1.0, shape=(4,), dtype=np.float32)

  def get_obs_space(self):
    return spaces.Box(low=-50, high=50, shape=(8,), dtype=np.float32)

# -----------------------------------------------------------------------------
# Fully Connected Neural Controller
# -----------------------------------------------------------------------------

class NeuronFullConParams:
  def __init__(self,alpha,beta,gamma,fc,fs):
    self.A = np.array([[0,alpha[0],alpha[1],alpha[2]],
                       [alpha[3],0,alpha[4],alpha[5]],
                       [alpha[6],alpha[7],0,alpha[8]],
                       [alpha[9],alpha[10],alpha[11],0]])
    self.B = np.array([[0,beta[0],beta[1],beta[2]],
                       [beta[3],0,beta[4],beta[5]],
                       [beta[6],beta[7],0,beta[8]],
                       [beta[9],beta[10],beta[11],0]])
    self.gamma = gamma
    self.fc = fc
    self.fs = fs

class NeuronFullConnectController(object):
  def __init__(self,p):

    self.B = p.B
    self.A = p.A
    self.IpBinv = np.linalg.pinv(np.eye(4) + p.B)
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
    normalize_factor = 0.6
    s = np.transpose(np.array( \
        [[self.gamma*data.actuator_velocity[0] \
          + data.actuator_length[0]/normalize_factor,
        self.gamma*data.actuator_velocity[1] \
          + data.actuator_length[1]/normalize_factor,
        self.gamma*data.actuator_velocity[2] \
          + data.actuator_length[2]/normalize_factor,
        self.gamma*data.actuator_velocity[3] \
          + data.actuator_length[3]/normalize_factor]]))

    ld = np.transpose(np.array( \
        [[self.action[0], \
          self.action[1], \
          self.action[2], \
          self.action[3]]]))

    z = np.dot(self.IpBinv,s-ld)
    z = np.clip(z,0,1)
    m = s - ld - np.dot(self.A,z)
    data.ctrl[0:4] = np.transpose(np.clip(m,0,1))

  def get_obs(self,data):
    q0_est = self.fq0.filter(data.qpos[0])
    q1_est = self.fq1.filter(data.qpos[1])
    v0_est = self.fv0.filter(data.qvel[0])
    v1_est = self.fv1.filter(data.qvel[1])
    self.obs = np.array([q0_est,q1_est,v0_est,v1_est])

  def set_action(self,newaction):
    self.action = newaction

  def getAlphaMatrix(self):
    return self.A

  def getBetaMatrix(self):
    return self.B

  def getIpBinvMatrix(self):
    return self.B

  def setAlphaMatrix(self,alpha):
    self.A = np.array([[0,alpha[0],alpha[1],alpha[2]],
                       [alpha[3],0,alpha[4],alpha[5]],
                       [alpha[6],alpha[7],0,alpha[8]],
                       [alpha[9],alpha[10],alpha[11],0]])
  def setBetaMatrix(self,beta):
    self.B = np.array([[0,beta[0],beta[1],beta[2]],
                       [beta[3],0,beta[4],beta[5]],
                       [beta[6],beta[7],0,beta[8]],
                       [beta[9],beta[10],beta[11],0]])
    self.IpBinv = np.linalg.pinv(np.eye(4) + self.B)

  def get_action_space(self):
    return  spaces.Box(low=0, high=1.0, shape=(4,), dtype=np.float32)

  def get_obs_space(self):
    return spaces.Box(low=-50, high=50, shape=(8,), dtype=np.float32)


# -----------------------------------------------------------------------------
# Fully Connected EP Neural Controller
# -----------------------------------------------------------------------------

class NeuronFuCoEpParams:
  def __init__(self,alpha,beta,gamma,fc,fs):
    self.A = np.array([[0,alpha[0],alpha[1],alpha[2]],
                       [alpha[3],0,alpha[4],alpha[5]],
                       [alpha[6],alpha[7],0,alpha[8]],
                       [alpha[9],alpha[10],alpha[11],0]])
    self.B = np.array([[0,beta[0],beta[1],beta[2]],
                       [beta[3],0,beta[4],beta[5]],
                       [beta[6],beta[7],0,beta[8]],
                       [beta[9],beta[10],beta[11],0]])
    self.gamma = gamma
    self.fc = fc
    self.fs = fs

class NeuronFuCoEpController(object):
  def __init__(self,p):

    self.B = p.B
    self.A = p.A
    self.IpBinv = np.linalg.pinv(np.eye(4) + p.B)
    self.gamma = p.gamma
    self.action = np.zeros(4)
    self.obs = np.zeros(4)

    self.Aep_inv = np.array([[1, 1, 0, 0],
                             [-1, 1, 0, 0],
                             [0, 0, 1, 1],
                             [0, 0, -1, 1]])

    b, a = signal.butter(1,p.fc,'low',fs=p.fs)
    self.fq0 = iir.IirFilt(b,a)
    self.fq1 = iir.IirFilt(b,a)
    self.fv0 = iir.IirFilt(b,a)
    self.fv1 = iir.IirFilt(b,a)

    #b, a = signal.butter(1,p.fc/10.0,'low',fs=p.fs)
    #self.fa = [None]*8
    #self.fa[0] = iir.IirFilt(b,a)
    #self.fa[1] = iir.IirFilt(b,a)
    #self.fa[2] = iir.IirFilt(b,a)
    #self.fa[3] = iir.IirFilt(b,a)


  def callback(self,model,data):
    self.get_obs(data)
    normalize_factor = 0.6
    s = np.transpose(np.array( \
        [self.gamma*data.actuator_velocity[0] \
          + data.actuator_length[0]/normalize_factor,
        self.gamma*data.actuator_velocity[1] \
          + data.actuator_length[1]/normalize_factor,
        self.gamma*data.actuator_velocity[2] \
          + data.actuator_length[2]/normalize_factor,
        self.gamma*data.actuator_velocity[3] \
          + data.actuator_length[3]/normalize_factor]))

    #action = np.array([self.fa[0].filter(self.action[0]),
    #                   self.fa[1].filter(self.action[1]),
    #                   self.fa[2].filter(self.action[2]),
    #                   self.fa[3].filter(self.action[3])])

    #ld = self.Aep_inv @ np.transpose(action)

    ld = self.Aep_inv @ np.transpose(np.array( \
                                      [self.action[0],
                                      self.action[1], \
                                      self.action[2], \
                                      self.action[3]]))

    z = self.IpBinv @ (s-ld)
    z = np.clip(z,0,1)
    m = s - ld - self.A @ z
    data.ctrl[0:4] = np.transpose(np.clip(m,0,1))

  def get_obs(self,data):
    q0_est = self.fq0.filter(data.qpos[0])
    q1_est = self.fq1.filter(data.qpos[1])
    v0_est = self.fv0.filter(data.qvel[0])
    v1_est = self.fv1.filter(data.qvel[1])
    self.obs = np.array([q0_est,q1_est,v0_est,v1_est])

  def set_action(self,newaction):
    self.action = newaction

  def getAlphaMatrix(self):
    return self.A

  def getBetaMatrix(self):
    return self.B

  def getIpBinvMatrix(self):
    return self.B

  def setAlphaMatrix(self,alpha):
    self.A = np.array([[0,alpha[0],alpha[1],alpha[2]],
                       [alpha[3],0,alpha[4],alpha[5]],
                       [alpha[6],alpha[7],0,alpha[8]],
                       [alpha[9],alpha[10],alpha[11],0]])
  def setBetaMatrix(self,beta):
    self.B = np.array([[0,beta[0],beta[1],beta[2]],
                       [beta[3],0,beta[4],beta[5]],
                       [beta[6],beta[7],0,beta[8]],
                       [beta[9],beta[10],beta[11],0]])
    self.IpBinv = np.linalg.pinv(np.eye(4) + self.B)

  def get_action_space(self):
    return spaces.Box(low=np.array([-1, 0, -1, 0]),
                      high=np.array([1, 1, 1, 1]),
                      dtype=np.float32)

  def get_obs_space(self):
    return spaces.Box(low=-50, high=50, shape=(8,), dtype=np.float32)

# -----------------------------------------------------------------------------
# EP Neural Controller stiffness and angle
# -----------------------------------------------------------------------------

class NeuronEP2Params:
  def __init__(self,gamma,fc,fs):
    self.gamma = gamma
    self.fc = fc
    self.fs = fs

class NeuronEP2Controller(object):
  def __init__(self,p):

    self.gamma = p.gamma
    self.action_size = 6
    self.action = np.zeros(self.action_size)
    self.obs = np.zeros(4)

    if self.action_size == 4:
        self.Aep_inv = np.array([[1, 1, 0, 0],
                                 [-1, 1, 0, 0],
                                 [0, 0, 1, 1],
                                 [0, 0, -1, 1]])
    elif self.action_size == 6:
        self.Aep_inv = np.array([[1, 1, 0, 0, 0, 0],
                                 [-1, 1, 0, 0, 0, 0],
                                 [0, 0, 1, 1, 0, 0],
                                 [0, 0, -1, 1, 0, 0],
                                 [0, 0, 0, 0, 1, 1],
                                 [0, 0, 0, 0, -1, 1]
                                 ])

    b, a = signal.butter(1,p.fc,'low',fs=p.fs)
    self.fq0 = iir.IirFilt(b,a)
    self.fq1 = iir.IirFilt(b,a)
    self.fv0 = iir.IirFilt(b,a)
    self.fv1 = iir.IirFilt(b,a)

  def callback(self,model,data):
    self.get_obs(data)
    if self.action_size == 4:
        data.ctrl[0:4] = self.Aep_inv @ np.transpose(np.array( \
                                          [self.action[0],
                                          self.action[1],
                                          self.action[2],
                                          self.action[3]]))
    if self.action_size == 6:
        data.ctrl[0:6] = self.Aep_inv @ np.transpose(np.array( \
                                          [self.action[0],
                                          self.action[1],
                                          self.action[2],
                                          self.action[3],
                                            self.action[4],
                                           self.action[5]
                                           ]))

  def get_obs(self,data):
    q0_est = self.fq0.filter(data.qpos[0])
    q1_est = self.fq1.filter(data.qpos[1])
    v0_est = self.fv0.filter(data.qvel[0])
    v1_est = self.fv1.filter(data.qvel[1])
    self.obs = np.array([q0_est,q1_est,v0_est,v1_est])

  def set_action(self,newaction):
    self.action = newaction

  def get_action_space(self):
    if self.action_size == 4:
        return spaces.Box(low=np.array([-1, 0, -1, 0]),
                          high=np.array([1, 1, 1, 1]),
                          dtype=np.float32)

    elif self.action_size == 6:
      return spaces.Box(low=np.array([-1, 0, -1, 0, -1, 0]),
                        high=np.array([1, 1, 1, 1, 1, 1]),
                        dtype=np.float32)

  def get_obs_space(self):
    return spaces.Box(low=-50, high=50, shape=(6,), dtype=np.float32)




# -----------------------------------------------------------------------------
# EP Neural Controller
# -----------------------------------------------------------------------------

class NeuronEPParams:
  def __init__(self,gamma,C,fc,fs):
    self.gamma = gamma
    self.C = C
    self.fc = fc
    self.fs = fs

class NeuronEPController(object):
  def __init__(self,p):

    self.gamma = p.gamma
    self.C = p.C
    self.action = np.zeros(4)
    self.obs = np.zeros(4)

    self.Aep_inv = np.array([[1, 1, 0, 0],
                             [-1, 1, 0, 0],
                             [0, 0, 1, 1],
                             [0, 0, -1, 1]])

    b, a = signal.butter(1,p.fc,'low',fs=p.fs)
    self.fq0 = iir.IirFilt(b,a)
    self.fq1 = iir.IirFilt(b,a)
    self.fv0 = iir.IirFilt(b,a)
    self.fv1 = iir.IirFilt(b,a)

#    b, a = signal.butter(1,p.fc/5,'low',fs=p.fs)
#    self.fa = [None]*2
#    self.fa[0] = iir.IirFilt(b,a)
#    self.fa[1] = iir.IirFilt(b,a)

  def callback(self,model,data):
    self.get_obs(data)
    normalize_factor = 1
    s = np.transpose(np.array( \
        [self.gamma*data.actuator_velocity[0] \
          + data.actuator_length[0]/normalize_factor,
        self.gamma*data.actuator_velocity[1] \
          + data.actuator_length[1]/normalize_factor,
        self.gamma*data.actuator_velocity[2] \
          + data.actuator_length[2]/normalize_factor,
        self.gamma*data.actuator_velocity[3] \
          + data.actuator_length[3]/normalize_factor]))

#    data.ctrl[0:4] = self.Aep_inv @ np.transpose(np.array( \
#                                      [2*self.fa[0].filter(self.action[0]),
#                                      self.C, \
#                                      2*self.fa[1].filter(self.action[1]), \
#                                      self.C]))

    data.ctrl[0:4] = self.Aep_inv @ np.transpose(np.array( \
                                      [self.action[0],
                                      self.C, \
                                      self.action[1], \
                                      self.C]))
  def get_obs(self,data):
    q0_est = self.fq0.filter(data.qpos[0])
    q1_est = self.fq1.filter(data.qpos[1])
    v0_est = self.fv0.filter(data.qvel[0])
    v1_est = self.fv1.filter(data.qvel[1])
    self.obs = np.array([q0_est,q1_est,v0_est,v1_est])

  def set_action(self,newaction):
    self.action = newaction

  def get_action_space(self):
    return spaces.Box(low=np.array([-1, -1]),
                      high=np.array([1, 1]),
                      dtype=np.float32)

  def get_obs_space(self):
    return spaces.Box(low=-50, high=50, shape=(8,), dtype=np.float32)
# -----------------------------------------------------------------------------
# Feedforward Controller
# -----------------------------------------------------------------------------
class FeedforwardParams:
  def __init__(self,fc,fs):
    self.fc = fc
    self.fs = fs

class FeedforwardParamsController(object):
    def __init__(self, p):
        self.obs = np.zeros(4)
        b, a = signal.butter(1, p.fc, 'low', fs=p.fs)
        self.fq0 = iir.IirFilt(b, a)
        self.fq1 = iir.IirFilt(b, a)
        self.fv0 = iir.IirFilt(b, a)
        self.fv1 = iir.IirFilt(b, a)

        self.enable_cocontraction = True
        self.joint_num = 2
        if self.enable_cocontraction:
            if self.joint_num == 2:
                self.action_dim = 4
            elif self.joint_num == 1:
                self.action_dim = 2
        else:
            if self.joint_num == 2:
                self.action_dim = 2
            elif self.joint_num == 1:
                self.action_dim = 1

        self.action = np.zeros(self.action_dim)
        self.target_pos = np.zeros(2)
        self.cocontraction = [0.1, 0.1] #range from 0 to 1
        weights_path = "./ff_optimal_1702622336.pth"
        self.ff_net = torch_net.FeedForwardNN(2, 32, 1, Control_Type.BASELINE)

        self.ff_net.load_state_dict(torch.load(weights_path))
        self.ff_net.eval()

    def set_action(self, newaction):
        for i in range(self.action_dim):
            self.action[i] = newaction[i]

    def callback(self, model, data):
        self.get_obs(data)

        for i in range(self.joint_num):
            if self.enable_cocontraction:
                self.cocontraction[i] = self.action[i*2+1]
            if self.action[i*2] >= 0:
                action_tensor = torch.tensor(np.array([self.action[i*2], self.cocontraction[i]]), dtype=torch.float32)
                u_tensor = self.ff_net(action_tensor.view(1, 2))
                data.ctrl[i * 2] = u_tensor[0][0].item() + self.cocontraction[i]
                data.ctrl[i * 2 + 1] = self.cocontraction[i]
            else:
                action_tensor = torch.tensor(np.array([-self.action[i*2], self.cocontraction[i]]), dtype=torch.float32)
                u_tensor = self.ff_net(action_tensor.view(1, 2))
                data.ctrl[i * 2] = self.cocontraction[i]
                data.ctrl[i * 2 + 1] = u_tensor[0][0].item() + self.cocontraction[i]
        # print(data.ctrl)

    def get_obs(self, data):
        q0_est = self.fq0.filter(data.qpos[0])
        q1_est = self.fq1.filter(data.qpos[1])
        v0_est = self.fv0.filter(data.qvel[0])
        v1_est = self.fv1.filter(data.qvel[1])

        self.obs = np.array([q0_est, q1_est, v0_est, v1_est])

    def get_action_space(self):
        if self.enable_cocontraction:
            if self.joint_num == 2:
                return spaces.Box(low=np.array([-5, 0, -5, 0]), high=np.array([5, 1, 5, 1]),  dtype=np.float32)
            elif self.joint_num == 1:
                return spaces.Box(low=np.array([-5, 0]), high=np.array([5, 1]),  dtype=np.float32)
        else:
            return spaces.Box(low=-5, high=5, shape=(self.joint_num,), dtype=np.float32)

    def get_obs_space(self):
        return spaces.Box(low=-100, high=100, shape=(6,), dtype=np.float32)
# -----------------------------------------------------------------------------
# Baseline Controller
#u-----------------------------------------------------------------------------
class BaselineParams:
  def __init__(self,fc,fs):
    self.fc = fc
    self.fs = fs

class BaselineController(object):
  def __init__(self,p):
    self.obs = np.zeros(4)
    self.action_size = 6

    b, a = signal.butter(1,p.fc,'low',fs=p.fs)
    self.fq0 = iir.IirFilt(b,a)
    self.fq1 = iir.IirFilt(b,a)
    self.fv0 = iir.IirFilt(b,a)
    self.fv1 = iir.IirFilt(b,a)

    self.action = np.zeros(self.action_size)

  def callback(self,model,data):
    self.get_obs(data)
    data.ctrl[0:self.action_size] = self.action

  def set_action(self,newaction):
    self.action = newaction

  def get_obs(self,data):
    q0_est = self.fq0.filter(data.qpos[0])
    q1_est = self.fq1.filter(data.qpos[1])
    v0_est = self.fv0.filter(data.qvel[0])
    v1_est = self.fv1.filter(data.qvel[1])

    self.obs = np.array([q0_est, q1_est, v0_est, v1_est])

  def get_action_space(self):
    return  spaces.Box(low=0, high=1.0, shape=(self.action_size,), dtype=np.float32)

  def get_obs_space(self):
    return spaces.Box(low=-50, high=50, shape=(6,), dtype=np.float32)
