import numpy as np
from control import *
import pickle

# Mujoco Sampling
fs = 200;

# loop factor for brain sampling
fs_brain_factor = 20
fc = (fs/fs_brain_factor)/2

# Choose controller here

# Neuron Parameters
#control_type = Control_Type.NEURON
#controller_params = NeuronParams(alpha=0.4691358024691358, \
#                                  beta=0.9, \
#                                  gamma=1, \
#                                  fc=10, \
#                                  fs=fs)

# Neuron Fully Connected Parameters
#control_type = Control_Type.NEURON_FULLCON
##p0 = pickle.load(open("p0_minimal.p", "rb"))
#p0 = pickle.load(open("p0_reflex_only.p", "rb"))
#b = p0[0:12]
#a = p0[12:]
#controller_params = NeuronFullConParams(alpha=a, \
#                                  beta=b, \
#                                  gamma=1, \
#                                  fc=10, \
#                                  fs=fs)


# Neuron Fully Connected EP Parameters
#control_type = Control_Type.NEURON_FC_EP
#p0 = pickle.load(open("p0_minimal.p", "rb"))
#b = p0[0:12]
#a = p0[12:]
#controller_params = NeuronFuCoEpParams(alpha=a, \
#                                  beta=b, \
#                                  gamma=1, \
#                                  fc=fc, \
#                                  fs=fs)




#
#control_type = Control_Type.NEURON_EP
#controller_params = NeuronEPParams(gamma=1,
#                                   C=1,
#                                   fc=fc,
#                                   fs=fs)

# Neuron Fully Connected EP Parameters
# control_type = Control_Type.NEURON_EP2
# controller_params = NeuronEP2Params(gamma=1,
#                                    fc=fc,
#                                    fs=fs)
# controller_params.RL_type = "SAC"

# Baseline models
# control_type = Control_Type.BASELINE
# controller_params = BaselineParams(fc=fc, \
#                                    fs=fs)
# controller_params.RL_type = "SAC"

# episode length
episode_sec = 5
episode_length = int(episode_sec*(fs/fs_brain_factor))
num_episodes = 100


