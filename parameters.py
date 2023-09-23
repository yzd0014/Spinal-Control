from control import *

# Choose controller here

# Neuron Parameters
control_type = Control_Type.NEURON_OPTIMAL
controller_params = NeuronParams(alpha=0.4691358024691358, \
                                  beta=0.9, \
                                  gamma=1, \
                                  fc=10, \
                                  fs=200)

# Baseline models
#control_type = Control_Type.BASELINE
#controller_params = BaselineParams(fc=10, \
#                                    fs=200)

# episode length
episode_length = 50

# loop factor for brain sampling
fs_brain_factor = 20

# For Simulations
#modelid = "neuron-iir-10hz-best"
#runid = "2137600"
#
##modelid = "baseline-iir-10hz"
##runid = "1973600"
