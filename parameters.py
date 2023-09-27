from control import *

# Choose controller here

# Neuron Parameters
control_type = Control_Type.PID
controller_params = ControllerParams(alpha=0.4691358024691358, \
                                    beta=0.9, \
                                    gamma=1, \
                                    fc=10, \
                                    episode_length_in_seconds=2.5,\
                                    brain_dt=0.05)
