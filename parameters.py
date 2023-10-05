from control import *

# Choose controller here

# Neuron Parameters
training_type = "N/A"
control_type = Control_Type.BASELINE
controller_params = ControllerParams(alpha=0.4691358024691358, \
                                    beta=0.9, \
                                    gamma=1, \
                                    fc=10, \
                                    input_size=6, \
                                    hidden_size=8, \
                                    output_size=4, \
                                    episode_length_in_seconds=2,\
                                    brain_dt=0.1)
