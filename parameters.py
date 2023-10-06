from control import *

# Choose controller here

# Neuron Parameters
training_type = "N/A"
control_type = Control_Type.BASELINE

if control_type == Control_Type.BASELINE:
    controller_input_size = 6
    controller_output_size = 4
elif control_type == Control_Type.PID:
    controller_input_size = 2
    controller_output_size = 2
controller_params = ControllerParams(alpha=0.4691358024691358, \
                                    beta=0.9, \
                                    gamma=1, \
                                    fc=10, \
                                    input_size=controller_input_size, \
                                    hidden_size=8, \
                                    output_size=controller_output_size, \
                                    episode_length_in_seconds=2,\
                                    brain_dt=0.1)
