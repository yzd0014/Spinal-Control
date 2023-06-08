import numpy as np

# all controllers are xml file (model) dependent
def baseline_controller(input_action, data):
    for i in range(4):
        data.ctrl[i] = input_action[i]

def RI_controller(input_action, data):
    for i in range(4):
        data.ctrl[i] = input_action[i]

    if input_action[0] > input_action[1]:
        data.ctrl[1] = 0
    if input_action[1] > input_action[0]:
        data.ctrl[0] = 0
    if input_action[2] > input_action[3]:
        data.ctrl[3] = 0
    if input_action[3] > input_action[2]:
        data.ctrl[2] = 0

def stretch_reflex_controller(input_action, data):
    normalize_factor = 0.677
    r_spindle = data.actuator_length[1] / normalize_factor
    l_spindle = data.actuator_length[2] / normalize_factor
    data.ctrl[1] = r_spindle - input_action[0]
    data.ctrl[2] = l_spindle - input_action[1]

def RI_and_stretch_reflex_controller(input_action, data):
    normalize_factor = 0.677
    r_spindle = data.actuator_length[1] / normalize_factor
    l_spindle = data.actuator_length[2] / normalize_factor
    data.ctrl[1] = r_spindle - input_action[0]
    data.ctrl[2] = l_spindle - input_action[1]

    a_square = input_action[0] * input_action[0]
    b_square = input_action[1] * input_action[1]
    theta_d = 0
    if a_square + b_square > 0.0000001:
        tmp = 3.08333 * (b_square - a_square) / (a_square + b_square)
        if tmp > 1:
            theta_d = 0
        elif tmp < -1:
            theta_d = np.pi
        else:
            theta_d = np.arccos(tmp)
        theta_d = np.pi * 0.5 - theta_d
    offset = 0.05
    if data.qpos[0] < theta_d - offset:
        data.ctrl[2] = 0
    if data.qpos[0] > theta_d + offset:
        data.ctrl[1] = 0

def neuron_controller(input_action, data):
    normalize_factor = 0.677
    l0 = input_action[0] * normalize_factor
    l1 = input_action[1] * normalize_factor

    r_spindle = 0.05 * data.actuator_velocity[1] + data.actuator_length[1]
    l_spindle = 0.05 * data.actuator_velocity[2] + data.actuator_length[2]
    inhibition_coeff = 2
    l_diff = inhibition_coeff * max(l_spindle - l1 - (r_spindle - l0), 0)
    r_diff = inhibition_coeff * max(r_spindle - l0 - (l_spindle - l1), 0)

    ctrl_coeff = 1
    data.ctrl[1] = max(ctrl_coeff * (r_spindle - l0 - l_diff), 0)
    data.ctrl[2] = max(ctrl_coeff * (l_spindle - l1 - r_diff), 0)