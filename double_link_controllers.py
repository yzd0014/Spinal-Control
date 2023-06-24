import numpy as np
from enum import Enum

class Control_Type(Enum):
    BASELINE = 1
    RI = 2
    REFLEX = 3
    RI_AND_REFLEX = 4
    NEURON = 5

control_typle_dic = {Control_Type.BASELINE: "baseline",
                     Control_Type.RI: "RI",
                     Control_Type.REFLEX: "strech reflex",
                     Control_Type.RI_AND_REFLEX: "RI + stretch refelx",
                     Control_Type.NEURON: "neuron model"}

# all controllers are xml file (model) dependent
def baseline_controller(input_action, data):
    for i in range(4):
        data.ctrl[i] = input_action[i]

def RI_controller(input_action, data):
    for i in range(4):
        data.ctrl[i] = input_action[i]

    offset = 0.2
    if input_action[0] > input_action[1] + offset:
        data.ctrl[1] = 0
    if input_action[1] > input_action[0] + offset :
        data.ctrl[0] = 0
    if input_action[2] > input_action[3] + offset:
        data.ctrl[3] = 0
    if input_action[3] > input_action[2] + offset:
        data.ctrl[2] = 0

def stretch_reflex_controller(input_action, data):
    normalize_factor = 0.677
    for i in range(4):
        data.ctrl[i+1] = data.actuator_length[i+1] / normalize_factor - input_action[i]

def RI_and_stretch_reflex_controller(input_action, data):
   pass

def neuron_controller(input_action, data):
    normalize_factor = 0.677
    for i in range(2):
        length_r = input_action[i*4] * normalize_factor
        length_l = input_action[i*4+1] * normalize_factor

        r_spindle = 0.05 * data.actuator_velocity[i*2+1] + data.actuator_length[i*2+1]
        l_spindle = 0.05 * data.actuator_velocity[i*2+2] + data.actuator_length[i*2+2]
        inhibition_coeff = 2
        beta = 0.9
        l_diff = inhibition_coeff / (beta * beta) * max((l_spindle - beta * r_spindle + beta * input_action[i*4+2] - input_action[i*4+3]), 0)
        r_diff = inhibition_coeff / (beta * beta) * max((r_spindle - beta * l_spindle + beta * input_action[i*4+3] - input_action[i*4+2]), 0)

        ctrl_coeff = 1
        data.ctrl[i*2+1] = max(ctrl_coeff * (r_spindle - length_r - l_diff), 0)
        data.ctrl[i*2+2] = max(ctrl_coeff * (l_spindle - length_l - r_diff), 0)