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
    data.ctrl[1] = input_action[0]
    data.ctrl[2] = input_action[1]

def RI_controller(input_action, data):
    data.ctrl[1] = input_action[0]
    data.ctrl[2] = input_action[1]

    if data.ctrl[2] > data.ctrl[1] + 0.2:
        data.ctrl[1] = 0
    if data.ctrl[1] > data.ctrl[2] + 0.2:
        data.ctrl[2] = 0

def stretch_reflex_controller(input_action, data):
    normalize_factor = 0.677
    r_spindle = data.actuator_length[1] / normalize_factor
    l_spindle = data.actuator_length[2] / normalize_factor
    data.ctrl[1] = r_spindle - input_action[0]
    data.ctrl[2] = l_spindle - input_action[1]

def neuron_controller(input_action, data):
    normalize_factor = 0.677
    l0 = input_action[0] * normalize_factor
    l1 = input_action[1] * normalize_factor

    r_spindle = 0.05 * data.actuator_velocity[1] + data.actuator_length[1]
    l_spindle = 0.05 * data.actuator_velocity[2] + data.actuator_length[2]
    inhibition_coeff = 0.15
    beta = 0.9

    l_diff = inhibition_coeff / (1-beta * beta) * max((l_spindle - beta * r_spindle + beta * input_action[2] - input_action[3]), 0)
    r_diff = inhibition_coeff / (1-beta * beta) * max((r_spindle - beta * l_spindle + beta * input_action[3] - input_action[2]), 0)

    ctrl_coeff = 1
    data.ctrl[1] = max(ctrl_coeff * (r_spindle - l0 - l_diff), 0)
    data.ctrl[2] = max(ctrl_coeff * (l_spindle - l1 - r_diff), 0)

def joint0_controller(model, data):
    kp = 0.5
    noise = np.random.randn(1)
    data.ctrl[0] = kp * noise
