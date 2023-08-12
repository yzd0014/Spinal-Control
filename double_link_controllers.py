import numpy as np
from enum import Enum

class Control_Type(Enum):
    BASELINE = 1
    NEURON_FILTER = 2
    REFLEX = 3
    NEURON_TEST = 4
    NEURON = 5
    NEURON_SIMPLE = 6

control_typle_dic = {Control_Type.BASELINE: "baseline",
                     Control_Type.NEURON_FILTER: "neuron filter",
                     Control_Type.REFLEX: "strech reflex",
                     Control_Type.NEURON_TEST: "neuron test",
                     Control_Type.NEURON: "neuron model",
                     Control_Type.NEURON_SIMPLE: "neuron model simple"
                     }

# all controllers are xml file (model) dependent
def baseline_controller(input_action, data):
    for i in range(4):
        data.ctrl[i] = input_action[i]

l_diff_prev = [0, 0]
r_diff_prev = [0, 0]
l_ctrl_prev = [0, 0]
r_ctrl_prev = [0, 0]
Ka = 0.3
def neuron_filter_controller(input_action, data):
    normalize_factor = 0.677
    for i in range(2):
        length_r = input_action[i * 2] * normalize_factor
        length_l = input_action[i * 2 + 1] * normalize_factor

        r_spindle = 0.05 * data.actuator_velocity[i * 2] + data.actuator_length[i * 2]
        l_spindle = 0.05 * data.actuator_velocity[i * 2 + 1] + data.actuator_length[i * 2 + 1]
        inhibition_coeff = 0.4691358024691358
        beta = 0.9
        l_diff = inhibition_coeff / (1 - beta * beta) * max((l_spindle - beta * r_spindle + beta * input_action[i * 2] - input_action[i * 2 + 1]), 0)
        l_diff = l_diff_prev[i] * (1 - Ka) + l_diff * Ka
        r_diff = inhibition_coeff / (1 - beta * beta) * max((r_spindle - beta * l_spindle + beta * input_action[i * 2 + 1] - input_action[i * 2]), 0)
        r_diff = r_diff_prev[i] * (1 - Ka) + r_diff * Ka
        l_diff_prev[i] = l_diff
        r_diff_prev[i] = r_diff

        ctrl_coeff = 1
        r_ctrl = max(ctrl_coeff * (r_spindle - length_r - l_diff), 0)
        data.ctrl[i * 2] = r_ctrl_prev[i] * (1 - Ka) + r_ctrl * Ka
        l_ctrl = max(ctrl_coeff * (l_spindle - length_l - r_diff), 0)
        data.ctrl[i * 2 + 1] = l_ctrl_prev[i] * (1 - Ka) + l_ctrl * Ka
        r_ctrl_prev[i] = data.ctrl[i * 2]
        l_ctrl_prev[i] = data.ctrl[i * 2 + 1]


def stretch_reflex_controller(input_action, data):
    normalize_factor = 0.677
    for i in range(4):
        data.ctrl[i] = data.actuator_length[i] / normalize_factor - input_action[i]

def neuron_controller(input_action, data):
    normalize_factor = 0.677
    for i in range(2):
        length_r = input_action[i*4] * normalize_factor
        length_l = input_action[i*4+1] * normalize_factor

        r_spindle = 0.05 * data.actuator_velocity[i*2] + data.actuator_length[i*2]
        l_spindle = 0.05 * data.actuator_velocity[i*2+1] + data.actuator_length[i*2+1]
        inhibition_coeff = 0.4691358024691358
        beta = 0.9
        l_diff = inhibition_coeff / (1-beta * beta) * max((l_spindle - beta * r_spindle + beta * input_action[i*4+2] - input_action[i*4+3]), 0)
        r_diff = inhibition_coeff / (1-beta * beta) * max((r_spindle - beta * l_spindle + beta * input_action[i*4+3] - input_action[i*4+2]), 0)

        # inhibition_coeff = 2.469135802469136
        # beta = 0.9
        # l_diff = inhibition_coeff * max((l_spindle - beta * r_spindle + beta * input_action[i*4+2] - input_action[i*4+3]), 0)
        # r_diff = inhibition_coeff * max((r_spindle - beta * l_spindle + beta * input_action[i*4+3] - input_action[i*4+2]), 0)

        ctrl_coeff = 1
        data.ctrl[i*2] = max(ctrl_coeff * (r_spindle - length_r - l_diff), 0)
        data.ctrl[i*2+1] = max(ctrl_coeff * (l_spindle - length_l - r_diff), 0)

def neuron_test_controller(input_action, data, i_alpha, i_beta):
    normalize_factor = 0.677
    for i in range(2):
        length_r = input_action[i * 4] * normalize_factor
        length_l = input_action[i * 4 + 1] * normalize_factor

        r_spindle = 0.05 * data.actuator_velocity[i * 2] + data.actuator_length[i * 2]
        l_spindle = 0.05 * data.actuator_velocity[i * 2 + 1] + data.actuator_length[i * 2 + 1]
        inhibition_coeff = i_alpha
        beta = i_beta
        l_diff = inhibition_coeff / (1 - beta * beta) * max((l_spindle - beta * r_spindle + beta * input_action[i * 4 + 2] - input_action[i * 4 + 3]), 0)
        r_diff = inhibition_coeff / (1 - beta * beta) * max((r_spindle - beta * l_spindle + beta * input_action[i * 4 + 3] - input_action[i * 4 + 2]), 0)

        ctrl_coeff = 1
        data.ctrl[i * 2] = max(ctrl_coeff * (r_spindle - length_r - l_diff), 0)
        data.ctrl[i * 2 + 1] = max(ctrl_coeff * (l_spindle - length_l - r_diff), 0)

def neuron_simple_controller(input_action, data):
    normalize_factor = 0.677
    for i in range(2):
        length_r = input_action[i*2] * normalize_factor
        length_l = input_action[i*2+1] * normalize_factor

        r_spindle = 0.05 * data.actuator_velocity[i*2] + data.actuator_length[i*2]
        l_spindle = 0.05 * data.actuator_velocity[i*2+1] + data.actuator_length[i*2+1]
        inhibition_coeff = 0.4691358024691358
        beta = 0.9
        l_diff = inhibition_coeff / (1-beta * beta) * max((l_spindle - beta * r_spindle + beta * input_action[i*2] - input_action[i*2+1]), 0)
        r_diff = inhibition_coeff / (1-beta * beta) * max((r_spindle - beta * l_spindle + beta * input_action[i*2+1] - input_action[i*2]), 0)

        ctrl_coeff = 1
        data.ctrl[i*2] = max(ctrl_coeff * (r_spindle - length_r - l_diff), 0)
        data.ctrl[i*2+1] = max(ctrl_coeff * (l_spindle - length_l - r_diff), 0)

    # for i in range(2):
    #     descend_ctrl = np.array([input_action[i * 2], input_action[i * 2 + 1]])
    #     # ctrl_mat = np.array([[-4, 4.5, 5, 4.5],[4.5, -4, 4.5, 5]])
    #     ctrl_mat = np.array([[-1, 1], [1, -1]])
    #     ctrl_output = np.matmul(ctrl_mat, descend_ctrl)
    #     data.ctrl[i * 2] = ctrl_output[0]
    #     data.ctrl[i * 2 + 1] = ctrl_output[1]

def joints_controller(data):
    kp = 1
    data.ctrl[4] = kp * np.random.randn(1)
    data.ctrl[5] = kp * np.random.randn(1)