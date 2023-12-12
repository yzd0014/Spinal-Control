from control import *

DOUBLE_PENDULUM_DAMPING = -1
DOUBLE_PENDULUM = 0
INVERTED_PENDULUM = 1
TOSS = 2
PUSH = 3
SWING = 4

# parameters that can be changed by users
control_type = Control_Type.DEFAULT
env_id = DOUBLE_PENDULUM_DAMPING
training_type = "feedforward"

if env_id == DOUBLE_PENDULUM:
    controller_input_size = 2
    # controller_input_size = 4
elif env_id == INVERTED_PENDULUM:
    controller_input_size = 6
else:
    controller_input_size = 6

if control_type == Control_Type.BASELINE:
    controller_output_size = 4
elif control_type == Control_Type.PID or control_type == Control_Type.EP:
    controller_output_size = 2
else:
    controller_output_size = 2

if env_id == DOUBLE_PENDULUM:
    xml = 'double_links_fast.xml'
    episode_length_in_seconds = 5
elif env_id == INVERTED_PENDULUM:
    xml = 'inverted_pendulum_fast.xml'
    episode_length_in_seconds = 120
elif env_id == TOSS:
    xml = 'toss.xml'
    episode_length_in_seconds = 1000
elif env_id == PUSH:
    xml = 'slider.xml'
    episode_length_in_seconds = 20
elif env_id == SWING:
    xml = 'inverted_pendulum_fast.xml'
    episode_length_in_seconds = 10
else:
    xml = 'double_links_damping.xml'
    episode_length_in_seconds = 5

controller_params = ControllerParams(alpha=0.4691358024691358, \
                                    beta=0.9, \
                                    gamma=1, \
                                    fc=10, \
                                    model_dir = xml, \
                                    input_size=controller_input_size, \
                                    hidden_size=8, \
                                    output_size=controller_output_size, \
                                    episode_length_in_seconds=episode_length_in_seconds,\
                                    brain_dt=0.1)
#############################################################################################
if training_type == "feedforward":
    # initialize mujoco
    xml_path = controller_params.model_dir
    dirname = os.path.dirname(__file__)
    abspath = os.path.join(dirname + "/" + xml_path)
    xml_path = abspath
    model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
    data = mj.MjData(model)  # MuJoCo data

    # create controllers
    if control_type == Control_Type.BASELINE:
        controller = BaselineController(controller_params, env_id)
    elif control_type == Control_Type.PID:
        controller = PIDController()
    elif control_type == Control_Type.EP:
        controller = EPController()
    else:
        controller = TemplateController(controller_params, env_id)
    mj.set_mjcb_control(controller.callback)