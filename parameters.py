from control import *

SINGLE_PENDULUM = -1
DOUBLE_PENDULUM = 0
INVERTED_PENDULUM = 1
TOSS = 2
PUSH = 3
SWING = 4

# parameters that can be changed by users
control_type = Control_Type.SAC
env_id = INVERTED_PENDULUM
training_type = "SAC"

if env_id == DOUBLE_PENDULUM:
    controller_input_size = 2
    # controller_input_size = 4
elif env_id == INVERTED_PENDULUM:
    controller_input_size = 6
else:
    controller_input_size = 2

if control_type == Control_Type.BASELINE:
    controller_output_size = 4
elif control_type == Control_Type.PID or control_type == Control_Type.EP:
    controller_output_size = 2
else:
    controller_output_size = 1

if env_id == DOUBLE_PENDULUM:
    # xml = 'double_links_fast.xml'
    xml = 'arm26.xml'
    episode_length_in_seconds = 10
elif env_id == INVERTED_PENDULUM:
    # xml = 'inverted_pendulum_fast.xml'
    xml = 'arm26_inverted.xml'
    episode_length_in_seconds = 180
elif env_id == TOSS:
    xml = 'toss.xml'
    episode_length_in_seconds = 10000
elif env_id == PUSH:
    xml = 'push.xml'
    episode_length_in_seconds = 10
elif env_id == SWING:
    xml = 'inverted_pendulum_fast.xml'
    episode_length_in_seconds = 1.5
elif env_id == SINGLE_PENDULUM:
    # xml = 'single_link.xml'
    xml = 'muscle_control_narrow.xml'
    episode_length_in_seconds = 10

controller_params = ControllerParams(alpha=0.4691358024691358, \
                                    beta=0.9, \
                                    gamma=1, \
                                    fc=10, \
                                    model_dir = xml, \
                                    input_size=controller_input_size, \
                                    hidden_size=32, \
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
    elif control_type == SINGLE_PENDULUM:
        controller = TemplateController(controller_params, env_id)
    mj.set_mjcb_control(controller.callback)