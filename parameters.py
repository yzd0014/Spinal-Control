from control import *

DOUBLE_PENDULUM = 0
INVERTED_PENDULUM = 1

# parameters that can be changed by users
control_type = Control_Type.BASELINE
env_id = INVERTED_PENDULUM
training_type = "N/A"

if control_type == Control_Type.BASELINE:
    controller_input_size = 6
    controller_output_size = 4
elif control_type == Control_Type.PID:
    if env_id == DOUBLE_PENDULUM:
        controller_input_size = 2
    elif env_id == INVERTED_PENDULUM:
        controller_input_size = 6
    controller_output_size = 2

if env_id == DOUBLE_PENDULUM:
    xml = 'double_links_fast.xml'
    episode_length_in_seconds = 2
elif env_id == INVERTED_PENDULUM:
    xml = 'inverted_pendulum_fast.xml'
    episode_length_in_seconds = 10

controller_params = ControllerParams(alpha=0.4691358024691358, \
                                    beta=0.9, \
                                    gamma=1, \
                                    fc=10, \
                                    model_dir = xml, \
                                    input_size=controller_input_size, \
                                    hidden_size=256, \
                                    output_size=controller_output_size, \
                                    episode_length_in_seconds=episode_length_in_seconds,\
                                    brain_dt=0.1)
#############################################################################################
# create controllers
if control_type == Control_Type.BASELINE:
    controller = BaselineController(controller_params)
elif control_type == Control_Type.PID:
    controller = PIDController()

# initialize mujoco
xml_path = controller_params.model_dir
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)
xml_path = abspath
model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
data = mj.MjData(model)  # MuJoCo data