from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
#from stable_baselines3 import DDPG
import os
import pendulum_env
import double_links_env
import time

env_id = 1

models_dir = f"models/{int(time.time())}/"

control_type = pendulum_env.Control_Type.NEURON
logdir = f"logs/{int(time.time())}-{pendulum_env.control_typle_dic[control_type]}/"
if env_id == 1:
	control_type = double_links_env.Control_Type.BASELINE
	logdir = f"logs/{int(time.time())}-{double_links_env.control_typle_dic[control_type]}/"


if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)

env = pendulum_env.PendulumEnv(control_type=control_type)
# vec_env = make_vec_env(lambda: pendulum_env.PendulumEnv(control_type=control_type), n_envs=4)
if env_id == 1:
	env = double_links_env.DoubleLinkEnv(control_type=control_type)

episode_length = 10000
if env_id == 0:
	num_episodes = 5
if env_id == 1:
	num_episodes = double_links_env.num_of_targets
	print(f"total number of targets: {num_episodes}")
m_steps = episode_length * num_episodes
model = PPO('MlpPolicy', env, device='cpu', n_steps=m_steps, batch_size=10000, n_epochs=10, verbose=1, tensorboard_log=logdir)
# model = PPO('MlpPolicy', env, device='cpu', n_steps=50000, batch_size=10000, n_epochs=100, verbose=1, tensorboard_log=logdir)
# PPO_model_path="models/1688353130/24640000.zip"
# model=PPO.load(PPO_model_path, env=env)
# model.verbose = 1
# model.tensorboard_log = logdir
# model.device="cuda"

#TIMESTEPS = 50000
TIMESTEPS = m_steps
iters = 0
while True:
	iters += 1
	model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
	model.save(f"{models_dir}/{TIMESTEPS * iters}")
	print(f"modle saved {iters}\n")