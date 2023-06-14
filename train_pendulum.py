from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
#from stable_baselines3 import DDPG
import os
import pendulum_env
import double_links_env
import time

control_type = double_links_env.Control_Type.BASELINE

models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}-{double_links_env.control_typle_dic[control_type]}/"

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)

# env = pendulum_env.PendulumEnv(control_type=control_type)
# vec_env = make_vec_env(lambda: pendulum_env.PendulumEnv(control_type=control_type), n_envs=4)

env = double_links_env.DoubleLinkEnv(control_type=control_type)


model = PPO('MlpPolicy', env, device='cpu', n_steps=50000, batch_size=10000, n_epochs=100, verbose=1, tensorboard_log=logdir)
# PPO_model_path="models/1683788483/11830000.zip"
# model=PPO.load(PPO_model_path, env=env)
# model.verbose = 1
# model.tensorboard_log = logdir
# model.device="cuda"

TIMESTEPS = 10000
iters = 0
while True:
	iters += 1
	model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
	model.save(f"{models_dir}/{TIMESTEPS * iters}")
	print(f"modle saved {iters}\n")