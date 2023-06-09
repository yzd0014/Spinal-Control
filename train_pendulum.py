from stable_baselines3 import PPO
#from stable_baselines3 import DDPG
import os
from pendulum_env import *
from double_links_env import *
import time

control_type = Control_Type.RI

models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}-{control_typle_dic[control_type]}/"

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)

# env = PendulumEnv(control_type=control_type)
env = DoubleLinkEnv(control_type=control_type)
model = PPO('MlpPolicy', env, n_steps=8192, batch_size=2048, n_epochs=100, verbose=1, tensorboard_log=logdir)
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