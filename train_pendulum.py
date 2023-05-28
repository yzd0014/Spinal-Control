from stable_baselines3 import PPO
#from stable_baselines3 import DDPG
import os
from pendulum_env import PendulumEnv
import time

models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)

env = PendulumEnv()
env.reset()

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)
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