import torch as th
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
import os
import double_links_env
import time
import pickle
import sys

from control import *
from parameters import *

# model to load
#modelid = "1694296421"
#modelid = "1694296257"


def cont_train_double_links(modelid):

  models_dir = "./models/" + modelid + "/"
  print("\n\n")
  print("loading env and control parameters " + models_dir + "\n")

  control_type, \
  episode_length, \
  num_episodes, \
  fs_brain_factor, \
  controller_params  = pickle.load(open(models_dir \
                                    + "env_contr_params.p", "rb"))
  logdir = "./logs/" + modelid + f"-{control_type_dic[control_type]}/"

  # Find most recent model
  allmodels = sorted(os.listdir(models_dir))
  allmodels.sort(key=lambda fn: \
                 os.path.getmtime(os.path.join(models_dir, fn)))

  runid = allmodels[-1].split(".")
  runid = runid[0]

  print("loading " + models_dir + runid + "\n")
  print("logging " + logdir + "\n")

  env = double_links_env.DoubleLinkEnv(control_type=control_type, \
                                fs_brain_factor=fs_brain_factor, \
                                c_params=controller_params)

  m_steps = episode_length*num_episodes
  TIMESTEPS = m_steps

  model2load = models_dir + runid + ".zip"
  model = PPO.load(model2load,env=env)
  iters = int(int(runid) / TIMESTEPS)

  while True:
    iters += 1
    model.learn(total_timesteps=TIMESTEPS,reset_num_timesteps=False, \
                tb_log_name=f"PPO",progress_bar=True)
    model.save(f"{models_dir}{TIMESTEPS*iters}")


if __name__=="__main__":
  cont_train_double_links(*sys.argv[1:])
