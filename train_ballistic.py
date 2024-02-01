import torch as th
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
import os
import double_links_ballistic_env
import time
import pickle

from control import *
from parameters import *

if __name__=="__main__":

  models_dir = f"models/{int(time.time())}/"
  logdir = f"logs/{int(time.time())}-{control_type_dic[control_type]}/"

  if not os.path.exists(models_dir):
    os.makedirs(models_dir)

  if not os.path.exists(logdir):
    os.makedirs(logdir)

  pickle.dump([control_type,
               episode_length,
               num_episodes,
               fs_brain_factor,
               controller_params],
               open(models_dir + "env_contr_params.p", "wb"))

  env = double_links_ballistic_env.DoubleLinkEnv(control_type=control_type,
                                fs_brain_factor=fs_brain_factor,
                                c_params=controller_params)


  policy_kwargs = dict(activation_fn=th.nn.Tanh,
                        net_arch=dict(pi=[8, 8],
                        vf=[64, 64]))

  m_steps = episode_length*num_episodes
  TIMESTEPS = m_steps

  print(episode_length)
  print(num_episodes)
  exit()
  model = PPO('MlpPolicy', env,
                policy_kwargs=policy_kwargs,
                device='auto',
                n_steps=m_steps*10,
                batch_size=10,
                n_epochs=10,
                verbose=0,
                tensorboard_log=logdir)
  print(model.policy)

  iters = 0
  while True:
    iters += 1
    model.learn(total_timesteps=TIMESTEPS,reset_num_timesteps=False,
                tb_log_name=f"PPO",progress_bar=True)
    model.save(f"{models_dir}/{TIMESTEPS * iters}")
