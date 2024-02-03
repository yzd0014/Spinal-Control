import torch as th
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
import os
import invertpendulum_env
import time
import pickle
from stable_baselines3.common.utils import safe_mean

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
               episode_sec,
               num_episodes,
               fs_brain_factor,
               controller_params],
               open(models_dir + "env_contr_params.p", "wb"))

  env = invertpendulum_env.InvertPendulumEnv(control_type=control_type,
                                             episode_sec=episode_sec,
                                             fs_brain_factor=fs_brain_factor,
                                             c_params=controller_params)


  m_steps = episode_length*num_episodes
  m_steps = 1000
  TIMESTEPS = m_steps

  # PPO -----------------------------------------------------------------------
  if controller_params.RL_type == "PPO":
    policy_kwargs = dict(activation_fn=th.nn.Tanh,
                          net_arch=dict(pi=[64, 64],
                          vf=[64, 64]))
    model = PPO('MlpPolicy', env,
                  device='cpu',
                  learning_rate=0.0003,
                  n_steps=m_steps,
                  batch_size=m_steps,
                  n_epochs=20,
                  gamma=0.99,
                  gae_lambda=0.95,
                  clip_range=0.2,
                  clip_range_vf=None,
                  ent_coef=0,
                  vf_coef=0.5,
                  target_kl=None,
                  use_sde=False,
                  policy_kwargs=policy_kwargs,
                  verbose=1,
                  tensorboard_log=logdir)


  # SAC -----------------------------------------------------------------------
  elif controller_params.RL_type == "SAC":
    policy_kwargs = dict(activation_fn=th.nn.ReLU,
                          net_arch=dict(pi=[256, 256],
                                        qf=[256, 256]))
    model = SAC("MlpPolicy", env,
                policy_kwargs=policy_kwargs,
                batch_size = 256,
                train_freq = (1, "episode"),
                buffer_size = 1000000,
                tau = 0.005,
                gamma = 0.99,
                target_update_interval = 1,
                learning_starts= 500,
                use_sde = True,
                use_sde_at_warmup = True,
                sde_sample_freq=64,
                verbose=1,
                gradient_steps = -1,
                tensorboard_log=logdir)

  print(model.policy)
  reward_target = 100

  iters = 0
  while True:
    iters += 1
    model.learn(total_timesteps=TIMESTEPS,reset_num_timesteps=False,
                tb_log_name=controller_params.RL_type,progress_bar=False)
    model.save(f"{models_dir}/{TIMESTEPS * iters}")
    mean_reward = safe_mean([ep_info["r"] for ep_info in model.ep_info_buffer])
    if mean_reward > reward_target:
        break
