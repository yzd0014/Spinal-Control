import torch as th
from stable_baselines3 import SAC
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
import os
import double_links_env
import time
import pickle

from control import *
from parameters import *

if __name__ == "__main__":

    models_dir = f"models/{int(time.time())}/"
    logdir = f"logs/{int(time.time())}-{control_type_dic[control_type]}/"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    env = double_links_env.DoubleLinkEnv(control_type=control_type, c_params=controller_params)#this will also update controller_params

    pickle.dump([control_type, \
                 controller_params], \
                open(models_dir + "env_contr_params.p", "wb"))

    num_episodes = env.get_num_of_targets()
    print(f"total number of targets: {num_episodes}")

    policy_kwargs = dict(activation_fn=th.nn.Tanh, \
                         net_arch=dict(pi=[64, 64], \
                                       vf=[64, 64]))
    m_steps = controller_params.episode_length_in_ticks * num_episodes

    TIMESTEPS = m_steps
    # model = PPO('MlpPolicy', env, \
    #             policy_kwargs=policy_kwargs, \
    #             device='cpu', \
    #             n_steps=m_steps, \
    #             batch_size=controller_params.episode_length_in_ticks, \
    #             n_epochs=10, \
    #             verbose=1, \
    #             tensorboard_log=logdir)

    model = SAC("MlpPolicy", env,  \
                device='cpu', \
                batch_size = controller_params.episode_length_in_ticks, \
                train_freq = controller_params.episode_length_in_ticks, \
                verbose=1,\
                tensorboard_log=logdir)
    print(model.policy)

    iters = 0
    while True:
        iters += 1
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, \
                    tb_log_name=f"SAC")
        model.save(f"{models_dir}/{TIMESTEPS * iters}")
