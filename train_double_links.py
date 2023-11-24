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
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import safe_mean
from control import *
from parameters import *

if __name__ == "__main__":

    models_dir = f"models/{int(time.time())}/"
    logdir = f"logs/{int(time.time())}-{control_type_dic[control_type]}/"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    env = double_links_env.DoubleLinkEnv(control_type=control_type, env_id=env_id, c_params=controller_params)#this will also update controller_params

    training_type = "PPO"
    pickle.dump([training_type, \
                control_type, \
                 env_id, \
                controller_params], \
                open(models_dir + "env_contr_params.p", "wb"))

    num_episodes = env.get_num_of_targets()
    print(f"total number of targets: {num_episodes}")

    policy_kwargs = dict(activation_fn=th.nn.Tanh, \
                         net_arch=dict(pi=[64, 64], \
                                       vf=[64, 64]))
    if env_id == DOUBLE_PENDULUM:
        n_steps = controller_params.episode_length_in_ticks * num_episodes
        batch_size = controller_params.episode_length_in_ticks
        n_epochs = 10
    elif env_id == INVERTED_PENDULUM:
        n_steps = int(100/controller_params.brain_dt)
        batch_size = n_steps
        n_epochs = 20

    TIMESTEPS = n_steps
    model = PPO('MlpPolicy', env, \
                policy_kwargs=policy_kwargs, \
                device='cpu', \
                n_steps=n_steps, \
                batch_size=batch_size, \
                n_epochs=n_epochs, \
                verbose=1, \
                tensorboard_log=logdir)

    # model = SAC("MlpPolicy", env,  \
    #             device='cpu', \
    #             batch_size = controller_params.episode_length_in_ticks, \
    #             train_freq = controller_params.episode_length_in_ticks, \
    #             verbose=1,\
    #             tensorboard_log=logdir)
    print(model.policy)

    iters = 0
    while True:
        iters += 1
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, \
                    tb_log_name=f"PPO")
        mean_reward = safe_mean([ep_info["r"] for ep_info in model.ep_info_buffer])
        if mean_reward > 120:
            break
        model.save(f"{models_dir}/{TIMESTEPS * iters}")
