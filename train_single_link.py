import torch as th
import single_link_env
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

    env = single_link_env.SingleLinkEnv(control_type=control_type, env_id=env_id, c_params=controller_params)#this will also update controller_params

    training_type = "PPO"
    pickle.dump([training_type, \
                control_type, \
                 env_id, \
                controller_params], \
                open(models_dir + "env_contr_params.p", "wb"))

    policy_kwargs = dict(activation_fn=th.nn.Tanh, \
                         net_arch=dict(pi=[64, 64], \
                                       vf=[64, 64]))
    reward_target = 0
    if env_id == SINGLE_PENDULUM:
        n_steps = controller_params.episode_length_in_ticks * 10
        batch_size = controller_params.episode_length_in_ticks
        n_epochs = 10
        reward_target = -0.8
        learning_rate = 0.0003

    TIMESTEPS = n_steps
    model = PPO('MlpPolicy', env, \
                policy_kwargs=policy_kwargs, \
                device='cpu', \
                n_steps=n_steps, \
                batch_size=batch_size, \
                n_epochs=n_epochs, \
                learning_rate=learning_rate, \
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
        if mean_reward > reward_target:
            break
        model.save(f"{models_dir}/{TIMESTEPS * iters}")
    file = open(logdir + "num_timesteps.txt", "w")
    file.write(str(TIMESTEPS * iters))
    file.close()