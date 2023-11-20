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

from control import *
from parameters import *

training = True
class StopTraining(BaseCallback):
    parent: EvalCallback

    def __init__(self, reward_threshold: float, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.reward_threshold = reward_threshold

    def _on_step(self) -> bool:
        global training
        assert self.parent is not None, "``StopTrainingOnMinimumReward`` callback must be used with an ``EvalCallback``"
        continue_training = bool(self.parent.best_mean_reward < self.reward_threshold)
        if self.verbose >= 1 and not continue_training:
            print(
                f"Stopping training because the mean reward {self.parent.best_mean_reward:.2f} "
                f" is above the threshold {self.reward_threshold}"
            )
            training = False
        return continue_training

if __name__ == "__main__":

    models_dir = f"models/{int(time.time())}/"
    logdir = f"logs/{int(time.time())}-{control_type_dic[control_type]}/"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    env = double_links_env.DoubleLinkEnv(control_type=control_type, env_id=env_id, c_params=controller_params)#this will also update controller_params

    eval_env = env
    callback_on_best = StopTraining(reward_threshold=1, verbose=1)
    eval_callback = EvalCallback(eval_env, callback_on_new_best=callback_on_best, verbose=0)

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
    while training:
        iters += 1
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, \
                    tb_log_name=f"PPO", callback=eval_callback)
        model.save(f"{models_dir}/{TIMESTEPS * iters}")
