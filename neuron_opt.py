from stable_baselines3 import PPO
import mujoco as mj
import numpy as np
import os
import double_links_env
import my_double_pendulum_env
import torch as th
from scipy.optimize import minimize
from scipy.optimize import OptimizeResult
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

episode_length =100
# env = double_links_env.DoubleLinkEnv(env_id=1, control_type=double_links_env.Control_Type.NEURON_TEST, speed_mode=FAST)
# env_with_monitor = Monitor(env)
env = my_double_pendulum_env.MyDoublePendulumEnv(i_episode_length=episode_length)
num_episodes = env.get_num_of_targets()

def callback_function(xk):
    current_value = my_f(xk)
    print(f"current value: {current_value}")

def my_f(x):
    Beta = np.zeros((4, 4))
    k = 0
    for i in range(4):
        for j in range(4):
            if i != j:
                Beta[i, j] = x[k]
                k += 1

    Alpha = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            Alpha[i, j] = x[k]
            k += 1

    env.set_spinal_weights(Beta, Alpha)

    total_pen = 0
    for i in range(num_episodes):
        env.reset()
        ep_pen = 0
        for j in range(episode_length):
            ep_pen += env.step()
        total_pen += ep_pen

    # print("objective function evaluated!")
    return total_pen

x0 = np.ones(28)
bnds=[(0, 1)] * 6
# reward = my_f(x0)
# print(reward)
# result = minimize(my_f, x0, bounds=bnds, callback=callback_function)
result = minimize(my_f, x0, method='CG', callback=callback_function)
if result.success:
    print(f"success with x: {result.x}")

env.close()