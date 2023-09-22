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
    A = np.zeros((4, 8))
    for i in range(4):
        A[0, i] = x[i]
    for i in range(0, 4, 2):
        A[1, i] = A[0, i+1]
        A[1, i+1] = A[0, i]
    for i in range(3, -1, -1):
        A[3, i] = x[i]
    for i in range(0, 4, 2):
        A[2, i] = A[3, i+1]
        A[2, i+1] = A[3, i]

    A[0, 4] = x[4]
    A[0, 5] = x[5]
    A[1, 4] =  A[0, 5]
    A[1, 5] =  A[0, 4]
    A[2, 6] =  A[0, 4]
    A[2, 7] =  A[0, 5]
    A[3, 6] =  A[0, 5]
    A[3, 7] =  A[0, 4]

    env.set_spinal_weights(A)

    total_pen = 0
    for i in range(num_episodes):
        env.reset()
        ep_pen = 0
        for j in range(episode_length):
            ep_pen += env.step()
        total_pen += ep_pen

    # print("objective function evaluated!")
    return total_pen

x0 = np.zeros(6)
bnds=[(0, 1)] * 6
# reward = my_f(x0)
# print(reward)
# result = minimize(my_f, x0, bounds=bnds, callback=callback_function)
result = minimize(my_f, x0, method='CG', callback=callback_function)
if result.success:
    print(f"success with x: {result.x}")

env.close()