from stable_baselines3 import PPO
import mujoco as mj
import numpy as np
import os
import double_links_env
import torch as th
from scipy.optimize import minimize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

SLOW = 0
FAST = 1

env = double_links_env.DoubleLinkEnv(env_id=1, control_type=double_links_env.Control_Type.NEURON_TEST, speed_mode=FAST)
env_with_monitor = Monitor(env)
num_episodes = env.get_num_of_targets()

policy_kwargs = dict(activation_fn=th.nn.Tanh, net_arch=dict(pi=[3, 3], vf=[3, 3]))
episode_length = 500
m_steps = episode_length * num_episodes

def callback_function(xk):
    print(f"alpha: {xk[0]}, beta: {xk[1]}")

def f(x):
    env.update_neuron_weights(x)
    env.reset()

    #train the model
    model = PPO('MlpPolicy', env_with_monitor, policy_kwargs=policy_kwargs, device='auto', n_steps=m_steps, batch_size=episode_length, n_epochs=10, verbose=1)
    training_steps = 500
    model.learn(total_timesteps=training_steps)

    #evaluate the model
    mean_reward, std_reward = evaluate_policy(model, env_with_monitor, n_eval_episodes=2)
    print(f"reward after {training_steps} steps: {mean_reward}")
    return mean_reward

x0 = [0.47, 0.9]
bounds = ((0, 4), (0, 0.9999))
result = minimize(f, x0, bounds=bounds, callback=callback_function)
if result.success:
    print(f"success with x: {result.x}")
