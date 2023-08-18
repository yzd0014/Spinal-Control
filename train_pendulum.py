import torch as th
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
import os
import pendulum_env
import double_links_env
import time

SINGLE_PENDULUM = 0
DOUBLE_PENDULUM = 1
INVERTED_PENDULUM = 2

SLOW = 0
FAST = 1

def make_env(instance_id, env_id, speed_mode, control_type):
	return lambda: Monitor(double_links_env.DoubleLinkEnv(control_type=control_type, env_id=env_id, instance_id=instance_id, speed_mode=speed_mode))

if __name__ =="__main__":
	# 0: single pendulum, 1: double pendulum, 2: inverted pendulum, 3: double pendulum with extra observation
	speed_mode = FAST
	env_id = DOUBLE_PENDULUM
	models_dir = f"models/{int(time.time())}/"

	if env_id == 0:
		control_type = pendulum_env.Control_Type.NEURON
		logdir = f"logs/{int(time.time())}-{pendulum_env.control_typle_dic[control_type]}/"
	elif env_id == 1 or env_id == 2 or env_id == 3:
		control_type = double_links_env.Control_Type.NEURON
		logdir = f"logs/{int(time.time())}-{double_links_env.control_typle_dic[control_type]}/"


	if not os.path.exists(models_dir):
		os.makedirs(models_dir)

	if not os.path.exists(logdir):
		os.makedirs(logdir)

	if speed_mode == SLOW:
		episode_length = 5000
	elif speed_mode == FAST:
		episode_length = 50
	if env_id == 0:
		num_episodes = 5
		env = pendulum_env.PendulumEnv(control_type=control_type, speed_mode=speed_mode)
		# vec_env = make_vec_env(lambda: pendulum_env.PendulumEnv(control_type=control_type), n_envs=4)
	elif env_id == 1:
		env = double_links_env.DoubleLinkEnv(control_type=control_type, speed_mode=speed_mode)
		num_episodes = env.get_num_of_targets()
		print(f"total number of targets: {num_episodes}")
	elif env_id == 2 or env_id == 3:
		dummy_env = double_links_env.DoubleLinkEnv(control_type=control_type, speed_mode=speed_mode)
		num_episodes = dummy_env.get_num_of_targets()
		print(f"total number of targets: {num_episodes}")

		THREADS_NUM = 16
		env_fns = [make_env(i, env_id, speed_mode, control_type) for i in range(THREADS_NUM)]
		env = SubprocVecEnv(env_fns)

	policy_kwargs = dict(activation_fn=th.nn.Tanh, net_arch=dict(pi=[8, 8], vf=[64, 64]))
	m_steps = episode_length * num_episodes
	#n_steps=50000 batch_size=10000,n_epochs=10 tested
	#model = PPO('MlpPolicy', env, device='cpu', n_steps=m_steps, batch_size=1000, n_epochs=100, verbose=1, tensorboard_log=logdir) - 1689548375
	if env_id == INVERTED_PENDULUM:
		TIMESTEPS = 10000
		model = PPO('MlpPolicy', env, policy_kwargs=policy_kwargs, device='cpu', n_steps=1000, batch_size=1000, n_epochs=10, verbose=1, tensorboard_log=logdir)
	else:
		TIMESTEPS = m_steps
		model = PPO('MlpPolicy', env, policy_kwargs=policy_kwargs, device='cpu', n_steps=m_steps, batch_size=episode_length, n_epochs=10, verbose=1, tensorboard_log=logdir)
		print(model.policy)
	# model = PPO('MlpPolicy', env, device='cpu', n_steps=50000, batch_size=10000, n_epochs=100, verbose=1, tensorboard_log=logdir)
	# PPO_model_path="models/1688353130/24640000.zip"
	# model=PPO.load(PPO_model_path, env=env)
	# model.verbose = 1
	# model.tensorboard_log = logdir
	# model.device="cuda"

	# model = A2C('MlpPolicy', env, device='auto', n_steps=10, gamma=0.1, gae_lambda = 0.5, verbose=1, tensorboard_log=logdir)
	# model = TD3('MlpPolicy', env, device='cpu', gamma=0.98, buffer_size=200000, learning_starts=10000, policy_kwargs=dict(net_arch=[400, 300]), verbose=1, tensorboard_log=logdir)
	# model =TD3('MlpPolicy', env, device='cpu', verbose=1,tensorboard_log=logdir)

	iters = 0
	while True:
		iters += 1
		model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
		model.save(f"{models_dir}/{TIMESTEPS * iters}")
		print(f"modle saved {iters}\n")