from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
import os
import pendulum_env
import double_links_env
import time

SINGLE_PENDULUM = 0
DOUBLE_PENDULUM = 1
INVERTED_PENDULUM = 2
DOUBLE_PENDULUM_WITH_EXTRA_OBSERVATION = 3

def make_env(instance_id, env_id):
	return lambda: Monitor(double_links_env.DoubleLinkEnv(control_type=control_type, env_id=env_id, instance_id=instance_id))

if __name__ =="__main__":
	# 0: single pendulum, 1: double pendulum, 2: inverted pendulum, 3: double pendulum with extra observation
	env_id = DOUBLE_PENDULUM
	models_dir = f"models/{int(time.time())}/"

	if env_id == 0:
		control_type = pendulum_env.Control_Type.NEURON
		logdir = f"logs/{int(time.time())}-{pendulum_env.control_typle_dic[control_type]}/"
	elif env_id == 1 or env_id == 2 or env_id == 3:
		control_type = double_links_env.Control_Type.BASELINE
		logdir = f"logs/{int(time.time())}-{double_links_env.control_typle_dic[control_type]}/"


	if not os.path.exists(models_dir):
		os.makedirs(models_dir)

	if not os.path.exists(logdir):
		os.makedirs(logdir)

	episode_length = 10000
	if env_id == 0:
		num_episodes = 5
		env = pendulum_env.PendulumEnv(control_type=control_type)
		# vec_env = make_vec_env(lambda: pendulum_env.PendulumEnv(control_type=control_type), n_envs=4)
	elif env_id == 1:
		env = double_links_env.DoubleLinkEnv(control_type=control_type)
		num_episodes = env.get_num_of_targets()
		print(f"total number of targets: {num_episodes}")
	elif env_id == 2 or env_id == 3:
		dummy_env = double_links_env.DoubleLinkEnv(control_type=control_type)
		num_episodes = dummy_env.get_num_of_targets()
		print(f"total number of targets: {num_episodes}")

		THREADS_NUM = 16
		env_fns = [make_env(i, env_id) for i in range(THREADS_NUM)]
		env = SubprocVecEnv(env_fns)

	m_steps = episode_length * num_episodes
	#n_steps=50000 batch_size=10000,n_epochs=10 tested
	#model = PPO('MlpPolicy', env, device='cpu', n_steps=m_steps, batch_size=1000, n_epochs=100, verbose=1, tensorboard_log=logdir) - 1689548375
	#model = PPO('MlpPolicy', env, device='cpu', n_steps=m_steps, batch_size=500, n_epochs=16, learning_rate=0.0002, verbose=1, tensorboard_log=logdir)
	model = PPO('MlpPolicy', env, device='auto', n_steps=5000, batch_size=5000, n_epochs=10, verbose=1,tensorboard_log=logdir)
	# model = PPO('MlpPolicy', env, device='cpu', n_steps=50000, batch_size=10000, n_epochs=100, verbose=1, tensorboard_log=logdir)
	# PPO_model_path="models/1688353130/24640000.zip"
	# model=PPO.load(PPO_model_path, env=env)
	# model.verbose = 1
	# model.tensorboard_log = logdir
	# model.device="cuda"
	model = A2C('MlpPolicy', env, device='auto', n_steps=m_steps, verbose=1, tensorboard_log=logdir)

	TIMESTEPS = 10000
	iters = 0
	while True:
		iters += 1
		model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"A2C")
		model.save(f"{models_dir}/{TIMESTEPS * iters}")
		print(f"modle saved {iters}\n")