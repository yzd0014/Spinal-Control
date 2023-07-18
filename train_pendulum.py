from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
#from stable_baselines3 import DDPG
import os
import pendulum_env
import double_links_env
import time

def make_env(instance_id=0):
	return lambda: double_links_env.DoubleLinkEnv(control_type=control_type, instance_id=instance_id)

if __name__ =="__main__":
	env_id = 1
	models_dir = f"models/{int(time.time())}/"

	if env_id == 0:
		control_type = pendulum_env.Control_Type.NEURON
		logdir = f"logs/{int(time.time())}-{pendulum_env.control_typle_dic[control_type]}/"
	elif env_id == 1:
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
		num_episodes = double_links_env.compute_total_num_of_targets()
		print(f"total number of targets: {num_episodes}")

		#env = double_links_env.DoubleLinkEnv(control_type=control_type)
		n_envs = 10
		env_fns = [make_env(i) for i in range(n_envs)]
		env = SubprocVecEnv(env_fns)

	m_steps = episode_length * num_episodes
	# m_steps = 50000
	#n_steps=50000 batch_size=10000,n_epochs=10 tested
	#model = PPO('MlpPolicy', env, device='cpu', n_steps=m_steps, batch_size=1000, n_epochs=100, verbose=1, tensorboard_log=logdir) - 1689548375
	#model = PPO('MlpPolicy', env, device='cpu', n_steps=m_steps, batch_size=500, n_epochs=16, learning_rate=0.0002, verbose=1, tensorboard_log=logdir)
	model = PPO('MlpPolicy', env, device='cuda', n_steps=m_steps, batch_size=1000, n_epochs=100, verbose=1, tensorboard_log=logdir)
	# model = PPO('MlpPolicy', env, device='cpu', n_steps=50000, batch_size=10000, n_epochs=100, verbose=1, tensorboard_log=logdir)
	# PPO_model_path="models/1688353130/24640000.zip"
	# model=PPO.load(PPO_model_path, env=env)
	# model.verbose = 1
	# model.tensorboard_log = logdir
	#model.device="cuda"

	#TIMESTEPS = 50000
	TIMESTEPS = m_steps
	iters = 0
	while True:
		iters += 1
		model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
		model.save(f"{models_dir}/{TIMESTEPS * iters}")
		print(f"modle saved {iters}\n")