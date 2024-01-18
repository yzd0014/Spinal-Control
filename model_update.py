import torch.nn.init as init
import time
from parameters import *
from stable_baselines3.common.utils import safe_mean
import double_links_env

env = double_links_env.DoubleLinkEnv(control_type=control_type, env_id=env_id, c_params=controller_params)

PPO_model_path = "./models/1705534107/585000.zip"
PPO_model = PPO.load(PPO_model_path, env=env, device='cpu')
PPO_parameters = PPO_model.get_parameters()
policy_weights = PPO_parameters['policy']
offset_tensor = torch.zeros([4, 64], dtype=torch.float64)
init.xavier_uniform_(offset_tensor)
offset_tensor = torch.mul(offset_tensor, 1)
# policy_weights['mlp_extractor.policy_net.2.weight'] = torch.add(policy_weights['mlp_extractor.policy_net.2.weight'], offset_tensor)
policy_weights['action_net.weight'] = torch.add(policy_weights['action_net.weight'], offset_tensor)
PPO_model.set_parameters(load_path_or_dict=PPO_parameters, device='cpu')
# PPO_parameters2 = PPO_model.get_parameters()

models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}-{control_type_dic[control_type]}/"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

reward_target = 100
TIMESTEPS =  int(100/controller_params.brain_dt)
iters = 0
PPO_model.save(f"{models_dir}/{TIMESTEPS * iters}")
while True:
    iters += 1
    PPO_model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, \
                tb_log_name=f"PPO")
    mean_reward = safe_mean([ep_info["r"] for ep_info in PPO_model.ep_info_buffer])
    if mean_reward > reward_target:
        break
    PPO_model.save(f"{models_dir}/{TIMESTEPS * iters}")
