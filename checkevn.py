from stable_baselines3.common.env_checker import check_env
from pendulum_env import PendulumEnv

env = PendulumEnv()
check_env(env)

episodes = 3
for i in range(episodes):
    done = False
    print("start episode")
    obs = env.reset()
    while done == False:
        random_action = env.action_space.sample()
        print("action", random_action)
        obs, reward, done, info = env.step(random_action)
        print('reward', reward)
