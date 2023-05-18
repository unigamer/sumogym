import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
from sumogym.envs.robot_sumo_single import RobotSumoSingleEnv

env = RobotSumoSingleEnv(max_time=20)
# vec_env = make_vec_env(env, n_envs=4)

model = PPO("MultiInputPolicy", env, verbose=0, tensorboard_log="./ppo_sumo/")
# model = PPO.load("ppo_sumo_2")
model.set_env(env)
model.learn(total_timesteps=10000)
model.save("ppo_sumo_no_time_penalty")

# del model # remove to demonstrate saving and loading

# model = PPO.load("ppo_sumo_1")
# del env

# env = RobotSumoSingleEnv(render_mode="human")
# obs, info = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, truncs, infos = env.step(action)

    
#     # env.render("human")