from stable_baselines3 import PPO
from sumogym.envs.robot_sumo_single import RobotSumoSingleEnv

import time
model = PPO.load("ppo_sumo_no_time_penalty")


env = RobotSumoSingleEnv(render_mode="human")
obs, info = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, truncs, infos = env.step(action)
    time.sleep(1/240.0)