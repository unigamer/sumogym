from gymnasium.envs.registration import register

register(
     id="sumogym/RobotSumo-v0",
     entry_point="sumogym.envs:RobotSumoEnv",
     max_episode_steps=300,
)