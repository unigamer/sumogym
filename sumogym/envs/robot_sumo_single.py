
from . import robot_sumo_parallel
import gymnasium as gym

class RobotSumoSingleEnv(robot_sumo_parallel.RobotSumoParallelEnv, gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, competitor_agent=None, max_time=-1):

        super().__init__(render_mode=render_mode, max_time=max_time)
        self.self_robot = "robotA"
        self.opponent_robot = "robotB"

        self.observation_space = super().observation_space(None)
        self.action_space = super().action_space(None)
  
    # def _get_info(self, agent):
    #     self.infos = super()._get_info(agent)
    #     return self.infos

    def step(self, action):
        opponent_action = [0, 0]
        actions = {self.self_robot: action,
                   self.opponent_robot: opponent_action}    
        self.observations, rewards, terminations, truncations, infos = super().step(actions)
        termination = any(terminations.values()) == True
        truncation = any(truncations.values()) == True
        return self.observations[self.self_robot], rewards[self.self_robot], termination, truncation, infos[self.self_robot]
    

    def reset(self, seed=None, options=None):
        self.observations, infos = super().reset(seed=seed, options=options)
        return self.observations[self.self_robot], infos[self.self_robot]
    
    def render(self):
        return super().render()
    
    def close(self):
        super().close()