from stable_baselines3.ppo import MlpPolicy
from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import VecMonitor
import supersuit as ss
# from petting_bubble_env_continuous import PettingBubblesEnvironment
from sumogym.envs.robot_sumo_parallel import RobotSumoParallelEnv
import gym

env = RobotSumoParallelEnv()
# env = ss.pad_observations_v0(env)
# env = ss.black_death_v3(env)
env = ss.pettingzoo_env_to_vec_env_v1(env)
env = ss.concat_vec_envs_v1(env, 4, num_cpus=4, base_class='stable_baselines3')

model = PPO('MultiInputPolicy', env, verbose=2, gamma=0.999, n_steps=1000, ent_coef=0.01, learning_rate=0.00025, vf_coef=0.5, max_grad_norm=0.5, gae_lambda=0.95, n_epochs=4, clip_range=0.2, clip_range_vf=1, tensorboard_log="./ppo_test/")
model.learn(total_timesteps=1000000, tb_log_name="test",  reset_num_timesteps=True)
model.save("bubble_policy_test")
