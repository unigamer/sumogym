import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.ppo import CnnPolicy

from sumogym.envs.robot_sumo_parallel import RobotSumoParallelEnv


def main():
    env = RobotSumoParallelEnv()
    # env = ss.frame_stack_v1(env, 3)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 8, num_cpus=2, base_class="stable_baselines3")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=3,
        gamma=0.95,
        n_steps=256,
        ent_coef=0.0905168,
        learning_rate=0.00062211,
        vf_coef=0.042202,
        max_grad_norm=0.9,
        gae_lambda=0.99,
        n_epochs=5,
        clip_range=0.3,
        batch_size=256,
    )
    # model.learn(total_timesteps=2000000)
    model.learn(total_timesteps=200)
    model.save("policy")

    # Rendering

    # env = pistonball_v6.env()
    # env = ss.color_reduction_v0(env, mode="B")
    # env = ss.resize_v1(env, x_size=84, y_size=84)
    # env = ss.frame_stack_v1(env, 3)

    # model = PPO.load("policy")

    # env.reset()
    # for agent in env.agent_iter():
    #     obs, reward, termination, truncation, info = env.last()
    #     act = (
    #         model.predict(obs, deterministic=True)[0]
    #         if not (termination or truncation)
    #         else None
    #     )
    #     env.step(act)
    #     env.render()


if __name__ == "__main__":
    main()