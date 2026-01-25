from super_bomberman_ma_env import CustomEnvironment
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.vec_env import VecMonitor
import supersuit as ss
import time

if __name__ == "__main__":
    steps = 10_000
    
    env = CustomEnvironment(individual_terminations=True)
    env = ss.black_death_v3(env)
    env = ss.frame_stack_v1(env, 4)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=1, base_class='stable_baselines3')
    env = VecMonitor(env)
    
    model = PPO(
        policy="CnnPolicy",
        env=env,
        n_steps=2048,
        ent_coef=0.02,
        verbose=1
    )

    model.learn(total_timesteps=steps, progress_bar=True)
    model.save("test_ppo_model_sb3")

    env.close()
