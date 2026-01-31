from super_bomberman_ma_env import CustomEnvironment
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.tune.registry import register_env
from pettingzoo.utils import frame_stack_v1
import supersuit as ss

# Register the custom environment
def make_env(config):
    env = CustomEnvironment()
    env = ss.frame_stack_v1(env, 4)
    return PettingZooEnv(env)
register_env("super_bomberman", make_env)

# Configure PPO for multi-agent training with two agents
config = (
    PPOConfig()
    # .environment(env="super_bomberman", env_config={"individual_terminations": True})
    .environment(env="super_bomberman", env_config={})
    .multi_agent(
        policies={"player_1", "player_2"},
        policy_mapping_fn=lambda agent_id, episode, **kwargs: agent_id,
    )
    .rl_module(
        rl_module_spec=MultiRLModuleSpec(
            rl_module_specs={
                "player_1": RLModuleSpec(),
                "player_2": RLModuleSpec(),
            }
        ),
    )
    .framework("torch")
    .training(
        train_batch_size=4000,
        sgd_minibatch_size=128,
        num_sgd_iter=30,
        lr=5e-5,
        gamma=0.99,
        lambda_=0.95,
        clip_param=0.2,
        vf_clip_param=10.0,
        entropy_coeff=0.02,
    )
    .rollouts(
        num_rollout_workers=2,
        num_envs_per_worker=1,
    )
    .resources(
        num_gpus=1,  # Set to 1 if you have a GPU
    )
    .debugging(
        log_level="INFO",
    )
)

if __name__ == "__main__":
    # Build the algorithm
    algo = config.build()
    
    # Training loop
    num_iterations = 100
    
    for i in range(num_iterations):
        result = algo.train()
        
        # Print training progress
        print(f"\n=== Iteration {i + 1} ===")
        print(f"Episode reward mean: {result.get('episode_reward_mean', 'N/A')}")
        print(f"Episode length mean: {result.get('episode_len_mean', 'N/A')}")
        
        # Save checkpoint every 10 iterations
        if (i + 1) % 10 == 0:
            checkpoint_dir = algo.save()
            print(f"Checkpoint saved at: {checkpoint_dir}")
    
    # Final save
    final_checkpoint = algo.save()
    print(f"\nTraining complete! Final checkpoint saved at: {final_checkpoint}")
    
    algo.stop()
