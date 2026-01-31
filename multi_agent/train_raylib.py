from super_bomberman_ma_env import CustomEnvironment
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.tune.registry import register_env
import supersuit as ss

# Register the custom environment with normalization
def make_env(config):
    env = CustomEnvironment()
    env = ss.black_death_v3(env)
    env = ss.color_reduction_v0(env, mode='full')  # Convert to grayscale (3 channels -> 1)
    env = ss.frame_stack_v1(env, 4)
    env = ss.dtype_v0(env, dtype='float32')  # Convert uint8 to float32
    env = ss.normalize_obs_v0(env, env_min=0, env_max=255)  # Normalize to [0, 1]
    return ParallelPettingZooEnv(env)

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
        model_config={
            "conv_filters": [
                [16, [8, 8], 4],
                [32, [4, 4], 2],
                [64, [3, 3], 2],
                [128, [3, 3], 2],
            ],
            "vf_share_layers": True,
        },
    )
    .framework("torch")
    .training(
        train_batch_size=256,
        num_epochs=3,
        lr=5e-5,
        gamma=0.99,
        lambda_=0.95,
        clip_param=0.2,
        vf_clip_param=10.0,
        entropy_coeff=0.01,
    )
    .env_runners(
        num_env_runners=1,
        num_envs_per_env_runner=1,
        sample_timeout_s=300.0,
        rollout_fragment_length=32,
    )
    .learners(
        num_learners=0,  # 0 means use local learner with GPU
        num_gpus_per_learner=1,
    )
    .resources(
        num_gpus=0,  # Set to 0 since GPU is now specified in learners
    )
    .fault_tolerance(
        restart_failed_env_runners=True,
    )
    .debugging(
        log_level="INFO",
    )
)

if __name__ == "__main__":
    # Build the algorithm
    algo = config.build_algo()
    
    # Training loop
    num_iterations = 100
    
    for i in range(num_iterations):
        result = algo.train()
        
        # Print training progress with more detailed metrics
        print(f"\n=== Iteration {i + 1} ===")
        
        # Try to get episode metrics from different possible locations in result
        env_runner_results = result.get('env_runners', {})
        sampler_results = result.get('sampler_results', {})
        
        # Debug: Print available keys in the first iteration
        if i == 0:
            print(f"DEBUG - Available result keys: {list(result.keys())}")
            if env_runner_results:
                print(f"DEBUG - env_runners keys: {list(env_runner_results.keys())}")
        
        # Episode statistics - correct keys for new RLlib API stack
        episode_return_mean = env_runner_results.get('episode_return_mean')
        episode_len_mean = env_runner_results.get('episode_len_mean')
        num_episodes = env_runner_results.get('num_episodes', 0)
        
        # Also get per-agent returns
        agent_returns = env_runner_results.get('agent_episode_returns_mean', {})
        module_returns = env_runner_results.get('module_episode_returns_mean', {})
        
        print(f"Episode return mean: {episode_return_mean if episode_return_mean is not None else 'N/A'}")
        print(f"Episode length mean: {episode_len_mean if episode_len_mean is not None else 'N/A'}")
        print(f"Episodes this iteration: {num_episodes}")
        if agent_returns:
            print(f"Agent returns: {agent_returns}")
        
        # Save checkpoint every 10 iterations
        if (i + 1) % 10 == 0:
            checkpoint_dir = algo.save()
            print(f"Checkpoint saved at: {checkpoint_dir}")
    
    # Final save
    final_checkpoint = algo.save()
    print(f"\nTraining complete! Final checkpoint saved at: {final_checkpoint}")
    
    algo.stop()
