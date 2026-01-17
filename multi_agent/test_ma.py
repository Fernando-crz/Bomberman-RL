from super_bomberman_ma_env import CustomEnvironment

from pettingzoo.test import parallel_api_test

if __name__ == "__main__":
    env = CustomEnvironment()
    parallel_api_test(env, num_cycles=1_000_000)
    env.close()

    env_stack = CustomEnvironment(frame_stack=4)
    parallel_api_test(env_stack, num_cycles=1_000_000)
    env_stack.close()

    env_indiv_term = CustomEnvironment(individual_terminations=True)
    parallel_api_test(env_indiv_term, num_cycles=1_000_000)
    env_indiv_term.close()

    env_stack_ind = CustomEnvironment(frame_stack=10, individual_terminations=True)
    parallel_api_test(env_stack_ind, num_cycles=1_000_000)
    env_stack_ind.close()