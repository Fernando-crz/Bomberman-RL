from gymnasium.spaces import Discrete, Box, MultiBinary
from pettingzoo import ParallelEnv
from pettingzoo.test import parallel_api_test
import numpy as np

HEIGHT, WIDTH, CHANNELS = 192, 256, 3 

# TODO Deal with hard-coded vars

class BombermanParallelEnv(ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    def __init__(self, retro_env):
        self.retro_env = retro_env
        
        self.possible_agents = ["player_0", "player_1"]
        self.agents = self.possible_agents[:]

        self.action_spaces = {
            "player_0": Discrete(10),
            "player_1": Discrete(10),
        }

        self.observation_spaces = {
            "player_0": Box(
                low=0,
                high=255,
                shape=(HEIGHT, WIDTH, CHANNELS),
                dtype=np.uint8
            ),
            "player_1": Box(
                low=0,
                high=255,
                shape=(HEIGHT, WIDTH, CHANNELS),
                dtype=np.uint8
            ),
        }
    
    def reset(self, seed=None, options=None):
        obs, info = self.retro_env.reset()

        self.agents = self.possible_agents[:]

        observations = {
            agent: obs
            for agent in self.agents
        }

        infos = {agent: {} for agent in self.agents}

        return observations, infos
    
    def close(self):
        self.retro_env.close()
    
    def step(self, actions):
        joint_action = np.concatenate([
            self.decode_action(actions["player_0"]),
            self.decode_action(actions["player_1"])
        ]).astype(np.int8)

        obs, _, _, _, info = self.retro_env.step(joint_action)

        # Compute observation with stacking

        observations = {
            agent: obs 
            for agent in self.agents
        }

        # Compute player rewards

        rewards = self.compute_rewards(info)

        # Compute Terminations/Truncations

        terminations = self.compute_terminations(info)
        truncations = {agent:False for agent in self.agents} # Never truncate since game will end before going over too many its.
        
        self.agents = [
            agent for agent in self.agents
            if not terminations[agent]
        ]

        # Compute Infos

        infos = {
            "player_0": {
                "is_dead": info.get("is_p1_dead", 0)
            },
            "player_1": {
                "is_dead": info.get("is_p2_dead", 0)
            },
        }

        return observations, rewards, terminations, truncations, infos
    
    def decode_action(self, action):
        buttons = np.zeros(12, dtype=np.int8)

        if action == 0:
            pass  # NOOP

        elif action == 1:      # UP
            buttons[4] = 1

        elif action == 2:      # DOWN
            buttons[5] = 1

        elif action == 3:      # LEFT
            buttons[6] = 1

        elif action == 4:      # RIGHT
            buttons[7] = 1

        elif action == 5:      # BOMB
            buttons[8] = 1

        elif action == 6:      # UP + BOMB
            buttons[4] = 1
            buttons[8] = 1

        elif action == 7:      # DOWN + BOMB
            buttons[5] = 1
            buttons[8] = 1

        elif action == 8:      # LEFT + BOMB
            buttons[6] = 1
            buttons[8] = 1

        elif action == 9:      # RIGHT + BOMB
            buttons[7] = 1
            buttons[8] = 1
        
        return buttons
    
    def compute_terminations(self, info):
        is_game_over = info.get("total_player_count") == 1 or info.get("active_player_count") == 0 
        return {
            "player_0": info.get("is_p1_dead") == 1 or is_game_over,
            "player_1": info.get("is_p2_dead") == 1 or is_game_over
            }
    
    def compute_rewards(self, info):
        P1_WIN_REWARD = 1.0
        P1_LOSE_PENALTY = 0.5
        P1_TIMER_PENALTY = 0.0001

        P2_WIN_REWARD = 1.0
        P2_LOSE_PENALTY = 0.5
        P2_TIMER_PENALTY = 0.0001
        return {
            "player_0": P1_WIN_REWARD * float(info.get("p1_win_blink_countdown") > 0) - P1_LOSE_PENALTY * float(info.get("is_p1_dead") == 1) - P1_TIMER_PENALTY,
            "player_1": P2_WIN_REWARD * float(info.get("p2_win_blink_countdown") > 0) - P2_LOSE_PENALTY * float(info.get("is_p2_dead") == 1) - P2_TIMER_PENALTY,
            }

    def render(self):
        self.retro_env.render()
    
    def action_space(self, agent):
        return self.action_spaces[agent]

    def observation_space(self, agent):
        return self.observation_spaces[agent]


def main():
    import stable_retro as retro
    env = retro.make("SuperBomberman-Snes", 
                        inttype=retro.data.Integrations.ALL, 
                        use_restricted_actions=retro.Actions.ALL,
                        record='.',
                        players=2,
                        render_mode='rgb_array')
    env.multi_rewards = True

    parallel_env = BombermanParallelEnv(env)
    
    obs, info = parallel_env.reset()

    # for i in range(10):
    #     actions = {
    #         "player_0": parallel_env.action_spaces["player_0"].sample(),
    #         "player_1": parallel_env.action_spaces["player_1"].sample(),
    #     }
    #     obs, rewards, terminations, truncations, infos = parallel_env.step(actions)
    #     print(f"\n\n=\n, rewards:{rewards}\n, terminations:{terminations}\n, truncations:{truncations}\n, infos:{infos}\n")

    #     if not parallel_env.agents:
    #         obs, infos = parallel_env.reset()

    parallel_api_test(parallel_env, num_cycles=10_000)

if __name__ == "__main__":
    main()