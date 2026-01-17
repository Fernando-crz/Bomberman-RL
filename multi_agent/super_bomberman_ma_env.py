import functools
import random
from copy import copy

import numpy as np

from gymnasium.spaces import Discrete, Box
from pettingzoo import ParallelEnv
import stable_retro

discrete_to_multibinary = {
    0: np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8),  # Stop
    1: np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8),  # Up
    2: np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=np.int8),  # Down
    3: np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.int8),  # Left
    4: np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.int8),  # Right
    5: np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], dtype=np.int8),  # Bomb
    6: np.array([0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], dtype=np.int8),  # Up + Bomb
    7: np.array([0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0], dtype=np.int8),  # Down + Bomb
    8: np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0], dtype=np.int8),  # Left + Bomb
    9: np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0], dtype=np.int8),  # Right + Bomb
}

class CustomEnvironment(ParallelEnv):
    metadata = {
        "name": "super_bomberman_ma_env_v0",
    }

    def __init__(self, render_mode='rgb_array', frame_stack=0, individual_terminations=False):
        self.possible_agents = ["player_1", "player_2"]
        self.render_mode = render_mode
        self.frame_stack = frame_stack
        self.individual_terminations = individual_terminations
        self.sr_env = None
        self.observation_stack = None

    def reset(self, seed=None, options=None):
        # Close previous emulator if it exists
        if self.sr_env is not None:
            try:
                self.sr_env.close()
            except Exception:
                pass
            self.sr_env = None

        # Initialize the Stable Retro environment
        self.sr_env = stable_retro.make("SuperBomberman-Snes", 
                            inttype=stable_retro.data.Integrations.ALL, 
                            use_restricted_actions=stable_retro.Actions.ALL,
                            players=2,
                            render_mode=self.render_mode)

        # Warm-up steps to avoid initial no-movement frames
        no_move = np.zeros(24, dtype=np.int8)
        obs, _ = self.sr_env.reset()
        for _ in range(5):
            obs, rew, terminate, truncate, _ = self.sr_env.step(no_move)
        
        # Set initial observations and infos(empty for now)
        self.agents = copy(self.possible_agents)
        if self.frame_stack > 0:
            obs = np.repeat(obs[np.newaxis, ...], self.frame_stack, axis=0)
            self.observation_stack = obs
        observations = {a: obs for a in self.agents}
        infos = {a: {} for a in self.agents}
        
        return observations, infos

    def close(self):
        if self.sr_env is not None:
            try:
                self.sr_env.close()
            except Exception:
                pass
            self.sr_env = None

    def step(self, actions):
        # Action mapping
        p1_action = discrete_to_multibinary[actions.get('player_1', 0)]
        p2_action = discrete_to_multibinary[actions.get('player_2', 0)]
        combined_action = np.concatenate([p1_action, p2_action])

        # Stable Retro Step
        obs, rew, terminated, truncated, info = self.sr_env.step(combined_action)

        # Rewards
        # TODO: Check info data.json variables to customize reward
        if self.sr_env.multi_rewards:
            rewards = {}
            if 'player_1' in self.agents:
                rewards['player_1'] = rew[0]
            if 'player_2' in self.agents:
                rewards['player_2'] = rew[1]
        else:
            rewards = {a: rew for a in self.agents}

        # Terminations and truncations
        if self.individual_terminations:
            p1_terminated = info['is_white_alive'] == 1943 or terminated
            p2_terminated = info['is_black_alive'] == 1943 or terminated
            terminations = {}
            if 'player_1' in self.agents:
                terminations['player_1'] = p1_terminated
            if 'player_2' in self.agents:
                terminations['player_2'] = p2_terminated
        else:
            terminations = {a: terminated for a in self.agents}
        truncations = {a: False for a in self.agents}

        # Observations and infos
        if self.frame_stack > 0:
            self.observation_stack = np.roll(self.observation_stack, shift=-1, axis=0)
            self.observation_stack[-1] = obs
            obs = self.observation_stack
        observations = {a: obs for a in self.agents}
        infos = {a: {} for a in self.agents}

        # Live agents
        self.agents = [a for a in self.agents if not (terminations[a] or truncations[a])]

        return observations, rewards, terminations, truncations, infos

    def render(self):
        return self.sr_env.render()

    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        if self.frame_stack > 0:
            return Box(low=0, high=255, shape=(self.frame_stack, 192, 256, 3), dtype=np.uint8)
        return Box(low=0, high=255, shape=(192, 256, 3), dtype=np.uint8)

    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(10)