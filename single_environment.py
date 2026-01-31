import stable_retro as retro

import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def create_env():
    retro.data.Integrations.add_custom_path(
                os.path.join(SCRIPT_DIR, "custom_integrations")
        )
    env = retro.make("SuperBomberman-Snes", 
                        inttype=retro.data.Integrations.ALL, 
                        use_restricted_actions=retro.Actions.ALL,
                        record='.',
                        players=2,
                        render_mode='human')
    env.multi_rewards = True

    return env

def main():
    env = create_env()

    no_move = np.zeros(24, dtype=np.int8)

    obs, _ = env.reset()

    for _ in range(10):
        obs, rew, terminate, truncate, info = env.step(no_move)
        print(f"\n\nobs: {obs.shape}\nrew: {rew}\nterminate: {terminate}\ntruncate: {truncate}\ninfo: {info}")
        env.render()
    


if __name__ == "__main__":
    main()