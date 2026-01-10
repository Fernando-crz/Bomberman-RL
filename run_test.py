import sys
import stable_retro
from PIL import Image
import numpy as np

frame_by_frame = False
count = 0

def repeat_action(env, action, times):
    global count
    accumulated_reward = [0, 0]

    for _ in range(times):
        # Take an action in the environment
        obs, rew, terminate, truncate, _ = env.step(action)

        # Accumulate rewards for both players
        if env.multi_rewards:
            accumulated_reward[0] += rew[0]
            accumulated_reward[1] += rew[1]
        else:
            accumulated_reward[0] += rew

        # Save frames to visualize later
        if frame_by_frame:
            img = Image.fromarray(obs)
            img.save(f"step_{count}.png")
        count += 1

        # Render the game state (may not work)
        env.render()
    return accumulated_reward

def main():
    env = stable_retro.make("SuperBomberman-Snes", 
                            inttype=stable_retro.data.Integrations.ALL, 
                            use_restricted_actions=stable_retro.Actions.ALL,
                            record='.',
                            players=2,
                            render_mode='human')

    print("Action random sample:", env.action_space.sample())
    print("Buttons:", env.buttons)
    # Buttons are: ['B', 'Y', 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A', 'X', 'L', 'R']

    # Define some example actions
    p1_first_move = np.zeros(12, dtype=np.int8)
    p1_first_move[7] = 1  # Right
    p2_first_move = np.zeros(12, dtype=np.int8)
    p2_first_move[6] = 1  # Left
    first_move = np.concatenate([p1_first_move, p2_first_move])

    p1_second_move = np.zeros(12, dtype=np.int8)
    p1_second_move[8] = 1  # Bomb
    p2_second_move = np.zeros(12, dtype=np.int8)
    p2_second_move[8] = 1  # Bomb
    second_move = np.concatenate([p1_second_move, p2_second_move])

    p1_third_move = np.zeros(12, dtype=np.int8)
    p1_third_move[6] = 1  # Left
    p2_third_move = np.zeros(12, dtype=np.int8)
    p2_third_move[7] = 1  # Right
    third_move = np.concatenate([p1_third_move, p2_third_move])

    p1_fourth_move = np.zeros(12, dtype=np.int8)
    p1_fourth_move[5] = 1  # Down
    p2_fourth_move = np.zeros(12, dtype=np.int8)
    p2_fourth_move[4] = 1  # Up
    fourth_move = np.concatenate([p1_fourth_move, p2_fourth_move])

    no_move = np.zeros(24, dtype=np.int8)


    # Initialize the environment and get the initial observation
    obs, _ = env.reset()

    # Depending on how the state was saved, the initial moments may not compute movements.
    # Because of these initial frames, we skip the first 5 frames
    for _ in range(5):
        obs, rew, terminate, truncate, _ = env.step(no_move)

    repeat_action(env, first_move, 20)
    rew = repeat_action(env, second_move, 1)
    print("Reward:", rew)
    repeat_action(env, third_move, 20)
    repeat_action(env, fourth_move, 20)
    repeat_action(env, no_move, 120)

    return


if __name__ == "__main__":
    # run "python run_test.py -frame" to save frames as images
    if "-frame" in sys.argv:
        frame_by_frame = True

    main()