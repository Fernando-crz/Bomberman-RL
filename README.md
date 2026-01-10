
# SuperBomberman (SNES) Integration for Stable Retro

This repository provides a custom environment for the SuperBomberman (SNES) game, designed for use with the Stable Retro framework. Follow the steps below to set up, configure, and run the environment.

## Installation

To install Stable Retro and prepare your system, refer to the following resources:

- [Stable Retro Documentation](https://stable-retro.farama.org/developing/)
- [Windows (WSL2) + Ubuntu 22.04 Setup guide for training models on videogames with stable-retro](https://www.youtube.com/watch?v=vPnJiUR21Og&t=133s)
- [OpenAI game integration tool - part 1](https://www.youtube.com/watch?v=lPYWaUAq_dY)

## Integrating SuperBomberman (SNES)

SuperBomberman (SNES) is a custom environment and must be integrated into Stable Retro manually:

1. In your `stable-retro` directory, run:
	 ```bash
	 ./gym-retro-integration
	 ```
2. In the integration tool, go to **Game → Integrate**, then select your `.sfc` ROM file for SuperBomberman.
3. When prompted, name the game `SuperBomberman-Snes`.
4. After integration, a new folder `SuperBomberman-Snes` will appear at `stable-retro/stable_retro/data/contrib/`.
5. Copy the contents of this repository's `game_info/` folder into the newly created folder.
6. You are now ready to run the provided scripts.

## Configuration

You can customize the environment by editing the following files in `stable-retro/stable_retro/data/contrib/SuperBomberman-Snes/`:

- `data.json`: define the variables extracted from the game memory.
- `metadata.json`: define the default state of the game.
- `scenario.json`: define the scenarios for the game, including frame crop, rewards and conditions of finish.

For more details on configuring custom environments, see this [YouTube playlist](https://www.youtube.com/playlist?list=PLmwlWbdWpZVvWqzOxu0jVBy-CaRpYha0t) and the [Stable Retro documentation](https://stable-retro.farama.org/integration/).

## Running the Environment

First, ensure SuperBomberman is integrated as described above. Then, run:

```bash
python integrate_game.py
```

To test the environment and save a gameplay recording in `.bk2` format, use:

```bash
python run_test.py
```

To save all gameplay frames as images for analysis, use:

```bash
python run_test.py -frame
```

The `./make_video.sh` script takes `.bk2` files as input and generates the corresponding `.mp4` video files, effectively converting gameplay recordings into standard video format for easier viewing and sharing. So run:

```bash
chmod +x make_video.sh
./make_video.sh
```

## Additional Notes

- By default, the reward in Stable Retro is set for a single player. To enable multi-player rewards, edit line 50 in `stable-retro/stable_retro/retro_env.py`:
	```python
	self.multi_rewards = True
	```
- The version of `retro_env.py` in this repository (located in `library_scripts/`) is already modified to support multi-player rewards. You can copy it to the framework directory if needed.
- Once you have the files `rom.sfc` and `rom.sha` from the integration process, you can run the scripts on Google Colab by opening the ColabAutomation.ipynb notebook, uploading the files at the `/content/` directory and running it.