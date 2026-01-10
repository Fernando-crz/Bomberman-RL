import stable_retro
import os

# SCRIPT_DIR = os.path.dirname(os.path.abspath("~/github/stable-retro/stable_retro/data/contrib/SuperBomberman-Snes"))
SCRIPT_DIR = "contrib/SuperBomberman-Snes"


def main():
        stable_retro.data.Integrations.add_custom_path(
                SCRIPT_DIR
        )
        print("SuperBomberman-Snes" in stable_retro.data.list_games(inttype=stable_retro.data.Integrations.ALL))
        env = stable_retro.make("SuperBomberman-Snes", inttype=stable_retro.data.Integrations.ALL)
        print(env)


if __name__ == "__main__":
        main()