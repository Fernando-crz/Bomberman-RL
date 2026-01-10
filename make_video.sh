#!/bin/bash
for file in *.bk2; do
    python3 -m stable_retro.scripts.playback_movie "$file"
done