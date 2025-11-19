#!/bin/bash

# Download checkpoints
if [ ! -d "soundfonts" ]; then
    gdown 11vPbKCIandwkxafLTqOvxHlKgkTtq1PI -O soundfonts.zip
    unzip -n soundfonts.zip
    rm soundfonts.zip
fi
