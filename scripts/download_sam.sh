#!/bin/bash

# Set your directory
dir="sam"

# Create the directory if it does not exist
mkdir -p "$dir"

# Use wget to download the file
wget -O "sam/sam_vit_h_4b8939.pth" "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

