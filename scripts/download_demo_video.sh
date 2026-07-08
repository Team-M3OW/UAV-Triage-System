#!/bin/bash
FILE="demo_video/monkey.mp4"
echo "downloading demo video from dropbox"
mkdir -p demo_video
wget --max-redirect=20 -O ${FILE} https://www.dropbox.com/scl/fi/placeholder/monkey.mp4?dl=1
