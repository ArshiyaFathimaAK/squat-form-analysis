#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
[ -f pose_landmarker_full.task ] || curl -L -o pose_landmarker_full.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
echo "Setup done. Each new terminal: source venv/bin/activate"
