@echo off
cd /d "%~dp0"
python -m venv venv
call venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
if not exist pose_landmarker_full.task curl -L -o pose_landmarker_full.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
echo.
echo Setup done. Each new terminal: venv\Scripts\activate
pause
