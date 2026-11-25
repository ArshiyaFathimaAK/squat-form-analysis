# Real-Time Squat Form Analysis (CS 4391)

## 1. Setup (once)
- Windows: double-click `setup.bat`
- Mac/Linux: `./setup.sh`

This creates a virtual environment, installs the packages and downloads the
MediaPipe model (`pose_landmarker_full.task`). Use Python 3.10-3.12.
If the model download fails, get it from the MediaPipe "Pose Landmarker" page
and put it in this folder.

Every new terminal, activate first:
- Windows: `venv\Scripts\activate`
- Mac/Linux: `source venv/bin/activate`

## 2. Add videos
Put the videos in `videos/` as clip01.mp4 ... clip20.mp4.

## 3. Fill in clips.csv (one row per video)
| column | example values |
|---|---|
| person | omar, arshiya, kiryang, anthony |
| view | side or front (exactly these words) |
| camera_angle | side_90, side_45, front_0 |
| lighting | bright, dim |
| speed | normal, fast, slow |
| true_reps | the number of reps you count by watching |
| split | calib (~6 clips) or test (~14 clips): split by clip, never by rep |

## 4. Run
```
python squat_pipeline.py extract videos/   # pose landmarks -> landmarks/
python squat_pipeline.py reps              # -> reps_to_label.csv
```
Test with 1 video first and check the rep count is right.

## 5. Label
Open `reps_to_label.csv`. For each rep, watch the video at `bottom_s` and enter
0/1 in `depth_err`, `lean_err` (side clips) and `valgus_err` (front clips).
Save as `labels.csv`. Two people label independently, then resolve disagreements.

## 6. Results
```
python squat_pipeline.py calibrate         # -> thresholds.json
python squat_pipeline.py evaluate          # precision/recall/F1, rep counting, robustness
python squat_pipeline.py live side         # webcam demo (press q to quit); FPS + latency
```
