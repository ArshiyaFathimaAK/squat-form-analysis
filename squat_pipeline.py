"""
Squat form analysis pipeline (CS 4391 project)
MediaPipe Pose -> joint angles -> rep segmentation -> rule-based errors -> evaluation.

pip install mediapipe opencv-python numpy pandas
Model file: download pose_landmarker_full.task from the MediaPipe Pose Landmarker page
and put it next to this script.

Usage:
  python squat_pipeline.py extract  videos/            # landmarks -> landmarks/<clip>.csv
  python squat_pipeline.py reps                         # rep segmentation + features -> reps_to_label.csv
  (fill in the 0/1 label columns in reps_to_label.csv and save it as labels.csv; fill clips.csv)
  python squat_pipeline.py calibrate                    # fit thresholds on split=calib clips -> thresholds.json
  python squat_pipeline.py evaluate                     # P/R/F1 on split=test + robustness table
  python squat_pipeline.py live                         # webcam real-time feedback + FPS/latency
"""
import sys, json, time, glob, os
os.environ.setdefault("GLOG_minloglevel", "2")       # hide MediaPipe INFO/WARNING spam
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
try:
    import pandas as pd
except ModuleNotFoundError:
    sys.exit("Missing dependency 'pandas'. Install it with: python -m pip install pandas")
import cv2

MODEL = "pose_landmarker_full.task"
# MediaPipe landmark indices
L_SH, R_SH, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK = 11, 12, 23, 24, 25, 26, 27, 28
KEEP = [L_SH, R_SH, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK]
ERRORS = ["depth_err", "valgus_err", "lean_err"]
# which camera views can see which error (depth shows in both: hip drops toward knee level)
VIEW_FOR = {"depth_err": ("side", "front"), "lean_err": ("side",), "valgus_err": ("front",)}


def make_landmarker():
    if not os.path.exists(MODEL):
        sys.exit(f"Missing {MODEL}. Run setup.bat (Windows) or setup.sh (Mac/Linux) first.")
    import mediapipe as mp
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision
    opts = vision.PoseLandmarkerOptions(
        base_options=mpt.BaseOptions(model_asset_path=MODEL),
        running_mode=vision.RunningMode.VIDEO, num_poses=1)
    return mp, vision.PoseLandmarker.create_from_options(opts)


def frame_landmarks(mp, lm, frame_bgr, ts_ms):
    """Return dict of pixel coords + visibility for the joints we use, or None."""
    h, w = frame_bgr.shape[:2]
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    res = lm.detect_for_video(img, ts_ms)
    if not res.pose_landmarks:
        return None
    p = res.pose_landmarks[0]
    out = {}
    for i in KEEP:
        out[f"x{i}"], out[f"y{i}"], out[f"v{i}"] = p[i].x * w, p[i].y * h, p[i].visibility
    return out


# ---------------------------------------------------------------- geometry
def angle(a, b, c):
    """Angle ABC in degrees; a, b, c are (N,2) arrays."""
    ba, bc = a - b, c - b
    cos = (ba * bc).sum(1) / (np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def pt(df, i):
    return df[[f"x{i}", f"y{i}"]].to_numpy(float)


def frame_features(df):
    """Per-frame features. Uses the body side the camera sees best (higher visibility)."""
    left = df[[f"v{L_HIP}", f"v{L_KNEE}", f"v{L_ANK}"]].mean().mean()
    right = df[[f"v{R_HIP}", f"v{R_KNEE}", f"v{R_ANK}"]].mean().mean()
    sh, hip, knee, ank = (L_SH, L_HIP, L_KNEE, L_ANK) if left >= right else (R_SH, R_HIP, R_KNEE, R_ANK)
    f = pd.DataFrame(index=df.index)
    f["knee_angle"] = angle(pt(df, hip), pt(df, knee), pt(df, ank))
    # torso lean from vertical (0 = upright). Proxy for "rounded back": MediaPipe has no spine points.
    v = pt(df, sh) - pt(df, hip)
    f["torso_lean"] = np.degrees(np.arctan2(np.abs(v[:, 0]), -v[:, 1]))
    # hip below knee, normalized by thigh length (>0 means hip crease below knee = full depth)
    thigh = np.linalg.norm(pt(df, hip) - pt(df, knee), axis=1) + 1e-9
    f["hip_depth"] = (pt(df, hip)[:, 1] - pt(df, knee)[:, 1]) / thigh
    # valgus (front view): knee width / ankle width. < ~1 means knees caving in.
    kw = np.abs(df[f"x{L_KNEE}"] - df[f"x{R_KNEE}"])
    aw = np.abs(df[f"x{L_ANK}"] - df[f"x{R_ANK}"]) + 1e-9
    f["knee_ankle_ratio"] = kw / aw
    return f.rolling(5, center=True, min_periods=1).mean()  # smooth jitter


# ---------------------------------------------------------------- rep counting
class RepCounter:
    """Knee-angle state machine with hysteresis: UP -> DOWN when < down_t, back UP when > up_t."""
    def __init__(self, down_t=140, up_t=160):
        self.down_t, self.up_t, self.state, self.cur = down_t, up_t, "UP", None

    def update(self, i, feats):
        """feats: dict of current-frame features. Returns finished rep dict or None."""
        ka = feats["knee_angle"]
        if self.state == "UP" and ka < self.down_t:
            self.state, self.cur = "DOWN", {"start": i, "bottom": i, "min_knee_angle": ka,
                                            "max_hip_depth": feats["hip_depth"],
                                            "max_torso_lean": feats["torso_lean"],
                                            "min_knee_ankle_ratio": feats["knee_ankle_ratio"]}
        elif self.state == "DOWN":
            c = self.cur
            if ka < c["min_knee_angle"]:
                c["min_knee_angle"], c["bottom"] = ka, i
            c["max_hip_depth"] = max(c["max_hip_depth"], feats["hip_depth"])
            c["max_torso_lean"] = max(c["max_torso_lean"], feats["torso_lean"])
            c["min_knee_ankle_ratio"] = min(c["min_knee_ankle_ratio"], feats["knee_ankle_ratio"])
            if ka > self.up_t:
                self.state, c["end"] = "UP", i
                self.cur = None
                return c
        return None


# ---------------------------------------------------------------- rules
def predict(rep, th, view):
    """Rule-based errors. Only predicts errors visible from this view."""
    p = {"depth_err": int(rep["min_knee_angle"] > th["depth_err"])}
    if view == "side":
        p["lean_err"] = int(rep["max_torso_lean"] > th["lean_err"])
    if view == "front":
        p["valgus_err"] = int(rep["min_knee_ankle_ratio"] < th["valgus_err"])
    return p


FEATURE_FOR = {"depth_err": ("min_knee_angle", ">"), "lean_err": ("max_torso_lean", ">"),
               "valgus_err": ("min_knee_ankle_ratio", "<")}
DEFAULT_TH = {"depth_err": 100.0, "lean_err": 45.0, "valgus_err": 0.9}


def prf(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    tp = int(((y == 1) & (yhat == 1)).sum()); fp = int(((y == 0) & (yhat == 1)).sum())
    fn = int(((y == 1) & (yhat == 0)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f, int(y.sum())


# ---------------------------------------------------------------- commands
def cmd_extract(video_dir):
    os.makedirs("landmarks", exist_ok=True)
    mp, lm = make_landmarker()
    exts = (".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm")
    vids = sorted(p for p in glob.glob(os.path.join(video_dir, "*")) if p.lower().endswith(exts))
    if not vids:
        sys.exit(f"No videos found in {video_dir}")
    for path in vids:
        clip = os.path.splitext(os.path.basename(path))[0]
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        rows, i = [], 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            d = frame_landmarks(mp, lm, frame, int(i * 1000 / fps))
            if d:
                d.update(frame=i, t=i / fps); rows.append(d)
            i += 1
        cap.release()
        pd.DataFrame(rows).to_csv(f"landmarks/{clip}.csv", index=False)
        print(f"{clip}: {i} frames, pose found in {len(rows)}")
        mp, lm = make_landmarker()  # fresh timestamps per video


def cmd_reps():
    out = []
    for path in sorted(glob.glob("landmarks/*.csv")):
        clip = os.path.splitext(os.path.basename(path))[0]
        if os.path.getsize(path) < 10:
            print(f"{clip}: empty landmarks file, skipped"); continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        f = frame_features(df)
        rc = RepCounter()
        n = 0
        for idx in range(len(df)):
            rep = rc.update(idx, f.iloc[idx].to_dict())
            if rep:
                n += 1
                out.append({"clip": clip, "rep": n,
                            "start_s": round(df.t.iloc[rep["start"]], 2),
                            "bottom_s": round(df.t.iloc[rep["bottom"]], 2),
                            "end_s": round(df.t.iloc[rep["end"]], 2),
                            **{k: round(v, 3) for k, v in rep.items() if k not in ("start", "bottom", "end")},
                            "depth_err": "", "valgus_err": "", "lean_err": ""})
        print(f"{clip}: {n} reps detected")
    pd.DataFrame(out).to_csv("reps_to_label.csv", index=False)
    print("Wrote reps_to_label.csv -> fill in 0/1 labels, save as labels.csv")


def load_labeled():
    reps = pd.read_csv("labels.csv")
    clips = pd.read_csv("clips.csv")  # clip, person, view, lighting, speed, camera_angle, split, true_reps
    return reps.merge(clips, on="clip")


def cmd_calibrate():
    d = load_labeled(); cal = d[d.split == "calib"]
    th = {}
    for err, (feat, op) in FEATURE_FOR.items():
        sub = cal[cal.view.isin(VIEW_FOR[err])].dropna(subset=[err])
        if sub.empty:
            th[err] = DEFAULT_TH[err]; continue
        best = (-1, -1, DEFAULT_TH[err])
        for t in np.unique(sub[feat]):
            yhat = (sub[feat] >= t) if op == ">" else (sub[feat] <= t)
            p, r, f, _ = prf(sub[err], yhat.astype(int))
            if (f, r) > best[:2]:  # max F1, tie-break on recall (your key metric)
                best = (f, r, float(t))
        th[err] = best[2]
        print(f"{err}: threshold {feat} {op}= {best[2]:.3f}  (calib F1={best[0]:.2f}, R={best[1]:.2f})")
    # store as strict thresholds used by predict()
    eps = {"depth_err": -1e-6, "lean_err": -1e-6, "valgus_err": 1e-6}
    json.dump({k: v + eps[k] for k, v in th.items()}, open("thresholds.json", "w"), indent=2)


def scores(d, th):
    rows = []
    for err in ERRORS:
        sub = d[d.view.isin(VIEW_FOR[err])].dropna(subset=[err])
        if sub.empty:
            continue
        yhat = [predict(r, th, r["view"])[err] for _, r in sub.iterrows()]
        p, r, f, n = prf(sub[err].astype(int), yhat)
        rows.append({"error": err, "precision": p, "recall": r, "f1": f, "positives": n, "reps": len(sub)})
    return pd.DataFrame(rows)


def cmd_evaluate():
    th = json.load(open("thresholds.json"))
    d = load_labeled(); test = d[d.split == "test"]
    print("\n== Overall test set ==")
    print(scores(test, th).round(3).to_string(index=False))
    # rep counting accuracy
    det = d.groupby("clip").rep.max().rename("detected")
    cnt = pd.read_csv("clips.csv").set_index("clip").join(det)
    cnt["abs_err"] = (cnt.detected - cnt.true_reps).abs()
    print(f"\nRep counting: mean abs error {cnt.abs_err.mean():.2f} reps/clip, "
          f"exact on {(cnt.abs_err == 0).mean():.0%} of clips")
    # robustness: vary one factor, compare F1 to baseline value of that factor
    print("\n== Robustness (F1 by condition) ==")
    for factor in ["camera_angle", "lighting", "speed", "person"]:
        if factor not in test:
            continue
        for val, g in test.groupby(factor):
            s = scores(g, th)
            if not s.empty:
                print(f"{factor}={val}: " + ", ".join(f"{r.error} F1={r.f1:.2f} (n={r.reps})" for r in s.itertuples()))
    print("\nNote: if a condition has <10 reps, report it but say the estimate is noisy.")


def cmd_live(view="side", src=0):
    th = json.load(open("thresholds.json")) if os.path.exists("thresholds.json") else DEFAULT_TH
    mp, lm = make_landmarker()
    cap = cv2.VideoCapture(int(src) if str(src).isdigit() else src)
    rc, buf, reps, msg = RepCounter(), [], 0, ""
    t0, lat, i = time.time(), [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = time.time()
        d = frame_landmarks(mp, lm, frame, int((t - t0) * 1000))
        if d:
            buf = (buf + [d])[-5:]
            feats = frame_features(pd.DataFrame(buf)).iloc[-1].to_dict()
            rep = rc.update(i, feats)
            if rep:
                reps += 1
                errs = [e for e, v in predict(rep, th, view).items() if v]
                msg = "GOOD REP" if not errs else " / ".join(
                    {"depth_err": "GO DEEPER", "lean_err": "CHEST UP", "valgus_err": "KNEES OUT"}[e] for e in errs)
            for i_ in KEEP:
                cv2.circle(frame, (int(d[f"x{i_}"]), int(d[f"y{i_}"])), 5, (0, 255, 0), -1)
        lat.append((time.time() - t) * 1000); i += 1
        fps = i / (time.time() - t0)
        cv2.putText(frame, f"Reps {reps}  {msg}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0) if msg == "GOOD REP" else (0, 0, 255), 2)
        cv2.putText(frame, f"FPS {fps:.1f}  latency {np.mean(lat[-30:]):.0f} ms", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("squat", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    print(f"Avg FPS {fps:.1f}, mean per-frame latency {np.mean(lat):.1f} ms")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "extract": cmd_extract(sys.argv[2] if len(sys.argv) > 2 else "videos")
    elif cmd == "reps": cmd_reps()
    elif cmd == "calibrate": cmd_calibrate()
    elif cmd == "evaluate": cmd_evaluate()
    elif cmd == "live": cmd_live(*(sys.argv[2:4]))
    else: print(__doc__)