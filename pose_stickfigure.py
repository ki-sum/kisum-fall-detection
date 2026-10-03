"""
pose_stickfigure.py — Convert a video (file or webcam) into anonymized stick-figure.

Uses MediaPipe Tasks v1.0.0:
- PoseLandmarker → 33 body keypoints per frame
- ImageSegmenter → person mask for blur/silhouette

Modes:
  overlay     — skeleton on top of original video (debug, NOT anonymized)
  blur        — heavy Gaussian blur on person area + skeleton overlay
  silhouette  — flat colour fills person area + skeleton overlay
  skeleton    — solid colour background (--colour), ONLY skeleton visible (max abstract)
  heatmap     — DensePose-style thermal gradient on person + skeleton (needs --palette)

Usage:
  # Batch: file in → file out
  python pose_stickfigure.py input.mp4 -o output.mp4 --mode blur

  # Live webcam preview only (no file save)
  python pose_stickfigure.py webcam --mode blur

  # Webcam preview + record (SPACE to start/stop recording, Q to quit)
  python pose_stickfigure.py webcam -o session.mp4 --mode blur

  # Webcam with specific device index (default 0)
  python pose_stickfigure.py webcam:1 --mode silhouette

First run auto-downloads MediaPipe models to ./mediapipe_models/ (~10 MB total).
"""

import argparse
import math
import os
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

MODEL_DIR = Path(__file__).parent / "mediapipe_models"
MODEL_URLS = {
    "pose_landmarker_lite.task":
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "pose_landmarker_full.task":
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
    "selfie_segmenter.tflite":
        "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/1/selfie_segmenter.tflite",
}

# 33 landmarks connection list (skeleton "bones")
# See https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
POSE_CONNECTIONS = [
    # torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # left arm
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    # right arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    # left leg
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    # right leg
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
    # face (minimal, keep it abstract)
    (9, 10),
    # nose to shoulders (neck)
    (0, 11), (0, 12),
]

# Skeleton colours (BGR — OpenCV order)
COLOUR_BONE = (110, 240, 90)     # bright green
COLOUR_JOINT = (100, 220, 255)   # cyan-yellow
COLOUR_SILHOUETTE_DEFAULT = (40, 40, 60)  # dark navy grey (BGR)

# Heatmap palettes for --mode heatmap (name → cv2 colormap constant)
PALETTES = {
    "viridis": cv2.COLORMAP_VIRIDIS,   # dark-blue → green → yellow (DensePose style)
    "inferno": cv2.COLORMAP_INFERNO,   # black → red → yellow
    "plasma":  cv2.COLORMAP_PLASMA,    # purple → pink → yellow
    "magma":   cv2.COLORMAP_MAGMA,     # black → purple → yellow
    "jet":     cv2.COLORMAP_JET,       # blue → green → red (classic rainbow)
    "hot":     cv2.COLORMAP_HOT,       # black → red → white (thermal camera)
    "cool":    cv2.COLORMAP_COOL,      # cyan → magenta
    "turbo":   cv2.COLORMAP_TURBO,     # improved rainbow
    "bone":    cv2.COLORMAP_BONE,      # blue-grey → white (medical x-ray)
}


class OneEuroFilter:
    """Real-time low-pass filter with velocity-adaptive cutoff.

    Reference: https://gery.casiez.net/1euro/ (Casiez, Roussel, Vogel 2012).
    Smooths jitter at low speeds; passes through fast motion with minimal lag.
    """
    def __init__(self, min_cutoff=1.5, beta=0.05, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.x_prev is None or self.t_prev is None:
            self.x_prev, self.t_prev = x, t
            return x
        dt = t - self.t_prev
        if dt <= 0:
            return self.x_prev
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat


class _SmoothedLandmark:
    __slots__ = ("x", "y", "z", "visibility")
    def __init__(self, x, y, z, v):
        self.x, self.y, self.z, self.visibility = x, y, z, v


class LandmarkSmoother:
    """Apply OneEuroFilter to each of 33 landmarks' (x, y, z) independently.

    strength=0.0  → no smoothing (raw)
    strength=0.5  → moderate (default, removes most standstill jitter)
    strength=1.0  → heavy (may lag on fast motion)
    """
    def __init__(self, n_landmarks=33, strength=0.5):
        # Map strength (0..1) → OneEuroFilter min_cutoff (higher = less smoothing)
        # strength 0.0 → min_cutoff 30 (essentially raw)
        # strength 0.5 → min_cutoff 1.5 (good default)
        # strength 1.0 → min_cutoff 0.3 (very smooth)
        min_cutoff = max(0.3, 30.0 - strength * 29.7)
        beta = 0.05 + strength * 0.1
        self.filters = [
            (OneEuroFilter(min_cutoff, beta),
             OneEuroFilter(min_cutoff, beta),
             OneEuroFilter(min_cutoff, beta))
            for _ in range(n_landmarks)
        ]
        self.strength = strength

    def smooth(self, landmarks, t):
        if self.strength <= 0.0:
            return landmarks  # bypass
        out = []
        for i, lm in enumerate(landmarks):
            fx, fy, fz = self.filters[i]
            out.append(_SmoothedLandmark(fx(lm.x, t), fy(lm.y, t), fz(lm.z, t), lm.visibility))
        return out


def ensure_model(name: str) -> Path:
    """Download the model file if not present. Returns local path."""
    MODEL_DIR.mkdir(exist_ok=True)
    dest = MODEL_DIR / name
    if dest.exists():
        return dest
    url = MODEL_URLS[name]
    print(f"downloading {name} from {url}...")
    try:
        urllib.request.urlretrieve(url, dest)
        size_kb = dest.stat().st_size / 1024
        print(f"  saved {dest} ({size_kb:.0f} KB)")
    except Exception as e:
        print(f"  ERROR downloading {name}: {e}", file=sys.stderr)
        if dest.exists():
            dest.unlink()
        raise
    return dest


def create_pose_landmarker(model_path: Path, want_seg=True):
    base_opts = mp_tasks.BaseOptions(model_asset_path=str(model_path))
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        # Pose-aligned person mask — cleaner than selfie_segmenter and matches
        # the detected pose exactly. Costs a bit more compute.
        output_segmentation_masks=want_seg,
    )
    return mp_vision.PoseLandmarker.create_from_options(opts)


def create_segmenter(model_path: Path):
    base_opts = mp_tasks.BaseOptions(model_asset_path=str(model_path))
    opts = mp_vision.ImageSegmenterOptions(
        base_options=base_opts,
        running_mode=mp_vision.RunningMode.VIDEO,
        output_category_mask=True,
        output_confidence_masks=False,
    )
    return mp_vision.ImageSegmenter.create_from_options(opts)


def _build_skeleton_heat(w, h, landmarks, bone_thickness=22, joint_radius=14, blur_kernel=71):
    """Rasterize bones + joints as bright shapes, blur into a smooth heat field.

    Returns uint8 (h, w). Hottest along the skeleton, cooling out toward body edges.
    """
    hot = np.zeros((h, w), dtype=np.float32)
    if landmarks is None:
        return hot.astype(np.uint8)
    pts = []
    for lm in landmarks:
        pts.append((int(lm.x * w), int(lm.y * h), lm.visibility))
    # Bones (thicker = wider heat plume)
    for a, b in POSE_CONNECTIONS:
        if a >= len(pts) or b >= len(pts):
            continue
        xa, ya, va = pts[a]
        xb, yb, vb = pts[b]
        if va < 0.3 or vb < 0.3:
            continue
        cv2.line(hot, (xa, ya), (xb, yb), 1.0, thickness=bone_thickness, lineType=cv2.LINE_AA)
    # Joints (brighter hot-spots at articulation points)
    for x, y, v in pts:
        if v < 0.3:
            continue
        cv2.circle(hot, (x, y), joint_radius, 1.3, -1, cv2.LINE_AA)
    # Big blur → smooth radial fall-off from the skeleton
    if blur_kernel > 1:
        # kernel must be odd
        k = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        hot = cv2.GaussianBlur(hot, (k, k), 0)
    m = float(hot.max())
    if m > 0:
        hot = (hot / m * 255.0).astype(np.uint8)
    else:
        hot = hot.astype(np.uint8)
    return hot


def apply_anonymization(frame_bgr, mask_np, landmarks, mode, silhouette_colour,
                        palette_cv=None, pose_seg=None):
    """Return frame with the person area replaced according to mode.

    mask_np    : selfie_segmenter uint8 mask (used by blur/silhouette).
    landmarks  : list of pose landmarks (used by heatmap for skeleton heat).
    pose_seg   : PoseLandmarker's built-in segmentation mask (float 0..1), preferred
                 for heatmap because it's aligned to the detected person.
    """
    if mode == "overlay":
        return frame_bgr  # no anonymization

    if mode == "skeleton":
        # Solid colour background — segmenter not needed. Skeleton drawn later.
        return np.full_like(frame_bgr, silhouette_colour, dtype=np.uint8)

    # heatmap runs on pose_seg (preferred) OR selfie mask_np OR falls back to
    # heat-as-alpha — it does NOT require selfie mask. Handle it BEFORE the
    # mask_np None-check (which is only relevant for blur/silhouette).
    if mode == "heatmap":
        h_, w_ = frame_bgr.shape[:2]
        hot = _build_skeleton_heat(w_, h_, landmarks)
        colormap = palette_cv if palette_cv is not None else cv2.COLORMAP_VIRIDIS
        heatmap = cv2.applyColorMap(hot, colormap)
        bg = np.full_like(frame_bgr, silhouette_colour, dtype=np.uint8)

        body_mask_bool = None
        if pose_seg is not None:
            pose_seg_np = pose_seg
            if pose_seg_np.ndim == 3:
                pose_seg_np = pose_seg_np.squeeze(-1)
            body_mask_bool = pose_seg_np > 0.5
        elif mask_np is not None:
            m = mask_np
            if m.ndim == 3:
                m = m.squeeze(-1)
            body_mask_bool = (m == 0)  # selfie: 0 = person

        if body_mask_bool is not None:
            body_mask_3 = np.repeat(body_mask_bool[:, :, None], 3, axis=2)
            out = np.where(body_mask_3, heatmap, bg)
        else:
            alpha = (hot.astype(np.float32) / 255.0)[:, :, None]
            out = (heatmap.astype(np.float32) * alpha +
                   bg.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
        return out

    if mask_np is None:
        return frame_bgr

    # MediaPipe may return mask as (H, W) or (H, W, 1). Force to 2D.
    if mask_np.ndim == 3:
        mask_np = mask_np.squeeze(-1)

    # selfie_segmenter.tflite emits category 0 = background, 1 = person? or vice-versa?
    # Testing empirically: person region has value 0 in category mask. Adjust if needed.
    # We treat non-zero as background to be safe; caller can pass inverted mask.
    person_mask = (mask_np == 0).astype(np.uint8)  # 1 where person

    if mode == "blur":
        blurred = cv2.GaussianBlur(frame_bgr, (51, 51), 0)
        # Also darken slightly to reduce recognizability
        blurred = (blurred * 0.7).astype(np.uint8)
        person_mask_3 = np.repeat(person_mask[:, :, None], 3, axis=2)
        out = np.where(person_mask_3 == 1, blurred, frame_bgr)
        return out

    if mode == "silhouette":
        colour_arr = np.full_like(frame_bgr, silhouette_colour, dtype=np.uint8)
        person_mask_3 = np.repeat(person_mask[:, :, None], 3, axis=2)
        out = np.where(person_mask_3 == 1, colour_arr, frame_bgr)
        return out

    return frame_bgr


def draw_skeleton(frame_bgr, landmarks_norm, w, h,
                  bone_colour=COLOUR_BONE, joint_colour=COLOUR_JOINT,
                  bone_thick=3, joint_radius=4):
    """Draw skeleton bones + joints on the frame in-place."""
    # landmarks_norm is list of NormalizedLandmark with .x .y .z .visibility
    pts = []
    for lm in landmarks_norm:
        x = int(lm.x * w)
        y = int(lm.y * h)
        pts.append((x, y, lm.visibility))
    # Bones
    for a, b in POSE_CONNECTIONS:
        if a >= len(pts) or b >= len(pts):
            continue
        xa, ya, va = pts[a]
        xb, yb, vb = pts[b]
        if va < 0.3 or vb < 0.3:
            continue
        cv2.line(frame_bgr, (xa, ya), (xb, yb), bone_colour, bone_thick, cv2.LINE_AA)
    # Joints
    for x, y, v in pts:
        if v < 0.3:
            continue
        cv2.circle(frame_bgr, (x, y), joint_radius, joint_colour, -1, cv2.LINE_AA)


def _open_source(source):
    """Return (VideoCapture, is_webcam, meta_dict)."""
    if isinstance(source, str) and source.startswith("webcam"):
        # "webcam" or "webcam:N"
        idx = 0
        if ":" in source:
            try:
                idx = int(source.split(":", 1)[1])
            except ValueError:
                pass
        print(f"opening webcam index {idx}...")
        # CAP_DSHOW is more reliable on Windows for USB webcams
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            # fall back to default backend
            cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            raise RuntimeError(f"could not open webcam index {idx}")
        # Ask for 720p / 30fps
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        print(f"  webcam opened: {w}x{h} @ {fps:.0f}fps")
        return cap, True, {"w": w, "h": h, "fps": fps, "n_frames": 0}

    # File input
    print(f"opening {source}...")
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  {w}x{h}  {fps:.1f}fps  {n_frames} frames  ({n_frames/fps:.1f}s)")
    return cap, False, {"w": w, "h": h, "fps": fps, "n_frames": n_frames}


def process_video(input_source, output_path, mode, silhouette_colour,
                  model_size="lite", every_n=1, preview=False, palette_cv=None,
                  smooth_strength=0.5):
    pose_model = "pose_landmarker_lite.task" if model_size == "lite" else "pose_landmarker_full.task"
    pose_path = ensure_model(pose_model)
    # segmenter needed for anything that shapes the person body
    seg_path = ensure_model("selfie_segmenter.tflite") if mode in ("blur", "silhouette", "heatmap") else None

    cap, is_webcam, meta = _open_source(input_source)
    w, h, fps, n_frames = meta["w"], meta["h"], meta["fps"], meta["n_frames"]

    # In webcam mode, always show preview. Recording is toggled by SPACE.
    if is_webcam:
        preview = True

    # Writer is set up on-demand for webcam (only when recording is toggled on)
    writer = None
    recording = False

    def _open_writer(path):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w_ = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
        if not w_.isOpened():
            raise RuntimeError(f"could not open output video for write: {path}")
        return w_

    if not is_webcam:
        if not output_path:
            raise RuntimeError("--output required when processing a file input")
        writer = _open_writer(output_path)
        recording = True

    # Enable pose seg mask for heatmap mode (nicer body outline than selfie_segmenter)
    landmarker = create_pose_landmarker(pose_path, want_seg=(mode == "heatmap"))
    segmenter = create_segmenter(seg_path) if seg_path else None
    smoother = LandmarkSmoother(strength=smooth_strength)

    idx = 0
    frames_recorded = 0
    t_start = time.time()
    last_landmarks = None
    last_mask = None
    last_pose_seg = None
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                if is_webcam:
                    print("webcam read failed, retrying...")
                    time.sleep(0.1)
                    continue
                break

            # Use wall-clock delta for webcam timestamps; frame index * 1000/fps for file
            if is_webcam:
                timestamp_ms = int((time.time() - t_start) * 1000)
            else:
                timestamp_ms = int(idx / fps * 1000)

            if idx % every_n == 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

                pose_result = landmarker.detect_for_video(mp_image, timestamp_ms)
                if pose_result.pose_landmarks and len(pose_result.pose_landmarks) > 0:
                    raw_landmarks = pose_result.pose_landmarks[0]
                    # Apply temporal smoothing (OneEuroFilter) to reduce standstill jitter
                    last_landmarks = smoother.smooth(raw_landmarks, timestamp_ms / 1000.0)
                else:
                    last_landmarks = None

                # Pose landmarker's built-in seg mask (only if enabled)
                if hasattr(pose_result, "segmentation_masks") and pose_result.segmentation_masks:
                    last_pose_seg = pose_result.segmentation_masks[0].numpy_view().copy()
                else:
                    last_pose_seg = None

                if segmenter is not None:
                    seg_result = segmenter.segment_for_video(mp_image, timestamp_ms)
                    if seg_result.category_mask is not None:
                        last_mask = seg_result.category_mask.numpy_view().copy()
                    else:
                        last_mask = None

            frame_out = apply_anonymization(frame_bgr, last_mask, last_landmarks,
                                            mode, silhouette_colour, palette_cv,
                                            pose_seg=last_pose_seg)
            if last_landmarks is not None:
                draw_skeleton(frame_out, last_landmarks, w, h)

            elapsed = time.time() - t_start
            if is_webcam:
                rec_flag = "● REC" if recording else "○ paused (SPACE=rec Q=quit)"
                info = f"{rec_flag}   mode={mode}   {idx} frames   {elapsed:.0f}s"
            else:
                eta = (elapsed / (idx + 1)) * (n_frames - idx - 1) if idx > 5 else 0
                info = f"{idx+1}/{n_frames}  {elapsed:.0f}s elapsed  ETA {eta:.0f}s"
            cv2.putText(frame_out, info, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 255, 0) if recording else (200, 200, 200), 2, cv2.LINE_AA)

            if recording and writer is not None:
                writer.write(frame_out)
                frames_recorded += 1

            if preview:
                cv2.imshow("pose stickfigure preview  (SPACE=rec  Q=quit)", frame_out)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("stopped by user")
                    break
                elif key == ord(" ") and is_webcam:
                    if not recording:
                        if not output_path:
                            print("no --output given, cannot record")
                        else:
                            writer = _open_writer(output_path)
                            recording = True
                            frames_recorded = 0
                            print(f"[REC ON] writing to {output_path}")
                    else:
                        recording = False
                        if writer is not None:
                            writer.release()
                            writer = None
                        print(f"[REC OFF] {frames_recorded} frames recorded")

            idx += 1
            if not is_webcam and idx % 30 == 0:
                elapsed = time.time() - t_start
                eta = (elapsed / (idx + 1)) * (n_frames - idx - 1) if idx > 5 else 0
                print(f"  frame {idx}/{n_frames}  ({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        landmarker.close()
        if segmenter is not None:
            segmenter.close()
        if preview:
            cv2.destroyAllWindows()

    total = time.time() - t_start
    if is_webcam:
        print(f"done — {frames_recorded} frames recorded in {total:.1f}s")
    else:
        print(f"done — wrote {output_path} ({idx} frames in {total:.1f}s, {idx/total:.1f} fps)")


def parse_colour(s: str):
    """Parse 'R,G,B' → BGR tuple."""
    parts = s.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("colour must be R,G,B (three ints 0-255)")
    r, g, b = [int(p) for p in parts]
    return (b, g, r)  # OpenCV BGR


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("input", help="input video file (mp4/mov) OR 'webcam' / 'webcam:N'")
    ap.add_argument("-o", "--output", default=None,
                    help="output MP4 file (required for file input; optional for webcam preview)")
    ap.add_argument("--mode", choices=["overlay", "blur", "silhouette", "skeleton", "heatmap"], default="blur",
                    help="anonymization mode (default: blur)")
    ap.add_argument("--colour", type=parse_colour, default=COLOUR_SILHOUETTE_DEFAULT,
                    help="fill colour for silhouette/skeleton/heatmap-bg modes as R,G,B (default: dark navy)")
    ap.add_argument("--palette", choices=list(PALETTES.keys()), default="viridis",
                    help="colour palette for heatmap mode (default: viridis, DensePose-style)")
    ap.add_argument("--smooth", type=float, default=0.5,
                    help="landmark smoothing strength 0.0-1.0 (default 0.5). "
                         "0=raw jitter, 0.5=removes standstill jitter, 1.0=very smooth (laggy on fast motion)")
    ap.add_argument("--model", choices=["lite", "full"], default="lite",
                    help="pose model size — lite = fast, full = better accuracy")
    ap.add_argument("--every", type=int, default=1,
                    help="run pose every N frames (default: 1). Set to 2-3 to speed up.")
    ap.add_argument("--preview", action="store_true",
                    help="show live preview window (slower)")
    args = ap.parse_args()

    is_webcam = args.input.startswith("webcam")
    if not is_webcam and not os.path.exists(args.input):
        print(f"input not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    if not is_webcam and not args.output:
        print("--output is required when input is a file", file=sys.stderr)
        sys.exit(1)

    process_video(args.input, args.output, args.mode, args.colour,
                  model_size=args.model, every_n=args.every, preview=args.preview,
                  palette_cv=PALETTES.get(args.palette),
                  smooth_strength=max(0.0, min(1.0, args.smooth)))


if __name__ == "__main__":
    main()
