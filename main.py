"""
Hand Tracking Drawing App  —  works with mediapipe >= 0.10
==========================================================
Install:
    pip install opencv-python mediapipe numpy

On first run the script downloads the hand-landmarker model (~9 MB) into the
same folder as this file.  Subsequent runs use the cached file.

Controls
--------
  Index finger only          -> DRAW with current colour
  Index + Middle finger up   -> ERASE  (large circle rubber)
  Thumb only (rest down)     -> CLEAR canvas  (hold ~1 s)
  Hover index tip on top bar -> SELECT colour swatch
  Q                          -> Quit
"""

import cv2
import numpy as np
import time
import urllib.request
import os
import sys

# ── mediapipe new-style imports ───────────────────────────────────────────────
import mediapipe as mp
from mediapipe.tasks                import python as mp_python
from mediapipe.tasks.python         import BaseOptions
from mediapipe.tasks.python.vision  import (
    HandLandmarker, HandLandmarkerOptions, HandLandmarkerResult,
    RunningMode, HandLandmarksConnections,
)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "hand_landmarker.task")

BRUSH_SIZE   = 8
ERASER_SIZE  = 50
SMOOTHING    = 0.35          # lower = more responsive, higher = smoother

PALETTE = [
    ("Red",    (0,   0,   220)),
    ("Orange", (0,   140, 255)),
    ("Yellow", (0,   220, 220)),
    ("Green",  (30,  180,  30)),
    ("Cyan",   (200, 200,   0)),
    ("Blue",   (220,  40,  40)),
    ("Purple", (200,   0, 200)),
    ("White",  (255, 255, 255)),
    ("Black",  (  0,   0,   0)),
]

SWATCH_W, SWATCH_H = 50, 40
SWATCH_GAP         = 6
BAR_H              = 70

# Landmark indices
TIP_IDS  = [4, 8, 12, 16, 20]   # thumb, index, middle, ring, pinky
PIP_IDS  = [2, 6, 10, 14, 18]


# ── Model download ────────────────────────────────────────────────────────────
def ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    print("[hand_draw] Downloading hand-landmarker model (~9 MB)...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"[hand_draw] Model saved to {MODEL_PATH}")
    except Exception as e:
        print(
            f"\n[hand_draw] ERROR: Could not auto-download the model.\n"
            f"  URL : {MODEL_URL}\n"
            f"  Save: {MODEL_PATH}\n"
            f"  Err : {e}\n\n"
            "Please download the file manually and place it next to this script."
        )
        sys.exit(1)


# ── Geometry helpers ──────────────────────────────────────────────────────────
def fingers_up(lms, is_right: bool):
    """Return [thumb, index, middle, ring, pinky] — 1 = extended."""
    up = []
    # Thumb: compare x (mirrored camera, so left/right swapped)
    if is_right:
        up.append(1 if lms[4].x < lms[3].x else 0)
    else:
        up.append(1 if lms[4].x > lms[3].x else 0)
    for tip, pip in zip(TIP_IDS[1:], PIP_IDS[1:]):
        up.append(1 if lms[tip].y < lms[pip].y else 0)
    return up


def swatch_rect(i):
    x1 = 10 + i * (SWATCH_W + SWATCH_GAP)
    y1 = (BAR_H - SWATCH_H) // 2
    return x1, y1, x1 + SWATCH_W, y1 + SWATCH_H


def hit_swatch(cx, cy):
    for i in range(len(PALETTE)):
        x1, y1, x2, y2 = swatch_rect(i)
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            return i
    return None


def rounded_rect(img, x1, y1, x2, y2, r, color, thickness=-1):
    cv2.rectangle(img, (x1+r, y1), (x2-r, y2), color, thickness)
    cv2.rectangle(img, (x1, y1+r), (x2, y2-r), color, thickness)
    for cx, cy in [(x1+r,y1+r),(x2-r,y1+r),(x1+r,y2-r),(x2-r,y2-r)]:
        cv2.circle(img, (cx, cy), r, color, thickness)


# ── UI overlay ────────────────────────────────────────────────────────────────
def draw_ui(frame, color_idx, mode):
    h, w = frame.shape[:2]
    # semi-transparent top bar
    bar_bg = frame.copy()
    cv2.rectangle(bar_bg, (0, 0), (w, BAR_H), (20, 20, 20), -1)
    cv2.addWeighted(bar_bg, 0.75, frame, 0.25, 0, frame)

    # colour swatches
    for i, (_, bgr) in enumerate(PALETTE):
        x1, y1, x2, y2 = swatch_rect(i)
        rounded_rect(frame, x1, y1, x2, y2, 6, bgr, -1)
        if i == color_idx:
            rounded_rect(frame, x1-3, y1-3, x2+3, y2+3, 8, (255,255,255), 2)

    # mode label
    font  = cv2.FONT_HERSHEY_SIMPLEX
    label = f"MODE: {mode}"
    (tw, th), _ = cv2.getTextSize(label, font, 0.65, 2)
    cv2.putText(frame, label, (w - tw - 14, BAR_H//2 + th//2),
                font, 0.65, (210, 210, 210), 2, cv2.LINE_AA)

    # bottom legend
    legend = [
        "Q = quit",
        "Hover index over bar = pick colour",
        "Thumb only = CLEAR (hold 1 s)",
        "Index + Middle = ERASE",
        "Index only = DRAW",
    ]
    for i, txt in enumerate(legend):
        cv2.putText(frame, txt, (10, h - 12 - i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (150,150,150), 1, cv2.LINE_AA)


# ── Hand skeleton ─────────────────────────────────────────────────────────────
CONNECTIONS = [(c.start, c.end) for c in HandLandmarksConnections.HAND_CONNECTIONS]

def draw_skeleton(frame, lms, w, h):
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
    for s, e in CONNECTIONS:
        cv2.line(frame, pts[s], pts[e], (80, 220, 80), 1, cv2.LINE_AA)
    for pt in pts:
        cv2.circle(frame, pt, 4, (255,255,255), -1)
        cv2.circle(frame, pt, 4, (60,200,60), 1)


# ── Smooth point ──────────────────────────────────────────────────────────────
def smooth(prev, raw):
    if prev is None:
        return raw
    return (
        int(prev[0]*SMOOTHING + raw[0]*(1-SMOOTHING)),
        int(prev[1]*SMOOTHING + raw[1]*(1-SMOOTHING)),
    )


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ensure_model()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    ret, frame = cap.read()
    if not ret:
        print("Cannot open camera.")
        return
    frame  = cv2.flip(frame, 1)
    h, w   = frame.shape[:2]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    color_idx  = 0
    prev_pt    = None
    smooth_pt  = None
    mode       = "IDLE"
    last_clear = 0

    # Async result container
    latest: list = [None]

    def on_result(result: HandLandmarkerResult, _img, _ts):
        latest[0] = result

    options = HandLandmarkerOptions(
        base_options      = BaseOptions(model_asset_path=MODEL_PATH),
        running_mode      = RunningMode.LIVE_STREAM,
        num_hands         = 1,
        min_hand_detection_confidence = 0.6,
        min_hand_presence_confidence  = 0.6,
        min_tracking_confidence       = 0.5,
        result_callback   = on_result,
    )

    with HandLandmarker.create_from_options(options) as detector:
        ts_ms = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame  = cv2.flip(frame, 1)
            ts_ms += 33   # monotonically increasing timestamp

            mp_img = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            )
            detector.detect_async(mp_img, ts_ms)

            result   = latest[0]
            cur_mode = "IDLE"

            if result and result.hand_landmarks:
                lms        = result.hand_landmarks[0]
                hand_label = result.handedness[0][0].category_name  # "Left"/"Right"
                is_right   = (hand_label == "Right")

                draw_skeleton(frame, lms, w, h)
                up = fingers_up(lms, is_right)

                # Index fingertip in pixel coords
                tip8 = (int(lms[8].x * w), int(lms[8].y * h))

                # ── CLEAR: thumb only ─────────────────────────────────────
                if up == [1, 0, 0, 0, 0]:
                    now = time.time()
                    if now - last_clear > 1.0:
                        canvas[:] = 0
                        last_clear = now
                    cur_mode  = "CLEAR"
                    prev_pt   = None
                    smooth_pt = None

                # ── ERASE: index + middle ─────────────────────────────────
                elif up[1] and up[2]:
                    cur_mode  = "ERASE"
                    smooth_pt = smooth(smooth_pt, tip8)
                    if smooth_pt[1] >= BAR_H:
                        cv2.circle(canvas, smooth_pt, ERASER_SIZE, (0,0,0), -1)
                        cv2.circle(frame,  smooth_pt, ERASER_SIZE, (180,180,180), 2)
                    prev_pt = None

                # ── DRAW: index only ──────────────────────────────────────
                elif up[1] and not up[2]:
                    cur_mode  = "DRAW"
                    smooth_pt = smooth(smooth_pt, tip8)

                    if smooth_pt[1] < BAR_H:           # toolbar area
                        idx = hit_swatch(*smooth_pt)
                        if idx is not None:
                            color_idx = idx
                        prev_pt = None
                    else:
                        color = PALETTE[color_idx][1]
                        if prev_pt and prev_pt[1] >= BAR_H:
                            cv2.line(canvas, prev_pt, smooth_pt,
                                     color, BRUSH_SIZE, cv2.LINE_AA)
                        else:
                            cv2.circle(canvas, smooth_pt, BRUSH_SIZE//2, color, -1)
                        prev_pt = smooth_pt
                        cv2.circle(frame, smooth_pt, BRUSH_SIZE, color, 2)

                else:
                    prev_pt   = None
                    smooth_pt = None
            else:
                prev_pt   = None
                smooth_pt = None

            mode = cur_mode

            # ── Merge canvas + frame ──────────────────────────────────────
            gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
            inv  = cv2.bitwise_not(mask)
            bg   = cv2.bitwise_and(frame,  frame,  mask=inv)
            fg   = cv2.bitwise_and(canvas, canvas, mask=mask)
            frame = cv2.add(bg, fg)

            draw_ui(frame, color_idx, mode)
            cv2.imshow("Hand Draw  |  Q = quit", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()