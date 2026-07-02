"""
Generate synthetic sensor + video-frame data that mimics a real capture session.

Resolves:
  1. Different sample rates (sensor ~200 Hz, video ~30 fps)
  2. A time offset between the two streams (they start at different moments)

Each "action" is a labeled segment with a characteristic accelerometer signature.
A sharp tap at the very start produces a spike in BOTH streams -- sync marker.
"""

import numpy as np
import pandas as pd

ACTIONS = ["stack", "flip", "screw"]
SENSOR_HZ = 200
VIDEO_FPS = 30

def _action_signal(action: str, n: int, rng: np.random.Generator) -> np.ndarray:
    """Return an accelerometer-magnitude signal characteristic of each action."""
    t = np.linspace(0, 1, n)
    if action == "stack":
        # discrete impacts: pick up, place down
        sig = np.zeros(n)
        for c in (0.25, 0.75):
            sig += 6 * np.exp(-((t - c) ** 2) / 0.0008)
    elif action == "flip":
        # one quick snap
        sig = 8 * np.exp(-((t - 0.5) ** 2) / 0.0004)
    else:  # screw
        # rhythmic oscillation
        sig = 2.5 * np.sin(2 * np.pi * 6 * t) ** 2
    sig += rng.normal(0, 0.3, n)  # sensor noise
    return sig + 9.81  # gravity baseline

def generate(seed: int = 0, offset_s: float = 1.37):
    """
    Build paired sensor + video streams with a known offset.

    Returns (sensor_df, video_df, true_offset_s). The true offset is returned
    only so you can check your sync-detection later -- in real life you don't get it.
    """
    rng = np.random.default_rng(seed)

    # --- build the master timeline of actions ---
    segments = []  # (action, duration_s)
    segments.append(("tap", 0.4))  # the sync marker -- wide enough for ~12 video frames
    segments.append(("idle", 0.5))
    for _ in range(9):  # 9 action reps, 3 of each, shuffled
        pass
    reps = ACTIONS * 3
    rng.shuffle(reps)
    for a in reps:
        segments.append((a, rng.uniform(1.2, 2.0)))
        segments.append(("idle", rng.uniform(0.3, 0.6)))

    # --- sensor stream (high rate) ---
    s_rows = []
    clock = 0.0
    for action, dur in segments:
        n = max(1, int(dur * SENSOR_HZ))
        ts = clock + np.arange(n) / SENSOR_HZ
        if action == "tap":
            # Wide, dominant spike: must survive 30fps video sampling AND be the
            # single largest event in the whole recording, so it's unambiguous in
            # both streams. (Real-world lesson: a sharp clap that's too brief can
            # fall between video frames -- make the marker firm and a touch held.)
            mag = np.full(n, 9.81) + 25 * np.exp(
                -((np.linspace(0, 1, n) - 0.3) ** 2) / 0.02
            )
            label = "tap"
        elif action == "idle":
            mag = np.full(n, 9.81) + rng.normal(0, 0.15, n)
            label = "idle"
        else:
            mag = _action_signal(action, n, rng)
            label = action
        for tt, mm in zip(ts, mag):
            s_rows.append((tt, mm, label))
        clock += dur

    sensor_df = pd.DataFrame(s_rows, columns=["t_sensor", "accel_mag", "label_true"])

    # --- video stream (low rate), shifted by the offset ---
    total = clock
    n_frames = int(total * VIDEO_FPS)
    frame_t = np.arange(n_frames) / VIDEO_FPS
    # "brightness of motion" proxy -- a crude visual signal that also spikes on the tap
    v_signal = np.interp(frame_t, sensor_df["t_sensor"], sensor_df["accel_mag"])
    v_signal = v_signal + rng.normal(0, 0.4, n_frames)
    video_df = pd.DataFrame(
        {"t_video": frame_t + offset_s, "motion_proxy": v_signal}
    )

    return sensor_df, video_df, offset_s

if __name__ == "__main__":
    sensor_df, video_df, offset = generate()
    sensor_df.to_csv("data/raw/sensor.csv", index=False)
    video_df.to_csv("data/raw/video.csv", index=False)
    print(f"Wrote {len(sensor_df)} sensor rows, {len(video_df)} video frames.")
    print(f"(Hidden) true offset between streams: {offset:.3f} s")
