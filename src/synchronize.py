"""
Synchronize two sensor streams recorded at different rates with an unknown offset.

This is the heart of the project and the part that maps directly to 6thSense's
bottleneck: vision and touch arrive on different clocks and have to be aligned
before they're useful.

Pipeline:
  1. Detect the sync marker (the tap) in each stream as its largest early spike.
  2. Compute the offset between the two detected spikes.
  3. Shift the video clock onto the sensor clock.
  4. Merge both streams onto one common timeline with pandas.merge_asof.
"""

import numpy as np
import pandas as pd


def detect_sync_spike(t: np.ndarray, signal: np.ndarray, search_window_s: float = 3.0) -> float:
    """
    Return the timestamp of the sync tap: the largest spike within the first
    `search_window_s` seconds. We search only the start so a big mid-recording
    impact can't be mistaken for the marker.
    """
    t = np.asarray(t)
    signal = np.asarray(signal)
    mask = t <= (t[0] + search_window_s)
    idx = np.argmax(signal[mask])
    return float(t[mask][idx])


def estimate_offset(sensor_df: pd.DataFrame, video_df: pd.DataFrame) -> float:
    """
    Estimate how far the video clock leads/lags the sensor clock, using the tap.
    Positive result => video timestamps are ahead and must be shifted back.
    """
    s_tap = detect_sync_spike(sensor_df["t_sensor"].values, sensor_df["accel_mag"].values)
    v_tap = detect_sync_spike(video_df["t_video"].values, video_df["motion_proxy"].values)
    return v_tap - s_tap


def synchronize(sensor_df: pd.DataFrame, video_df: pd.DataFrame, tolerance_s: float = 0.02) -> pd.DataFrame:
    """
    Align the two streams onto the sensor clock and merge them.

    merge_asof matches each sensor sample to the nearest video frame within
    `tolerance_s`. This is the standard tool for as-of joins on time-series and
    exactly what a real vision+touch fusion pipeline uses.
    """
    offset = estimate_offset(sensor_df, video_df)

    # shift video onto the sensor clock
    video_aligned = video_df.copy()
    video_aligned["t"] = video_aligned["t_video"] - offset

    sensor = sensor_df.copy()
    sensor["t"] = sensor["t_sensor"]

    # merge_asof requires both keys sorted
    sensor = sensor.sort_values("t")
    video_aligned = video_aligned.sort_values("t")

    merged = pd.merge_asof(
        sensor,
        video_aligned[["t", "motion_proxy"]],
        on="t",
        direction="nearest",
        tolerance=tolerance_s,
    )
    return merged, offset


if __name__ == "__main__":
    sensor_df = pd.read_csv("data/raw/sensor.csv")
    video_df = pd.read_csv("data/raw/video.csv")

    merged, offset = synchronize(sensor_df, video_df)
    merged.to_csv("data/processed/merged.csv", index=False)

    matched = merged["motion_proxy"].notna().mean()
    print(f"Estimated offset: {offset:.3f} s")
    print(f"Fraction of sensor samples matched to a video frame: {matched:.1%}")
    print(f"Wrote {len(merged)} aligned rows to data/processed/merged.csv")
