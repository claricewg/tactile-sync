"""
Plot the aligned streams to sanity-check synchronization by eye.
If the tap spikes in both streams line up after alignment, the sync worked.
"""

import matplotlib
matplotlib.use("Agg")  # headless: save to file instead of opening a window
import matplotlib.pyplot as plt
import pandas as pd

from synchronize import estimate_offset

def plot_alignment(sensor_df, video_df, merged, offset, out_path="figures/alignment.png"):
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    # before alignment: raw clocks
    axes[0].plot(sensor_df["t_sensor"], sensor_df["accel_mag"], lw=0.8, label="sensor (accel mag)")
    axes[0].plot(video_df["t_video"], video_df["motion_proxy"], lw=0.8, alpha=0.7, label="video (motion proxy)")
    axes[0].set_title("Before alignment — streams on their own clocks")
    axes[0].set_ylabel("signal")
    axes[0].legend(loc="upper right")

    # after alignment: merged onto common timeline
    axes[1].plot(merged["t"], merged["accel_mag"], lw=0.8, label="sensor")
    axes[1].plot(merged["t"], merged["motion_proxy"], lw=0.8, alpha=0.7, label="video (shifted)")
    axes[1].set_title(f"After alignment — offset of {offset:.3f}s removed, merged with merge_asof")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("signal")
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"Saved {out_path}")

if __name__ == "__main__":
    sensor_df = pd.read_csv("data/raw/sensor.csv")
    video_df = pd.read_csv("data/raw/video.csv")
    merged = pd.read_csv("data/processed/merged.csv")
    offset = estimate_offset(sensor_df, video_df)
    plot_alignment(sensor_df, video_df, merged, offset)
