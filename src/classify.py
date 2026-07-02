"""
Classify the action from the FUSED (vision + sensor) signal.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

WINDOW_S = 0.5
STEP_S = 0.25
ACTION_LABELS = {"stack", "flip", "screw"}


def make_windows(merged: pd.DataFrame, use_video: bool = True):
    """Slice into overlapping windows; keep windows that are mostly one action."""
    t = merged["t"].values
    feats, labels = [], []
    start = t[0]
    while start + WINDOW_S <= t[-1]:
        w = merged[(merged["t"] >= start) & (merged["t"] < start + WINDOW_S)]
        start += STEP_S
        if len(w) < 5:
            continue
        # majority label in the window; skip mixed/idle/tap windows
        lab = w["label_true"].mode().iloc[0]
        if lab not in ACTION_LABELS:
            continue
        # only keep windows that are clearly one action (>70% of samples)
        if (w["label_true"] == lab).mean() < 0.7:
            continue

        a = w["accel_mag"].values
        f = [a.mean(), a.std(), a.max(), a.min(), np.ptp(a), np.median(a)]
        if use_video:
            v = w["motion_proxy"].values
            f += [np.nanmean(v), np.nanstd(v), np.nanmax(v)]
        feats.append(f)
        labels.append(lab)
    return np.array(feats), np.array(labels)


def run(use_video: bool = True, seed: int = 0):
    merged = pd.read_csv("data/processed/merged.csv")
    X, y = make_windows(merged, use_video=use_video)
    if len(set(y)) < 2:
        print("Not enough distinct actions to classify -- generate more reps.")
        return
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    clf = RandomForestClassifier(n_estimators=200, random_state=seed)
    clf.fit(Xtr, ytr)
    acc = clf.score(Xte, yte)
    tag = "sensor+video" if use_video else "sensor-only"
    print(f"[{tag}] test accuracy: {acc:.1%}  (n_windows={len(y)})")
    return acc


if __name__ == "__main__":
    print("Comparing modalities:")
    run(use_video=False)
    run(use_video=True)
