#!/usr/bin/env python3
"""
Visualize sample raw traces (vs time) for recorded gestures and training labels.

Usage examples:
  - Raw sequences (phases shaded), one figure per gesture type:
      python tools/visualize_traces.py --source raw --out gesture_data/viz

  - Training windows, N samples per label:
      python tools/visualize_traces.py --source train --n-samples 2 --out gesture_data/viz_train
"""

import argparse
import json
import os
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


FINGERTIPS = [4, 8, 12, 16, 20]
PALM_IDX = [0, 5, 9, 13, 17]


def load_raw_sequences(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw sequences file not found: {path}")
    with open(path, "r") as f:
        data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Expected a list in continuous_sequences.json")
        return data


def load_training_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Training data file not found: {path}")
    with open(path, "r") as f:
        data = json.load(f)
        if not ("sequences" in data and "labels" in data):
            raise ValueError("Expected keys 'sequences' and 'labels' in training_data.json")
        return data


def compute_signals_from_landmarks_seq(landmarks_seq):
    """Compute simple, interpretable signals for a sequence of 21x3 landmarks per frame.

    Returns dict of np.ndarrays keyed by signal name.
    """
    arr = np.array(landmarks_seq)  # (T, 21, 3) normalized coords
    if arr.ndim != 3 or arr.shape[1] != 21 or arr.shape[2] != 3:
        raise ValueError("Expected shape (T, 21, 3)")
    T = arr.shape[0]

    palm_center = arr[:, PALM_IDX, :].mean(axis=1)  # (T, 3)
    index_tip = arr[:, 8, :]
    thumb_tip = arr[:, 4, :]

    # Spread: average distance from palm center to fingertips
    spread = []
    for t in range(T):
        pc = palm_center[t, :2]
        dists = [np.linalg.norm(arr[t, i, :2] - pc) for i in FINGERTIPS]
        spread.append(np.mean(dists))
    spread = np.array(spread)

    thumb_index_dist = np.linalg.norm((thumb_tip[:, :2] - index_tip[:, :2]), axis=1)

    signals = {
        "palm_center_x": palm_center[:, 0],
        "palm_center_y": palm_center[:, 1],
        "index_tip_x": index_tip[:, 0],
        "index_tip_y": index_tip[:, 1],
        "spread": spread,
        "thumb_index_dist": thumb_index_dist,
    }
    return signals


def shade_phases(ax, phases, alpha=0.12):
    colors = {
        "neutral": (0.7, 0.7, 0.7),
        "transitioning_to_gesture": (0.95, 0.85, 0.2),
        "active_gesture": (0.1, 0.8, 0.1),
        "transitioning_to_neutral": (0.95, 0.55, 0.1),
    }
    n = len(phases)
    if n == 0:
        return
    start = 0
    cur = phases[0]
    for i in range(1, n + 1):
        if i == n or phases[i] != cur:
            color = colors.get(cur, (0.9, 0.9, 0.9))
            ax.axvspan(start, i - 1, color=color, alpha=alpha, lw=0)
            if i < n:
                start = i
                cur = phases[i]


def plot_raw_sequences(sequences, outdir, gestures=None):
    os.makedirs(outdir, exist_ok=True)

    by_type = defaultdict(list)
    for seq in sequences:
        by_type[seq.get("gesture_type", "unknown")].append(seq)

    selected_gestures = gestures or sorted(by_type.keys())

    for g in selected_gestures:
        seqs = by_type.get(g, [])
        if not seqs:
            continue
        # plot each recorded sequence for this gesture type
        for si, seq in enumerate(seqs):
            phases = seq.get("phases", [])
            if not phases:
                continue
            lm_seq = [p["landmarks"] for p in phases]
            phase_names = [p.get("phase", "") for p in phases]

            signals = compute_signals_from_landmarks_seq(lm_seq)
            T = len(lm_seq)
            x = np.arange(T)

            fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
            fig.suptitle(f"Raw trace: {g} (sequence {si+1}/{len(seqs)})")

            # Palm center
            axes[0].plot(x, signals["palm_center_x"], label="palm_center_x")
            axes[0].plot(x, signals["palm_center_y"], label="palm_center_y")
            axes[0].set_ylabel("Palm center")
            axes[0].legend(loc="upper right")
            shade_phases(axes[0], phase_names)

            # Spread
            axes[1].plot(x, signals["spread"], color="tab:green")
            axes[1].set_ylabel("Spread")
            shade_phases(axes[1], phase_names)

            # Index tip
            axes[2].plot(x, signals["index_tip_x"], label="idx_x", color="tab:orange")
            axes[2].plot(x, signals["index_tip_y"], label="idx_y", color="tab:red")
            axes[2].set_ylabel("Index tip")
            axes[2].legend(loc="upper right")
            shade_phases(axes[2], phase_names)

            # Thumb-index distance
            axes[3].plot(x, signals["thumb_index_dist"], color="tab:purple")
            axes[3].set_ylabel("Thumb-Index")
            shade_phases(axes[3], phase_names)
            axes[3].set_xlabel("Frame")

            outpath = os.path.join(outdir, f"raw_{g}_seq{si+1}.png")
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig.savefig(outpath, dpi=150)
            plt.close(fig)
            print(f"Saved: {outpath}")


def plot_training_windows(train_data, outdir, n_samples=1, labels=None, seed=42):
    os.makedirs(outdir, exist_ok=True)
    rnd = random.Random(seed)

    X = train_data["sequences"]
    y = train_data["labels"]

    by_label = defaultdict(list)
    for i, lbl in enumerate(y):
        by_label[lbl].append(i)

    selected_labels = labels or sorted(by_label.keys())

    for lbl in selected_labels:
        idxs = by_label.get(lbl, [])
        if not idxs:
            continue
        rnd.shuffle(idxs)
        for j, idx in enumerate(idxs[:n_samples]):
            lm_seq = X[idx]  # list[T][21][3]
            signals = compute_signals_from_landmarks_seq(lm_seq)
            T = len(lm_seq)
            x = np.arange(T)

            fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
            fig.suptitle(f"Training window: {lbl} (sample {j+1}/{n_samples})")

            axes[0].plot(x, signals["palm_center_x"], label="palm_center_x")
            axes[0].plot(x, signals["palm_center_y"], label="palm_center_y")
            axes[0].set_ylabel("Palm center")
            axes[0].legend(loc="upper right")

            axes[1].plot(x, signals["spread"], color="tab:green")
            axes[1].set_ylabel("Spread")

            axes[2].plot(x, signals["index_tip_x"], label="idx_x", color="tab:orange")
            axes[2].plot(x, signals["index_tip_y"], label="idx_y", color="tab:red")
            axes[2].set_ylabel("Index tip")
            axes[2].legend(loc="upper right")

            axes[3].plot(x, signals["thumb_index_dist"], color="tab:purple")
            axes[3].set_ylabel("Thumb-Index")
            axes[3].set_xlabel("Frame")

            outpath = os.path.join(outdir, f"train_{lbl}_sample{j+1}.png")
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig.savefig(outpath, dpi=150)
            plt.close(fig)
            print(f"Saved: {outpath}")


def main():
    parser = argparse.ArgumentParser(description="Visualize gesture traces vs time")
    parser.add_argument("--source", choices=["raw", "train"], default="raw")
    parser.add_argument("--data-dir", default="gesture_data")
    parser.add_argument("--out", default="gesture_data/viz")
    parser.add_argument("--n-samples", type=int, default=1, help="Samples per label (train mode)")
    parser.add_argument("--labels", nargs="*", help="Subset of labels to visualize (train mode)")
    parser.add_argument("--gestures", nargs="*", help="Subset of gesture types (raw mode)")
    args = parser.parse_args()

    if args.source == "raw":
        raw_path = os.path.join(args.data_dir, "continuous_sequences.json")
        sequences = load_raw_sequences(raw_path)
        plot_raw_sequences(sequences, outdir=args.out, gestures=args.gestures)
    else:
        train_path = os.path.join(args.data_dir, "training_data.json")
        data = load_training_data(train_path)
        plot_training_windows(data, outdir=args.out, n_samples=args.n_samples, labels=args.labels)


if __name__ == "__main__":
    main()

