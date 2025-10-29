#!/usr/bin/env python3
"""
Regenerate training_data.json with center-frame labeling from continuous_sequences.json.

This avoids re-recording: it reads your existing raw sequences and writes
center-labeled training windows, consistent with the updated recorder logic.

Usage:
  python tools/generate_centered_training.py \
      --input gesture_data/continuous_sequences.json \
      --output gesture_data/training_data.json \
      --window-size 20 --stride 2
"""

import argparse
import json
import os
from collections import Counter


def center_label_training(raw_sequences, window_size=20, stride=2):
    training = {"sequences": [], "labels": []}

    for seq_obj in raw_sequences:
        phases = seq_obj.get("phases", [])
        gesture_type = seq_obj.get("gesture_type", "neutral")
        n = len(phases)
        if n < window_size:
            continue

        for i in range(0, n - window_size, stride):
            window_phases = phases[i : i + window_size]
            landmarks = [p["landmarks"] for p in window_phases]

            center_idx = i + (window_size // 2)
            center_phase = phases[center_idx]["phase"] if 0 <= center_idx < n else "neutral"

            label = "neutral"
            if gesture_type != "neutral":
                if center_phase == "active_gesture":
                    label = gesture_type
                elif center_phase == "transitioning_to_neutral":
                    label = f"{gesture_type}_return"
                elif center_phase == "transitioning_to_gesture":
                    label = f"{gesture_type}_start"

            if landmarks:
                training["sequences"].append(landmarks)
                training["labels"].append(label)

    return training


def main():
    ap = argparse.ArgumentParser(description="Regenerate centered training windows")
    ap.add_argument("--input", default="gesture_data/continuous_sequences.json")
    ap.add_argument("--output", default="gesture_data/training_data.json")
    ap.add_argument("--window-size", type=int, default=20)
    ap.add_argument("--stride", type=int, default=2)
    args = ap.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input raw sequences not found: {args.input}")

    with open(args.input, "r") as f:
        raw = json.load(f)
        if not isinstance(raw, list):
            raise ValueError("Expected a list in continuous_sequences.json")

    training = center_label_training(raw, window_size=args.window_size, stride=args.stride)

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(training, f)

    # Report
    labels = training.get("labels", [])
    counts = Counter(labels)
    total = len(training.get("sequences", []))
    print(f"✓ Regenerated centered training data -> {args.output}")
    print(f"  Windows: {total}")
    print(f"  Label distribution: {counts}")


if __name__ == "__main__":
    main()

