#!/usr/bin/env python3
"""
calc_est_time.py — Estimate SAM2 segmentation time per speaker.

Counts total video frames per speaker under the configured data_dir/video_subdir
and estimates processing time at ~3.75 min per 1000 frames on 1 GPU.

Usage:
    python calc_est_time.py --dataset prompt
    python calc_est_time.py --dataset gloss --gpus 4
"""

import argparse
import json
import os
import sys

import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))

MINUTES_PER_1000_FRAMES = 3.75


def _load_config(config_path: str) -> dict:
    if os.path.exists(config_path):
        with open(config_path) as f:
            return json.load(f)
    return {}


def count_frames(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def main():
    parser = argparse.ArgumentParser(
        description="Estimate SAM2 segmentation time per speaker."
    )
    parser.add_argument("--dataset", required=True,
                        help="Which dataset to use, used to select config.")
    parser.add_argument("--gpus", type=int, default=1,
                        help="Number of GPUs (divides total time). Default: 1")
    args = parser.parse_args()

    config_path = os.path.join(_HERE, "prompting_configs", f"sam2_{args.dataset}_config.json")
    _cfg = _load_config(config_path)
    data_dir = _cfg.get("data_dir")
    video_subdir = _cfg.get("video_subdir")

    if not data_dir or not video_subdir:
        print(f"Missing data_dir or video_subdir in {config_path}")
        sys.exit(1)

    if not os.path.isdir(data_dir):
        print(f"Data directory not found: {data_dir}")
        sys.exit(1)

    speakers = sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith(".")
    )

    total_frames_all = 0
    results = []
    exc_files = ['cop_cop', 'top_cop', 'top_top', 'ptk1', 'ptk2', 'ptk3'] 

    for spk in speakers:
        video_dir = os.path.join(data_dir, spk, *video_subdir.split("/"))
        if not os.path.isdir(video_dir):
            continue

        videos = [f for f in os.listdir(video_dir) if f.lower().endswith(".avi") and not any(exc in f for exc in exc_files)]
        if not videos:
            continue

        spk_frames = 0
        for v in videos:
            spk_frames += count_frames(os.path.join(video_dir, v))

        est_minutes = (spk_frames / 1000.0) * MINUTES_PER_1000_FRAMES / args.gpus
        hours = int(est_minutes // 60)
        mins = est_minutes % 60
        results.append((spk, len(videos), spk_frames, hours, mins))
        total_frames_all += spk_frames

    if not results:
        print("No speakers with videos found.")
        sys.exit(1)

    # Print results
    print(f"{'Speaker':<12} {'Videos':>6} {'Frames':>10} {'Est. Time':>12}")
    print("-" * 44)
    for spk, n_vids, n_frames, h, m in results:
        print(f"{spk:<12} {n_vids:>6} {n_frames:>10} {h:>5}h {m:>5.1f}m")

    total_minutes = (total_frames_all / 1000.0) * MINUTES_PER_1000_FRAMES / args.gpus
    total_h = int(total_minutes // 60)
    total_m = total_minutes % 60
    print("-" * 44)
    print(f"{'TOTAL':<12} {sum(r[1] for r in results):>6} {total_frames_all:>10} {total_h:>5}h {total_m:>5.1f}m")
    print(f"\n({args.gpus} GPU(s), {MINUTES_PER_1000_FRAMES} min/1000 frames)")


if __name__ == "__main__":
    main()