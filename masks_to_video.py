#!/usr/bin/env python3
"""
masks_to_video.py — Render mask-only MP4s (solid green on black) for a sample of videos.

Usage:
    python masks_to_video.py --spk spk15 [--session 1] [--n 10] [--color 0,255,0]

Reads .npz mask files from {data_dir}/{spk}/sam_seg/masks/ and writes
mask-only videos to {data_dir}/{spk}/sam_seg/masks_vids/.
"""

import argparse
import json
import os
import random
import subprocess
import sys

import cv2
import numpy as np

# ── Config ─────────────────────────────────────────────────────────────────────

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_config(config_path: str) -> dict:
    if os.path.exists(config_path):
        with open(config_path) as f:
            return json.load(f)
    return {}


def _get_fps(video_path: str) -> float:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate",
             "-of", "default=noprint_wrappers=1:nokey=1",
             video_path],
            capture_output=True, text=True, check=True,
        )
        num, den = result.stdout.strip().split("/")
        return float(num) / float(den)
    except Exception:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return fps or 25.0


def render_mask_video(npz_path: str, video_path: str, output_path: str,
                      color_bgr: tuple) -> None:
    """Render all masks as a single solid color on a black background."""
    data = np.load(npz_path)
    region_names = list(data.keys())
    if not region_names:
        print(f"  No regions in {npz_path}, skipping.")
        return

    # Stack all regions into a combined mask (logical OR)
    first = data[region_names[0]]  # (T, H, W)
    combined = first.copy()
    for name in region_names[1:]:
        combined = combined | data[name]

    T, H, W = combined.shape

    # Get fps from the source video if available, else default
    if os.path.exists(video_path):
        fps = _get_fps(video_path)
        # Use source video dimensions for output
        cap = cv2.VideoCapture(video_path)
        vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
    else:
        fps = 25.0
        vid_w, vid_h = W, H

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    ffmpeg_proc = subprocess.Popen(
        [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{vid_w}x{vid_h}",
            "-r", str(fps),
            "-i", "pipe:0",
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "23",
            output_path,
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    for t in range(T):
        mask = combined[t]  # (H, W) bool
        if mask.shape != (vid_h, vid_w):
            mask = cv2.resize(mask.astype(np.uint8), (vid_w, vid_h),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
        frame = np.zeros((vid_h, vid_w, 3), dtype=np.uint8)
        frame[mask] = color_bgr
        ffmpeg_proc.stdin.write(frame.tobytes())

    ffmpeg_proc.stdin.close()
    stderr_bytes = ffmpeg_proc.stderr.read()
    ffmpeg_proc.wait()
    if ffmpeg_proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{stderr_bytes.decode()}")

    print(f"  Wrote {T} frames → {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Render mask-only videos (solid color on black) for a sample of files."
    )
    parser.add_argument("--spk", required=True, help="Speaker ID (e.g. spk15)")
    parser.add_argument("--dataset", required=True,
                        help="Which dataset to use, used to select config.")
    parser.add_argument("--session", type=int, default=1, help="Session number (default: 1)")
    parser.add_argument("--n", type=int, default=10, help="Number of files to sample (default: 10)")
    parser.add_argument("--color", default="0,255,0",
                        help="BGR color as R,G,B (default: 0,255,0 = green)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    config_path = os.path.join(_HERE, 'prompting_configs', f"sam2_{args.dataset}_config.json")
    _cfg = _load_config(config_path)
    DATA_DIR = _cfg.get("data_dir")
    VIDEO_SUBDIR = _cfg.get("video_subdir")

    spk = args.spk
    r, g, b = (int(c) for c in args.color.split(","))
    color_bgr = (b, g, r)

    masks_dir = os.path.join(DATA_DIR, spk, "sam_seg", "masks")
    video_dir = os.path.join(DATA_DIR, spk, *VIDEO_SUBDIR.split("/"))
    output_dir = os.path.join(DATA_DIR, spk, "sam_seg", "masks_vids")

    if not os.path.isdir(masks_dir):
        print(f"Masks directory not found: {masks_dir}")
        sys.exit(1)

    npz_files = sorted(f for f in os.listdir(masks_dir) if f.endswith(".npz"))
    if not npz_files:
        print(f"No .npz mask files found in {masks_dir}")
        sys.exit(1)

    # Sample
    random.seed(args.seed)
    n = min(args.n, len(npz_files))
    sample = random.sample(npz_files, n)
    print(f"Sampling {n}/{len(npz_files)} mask files for {spk}")

    for npz_name in sample:
        print(f"\n[{npz_name}]")
        npz_path = os.path.join(masks_dir, npz_name)

        # Derive video name: mask file is {spk}_{video_basename}.npz
        video_basename = npz_name.replace(f"{spk}_", "", 1).replace(".npz", "")
        video_path = os.path.join(video_dir, video_basename + ".avi")
        output_path = os.path.join(output_dir, video_basename + ".mp4")

        try:
            render_mask_video(npz_path, video_path, output_path, color_bgr)
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\nDone. Videos saved to {output_dir}")


if __name__ == "__main__":
    main()
