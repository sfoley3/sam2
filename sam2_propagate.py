#!/usr/bin/env python3
"""
sam2_propagate.py — Batch propagation of SAM2 prompts across all videos for a speaker.

Usage:
    # single GPU
    python sam2_propagate.py --spk spk15 [--device cuda:0] [--chunk 150]

    # multi-GPU (distribute videos across GPUs 0,1,2,3)
    python sam2_propagate.py --spk spk15 --gpus 0,1,2,3

Session JSON is read automatically from {data_dir}/{spk}/sam_seg/session.json
as configured in sam2_gui_config.json.

Outputs per video:
    data_dir/spk/sam2/masks/{video_basename}/{region_name}.npy   (bool T×H×W)
    data_dir/spk/sam2/overlays/{video_basename}.mp4
"""

import argparse
import json
import multiprocessing
import os
import queue
import shutil
import subprocess
import sys

import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor

# ── Config ─────────────────────────────────────────────────────────────────────

_HERE = os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH = os.path.join(_HERE, "sam2_gui_config.json")


def _load_config() -> dict:
    if os.path.exists(_CONFIG_PATH):
        with open(_CONFIG_PATH) as f:
            return json.load(f)
    return {}


_cfg = _load_config()

DATA_DIR = _cfg.get("data_dir", "/data")
VIDEO_SUBDIR = _cfg.get("video_subdir", "video/video")
FRAMES_TEMP = _cfg.get("frames_temp_dir", "/tmp/sam2_frames")
CHECKPOINT = _cfg.get("checkpoint", "checkpoints/sam2.1_hiera_large.pt")
MODEL_CFG = _cfg.get("model_cfg", "configs/sam2.1/sam2.1_hiera_l.yaml")

# ── Path helpers ───────────────────────────────────────────────────────────────


def get_video_dir(spk: str) -> str:
    return os.path.join(DATA_DIR, spk, *VIDEO_SUBDIR.split("/"))


def get_video_files(spk: str) -> list:
    vd = get_video_dir(spk)
    if not os.path.isdir(vd):
        return []
    return sorted(f for f in os.listdir(vd) if f.lower().endswith(".avi"))


def get_frames_dir(spk: str, video_basename: str) -> str:
    return os.path.join(FRAMES_TEMP, spk, video_basename)


def get_mask_dir(spk: str, video_basename: str) -> str:
    return os.path.join(DATA_DIR, spk, "sam2", "masks", video_basename)


def get_overlay_path(spk: str, video_basename: str) -> str:
    return os.path.join(DATA_DIR, spk, "sam2", "overlays", f"{video_basename}.mp4")


# ── Frame extraction ───────────────────────────────────────────────────────────


def extract_frames(
    video_path: str, frames_dir: str, init_frame_path: str = None
) -> int:
    """Extract all frames to frames_dir as %05d.jpg using cv2 (matches notebook).

    If init_frame_path is given, copies it as 00000.jpg and numbers video
    frames from 00001.jpg onward so SAM2 always anchors on the same reference
    frame regardless of which video is being processed.

    Returns the frame offset (1 if init frame was prepended, else 0).
    """
    os.makedirs(frames_dir, exist_ok=True)
    offset = 0
    if init_frame_path and os.path.exists(init_frame_path):
        shutil.copy2(init_frame_path, os.path.join(frames_dir, "00000.jpg"))
        offset = 1

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    idx = offset
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(
            os.path.join(frames_dir, f"{idx:05d}.jpg"),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )
        idx += 1
    cap.release()
    return offset


# ── Overlay rendering ──────────────────────────────────────────────────────────


def _hex_to_bgr(hex_color: str) -> tuple:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def write_overlay_video(
    video_path: str,
    masks_per_region: dict,  # region_name → (T, H, W) bool array
    colors: dict,  # region_name → hex color string
    output_path: str,
    alpha: float = 0.7,
) -> None:
    """Write an MP4 with coloured mask overlays burned in."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        overlay = frame.copy()
        for name, masks in masks_per_region.items():
            if frame_idx >= masks.shape[0]:
                continue
            mask = masks[frame_idx]  # (H, W) bool
            bgr = _hex_to_bgr(colors[name])
            overlay[mask] = (
                (1 - alpha) * frame[mask].astype(np.float32)
                + alpha * np.array(bgr, dtype=np.float32)
            ).astype(np.uint8)
        writer.write(overlay)
        frame_idx += 1

    cap.release()
    writer.release()

    # Re-encode to libx264/yuv420p so timing metadata is correct for playback
    raw_path = output_path + ".raw.mp4"
    os.rename(output_path, raw_path)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            raw_path,
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "23",
            output_path,
        ],
        check=True,
        capture_output=True,
    )
    os.remove(raw_path)


# ── Propagation core ───────────────────────────────────────────────────────────


def propagate_video(
    predictor,
    session: dict,
    video_path: str,
    spk: str,
    chunk: int,
) -> None:
    """Propagate saved prompts through one video and save masks + overlay."""
    regions = session["regions"]
    init_frame_path = session.get("init_frame_path")
    video_basename = os.path.splitext(os.path.basename(video_path))[0]

    # ── extract frames, prepending the saved init frame as frame 0 ──────────
    frames_dir = get_frames_dir(spk, video_basename)
    print(f"  Extracting frames → {frames_dir}")
    frame_offset = extract_frames(video_path, frames_dir, init_frame_path)
    if frame_offset:
        print("  Prepended init frame as 00000.jpg (anchor)")

    frame_files = sorted(
        f for f in os.listdir(frames_dir) if f.lower().endswith(".jpg")
    )
    n_total_frames = len(frame_files)
    n_video_frames = n_total_frames - frame_offset
    if n_video_frames <= 0:
        print(f"  WARNING: no frames extracted for {video_basename}; skipping.")
        return

    # ── init SAM2 inference state ────────────────────────────────────────────
    print(f"  Initialising SAM2 state ({n_total_frames} frames, anchor at frame 0)…")
    inference_state = predictor.init_state(
        video_path=frames_dir,
        offload_video_to_cpu=True,
        offload_state_to_cpu=False,
    )
    predictor.reset_state(inference_state)

    # ── add prompts at frame 0 (always the saved init frame) ────────────────
    for region in regions:
        if not region["points"]:
            continue
        predictor.add_new_points_or_box(
            inference_state,
            frame_idx=0,
            obj_id=region["obj_id"],
            points=np.array(region["points"], dtype=np.float32),
            labels=np.array(region["labels"], dtype=np.int32),
            clear_old_points=True,
        )

    # ── propagate in chunks ──────────────────────────────────────────────────
    obj_id_to_name = {r["obj_id"]: r["name"] for r in regions}
    obj_id_to_color = {r["obj_id"]: r["color"] for r in regions}

    # Pre-allocate lists indexed by video frame (excluding the synthetic anchor)
    collected: dict = {r["obj_id"]: [None] * n_video_frames for r in regions}

    print(f"  Propagating (chunk={chunk})…")
    start = 0
    while start < n_total_frames:
        end = min(start + chunk, n_total_frames)
        for frame_idx, obj_ids, masks in predictor.propagate_in_video(
            inference_state,
            start_frame_idx=start,
            max_frame_num_to_track=end - start,
        ):
            masks_np = masks.squeeze(1).cpu().numpy() > 0  # (N_objs, H, W)
            video_fid = frame_idx - frame_offset
            if video_fid < 0:
                continue  # skip the synthetic anchor frame
            for oid, m in zip(obj_ids, masks_np):
                if oid in collected and video_fid < n_video_frames:
                    collected[oid][video_fid] = m
        start = end

    # ── fill any None frames with empty masks ────────────────────────────────
    cap_tmp = cv2.VideoCapture(video_path)
    vid_h = int(cap_tmp.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_w = int(cap_tmp.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_tmp.release()
    empty = np.zeros((vid_h, vid_w), dtype=bool)

    for oid in collected:
        for i in range(n_video_frames):
            if collected[oid][i] is None:
                collected[oid][i] = empty

    # ── save .npy masks ──────────────────────────────────────────────────────
    mask_dir = get_mask_dir(spk, video_basename)
    os.makedirs(mask_dir, exist_ok=True)

    masks_per_region = {}
    colors_per_region = {}
    for oid, frames_list in collected.items():
        name = obj_id_to_name.get(oid, f"obj_{oid}")
        stack = np.stack(frames_list, axis=0)  # (T, H, W)
        npy_path = os.path.join(mask_dir, f"{name}.npy")
        np.save(npy_path, stack)
        print(f"  Saved mask {stack.shape} → {npy_path}")
        masks_per_region[name] = stack
        colors_per_region[name] = obj_id_to_color.get(oid, "#ffffff")

    # ── write overlay video ──────────────────────────────────────────────────
    overlay_path = get_overlay_path(spk, video_basename)
    print(f"  Writing overlay video → {overlay_path}")
    write_overlay_video(video_path, masks_per_region, colors_per_region, overlay_path)
    print(f"  Done: {video_basename}")


# ── Multi-GPU worker ───────────────────────────────────────────────────────────


def _gpu_worker(
    gpu_id: int,
    session: dict,
    chunk: int,
    work_q: multiprocessing.Queue,
    done_q: multiprocessing.Queue,
) -> None:
    """Load one model instance on gpu_id, drain work_q, report results to done_q."""
    device = f"cuda:{gpu_id}"
    spk = session["speaker"]

    ckpt = os.path.join(_HERE, CHECKPOINT)
    print(f"[GPU {gpu_id}] Loading SAM2 model on {device}…", flush=True)
    predictor = build_sam2_video_predictor(MODEL_CFG, ckpt_path=ckpt, device=device)
    print(f"[GPU {gpu_id}] Model ready.", flush=True)

    while True:
        try:
            vid_name = work_q.get(timeout=5)
        except queue.Empty:
            break

        video_path = os.path.join(get_video_dir(spk), vid_name)
        print(f"\n[GPU {gpu_id}] → {vid_name}", flush=True)
        try:
            propagate_video(predictor, session, video_path, spk, chunk=chunk)
            done_q.put((gpu_id, vid_name, None))
        except Exception as e:
            done_q.put((gpu_id, vid_name, str(e)))

    print(f"[GPU {gpu_id}] Done.", flush=True)


# ── CLI entry point ────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Batch-propagate SAM2 prompts across all videos for a speaker."
    )
    parser.add_argument(
        "--spk",
        required=True,
        help="Speaker ID (e.g. spk15). Session JSON is read from "
        "{data_dir}/{spk}/sam_seg/session.json as set in config.",
    )
    parser.add_argument(
        "--gpus",
        default=None,
        help="Comma-separated GPU indices for multi-GPU mode "
        "(e.g. 0,1,2,3). Each GPU runs its own model instance "
        "and pulls videos from a shared queue.",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=150,
        help="Frames per propagation chunk (default: 150)",
    )
    args = parser.parse_args()

    # ── load session ─────────────────────────────────────────────────────────
    spk = args.spk
    session_path = os.path.join(DATA_DIR, spk, "sam_seg", "session.json")
    if not os.path.exists(session_path):
        print(f"No session found for {spk} at {session_path}")
        sys.exit(1)
    with open(session_path) as f:
        session = json.load(f)

    spk = session["speaker"]
    video_files = get_video_files(spk)
    if not video_files:
        print(f"No .avi files found for speaker {spk} in {get_video_dir(spk)}")
        sys.exit(1)

    print(f"Speaker: {spk}  ({len(video_files)} videos)")

    # ── multi-GPU path ────────────────────────────────────────────────────────
    if args.gpus:
        gpu_ids = [int(g.strip()) for g in args.gpus.split(",")]
        print(f"Multi-GPU mode: {gpu_ids}")

        work_q = multiprocessing.Queue()
        done_q = multiprocessing.Queue()

        for vid_name in video_files:
            work_q.put(vid_name)

        workers = []
        for gpu_id in gpu_ids:
            p = multiprocessing.Process(
                target=_gpu_worker,
                args=(gpu_id, session, args.chunk, work_q, done_q),
                daemon=True,
            )
            p.start()
            workers.append(p)

        for _ in video_files:
            gpu_id, vid_name, err = done_q.get()
            if err:
                print(f"  [GPU {gpu_id}] ERROR on {vid_name}: {err}")
            else:
                print(f"  [GPU {gpu_id}] Finished {vid_name}")

        for p in workers:
            p.join()

    # ── single-GPU path ───────────────────────────────────────────────────────
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        ckpt = os.path.join(_HERE, CHECKPOINT)
        print(f"Loading SAM2 model on {device}…")
        predictor = build_sam2_video_predictor(MODEL_CFG, ckpt_path=ckpt, device=device)
        print("Model loaded.")

        for vid_name in video_files:
            if "picture_description1" not in vid_name:
                continue
            video_path = os.path.join(get_video_dir(spk), vid_name)
            print(f"\n[{vid_name}]")
            try:
                propagate_video(predictor, session, video_path, spk, chunk=args.chunk)
            except Exception as e:
                print(f"  ERROR: {e}")
                continue

    print("\nAll done.")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
