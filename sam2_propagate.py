#!/usr/bin/env python3
"""
sam2_propagate.py — Batch propagation of SAM2 prompts across all videos for a speaker.

Usage:
    # single GPU
    python sam2_propagate.py --spk spk15 [--chunk 150] [--subset 200]

    # multi-GPU (distribute videos across GPUs 0,1,2,3)
    python sam2_propagate.py --spk spk15 --gpus 0,1,2,3

Session JSON is read automatically from {data_dir}/{spk}/sam_seg/session.json
as configured in sam2_gui_config.json.

Outputs per video:
    data_dir/spk/sam_seg/masks/{spk}_{video_basename}.npz   (keys=region names, each bool T×H×W)
    data_dir/spk/sam_seg/overlays/{video_basename}.mp4
"""

import argparse
import gc
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
from tqdm import tqdm

# Suppress SAM2's internal tqdm bars (forward + reverse pass)
os.environ["TQDM_DISABLE"] = "1"
from sam2.build_sam import build_sam2_video_predictor

_HERE = os.path.dirname(os.path.abspath(__file__))

# ── Path helpers ───────────────────────────────────────────────────────────────


def _load_config(CONFIG_PATH) -> dict:
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return {}


def get_video_dir(data_dir: str, video_subdir: str, spk: str) -> str:
    return os.path.join(data_dir, spk, *video_subdir.split("/"))


def get_video_files(data_dir: str, video_subdir: str, spk: str) -> list:
    vd = get_video_dir(data_dir, video_subdir, spk)
    if not os.path.isdir(vd):
        return []
    return sorted(f for f in os.listdir(vd) if f.lower().endswith(".avi"))


def get_frames_dir(frames_temp: str, spk: str, video_basename: str) -> str:
    return os.path.join(frames_temp, spk, video_basename)


def get_mask_path(data_dir: str, spk: str, video_basename: str) -> str:
    return os.path.join(data_dir, spk, "sam_seg", "masks", f"{spk}_{video_basename}.npz")


def get_overlay_path(data_dir: str, spk: str, video_basename: str) -> str:
    return os.path.join(data_dir, spk, "sam_seg", "overlays", f"{video_basename}.mp4")


# ── Frame extraction ───────────────────────────────────────────────────────────


def extract_frames(
    video_path: str, frames_dir: str, init_frame_path: str = None,
    standard_size: int = None,
) -> int:
    """Extract all frames to frames_dir as %05d.jpg using cv2 (matches notebook).

    If init_frame_path is given, copies it as 00000.jpg and numbers video
    frames from 00001.jpg onward so SAM2 always anchors on the same reference
    frame regardless of which video is being processed.

    If standard_size is given, all frames (including the init frame) are
    resized to standard_size × standard_size so that prompt points in the
    standardised coordinate space are directly usable.

    Returns the frame offset (1 if init frame was prepended, else 0).
    """
    os.makedirs(frames_dir, exist_ok=True)
    offset = 0
    if init_frame_path and os.path.exists(init_frame_path):
        init_img = cv2.imread(init_frame_path)
        if standard_size:
            init_img = cv2.resize(init_img, (standard_size, standard_size),
                                  interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(os.path.join(frames_dir, "00000.jpg"), init_img,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        offset = 1

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    idx = offset
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if standard_size:
            frame = cv2.resize(frame, (standard_size, standard_size),
                               interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(
            os.path.join(frames_dir, f"{idx:05d}.jpg"),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )
        idx += 1
    cap.release()
    return offset


# ── Overlay rendering ──────────────────────────────────────────────────────────


def _get_fps(video_path: str) -> float:
    """Get FPS via ffprobe — more reliable than cv2 for AVI files."""
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
        fps = cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS)
        return fps or 25.0


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
    fps = _get_fps(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames_in = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  [FPS] ffprobe → {fps:.3f} fps | source: {total_frames_in} frames "
          f"= {total_frames_in / fps:.1f}s")

    n_mask_frames = (
        max(m.shape[0] for m in masks_per_region.values())
        if masks_per_region else total_frames_in
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Pipe raw BGR frames directly to ffmpeg — avoids cv2.VideoWriter container
    # metadata issues that cause incorrect fps at high frame rates (e.g. 99 fps).
    ffmpeg_proc = subprocess.Popen(
        [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{w}x{h}",
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

    frame_idx = 0
    while frame_idx < n_mask_frames:
        ret, frame = cap.read()
        if not ret:
            break
        overlay = frame.copy()
        for name, masks in masks_per_region.items():
            if frame_idx >= masks.shape[0]:
                continue
            mask = masks[frame_idx]  # (H, W) bool
            if mask.shape != (h, w):
                mask = cv2.resize(mask.astype(np.uint8), (w, h),
                                  interpolation=cv2.INTER_NEAREST).astype(bool)
            bgr = _hex_to_bgr(colors[name])
            overlay[mask] = (
                (1 - alpha) * frame[mask].astype(np.float32)
                + alpha * np.array(bgr, dtype=np.float32)
            ).astype(np.uint8)
        ffmpeg_proc.stdin.write(overlay.tobytes())
        frame_idx += 1

    cap.release()
    ffmpeg_proc.stdin.close()
    stderr_bytes = ffmpeg_proc.stderr.read()
    ffmpeg_proc.wait()
    if ffmpeg_proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed:\n{stderr_bytes.decode()}"
        )
    print(f"  [FPS] wrote {frame_idx} frames → {frame_idx / fps:.1f}s at {fps:.3f} fps (intended)")

    # Probe the output file to confirm actual stored fps and duration
    _probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate,nb_frames,duration",
            "-of", "default=noprint_wrappers=1",
            output_path,
        ],
        capture_output=True, text=True,
    )
    print(f"  [FPS] output file probe:\n    " + _probe.stdout.strip().replace("\n", "\n    "))


# ── Propagation core ───────────────────────────────────────────────────────────


def propagate_video(
    predictor,
    session: dict,
    video_path: str,
    spk: str,
    chunk: int,
    data_dir: str,
    frames_temp: str,
    subset: int = None,
) -> None:
    """Propagate saved prompts through one video and save masks + overlay."""
    regions = session["regions"]
    standard_size = session.get("standard_size")
    init_frame_path = session.get("init_frame_path")
    if not init_frame_path:
        # Fallback for sessions saved before init_frame_path was added
        init_frame_path = os.path.join(data_dir, spk, "sam_seg", f"{spk}_frame.jpg")
        if os.path.exists(init_frame_path):
            print(f"  Using derived init frame path: {init_frame_path}")
        else:
            init_frame_path = None
    video_basename = os.path.splitext(os.path.basename(video_path))[0]

    # ── extract frames, prepending the saved init frame as frame 0 ──────────
    frames_dir = get_frames_dir(frames_temp, spk, video_basename)
    print(f"  Extracting frames → {frames_dir}")
    frame_offset = extract_frames(video_path, frames_dir, init_frame_path,
                                  standard_size=standard_size)
    if frame_offset:
        print("  Prepended init frame as 00000.jpg (anchor)")
    if standard_size:
        print(f"  All frames resized to {standard_size}×{standard_size}")

    frame_files = sorted(
        f for f in os.listdir(frames_dir) if f.lower().endswith(".jpg")
    )
    n_total_frames = len(frame_files)
    n_video_frames = n_total_frames - frame_offset
    if n_video_frames <= 0:
        print(f"  WARNING: no frames extracted for {video_basename}; skipping.")
        return

    if subset is not None:
        n_video_frames = min(n_video_frames, subset)
        n_total_frames = n_video_frames + frame_offset
        print(f"  [SUBSET] Limiting to first {n_video_frames} video frames")

    _src_fps = _get_fps(video_path)
    print(f"  Source FPS (ffprobe): {_src_fps:.3f} | "
          f"video frames: {n_video_frames} | "
          f"duration: {n_video_frames / _src_fps:.1f}s")

    # ── init SAM2 inference state ────────────────────────────────────────────
    print(f"  Initialising SAM2 state ({n_total_frames} frames, anchor at frame 0)…")
    inference_state = predictor.init_state(
        video_path=frames_dir,
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
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
    pbar = tqdm(total=n_video_frames, desc=f"  {video_basename}", disable=False)
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
            pbar.update(1)
            for oid, m in zip(obj_ids, masks_np):
                if oid in collected and video_fid < n_video_frames:
                    collected[oid][video_fid] = m
        start = end
    pbar.close()

    # ── fill any None frames with empty masks ────────────────────────────────
    cap_tmp = cv2.VideoCapture(video_path)
    vid_h = int(cap_tmp.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_w = int(cap_tmp.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_tmp.release()

    mask_h = standard_size if standard_size else vid_h
    mask_w = standard_size if standard_size else vid_w

    for oid in collected:
        for i in range(n_video_frames):
            if collected[oid][i] is None:
                collected[oid][i] = np.zeros((mask_h, mask_w), dtype=bool)

    # ── mask coverage diagnostics ─────────────────────────────────────────────
    for oid, frames_list in collected.items():
        name = obj_id_to_name.get(oid, f"obj_{oid}")
        filled = [m for m in frames_list if m.any()]
        if filled:
            coverage = np.mean([m.mean() for m in filled])
            print(f"  {name}: {len(filled)}/{n_video_frames} frames with mask, "
                  f"mean coverage {coverage:.3%}")
        else:
            print(f"  {name}: no mask found in any frame")

    # ── save masks as single .npz ─────────────────────────────────────────────
    mask_path = get_mask_path(data_dir, spk, video_basename)
    os.makedirs(os.path.dirname(mask_path), exist_ok=True)

    masks_per_region = {}
    colors_per_region = {}
    for oid, frames_list in collected.items():
        name = obj_id_to_name.get(oid, f"obj_{oid}")
        stack = np.stack(frames_list, axis=0)  # (T, H, W)
        masks_per_region[name] = stack
        colors_per_region[name] = obj_id_to_color.get(oid, "#ffffff")

    np.savez(mask_path, **masks_per_region)
    print(f"  Saved masks {list(masks_per_region.keys())} → {mask_path}")

    # ── write overlay video ──────────────────────────────────────────────────
    overlay_path = get_overlay_path(data_dir, spk, video_basename)
    print(f"  Writing overlay video → {overlay_path}")
    write_overlay_video(video_path, masks_per_region, colors_per_region, overlay_path)

    # ── cleanup GPU memory & temp frames ─────────────────────────────────────
    predictor.reset_state(inference_state)
    del inference_state, collected, masks_per_region
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if os.path.isdir(frames_dir):
        shutil.rmtree(frames_dir)
    print(f"  Done: {video_basename}")


# ── Multi-GPU worker ───────────────────────────────────────────────────────────


def _gpu_worker(
    gpu_id: int,
    session: dict,
    chunk: int,
    work_q: multiprocessing.Queue,
    done_q: multiprocessing.Queue,
    data_dir: str,
    video_subdir: str,
    frames_temp: str,
    checkpoint: str,
    model_cfg: str,
    subset: int = None,
) -> None:
    """Load one model instance on gpu_id, drain work_q, report results to done_q."""
    device = f"cuda:{gpu_id}"
    spk = session["speaker"]

    ckpt = os.path.join(_HERE, checkpoint)
    print(f"[GPU {gpu_id}] Loading SAM2 model on {device}…", flush=True)
    predictor = build_sam2_video_predictor(model_cfg, ckpt_path=ckpt, device=device)
    print(f"[GPU {gpu_id}] Model ready.", flush=True)

    while True:
        try:
            vid_name = work_q.get(timeout=5)
        except queue.Empty:
            break

        video_path = os.path.join(get_video_dir(data_dir, video_subdir, spk), vid_name)
        print(f"\n[GPU {gpu_id}] → {vid_name}", flush=True)
        try:
            with torch.inference_mode():
                propagate_video(predictor, session, video_path, spk, chunk=chunk,
                                data_dir=data_dir, frames_temp=frames_temp, subset=subset)
            done_q.put((gpu_id, vid_name, None))
        except Exception as e:
            torch.cuda.empty_cache()
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
        "--dataset",
        required=True,
        help="which dataset to use, used to select config.",
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
    parser.add_argument(
        "--session",
        type=int,
        default=1,
        help="Which SAM2 session to load (default: 1, corresponding to session1.json). ",
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        metavar="N",
        help="Only propagate the first N frames of each video (diagnostic mode).",
    )
    args = parser.parse_args()

    CONFIG_PATH = os.path.join('./prompting_configs', f"sam2_{args.dataset}_config.json")

    _cfg = _load_config(CONFIG_PATH)

    DATA_DIR = _cfg.get("data_dir")
    VIDEO_SUBDIR = _cfg.get("video_subdir")
    FRAMES_TEMP = _cfg.get("frames_temp_dir", "/tmp/sam2_frames")
    CHECKPOINT = _cfg.get("checkpoint", "checkpoints/sam2.1_hiera_large.pt")
    MODEL_CFG = _cfg.get("model_cfg", "configs/sam2.1/sam2.1_hiera_l.yaml")

    # ── load session ─────────────────────────────────────────────────────────
    spk = args.spk
    session_path = os.path.join(DATA_DIR, spk, "sam_seg", "sessions", f"session{args.session}.json")
    print(f"Looking for session JSON at {session_path}…")
    if not os.path.exists(session_path):
        print(f"No session found for {spk} at {session_path}")
        sys.exit(1)
    with open(session_path) as f:
        session = json.load(f)

    spk = session["speaker"]
    video_files = get_video_files(DATA_DIR, VIDEO_SUBDIR, spk)
    if not video_files:
        print(f"No .avi files found for speaker {spk} in {get_video_dir(DATA_DIR, VIDEO_SUBDIR, spk)}")
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
                args=(gpu_id, session, args.chunk, work_q, done_q,
                      DATA_DIR, VIDEO_SUBDIR, FRAMES_TEMP, CHECKPOINT, MODEL_CFG,
                      args.subset),
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

        for vid_name in tqdm(video_files):
            if not "picture_description2" in vid_name:
                continue
            video_path = os.path.join(get_video_dir(DATA_DIR, VIDEO_SUBDIR, spk), vid_name)
            print(f"\n[{vid_name}]")
            try:
                with torch.inference_mode():
                    propagate_video(predictor, session, video_path, spk, chunk=args.chunk,
                                    data_dir=DATA_DIR, frames_temp=FRAMES_TEMP, subset=args.subset)
            except Exception as e:
                print(f"  ERROR: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

    print("\nAll done.")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
