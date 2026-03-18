#!/usr/bin/env python3
"""
sam2_propagate.py — Batch propagation of SAM2 prompts across all videos for a speaker.

Usage:
    python sam2_propagate.py --session path/to/session.json [--device cuda:0] [--chunk 150]

For multi-GPU runs, launch one process per GPU:
    python sam2_propagate.py --session session.json --device cuda:0 &
    python sam2_propagate.py --session session.json --device cuda:1 &

Outputs per video:
    data_dir/spk/sam2/masks/{video_basename}/{region_name}.npy   (bool T×H×W)
    data_dir/spk/sam2/overlays/{video_basename}.mp4
"""

import argparse
import json
import os
import subprocess
import sys

import cv2
import numpy as np

# ── Config ─────────────────────────────────────────────────────────────────────

_HERE = os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH = os.path.join(_HERE, 'sam2_gui_config.json')


def _load_config() -> dict:
    if os.path.exists(_CONFIG_PATH):
        with open(_CONFIG_PATH) as f:
            return json.load(f)
    return {}


_cfg = _load_config()

DATA_DIR     = _cfg.get('data_dir',        '/data')
VIDEO_SUBDIR = _cfg.get('video_subdir',    'video/video')
FRAMES_TEMP  = _cfg.get('frames_temp_dir', '/tmp/sam2_frames')
CHECKPOINT   = _cfg.get('checkpoint',      'checkpoints/sam2.1_hiera_large.pt')
MODEL_CFG    = _cfg.get('model_cfg',       'configs/sam2.1/sam2.1_hiera_l.yaml')

# ── Path helpers ───────────────────────────────────────────────────────────────

def get_video_dir(spk: str) -> str:
    return os.path.join(DATA_DIR, spk, *VIDEO_SUBDIR.split('/'))


def get_video_files(spk: str) -> list:
    vd = get_video_dir(spk)
    if not os.path.isdir(vd):
        return []
    return sorted(f for f in os.listdir(vd) if f.lower().endswith('.avi'))


def get_frames_dir(spk: str, video_basename: str) -> str:
    return os.path.join(FRAMES_TEMP, spk, video_basename)


def get_mask_dir(spk: str, video_basename: str) -> str:
    return os.path.join(DATA_DIR, spk, 'sam2', 'masks', video_basename)


def get_overlay_path(spk: str, video_basename: str) -> str:
    return os.path.join(DATA_DIR, spk, 'sam2', 'overlays',
                        f'{video_basename}.mp4')

# ── Frame extraction ───────────────────────────────────────────────────────────

def extract_frames(video_path: str, frames_dir: str) -> None:
    """Extract all frames to frames_dir as %05d.jpg (ffmpeg, q:v 2)."""
    os.makedirs(frames_dir, exist_ok=True)
    subprocess.run(
        ['ffmpeg', '-y', '-i', video_path,
         '-q:v', '2', '-start_number', '0',
         os.path.join(frames_dir, '%05d.jpg')],
        check=True, capture_output=True)


# ── Overlay rendering ──────────────────────────────────────────────────────────

def _hex_to_bgr(hex_color: str) -> tuple:
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def write_overlay_video(
    video_path:   str,
    masks_per_region: dict,   # region_name → (T, H, W) bool array
    colors:       dict,       # region_name → hex color string
    output_path:  str,
    alpha:        float = 0.45,
) -> None:
    """Write an MP4 with coloured mask overlays burned in."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open {video_path}')
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps, (w, h))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        overlay = frame.copy()
        for name, masks in masks_per_region.items():
            if frame_idx >= masks.shape[0]:
                continue
            mask = masks[frame_idx]   # (H, W) bool
            bgr  = _hex_to_bgr(colors[name])
            overlay[mask] = (
                (1 - alpha) * frame[mask].astype(np.float32) +
                alpha * np.array(bgr, dtype=np.float32)
            ).astype(np.uint8)
        cv2.addWeighted(overlay, 1.0, frame, 0.0, 0, overlay)
        writer.write(overlay)
        frame_idx += 1

    cap.release()
    writer.release()


# ── Propagation core ───────────────────────────────────────────────────────────

def propagate_video(
    predictor,
    session:    dict,
    video_path: str,
    spk:        str,
    chunk:      int,
    device:     str,
) -> None:
    """Propagate saved prompts through one video and save masks + overlay."""
    regions         = session['regions']
    init_frame_idx  = session.get('initial_frame_idx', 0)
    video_basename  = os.path.splitext(os.path.basename(video_path))[0]

    # ── extract frames ───────────────────────────────────────────────────────
    frames_dir = get_frames_dir(spk, video_basename)
    print(f'  Extracting frames → {frames_dir}')
    extract_frames(video_path, frames_dir)

    frame_files = sorted(f for f in os.listdir(frames_dir)
                         if f.lower().endswith('.jpg'))
    n_frames = len(frame_files)
    if n_frames == 0:
        print(f'  WARNING: no frames extracted for {video_basename}; skipping.')
        return

    # ── init SAM2 inference state ────────────────────────────────────────────
    print(f'  Initialising SAM2 state ({n_frames} frames)…')
    inference_state = predictor.init_state(
        video_path=frames_dir,
        offload_video_to_cpu=True,
        offload_state_to_cpu=False,
    )
    predictor.reset_state(inference_state)

    # ── add prompts on the saved initial frame ───────────────────────────────
    for region in regions:
        if not region['points']:
            continue
        predictor.add_new_points_or_box(
            inference_state,
            frame_idx=init_frame_idx,
            obj_id=region['obj_id'],
            points=region['points'],
            labels=region['labels'],
            clear_old_points=True,
        )

    # ── propagate in chunks ──────────────────────────────────────────────────
    # Collect per-object mask lists: obj_id → list of (H,W) bool arrays
    obj_id_to_name  = {r['obj_id']: r['name']  for r in regions}
    obj_id_to_color = {r['obj_id']: r['color'] for r in regions}

    # Pre-allocate lists indexed by frame
    collected: dict = {r['obj_id']: [None] * n_frames for r in regions}

    print(f'  Propagating (chunk={chunk})…')
    start = init_frame_idx
    while start < n_frames:
        end = min(start + chunk, n_frames)
        for frame_idx, obj_ids, masks in predictor.propagate_in_video(
            inference_state,
            start_frame_idx=start,
            max_frame_num_to_track=end - start,
        ):
            # masks: tensor (N_objs, 1, H, W)
            import torch
            masks_np = (masks.squeeze(1).cpu().numpy() > 0)  # (N_objs, H, W)
            for oid, m in zip(obj_ids, masks_np):
                if oid in collected and frame_idx < n_frames:
                    collected[oid][frame_idx] = m
        start = end

    # Also propagate backward from init frame if it is not frame 0
    if init_frame_idx > 0:
        print(f'  Propagating backward from frame {init_frame_idx}…')
        start = init_frame_idx
        while start > 0:
            begin = max(start - chunk, 0)
            for frame_idx, obj_ids, masks in predictor.propagate_in_video(
                inference_state,
                start_frame_idx=start,
                max_frame_num_to_track=start - begin,
                reverse=True,
            ):
                import torch
                masks_np = (masks.squeeze(1).cpu().numpy() > 0)
                for oid, m in zip(obj_ids, masks_np):
                    if oid in collected and 0 <= frame_idx < n_frames:
                        collected[oid][frame_idx] = m
            start = begin

    # ── fill any None frames with empty masks ────────────────────────────────
    cap_tmp = cv2.VideoCapture(video_path)
    vid_h   = int(cap_tmp.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_w   = int(cap_tmp.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_tmp.release()
    empty   = np.zeros((vid_h, vid_w), dtype=bool)

    for oid in collected:
        for i in range(n_frames):
            if collected[oid][i] is None:
                collected[oid][i] = empty

    # ── save .npy masks ──────────────────────────────────────────────────────
    mask_dir = get_mask_dir(spk, video_basename)
    os.makedirs(mask_dir, exist_ok=True)

    masks_per_region = {}
    colors_per_region = {}
    for oid, frames_list in collected.items():
        name  = obj_id_to_name.get(oid, f'obj_{oid}')
        stack = np.stack(frames_list, axis=0)   # (T, H, W)
        npy_path = os.path.join(mask_dir, f'{name}.npy')
        np.save(npy_path, stack)
        print(f'  Saved mask {stack.shape} → {npy_path}')
        masks_per_region[name]  = stack
        colors_per_region[name] = obj_id_to_color.get(oid, '#ffffff')

    # ── write overlay video ──────────────────────────────────────────────────
    overlay_path = get_overlay_path(spk, video_basename)
    print(f'  Writing overlay video → {overlay_path}')
    write_overlay_video(video_path, masks_per_region, colors_per_region,
                        overlay_path)
    print(f'  Done: {video_basename}')


# ── CLI entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Batch-propagate SAM2 prompts across all videos for a speaker.')
    parser.add_argument('--session', required=True,
                        help='Path to the session JSON saved by sam2_gui.py')
    parser.add_argument('--device',  default=None,
                        help='PyTorch device (e.g. cuda, cuda:0, cpu). '
                             'Overrides config.')
    parser.add_argument('--chunk',   type=int, default=150,
                        help='Frames per propagation chunk (default: 150)')
    args = parser.parse_args()

    # ── load session ─────────────────────────────────────────────────────────
    with open(args.session) as f:
        session = json.load(f)

    spk     = session['speaker']
    device  = args.device or _cfg.get('device', 'cuda')

    # ── build predictor ───────────────────────────────────────────────────────
    from sam2.build_sam import build_sam2_video_predictor
    ckpt = os.path.join(_HERE, CHECKPOINT)
    print(f'Loading SAM2 model on {device}…')
    predictor = build_sam2_video_predictor(MODEL_CFG, ckpt_path=ckpt,
                                           device=device)
    print('Model loaded.')

    # ── enumerate videos ──────────────────────────────────────────────────────
    video_files = get_video_files(spk)
    if not video_files:
        print(f'No .avi files found for speaker {spk} in {get_video_dir(spk)}')
        sys.exit(1)

    print(f'Speaker: {spk}  ({len(video_files)} videos)')

    for vid_name in video_files:
        video_path = os.path.join(get_video_dir(spk), vid_name)
        print(f'\n[{vid_name}]')
        try:
            propagate_video(predictor, session, video_path, spk,
                            chunk=args.chunk, device=device)
        except Exception as e:
            print(f'  ERROR: {e}')
            continue

    print('\nAll done.')


if __name__ == '__main__':
    main()
