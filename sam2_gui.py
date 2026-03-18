#!/usr/bin/env python3
"""
sam2_gui.py — Interactive SAM2 vocal tract segmentation GUI.

Workflow:
  1. Select speaker and video from dropdowns.
  2. Scrub to the desired initial frame; click "Set Init Frame".
     → ffmpeg extracts all frames; SAM2 initialises its inference state.
  3. Add regions (tongue_tip, lips, …); click on the canvas to place
     positive / negative prompt points → live mask overlay appears.
  4. Save the session (JSON).  Run sam2_propagate.py to batch-propagate
     the saved prompts across all videos for that speaker.
"""

import os
import json
import threading
import subprocess
from dataclasses import dataclass, field
from typing import Optional

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog

import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import torch

# ── Config ─────────────────────────────────────────────────────────────────────

_HERE = os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH = os.path.join(_HERE, 'sam2_gui_config.json')


def _load_config() -> dict:
    if os.path.exists(_CONFIG_PATH):
        with open(_CONFIG_PATH) as f:
            return json.load(f)
    return {}


_cfg = _load_config()

DATA_DIR     = _cfg.get('data_dir',       '/data')
VIDEO_SUBDIR = _cfg.get('video_subdir',   'video/video')
FRAMES_TEMP  = _cfg.get('frames_temp_dir', '/tmp/sam2_frames')
CHECKPOINT   = _cfg.get('checkpoint',     'checkpoints/sam2.1_hiera_large.pt')
MODEL_CFG    = _cfg.get('model_cfg',      'configs/sam2.1/sam2.1_hiera_l.yaml')
DEVICE       = _cfg.get('device',         'cuda' if torch.cuda.is_available() else 'cpu')

# Cycling palette for region colours
_PALETTE = [
    '#3cb44b', '#e6194b', '#4363d8', '#f58231', '#911eb4',
    '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990',
]

# ── Data models ────────────────────────────────────────────────────────────────

@dataclass
class SegRegion:
    name:   str
    obj_id: int
    points: list = field(default_factory=list)   # [[x, y], …]
    labels: list = field(default_factory=list)   # 1=positive, 0=negative
    color:  str  = '#3cb44b'

    def to_dict(self) -> dict:
        return {
            'name':   self.name,
            'obj_id': self.obj_id,
            'points': self.points,
            'labels': self.labels,
            'color':  self.color,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'SegRegion':
        return cls(
            name=d['name'],
            obj_id=int(d['obj_id']),
            points=list(d.get('points', [])),
            labels=list(d.get('labels', [])),
            color=d.get('color', '#3cb44b'),
        )

# ── Path helpers ───────────────────────────────────────────────────────────────

def get_speaker_dirs() -> list:
    if not os.path.isdir(DATA_DIR):
        return []
    return sorted(
        d for d in os.listdir(DATA_DIR)
        if d.startswith('spk') and os.path.isdir(os.path.join(DATA_DIR, d))
    )


def get_video_files(spk: str) -> list:
    vd = os.path.join(DATA_DIR, spk, *VIDEO_SUBDIR.split('/'))
    if not os.path.isdir(vd):
        return []
    return sorted(f for f in os.listdir(vd) if f.lower().endswith('.avi'))


def get_video_path(spk: str, filename: str) -> str:
    return os.path.join(DATA_DIR, spk, *VIDEO_SUBDIR.split('/'), filename)


def get_frames_dir(spk: str, video_basename: str) -> str:
    return os.path.join(FRAMES_TEMP, spk, video_basename)


def get_session_dir(spk: str) -> str:
    return os.path.join(DATA_DIR, spk, 'sam2', 'sessions')

# ── Main application ───────────────────────────────────────────────────────────

class SAM2GUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title('SAM2 Segmentation GUI')
        self.root.geometry('1300x800')

        # ── internal state ──────────────────────────────────────────────────
        self._predictor         = None
        self._model_ready       = False
        self._inference_state   = None
        self._sam2_active       = False   # True after init_state() succeeds

        self._spk:              Optional[str] = None
        self._video_file:       Optional[str] = None
        self._cap:              Optional[cv2.VideoCapture] = None
        self._n_frames          = 0
        self._fps               = 25.0
        self._current_frame_idx = 0
        self._init_frame_idx:   Optional[int] = None
        self._current_frame_rgb: Optional[np.ndarray] = None  # (H,W,3) uint8

        self._regions:          list = []              # [SegRegion, …]
        self._active_region_idx: Optional[int] = None
        self._point_mode        = tk.IntVar(value=1)   # 1=positive, 0=negative
        self._mask_overlays:    dict = {}              # obj_id → bool (H,W)

        self._playing           = False
        self._play_timer        = None

        self._build_ui()
        self._load_model_async()

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self):
        # ── top bar ─────────────────────────────────────────────────────────
        top = tk.Frame(self.root)
        top.pack(side=tk.TOP, fill=tk.X, padx=5, pady=4)

        tk.Label(top, text='Speaker:').pack(side=tk.LEFT)
        self._spk_var = tk.StringVar()
        self._spk_cb  = ttk.Combobox(top, textvariable=self._spk_var,
                                     width=12, state='readonly')
        self._spk_cb.pack(side=tk.LEFT, padx=4)
        self._spk_cb.bind('<<ComboboxSelected>>', self._on_speaker_change)

        tk.Label(top, text='Video:').pack(side=tk.LEFT, padx=(10, 0))
        self._vid_var = tk.StringVar()
        self._vid_cb  = ttk.Combobox(top, textvariable=self._vid_var,
                                     width=30, state='readonly')
        self._vid_cb.pack(side=tk.LEFT, padx=4)
        self._vid_cb.bind('<<ComboboxSelected>>', self._on_video_change)

        self._status_lbl = tk.Label(top, text='Loading model…', fg='gray')
        self._status_lbl.pack(side=tk.RIGHT, padx=8)

        # ── middle: canvas (left) + control panel (right) ───────────────────
        mid = tk.Frame(self.root)
        mid.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5)

        self._fig    = Figure(figsize=(7, 5), tight_layout=True)
        self._ax     = self._fig.add_subplot(111)
        self._ax.set_axis_off()
        self._canvas = FigureCanvasTkAgg(self._fig, master=mid)
        self._canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._canvas.mpl_connect('button_press_event', self._on_canvas_click)

        # right panel
        rp = tk.Frame(mid, width=270, bd=1, relief=tk.SUNKEN)
        rp.pack(side=tk.RIGHT, fill=tk.Y, padx=(4, 0))
        rp.pack_propagate(False)

        tk.Label(rp, text='Regions', font=('Helvetica', 11, 'bold')).pack(
            anchor='w', padx=6, pady=(6, 2))
        tk.Button(rp, text='[+] Add region', command=self._add_region).pack(
            anchor='w', padx=6)
        self._region_listbox = tk.Listbox(rp, height=8, selectmode=tk.SINGLE,
                                          exportselection=False)
        self._region_listbox.pack(fill=tk.X, padx=6, pady=4)
        self._region_listbox.bind('<<ListboxSelect>>', self._on_region_select)
        tk.Button(rp, text='Delete selected region',
                  command=self._delete_region).pack(anchor='w', padx=6)

        ttk.Separator(rp, orient='horizontal').pack(fill=tk.X, pady=6)

        tk.Label(rp, text='Point mode:', font=('Helvetica', 10, 'bold')).pack(
            anchor='w', padx=6)
        tk.Radiobutton(rp, text='● Positive', variable=self._point_mode,
                       value=1).pack(anchor='w', padx=14)
        tk.Radiobutton(rp, text='○ Negative', variable=self._point_mode,
                       value=0).pack(anchor='w', padx=14)
        tk.Button(rp, text='Clear points',
                  command=self._clear_points).pack(anchor='w', padx=6, pady=6)

        # ── bottom: frame scrubber + action row ─────────────────────────────
        bot = tk.Frame(self.root, bd=1, relief=tk.RIDGE)
        bot.pack(side=tk.TOP, fill=tk.X, padx=5, pady=4)

        nav = tk.Frame(bot)
        nav.pack(side=tk.TOP, fill=tk.X, padx=4, pady=2)

        tk.Button(nav, text='◀◀', command=self._prev_frame).pack(side=tk.LEFT)
        self._play_btn = tk.Button(nav, text='▶', command=self._toggle_play)
        self._play_btn.pack(side=tk.LEFT)
        tk.Button(nav, text='▶▶', command=self._next_frame).pack(side=tk.LEFT)

        self._frame_slider = tk.Scale(nav, from_=0, to=1, orient=tk.HORIZONTAL,
                                      length=500, command=self._on_slider,
                                      showvalue=False)
        self._frame_slider.pack(side=tk.LEFT, padx=8, fill=tk.X, expand=True)

        self._frame_lbl = tk.Label(nav, text='Frame: 0 / 0', width=16)
        self._frame_lbl.pack(side=tk.LEFT)

        tk.Button(nav, text='Set Init Frame', command=self._set_init_frame,
                  bg='#d0e8ff').pack(side=tk.LEFT, padx=8)

        act = tk.Frame(bot)
        act.pack(side=tk.TOP, fill=tk.X, padx=4, pady=2)
        tk.Button(act, text='Save Session',
                  command=self._save_session).pack(side=tk.LEFT, padx=4)
        tk.Button(act, text='Load Session',
                  command=self._load_session).pack(side=tk.LEFT, padx=4)
        tk.Button(act, text='Export / Propagate', command=self._export_propagate,
                  bg='#ffd0a0').pack(side=tk.LEFT, padx=4)

        # populate speakers
        speakers = get_speaker_dirs()
        self._spk_cb['values'] = speakers
        if speakers:
            self._spk_var.set(speakers[0])
            self._on_speaker_change()

    # ── Model loading (background thread) ─────────────────────────────────────

    def _load_model_async(self):
        threading.Thread(target=self._load_model, daemon=True).start()

    def _load_model(self):
        try:
            from sam2.build_sam import build_sam2_video_predictor
            ckpt = os.path.join(_HERE, CHECKPOINT)
            self._predictor = build_sam2_video_predictor(
                MODEL_CFG, ckpt_path=ckpt, device=DEVICE)
            self._model_ready = True
            self.root.after(0, lambda: self._status_lbl.config(
                text='Model ready', fg='green'))
        except Exception as e:
            msg = str(e)
            self.root.after(0, lambda msg=msg: self._status_lbl.config(
                text=f'Model error: {msg}', fg='red'))

    # ── Speaker / video selection ──────────────────────────────────────────────

    def _on_speaker_change(self, event=None):
        self._spk   = self._spk_var.get()
        videos      = get_video_files(self._spk)
        self._vid_cb['values'] = videos
        if videos:
            self._vid_var.set(videos[0])
            self._on_video_change()
        else:
            self._vid_var.set('')

    def _on_video_change(self, event=None):
        self._video_file = self._vid_var.get()
        if not self._video_file:
            return
        # reset SAM2 state for new video
        self._sam2_active       = False
        self._inference_state   = None
        self._init_frame_idx    = None
        self._regions           = []
        self._active_region_idx = None
        self._mask_overlays     = {}
        self._refresh_region_list()
        self._open_video(get_video_path(self._spk, self._video_file))

    def _open_video(self, path: str):
        if self._cap is not None:
            self._cap.release()
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            messagebox.showerror('Error', f'Cannot open video:\n{path}')
            return
        self._n_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps      = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
        self._current_frame_idx = 0
        self._frame_slider.config(to=max(0, self._n_frames - 1))
        self._frame_slider.set(0)
        self._display_frame(0)

    # ── Frame navigation ───────────────────────────────────────────────────────

    def _display_frame(self, idx: int):
        if self._cap is None:
            return
        self._current_frame_idx = max(0, min(int(idx), self._n_frames - 1))
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, self._current_frame_idx)
        ret, frame = self._cap.read()
        if not ret:
            return
        self._current_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._frame_lbl.config(
            text=f'Frame: {self._current_frame_idx} / {self._n_frames - 1}')
        self._redraw()

    def _on_slider(self, val):
        self._display_frame(int(float(val)))

    def _prev_frame(self):
        self._frame_slider.set(max(0, self._current_frame_idx - 1))

    def _next_frame(self):
        self._frame_slider.set(min(self._n_frames - 1, self._current_frame_idx + 1))

    def _toggle_play(self):
        self._playing = not self._playing
        self._play_btn.config(text='⏸' if self._playing else '▶')
        if self._playing:
            self._play_step()

    def _play_step(self):
        if not self._playing:
            return
        nxt = self._current_frame_idx + 1
        if nxt >= self._n_frames:
            self._playing = False
            self._play_btn.config(text='▶')
            return
        self._frame_slider.set(nxt)
        delay = max(1, int(1000 / self._fps))
        self._play_timer = self.root.after(delay, self._play_step)

    # ── SAM2 initialisation ────────────────────────────────────────────────────

    def _set_init_frame(self):
        if not self._model_ready:
            messagebox.showwarning('Not ready', 'Model is still loading.')
            return
        if self._cap is None:
            return
        self._init_frame_idx  = self._current_frame_idx
        self._sam2_active     = False
        self._inference_state = None
        self._mask_overlays   = {}
        self._status_lbl.config(text='Extracting frames…', fg='orange')
        threading.Thread(target=self._extract_and_init, daemon=True).start()

    def _extract_and_init(self):
        video_path = get_video_path(self._spk, self._video_file)
        basename   = os.path.splitext(self._video_file)[0]
        frames_dir = get_frames_dir(self._spk, basename)
        os.makedirs(frames_dir, exist_ok=True)

        try:
            subprocess.run(
                ['ffmpeg', '-y', '-i', video_path,
                 '-q:v', '2', '-start_number', '0',
                 os.path.join(frames_dir, '%05d.jpg')],
                check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            err = e.stderr.decode(errors='replace')[-200:]
            self.root.after(0, lambda: self._status_lbl.config(
                text=f'ffmpeg failed: {err}', fg='red'))
            return

        try:
            self._inference_state = self._predictor.init_state(
                video_path=frames_dir)
            self._sam2_active = True
            # re-add any prompts that were already placed
            for region in self._regions:
                if region.points:
                    self._predictor.add_new_points_or_box(
                        self._inference_state,
                        frame_idx=self._init_frame_idx,
                        obj_id=region.obj_id,
                        points=region.points,
                        labels=region.labels,
                        clear_old_points=True,
                    )
            self.root.after(0, self._on_init_done)
        except Exception as e:
            msg = str(e)
            self.root.after(0, lambda msg=msg: self._status_lbl.config(
                text=f'Init error: {msg}', fg='red'))

    def _on_init_done(self):
        self._status_lbl.config(
            text=f'SAM2 active (init frame {self._init_frame_idx})', fg='blue')
        if self._regions:
            self._update_all_overlays()
        self._redraw()

    # ── Click interaction ──────────────────────────────────────────────────────

    def _on_canvas_click(self, event):
        if event.inaxes != self._ax:
            return
        if not self._sam2_active:
            messagebox.showinfo(
                'Not initialised',
                'Click "Set Init Frame" first to enable SAM2 interaction.')
            return
        if self._active_region_idx is None:
            messagebox.showinfo('No region selected',
                                'Select or add a region before clicking.')
            return

        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        px, py = int(round(x)), int(round(y))
        h, w   = self._current_frame_rgb.shape[:2]
        if not (0 <= px < w and 0 <= py < h):
            return

        region = self._regions[self._active_region_idx]

        if event.button == 3:          # right-click: remove last point
            if region.points:
                region.points.pop()
                region.labels.pop()
        else:
            label = self._point_mode.get()   # 1 or 0
            region.points.append([px, py])
            region.labels.append(label)

        self._run_prediction(region)

    def _run_prediction(self, region: SegRegion):
        if not region.points:
            self._mask_overlays.pop(region.obj_id, None)
            self._redraw()
            return
        try:
            import torch
            _, obj_ids, masks = self._predictor.add_new_points_or_box(
                self._inference_state,
                frame_idx=self._init_frame_idx,
                obj_id=region.obj_id,
                points=region.points,
                labels=region.labels,
                clear_old_points=True,
            )
            # masks: tensor (N_objs, 1, H, W)
            masks_np = (masks.squeeze(1).cpu().numpy() > 0)  # (N_objs, H, W)
            for oid, mask in zip(obj_ids, masks_np):
                self._mask_overlays[oid] = mask
        except Exception as e:
            self._status_lbl.config(text=f'Predict error: {e}', fg='red')
        self._redraw()

    def _update_all_overlays(self):
        for region in self._regions:
            if region.points:
                self._run_prediction(region)

    # ── Canvas rendering ───────────────────────────────────────────────────────

    def _redraw(self):
        if self._current_frame_rgb is None:
            return
        self._ax.clear()
        self._ax.set_axis_off()
        self._ax.imshow(self._current_frame_rgb)

        h, w = self._current_frame_rgb.shape[:2]

        # mask overlays (semi-transparent colour fills)
        for region in self._regions:
            mask = self._mask_overlays.get(region.obj_id)
            if mask is None:
                continue
            hex_col = region.color.lstrip('#')
            r, g, b = (int(hex_col[i:i+2], 16) / 255.0 for i in (0, 2, 4))
            overlay          = np.zeros((h, w, 4), dtype=np.float32)
            overlay[mask, 0] = r
            overlay[mask, 1] = g
            overlay[mask, 2] = b
            overlay[mask, 3] = 0.45
            self._ax.imshow(overlay)

        # prompt point markers
        for region in self._regions:
            for (px, py), lbl in zip(region.points, region.labels):
                marker = '+' if lbl == 1 else 'x'
                col    = region.color if lbl == 1 else 'red'
                self._ax.plot(px, py, marker, color=col, markersize=10, mew=2.5)

        self._canvas.draw_idle()

    # ── Region management ──────────────────────────────────────────────────────

    def _add_region(self):
        name = simpledialog.askstring('New region', 'Region name:',
                                      parent=self.root)
        if not name:
            return
        obj_id = len(self._regions)
        color  = _PALETTE[obj_id % len(_PALETTE)]
        region = SegRegion(name=name, obj_id=obj_id, color=color)
        self._regions.append(region)
        self._refresh_region_list()
        self._region_listbox.selection_clear(0, tk.END)
        self._region_listbox.selection_set(len(self._regions) - 1)
        self._active_region_idx = len(self._regions) - 1

    def _delete_region(self):
        if self._active_region_idx is None:
            return
        region = self._regions[self._active_region_idx]
        self._mask_overlays.pop(region.obj_id, None)
        self._regions.pop(self._active_region_idx)
        self._active_region_idx = None
        self._refresh_region_list()
        self._redraw()

    def _on_region_select(self, event=None):
        sel = self._region_listbox.curselection()
        self._active_region_idx = sel[0] if sel else None

    def _refresh_region_list(self):
        self._region_listbox.delete(0, tk.END)
        for r in self._regions:
            self._region_listbox.insert(tk.END, f'[{r.obj_id}] {r.name}')

    def _clear_points(self):
        if self._active_region_idx is None:
            return
        region = self._regions[self._active_region_idx]
        region.points.clear()
        region.labels.clear()
        self._mask_overlays.pop(region.obj_id, None)
        self._redraw()

    # ── Save / Load session ────────────────────────────────────────────────────

    def _session_payload(self) -> dict:
        return {
            'speaker':           self._spk,
            'video':             self._video_file,
            'initial_frame_idx': self._init_frame_idx,
            'regions':           [r.to_dict() for r in self._regions],
        }

    def _save_session(self):
        if not self._spk or not self._video_file:
            messagebox.showwarning('Nothing to save',
                                   'Select a speaker and video first.')
            return
        session_dir = get_session_dir(self._spk)
        os.makedirs(session_dir, exist_ok=True)
        default_name = os.path.splitext(self._video_file)[0] + '.json'
        path = filedialog.asksaveasfilename(
            initialdir=session_dir,
            initialfile=default_name,
            defaultextension='.json',
            filetypes=[('JSON', '*.json')])
        if not path:
            return
        with open(path, 'w') as f:
            json.dump(self._session_payload(), f, indent=2)
        self._status_lbl.config(
            text=f'Saved → {os.path.basename(path)}', fg='green')

    def _load_session(self):
        start_dir = (get_session_dir(self._spk)
                     if self._spk and os.path.isdir(get_session_dir(self._spk))
                     else os.getcwd())
        path = filedialog.askopenfilename(
            initialdir=start_dir, filetypes=[('JSON', '*.json')])
        if not path:
            return
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as e:
            messagebox.showerror('Error', f'Cannot read session:\n{e}')
            return

        spk   = data.get('speaker',   '')
        video = data.get('video',     '')

        if spk and spk != self._spk:
            self._spk_var.set(spk)
            self._on_speaker_change()
        if video and video != self._video_file:
            self._vid_var.set(video)
            self._on_video_change()

        self._init_frame_idx    = data.get('initial_frame_idx')
        self._regions           = [SegRegion.from_dict(r)
                                   for r in data.get('regions', [])]
        self._mask_overlays     = {}
        self._active_region_idx = None
        self._refresh_region_list()

        if self._init_frame_idx is not None:
            self._frame_slider.set(self._init_frame_idx)

        if self._model_ready:
            # re-extract frames and reinitialise (also re-adds prompts)
            self._set_init_frame()
        else:
            self._redraw()

        self._status_lbl.config(
            text=f'Loaded: {os.path.basename(path)}', fg='blue')

    # ── Export / Propagate ─────────────────────────────────────────────────────

    def _export_propagate(self):
        if not self._sam2_active:
            messagebox.showwarning(
                'SAM2 not active',
                'Set Init Frame and place at least one prompt point first.')
            return
        if not self._regions:
            messagebox.showwarning('No regions', 'Add at least one region.')
            return

        session_dir = get_session_dir(self._spk)
        os.makedirs(session_dir, exist_ok=True)
        basename     = os.path.splitext(self._video_file)[0]
        session_path = os.path.join(session_dir, f'{basename}.json')

        with open(session_path, 'w') as f:
            json.dump(self._session_payload(), f, indent=2)

        cmd = (f'python sam2_propagate.py '
               f'--session "{session_path}" --device {DEVICE}')
        messagebox.showinfo(
            'Session exported',
            f'Session saved to:\n{session_path}\n\n'
            f'Run batch propagation with:\n{cmd}')


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    SAM2GUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
