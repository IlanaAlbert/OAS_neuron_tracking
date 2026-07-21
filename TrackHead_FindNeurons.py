"""
worm_tracker_gui.py
───────────────────
Run from terminal:  python worm_tracker_gui.py

TWO-STAGE WORM ANALYSIS TOOL
═════════════════════════════

Stage 1 — Event Picker
  A fast scrubber showing only the behavior channel at reduced resolution.
  Mark frames of interest with the spacebar (or the Mark button), add a note,
  then click "Continue to Neuron Labeling" when done.
  Events are also saved immediately to events_NNN.json.

Stage 2 — Neuron Labeler
  For each marked event a ±500-frame window opens in the GCaMP channel.
  Keys 1-5 select the active neuron slot; clicking places/corrects that neuron.
  Positions carry forward from the last clicked frame (interpolation-by-default).
  Results are saved to neurons_NNN_frameXXXXX.npz per event window.

────────────────────────────────────────────────────────────────────────────
CONFIGURATION — edit below before running
────────────────────────────────────────────────────────────────────────────
"""

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import threading
import os
import json
from io import BytesIO
from PIL import Image, ImageTk, ImageDraw   # pip install pillow
import glob

import numpy as np
import h5py
import cv2
from scipy import ndimage

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.optimize import linear_sum_assignment
import matplotlib.patches as patches
from skimage import morphology
from scipy.ndimage import uniform_filter, binary_fill_holes

from openpyxl import Workbook
from concurrent.futures import ThreadPoolExecutor




# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION  ← edit these
# ══════════════════════════════════════════════════════════════════════════════

B_FP   = r""   # set at startup via the path selector dialog
GC_FP  = r""   # set at startup via the path selector dialog


# Filename patterns — {idx} is replaced by the zero-padded index ("000", "001", …)
# Change these to match your actual naming scheme:
B_PATTERN  = "{idx}.h5"
GC_PATTERN = "{idx}.h5"

AMPLIFICATION = 10
SHIFT_X = -35          # pixel shift to align behavior → gcamp
SHIFT_Y = -10

DEBUG_MAX_FRAMES = None   # set to None to load all frames

INDICES_DEADPIXELS = np.array([
    [124,  55],
    [124,  56],
    [125,  56],
    [193,  51],
    [213, 187],
    [374, 187],
    [484,  46],
], dtype=np.int64)

EVENT_WINDOW = 500          # frames on each side of a marked event
DISPLAY_SCALE = 0.4         # fraction of original resolution for the event picker
NEURON_NAMES  = ["Neuron 1", "Neuron 2", "Neuron 3", "Neuron 4", "Neuron 5"]
NEURON_COLORS = ["#ff4444", "#44ff88", "#4488ff", "#ffcc00", "#ff44ff"]

EXPORT_EXCEL    = ""   # set by _initialize_paths
EXPORT_METADATA = ""   # set by _initialize_paths

# ── stubs: replace with your real implementations ────────────────────────────
def fix_dead(g_arr, deadpixels = INDICES_DEADPIXELS):
    """Replace dead pixels in GCaMP array."""
    for i,j in deadpixels:
        g_arr[i,j] = 0
    return g_arr

def build_frame_alignment(b_times, g_times):
    """Find the correct g_frame for b_frame"""
    cost_matrix = np.abs(
        b_times[:, None] - g_times[None, :]
    ) # cost matrix of absolute time differences - shape: (n_bheavior, n_gcamp)
    # optimal one-to-one assignment minimizing total time difference
    b_indicies, g_inidices = linear_sum_assignment(cost_matrix)
    
    alignment = {b: g for b, g in zip(b_indicies, g_inidices)}
    return alignment

def get_g_frame_idx(file_idx, b_frame_idx, alignment_dict):
    """Map behavior frame index → GCaMP frame index."""
    g_idx = alignment_dict[file_idx][b_frame_idx]
    return g_idx

# ── brightness helpers ────────────────────────────────────────────────────────
# Precompute relative pixel offsets for a circle of radius BRIGHTNESS_RADIUS once
# at import time so every call to percentile_in_circle_fast skips the meshgrid.
BRIGHTNESS_RADIUS = 10
_bdy, _bdx = np.mgrid[-BRIGHTNESS_RADIUS:BRIGHTNESS_RADIUS+1,
                       -BRIGHTNESS_RADIUS:BRIGHTNESS_RADIUS+1]
_bmask = (_bdy**2 + _bdx**2) <= BRIGHTNESS_RADIUS**2
_CIRC_DY = _bdy[_bmask]   # shape (N_pixels_in_circle,)
_CIRC_DX = _bdx[_bmask]

def percentile_in_circle_fast(arr, center, percentile=99.99):
    """99.99th-percentile pixel value inside a circle centered on `center`=(row,col)."""
    r = int(round(float(center[0])))
    c = int(round(float(center[1])))
    h, w = arr.shape
    rows = r + _CIRC_DY
    cols = c + _CIRC_DX
    valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    vals = arr[rows[valid], cols[valid]]
    return float(np.percentile(vals, percentile)) if vals.size else 0.0


def _process_one_neuron_file(neuron_file, gc_file_list):
    """
    Load one saved .npz, read GCaMP frames from the matching HDF5, and compute
    per-neuron brightness at every annotated frame.  Runs in a worker thread so
    each call opens its own h5py handle — safe for concurrent reads across files.
    Returns a result dict, or None on error.
    """
    try:
        with np.load(neuron_file) as data:
            neuron_pos   = data['positions']                              # (n, n_frames, 2)
            frames       = data['frames']                                 # behavior frame indices
            gcamp_frames = data['gcamp_frames']                          # GCaMP frame indices
            fileidx      = int(data['fileidx'])
            event_frame  = int(data['event_frame'])
            note         = str(data['note']) if data['note'].size else ''

        num_neurons = neuron_pos.shape[0]
        n_frames    = len(frames)
        brightness  = np.zeros((num_neurons, n_frames))
        times       = np.empty(n_frames)

        alignment = ALIGNMENT_DICT[fileidx]
        with h5py.File(gc_file_list[fileidx], 'r') as gf:
            event_time = float(gf['times'][alignment[event_frame]])
            for ci, g_idx in enumerate(gcamp_frames):
                arr       = np.array(gf['data'][int(g_idx)], dtype=np.float32)
                times[ci] = float(gf['times'][int(g_idx)]) - event_time
                for ni in range(num_neurons):
                    rx, ry = neuron_pos[ni, ci, :]
                    if np.isnan(rx):
                        continue
                    brightness[ni, ci] = percentile_in_circle_fast(arr, (rx, ry))

        return {
            'file':          neuron_file,
            'fileidx':       fileidx,
            'event_frame':   event_frame,
            'note':          note,
            'frames':        frames,        # behavior frame indices
            'gcamp_frames':  gcamp_frames,  # GCaMP frame indices
            'times':         times,
            'brightness':   brightness,
            'positions':    neuron_pos,
            'num_neurons':  num_neurons,
        }
    except Exception as e:
        print(f"[export] {neuron_file}: {e}")
        return None

ALIGNMENT_DICT = {}
b_files        = []
gc_files       = []

def _initialize_paths(b_fp, gc_fp):
    """
    Set the global folder paths and build the frame-alignment dict.
    Called once after the startup dialog confirms paths.
    """
    global B_FP, GC_FP, EXPORT_EXCEL, EXPORT_METADATA
    global b_files, gc_files, ALIGNMENT_DICT

    B_FP  = b_fp
    GC_FP = gc_fp
    EXPORT_EXCEL    = os.path.join(B_FP, "neuron_brightness.xlsx")
    EXPORT_METADATA = os.path.join(B_FP, "metadata.txt")

    b_files  = sorted(glob.glob(os.path.join(B_FP,  "*.h5")))
    gc_files = sorted(glob.glob(os.path.join(GC_FP, "*.h5")))

    ALIGNMENT_DICT = {}
    frames_loaded  = 0
    for file_idx in range(len(b_files)):
        if DEBUG_MAX_FRAMES is not None and frames_loaded >= DEBUG_MAX_FRAMES:
            break
        with h5py.File(b_files[file_idx], 'r') as bf:
            with h5py.File(gc_files[file_idx], 'r') as gf:
                alignment = build_frame_alignment(
                    np.array(bf['times'][:]).flatten(),
                    np.array(gf['times'][:]).flatten()
                )
                ALIGNMENT_DICT[file_idx] = alignment
                frames_loaded += int(bf['data'].shape[0])

# ══════════════════════════════════════════════════════════════════════════════
#  FILE DISCOVERY  (shared by both windows)
# ══════════════════════════════════════════════════════════════════════════════

def discover_file_pairs(g_folder, b_folder, b_pattern, gc_pattern):
    pairs = []
    idx = 0
    while True:
        tag     = str(idx).zfill(6)
        b_path  = os.path.join(b_folder, b_pattern.format(idx=tag))
        gc_path = os.path.join(g_folder, gc_pattern.format(idx=tag))

        b_exists  = os.path.exists(b_path)
        gc_exists = os.path.exists(gc_path)

        if not b_exists and not gc_exists:
            break   # no more pairs at this index

        if b_exists and gc_exists:
            try:
                with h5py.File(b_path, 'r') as f:
                    n_frames = int(f['data'].shape[0])
                pairs.append({"idx": idx, "b_path": b_path,
                               "gc_path": gc_path, "n_frames": n_frames})
            except Exception as e:
                print(f"[discover] skipping index {idx}: {e}")
        else:
            missing = b_path if not b_exists else gc_path
            print(f"[discover] index {idx}: {missing} not found — skipping pair")

        idx += 1

    return pairs


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED IMAGE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_behavior_raw(bf, frame):
    """Load a single behavior frame as a uint8 array (no mask processing)."""
    arr = bf['data'][frame]
    shifted = ndimage.shift(arr.astype(float), shift=(SHIFT_X, SHIFT_Y),
                            order=1, cval=255)
    return np.clip(shifted, 0, 255).astype(np.uint8)


def load_gcamp_raw(gf, fileidx, frame):
    """Load behavior + GCaMP for a single frame, return (g_uint8, b_uint8)."""
    g_idx = get_g_frame_idx(fileidx, frame, ALIGNMENT_DICT)
    g_arr = fix_dead(gf['data'][g_idx])
    g_arr = np.clip(g_arr * AMPLIFICATION, 0, 255).astype(np.uint8)
    return g_arr


def arr_to_photoimage(arr, scale=1.0):
    """
    Convert a uint8 numpy array to a Tk PhotoImage.
    Uses PIL for speed and optional downscaling.
    This avoids matplotlib overhead entirely for the fast scrubber.
    """
    if arr.ndim == 2:
        img = Image.fromarray(arr, mode='L').convert('RGB')
    else:
        img = Image.fromarray(arr)
    if scale != 1.0:
        w = max(1, int(img.width  * scale))
        h = max(1, int(img.height * scale))
        img = img.resize((w, h), Image.NEAREST)  # NEAREST = fastest
    return ImageTk.PhotoImage(img)


# ══════════════════════════════════════════════════════════════════════════════
#  IMAGE PROCESSING
# ══════════════════════════════════════════════════════════════════════════════
# def make_worm_mask(bf, frame):
#     # Remember will need to get the correct file and frame from the continuous frame index
#     b_arr = load_behavior_raw(bf, frame)
#     # Binary worm mask: worm body is dark on a bright background
#     mask = (b_arr < 75).astype(np.uint8)
#     mask = morphology.binary_closing(mask, morphology.disk(10))
#     mask = np.array(binary_fill_holes(mask)).astype(np.uint8)

#     # Keep only the largest connected component (discard debris / noise blobs)
#     n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
#     if n_labels > 1:
#         largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
#         mask = (labels == largest).astype(np.uint8)
#     return mask

# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 1 — EVENT PICKER
# ══════════════════════════════════════════════════════════════════════════════

class EventPickerWindow:
    """
    Fast behavior-only scrubber.  Mark frames with Space or the Mark button.
    Add a note per event in the text field before marking.
    When done, saves events_NNN.json and opens the Neuron Labeler.
    """

    def __init__(self, root):
        self.root = root
        self.root.title("Stage 1 — Event Picker")
        self.root.configure(bg="#1a1a2e")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # ──────────────────── virtual timeline state ────────────────────────
        self.file_pairs   = []          # populated by _discover_and_populate
        self._offset = []               # offsets[i] = global frame index of file i frame 0
        self._total_frames = 0

        # Currently open HDF5 nadle + which fileidx it belongs to
        self.bf = None
        self.gf = None 
        self._open_fidx = -1

        # Global frame position
        self.current_gframe = 0

        # All marked events: list of {"grame": int, "fileidx": int,
        #                               "local_frame": int, "note": str}
        self.events = []

        # ──────────────────── head tracking state ────────────────────────
        # head_positions[gframe] = (row, col) in FULL-resolution pixels
        self.head_positions = {}
        self._pending_seed = None # (grame, full_row, rull_col) set by click
        self._is_tracking = False
        self._show_overlay = True # press H to toggle dot overlay

        # Frame image cache: {gframe: PhotoImage} (capped at _CACHE_SIZE)
        self._img_cache = {}
        self._CACHE_SIZE   = 120

        # Playback
        self._playing      = False
        self._play_speed   = 10           # ms between frames (~50 fps default)

        self._build_ui()
        self._discover_and_populate()
        self.root.bind("<Left>",       lambda e: self._step(-1))
        self.root.bind("<Right>",      lambda e: self._step(1))
        self.root.bind("<Delete>",     lambda e: self._remove_last_event())
        self.root.bind("h",     lambda e: self._toggle_overlay())
        self.img_canvas.bind("<Button-1>",  self._on_canvas_click)
        self.img_canvas.bind("<Configure>", self._on_canvas_resize)

    # ─────── virtual timeline helpers ─────────────────────────────────────────

    def _global_to_local(self, gframe):
        """
        Convert a global frame number to (fileidx, local_frame).
        Uses a binary search over the pre-computed offsets array for speed.
        """
        # np.searchsorted finds the insertion point; subtract 1 to get the file 
        # whose offset is <= gframe
        i = int(np.searchsorted(self._offsets, gframe, side='right')) - 1
        i = max(0, min(i, len(self.file_pairs) -1))
        local = gframe - self._offsets[i]
        return self.file_pairs[i]["idx"], local, i
    
    def _ensure_file_open(self, fileidx, list_idx):
        """
        Open the behavior AND GCaMP HDF5 for 'fileidx' if it isn't already open.
        Closes the previously open handle first.
        """
        if self._open_fidx == fileidx:
            return # already open, notheing to do
        for h in (self.bf, self.gf):
            if h is not None:
                try: h.close()
                except: pass
        
        self.bf = None
        self.gf = None

        
        pair = self.file_pairs[list_idx]
        if not os.path.exists(pair["b_path"]):
            self._demo_mode = True
            self._open_fidx = fileidx
            return
        
        self._demo_mode = False
        try:
            self.bf = h5py.File(pair["b_path"], 'r')
            self.gf = h5py.File(pair["gc_path"], 'r')
            self._open_fidx = fileidx
        except Exception as e:
            messagebox.showerror("File error", str(e))
            self._demo_mode = True
            self._open_fidx = fileidx

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        BG     = "#1a1a2e"
        PANEL  = "#16213e"
        ACCENT = "#e94560"
        TEXT   = "#eaeaea"
        MUTED  = "#8892a4"
        BTN_BG = "#0f3460"
        BTN_A  = "#e94560"

        def btn(parent, label, cmd, highlight=False, side="left", padx=4, pady=0):
            b = tk.Button(parent, text=label, command=cmd,
                          bg=ACCENT if highlight else BTN_BG,
                          fg=TEXT, relief="flat",
                          font=("Courier New", 9, "bold"),
                          padx=8, pady=pady,
                          activebackground=BTN_A, cursor="hand2")
            b.pack(side=side, padx=padx)
            return b

        # ── top bar ───────────────────────────────────────────────────────────
        top = tk.Frame(self.root, bg=BG, pady=6)
        top.pack(fill="x", padx=10)

        tk.Label(top, text="EVENT PICKER",
                 font=("Courier New", 13, "bold"),
                 fg=ACCENT, bg=BG).pack(side="left")
        
        # Current file indiciator (read-only, updates as you scroll)
        self.file_indicator = tk.Label(top, text = "",
                                       font = ("Courier New", 9), fg = TEXT, bg=BG)
        self.file_indicator.pack(side="left", padx = 20)

        self.status_var = tk.StringVar(value="Scanning for files...")
        tk.Label(
            top, textvariable=self.status_var,
            font=("Courier New", 9), fg=MUTED, bg=BG).pack(side="right")

        # ── image canvas (raw Tk canvas — no matplotlib overhead) ─────────────
        self.canvas_frame = tk.Frame(self.root, bg="black")
        self.canvas_frame.pack(fill="both", expand=True, padx=10, pady=4)

        self.img_canvas = tk.Canvas(self.canvas_frame, bg="black",
                                    highlightthickness=0)
        self.img_canvas.pack(fill="both", expand=True)
        self._tk_img = None     # hold reference so GC doesn't collect it

        # ── slider ────────────────────────────────────────────────────────────
        sf = tk.Frame(self.root, bg=PANEL, pady=6)
        sf.pack(fill="x", padx=10, pady=(0, 2))

        tk.Button(sf, text="◀", command=lambda: self._step(-1),
                  bg=BTN_BG, fg=TEXT, relief="flat",
                  font=("Courier New", 10, "bold"),
                  activebackground=BTN_A, cursor="hand2").pack(side="left", padx=4)

        self.frame_var = tk.IntVar(value=0)
        # 'to' is set in _discover_and_populate once total frames are known
        self.slider = ttk.Scale(sf, from_=0, to=0,
                                variable=self.frame_var, orient="horizontal",
                                command=self._on_slider)
        self.slider.pack(side="left", fill="x", expand=True, padx=6)

        tk.Button(sf, text="▶", command=lambda: self._step(1),
                  bg=BTN_BG, fg=TEXT, relief="flat",
                  font=("Courier New", 10, "bold"),
                  activebackground=BTN_A, cursor="hand2").pack(side="left", padx=4)

        self.play_btn = btn(sf, "⏵", self._toggle_play, padx=6)

        # Speed slider
        tk.Label(sf, text="speed:", font=("Courier New", 8),
                 fg=MUTED, bg=PANEL).pack(side="left", padx=(8, 2))
        self.speed_var = tk.IntVar(value=50)
        tk.Scale(sf, from_=10, to=200, orient="horizontal",
                 variable=self.speed_var, length=80,
                 bg=PANEL, fg=TEXT, highlightthickness=0,
                 troughcolor=BTN_BG, showvalue=False,
                 command=lambda v: setattr(self, '_play_speed', int(float(v)))
                 ).pack(side="left", padx=2)

        self.frame_label = tk.Label(sf, text="Frame: —", width=16,
                                    font=("Courier New", 9), fg=TEXT, bg=PANEL)
        self.frame_label.pack(side="left", padx=8)

        # ── control bar ───────────────────────────────────────────────────────
        ctrl = tk.Frame(self.root, bg=PANEL, pady=6)
        ctrl.pack(fill="x", padx=10, pady=(0, 4))

        tk.Label(ctrl, text="Note:", font=("Courier New", 9),
                 fg=TEXT, bg=PANEL).pack(side="left", padx=(10, 4))
        self.note_var = tk.StringVar()
        tk.Entry(ctrl, textvariable=self.note_var, width=24,
                 bg=BTN_BG, fg=TEXT, insertbackground=TEXT,
                 relief="flat", font=("Courier New", 9)
                 ).pack(side="left", padx=4)

        btn(ctrl, "⚑  Mark Frame", self._mark_frame, highlight=True, padx=8)
        btn(ctrl, "✕  Remove Last",         self._remove_last_event, padx=4)
        btn(ctrl, "▶▶  Continue →",         self._continue_to_labeler,
            highlight=False, side="right", padx=12)
        
        # ── head tracking bar ───────────────────────────────────────────────────────
        # DEBUG idk if I ever use this...
        track_bar = tk.Frame(self.root, bg=PANEL, pady=5)
        track_bar.pack(fill="x", padx=10, pady=(0,2))

        tk.Label(track_bar, text="HEAD TRACK:",
                 font=("Courier New", 9, "bold"), fg=ACCENT, bg=PANEL).pack(side="left", padx=(10,8))
        
        self.seed_label = tk.Label(track_bar, text="No seed - click image to set",
                                   font=("Courier New", 8), fg=MUTED, bg=PANEL)
        self.seed_label.pack(side="left", padx=4)

        tk.Label(track_bar, text="N:", font=("Courier New", 9),
                 fg=TEXT, bg=PANEL).pack(side="left", padx=(12, 2))
        self.track_n_var = tk.IntVar(value=200)
        tk.Spinbox(track_bar, from_=1, to=99999, textvariable=self.track_n_var,
                   width=6, font=("Courier New", 9),
                   bg=BTN_BG, fg=TEXT, relief="flat").pack(side="left", padx=4)
        
        self.track_btn = btn(track_bar, "▶ Track N Frames",
                             self._start_tracking, highlight=True, padx=8)
        
        self.progress = ttk.Progressbar(track_bar, mode="determinate", length=100)
        self.progress.pack(side="left", padx=8)

        self.overlay_btn = btn(track_bar, "• Overlay ON",
                               self._toggle_overlay, padx=6),
        tk.Label(track_bar, text="(H)", font=("Courier New", 7),
                 fg=MUTED, bg=PANEL).pack(side="left", padx=2)

        # ── event list ────────────────────────────────────────────────────────
        list_frame = tk.Frame(self.root, bg=BG)
        list_frame.pack(fill="x", padx=10, pady=(0, 6))

        tk.Label(list_frame, text="Marked events:",
                 font=("Courier New", 8), fg=MUTED, bg=BG).pack(anchor="w")

        self.event_listbox = tk.Listbox(
            list_frame, height=4, bg=PANEL, fg=TEXT,
            font=("Courier New", 8), relief="flat",
            selectbackground=ACCENT, activestyle="none"
        )
        self.event_listbox.pack(fill="x")
        self.event_listbox.bind("<Double-Button-1>", self._jump_to_event)

        # ttk styling
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Horizontal.TScale",   background=PANEL, troughcolor=BTN_BG)
        # style.configure("TCombobox", fieldbackground=BTN_BG, background=BTN_BG,
        #                 foreground=TEXT, selectbackground=ACCENT)

    # ── file handling ─────────────────────────────────────────────────────────

    def _discover_and_populate(self):
        if os.path.isdir(B_FP):
            self.file_pairs = discover_file_pairs(GC_FP, B_FP, B_PATTERN, GC_PATTERN)
        if not self.file_pairs:
            # Demo mode: two synthetic files
            self.file_pairs = [
                {"idx": 0, "b_path": "", "gc_path": "", "n_frames": 300},
                {"idx": 1, "b_path": "", "gc_path": "", "n_frames": 300},
            ]
            self._demo_mode = True
            self._set_status("⚠  Demo mode")
        else:
            self._demo_mode = False
        
        # Apply debug frame cap: truncate file list and clamp last file's n_frames
        if DEBUG_MAX_FRAMES is not None:
            capped, count = [], 0
            for p in self.file_pairs:
                if count >= DEBUG_MAX_FRAMES:
                    break
                remaining = DEBUG_MAX_FRAMES - count
                if p["n_frames"] > remaining:
                    p = dict(p, n_frames=remaining)
                capped.append(p)
                count += p["n_frames"]
            self.file_pairs = capped

        # Build cumulative offset array: offsets[i] = first global frame of file i
        self._offsets = []
        cumulative = 0
        for p in self.file_pairs:
            self._offsets.append(cumulative)
            cumulative += p["n_frames"]
        self._offsets = np.array(self._offsets)
        self._total_frames = int(cumulative)

        self.slider.configure(to=self._total_frames - 1)

        n_files = len(self.file_pairs)
        self._set_status(
            f"Loaded {n_files} file{'s' if n_files!=1 else ''} - "
            f"{self._total_frames} total frames. Scroll freely." 
        )
        self._show_frame(0)

    # ── frame display (pure Tk, no matplotlib) ────────────────────────────────

    def _get_display_image(self, gframe):
        """
        Return a cached PIL Image for global `gframe` at DISPLAY_SCALE resolution.
        The cache stores PIL Images (not PhotoImages) so _show_frame can resize
        them to fit the canvas at whatever size the window currently is.
        Cache key includes overlay state so toggling refreshes the dot correctly.
        """
        cache_key = (gframe, self._show_overlay)
        if cache_key in self._img_cache:
            return self._img_cache[cache_key]

        fidx, local_frame, list_idx = self._global_to_local(gframe)
        self._ensure_file_open(fidx, list_idx)

        if self._demo_mode or self.bf is None:
            arr = self._synthetic_beh(local_frame)
        else:
            arr = load_behavior_raw(self.bf, local_frame)

        # Downsample to DISPLAY_SCALE for the cache (cheap intermediate)
        thumb_w = max(1, int(arr.shape[1] * DISPLAY_SCALE))
        thumb_h = max(1, int(arr.shape[0] * DISPLAY_SCALE))
        img = Image.fromarray(arr, mode="L").convert('RGB').resize(
            (thumb_w, thumb_h), Image.NEAREST)

        # Draw head position dot if overlay is on and a position exists
        if self._show_overlay and gframe in self.head_positions:
            hr, hc = self.head_positions[gframe]
            dr = int(hr * DISPLAY_SCALE)
            dc = int(hc * DISPLAY_SCALE)
            draw  = ImageDraw.Draw(img)
            r_dot = max(3, int(6 * DISPLAY_SCALE))
            draw.ellipse([dc-r_dot-1, dr-r_dot-1, dc+r_dot+1, dr+r_dot+1],
                         outline=(255, 255, 255), width=1)
            draw.ellipse([dc-r_dot-1, dr-r_dot-1, dc+r_dot+1, dr+r_dot+1],
                         fill=(220, 60, 60))

        # Cap cache size: evict the oldest entry
        if len(self._img_cache) >= self._CACHE_SIZE:
            oldest = next(iter(self._img_cache))
            del self._img_cache[oldest]
        self._img_cache[cache_key] = img
        return img

    def _show_frame(self, gframe):
        self.current_gframe = gframe
        self.frame_var.set(gframe)

        fidx, local_frame, list_idx = self._global_to_local(gframe)

        fname = os.path.basename(self.file_pairs[list_idx]["b_path"]) or f"[demo {fidx}]"
        self.file_indicator.config(text=f"file [{fidx:03d} {fname}]")
        self.frame_label.config(text=f"global {gframe} / local {local_frame}")

        pil_img = self._get_display_image(gframe)

        # Fit the image into the current canvas size, preserving aspect ratio
        cw = self.img_canvas.winfo_width()  or pil_img.width
        ch = self.img_canvas.winfo_height() or pil_img.height
        scale = min(cw / pil_img.width, ch / pil_img.height)
        disp_w = max(1, int(pil_img.width  * scale))
        disp_h = max(1, int(pil_img.height * scale))
        fitted = pil_img.resize((disp_w, disp_h), Image.NEAREST)

        photo = ImageTk.PhotoImage(fitted)
        self._tk_img = photo   # keep reference so GC doesn't collect it

        self.img_canvas.delete("all")
        self.img_canvas.create_image(cw // 2, ch // 2, anchor="center", image=photo)

        is_marked = any(e["gframe"] == gframe for e in self.events)
        self.img_canvas.config(bg="#2a0a0a" if is_marked else "black")

    def _on_canvas_resize(self, _event=None):
        """Redraw current frame when the canvas is resized; debounced to ~60 ms."""
        if hasattr(self, '_resize_after_id'):
            self.root.after_cancel(self._resize_after_id)
        self._resize_after_id = self.root.after(
            60, lambda: self._show_frame(self.current_gframe)
        )

    @staticmethod
    def _synthetic_beh(local_frame):
        rng = np.random.default_rng(local_frame + 9999)
        h, w = 512, 512
        arr  = np.full((h, w), 180, dtype=np.uint8)
        cx   = int(w * 0.3 + (local_frame % 200) * 0.5)
        cy   = int(h * 0.5 + 50 * np.sin(local_frame / 30))
        for i in range(h):
            c = int(cx + (i - cy) * 0.3)
            arr[i, max(0, c - 20):min(w, c + 20)] = 40
        return arr

    # ── slider / playback ─────────────────────────────────────────────────────

    def _on_slider(self, val):
        gframe = int(float(val))
        if gframe != self.current_gframe:
            self._show_frame(gframe)

    def _step(self, delta):
        # n     = self.current_pair["n_frames"] if self.current_pair else 1
        gframe = max(0, min(self._total_frames - 1, self.current_gframe + delta))
        self.frame_var.set(gframe)
        self._show_frame(gframe)

    def _toggle_play(self):
        if self._playing:
            self._playing = False
            self.play_btn.config(text="⏵")
        else:
            self._playing = True
            self.play_btn.config(text="⏸")
            self._play_loop()

    def _play_loop(self):
        if not self._playing:
            return
        if self.current_gframe >= self._total_frames - 1:
            self._playing = False
            self.play_btn.config(text="⏵")
            return
        self._step(1)
        self.root.after(self._play_speed, self._play_loop)

    # ── event marking ─────────────────────────────────────────────────────────

    def _mark_frame(self):
        gframe = self.current_gframe
        if any(e["gframe"] == gframe for e in self.events):
            self._set_status(f"Global frame {gframe} already marked.")
            return
        fidx, local_frame, _ = self._global_to_local(gframe)
        note = self.note_var.get().strip()
        self.events.append({
            "gframe": gframe,
            "fileidx": fidx,
            "local_frame": local_frame,
            "note": note,
        })
        
        self.events.sort(key=lambda e: e["gframe"])
        self.note_var.set("")
        self._refresh_event_list()
        self._show_frame(gframe)   # re-draw with highlight
        self._set_status(f"✓ Marked frame [{fidx:03d}] local frame {local_frame}"
                         f"({len(self.events)} events total)")

    def _remove_last_event(self):
        if self.events:
            removed = self.events.pop()
            self._refresh_event_list()
            self._set_status(f"Removed event at file [{removed['fileidx']:03d}]"
                             f"local frame {removed['local_frame']}")

    def _jump_to_event(self, _=None):
        sel = self.event_listbox.curselection()
        if sel:
            gframe = self.events[sel[0]]["gframe"]
            self.frame_var.set(gframe)
            self._show_frame(gframe)

    def _refresh_event_list(self):
        self.event_listbox.delete(0, "end")
        for e in self.events:
            note_str = f"  —  {e['note']}" if e["note"] else ""
            self.event_listbox.insert("end", 
                                      f"[{e['fileidx']:03d}] local {e['local_frame']:>6}"
                                      f"    (global {e['gframe']}){note_str}")

    # ── continue ──────────────────────────────────────────────────────────────

    def _continue_to_labeler(self):
        if not self.events:
            messagebox.showwarning("No events", "Mark at least one frame first.")
            return

        # Save events to JSON
        out_path = os.path.join(B_FP, f"events_all.json")
        with open(out_path, "w") as f:
            head_pos_serializable = {
                int(k): [int(v[0]), int(v[1])]
                for k, v in self.head_positions.items()
            }

            json.dump({"events": self.events,
                       "head_positions": head_pos_serializable}, f, indent=2, 
                      default=lambda x: int(x) if hasattr(x, 'item') else x)
        self._set_status(f"✓ Saved {out_path}")

        # Open a single labeler window with all events across all files
        self._playing = False
        labeler_events = [
            {"frame": e["local_frame"], "note": e["note"], "fileidx": e["fileidx"]}
            for e in self.events
        ]
        labeler_win = tk.Toplevel(self.root)
        self._labeler_ref = NeuronLabelerWindow(labeler_win, self.file_pairs, labeler_events)

    # ── head tracking ───────────────────────────────────────────────────────────────
    # DEBUG do I even use this??
    def _on_canvas_click(self, event):
        """
        Register a click on the behavior canvas as a head seed position.
        Converts display-resolution (x,y) back to full-resolution (row,col)
        """
        cw = self.img_canvas.winfo_width()
        ch = self.img_canvas.winfo_height()

        fidx, local_frame, list_idx = self._global_to_local(self.current_gframe)
        self._ensure_file_open(fidx, list_idx)

        # Inter full-res image dimensions
        if not self._demo_mode and self.bf is not None:
            full_h = self.bf['data'].shape[1]
            full_w = self.bf['data'].shape[2]
        else:
            full_h, full_w = 512, 512

        disp_w = int(full_w * DISPLAY_SCALE)
        disp_h = int(full_h * DISPLAY_SCALE)

        # Canvas centers the image - compute top-left offset
        x0 = (cw - disp_w) // 2
        y0 = (ch - disp_h) // 2

        ix = event.x - x0
        iy = event.y - y0

        if ix < 0 or iy < 0 or ix >= disp_w or iy >= disp_h:
            return # click outside image area
        
        # Scale up to full resolution
        full_r = int(iy / DISPLAY_SCALE)
        full_c = int(ix / DISPLAY_SCALE)

        self._pending_seed = (self.current_gframe, full_r, full_c)
        self.seed_label.config(
            text=f"Seed: row={full_r}, col={full_c} frame {self.current_gframe}",
            fg="#00ffcc"
        )
        # Store immediately and refresh display with dot
        self.head_positions[self.current_gframe] = (full_r, full_c)
        self._invalidate_cache(self.current_gframe)
        self._show_frame(self.current_gframe)

    def _invalidate_cache(self, gframe):
        """Remove both overlay states for gframe from the image cache."""
        for key in [(gframe, True), (gframe, False)]:
            self._img_cache.pop(key, None)

    def _start_tracking(self):
        """Validate seed and kick off background tracking thread."""
        if self._is_tracking:
            return
        if self._pending_seed is None:
            messagebox.showwarning(
                "No seed",
                "Click the worm head on the image first to set a seed position"
            )
            return
        seed_gframe, seed_r, seed_c = self._pending_seed
        n = self.track_n_var.get()
        end_gframe = min(seed_gframe + n, self._total_frames - 1)

        self._is_tracking = True
        self._playing = False
        self.play_btn.config(text="▶")
        self.track_btn.config(state="disabled")
        self.progress["value"] = 0
        self.progress["maximum"] = max(1, end_gframe - seed_gframe)
        self._set_status(
            f"Tracking frames {seed_gframe} to {end_gframe} in background..."
        )
        threading.Thread(
            targe=self._tracking_worker,
            args=(seed_gframe, end_gframe, seed_r, seed_c),
            daemon=True
        ).start()

    def _tracking_worker(self, start_gframe, end_gframe, seed_r, seed_c):
        """
        Background thread: run find_head() frame-by-frame on the GCaMP channel.
        Results are stored into self.head_positions keyed by global frame number.
        """
        prev_head = (seed_r, seed_c)
        total = max(1, end_gframe - start_gframe)

        for i, gframe in enumerate(range(start_gframe+1, end_gframe+1)):
            if not self._is_tracking:
                break
            try:
                fidx, local_frame, list_idx = self._global_to_local(gframe)
                self._ensure_file_open(fidx, list_idx)

                if self._demo_mode or self.bf is None or self.gf is None:
                    # Demo mode: jitter seed slightly
                    pr, pc = prev_head
                    rng = np.random.default_rng(gframe)
                    prev_head = (pr + int(rng.integers(-3,4)),
                                 pc + int(rng.integers(-3,4)))
                    self.head_positions[gframe] = prev_head
                else:
                    g_arr, mask, _ = load_frame(
                        self.bf, self.gf, fidx, local_frame
                    )
                    hr, hc, _box, _ = find_head(g_arr, mask, prev_head=prev_head)
                    self.head_positions[gframe] = (int(hr), int(hc))
                    prev_head = (int(hr), int(hc))

                # Invalidate cache so dot appears when user scrubs to this frame
                self._invalidate_cache(gframe)
            
            except Exception as e:
                print(f"[tracking frame {gframe}] {e}")

            pct = (i + 1) / total * 100
            self.root.after(0, lambda p=pct: self.progress.configure(value=p))    
        
        self.root.after(0, self._tracking_done, start_gframe, end_gframe)

    def _tracking_done(self, start_gframe, end_gframe):
        """Called on the main thread when tracking finishes."""
        self._is_tracking = False
        self.track_btn.config(state="normal")
        self.progress["value"] = self.progress["maximum"]
        self._pending_seed = None
        self.seed_label.config(
            text="trcaking complete - scrub to verify, click to re-seed",
            fg="#44ff88"
        )
        # Auto-payse and jump to end of tracked window for review
        self._playing = False
        self.play_btn.config(text="▶")
        self.frame_var.set(end_gframe)
        self._show_frame(end_gframe)
        n_tracked = end_gframe - start_gframe
        self._set_status(
            f"Tracked {n_tracked} frames ({start_gframe} to {end_gframe})."
            "Scrub to verify. Click + Track to re-seed and correct."
        )

    def _toggle_overlay(self):
        """Toggle the head position dot overlay on/off (also bound to H key)."""
        self._show_overlay = not self._show_overlay
        label = "• Overlay ON" if self._show_overlay else "◦ Overlay OFF"
        self.overlay_btn.config(text=label)
        self._show_frame(self.current_gframe)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _set_status(self, msg):
        self.status_var.set(msg)

    def _on_close(self):
        self._playing = False
        self._is_tracking = False
        for h in (self.bf, self.gf):
            if h is not None:
                try: h.close()
                except: pass
        self.root.destroy()


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 2 — NEURON LABELER
# ══════════════════════════════════════════════════════════════════════════════

class NeuronLabelerWindow:
    """
    For each event, shows the GCaMP channel over a ±EVENT_WINDOW frame range.
    Keys 1-5 select the active neuron. Clicking places/corrects that neuron.
    Positions carry forward from the last clicked frame until corrected.
    Rename neurons by double-clicking the neuron button label.
    """

    def __init__(self, root, file_pairs, events):
        self.root       = root
        self.file_pairs = file_pairs   # list of all pair dicts
        self.events     = events       # list of {"frame": int, "note": str, "fileidx": int}

        self.root.title("Stage 2 — Neuron Labeler")
        self.root.configure(bg="#1a1a2e")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # HDF5 handles — opened on demand per event, switched as needed
        self.bf  = None
        self.gf  = None
        self._open_fidx = -1
        self._demo_mode = False

        # ── event navigation ──────────────────────────────────────────────────
        self.event_idx     = 0
        self.current_frame = 0   # absolute frame index in the file

        # ── neuron state ──────────────────────────────────────────────────────
        # neuron_pos[event_idx][neuron_idx][frame] = (row, col)
        # We store only explicitly clicked frames; carry-forward is computed
        # on the fly in _get_neuron_pos_for_frame.
        self.neuron_pos    = [{} for _ in events]   # per-event dict of dicts
        self.active_neuron = 0   # 0-based index of the selected neuron slot
        self.neuron_names  = list(NEURON_NAMES)

        # Single-frame GCaMP cache
        self._cache_key  = -1
        self._cache_img  = None

        self._build_ui()
        self._load_event(0)

        # Key bindings: 1-5 select neuron, arrows navigate, space = next event
        for i in range(5):
            self.root.bind(str(i + 1), lambda e, n=i: self._select_neuron(n))
        self.root.bind("<Left>",  lambda e: self._step(-1))
        self.root.bind("<Right>", lambda e: self._step(1))
        self.root.bind("<space>", lambda e: self._next_event())

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        BG     = "#1a1a2e"
        PANEL  = "#16213e"
        ACCENT = "#e94560"
        TEXT   = "#eaeaea"
        MUTED  = "#8892a4"
        BTN_BG = "#0f3460"
        BTN_A  = "#e94560"

        def btn(parent, label, cmd, highlight=False, side="left", padx=4):
            b = tk.Button(parent, text=label, command=cmd,
                          bg=ACCENT if highlight else BTN_BG,
                          fg=TEXT, relief="flat",
                          font=("Courier New", 9, "bold"),
                          padx=8, pady=4,
                          activebackground=BTN_A, cursor="hand2")
            b.pack(side=side, padx=padx)
            return b

        # ── top bar ───────────────────────────────────────────────────────────
        top = tk.Frame(self.root, bg=BG, pady=6)
        top.pack(fill="x", padx=10)

        tk.Label(top, text="NEURON LABELER",
                 font=("Courier New", 13, "bold"),
                 fg=ACCENT, bg=BG).pack(side="left")

        self.event_label = tk.Label(top, text="",
                                    font=("Courier New", 10), fg=TEXT, bg=BG)
        self.event_label.pack(side="left", padx=20)

        self.status_var = tk.StringVar(value="")
        tk.Label(top, textvariable=self.status_var,
                 font=("Courier New", 9), fg=MUTED, bg=BG).pack(side="right")

        # ── neuron selector buttons (1–5) ─────────────────────────────────────
        nbar = tk.Frame(self.root, bg=PANEL, pady=6)
        nbar.pack(fill="x", padx=10, pady=(0, 2))

        tk.Label(nbar, text="Active neuron (keys 1–5):",
                 font=("Courier New", 9), fg=MUTED, bg=PANEL
                 ).pack(side="left", padx=(10, 8))

        self._neuron_btns = []
        for i in range(5):
            b = tk.Button(
                nbar, text=f"{i+1}: {self.neuron_names[i]}",
                bg=NEURON_COLORS[i], fg="#111111",
                relief="flat", font=("Courier New", 9, "bold"),
                padx=8, pady=3, cursor="hand2",
                command=lambda n=i: self._select_neuron(n)
            )
            b.pack(side="left", padx=3)
            # Double-click to rename
            b.bind("<Double-Button-1>", lambda e, n=i: self._rename_neuron(n))
            self._neuron_btns.append(b)

        tk.Label(nbar, text="(double-click to rename)",
                 font=("Courier New", 7), fg=MUTED, bg=PANEL
                 ).pack(side="left", padx=8)

        # ── GCaMP image canvas ────────────────────────────────────────────────
        self.fig      = Figure(figsize=(7, 6), facecolor="#0d0d1a")
        self.ax       = self.fig.add_subplot(1, 1, 1)
        self.ax.set_facecolor("#0d0d1a")
        self.ax.tick_params(colors="#8892a4")
        for spine in self.ax.spines.values():
            spine.set_edgecolor("#333355")
        self.fig.set_tight_layout({"pad": 1.5})   # auto-adjusts on every resize

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10)
        self.canvas.mpl_connect("button_press_event", self._on_image_click)
        self.canvas.mpl_connect("resize_event",
                                lambda _e: self.canvas.draw_idle())

        # ── slider ────────────────────────────────────────────────────────────
        sf = tk.Frame(self.root, bg=PANEL, pady=6)
        sf.pack(fill="x", padx=10, pady=(2, 2))

        tk.Button(sf, text="◀", command=lambda: self._step(-1),
                  bg=BTN_BG, fg=TEXT, relief="flat",
                  font=("Courier New", 10, "bold"),
                  activebackground=BTN_A, cursor="hand2").pack(side="left", padx=4)

        self.frame_var = tk.IntVar(value=0)
        self.slider = ttk.Scale(sf, from_=0, to=1,
                                variable=self.frame_var, orient="horizontal",
                                command=self._on_slider)
        self.slider.pack(side="left", fill="x", expand=True, padx=6)

        tk.Button(sf, text="▶", command=lambda: self._step(1),
                  bg=BTN_BG, fg=TEXT, relief="flat",
                  font=("Courier New", 10, "bold"),
                  activebackground=BTN_A, cursor="hand2").pack(side="left", padx=4)

        self.frame_label = tk.Label(sf, text="", width=20,
                                    font=("Courier New", 9), fg=TEXT, bg=PANEL)
        self.frame_label.pack(side="left", padx=8)

        # ── bottom bar ────────────────────────────────────────────────────────
        bot = tk.Frame(self.root, bg=PANEL, pady=6)
        bot.pack(fill="x", padx=10, pady=(0, 6))

        btn(bot, "◀ Prev Event",  self._prev_event,  padx=6)
        btn(bot, "▶ Next Event (Space)", self._next_event, highlight=True, padx=8)
        btn(bot, "💾 Save This Event", self._save_event, padx=6)
        btn(bot, "💾 Save All",        self._save_all,           side="right", padx=12)
        btn(bot, "📊 Export Brightness", self._compute_and_export, side="right", padx=6)
        btn(bot, "x Clear Neuron Here", self._clear_neuron_at_frame, padx=6)

        # ttk styling
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Horizontal.TScale",
                        background=PANEL, troughcolor=BTN_BG)

    # ── file switching ────────────────────────────────────────────────────────

    def _ensure_file_open(self, fidx):
        """Open behavior + GCaMP HDF5 for fidx if not already open."""
        if self._open_fidx == fidx:
            return
        for h in (self.bf, self.gf):
            if h is not None:
                try: h.close()
                except: pass
        self.bf = self.gf = None

        pair = next((p for p in self.file_pairs if p["idx"] == fidx), None)
        if pair is None or not os.path.exists(pair["b_path"]):
            self._demo_mode = True
            self._open_fidx = fidx
            return
        try:
            self.bf = h5py.File(pair["b_path"],  'r')
            self.gf = h5py.File(pair["gc_path"], 'r')
            self._demo_mode = False
        except Exception as e:
            messagebox.showerror("File error", str(e))
            self._demo_mode = True
        self._open_fidx = fidx

    def _current_pair(self):
        """Return the pair dict for the currently open file."""
        return next((p for p in self.file_pairs if p["idx"] == self._open_fidx), self.file_pairs[0])

    # ── event navigation ──────────────────────────────────────────────────────

    def _load_event(self, event_idx):
        """Switch to a different event window."""
        if event_idx < 0 or event_idx >= len(self.events):
            return
        self.event_idx = event_idx
        ev = self.events[event_idx]

        fidx = ev.get("fileidx", 0)
        self._ensure_file_open(fidx)

        n          = self._current_pair()["n_frames"]
        self._win_start = max(0, ev["frame"] - EVENT_WINDOW//5)
        self._win_end   = min(n - 1, ev["frame"] + EVENT_WINDOW)
        self._event_frame = ev["frame"]   # absolute centre frame

        self.slider.configure(from_=self._win_start, to=self._win_end)

        note_str = ('  "' + ev["note"] + '"') if ev["note"] else ""
        self.event_label.config(
            text=f"Event {event_idx + 1}/{len(self.events)}"
                 f"  —  frame {ev['frame']}{note_str}"
        )

        # Ensure neuron_pos dict exists for this event
        if not self.neuron_pos[event_idx]:
            self.neuron_pos[event_idx] = {i: {} for i in range(5)}

        self._cache_key = -1   # invalidate cache on event switch
        self._show_frame(ev["frame"])
        self._set_status(
            f"Event {event_idx + 1}: frames {self._win_start}–{self._win_end}. "
            "Click neurons, use keys 1–5 to switch slot."
        )

    def _prev_event(self):
        self._load_event(self.event_idx - 1)

    def _next_event(self):
        if self.event_idx < len(self.events) - 1:
            self._load_event(self.event_idx + 1)
        else:
            self._set_status("✓ All events done. Use Save All to write results.")

    # ── frame display ─────────────────────────────────────────────────────────

    def _get_gcamp(self, frame):
        """Return cached (or freshly loaded) GCaMP uint8 array for `frame`."""
        if self._cache_key == frame and self._cache_img is not None:
            return self._cache_img
        if self._demo_mode:
            img = self._synthetic_gcamp(frame)
        else:
            img = load_gcamp_raw(self.gf, self._open_fidx, frame)
        self._cache_key = frame
        self._cache_img = img
        return img

    @staticmethod
    def _synthetic_gcamp(frame):
        rng = np.random.default_rng(frame)
        h, w = 512, 512
        img  = rng.integers(5, 30, (h, w), dtype=np.uint8)
        for _ in range(3):
            r = rng.integers(50, h - 50)
            c = rng.integers(50, w - 50)
            rr, cc = np.ogrid[:h, :w]
            img = img + (60 * np.exp(-((rr-r)**2+(cc-c)**2)/300)).astype(np.uint8)
        return np.clip(img, 0, 255).astype(np.uint8)

    def _show_frame(self, frame):
        self.current_frame = frame
        self.frame_var.set(frame)
        rel = frame - self._event_frame
        self.frame_label.config(
            text=f"Frame {frame}  ({rel:+d} from event)"
        )

        img = self._get_gcamp(frame)
        self.ax.cla()
        self.ax.imshow(img, cmap="gray", vmin=0, vmax=255)

        # Draw a vertical reference line at the event centre
        h, w = img.shape[:2]
        self.ax.axvline(x=w // 2, color="#ffffff", alpha=0.05, linewidth=0.5)

        # Draw all neuron positions at this frame (carry-forward)
        for n_idx in range(5):
            pos = self._get_neuron_pos_for_frame(n_idx, frame)
            if pos is not None:
                r, c = pos
                color   = NEURON_COLORS[n_idx]
                is_active = (n_idx == self.active_neuron)
                marker    = "o" if is_active else "."
                size      = 40  if is_active else 30
                self.ax.scatter(c, r, c=color, s=size, marker=marker,
                                zorder=5, linewidths=1,
                                edgecolors="white" if is_active else color,
                                alpha=0.2)
                # self.ax.annotate(
                #     f"{n_idx+1}:{self.neuron_names[n_idx]}",
                #     (c, r), xytext=(5, -10), textcoords="offset points",
                #     color=color, fontsize=7, fontfamily="Courier New"
                # )

        fidx = self._open_fidx
        self.ax.set_title(
            f"GCaMP  file [{fidx:03d}]  frame {frame}  "
            f"  active: {self.active_neuron+1} — {self.neuron_names[self.active_neuron]}",
            color="#eaeaea", fontsize=9, fontfamily="Courier New"
        )
        self.ax.set_facecolor("#0d0d1a")
        self.ax.tick_params(colors="#8892a4")
        self.canvas.draw_idle()

    # ── neuron positions ──────────────────────────────────────────────────────

    def _get_neuron_pos_for_frame(self, n_idx, frame):
        """
        Return the position (row, col) for neuron `n_idx` at `frame`.
        Uses carry-forward: returns the most recent clicked position at or
        before `frame` within the current event window.
        Returns None if the neuron has never been clicked in this window.
        """
        clicks = self.neuron_pos[self.event_idx].get(n_idx, {})
        # Find the latest clicked frame <= current frame
        candidates = [f for f in clicks if f <= frame]
        if not candidates:
            return None
        return clicks[max(candidates)]

    def _on_image_click(self, event):
        if event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        row = int(round(event.ydata))
        col = int(round(event.xdata))
        frame = self.current_frame

        # Store click for the active neuron at this frame
        if self.active_neuron not in self.neuron_pos[self.event_idx]:
            self.neuron_pos[self.event_idx][self.active_neuron] = {}
        self.neuron_pos[self.event_idx][self.active_neuron][frame] = (row, col)

        self._set_status(
            f"Neuron {self.active_neuron+1} ({self.neuron_names[self.active_neuron]})"
            f" → row={row}, col={col}  frame {frame}"
        )
        self._show_frame(frame)

    def _clear_neuron_at_frame(self):
        """Remove the clicked position for the active neuron at the current frame only"""
        frame = self.current_frame
        clicks = self.neuron_pos[self.event_idx].get(self.active_neuron, {})
        if frame in clicks:
            del clicks[frame]
            self.status(
                f"Cleared neuron {self.active_neuron} at frame {frame}"
            )
        else:
            self.set_status(
                f"No click to clear for Neuron {self.active_neuron + 1} at frame {frame}"
            )

    # ── neuron selection & naming ─────────────────────────────────────────────

    def _select_neuron(self, n_idx):
        if n_idx >= 5:
            return
        self.active_neuron = n_idx
        # Highlight active button with a border
        for i, b in enumerate(self._neuron_btns):
            relief = "solid" if i == n_idx else "flat"
            b.config(relief=relief)
        self._set_status(
            f"Active neuron: {n_idx+1} — {self.neuron_names[n_idx]}"
        )
        self._show_frame(self.current_frame)

    def _rename_neuron(self, n_idx):
        new_name = simpledialog.askstring(
            "Rename neuron",
            f"New name for slot {n_idx + 1}:",
            initialvalue=self.neuron_names[n_idx],
            parent=self.root
        )
        if new_name:
            self.neuron_names[n_idx] = new_name.strip()
            self._neuron_btns[n_idx].config(
                text=f"{n_idx+1}: {self.neuron_names[n_idx]}"
            )
            self._show_frame(self.current_frame)

    # ── slider / step ─────────────────────────────────────────────────────────

    def _on_slider(self, val):
        frame = int(float(val))
        if frame != self.current_frame:
            self._show_frame(frame)

    def _step(self, delta):
        frame = max(self._win_start,
                    min(self._win_end, self.current_frame + delta))
        self.frame_var.set(frame)
        self._show_frame(frame)

    # ── save ──────────────────────────────────────────────────────────────────

    def _save_event(self):
        self._write_event(self.event_idx)
        self._set_status(f"✓ Saved event {self.event_idx + 1}")

    def _save_all(self):
        for i in range(len(self.events)):
            self._write_event(i)
        self._set_status(f"✓ Saved all {len(self.events)} events.")
        messagebox.showinfo(
            "Saved",
            f"Saved {len(self.events)} event files in:\n{os.path.abspath('.')}"
        )

    def _write_event(self, event_idx):
        """
        Write neuron positions for one event window to a .npz file.
        Only frames where a position was explicitly clicked are saved.
        """
        ev   = self.events[event_idx]
        fidx = ev.get("fileidx", self._open_fidx)
        out  = os.path.join(B_FP, f"neurons_{fidx:03d}_frame{ev['frame']:06d}.npz")

        clicks_for_event = self.neuron_pos[event_idx]

        # Collect only explicitly annotated frames across all neurons
        annotated_frames = sorted({
            frame
            for n_idx in range(5)
            for frame in clicks_for_event.get(n_idx, {})
        })

        frames  = np.array(annotated_frames, dtype=np.int64)
        n_ann   = len(frames)
        pos_arr = np.full((5, n_ann, 2), np.nan)

        for n_idx in range(5):
            clicks = clicks_for_event.get(n_idx, {})
            for fi, frame in enumerate(frames):
                if frame in clicks:
                    pos_arr[n_idx, fi, 0] = clicks[frame][0]   # row
                    pos_arr[n_idx, fi, 1] = clicks[frame][1]   # col

        alignment    = ALIGNMENT_DICT[fidx]
        gcamp_frames = np.array([alignment[int(f)] for f in frames], dtype=np.int64)

        np.savez(
            out,
            fileidx      = fidx,
            event_frame  = ev["frame"],
            note         = ev.get("note", ""),
            frames       = frames,          # behavior frame indices
            gcamp_frames = gcamp_frames,    # corresponding GCaMP frame indices
            positions    = pos_arr,         # (5, n_annotated_frames, 2) — NaN if not clicked
            neuron_names = np.array(self.neuron_names)
        )

    # ── brightness export ─────────────────────────────────────────────────────

    def _compute_and_export(self):
        """
        Find all neurons_*.npz files saved by _write_event, compute GCaMP
        brightness at every annotated position using ThreadPoolExecutor
        (one thread per file → overlapping HDF5 I/O), then write the
        Excel workbook and metadata text file.
        """
        neuron_files = sorted(glob.glob(os.path.join(B_FP, "neurons_*.npz")))
        if not neuron_files:
            messagebox.showwarning(
                "No data", "No neurons_*.npz files found. Save events first.")
            return

        gc_file_list = sorted(glob.glob(os.path.join(GC_FP, "*.h5")))
        n = len(neuron_files)
        self._set_status(f"Computing brightness for {n} event file(s)…")
        self.root.update_idletasks()

        max_workers = min(4, n)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_process_one_neuron_file, nf, gc_file_list)
                       for nf in neuron_files]
            results = [f.result() for f in futures]

        results = [r for r in results if r is not None]
        if not results:
            messagebox.showerror("Export failed", "No events could be processed.")
            return

        wb   = Workbook()
        ws_b = wb.active
        ws_b.title = "Neuron_brightness"
        ws_p = wb.create_sheet("Neuron Positions")
        metadata_lines = []

        for res in results:
            n_neu        = res['num_neurons']
            frames       = res['frames']        # behavior frame indices
            gcamp_frames = res['gcamp_frames']  # GCaMP frame indices
            times        = res['times']
            brt          = res['brightness']    # (n_neurons, n_frames)
            pos          = res['positions']     # (n_neurons, n_frames, 2)

            # brightness sheet
            ws_b.append(['Behavior Frame', 'GCaMP Frame', 'seconds to event'] +
                        [f'Neuron {i}' for i in range(n_neu)])
            for ci in range(len(frames)):
                ws_b.append([int(frames[ci]), int(gcamp_frames[ci]), float(times[ci])] +
                             brt[:, ci].tolist())
            ws_b.append([])

            # positions sheet
            ws_p.append(['Behavior Frame', 'GCaMP Frame', 'seconds to event'] +
                        [f'Neuron {i} X' for i in range(n_neu)] +
                        [f'Neuron {i} Y' for i in range(n_neu)])
            for ci in range(len(frames)):
                ws_p.append([int(frames[ci]), int(gcamp_frames[ci]), float(times[ci])] +
                             pos[:, ci, 0].tolist() +
                             pos[:, ci, 1].tolist())
            ws_p.append([])

            metadata_lines += [
                f"File: {res['file']}",
                f"  File Index: {res['fileidx']}",
                f"  Event Frame: {res['event_frame']}",
                f"  Note: {res['note']}",
                '',
            ]

        wb.save(EXPORT_EXCEL)
        with open(EXPORT_METADATA, 'w') as mf:
            mf.write('\n'.join(metadata_lines))

        self._set_status(
            f"✓ Exported {len(results)} event(s) → {EXPORT_EXCEL}")
        messagebox.showinfo(
            "Export complete",
            f"Saved {len(results)} event(s):\n{EXPORT_EXCEL}\n{EXPORT_METADATA}")

    # ── helpers ───────────────────────────────────────────────────────────────

    def _set_status(self, msg):
        self.status_var.set(msg)

    def _on_close(self):
        for h in (self.bf, self.gf):
            if h:
                try: h.close()
                except: pass
        self.root.destroy()


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

class StartupDialog:
    """
    Modal dialog shown before the main window.
    User browses for the behavior and GCaMP folders, then clicks Launch.
    """

    # Pre-fill with the last-used paths (persisted next to the script)
    _PREFS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               ".tracker_paths.json")

    def __init__(self, root):
        self.root = root
        self.root.title("Select Data Folders")
        self.root.configure(bg="#1a1a2e")
        self.root.resizable(False, False)
        self.confirmed = False

        BG     = "#1a1a2e"
        PANEL  = "#16213e"
        ACCENT = "#e94560"
        TEXT   = "#eaeaea"
        MUTED  = "#8892a4"
        BTN_BG = "#0f3460"

        # ── load saved paths ─────────────────────────────────────────────────
        saved = {}
        try:
            with open(self._PREFS_FILE) as f:
                saved = json.load(f)
        except Exception:
            pass

        # ── title ─────────────────────────────────────────────────────────────
        tk.Label(root, text="WORM TRACKER",
                 font=("Courier New", 14, "bold"),
                 fg=ACCENT, bg=BG).pack(pady=(18, 4))
        tk.Label(root, text="Select the behavior and GCaMP data folders to begin.",
                 font=("Courier New", 9), fg=MUTED, bg=BG).pack(pady=(0, 14))

        # ── folder rows ───────────────────────────────────────────────────────
        frame = tk.Frame(root, bg=BG)
        frame.pack(padx=30, fill="x")

        self.b_var  = tk.StringVar(value=saved.get("b_fp",  ""))
        self.gc_var = tk.StringVar(value=saved.get("gc_fp", ""))

        for label_text, var in [("Behavior folder:", self.b_var),
                                 ("GCaMP folder:",    self.gc_var)]:
            row = tk.Frame(frame, bg=BG)
            row.pack(fill="x", pady=5)
            tk.Label(row, text=label_text, width=16, anchor="w",
                     font=("Courier New", 9), fg=TEXT, bg=BG).pack(side="left")
            tk.Entry(row, textvariable=var, width=52,
                     bg=PANEL, fg=TEXT, insertbackground=TEXT,
                     relief="flat", font=("Courier New", 9)).pack(side="left", padx=(4, 6))
            tk.Button(row, text="Browse…",
                      command=lambda v=var: self._browse(v),
                      bg=BTN_BG, fg=TEXT, relief="flat",
                      font=("Courier New", 9), padx=6, cursor="hand2"
                      ).pack(side="left")

        # ── status label (shows file counts after validation) ─────────────────
        self.status_var = tk.StringVar(value="")
        tk.Label(root, textvariable=self.status_var,
                 font=("Courier New", 8), fg=MUTED, bg=BG).pack(pady=(8, 0))

        # ── buttons ───────────────────────────────────────────────────────────
        btn_row = tk.Frame(root, bg=BG)
        btn_row.pack(pady=16)
        tk.Button(btn_row, text="Launch",
                  command=self._launch,
                  bg=ACCENT, fg=TEXT, relief="flat",
                  font=("Courier New", 10, "bold"),
                  padx=20, pady=6, cursor="hand2").pack(side="left", padx=8)
        tk.Button(btn_row, text="Cancel",
                  command=root.destroy,
                  bg=BTN_BG, fg=TEXT, relief="flat",
                  font=("Courier New", 10),
                  padx=20, pady=6, cursor="hand2").pack(side="left", padx=8)

        root.update_idletasks()
        self._center(root)

    def _browse(self, var):
        from tkinter import filedialog
        path = filedialog.askdirectory(initialdir=var.get() or os.path.expanduser("~"))
        if path:
            var.set(path)

    def _launch(self):
        b_fp  = self.b_var.get().strip()
        gc_fp = self.gc_var.get().strip()

        if not os.path.isdir(b_fp):
            self.status_var.set("⚠  Behavior folder not found.")
            return
        if not os.path.isdir(gc_fp):
            self.status_var.set("⚠  GCaMP folder not found.")
            return

        n_b  = len(glob.glob(os.path.join(b_fp,  "*.h5")))
        n_gc = len(glob.glob(os.path.join(gc_fp, "*.h5")))
        if n_b == 0 or n_gc == 0:
            self.status_var.set(
                f"⚠  Found {n_b} behavior file(s) and {n_gc} GCaMP file(s) — need at least 1 each.")
            return

        self.status_var.set(f"Found {n_b} behavior / {n_gc} GCaMP files. Loading alignments…")
        self.root.update_idletasks()

        try:
            _initialize_paths(b_fp, gc_fp)
        except Exception as e:
            self.status_var.set(f"⚠  Error loading alignments: {e}")
            return

        # persist paths for next run
        try:
            with open(self._PREFS_FILE, "w") as f:
                json.dump({"b_fp": b_fp, "gc_fp": gc_fp}, f)
        except Exception:
            pass

        self.confirmed = True
        self.root.destroy()

    @staticmethod
    def _center(win):
        win.update_idletasks()
        w = win.winfo_reqwidth()
        h = win.winfo_reqheight()
        sw = win.winfo_screenwidth()
        sh = win.winfo_screenheight()
        win.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")


def main():
    # ── startup dialog ────────────────────────────────────────────────────────
    dialog_root = tk.Tk()
    dialog = StartupDialog(dialog_root)
    dialog_root.mainloop()

    if not dialog.confirmed:
        return   # user cancelled

    # ── main window ───────────────────────────────────────────────────────────
    root = tk.Tk()
    root.geometry("900x780")
    root.minsize(700, 500)
    EventPickerWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()