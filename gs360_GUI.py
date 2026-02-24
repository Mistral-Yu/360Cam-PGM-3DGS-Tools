#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI preview and execution tool for gs360_360PerspCut."""

import argparse
import csv
import copy
import importlib
import itertools
import math
import pathlib
import os
import shutil
from collections import defaultdict
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox, ttk
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    from PIL import Image, ImageDraw, ImageTk
except ImportError as exc:  # pragma: no cover - environment guard
    print("[ERR] Pillow (PIL) is required: pip install Pillow", file=sys.stderr)
    raise SystemExit(1) from exc

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - environment guard
    print("[ERR] NumPy is required: pip install numpy", file=sys.stderr)
    raise SystemExit(1) from exc

SCRIPT_DIR = Path(__file__).resolve().parent
CLI_TOOLS_DIR = SCRIPT_DIR / "cli_tools"
if str(CLI_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(CLI_TOOLS_DIR))

cutter = importlib.import_module("gs360_360PerspCut")

COLOR_CYCLE = [
    "#ff6b6b",
    "#4ecdc4",
    "#ffd166",
    "#5a189a",
    "#118ab2",
    "#f9844a",
    "#06d6a0",
    "#ef476f",
    "#073b4c",
    "#9b5de5",
]

PRESET_CHOICES = ["default", "fisheyelike", "full360coverage", "2views", "evenMinus30", "evenPlus30", "fisheyeXY"]

HUMAN_MODE_CHOICES = ["mask", "alpha", "inpaint"]
HUMAN_TARGET_CHOICES = ["person", "bicycle", "car", "animal"]

MSXML_PRESET_CHOICES = [
    "default",
    "fisheyelike",
    "full360coverage",
    "2views",
    "evenMinus30",
    "evenPlus30",
    "cube105",
]
MSXML_FORMAT_CHOICES = [
    "Metashape",
    "Metashape-Multi-Camera-System",
    "transforms",
    "COLMAP",
    "RealityScan",
    "all",
]
MSXML_FORMAT_TO_CLI = {
    "Metashape": "metashape",
    "Metashape-Multi-Camera-System": "metashape-multi-camera-system",
    "transforms": "transforms",
    "COLMAP": "colmap",
    "RealityScan": "realityscan",
    "all": "all",
}

DEFAULT_SELECTOR_CSV_NAME = "selected_image_list.csv"

PLY_VIEW_CANVAS_WIDTH = 840
PLY_VIEW_CANVAS_HEIGHT = 840
PLY_VIEW_MAX_POINTS = 500_000
PLY_VIEW_INTERACTIVE_MAX_POINTS = 100_000
PLY_INTERACTION_SETTLE_DELAY_MS = 350
PLY_VIEW_MAX_ZOOM = 24.0
PLY_PROPERTY_TYPES = {
    "char": ("b", 1, "i1"),
    "uchar": ("B", 1, "u1"),
    "int8": ("b", 1, "i1"),
    "uint8": ("B", 1, "u1"),
    "short": ("h", 2, "<i2"),
    "ushort": ("H", 2, "<u2"),
    "int16": ("h", 2, "<i2"),
    "uint16": ("H", 2, "<u2"),
    "int": ("i", 4, "<i4"),
    "int32": ("i", 4, "<i4"),
    "uint": ("I", 4, "<u4"),
    "uint32": ("I", 4, "<u4"),
    "float": ("f", 4, "<f4"),
    "float32": ("f", 4, "<f4"),
    "double": ("d", 8, "<f8"),
    "float64": ("d", 8, "<f8"),
}


class ToolTip:
    """Simple tooltip that appears near widgets on hover."""

    def __init__(self, widget: tk.Widget, text: str, delay: int = 400):
        self.widget = widget
        self.text = text
        self.delay = delay
        self.tipwindow: Optional[tk.Toplevel] = None
        self._after_id: Optional[str] = None
        widget.bind("<Enter>", self._on_enter, add="+")
        widget.bind("<Leave>", self._on_leave, add="+")
        widget.bind("<ButtonPress>", self._on_leave, add="+")

    def _on_enter(self, _event=None) -> None:
        self._schedule()

    def _schedule(self) -> None:
        self._cancel()
        self._after_id = self.widget.after(self.delay, self._show)

    def _on_leave(self, _event=None) -> None:
        self._cancel()
        self._hide()

    def _cancel(self) -> None:
        if self._after_id is not None:
            try:
                self.widget.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

    def _show(self) -> None:
        if self.tipwindow or not self.text:
            return
        x, y = self.widget.winfo_pointerxy()
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x + 12}+{y + 12}")
        label = tk.Label(
            tw,
            text=self.text,
            justify=tk.LEFT,
            background="#ffffe0",
            relief=tk.SOLID,
            borderwidth=1,
            font=("TkDefaultFont", 9),
            wraplength=320,
        )
        label.pack(ipadx=6, ipady=4)

    def _hide(self) -> None:
        if self.tipwindow is not None:
            try:
                self.tipwindow.destroy()
            except Exception:
                pass
            self.tipwindow = None

HELP_TEXT = """
- preset: Output preset (default / fisheyelike / 2views / evenMinus30 / evenPlus30 / fisheyeXY)
- count: Horizontal divisions (360 deg / count gives yaw step)
- addcam: Extra cameras, e.g. B, D:U15, H:D20
- addcam delta (deg): Default delta when U/D omit a value
- delcam: Remove baseline cameras, e.g. B,D
- setcam: Override/adjust cameras, e.g. A=U30, B:-5, A_U=10 (upper), A_D:-5 (lower)
- Size(px): Output square size per view
- HFOV (deg): Horizontal FOV (overrides focal length)
- Focal(mm): Focal length when HFOV is not set
- Sensor(mm): Sensor width/height (e.g. 36 24 or 36x24)
- Add top: Include a cube-map style top view (pitch +90 deg)
- Add bottom: Include a cube-map style bottom view (pitch -90 deg)
- Video input: Use Browse... to select a video file (mp4/mov/etc.); choose preview FPS and ensure ffmpeg path is configured.

Update: Refresh preview only with the current parameters
Run Export: Execute FFmpeg jobs and write outputs (with confirmation)
"""

FIELD_HELP_TEXT = {
    "preset": "Choose a preset layout for virtual cameras. Defaults to an 8-view configuration.",
    "addcam": "Add extra cameras using codes like B, D:U20, H:D to offset pitch or duplicate slots.",
    "addcam_deg": "Default pitch delta (degrees) when U/D is specified without a value in addcam/setcam.",
    "delcam": "Remove baseline cameras by letter, e.g. B,D to skip those outputs.",
    "setcam": "Override existing camera pitch. Use A=U20, C:+15, D:-5, or target extras with A_U=10 / A_D:-5.",
    "size": "Square output size (pixels) for each perspective view.",
    "focal_mm": "Virtual focal length (mm). Used when HFOV is blank.",
    "hfov": "Horizontal field-of-view override (degrees). Takes priority over focal length when provided.",
    "sensor_mm": "Sensor width/height (mm). Example: 36 24 or 36x24.",
    "count": "Number of evenly spaced horizontal cameras (360° / count determines yaw step).",
    "add_top": "Enable an additional top view at pitch +90°.",
    "add_bottom": "Enable an additional bottom view at pitch -90°.",
    "input_path": "Select an image folder or Browse to choose a video file (mp4/mov/etc.). For videos, set preview FPS and ensure ffmpeg is configured.",
    "show_seam_overlay": "Overlay a translucent band along the panorama seam to visualise potential stitching artifacts.",
    "ffmpeg": "Path to the ffmpeg executable. Leave blank to use the system PATH.",
    "jobs": "Number of parallel ffmpeg processes. 'auto' uses approximately half the CPU cores (video direct export halves this again).",
    "fps": "Frame extraction rate (fps) required when processing a video source.",
    "start": "Optional start time (seconds) when exporting directly from video using FPS.",
    "end": "Optional end time (seconds) when exporting directly from video using FPS.",
    "keep_rec709": "Convert Rec.709 to sRGB (unchecked = keep Rec.709).",
    "jpeg_quality_95": "When checked, save JPG outputs with approximately 95% quality rather than maximum quality.",
}

# Override legacy text with clarified wording.
FIELD_HELP_TEXT.update(
    {
        "count": "Number of evenly spaced horizontal cameras (360 deg / count determines yaw step).",
        "add_top": "Enable an additional top view at pitch +90 deg.",
        "add_bottom": "Enable an additional bottom view at pitch -90 deg.",
        "keep_rec709": "Convert Rec.709 to sRGB (uncheck to keep Rec.709).",
    }
)


def parse_arguments() -> argparse.Namespace:
    """Build the preview CLI parser and return parsed arguments."""

    parser = cutter.create_arg_parser()
    for action in parser._actions:
        if action.dest == "input_dir":
            action.required = False
    parser.description = "Visualize and execute gs360_360PerspCut camera layouts."
    parser.add_argument(
        "--image",
        help="Specific panorama to preview. Defaults to the first supported file in the selected directory.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Optional fixed scale factor (0 < scale <= 1). Overrides --max-width/--max-height.",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=1600,
        help="Maximum display width when --scale is not supplied.",
    )
    parser.add_argument(
        "--max-height",
        type=int,
        default=900,
        help="Maximum display height when --scale is not supplied.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=90,
        help="Edge sampling count per side when tracing polygons (higher = smoother).",
    )
    parser.add_argument(
        "--hide-labels",
        action="store_true",
        help="Hide overlay labels if you prefer a clean view.",
    )
    return parser.parse_args()


def normalize_vector(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """Normalize a 3D vector."""

    length = math.sqrt(x * x + y * y + z * z)
    if length <= 0.0:
        return 0.0, 0.0, 1.0
    return x / length, y / length, z / length


def rotate_pitch(vec: Sequence[float], pitch_rad: float) -> Tuple[float, float, float]:
    """Rotate vector around the X-axis by pitch (radians)."""

    x, y, z = vec
    cos_p = math.cos(pitch_rad)
    sin_p = math.sin(pitch_rad)
    return (
        x,
        cos_p * y + sin_p * z,
        -sin_p * y + cos_p * z,
    )


def rotate_yaw(vec: Sequence[float], yaw_rad: float) -> Tuple[float, float, float]:
    """Rotate vector around the Y-axis by yaw (radians)."""

    x, y, z = vec
    cos_y = math.cos(yaw_rad)
    sin_y = math.sin(yaw_rad)
    return (
        cos_y * x + sin_y * z,
        y,
        -sin_y * x + cos_y * z,
    )


def direction_from_uv(
    u: float,
    v: float,
    hfov_rad: float,
    vfov_rad: float,
    yaw_rad: float,
    pitch_rad: float,
) -> Tuple[float, float]:
    """Convert normalized viewport coordinates to longitude/latitude in radians."""

    x = math.tan(hfov_rad / 2.0) * u
    y = math.tan(vfov_rad / 2.0) * (-v)
    z = 1.0
    vec = normalize_vector(x, y, z)
    vec = rotate_pitch(vec, pitch_rad)
    vec = rotate_yaw(vec, yaw_rad)
    lon = math.atan2(vec[0], vec[2])
    lat = math.asin(max(-1.0, min(1.0, vec[1])))
    return lon, lat


def unwrap_angles(values: Iterable[float]) -> List[float]:
    """Unwrap angle list to maintain continuity across the 2*pi seam."""

    iterator = iter(values)
    try:
        first = next(iterator)
    except StopIteration:
        return []
    unwrapped = [first]
    prev = first
    for angle in iterator:
        candidate = angle
        while candidate - prev > math.pi:
            candidate -= 2.0 * math.pi
        while candidate - prev < -math.pi:
            candidate += 2.0 * math.pi
        unwrapped.append(candidate)
        prev = candidate
    return unwrapped


def lonlat_to_xy(lon: float, lat: float, width: int, height: int) -> Tuple[float, float]:
    """Convert spherical lon/lat (radians) into equirectangular pixel coordinates."""

    x = ((lon / (2.0 * math.pi)) + 0.5) * width
    y = (0.5 - (lat / math.pi)) * height
    return x, y




def sample_view_segments(
    spec: cutter.ViewSpec,
    pano_width: int,
    pano_height: int,
    samples: int,
) -> Tuple[List[List[Tuple[float, float]]], Tuple[float, float]]:
    """Return edge segments and centroid (both in pixel space)."""

    hfov_rad = math.radians(min(max(spec.hfov_deg, 1e-3), 179.9))
    vfov_rad = math.radians(min(max(spec.vfov_deg, 1e-3), 179.9))
    yaw_rad = math.radians(spec.yaw_deg)
    pitch_rad = math.radians(spec.pitch_deg)

    per_side = max(8, samples)

    def sample_edge(
        u_start: float,
        u_end: float,
        v_start: float,
        v_end: float,
        count: int,
    ) -> List[List[Tuple[float, float]]]:
        if count <= 0:
            return []
        lon_samples: List[float] = []
        lat_samples: List[float] = []
        for i in range(count + 1):
            t = i / count
            u = u_start + (u_end - u_start) * t
            v = v_start + (v_end - v_start) * t
            lon, lat = direction_from_uv(u, v, hfov_rad, vfov_rad, yaw_rad, pitch_rad)
            lon_samples.append(lon)
            lat_samples.append(lat)
        lon_samples = unwrap_angles(lon_samples)
        segments: List[List[Tuple[float, float]]] = []
        current: List[Tuple[float, float]] = []
        current_wrap: Optional[int] = None
        for lon, lat in zip(lon_samples, lat_samples):
            x_unwrapped = ((lon / (2.0 * math.pi)) + 0.5) * pano_width
            wrap_index = math.floor(x_unwrapped / pano_width)
            x = x_unwrapped - wrap_index * pano_width
            if x < 0.0:
                x += pano_width
            elif x >= pano_width:
                x -= pano_width
            y = max(0.0, min(float(pano_height), (0.5 - (lat / math.pi)) * pano_height))
            if current_wrap is None:
                current_wrap = wrap_index
            elif wrap_index != current_wrap:
                if current:
                    segments.append(current)
                current = []
                current_wrap = wrap_index
            current.append((x, y))
        if current:
            segments.append(current)
        return segments

    segments: List[List[Tuple[float, float]]] = []
    segments.extend(sample_edge(-1.0, 1.0, -1.0, -1.0, per_side))
    segments.extend(sample_edge(1.0, 1.0, -1.0, 1.0, per_side))
    segments.extend(sample_edge(1.0, -1.0, 1.0, 1.0, per_side))
    segments.extend(sample_edge(-1.0, -1.0, 1.0, -1.0, per_side))

    center_lon, center_lat = direction_from_uv(0.0, 0.0, hfov_rad, vfov_rad, yaw_rad, pitch_rad)
    center_x = ((center_lon / (2.0 * math.pi)) + 0.5) * pano_width
    center_x = center_x % pano_width
    center_y = (0.5 - (center_lat / math.pi)) * pano_height
    center_y = max(0.0, min(float(pano_height), center_y))

    return segments, (center_x, center_y)

def compute_scale(
    width: int,
    height: int,
    scale_override: Optional[float],
    max_width: int,
    max_height: int,
) -> float:
    """Compute the display scale factor."""

    if scale_override:
        if scale_override <= 0:
            raise ValueError("--scale must be > 0")
        return min(1.0, float(scale_override))
    scale_w = max_width / float(width)
    scale_h = max_height / float(height)
    return min(1.0, scale_w, scale_h)


def build_info_lines(specs: Sequence[cutter.ViewSpec]) -> List[str]:
    """Create textual summaries for each view."""

    lines: List[str] = []
    for spec in specs:
        lines.append(
            f"{spec.view_id:>6}: yaw={spec.yaw_deg:+6.1f}°  "
            f"pitch={spec.pitch_deg:+6.1f}°  hfov={spec.hfov_deg:5.1f}°  "
            f"vfov={spec.vfov_deg:5.1f}°"
        )
    return lines


def clone_namespace(ns: argparse.Namespace) -> argparse.Namespace:
    """Return a deep copy of an argparse namespace."""

    return copy.deepcopy(ns)


def ensure_explicit_flags(args: argparse.Namespace) -> None:
    """Ensure *_explicit attributes exist for shared CLI."""

    for attr in ("size", "hfov", "focal_mm"):
        flag = f"{attr}_explicit"
        if not hasattr(args, flag):
            setattr(args, flag, False)


class PreviewApp:
    """Interactive GUI to tweak gs360_360 cuts and visualise overlays."""

    VIDEO_PERSIST_FIELDS = {"fps", "keep_rec709", "start", "end"}
    FIELD_DEFS = [
        {"name": "preset", "label": "Preset", "type": "choice", "choices": PRESET_CHOICES, "width": 14},
        {"name": "addcam", "label": "AddCam", "type": "str", "width": 18},
        {"name": "addcam_deg", "label": "CamDeg", "type": "float", "width": 10},
        {"name": "delcam", "label": "DelCam", "type": "str", "width": 16},
        {"name": "setcam", "label": "SetCam", "type": "str", "width": 18},
        {"name": "size", "label": "Size", "type": "int", "width": 10},
        {"name": "focal_mm", "label": "Focal", "type": "float", "width": 10},
        {"name": "hfov", "label": "HFOV", "type": "float_optional", "width": 10},
        {"name": "sensor_mm", "label": "Sensor", "type": "str", "width": 14},
        {"name": "count", "label": "Count", "type": "int", "width": 8},
        {"name": "ext", "label": "Ext", "type": "choice", "choices": ["jpg", "tif", "png"], "width": 8},
        {
            "name": "jpeg_quality_95",
            "label": "JPEG 95%",
            "type": "bool",
            "align_with": "ext",
            "col_shift": 2,
        },
        {
            "name": "fps",
            "label": "FPS",
            "type": "float_optional",
            "width": 8,
            "video_only": True,
            "align_with": "count",
            "col_shift": 0,
            "row_shift": 2,
            "required_if_video": True,
        },
        {
            "name": "start",
            "label": "Start (s)",
            "type": "float_optional",
            "width": 10,
            "video_only": True,
            "align_with": "count",
            "col_shift": 0,
            "row_shift": 2,
        },
        {
            "name": "end",
            "label": "End (s)",
            "type": "float_optional",
            "width": 10,
            "video_only": True,
            "align_with": "count",
            "col_shift": 0,
            "row_shift": 2,
        },
        {
            "name": "add_top",
            "label": "Add top",
            "type": "bool",
            "align_with": "count",
            "col_shift": 0,
            "row_shift": 1,
        },
        {
            "name": "add_bottom",
            "label": "Add bottom",
            "type": "bool",
            "align_with": "count",
            "col_shift": 2,
            "row_shift": 1,
        },
        {
            "name": "keep_rec709",
            "label": "Convert Rec.709 to sRGB",
            "type": "bool",
            "video_only": True,
            "align_with": "ext",
            "col_shift": 0,
            "row_shift": 2,
        },
        {
            "name": "show_seam_overlay",
            "label": "Show seam hint",
            "type": "bool",
            "align_with": "count",
            "col_shift": 4,
            "row_shift": 1,
        },
    ]

    EXPLICIT_FIELDS = {"size", "hfov", "focal_mm"}

    def __init__(self, args: argparse.Namespace):
        ensure_explicit_flags(args)
        self.original_args = args
        self.current_args = clone_namespace(args)
        if not hasattr(self.current_args, "input_is_video"):
            setattr(self.current_args, "input_is_video", False)
        if not hasattr(self.current_args, "video_bit_depth"):
            setattr(self.current_args, "video_bit_depth", 8)
        if not hasattr(self.current_args, "jpeg_quality_95"):
            setattr(self.current_args, "jpeg_quality_95", False)
        if not hasattr(self.current_args, "show_seam_overlay"):
            setattr(self.current_args, "show_seam_overlay", False)
        self.defaults = clone_namespace(self.current_args)
        ensure_explicit_flags(self.current_args)
        self.form_snapshot: Dict[str, str] = {}
        self.field_vars: Dict[str, tk.Variable] = {}
        self.field_widgets: Dict[str, tk.Widget] = {}
        self.video_only_fields: Set[str] = {
            definition["name"]
            for definition in self.FIELD_DEFS
            if definition.get("video_only")
        }
        self.result: Optional[cutter.BuildResult] = None
        self.matched_specs: List[cutter.ViewSpec] = []
        self.is_executing = False
        self.source_is_video = False
        self._video_preview_signature: Optional[Tuple[pathlib.Path, bool]] = None
        self.video_persist_state: Dict[str, Any] = {"fps": 1.0, "keep_rec709": True, "start": None, "end": None}

        self.scale_override = args.scale
        self.max_width_limit = int(args.max_width)
        self.max_height_limit = int(args.max_height)

        self.in_dir: Optional[pathlib.Path] = None
        self.files: List[pathlib.Path] = []
        self.image_path: Optional[pathlib.Path] = None
        self.out_dir: Optional[pathlib.Path] = None

        self.samples = max(8, int(args.samples))
        self.hide_labels = bool(args.hide_labels)

        self.pano_image: Optional[Image.Image] = None
        self.display_image: Optional[Image.Image] = None
        self.pano_width = 0
        self.pano_height = 0
        self.scale = 1.0
        self.display_width = 640
        self.display_height = 360

        self.root = tk.Tk()
        self.root.withdraw()
        self.base_dir = SCRIPT_DIR
        self.cli_tools_dir = CLI_TOOLS_DIR
        if not self.cli_tools_dir.exists():
            self.cli_tools_dir = self.base_dir
        screen_w = max(1, self.root.winfo_screenwidth())
        screen_h = max(1, self.root.winfo_screenheight())
        self.view_max_width = max(1, min(self.max_width_limit, screen_w // 3))
        self.view_max_width = max(1, int(math.ceil(self.view_max_width * 1.1)))
        self.view_max_height = max(1, min(self.max_height_limit, screen_h // 3))

        self.photo: Optional[ImageTk.PhotoImage] = None
        self.preview_frame: Optional[tk.LabelFrame] = None
        self.controls_frame: Optional[tk.LabelFrame] = None
        self.left_frame: Optional[tk.Frame] = None
        self.canvas: Optional[tk.Canvas] = None
        self.canvas_image_id: Optional[int] = None
        self.canvas_offset_x = 0.0
        self.canvas_offset_y = 0.0
        self.folder_path_var = tk.StringVar()
        self.output_path_var = tk.StringVar()
        self._out_dir_custom = False
        self._tooltips: List[ToolTip] = []
        self.ffmpeg_path_var = tk.StringVar(value=str(getattr(self.current_args, "ffmpeg", "ffmpeg")))
        self.jobs_var = tk.StringVar(value=str(getattr(self.current_args, "jobs", "auto")))
        self.log_text: Optional[tk.Text] = None
        self.help_text: Optional[tk.Text] = None
        self.update_button: Optional[tk.Button] = None
        self.execute_button: Optional[tk.Button] = None
        self.preview_csv_var = tk.StringVar()
        self.preview_csv_entry: Optional[tk.Entry] = None
        self.preview_csv_button: Optional[tk.Button] = None
        self._preview_estimated_frames: Optional[int] = None
        self._video_estimated_cache: Dict[str, Dict[str, Any]] = {}
        self.right_inner: Optional[tk.Frame] = None
        self.notebook: Optional[ttk.Notebook] = None

        self.video_vars: Dict[str, tk.Variable] = {}
        self.video_log: Optional[tk.Text] = None
        self.video_run_button: Optional[tk.Button] = None
        self.video_inspect_button: Optional[tk.Button] = None
        self.fisheye_persp_check: Optional[tk.Checkbutton] = None
        self.fisheye_fov_entry: Optional[tk.Entry] = None
        self.fisheye_focal_entry: Optional[tk.Entry] = None
        self.fisheye_size_entry: Optional[tk.Entry] = None
        self.fisheye_projection_combo: Optional[ttk.Combobox] = None

        self.preview_inspect_button: Optional[tk.Button] = None

        self.selector_vars: Dict[str, tk.Variable] = {}
        self.selector_log: Optional[tk.Text] = None
        self.selector_run_button: Optional[tk.Button] = None
        self.selector_show_score_button: Optional[tk.Button] = None
        self.selector_open_suspects_button: Optional[tk.Button] = None
        self.selector_manual_apply_button: Optional[tk.Button] = None
        self.selector_manual_reset_button: Optional[tk.Button] = None
        self.selector_count_var: Optional[tk.StringVar] = None
        self.selector_summary_label: Optional[tk.Label] = None
        self.selector_score_canvas: Optional[tk.Canvas] = None
        self.selector_last_scores: List[Tuple[int, bool, Optional[float]]] = []
        self.selector_score_entries: List[Dict[str, Any]] = []
        self.selector_score_suspect_positions: Set[int] = set()
        self.selector_score_csv_path_loaded: Optional[Path] = None
        self.selector_score_csv_fieldnames: List[str] = []
        self.selector_score_selected_key: Optional[str] = None
        self.selector_score_zoom = 1.0
        self.selector_score_bar_width = 4.0
        self.selector_score_bar_area_height = 0
        self.selector_score_total_width = 0.0
        self.selector_score_value_min = 0.0
        self.selector_score_value_range = 0.0
        self.selector_preview_panel_window: Optional[tk.Toplevel] = None
        self.selector_preview_panel_image_label: Optional[tk.Label] = None
        self.selector_preview_panel_image_canvas: Optional[tk.Canvas] = None
        self.selector_preview_panel_canvas_image_id: Optional[int] = None
        self.selector_preview_panel_info_label: Optional[tk.Label] = None
        self.selector_preview_panel_status_label: Optional[tk.Label] = None
        self.selector_preview_panel_index_label: Optional[tk.Label] = None
        self.selector_preview_panel_slider: Optional[tk.Scale] = None
        self.selector_preview_panel_slider_var: Optional[tk.IntVar] = None
        self.selector_preview_panel_zoom_reset_button: Optional[tk.Button] = None
        self.selector_preview_panel_zoom_25_button: Optional[tk.Button] = None
        self.selector_preview_panel_zoom_50_button: Optional[tk.Button] = None
        self.selector_preview_panel_zoom_100_button: Optional[tk.Button] = None
        self.selector_preview_panel_close_current_button: Optional[tk.Button] = None
        self.selector_preview_panel_close_all_button: Optional[tk.Button] = None
        self.selector_preview_items: Dict[int, Dict[str, Any]] = {}
        self.selector_preview_panel_active_idx: Optional[int] = None
        self.selector_preview_panel_zoom_ratio = 1.0
        self.selector_auto_fetch_pending = False
        self.selector_csv_auto = True
        self.selector_csv_auto_value = ""
        self._selector_csv_updating = False

        self.human_vars: Dict[str, tk.Variable] = {}
        self.human_log: Optional[tk.Text] = None
        self.human_run_button: Optional[tk.Button] = None

        self.msxml_vars: Dict[str, tk.Variable] = {}
        self.msxml_log: Optional[tk.Text] = None
        self.msxml_run_button: Optional[tk.Button] = None
        self.msxml_stop_button: Optional[tk.Button] = None
        self.msxml_cut_input_entry: Optional[tk.Entry] = None
        self.msxml_cut_out_entry: Optional[tk.Entry] = None
        self.msxml_preset_combo: Optional[ttk.Combobox] = None
        self.msxml_points_entry: Optional[tk.Entry] = None
        self.msxml_points_button: Optional[tk.Button] = None
        self.msxml_points_rotate_check: Optional[tk.Checkbutton] = None
        self.msxml_multicam_vars: Dict[str, tk.Variable] = {}
        self.msxml_multicam_run_button: Optional[tk.Button] = None

        self.ply_vars: Dict[str, tk.Variable] = {}
        self.ply_log: Optional[tk.Text] = None
        self.ply_run_button: Optional[tk.Button] = None
        self.ply_input_view_button: Optional[tk.Button] = None
        self.ply_view_button: Optional[tk.Button] = None
        self.ply_clear_view_button: Optional[tk.Button] = None
        self.ply_append_entry: Optional[tk.Entry] = None
        self.ply_adaptive_weight_entry: Optional[tk.Entry] = None
        self.ply_keep_menu: Optional[ttk.Combobox] = None
        self.ply_target_mode_var = tk.StringVar(value="Target points")
        self.ply_downsample_method_var = tk.StringVar(value="Voxel")
        self._ply_target_value_entry: Optional[tk.Entry] = None
        self._ply_target_value_label: Optional[tk.Label] = None
        self._ply_target_var_map: Dict[str, tk.StringVar] = {}
        self._ply_mode_key_map: Dict[str, str] = {
            "Target points": "points",
            "Target percent": "percent",
            "Voxel size": "voxel",
        }
        self._ply_downsample_method_key_map: Dict[str, str] = {
            "Voxel": "voxel",
            "spatial-hash": "spatial-hash",
            "Adaptive (Octree)": "adaptive",
        }
        self._last_ply_output_path: Optional[Path] = None
        self._ply_current_file_path: Optional[Path] = None
        self._ply_viewer_root: Optional[tk.Widget] = None
        self._ply_view_canvas: Optional[tk.Canvas] = None
        self._ply_canvas_image_id: Optional[int] = None
        self._ply_canvas_photo: Optional[ImageTk.PhotoImage] = None
        self._ply_view_info_var = tk.StringVar(value="PLY viewer is idle")
        self._ply_view_points: Optional[np.ndarray] = None
        self._ply_view_points_centered: Optional[np.ndarray] = None
        self._ply_view_colors: Optional[np.ndarray] = None
        self._ply_view_center = np.zeros(3, dtype=np.float32)
        self._ply_view_total_points = 0
        self._ply_view_sample_step = 1
        self._ply_view_source_label = "PLY"
        self._ply_view_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._ply_view_zoom = 1.0
        self._ply_view_pan: Tuple[float, float] = (0.0, 0.0)
        self._ply_drag_last: Optional[Tuple[int, int]] = None
        self._ply_pan_last: Optional[Tuple[int, int]] = None
        self._ply_view_is_loading = False
        self._ply_view_load_token: Optional[object] = None
        self._ply_loader_thread: Optional[threading.Thread] = None
        self._ply_view_depth_offset = 1.0
        self._ply_view_max_extent = 1.0
        self._ply_monochrome_var = tk.BooleanVar(value=False)
        self._ply_projection_mode = tk.StringVar(value="Orthographic")
        self._ply_view_interactive_max_points_var = tk.StringVar(
            value=str(PLY_VIEW_INTERACTIVE_MAX_POINTS)
        )
        self._ply_view_high_max_points_var = tk.StringVar(
            value=str(PLY_VIEW_MAX_POINTS)
        )
        self._ply_projection_combo: Optional[ttk.Combobox] = None
        self._ply_interactive_max_points_entry: Optional[tk.Entry] = None
        self._ply_high_max_points_entry: Optional[tk.Entry] = None
        self._ply_sky_axis_var = tk.StringVar(value="+Z")
        self._ply_sky_scale_var = tk.StringVar(value="100")
        self._ply_sky_count_var = tk.StringVar(value="4000")
        self._ply_sky_color_var = tk.StringVar(value="#87cefa")
        self._ply_sky_color_rgb_var = tk.StringVar(value="RGB(135, 206, 250)")
        self._ply_sky_color_label: Optional[tk.Label] = None
        self._ply_remove_color_var = tk.StringVar(
            value=self._ply_sky_color_var.get()
        )
        self._ply_remove_color_rgb_var = tk.StringVar(
            value=self._ply_sky_color_rgb_var.get()
        )
        self._ply_remove_color_tol_var = tk.StringVar(value="50")
        self._ply_remove_color_label: Optional[tk.Label] = None
        self._ply_sky_save_path_var = tk.StringVar()
        self._ply_sky_points: Optional[np.ndarray] = None
        self._ply_sky_colors: Optional[np.ndarray] = None
        self._ply_view_rgb_mean: Optional[Tuple[float, float, float]] = None
        self._ply_loaded_points: Optional[np.ndarray] = None
        self._ply_loaded_colors: Optional[np.ndarray] = None
        self._ply_loaded_total_points = 0
        self._ply_loaded_sample_step = 1
        self._ply_is_interacting = False
        self._ply_interaction_after_id: Optional[str] = None
        self._ply_high_max_points_auto = True
        self._ply_high_max_points_updating = False
        self._ply_last_auto_high_max_points = str(PLY_VIEW_MAX_POINTS)

        self.video_stop_button: Optional[tk.Button] = None
        self.selector_stop_button: Optional[tk.Button] = None
        self.human_stop_button: Optional[tk.Button] = None
        self.ply_stop_button: Optional[tk.Button] = None
        self.preview_stop_button: Optional[tk.Button] = None

        self._process_lock = threading.Lock()
        self._active_processes: Dict[str, subprocess.Popen] = {}
        self._process_ui: Dict[str, Dict[str, Any]] = {}
        self._process_start_times: Dict[str, float] = {}
        self._queued_cli_commands: Dict[
            str,
            List[Tuple[List[str], Optional[Path], bool, Optional[str]]],
        ] = defaultdict(list)

        self._video_output_auto = True
        self._video_last_auto_output = ""
        self._video_output_updating = False
        self._video_prefix_auto = True
        self._video_last_auto_prefix = "out"
        self._video_prefix_updating = False
        self._human_output_auto = True
        self._human_last_auto_output = ""
        self._human_output_updating = False
        self._msxml_output_auto = True
        self._msxml_last_auto_output = ""
        self._msxml_output_updating = False
        self._msxml_cut_input_auto = True
        self._msxml_last_auto_cut_input = ""
        self._msxml_cut_input_updating = False
        self._msxml_points_ply_auto = True
        self._msxml_last_auto_points_ply = ""
        self._msxml_points_ply_updating = False
        self._ply_output_auto = True
        self._ply_last_auto_output = ""
        self._ply_output_updating = False
        self.selector_csv_entry: Optional[tk.Entry] = None
        self.selector_dry_run_check: Optional[tk.Checkbutton] = None
        self.selector_csv_button: Optional[tk.Button] = None
        self._selector_duration_message: Optional[str] = None
        self._selector_duration_pending = False

        self._preview_frame_padding = 0
        self._controls_frame_padding = 0

        self.root.title("360Cam-PGM-3DGS-Tools")
        self._output_monitor_stop = threading.Event()
        self._output_monitor_thread: Optional[threading.Thread] = None
        self.build_ui()
        self.set_form_values()
        self.root.update_idletasks()

        loaded = False
        if getattr(args, "input_dir", None):
            loaded = self.try_load_directory(args.input_dir, image_hint=args.image)
            if loaded:
                self.refresh_overlays(initial=True)

        if not loaded:
            self.set_log_text("Select an input folder to preview camera layouts.")

        self.root.deiconify()



    def build_ui(self) -> None:
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True)
        ttk.Separator(self.root, orient="horizontal").pack(fill="x")

        video_tab = tk.Frame(self.notebook)
        selector_tab = tk.Frame(self.notebook)
        preview_tab = tk.Frame(self.notebook)
        human_tab = tk.Frame(self.notebook)
        ply_tab = tk.Frame(self.notebook)
        msxml_tab = tk.Frame(self.notebook)
        config_tab = tk.Frame(self.notebook)

        self.notebook.add(video_tab, text="Video2Frames")
        self.notebook.add(selector_tab, text="FrameSelector")
        self.notebook.add(preview_tab, text="360PerspCut")
        self.notebook.add(human_tab, text="SegmentationMaskTool")
        self.notebook.add(ply_tab, text="PlyOptimizer")
        self.notebook.add(msxml_tab, text="MS360xmlToPersCams")
        self.notebook.add(config_tab, text="Config")

        self._build_video_tab(video_tab)
        self._build_frame_selector_tab(selector_tab)
        self._build_preview_tab(preview_tab)
        self._build_human_mask_tab(human_tab)
        self._build_ply_tab(ply_tab)
        self._build_msxml_tab(msxml_tab)
        self._build_config_tab(config_tab)

        self.notebook.select(preview_tab)

    def _build_video_tab(self, parent: tk.Widget) -> None:
        container = tk.Frame(parent)
        container.pack(fill="both", expand=True)

        params = tk.LabelFrame(container, text="Parameters")
        params.pack(fill="x", padx=8, pady=8)

        self.video_vars = {
            "prefix": tk.StringVar(value="out"),
            "video": tk.StringVar(),
            "output": tk.StringVar(),
            "fps": tk.StringVar(value="1"),
            "ext": tk.StringVar(value="jpg"),
            "start": tk.StringVar(),
            "end": tk.StringVar(),
            "keep_rec709": tk.BooleanVar(value=True),
            "overwrite": tk.BooleanVar(value=False),
            "fisheye_experimental": tk.BooleanVar(value=False),
            "fisheye_perspective": tk.BooleanVar(value=False),
            "fisheye_input_fov": tk.StringVar(value="190"),
            "fisheye_persp_focal": tk.StringVar(value="8"),
            "fisheye_persp_size": tk.StringVar(value="3840"),
            "fisheye_projection": tk.StringVar(value="equisolid"),
        }

        self.video_vars["video"].trace_add("write", self._on_video_input_changed)
        self.video_vars["output"].trace_add("write", self._on_video_output_changed)
        self.video_vars["fps"].trace_add("write", self._on_video_fps_changed)
        self.video_vars["prefix"].trace_add("write", self._on_video_prefix_changed)
        self.video_vars["fisheye_experimental"].trace_add(
            "write", self._on_fisheye_experimental_changed
        )
        self.video_vars["fisheye_perspective"].trace_add(
            "write", self._on_fisheye_perspective_changed
        )
        self.video_set_fps_var = tk.BooleanVar(value=True)

        row = 0
        tk.Label(params, text="Input video").grid(row=row, column=0, sticky="e", padx=4, pady=4)
        video_entry = tk.Entry(params, textvariable=self.video_vars["video"], width=52)
        video_entry.grid(row=row, column=1, sticky="we", padx=4, pady=4)
        tk.Button(
            params,
            text="Browse...",
            command=lambda: self._select_file(self.video_vars["video"], title="Select video file"),
        ).grid(row=row, column=2, padx=4, pady=4)

        row += 1
        tk.Label(params, text="Output folder").grid(row=row, column=0, sticky="e", padx=4, pady=4)
        out_entry = tk.Entry(params, textvariable=self.video_vars["output"], width=52)
        out_entry.grid(row=row, column=1, sticky="we", padx=4, pady=4)
        tk.Button(
            params,
            text="Browse...",
            command=lambda: self._select_directory(self.video_vars["output"], title="Select output folder"),
        ).grid(row=row, column=2, padx=4, pady=4)

        row += 1
        fps_ext_frame = tk.Frame(params)
        fps_ext_frame.grid(row=row, column=0, columnspan=3, sticky="we", pady=4)
        tk.Label(fps_ext_frame, text="FPS").pack(side=tk.LEFT, padx=(0, 4))
        tk.Entry(fps_ext_frame, textvariable=self.video_vars["fps"], width=10).pack(side=tk.LEFT, padx=(0, 12))
        tk.Label(fps_ext_frame, text="Extension").pack(side=tk.LEFT, padx=(0, 4))
        ext_combo = ttk.Combobox(
            fps_ext_frame,
            textvariable=self.video_vars["ext"],
            values=("jpg", "tif", "png"),
            width=8,
            state="readonly",
        )
        ext_combo.pack(side=tk.LEFT, padx=(0, 4))
        tk.Label(fps_ext_frame, text="Prefix").pack(side=tk.LEFT, padx=(12, 4))
        tk.Entry(fps_ext_frame, textvariable=self.video_vars["prefix"], width=12).pack(side=tk.LEFT, padx=(0, 4))

        row += 1
        range_frame = tk.Frame(params)
        range_frame.grid(row=row, column=0, columnspan=3, sticky="we", pady=4)
        range_frame.columnconfigure(1, weight=0)
        range_frame.columnconfigure(3, weight=0)
        tk.Label(range_frame, text="Start (s)").grid(row=0, column=0, sticky="e", padx=(0, 4))
        tk.Entry(
            range_frame,
            textvariable=self.video_vars["start"],
            width=10,
        ).grid(row=0, column=1, sticky="w", padx=(0, 12))
        tk.Label(range_frame, text="End (s)").grid(row=0, column=2, sticky="e", padx=(0, 4))
        tk.Entry(
            range_frame,
            textvariable=self.video_vars["end"],
            width=10,
        ).grid(row=0, column=3, sticky="w", padx=(0, 12))

        row += 1
        flags_frame = tk.Frame(params)
        flags_frame.grid(row=row, column=0, columnspan=3, sticky="w", pady=4)
        tk.Checkbutton(
            flags_frame,
            text="Convert Rec.709 to sRGB",
            variable=self.video_vars["keep_rec709"],
        ).pack(side=tk.LEFT, padx=(0, 12))
        tk.Checkbutton(
            flags_frame,
            text="Overwrite output",
            variable=self.video_vars["overwrite"],
        ).pack(side=tk.LEFT, padx=(0, 12))

        row += 1
        experimental_frame = tk.Frame(params)
        experimental_frame.grid(row=row, column=0, columnspan=3, sticky="w", pady=4)
        fisheye_main_cb = tk.Checkbutton(
            experimental_frame,
            text="Experimental: Fisheye extraction (raw video input)",
            variable=self.video_vars["fisheye_experimental"],
        )
        fisheye_main_cb.pack(side=tk.LEFT, padx=(0, 12))
        self.fisheye_persp_check = tk.Checkbutton(
            experimental_frame,
            text="Experimental: Dual fisheye to perspective",
            variable=self.video_vars["fisheye_perspective"],
        )
        self.fisheye_persp_check.pack(side=tk.LEFT, padx=(0, 12))
        tk.Label(experimental_frame, text="Input FOV (deg)").pack(side=tk.LEFT, padx=(0, 4))
        self.fisheye_fov_entry = tk.Entry(
            experimental_frame,
            textvariable=self.video_vars["fisheye_input_fov"],
            width=6,
        )
        self.fisheye_fov_entry.pack(side=tk.LEFT, padx=(0, 12))
        tk.Label(experimental_frame, text="Output focal (mm, approx)").pack(side=tk.LEFT, padx=(0, 4))
        self.fisheye_focal_entry = tk.Entry(
            experimental_frame,
            textvariable=self.video_vars["fisheye_persp_focal"],
            width=6,
        )
        self.fisheye_focal_entry.pack(side=tk.LEFT, padx=(0, 12))
        tk.Label(experimental_frame, text="Size (px)").pack(side=tk.LEFT, padx=(0, 4))
        self.fisheye_size_entry = tk.Entry(
            experimental_frame,
            textvariable=self.video_vars["fisheye_persp_size"],
            width=6,
        )
        self.fisheye_size_entry.pack(side=tk.LEFT, padx=(0, 4))
        tk.Label(experimental_frame, text="Projection").pack(side=tk.LEFT, padx=(8, 4))
        self.fisheye_projection_combo = ttk.Combobox(
            experimental_frame,
            textvariable=self.video_vars["fisheye_projection"],
            values=("equidistant", "equisolid"),
            width=12,
            state="readonly",
        )
        self.fisheye_projection_combo.pack(side=tk.LEFT, padx=(0, 4))

        for col in range(3):
            params.grid_columnconfigure(col, weight=1 if col == 1 else 0)

        actions = tk.Frame(container)
        actions.pack(fill="x", padx=8, pady=(0, 8))
        self.video_inspect_button = tk.Button(
            actions,
            text="Inspect video",
            command=self._inspect_video_metadata,
            state="disabled",
        )
        self.video_inspect_button.pack(side=tk.LEFT, padx=4, pady=4)
        self.video_set_fps_check = tk.Checkbutton(
            actions,
            text="Set FPS",
            variable=self.video_set_fps_var,
        )
        self.video_set_fps_check.pack(side=tk.LEFT, padx=4, pady=4)
        self.video_stop_button = tk.Button(
            actions,
            text="Stop",
            command=lambda: self._stop_cli_process("video"),
        )
        self.video_stop_button.pack(side=tk.RIGHT, padx=4, pady=4)
        self.video_stop_button.configure(state="disabled")
        self.video_run_button = tk.Button(
            actions,
            text="Run gs360_Video2Frames",
            command=self._run_video_tool,
        )
        self.video_run_button.pack(side=tk.RIGHT, padx=4, pady=4)

        log_frame = tk.LabelFrame(container, text="Log")
        log_frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.video_log = tk.Text(log_frame, wrap="word", height=18, cursor="arrow")
        self.video_log.pack(fill="both", expand=True, padx=6, pady=4)
        self.video_log.bind("<Key>", self._block_text_edit)
        self.video_log.bind("<Button-1>", lambda event: self.video_log.focus_set())
        self._set_text_widget(self.video_log, "")

        self._update_video_default_output(force=False)
        self._update_fisheye_controls()

    def _on_video_input_changed(self, *_args) -> None:
        value = self.video_vars.get("video")
        if value is None:
            return
        if not value.get().strip():
            return
        self._update_video_default_output(force=True)

    def _on_video_fps_changed(self, *_args) -> None:
        self._update_video_default_output(force=False)

    def _on_video_output_changed(self, *_args) -> None:
        if self._video_output_updating:
            return
        self._video_output_auto = False

    def _on_video_prefix_changed(self, *_args) -> None:
        if self._video_prefix_updating:
            return
        self._video_prefix_auto = False

    def _on_fisheye_experimental_changed(self, *_args) -> None:
        self._update_fisheye_controls()

    def _on_fisheye_perspective_changed(self, *_args) -> None:
        self._update_fisheye_controls()

    def _update_fisheye_controls(self) -> None:
        experimental_var = self.video_vars.get("fisheye_experimental")
        perspective_var = self.video_vars.get("fisheye_perspective")
        experimental_on = bool(experimental_var.get()) if experimental_var is not None else False
        if not experimental_on and perspective_var is not None and perspective_var.get():
            perspective_var.set(False)
        perspective_on = experimental_on and bool(perspective_var.get()) if perspective_var is not None else False
        check = self.fisheye_persp_check
        if check is not None:
            state = "normal" if experimental_on else "disabled"
            try:
                check.configure(state=state)
            except tk.TclError:
                pass
        entry_state = "normal" if perspective_on else "disabled"
        for entry in (
            self.fisheye_fov_entry,
            self.fisheye_focal_entry,
            self.fisheye_size_entry,
        ):
            if entry is not None:
                try:
                    entry.configure(state=entry_state)
                except tk.TclError:
                    pass
        combo = self.fisheye_projection_combo
        if combo is not None:
            try:
                combo.configure(state="readonly" if perspective_on else "disabled")
            except tk.TclError:
                pass


    def _format_fps_for_output(self, fps_value: str) -> Optional[str]:
        if not fps_value:
            return None
        try:
            fps = float(fps_value)
        except ValueError:
            return None
        fps = max(fps, 0.0)
        text = f"{fps}"
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return text or "0"

    @staticmethod
    def _parse_positive_float(value: str) -> Optional[float]:
        text = value.strip()
        if not text:
            return None
        try:
            parsed = float(text)
        except ValueError:
            return None
        if parsed <= 0:
            return None
        return parsed

    def _estimate_output_frames_from_metadata(
        self,
        metadata: Optional[Dict[str, Any]],
        fps_out: Optional[float],
        start_text: Optional[str] = None,
        end_text: Optional[str] = None,
    ) -> Optional[int]:
        if not metadata or fps_out is None or fps_out <= 0:
            return None
        try:
            frames_est = int(metadata.get("estimated_frames"))
        except (TypeError, ValueError):
            return None
        try:
            fps_in = float(metadata.get("frame_rate"))
        except (TypeError, ValueError):
            return None
        if frames_est <= 0 or fps_in <= 0:
            return None
        frames_effective = frames_est
        if start_text or end_text:
            duration_est = frames_est / fps_in
            start_sec = 0.0
            if start_text:
                try:
                    start_sec = max(0.0, float(start_text))
                except ValueError:
                    start_sec = 0.0
            end_limit = duration_est
            if end_text:
                try:
                    end_limit = max(0.0, min(float(end_text), duration_est))
                except ValueError:
                    end_limit = duration_est
            trimmed = max(end_limit - start_sec, 0.0)
            frames_effective = int(round(trimmed * fps_in))
        frames_effective = max(frames_effective, 0)
        output_frames = int(round(frames_effective * (float(fps_out) / fps_in)))
        return max(output_frames, 0)

    def _estimate_preview_view_count(self, video_path: Path) -> Optional[int]:
        target_path = video_path
        try:
            target_path = video_path.resolve()
        except FileNotFoundError:
            target_path = video_path
        except Exception:
            pass
        for specs in (self.matched_specs, self.result.view_specs if self.result else None):
            if not specs:
                continue
            view_ids: Set[str] = set()
            for spec in specs:
                try:
                    spec_path = spec.source_path.resolve()
                except FileNotFoundError:
                    spec_path = spec.source_path
                except Exception:
                    spec_path = spec.source_path
                if spec_path == target_path:
                    view_ids.add(spec.view_id)
            if view_ids:
                return len(view_ids)
        try:
            args_for_build = clone_namespace(self.current_args)
            ensure_explicit_flags(args_for_build)
            args_for_build.input_dir = str(self.in_dir) if self.in_dir else str(video_path)
            args_for_build.out_dir = str(self.out_dir) if self.out_dir else None
            setattr(args_for_build, "input_is_video", True)
            setattr(
                args_for_build,
                "video_bit_depth",
                getattr(self.current_args, "video_bit_depth", 8),
            )
            output_root = self.out_dir or self._default_output_path() or video_path.parent
            result = cutter.build_view_jobs(args_for_build, [video_path], output_root)
            view_ids = {spec.view_id for spec in result.view_specs}
            return len(view_ids) if view_ids else None
        except Exception:
            return None

    def _update_video_default_output(self, force: bool = False) -> None:
        if "video" not in self.video_vars or "output" not in self.video_vars:
            return
        video_value = self.video_vars["video"].get().strip()
        self._update_video_inspect_state()
        prefix_var = self.video_vars.get("prefix")
        if not video_value:
            if prefix_var is not None and (self._video_prefix_auto or not prefix_var.get().strip()):
                self._video_prefix_auto = True
                self._video_last_auto_prefix = "out"
                self._video_prefix_updating = True
                try:
                    prefix_var.set("out")
                finally:
                    self._video_prefix_updating = False
            return

        fps_text = self.video_vars["fps"].get().strip()
        fps_formatted = self._format_fps_for_output(fps_text) or "fps"

        path = Path(video_value).expanduser()
        try:
            if not path.is_absolute():
                path = path.resolve()
        except FileNotFoundError:
            return
        if not path.suffix:
            return

        parent = path.parent if path.parent != Path("") else Path.cwd()
        default_output = str(parent / f"{path.stem}_frames_{fps_formatted}fps")

        if prefix_var is not None:
            default_prefix = re.sub(r"\s+", "_", path.stem) if path.stem else "out"
            if not default_prefix:
                default_prefix = "out"
            current_prefix = prefix_var.get().strip()
            should_update_prefix = (
                force
                or self._video_prefix_auto
                or not current_prefix
                or current_prefix == self._video_last_auto_prefix
            )
            if should_update_prefix:
                self._video_prefix_auto = True
                self._video_last_auto_prefix = default_prefix
                self._video_prefix_updating = True
                try:
                    prefix_var.set(default_prefix)
                finally:
                    self._video_prefix_updating = False

        current_output = self.video_vars["output"].get().strip()
        should_update = force or self._video_output_auto or not current_output or current_output == self._video_last_auto_output
        if not should_update:
            return

        self._video_output_auto = True
        self._video_last_auto_output = default_output
        self._video_output_updating = True
        try:
            self.video_vars["output"].set(default_output)
        finally:
            self._video_output_updating = False

    def _update_video_inspect_state(self) -> None:
        button = self.video_inspect_button
        if button is None:
            return
        video_var = self.video_vars.get("video")
        if video_var is None:
            button.configure(state="disabled")
            return
        path_text = video_var.get().strip()
        if not path_text:
            button.configure(state="disabled")
            return
        try:
            path = Path(path_text).expanduser()
        except Exception:
            button.configure(state="disabled")
            return
        enabled = path.exists() and path.is_file()
        button.configure(state="normal" if enabled else "disabled")
        checkbox = getattr(self, "video_set_fps_check", None)
        if checkbox is not None:
            checkbox.configure(state="normal" if enabled else "disabled")

    def _update_preview_inspect_state(self) -> None:
        button = self.preview_inspect_button
        if button is None:
            return
        if not self.source_is_video:
            button.configure(state="disabled")
            self._update_preview_csv_state()
            return
        video_path = None
        if self.files:
            candidate = self.files[0]
            if isinstance(candidate, Path):
                video_path = candidate
            else:
                try:
                    video_path = Path(candidate)
                except Exception:
                    video_path = None
        if video_path is None:
            button.configure(state="disabled")
            self._update_preview_csv_state()
            return
        try:
            exists = video_path.exists()
        except Exception:
            exists = False
        button.configure(state="normal" if exists else "disabled")
        self._update_preview_csv_state()

    def _resolve_ffmpeg_path(self) -> str:
        ffmpeg_value = self.ffmpeg_path_var.get().strip()
        if not ffmpeg_value:
            ffmpeg_value = str(getattr(self.defaults, "ffmpeg", "ffmpeg"))
        if not ffmpeg_value:
            return "ffmpeg"
        try:
            ffmpeg_path = Path(ffmpeg_value).expanduser()
        except Exception:
            return ffmpeg_value
        if ffmpeg_path.is_file():
            return str(ffmpeg_path)
        return ffmpeg_value

    @staticmethod
    def _parse_duration_text(duration_text: str) -> Optional[float]:
        if not duration_text:
            return None
        match = re.match(r"(?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2}(?:\.\d+)?)", duration_text.strip())
        if not match:
            return None
        try:
            hours = int(match.group("h"))
            minutes = int(match.group("m"))
            seconds = float(match.group("s"))
            return hours * 3600.0 + minutes * 60.0 + seconds
        except ValueError:
            return None

    @staticmethod
    def _parse_fps_from_stream(stream_text: str) -> Optional[float]:
        if not stream_text:
            return None
        fps_match = re.search(r"(\d+(?:\.\d+)?)\s*fps", stream_text)
        if not fps_match:
            return None
        try:
            return float(fps_match.group(1))
        except ValueError:
            return None

    @staticmethod
    def _format_metadata_entries(entries: Sequence[str]) -> List[str]:
        formatted: List[str] = []
        for raw in entries:
            if raw is None:
                continue
            text = str(raw).strip()
            if not text:
                continue
            if ":" in text:
                key, value = text.split(":", 1)
                formatted.append(f"{key.strip()}: {value.strip()}")
            else:
                formatted.append(text)
        return formatted

    @staticmethod
    def _format_duration(seconds: float) -> str:
        seconds = max(0.0, seconds)
        total_ms = int(round(seconds * 1000))
        ms = total_ms % 1000
        total_seconds = total_ms // 1000
        s = total_seconds % 60
        total_minutes = total_seconds // 60
        m = total_minutes % 60
        h = total_minutes // 60
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

    def _store_estimated_frames(self, video_path: Path, metadata: Optional[Dict[str, Any]]) -> None:
        if not metadata:
            return
        estimated = metadata.get("estimated_frames")
        if estimated is None:
            return
        try:
            est_int = int(estimated)
        except Exception:
            return
        frame_rate = metadata.get("frame_rate")
        try:
            frame_rate = float(frame_rate)
        except Exception:
            frame_rate = None
        key = str(Path(video_path).expanduser().resolve())
        self._video_estimated_cache[key] = {"estimated_frames": est_int, "frame_rate": frame_rate}
        self._preview_estimated_frames = est_int

    def _get_estimated_frames_info(self, video_path: Path) -> Tuple[Optional[int], Optional[float]]:
        key = None
        try:
            key = str(Path(video_path).expanduser().resolve())
        except Exception:
            key = str(video_path)
        if key and key in self._video_estimated_cache:
            cached = self._video_estimated_cache.get(key, {})
            return cached.get("estimated_frames"), cached.get("frame_rate")
        result = self._collect_video_metadata_lines(video_path, "Frame count probe")
        if not result:
            return None, None
        _lines, metadata = result
        self._store_estimated_frames(video_path, metadata)
        frames = None
        fps = None
        if metadata:
            try:
                frames = int(metadata.get("estimated_frames"))
            except Exception:
                frames = None
            try:
                fps = float(metadata.get("frame_rate"))
            except Exception:
                fps = None
        return frames, fps

    def _get_estimated_frames(self, video_path: Path) -> Optional[int]:
        frames, _fps = self._get_estimated_frames_info(video_path)
        return frames

    def _apply_detected_video_fps(self, metadata: Optional[Dict[str, Any]]) -> Tuple[Optional[str], bool]:
        if not metadata:
            return None, False
        fps_value = metadata.get("frame_rate")
        fps_text_raw = metadata.get("frame_rate_text")
        detected = None
        if fps_value is not None:
            try:
                detected = float(fps_value)
            except (TypeError, ValueError):
                detected = None
        if detected is None and fps_text_raw:
            detected = self._parse_fps_from_stream(str(fps_text_raw))
        if detected is None or detected <= 0.0:
            return None, False
        formatted = self._format_fps_for_output(f"{detected}")
        if not formatted:
            return None, False
        fps_var = self.video_vars.get("fps") if self.video_vars else None
        if fps_var is None:
            return None, False
        should_apply = True
        if hasattr(self, "video_set_fps_var") and self.video_set_fps_var is not None:
            try:
                should_apply = bool(self.video_set_fps_var.get())
            except Exception:
                should_apply = True
        if should_apply:
            fps_var.set(formatted)
            return formatted, True
        return formatted, False

    def _inspect_video_metadata(self) -> None:
        if not self.video_vars:
            return
        video_value = self.video_vars.get("video")
        if video_value is None:
            return
        video_path_text = video_value.get().strip()
        if not video_path_text:
            messagebox.showerror("Video metadata", "Select an input video first.")
            return
        try:
            video_path = Path(video_path_text).expanduser().resolve(strict=True)
        except FileNotFoundError:
            messagebox.showerror("Video metadata", f"Video file not found:\n{video_path_text}")
            return
        except Exception as exc:
            messagebox.showerror("Video metadata", f"Failed to resolve video path:\n{exc}")
            return

        result = self._collect_video_metadata_lines(video_path, "Video metadata")
        if not result:
            return
        lines, metadata = result
        self._set_text_widget(self.video_log, "")
        for line in lines:
            self._append_text_widget(self.video_log, line)
        fps_updated, applied = self._apply_detected_video_fps(metadata)
        self._store_estimated_frames(video_path, metadata)
        if fps_updated:
            if applied:
                self._append_text_widget(self.video_log, f"[fps] Updated FPS to detected value ({fps_updated})")
            else:
                self._append_text_widget(
                    self.video_log,
                    f"[fps] Detected FPS {fps_updated} (Set FPS unchecked; entry left unchanged)",
                )
                fps_var = self.video_vars.get("fps") if self.video_vars else None
                start_var = self.video_vars.get("start") if self.video_vars else None
                end_var = self.video_vars.get("end") if self.video_vars else None
                fps_text = fps_var.get().strip() if fps_var is not None else ""
                start_text = start_var.get().strip() if start_var is not None else ""
                end_text = end_var.get().strip() if end_var is not None else ""
                fps_out = self._parse_positive_float(fps_text)
                estimate = self._estimate_output_frames_from_metadata(
                    metadata,
                    fps_out,
                    start_text=start_text,
                    end_text=end_text,
                )
                if estimate is not None:
                    fps_label = f"{fps_out:g}" if fps_out is not None else "?"
                    self._append_text_widget(
                        self.video_log,
                        f"[estimate] Output {estimate} frames @ fps={fps_label}",
                    )
                else:
                    self._append_text_widget(
                        self.video_log,
                        "[estimate] Output frames unavailable (missing input fps/frames or fps entry).",
                    )

    def _inspect_preview_video_metadata(self) -> None:
        if not self.source_is_video:
            messagebox.showerror("Preview video metadata", "Load a video in gs360_360PerspCut first.")
            return
        if not self.files:
            messagebox.showerror("Preview video metadata", "No video source is available in gs360_360PerspCut.")
            return
        candidate = self.files[0]
        raw_text = str(candidate)
        try:
            video_path = candidate if isinstance(candidate, Path) else Path(candidate)
            video_path = video_path.expanduser().resolve(strict=True)
        except FileNotFoundError:
            messagebox.showerror("Preview video metadata", f"Video file not found:\n{raw_text}")
            return
        except Exception as exc:
            messagebox.showerror("Preview video metadata", f"Failed to resolve video path:\n{exc}")
            return

        result = self._collect_video_metadata_lines(video_path, "Preview video metadata")
        if not result:
            return
        lines, _metadata = result
        self.set_log_text("")
        for line in lines:
            self._append_text_widget(self.log_text, line)
        self._store_estimated_frames(video_path, _metadata)
        view_count = self._estimate_preview_view_count(video_path)
        csv_path_text = self.preview_csv_var.get().strip() if self.preview_csv_var is not None else ""
        if csv_path_text:
            try:
                indices, total_rows = self._load_selected_frames_from_csv(Path(csv_path_text))
            except FileNotFoundError as exc:
                messagebox.showerror("Selection CSV", str(exc))
                return
            except Exception as exc:
                messagebox.showerror("Selection CSV", f"Failed to read CSV:\n{exc}")
                return
            self._append_text_widget(
                self.log_text,
                f"[estimate] CSV selected frames: {len(indices)} (rows: {total_rows})",
            )
            if view_count is not None:
                total_outputs = len(indices) * view_count
                self._append_text_widget(
                    self.log_text,
                    f"[estimate] Total outputs: {total_outputs}",
                )
            else:
                self._append_text_widget(
                    self.log_text,
                    "[estimate] Total outputs: unavailable (camera count unknown).",
                )
        else:
            fps_var = self.field_vars.get("fps")
            start_var = self.field_vars.get("start")
            end_var = self.field_vars.get("end")
            fps_text = fps_var.get().strip() if fps_var is not None else ""
            start_text = start_var.get().strip() if start_var is not None else ""
            end_text = end_var.get().strip() if end_var is not None else ""
            fps_out = self._parse_positive_float(fps_text)
            estimate = self._estimate_output_frames_from_metadata(
                _metadata,
                fps_out,
                start_text=start_text,
                end_text=end_text,
            )
            if estimate is not None:
                fps_label = f"{fps_out:g}" if fps_out is not None else "?"
                self._append_text_widget(
                    self.log_text,
                    f"[estimate] Output {estimate} frames @ fps={fps_label}",
                )
                if view_count is not None:
                    total_outputs = estimate * view_count
                    self._append_text_widget(
                        self.log_text,
                        f"[estimate] Total outputs: {total_outputs}",
                    )
                else:
                    self._append_text_widget(
                        self.log_text,
                        "[estimate] Total outputs: unavailable (camera count unknown).",
                    )
            else:
                self._append_text_widget(
                    self.log_text,
                    "[estimate] Output frames unavailable (missing input fps/frames or fps entry).",
                )

    def _collect_video_metadata_lines(
        self,
        video_path: Path,
        dialog_title: str,
    ) -> Optional[Tuple[List[str], Dict[str, Any]]]:
        ffmpeg_path = self._resolve_ffmpeg_path()
        cmd = [
            ffmpeg_path,
            "-hide_banner",
            "-i",
            str(video_path),
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            messagebox.showerror(
                dialog_title,
                f"ffmpeg not found: {ffmpeg_path}\nAdjust the ffmpeg path in the Config tab.",
            )
            return None
        except Exception as exc:
            messagebox.showerror(dialog_title, f"ffmpeg execution failed:\n{exc}")
            return None

        combined_output = (result.stderr or "") + "\n" + (result.stdout or "")
        if not combined_output.strip():
            messagebox.showinfo(dialog_title, "ffmpeg did not produce any output to inspect.")
            return None

        raw_lines = combined_output.splitlines()

        def collect_indented_block(start_index: int) -> List[str]:
            block: List[str] = []
            for j in range(start_index + 1, len(raw_lines)):
                candidate = raw_lines[j]
                if not candidate.strip():
                    break
                if candidate.lstrip() == candidate:
                    break
                block.append(candidate.strip())
            return block

        def previous_non_empty(index: int) -> str:
            for j in range(index - 1, -1, -1):
                candidate = raw_lines[j].strip()
                if candidate:
                    return candidate
            return ""

        stream_metadata_raw: Dict[str, List[str]] = defaultdict(list)
        stream_side_raw: Dict[str, List[str]] = defaultdict(list)

        for idx, raw_line in enumerate(raw_lines):
            stripped = raw_line.strip()
            if stripped.startswith("Metadata:"):
                prev = previous_non_empty(idx)
                block = collect_indented_block(idx)
                if prev.startswith("Stream #"):
                    stream_metadata_raw[prev].extend(block)
            elif stripped.startswith("Side data:"):
                prev = previous_non_empty(idx)
                block = collect_indented_block(idx)
                if prev.startswith("Stream #"):
                    stream_side_raw[prev].extend(block)

        stream_pattern = re.compile(
            r"Stream #(?P<id>\d+:\d+)(?:\[[^\]]+\])?(?:\((?P<lang>[^\)]*)\))?: "
            r"(?P<type>Video|Audio): (?P<details>.+)"
        )
        stream_infos: List[Dict[str, Any]] = []
        for raw_line in raw_lines:
            stripped = raw_line.strip()
            match = stream_pattern.match(stripped)
            if not match:
                continue
            key = stripped
            stream_infos.append(
                {
                    "id": match.group("id"),
                    "lang": (match.group("lang") or "").strip(),
                    "type": match.group("type"),
                    "details": match.group("details").strip(),
                    "metadata": self._format_metadata_entries(stream_metadata_raw.get(key, [])),
                    "side": self._format_metadata_entries(stream_side_raw.get(key, [])),
                }
            )

        def parse_video_stream(info: Dict[str, Any]) -> Dict[str, Any]:
            tokens = [token.strip() for token in info["details"].split(",") if token.strip()]
            codec = tokens[0] if tokens else info["details"]
            pixel_format = ""
            pixel_details: List[str] = []
            color_tags: List[str] = []
            resolution = ""
            sar_dar = ""
            bitrate = ""
            frame_rate_value = self._parse_fps_from_stream(info["details"])
            frame_rate_text_local = ""
            other_tags: List[str] = []

            if len(tokens) > 1:
                fmt_token = tokens[1]
                if "(" in fmt_token:
                    fmt_name, _, rest = fmt_token.partition("(")
                    pixel_format = fmt_name.strip()
                    rest = rest.rstrip(")")
                    if rest:
                        pixel_details = [part.strip() for part in rest.split(",") if part.strip()]
                else:
                    pixel_format = fmt_token
                color_tags.extend(pixel_details)

            for token in tokens[2:]:
                token = token.strip()
                if not token:
                    continue
                lower = token.lower()
                if re.match(r"\d{2,5}x\d{2,5}", token):
                    resolution = token
                elif token.startswith("[") and token.endswith("]"):
                    sar_dar = token
                elif "kb/s" in lower:
                    bitrate = token
                elif lower.endswith("fps"):
                    if not frame_rate_value and not frame_rate_text_local:
                        frame_rate_text_local = token
                elif any(keyword in lower for keyword in ("bt", "progressive", "interlaced", "tv", "pc")):
                    color_tags.append(token)
                elif lower.endswith("tbr") or lower.endswith("tbn") or lower.endswith("tbc"):
                    other_tags.append(token)
                else:
                    other_tags.append(token)

            color_tags = list(dict.fromkeys(color_tags))
            other_tags = list(dict.fromkeys(other_tags))

            return {
                "id": info["id"],
                "lang": info["lang"],
                "codec": codec,
                "pixel_format": pixel_format,
                "pixel_details": pixel_details,
                "resolution": resolution,
                "sar_dar": sar_dar,
                "bitrate": bitrate,
                "frame_rate": frame_rate_value,
                "frame_rate_text": frame_rate_text_local,
                "color_tags": color_tags,
                "other": other_tags,
                "metadata": info["metadata"],
                "side": info["side"],
            }

        video_streams = [parse_video_stream(info) for info in stream_infos if info["type"] == "Video"]

        lines: List[str] = [f"[inspect] {video_path.name}"]
        primary_metadata: Dict[str, Any] = {}

        duration_match = re.search(r"Duration:\s*(\d{2}:\d{2}:\d{2}\.\d+)", combined_output)
        duration_seconds: Optional[float] = None
        if duration_match:
            duration_seconds = self._parse_duration_text(duration_match.group(1))
        estimated_frames: Optional[int] = None

        if video_streams:
            video = video_streams[0]
            primary_metadata = {
                "frame_rate": video.get("frame_rate"),
                "frame_rate_text": video.get("frame_rate_text"),
            }
            lines.append("Video stream:")
            stream_label = f"#{video['id']}"
            if video["lang"]:
                stream_label += f" ({video['lang']})"
            lines.append(f"  Stream: {stream_label}")
            lines.append(f"  Codec: {video['codec']}")
            if video["pixel_format"]:
                pixel_line = f"  Pixel format: {video['pixel_format']}"
                if video["pixel_details"]:
                    pixel_line += f" ({', '.join(video['pixel_details'])})"
                lines.append(pixel_line)
            if video["resolution"]:
                resolution_line = video["resolution"]
                if video["sar_dar"]:
                    resolution_line += f" {video['sar_dar']}"
                lines.append(f"  Resolution: {resolution_line}")
            if video["bitrate"]:
                lines.append(f"  Bit rate: {video['bitrate']}")
            if video["frame_rate"]:
                lines.append(f"  Frame rate:  {video['frame_rate']:.3f} fps")
            elif video["frame_rate_text"]:
                lines.append(f"  Frame rate:  {video['frame_rate_text']}")
            if video["frame_rate"] and duration_seconds is not None:
                estimated_frames = int(round(duration_seconds * video["frame_rate"]))
                lines.append(f"  Estimated frames:  {estimated_frames}")
            if video["color_tags"]:
                clean_color = [tag.rstrip(")") for tag in video["color_tags"]]
                lines.append(f"  Color: {', '.join(clean_color)}")
            if video["other"]:
                lines.append(f"  Other: {', '.join(video['other'])}")
            if video["metadata"]:
                lines.append("  Metadata:")
                for item in video["metadata"]:
                    lines.append(f"    {item}")
            if video["side"]:
                lines.append("  Side data:")
                for item in video["side"]:
                    lines.append(f"    {item}")
            if estimated_frames is not None:
                primary_metadata["estimated_frames"] = estimated_frames

        if result.returncode != 0 and "At least one output file must be specified" not in combined_output:
            lines.append(f"[warn] ffmpeg exited with code {result.returncode}")

        return lines, primary_metadata


    def _build_frame_selector_tab(self, parent: tk.Widget) -> None:
        container = tk.Frame(parent)
        container.pack(fill="both", expand=True)

        params = tk.LabelFrame(container, text="Parameters")
        params.pack(fill="x", padx=8, pady=8)

        self.selector_vars = {
            "in_dir": tk.StringVar(),
            "segment_size": tk.StringVar(value="10"),
            "dry_run": tk.BooleanVar(value=True),
            "workers": tk.StringVar(value="auto"),
            "ext": tk.StringVar(value="all"),
            "sort": tk.StringVar(value="lastnum"),
            "score_backend": tk.StringVar(value="opencv"),
            "csv_mode": tk.StringVar(value="write"),
            "csv_path": tk.StringVar(),
            "crop_ratio": tk.StringVar(value="0.8"),
            "min_spacing_frames": tk.StringVar(value="auto"),
            "augment_gap": tk.BooleanVar(value=True),
            "augment_lowlight": tk.BooleanVar(value=False),
            "augment_motion": tk.BooleanVar(value=False),
            "segment_boundary_reopt": tk.BooleanVar(value=True),
            "ignore_highlights": tk.BooleanVar(value=True),
        }

        self.selector_vars["csv_mode"].trace_add("write", self._on_selector_csv_mode_changed)
        self.selector_vars["in_dir"].trace_add("write", self._on_selector_in_dir_changed)
        self.selector_vars["csv_path"].trace_add("write", self._on_selector_csv_path_changed)

        row = 0
        tk.Label(params, text="Input folder").grid(row=row, column=0, sticky="e", padx=4, pady=4)
        tk.Entry(params, textvariable=self.selector_vars["in_dir"], width=52).grid(row=row, column=1, sticky="we", padx=4, pady=4)
        tk.Button(
            params,
            text="Browse...",
            command=lambda: self._select_directory(
                self.selector_vars["in_dir"],
                title="Select input folder",
                on_select=self._on_selector_input_selected,
            ),
        ).grid(row=row, column=2, padx=4, pady=4)

        row += 1
        segment_frame = tk.Frame(params)
        segment_frame.grid(row=row, column=0, columnspan=3, sticky="we", pady=4)
        tk.Label(segment_frame, text="Segment size").pack(side=tk.LEFT, padx=(0, 4))
        tk.Entry(segment_frame, textvariable=self.selector_vars["segment_size"], width=10).pack(side=tk.LEFT, padx=(0, 12))
        tk.Label(segment_frame, text="Workers").pack(side=tk.LEFT, padx=(0, 4))
        tk.Entry(segment_frame, textvariable=self.selector_vars["workers"], width=10).pack(side=tk.LEFT, padx=(0, 12))
        self.selector_dry_run_check = tk.Checkbutton(
            segment_frame,
            text="Dry run",
            variable=self.selector_vars["dry_run"],
        )
        self.selector_dry_run_check.pack(side=tk.LEFT, padx=(0, 12))

        row += 1
        esm_frame = tk.Frame(params)
        esm_frame.grid(row=row, column=0, columnspan=3, sticky="we", pady=4)
        tk.Label(esm_frame, text="Extension").pack(side=tk.LEFT, padx=(0, 4))
        selector_ext_combo = ttk.Combobox(
            esm_frame,
            textvariable=self.selector_vars["ext"],
            values=("all", "tif", "jpg", "png"),
            state="readonly",
            width=8,
        )
        selector_ext_combo.pack(side=tk.LEFT, padx=(0, 12))
        tk.Label(esm_frame, text="Sort").pack(side=tk.LEFT, padx=(0, 4))
        selector_sort_combo = ttk.Combobox(
            esm_frame,
            textvariable=self.selector_vars["sort"],
            values=("lastnum", "firstnum", "name", "mtime"),
            state="readonly",
            width=10,
        )
        selector_sort_combo.pack(side=tk.LEFT, padx=(0, 12))
        tk.Label(esm_frame, text="Sharpness analysis backend").pack(side=tk.LEFT, padx=(0, 4))
        selector_score_combo = ttk.Combobox(
            esm_frame,
            textvariable=self.selector_vars["score_backend"],
            values=("ffmpeg", "opencv"),
            state="readonly",
            width=10,
        )
        selector_score_combo.pack(side=tk.LEFT, padx=(0, 12))

        row += 1
        csv_frame = tk.Frame(params)
        csv_frame.grid(row=row, column=0, columnspan=3, sticky="we", pady=4)
        tk.Label(csv_frame, text="CSV mode").pack(side=tk.LEFT, padx=(0, 4))
        selector_csv_combo = ttk.Combobox(
            csv_frame,
            textvariable=self.selector_vars["csv_mode"],
            values=("write", "reselect", "apply", "none"),
            state="readonly",
            width=10,
        )
        selector_csv_combo.pack(side=tk.LEFT, padx=(0, 12))
        tk.Label(csv_frame, text="Path").pack(side=tk.LEFT, padx=(0, 4))
        self.selector_csv_entry = tk.Entry(csv_frame, textvariable=self.selector_vars["csv_path"], width=40)
        self.selector_csv_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        self.selector_csv_button = tk.Button(csv_frame, text="Browse...", command=self._browse_selector_csv)
        self.selector_csv_button.pack(side=tk.LEFT, padx=(0, 4))

        row += 1
        tk.Label(params, text="Min spacing").grid(row=row, column=0, sticky="e", padx=4, pady=4)
        tk.Entry(params, textvariable=self.selector_vars["min_spacing_frames"], width=10).grid(row=row, column=1, sticky="w", padx=4, pady=4)

        row += 1
        augment_frame = tk.Frame(params)
        augment_frame.grid(row=row, column=0, columnspan=3, sticky="w", pady=4)
        tk.Checkbutton(augment_frame, text="Augment gap", variable=self.selector_vars["augment_gap"]).pack(side=tk.LEFT, padx=(0, 12))
        tk.Checkbutton(
            augment_frame,
            text="Augment lowlight (Experimental)",
            variable=self.selector_vars["augment_lowlight"],
        ).pack(side=tk.LEFT, padx=(0, 12))
        tk.Checkbutton(
            augment_frame,
            text="Augment motion (Experimental)",
            variable=self.selector_vars["augment_motion"],
        ).pack(side=tk.LEFT, padx=(0, 12))
        tk.Checkbutton(
            augment_frame,
            text="Segment Top-K + Boundary Local Reopt",
            variable=self.selector_vars["segment_boundary_reopt"],
        ).pack(side=tk.LEFT, padx=(0, 12))

        row += 1
        final_frame = tk.Frame(params)
        final_frame.grid(row=row, column=0, columnspan=3, sticky="we", pady=4)
        tk.Label(final_frame, text="Score crop ratio (crop top/bottom of 360deg pano)").pack(side=tk.LEFT, padx=(0, 4))
        tk.Entry(final_frame, textvariable=self.selector_vars["crop_ratio"], width=10).pack(side=tk.LEFT, padx=(0, 12))
        tk.Checkbutton(
            final_frame,
            text="Ignore clipped highlights",
            variable=self.selector_vars["ignore_highlights"],
        ).pack(side=tk.LEFT, padx=(0, 12))

        for col in range(3):
            params.grid_columnconfigure(col, weight=1 if col == 1 else 0)

        actions = tk.Frame(container)
        actions.pack(fill="x", padx=8, pady=(0, 8))
        self.selector_stop_button = tk.Button(
            actions,
            text="Stop",
            command=lambda: self._stop_cli_process("selector"),
        )
        self.selector_stop_button.pack(side=tk.RIGHT, padx=4, pady=4)
        self.selector_stop_button.configure(state="disabled")
        self.selector_run_button = tk.Button(
            actions,
            text="Run gs360_FrameSelector",
            command=self._run_frame_selector,
        )
        self.selector_run_button.pack(side=tk.RIGHT, padx=4, pady=4)

        summary_frame = tk.LabelFrame(container, text="Sharpness OverView Frame")
        summary_frame.pack(fill="x", padx=8, pady=(0, 8))
        overview_controls = tk.Frame(summary_frame)
        overview_controls.pack(fill="x", padx=6, pady=(4, 2))
        self.selector_count_var = tk.StringVar(value="1")
        tk.Label(overview_controls, text="Suspects").pack(side=tk.LEFT, padx=(0, 4))
        count_entry = tk.Entry(overview_controls, textvariable=self.selector_count_var, width=6)
        count_entry.pack(side=tk.LEFT, padx=(0, 0))
        tk.Label(overview_controls, text="%").pack(side=tk.LEFT, padx=(2, 8))
        self.selector_show_score_button = tk.Button(
            overview_controls,
            text="Check Selection",
            command=self._show_selector_scores,
        )
        self.selector_show_score_button.pack(side=tk.LEFT, padx=(0, 8))
        self.selector_open_suspects_button = tk.Button(
            overview_controls,
            text="Open Suspects",
            command=self._open_selector_suspects_preview,
        )
        self.selector_open_suspects_button.pack(side=tk.LEFT, padx=(0, 12))
        self.selector_manual_apply_button = tk.Button(
            overview_controls,
            text="Manual Selection Apply",
            command=self._confirm_selector_manual_selection,
            state="disabled",
        )
        self.selector_manual_apply_button.pack(side=tk.LEFT, padx=(4, 4))
        self.selector_manual_reset_button = tk.Button(
            overview_controls,
            text="Reset",
            command=self._reset_selector_manual_selection,
            state="disabled",
        )
        self.selector_manual_reset_button.pack(side=tk.LEFT, padx=(0, 4))
        self.selector_summary_label = tk.Label(
            summary_frame,
            text="No CSV loaded.",
            anchor="w",
            font=("TkDefaultFont", 10, "bold"),
        )
        self.selector_summary_label.pack(fill="x", padx=6, pady=(4, 2))
        score_canvas_container = tk.Frame(summary_frame)
        score_canvas_container.pack(fill="x", expand=True, padx=6, pady=(0, 4))
        self.selector_score_canvas = tk.Canvas(
            score_canvas_container,
            height=92,
            bg="#f4f4f4",
            highlightthickness=0,
            xscrollincrement=1,
        )
        self.selector_score_canvas.pack(fill="x", expand=True, side=tk.TOP)
        self.selector_score_canvas.bind("<Button-1>", self._on_selector_score_canvas_click)
        self.selector_score_canvas.bind("<Button-3>", self._on_selector_score_canvas_right_click)
        self.selector_score_canvas.bind("<MouseWheel>", self._on_selector_score_canvas_mousewheel)
        self.selector_score_canvas.bind("<Button-4>", self._on_selector_score_canvas_mousewheel)
        self.selector_score_canvas.bind("<Button-5>", self._on_selector_score_canvas_mousewheel)
        score_scrollbar = tk.Scrollbar(
            score_canvas_container,
            orient="horizontal",
            command=self.selector_score_canvas.xview,
        )
        score_scrollbar.pack(fill="x", side=tk.TOP)
        self.selector_score_canvas.configure(xscrollcommand=score_scrollbar.set)
        tk.Label(
            summary_frame,
            text="Legend: selected=teal, unselected=gray, suspect=red outline, preview set=blue outline, preview current=deep blue outline, manual edit=orange outline, left-click=toggle, right-click=preview toggle, wheel=zoom X",
            anchor="w",
        ).pack(fill="x", padx=6, pady=(0, 4))

        log_frame = tk.LabelFrame(container, text="Log")
        log_frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.selector_log = tk.Text(log_frame, wrap="word", height=18, cursor="arrow")
        self.selector_log.pack(fill="both", expand=True, padx=6, pady=4)
        self.selector_log.bind("<Key>", self._block_text_edit)
        self.selector_log.bind("<Button-1>", lambda event: self.selector_log.focus_set())
        self._set_text_widget(self.selector_log, "")

        self._on_selector_csv_mode_changed()

    def _build_human_mask_tab(self, parent: tk.Widget) -> None:
        container = tk.Frame(parent)
        container.pack(fill="both", expand=True)

        params = tk.LabelFrame(container, text="Parameters")
        params.pack(fill="x", padx=8, pady=8)

        self.human_vars = {
            "input": tk.StringVar(),
            "output": tk.StringVar(),
            "mode": tk.StringVar(value="mask"),
            "cpu": tk.BooleanVar(value=False),
            "include_shadow": tk.BooleanVar(value=False),
            "target_person": tk.BooleanVar(value=True),
            "target_bicycle": tk.BooleanVar(value=False),
            "target_car": tk.BooleanVar(value=False),
            "target_animal": tk.BooleanVar(value=False),
        }
        self.human_vars["input"].trace_add("write", self._on_human_input_changed)
        self.human_vars["output"].trace_add("write", self._on_human_output_changed)

        row = 0
        tk.Label(params, text="Input folder").grid(row=row, column=0, sticky="e", padx=4, pady=4)
        tk.Entry(params, textvariable=self.human_vars["input"], width=52).grid(row=row, column=1, sticky="we", padx=4, pady=4)
        tk.Button(
            params,
            text="Browse...",
            command=lambda: self._select_directory(
                self.human_vars["input"],
                title="Select input folder",
                on_select=self._on_human_input_selected,
            ),
        ).grid(row=row, column=2, padx=4, pady=4)

        row += 1
        tk.Label(params, text="Output folder").grid(row=row, column=0, sticky="e", padx=4, pady=4)
        tk.Entry(params, textvariable=self.human_vars["output"], width=52).grid(row=row, column=1, sticky="we", padx=4, pady=4)
        tk.Button(
            params,
            text="Browse...",
            command=lambda: self._select_directory(self.human_vars["output"], title="Select output folder"),
        ).grid(row=row, column=2, padx=4, pady=4)

        row += 1
        mode_frame = tk.Frame(params)
        mode_frame.grid(row=row, column=0, columnspan=3, sticky="we", pady=4)
        tk.Label(mode_frame, text="Mode").pack(side=tk.LEFT, padx=(0, 4))
        mode_combo = ttk.Combobox(
            mode_frame,
            textvariable=self.human_vars["mode"],
            values=HUMAN_MODE_CHOICES,
            state="readonly",
            width=16,
        )
        mode_combo.pack(side=tk.LEFT, padx=(0, 12))
        mode_combo.set(self.human_vars["mode"].get())

        row += 1
        target_frame = tk.Frame(params)
        target_frame.grid(row=row, column=0, columnspan=3, sticky="w", pady=4)
        tk.Label(target_frame, text="Targets").pack(side=tk.LEFT, padx=(0, 8))
        tk.Checkbutton(
            target_frame,
            text="Person",
            variable=self.human_vars["target_person"],
        ).pack(side=tk.LEFT, padx=(0, 8))
        tk.Checkbutton(
            target_frame,
            text="Bicycle",
            variable=self.human_vars["target_bicycle"],
        ).pack(side=tk.LEFT, padx=(0, 8))
        tk.Checkbutton(
            target_frame,
            text="Car",
            variable=self.human_vars["target_car"],
        ).pack(side=tk.LEFT, padx=(0, 8))
        tk.Checkbutton(
            target_frame,
            text="Animal (Bird/Cat/Dog)",
            variable=self.human_vars["target_animal"],
        ).pack(side=tk.LEFT, padx=(0, 8))

        row += 1
        flags_frame = tk.Frame(params)
        flags_frame.grid(row=row, column=0, columnspan=3, sticky="w", pady=4)
        tk.Checkbutton(flags_frame, text="Force CPU", variable=self.human_vars["cpu"]).pack(side=tk.LEFT, padx=(0, 12))
        tk.Checkbutton(
            flags_frame,
            text="Include shadow",
            variable=self.human_vars["include_shadow"],
        ).pack(side=tk.LEFT, padx=(0, 12))

        for col in range(3):
            params.grid_columnconfigure(col, weight=1 if col == 1 else 0)

        actions = tk.Frame(container)
        actions.pack(fill="x", padx=8, pady=(0, 8))
        self.human_stop_button = tk.Button(
            actions,
            text="Stop",
            command=lambda: self._stop_cli_process("human"),
        )
        self.human_stop_button.pack(side=tk.RIGHT, padx=4, pady=4)
        self.human_stop_button.configure(state="disabled")
        self.human_run_button = tk.Button(
            actions,
            text="Run gs360_SegmentationMaskTool",
            command=self._run_human_mask_tool,
        )
        self.human_run_button.pack(side=tk.RIGHT, padx=4, pady=4)

        log_frame = tk.LabelFrame(container, text="Log")
        log_frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.human_log = tk.Text(log_frame, wrap="word", height=18, cursor="arrow")
        self.human_log.pack(fill="both", expand=True, padx=6, pady=4)
        self.human_log.bind("<Key>", self._block_text_edit)
        self.human_log.bind("<Button-1>", lambda event: self.human_log.focus_set())
        self._set_text_widget(self.human_log, "")

    def _on_human_input_changed(self, *_args) -> None:
        self._update_human_default_output()

    def _on_human_output_changed(self, *_args) -> None:
        if self._human_output_updating:
            return
        self._human_output_auto = False

    def _human_default_output_for_input(self, input_text: str) -> Optional[str]:
        text = input_text.strip()
        if not text:
            return None
        try:
            base_path = Path(text).expanduser()
        except Exception:
            return None
        source_dir = base_path.parent if base_path.suffix else base_path
        parent_dir = source_dir.parent
        if parent_dir == source_dir:
            return str(source_dir / "_mask")
        return str(parent_dir / "_mask")

    def _update_human_default_output(self, input_text: Optional[str] = None) -> None:
        if not self.human_vars:
            return
        if input_text is None:
            input_text = self.human_vars["input"].get()
        default_out = self._human_default_output_for_input(input_text)
        if not default_out:
            return
        current_output = self.human_vars["output"].get().strip()
        should_update = (
            self._human_output_auto
            or not current_output
            or current_output == self._human_last_auto_output
        )
        if not should_update:
            return
        self._human_output_auto = True
        self._human_last_auto_output = default_out
        self._human_output_updating = True
        try:
            self.human_vars["output"].set(default_out)
        finally:
            self._human_output_updating = False

    def _on_human_input_selected(self, selected_path: str) -> None:
        self._update_human_default_output(selected_path)

    def _msxml_base_dir(self, input_text: str) -> Optional[Path]:
        text = input_text.strip()
        if not text:
            return None
        try:
            path_obj = Path(text).expanduser()
        except Exception:
            return None
        return path_obj.parent if path_obj.suffix else path_obj

    def _msxml_default_output_for_input(self, input_text: str) -> Optional[str]:
        base_dir = self._msxml_base_dir(input_text)
        if base_dir is None:
            return None
        return str(base_dir / "perspective_cams")

    def _msxml_default_cut_input(self, input_text: str) -> Optional[str]:
        base_dir = self._msxml_base_dir(input_text)
        if base_dir is None:
            return None
        return str(base_dir / "360imgs")

    def _msxml_default_points_ply(self, input_text: str) -> Optional[str]:
        base_dir = self._msxml_base_dir(input_text)
        if base_dir is None:
            return None
        candidate = base_dir / "pointcloud" / "pointcloud.ply"
        if candidate.exists():
            return str(candidate)
        fallback = base_dir / "pointcloud.ply"
        if fallback.exists():
            return str(fallback)
        return str(candidate)

    def _update_msxml_auto_paths(self, input_text: Optional[str] = None) -> None:
        if not self.msxml_vars:
            return
        if input_text is None:
            input_text = self.msxml_vars["xml"].get()

        default_out = self._msxml_default_output_for_input(input_text)
        if default_out:
            current_output = self.msxml_vars["output"].get().strip()
            should_update = (
                self._msxml_output_auto
                or not current_output
                or current_output == self._msxml_last_auto_output
            )
            if should_update:
                self._msxml_output_auto = True
                self._msxml_last_auto_output = default_out
                self._msxml_output_updating = True
                try:
                    self.msxml_vars["output"].set(default_out)
                finally:
                    self._msxml_output_updating = False

        # NOTE: cut input and points PLY are user-specified (no auto-fill).

    def _on_msxml_input_changed(self, *_args) -> None:
        self._update_msxml_auto_paths()

    def _on_msxml_input_selected(self, selected_path: str) -> None:
        self._update_msxml_auto_paths(selected_path)

    def _on_msxml_output_changed(self, *_args) -> None:
        if self._msxml_output_updating:
            return
        self._msxml_output_auto = False

    def _on_msxml_cut_input_changed(self, *_args) -> None:
        if self._msxml_cut_input_updating:
            return
        self._msxml_cut_input_auto = False

    def _on_msxml_points_ply_changed(self, *_args) -> None:
        if self._msxml_points_ply_updating:
            return
        self._msxml_points_ply_auto = False

    def _update_msxml_cut_state(self) -> None:
        if not self.msxml_vars:
            return
        enabled = bool(self.msxml_vars["cut"].get())
        state = "normal" if enabled else "disabled"
        for widget in (self.msxml_cut_input_entry, self.msxml_cut_out_entry):
            if widget is None:
                continue
            try:
                widget.configure(state=state)
            except tk.TclError:
                pass

    def _format_allows_points_ply(self, format_value: str) -> bool:
        return format_value in {"colmap", "all", "transforms"}

    def _update_msxml_format_state(self) -> None:
        if not self.msxml_vars:
            return
        fmt_display = self.msxml_vars["format"].get().strip()
        fmt = MSXML_FORMAT_TO_CLI.get(fmt_display, fmt_display.lower())
        preset_combo = self.msxml_preset_combo
        if fmt == "metashape-multi-camera-system":
            if self.msxml_vars["preset"].get().strip() != "fisheyelike":
                self.msxml_vars["preset"].set("fisheyelike")
            if preset_combo is not None:
                try:
                    preset_combo.configure(
                        values=("fisheyelike",),
                        state="readonly",
                    )
                except tk.TclError:
                    pass
        elif preset_combo is not None:
            try:
                preset_combo.configure(
                    values=MSXML_PRESET_CHOICES,
                    state="readonly",
                )
            except tk.TclError:
                pass
        enabled = self._format_allows_points_ply(fmt)
        state = "normal" if enabled else "disabled"
        for widget in (
            self.msxml_points_entry,
            self.msxml_points_button,
            self.msxml_points_rotate_check,
        ):
            if widget is None:
                continue
            try:
                widget.configure(state=state)
            except tk.TclError:
                pass

    @staticmethod
    def _extract_multicam_view_id(stem: str) -> Optional[str]:
        """Extract trailing view ID token (e.g. A, A_U, A_D20) from a file stem."""
        pattern = r"_((?:[A-Z]|\d{2,})(?:_(?:U|D|U\d+|D\d+))?)$"
        match = re.search(pattern, stem.upper())
        if not match:
            return None
        return match.group(1)

    @staticmethod
    def _next_available_path(path: Path) -> Path:
        """Avoid collisions by appending a numeric suffix when needed."""
        if not path.exists():
            return path
        stem = path.stem
        suffix = path.suffix
        parent = path.parent
        index = 1
        while True:
            candidate = parent / f"{stem}_{index:03d}{suffix}"
            if not candidate.exists():
                return candidate
            index += 1

    def _run_msxml_multicam_folder_split(self) -> None:
        """Group perspective images into camera-ID folders for Metashape multi-cam import."""
        if not self.msxml_multicam_vars:
            return
        source_text = self.msxml_multicam_vars["source"].get().strip()
        dry_run = bool(self.msxml_multicam_vars["dry_run"].get())
        if not source_text:
            messagebox.showerror(
                "MS360xmlToPersCams",
                "Source image folder is required.",
            )
            return
        try:
            source_dir = Path(source_text).expanduser().resolve()
        except Exception as exc:
            messagebox.showerror("MS360xmlToPersCams", f"Invalid source path:\n{exc}")
            return
        if not source_dir.exists() or not source_dir.is_dir():
            messagebox.showerror(
                "MS360xmlToPersCams",
                f"Source folder not found:\n{source_dir}",
            )
            return

        image_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".exr"}
        files = sorted(
            p for p in source_dir.iterdir()
            if p.is_file() and p.suffix.lower() in image_exts
        )
        if not files:
            messagebox.showinfo(
                "MS360xmlToPersCams",
                f"No image files found in:\n{source_dir}",
            )
            return

        moved = 0
        skipped = 0
        split_counts: Dict[str, int] = defaultdict(int)
        unrecognized: List[str] = []
        for src_path in files:
            view_id = self._extract_multicam_view_id(src_path.stem)
            if not view_id:
                skipped += 1
                unrecognized.append(src_path.name)
                continue

            dest_dir = source_dir / view_id
            if not dry_run:
                dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / src_path.name
            try:
                if src_path.resolve() == dest_path.resolve():
                    skipped += 1
                    continue
            except Exception:
                pass
            dest_path = self._next_available_path(dest_path)
            try:
                if not dry_run:
                    shutil.move(str(src_path), str(dest_path))
                moved += 1
                split_counts[view_id] += 1
            except Exception:
                skipped += 1
                unrecognized.append(src_path.name)

        self._append_text_widget(
            self.msxml_log,
            (
                "[multicam] folder split completed"
                + (" [dry-run]" if dry_run else "")
                + ": "
                f"moved={moved}, skipped={skipped}, root={source_dir}"
            ),
        )
        if unrecognized:
            sample = ", ".join(unrecognized[:6])
            more = f" (+{len(unrecognized) - 6} more)" if len(unrecognized) > 6 else ""
            self._append_text_widget(
                self.msxml_log,
                f"[multicam] skipped examples: {sample}{more}",
            )
        if dry_run and split_counts:
            self._append_text_widget(
                self.msxml_log,
                "[multicam][dry-run] planned subfolders and counts:",
            )
            for folder_name in sorted(split_counts):
                self._append_text_widget(
                    self.msxml_log,
                    f"  - {folder_name}: {split_counts[folder_name]}",
                )
        messagebox.showinfo(
            "MS360xmlToPersCams",
            (
                "Folder split completed"
                + (" (dry run)." if dry_run else ".")
                + "\n"
                f"Moved: {moved}\nSkipped: {skipped}\n"
                f"Source folder: {source_dir}"
            ),
        )

    def _build_msxml_tab(self, parent: tk.Widget) -> None:
        container = tk.Frame(parent)
        container.pack(fill="both", expand=True)

        params = tk.LabelFrame(container, text="Parameters")
        params.pack(fill="x", padx=8, pady=8)

        self.msxml_vars = {
            "xml": tk.StringVar(),
            "output": tk.StringVar(),
            "preset": tk.StringVar(value="full360coverage"),
            "format": tk.StringVar(value="Metashape"),
            "ext": tk.StringVar(value="jpg"),
            "scale": tk.StringVar(value="1.0"),
            "world_axis": tk.StringVar(value="0 1 0"),
            "world_deg": tk.StringVar(value="0"),
            "cut": tk.BooleanVar(value=False),
            "cut_input": tk.StringVar(),
            "cut_out": tk.StringVar(),
            "points_ply": tk.StringVar(),
            "pc_rotate_x": tk.BooleanVar(value=True),
        }
        self.msxml_multicam_vars = {
            "source": tk.StringVar(),
            "dry_run": tk.BooleanVar(value=False),
        }
        self.msxml_vars["xml"].trace_add("write", self._on_msxml_input_changed)
        self.msxml_vars["output"].trace_add("write", self._on_msxml_output_changed)
        self.msxml_vars["cut_input"].trace_add(
            "write", self._on_msxml_cut_input_changed
        )
        self.msxml_vars["points_ply"].trace_add(
            "write", self._on_msxml_points_ply_changed
        )
        self.msxml_vars["cut"].trace_add(
            "write", lambda *_args: self._update_msxml_cut_state()
        )
        self.msxml_vars["format"].trace_add(
            "write", lambda *_args: self._update_msxml_format_state()
        )

        row = 0
        tk.Label(params, text="Input XML").grid(
            row=row, column=0, sticky="e", padx=4, pady=4
        )
        tk.Entry(params, textvariable=self.msxml_vars["xml"], width=52).grid(
            row=row, column=1, sticky="we", padx=4, pady=4
        )
        tk.Button(
            params,
            text="Browse...",
            command=lambda: self._select_file(
                self.msxml_vars["xml"],
                title="Select Metashape XML",
                filetypes=[("XML files", "*.xml"), ("All files", "*.*")],
                on_select=self._on_msxml_input_selected,
            ),
        ).grid(row=row, column=2, padx=4, pady=4)

        row += 1
        tk.Label(params, text="Output folder").grid(
            row=row, column=0, sticky="e", padx=4, pady=4
        )
        tk.Entry(params, textvariable=self.msxml_vars["output"], width=52).grid(
            row=row, column=1, sticky="we", padx=4, pady=4
        )
        tk.Button(
            params,
            text="Browse...",
            command=lambda: self._select_directory(
                self.msxml_vars["output"],
                title="Select output folder",
            ),
        ).grid(row=row, column=2, padx=4, pady=4)

        row += 1
        tk.Label(params, text="Preset").grid(
            row=row, column=0, sticky="e", padx=4, pady=4
        )
        preset_combo = ttk.Combobox(
            params,
            textvariable=self.msxml_vars["preset"],
            values=MSXML_PRESET_CHOICES,
            state="readonly",
            width=22,
        )
        preset_combo.grid(row=row, column=1, sticky="w", padx=4, pady=4)
        self.msxml_preset_combo = preset_combo

        row += 1
        tk.Label(params, text="Format / Ext").grid(
            row=row, column=0, sticky="e", padx=4, pady=4
        )
        format_frame = tk.Frame(params)
        format_frame.grid(row=row, column=1, sticky="w", padx=4, pady=4)
        tk.Label(format_frame, text="Format").pack(side=tk.LEFT, padx=(0, 4))
        format_combo = ttk.Combobox(
            format_frame,
            textvariable=self.msxml_vars["format"],
            values=MSXML_FORMAT_CHOICES,
            state="readonly",
            width=32,
        )
        format_combo.pack(side=tk.LEFT, padx=(0, 10))
        tk.Label(format_frame, text="Ext").pack(side=tk.LEFT, padx=(0, 4))
        ext_combo = ttk.Combobox(
            format_frame,
            textvariable=self.msxml_vars["ext"],
            values=("jpg", "png", "tif"),
            state="readonly",
            width=8,
        )
        ext_combo.pack(side=tk.LEFT)

        row += 1
        tk.Label(params, text="Scale factor / World").grid(
            row=row, column=0, sticky="e", padx=4, pady=4
        )
        world_frame = tk.Frame(params)
        world_frame.grid(row=row, column=1, sticky="w", padx=4, pady=4)
        tk.Label(world_frame, text="Scale factor").pack(side=tk.LEFT, padx=(0, 4))
        tk.Entry(
            world_frame,
            textvariable=self.msxml_vars["scale"],
            width=8,
        ).pack(side=tk.LEFT, padx=(0, 10))
        tk.Label(world_frame, text="Axis").pack(side=tk.LEFT, padx=(0, 4))
        tk.Entry(
            world_frame,
            textvariable=self.msxml_vars["world_axis"],
            width=12,
        ).pack(side=tk.LEFT, padx=(0, 10))
        tk.Label(world_frame, text="Deg").pack(side=tk.LEFT, padx=(0, 4))
        tk.Entry(
            world_frame,
            textvariable=self.msxml_vars["world_deg"],
            width=6,
        ).pack(side=tk.LEFT)

        row += 1
        persp_frame = tk.LabelFrame(params, text="PerspCut")
        persp_frame.grid(
            row=row,
            column=0,
            columnspan=3,
            sticky="we",
            padx=4,
            pady=(6, 4),
        )
        persp_frame.grid_columnconfigure(1, weight=1)

        p_row = 0
        tk.Checkbutton(
            persp_frame,
            text="PerspCut",
            variable=self.msxml_vars["cut"],
            command=self._update_msxml_cut_state,
        ).grid(row=p_row, column=0, columnspan=3, sticky="w", padx=4, pady=4)

        p_row += 1
        tk.Label(persp_frame, text="PerspCut input").grid(
            row=p_row, column=0, sticky="e", padx=4, pady=4
        )
        self.msxml_cut_input_entry = tk.Entry(
            persp_frame, textvariable=self.msxml_vars["cut_input"], width=52
        )
        self.msxml_cut_input_entry.grid(
            row=p_row, column=1, sticky="we", padx=4, pady=4
        )
        tk.Button(
            persp_frame,
            text="Browse...",
            command=lambda: self._select_directory(
                self.msxml_vars["cut_input"],
                title="Select cut input folder",
            ),
        ).grid(row=p_row, column=2, padx=4, pady=4)

        p_row += 1
        tk.Label(persp_frame, text="Cut out").grid(
            row=p_row, column=0, sticky="e", padx=4, pady=4
        )
        self.msxml_cut_out_entry = tk.Entry(
            persp_frame, textvariable=self.msxml_vars["cut_out"], width=52
        )
        self.msxml_cut_out_entry.grid(
            row=p_row, column=1, sticky="we", padx=4, pady=4
        )
        tk.Button(
            persp_frame,
            text="Browse...",
            command=lambda: self._select_directory(
                self.msxml_vars["cut_out"],
                title="Select cut output folder",
            ),
        ).grid(row=p_row, column=2, padx=4, pady=4)

        p_row += 1
        tk.Label(persp_frame, text="Points PLY").grid(
            row=p_row, column=0, sticky="e", padx=4, pady=4
        )
        self.msxml_points_entry = tk.Entry(
            persp_frame, textvariable=self.msxml_vars["points_ply"], width=52
        )
        self.msxml_points_entry.grid(
            row=p_row, column=1, sticky="we", padx=4, pady=4
        )
        self.msxml_points_button = tk.Button(
            persp_frame,
            text="Browse...",
            command=lambda: self._select_file(
                self.msxml_vars["points_ply"],
                title="Select points PLY",
                filetypes=[("PLY files", "*.ply"), ("All files", "*.*")],
            ),
        )
        self.msxml_points_button.grid(row=p_row, column=2, padx=4, pady=4)

        p_row += 1
        self.msxml_points_rotate_check = tk.Checkbutton(
            persp_frame,
            text="Rotate pointcloud X +180",
            variable=self.msxml_vars["pc_rotate_x"],
        )
        self.msxml_points_rotate_check.grid(
            row=p_row, column=1, sticky="w", padx=4, pady=4
        )

        params.grid_columnconfigure(1, weight=1)

        actions = tk.Frame(container)
        actions.pack(fill="x", padx=8, pady=(0, 8))
        self.msxml_stop_button = tk.Button(
            actions,
            text="Stop",
            command=lambda: self._stop_cli_process("msxml"),
        )
        self.msxml_stop_button.pack(side=tk.RIGHT, padx=4, pady=4)
        self.msxml_stop_button.configure(state="disabled")
        self.msxml_run_button = tk.Button(
            actions,
            text="Run gs360_MS360xmlToPersCams",
            command=self._run_msxml_tool,
        )
        self.msxml_run_button.pack(side=tk.RIGHT, padx=4, pady=4)

        split_frame = tk.LabelFrame(
            container,
            text="Folder Split for Metashape Multi-Camera System",
        )
        split_frame.pack(fill="x", padx=8, pady=(0, 8))
        split_frame.grid_columnconfigure(1, weight=1)

        s_row = 0
        tk.Label(split_frame, text="Source images").grid(
            row=s_row, column=0, sticky="e", padx=4, pady=4
        )
        tk.Entry(
            split_frame,
            textvariable=self.msxml_multicam_vars["source"],
            width=52,
        ).grid(row=s_row, column=1, sticky="we", padx=4, pady=4)
        tk.Button(
            split_frame,
            text="Browse...",
            command=lambda: self._select_directory(
                self.msxml_multicam_vars["source"],
                title="Select source image folder",
            ),
        ).grid(row=s_row, column=2, padx=4, pady=4)

        s_row += 1
        tk.Checkbutton(
            split_frame,
            text="Dry Run",
            variable=self.msxml_multicam_vars["dry_run"],
            anchor="w",
        ).grid(row=s_row, column=1, sticky="w", padx=4, pady=4)
        self.msxml_multicam_run_button = tk.Button(
            split_frame,
            text="Run Folder Split",
            command=self._run_msxml_multicam_folder_split,
        )
        self.msxml_multicam_run_button.grid(
            row=s_row, column=2, sticky="e", padx=4, pady=4
        )

        log_frame = tk.LabelFrame(container, text="Log")
        log_frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.msxml_log = tk.Text(log_frame, wrap="word", height=16, cursor="arrow")
        self.msxml_log.pack(fill="both", expand=True, padx=6, pady=4)
        self.msxml_log.bind("<Key>", self._block_text_edit)
        self.msxml_log.bind("<Button-1>", lambda event: self.msxml_log.focus_set())
        self._set_text_widget(self.msxml_log, "")
        self._update_msxml_cut_state()
        self._update_msxml_format_state()

    def _build_ply_tab(self, parent: tk.Widget) -> None:
        container = tk.Frame(parent)
        container.pack(fill="both", expand=True)

        body_pane = tk.PanedWindow(
            container,
            orient=tk.HORIZONTAL,
            sashrelief=tk.RAISED,
            sashwidth=6,
        )
        body_pane.pack(fill="both", expand=True, padx=8, pady=8)
        left_panel = tk.Frame(body_pane)
        right_panel = tk.Frame(body_pane)
        body_pane.add(left_panel, minsize=700, stretch="always")
        body_pane.add(right_panel, minsize=260, stretch="never")

        params = tk.LabelFrame(left_panel, text="Parameters")
        params.pack(fill="x", padx=0, pady=(0, 8))

        self.ply_vars = {
            "input": tk.StringVar(),
            "output": tk.StringVar(),
            "target_points": tk.StringVar(value="100000"),
            "target_percent": tk.StringVar(value="10"),
            "voxel_size": tk.StringVar(value="1"),
            "downsample_method": self.ply_downsample_method_var,
            "adaptive_weight": tk.StringVar(value="1.0"),
            "keep_strategy": tk.StringVar(value="centroid"),
            "append_ply": tk.StringVar(),
        }
        self.ply_vars["input"].trace_add("write", self._on_ply_input_changed)
        self.ply_vars["output"].trace_add("write", self._on_ply_output_changed)
        self.ply_vars["downsample_method"].trace_add(
            "write", self._update_ply_adaptive_state
        )
        self._ply_view_high_max_points_var.trace_add(
            "write", self._on_ply_high_max_points_var_changed
        )
        self._ply_target_var_map = {
            "points": self.ply_vars["target_points"],
            "percent": self.ply_vars["target_percent"],
            "voxel": self.ply_vars["voxel_size"],
        }

        row = 0
        tk.Label(params, text="Input PLY").grid(row=row, column=0, sticky="e", padx=4, pady=4)
        tk.Entry(params, textvariable=self.ply_vars["input"], width=52).grid(row=row, column=1, sticky="we", padx=4, pady=4)
        tk.Button(
            params,
            text="Browse...",
            command=lambda: self._select_file(
                self.ply_vars["input"],
                title="Select input PLY",
                filetypes=[("PLY files", "*.ply"), ("All files", "*.*")],
                on_select=self._on_ply_input_selected,
            ),
        ).grid(row=row, column=2, padx=4, pady=4)

        row += 1
        tk.Label(params, text="Output PLY").grid(row=row, column=0, sticky="e", padx=4, pady=4)
        tk.Entry(params, textvariable=self.ply_vars["output"], width=52).grid(row=row, column=1, sticky="we", padx=4, pady=4)
        tk.Button(
            params,
            text="Browse...",
            command=lambda: self._select_save_file(self.ply_vars["output"], title="Select output PLY", defaultextension=".ply", filetypes=[("PLY files", "*.ply"), ("All files", "*.*")]),
        ).grid(row=row, column=2, padx=4, pady=4)

        row += 1
        downsample_frame = tk.Frame(params)
        downsample_frame.grid(row=row, column=0, columnspan=3, sticky="we", pady=4)
        tk.Label(downsample_frame, text="Target mode").pack(side=tk.LEFT, padx=(0, 4))
        mode_labels = tuple(self._ply_mode_key_map.keys())
        mode_combo = ttk.Combobox(
            downsample_frame,
            textvariable=self.ply_target_mode_var,
            values=mode_labels,
            state="readonly",
            width=18,
        )
        mode_combo.pack(side=tk.LEFT, padx=(0, 12))
        mode_combo.bind("<<ComboboxSelected>>", self._on_ply_target_mode_changed)
        self._ply_target_value_label = tk.Label(downsample_frame, text="Points")
        self._ply_target_value_label.pack(side=tk.LEFT, padx=(0, 4))
        current_mode_key = self._ply_mode_key_map.get(self.ply_target_mode_var.get(), "points")
        current_var = self._ply_target_var_map.get(current_mode_key, self.ply_vars["target_points"])
        self._ply_target_value_entry = tk.Entry(
            downsample_frame,
            textvariable=current_var,
            width=12,
        )
        self._ply_target_value_entry.pack(side=tk.LEFT, padx=(0, 12))
        tk.Label(downsample_frame, text="Downsample").pack(side=tk.LEFT, padx=(0, 4))
        method_labels = tuple(self._ply_downsample_method_key_map.keys())
        ttk.Combobox(
            downsample_frame,
            textvariable=self.ply_downsample_method_var,
            values=method_labels,
            state="readonly",
            width=22,
        ).pack(side=tk.LEFT, padx=(0, 12))
        tk.Label(downsample_frame, text="Weight").pack(side=tk.LEFT, padx=(0, 4))
        self.ply_adaptive_weight_entry = tk.Entry(
            downsample_frame,
            textvariable=self.ply_vars["adaptive_weight"],
            width=8,
        )
        self.ply_adaptive_weight_entry.pack(side=tk.LEFT, padx=(0, 12))
        tk.Label(downsample_frame, text="Keep strategy").pack(side=tk.LEFT, padx=(0, 4))
        self.ply_keep_menu = ttk.Combobox(
            downsample_frame,
            textvariable=self.ply_vars["keep_strategy"],
            values=("centroid", "center", "first", "random"),
            state="readonly",
            width=12,
        )
        self.ply_keep_menu.pack(side=tk.LEFT, padx=(0, 4))
        self._update_ply_target_value_widgets()

        row += 1
        tk.Label(params, text="Append PLY files").grid(
            row=row, column=0, sticky="e", padx=4, pady=4
        )
        append_entry = tk.Entry(
            params,
            textvariable=self.ply_vars["append_ply"],
            width=52,
        )
        append_entry.grid(row=row, column=1, sticky="we", padx=4, pady=4)
        self.ply_append_entry = append_entry
        tk.Button(
            params,
            text="Browse...",
            command=self._browse_ply_append_files,
        ).grid(row=row, column=2, padx=4, pady=4)

        row += 1
        params_actions = tk.Frame(params)
        params_actions.grid(row=row, column=0, columnspan=3, sticky="we", pady=(4, 4))
        self.ply_stop_button = tk.Button(
            params_actions,
            text="Stop",
            command=lambda: self._stop_cli_process("ply"),
        )
        self.ply_stop_button.pack(side=tk.RIGHT, padx=4, pady=4)
        self.ply_stop_button.configure(state="disabled")
        self.ply_run_button = tk.Button(
            params_actions,
            text="Run gs360_PlyOptimizer",
            command=self._run_ply_optimizer,
        )
        self.ply_run_button.pack(side=tk.RIGHT, padx=4, pady=4)

        params.grid_columnconfigure(1, weight=1)

        viewer_frame = tk.LabelFrame(left_panel, text="Ply Viewer")
        viewer_frame.pack(fill="both", expand=True, padx=0, pady=0)
        self._ply_viewer_root = viewer_frame

        save_top_controls = tk.Frame(viewer_frame)
        save_top_controls.pack(fill="x", padx=12, pady=(8, 0))
        tk.Label(save_top_controls, text="Save viewed PLY").pack(
            side=tk.LEFT, padx=(0, 4)
        )
        sky_save_entry = tk.Entry(
            save_top_controls,
            textvariable=self._ply_sky_save_path_var,
            width=56,
        )
        sky_save_entry.pack(side=tk.LEFT, fill="x", expand=True)
        tk.Button(
            save_top_controls,
            text="Browse...",
            command=self._on_browse_sky_save_path,
        ).pack(side=tk.LEFT, padx=(4, 4))
        tk.Button(
            save_top_controls,
            text="Save",
            command=self._on_save_sky_points,
        ).pack(side=tk.LEFT)

        viewer_actions = tk.Frame(viewer_frame)
        viewer_actions.pack(fill="x", padx=12, pady=(6, 0))
        self.ply_input_view_button = tk.Button(
            viewer_actions,
            text="Show Input PLY",
            command=self._on_show_input_ply,
        )
        self.ply_input_view_button.pack(side=tk.LEFT, padx=(0, 4), pady=0)
        self.ply_view_button = tk.Button(
            viewer_actions,
            text="Show Output PLY",
            command=self._on_show_ply,
        )
        self.ply_view_button.pack(side=tk.LEFT, padx=4, pady=0)
        self.ply_clear_view_button = tk.Button(
            viewer_actions,
            text="Clear",
            command=self._on_clear_ply_view,
        )
        self.ply_clear_view_button.pack(side=tk.LEFT, padx=(4, 0), pady=0)
        tk.Checkbutton(
            viewer_actions,
            text="Monochrome",
            variable=self._ply_monochrome_var,
            command=self._redraw_ply_canvas,
        ).pack(side=tk.LEFT, padx=(12, 0))
        tk.Label(viewer_actions, text="Projection").pack(side=tk.LEFT, padx=(12, 4))
        self._ply_projection_combo = ttk.Combobox(
            viewer_actions,
            textvariable=self._ply_projection_mode,
            values=("Orthographic", "Perspective"),
            state="readonly",
            width=14,
        )
        self._ply_projection_combo.pack(side=tk.LEFT, padx=(0, 4))
        self._ply_projection_combo.bind("<<ComboboxSelected>>", self._on_ply_projection_changed)
        tk.Label(viewer_actions, text="Low quality pts").pack(
            side=tk.LEFT, padx=(12, 4)
        )
        interactive_max_points_entry = tk.Entry(
            viewer_actions,
            textvariable=self._ply_view_interactive_max_points_var,
            width=9,
        )
        interactive_max_points_entry.pack(side=tk.LEFT)
        interactive_max_points_entry.bind(
            "<Return>", self._on_ply_interactive_max_points_commit
        )
        interactive_max_points_entry.bind(
            "<FocusOut>", self._on_ply_interactive_max_points_commit
        )
        self._ply_interactive_max_points_entry = interactive_max_points_entry
        tk.Label(viewer_actions, text="High quality pts").pack(
            side=tk.LEFT, padx=(12, 4)
        )
        high_max_points_entry = tk.Entry(
            viewer_actions,
            textvariable=self._ply_view_high_max_points_var,
            width=9,
        )
        high_max_points_entry.pack(side=tk.LEFT)
        high_max_points_entry.bind(
            "<Return>", self._on_ply_high_max_points_commit
        )
        high_max_points_entry.bind(
            "<FocusOut>", self._on_ply_high_max_points_commit
        )
        self._ply_high_max_points_entry = high_max_points_entry

        remove_color_controls = tk.Frame(viewer_frame)
        remove_color_controls.pack(fill="x", padx=12, pady=(4, 0))
        tk.Label(remove_color_controls, text="Remove color").pack(
            side=tk.LEFT, padx=(0, 4)
        )
        remove_color_display = tk.Label(
            remove_color_controls,
            textvariable=self._ply_remove_color_rgb_var,
            fg="#87cefa",
        )
        remove_color_display.pack(side=tk.LEFT, padx=(4, 4))
        self._ply_remove_color_label = remove_color_display
        self._update_remove_color_display(
            (135, 206, 250),
            self._ply_remove_color_var.get(),
        )
        tk.Button(
            remove_color_controls,
            text="Pick Remove Color",
            command=self._on_pick_remove_color,
        ).pack(side=tk.LEFT, padx=(4, 4))
        tk.Label(remove_color_controls, text="Tol (RGB)").pack(
            side=tk.LEFT, padx=(12, 4)
        )
        tk.Entry(
            remove_color_controls,
            textvariable=self._ply_remove_color_tol_var,
            width=6,
        ).pack(side=tk.LEFT)
        tk.Button(
            remove_color_controls,
            text="Remove Color Points",
            command=self._on_remove_color_points,
        ).pack(side=tk.LEFT, padx=(8, 4))
        tk.Button(
            remove_color_controls,
            text="Reset",
            command=self._on_reset_ply_view_state,
        ).pack(side=tk.LEFT, padx=(4, 0))

        sky_controls = tk.Frame(viewer_frame)
        sky_controls.pack(fill="x", padx=12, pady=(4, 0))
        tk.Label(sky_controls, text="Sky color").pack(side=tk.LEFT, padx=(12, 4))
        sky_color_display = tk.Label(
            sky_controls,
            textvariable=self._ply_sky_color_rgb_var,
            fg="#87cefa",
        )
        sky_color_display.pack(side=tk.LEFT, padx=(4, 4))
        self._ply_sky_color_label = sky_color_display
        self._update_sky_color_display((135, 206, 250), self._ply_sky_color_var.get())
        tk.Button(
            sky_controls,
            text="Pick Sky Color",
            command=self._on_pick_sky_color,
        ).pack(side=tk.LEFT, padx=(4, 4))
        tk.Button(
            sky_controls,
            text="Auto Pick Sky Color",
            command=self._on_auto_pick_sky_color,
        ).pack(side=tk.LEFT, padx=(4, 0))

        sky_action_controls = tk.Frame(viewer_frame)
        sky_action_controls.pack(fill="x", padx=12, pady=(4, 0))
        axis_options = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")
        button_row = tk.Frame(sky_action_controls)
        button_row.pack(side=tk.LEFT, padx=(0, 12))
        tk.Label(button_row, text="Sky axis").pack(side=tk.LEFT, padx=(0, 4))
        axis_combo = ttk.Combobox(
            button_row,
            textvariable=self._ply_sky_axis_var,
            values=axis_options,
            state="readonly",
            width=6,
        )
        axis_combo.pack(side=tk.LEFT)
        tk.Label(button_row, text="Sky scale").pack(side=tk.LEFT, padx=(12, 4))
        sky_scale_entry = tk.Entry(
            button_row,
            textvariable=self._ply_sky_scale_var,
            width=8,
        )
        sky_scale_entry.pack(side=tk.LEFT)
        tk.Label(button_row, text="Sky points").pack(side=tk.LEFT, padx=(12, 4))
        sky_count_entry = tk.Entry(
            button_row,
            textvariable=self._ply_sky_count_var,
            width=10,
        )
        sky_count_entry.pack(side=tk.LEFT, padx=(0, 12))
        tk.Button(
            button_row,
            text="Add Sky PointClouds (Preview)",
            command=self._on_add_sky_points,
        ).pack(side=tk.LEFT, padx=(0, 12))
        tk.Button(
            button_row,
            text="Clear Sky",
            command=self._on_clear_sky_points,
        ).pack(side=tk.LEFT)

        canvas = tk.Canvas(
            viewer_frame,
            width=PLY_VIEW_CANVAS_WIDTH,
            height=PLY_VIEW_CANVAS_HEIGHT,
            bg="#101010",
            highlightthickness=0,
        )
        canvas.pack(padx=12, pady=12, anchor="center")
        self._ply_canvas_image_id = canvas.create_image(0, 0, anchor="nw")
        self._ply_view_canvas = canvas
        canvas.bind("<ButtonPress-1>", self._on_ply_drag_start)
        canvas.bind("<B1-Motion>", self._on_ply_drag_move)
        canvas.bind("<ButtonRelease-1>", self._on_ply_drag_end)
        canvas.bind("<ButtonPress-3>", self._on_ply_pan_start)
        canvas.bind("<B3-Motion>", self._on_ply_pan_move)
        canvas.bind("<ButtonRelease-3>", self._on_ply_pan_end)
        canvas.bind("<Leave>", self._on_ply_drag_end)
        canvas.bind("<MouseWheel>", self._on_ply_zoom)
        canvas.bind("<Button-4>", self._on_ply_zoom)
        canvas.bind("<Button-5>", self._on_ply_zoom)

        log_frame = tk.LabelFrame(right_panel, text="Log")
        log_frame.pack(fill="both", expand=True, padx=0, pady=0)
        self.ply_log = tk.Text(log_frame, wrap="word", height=28, cursor="arrow")
        self.ply_log.pack(fill="both", expand=True, padx=6, pady=4)
        self.ply_log.bind("<Key>", self._block_text_edit)
        self.ply_log.bind("<Button-1>", lambda event: self.ply_log.focus_set())
        self._set_text_widget(self.ply_log, "")

        def _init_ply_sash() -> None:
            try:
                w = max(640, body_pane.winfo_width())
                body_pane.sash_place(0, int(w * 0.74), 1)
            except Exception:
                pass

        self.root.after_idle(_init_ply_sash)
        self._redraw_ply_canvas()
        self._update_ply_adaptive_state()

    def _build_preview_tab(self, parent: tk.Widget) -> None:
        main = tk.Frame(parent)
        main.pack(fill="both", expand=True)

        work_frame = tk.Frame(main)
        work_frame.pack(fill="both", expand=True, padx=8, pady=(8, 4))

        left_right_wrapper = tk.Frame(work_frame)
        left_right_wrapper.pack(fill="both", expand=True)

        self.preview_frame = tk.LabelFrame(left_right_wrapper, text="Preview")
        self.preview_frame.pack(side=tk.LEFT, padx=(0, 6), pady=0, fill="y", expand=False, anchor="n")
        self.preview_frame.pack_propagate(False)
        self.preview_frame.configure(
            width=self.display_width + 16,
            height=self.display_height + 16,
        )

        self.left_frame = tk.Frame(self.preview_frame, bg=self.root.cget("bg"), width=self.display_width, height=self.display_height)
        self.left_frame.pack(fill="y", expand=False)
        self.left_frame.pack_propagate(False)

        self.canvas = tk.Canvas(
            self.left_frame,
            width=self.display_width,
            height=self.display_height,
            highlightthickness=0,
            bg="#202020",
        )
        self.canvas.pack(side=tk.TOP, fill="both", expand=True)
        self.canvas.create_text(
            self.display_width / 2,
            self.display_height / 2,
            text="Select an input folder",
            fill="#d0d0d0",
            tags=("placeholder",),
        )

        self.controls_frame = tk.LabelFrame(left_right_wrapper, text="Controls", width=520)
        self.controls_frame.pack(side=tk.LEFT, fill="y", expand=False, padx=(6, 0), pady=0)
        self.controls_frame.pack_propagate(False)
        self.right_inner = tk.Frame(self.controls_frame)
        self.right_inner.pack(fill="both", expand=True)

        folder_frame = tk.LabelFrame(self.right_inner, text="Input (image folder / video)")
        folder_frame.pack(fill="x")
        path_entry = tk.Entry(folder_frame, textvariable=self.folder_path_var, width=38)
        path_entry.pack(side=tk.LEFT, fill="x", expand=True, padx=(8, 4), pady=4)
        self._bind_help(path_entry, "input_path")
        tk.Button(
            folder_frame,
            text="Browse video...",
            command=self._browse_video_input,
        ).pack(side=tk.RIGHT, padx=(4, 8), pady=4)
        tk.Button(
            folder_frame,
            text="Browse images...",
            command=self._browse_image_input,
        ).pack(side=tk.RIGHT, padx=(4, 4), pady=4)

        output_frame = tk.LabelFrame(self.right_inner, text="Output Folder")
        output_frame.pack(fill="x", pady=(4, 4))
        out_entry = tk.Entry(output_frame, textvariable=self.output_path_var, width=38)
        out_entry.pack(side=tk.LEFT, fill="x", expand=True, padx=(8, 4), pady=4)
        tk.Button(
            output_frame,
            text="Browse...",
            command=self.browse_output_directory,
        ).pack(side=tk.RIGHT, padx=(4, 8), pady=4)

        controls = tk.LabelFrame(self.right_inner, text="Crop Parameters")
        controls.pack(fill="x", pady=(8, 4))

        columns = 3
        field_positions: Dict[str, Tuple[int, int]] = {}
        for idx, definition in enumerate(self.FIELD_DEFS):
            name = definition["name"]
            field_type = definition["type"]
            base_row = idx // columns
            base_col = (idx % columns) * 2
            row = base_row + int(definition.get("row_shift", 0))
            col = base_col
            anchor = definition.get("align_with")
            if anchor:
                anchor_pos = field_positions.get(anchor)
                if anchor_pos:
                    row = anchor_pos[0] + int(definition.get("row_shift", 0))
                    col = anchor_pos[1] + int(definition.get("col_shift", 2))
            field_positions[name] = (row, col)
            video_only = bool(definition.get("video_only"))
            # Defer video-only fps/keep_rec709 layout to custom frame below.
            if name in {"fps", "start", "end"}:
                var = tk.StringVar()
                self.field_vars[name] = var
                self.field_widgets[name] = None
                continue
            if name == "keep_rec709":
                var = tk.BooleanVar(value=False)
                self.field_vars[name] = var
                self.field_widgets[name] = None
                continue
            if field_type == "bool":
                var = tk.BooleanVar(value=False)
                widget = tk.Checkbutton(controls, text=definition["label"], variable=var, anchor="w")
                widget.configure(width=20)
                if name == "show_seam_overlay":
                    widget.configure(command=lambda v=var: self._on_show_seam_toggle(v))
                widget.grid(row=row, column=col, columnspan=2, sticky="w", padx=4, pady=2)
                self._bind_help(widget, name)
                self.field_vars[name] = var
                self.field_widgets[name] = widget
                if video_only and not self.source_is_video:
                    widget.configure(state="disabled")
                continue
            if field_type == "choice":
                var = tk.StringVar(value=definition["choices"][0])
                label = tk.Label(controls, text=definition["label"])
                label.grid(row=row, column=col, sticky="e", padx=4, pady=2)
                self._bind_help(label, name)
                combo = ttk.Combobox(
                    controls,
                    textvariable=var,
                    values=definition["choices"],
                    state="readonly",
                    width=definition.get("width", 12),
                )
                if name == "preset":
                    combo.bind("<<ComboboxSelected>>", lambda _event, v=var: self.on_preset_changed(v.get()))
                elif name == "ext":
                    combo.bind("<<ComboboxSelected>>", lambda _event, v=var: self.on_ext_changed(v.get()))
                combo.grid(row=row, column=col + 1, sticky="we", padx=4, pady=2)
                self._bind_help(combo, name)
                self.field_vars[name] = var
                self.field_widgets[name] = combo
                if video_only and not self.source_is_video:
                    combo.configure(state="disabled")
                continue
            label = tk.Label(controls, text=definition["label"])
            label.grid(row=row, column=col, sticky="e", padx=4, pady=2)
            self._bind_help(label, name)
            var = tk.StringVar()
            entry = tk.Entry(controls, textvariable=var, width=definition.get("width", 12))
            entry.grid(row=row, column=col + 1, sticky="we", padx=4, pady=2)
            self._bind_help(entry, name)
            self.field_vars[name] = var
            self.field_widgets[name] = entry
            if video_only and not self.source_is_video:
                entry.configure(state="disabled")

        max_col_index = 0
        if field_positions:
            max_col_index = max(pos[1] for pos in field_positions.values())
        for col in range(max_col_index + 2):
            controls.grid_columnconfigure(col, weight=1 if col % 2 else 0)

        video_frame = tk.LabelFrame(controls, text="Video (direct export)")
        video_frame.grid(
            row=max(pos[0] for pos in field_positions.values()) + 1 if field_positions else 0,
            column=0,
            columnspan=max_col_index + 2,
            sticky="we",
            padx=4,
            pady=(6, 4),
        )
        for col in (1, 3, 5, 6, 7):
            video_frame.columnconfigure(col, weight=1, minsize=25)

        keep_var = self.field_vars.get("keep_rec709")
        keep_chk = self.field_widgets.get("keep_rec709")
        if keep_var is not None and keep_chk is None:
            keep_chk = tk.Checkbutton(video_frame, text="Convert Rec.709 to sRGB", variable=keep_var, anchor="w")
            keep_chk.configure(width=20)
            self.field_widgets["keep_rec709"] = keep_chk
        if keep_chk is not None:
            keep_chk.grid(row=0, column=0, columnspan=2, sticky="w", padx=6, pady=2)

        tk.Label(video_frame, text="FPS").grid(row=0, column=2, sticky="e", padx=4, pady=2)
        fps_var = self.field_vars.get("fps")
        fps_entry = self.field_widgets.get("fps")
        if fps_var is not None and fps_entry is None:
            fps_entry = tk.Entry(video_frame, textvariable=fps_var, width=6)
            self.field_widgets["fps"] = fps_entry
        if fps_entry is not None:
            fps_entry.grid(row=0, column=3, sticky="we", padx=4, pady=2)

        tk.Label(video_frame, text="Start (s)").grid(row=0, column=4, sticky="e", padx=4, pady=2)
        start_var = self.field_vars.get("start")
        start_entry = self.field_widgets.get("start")
        if start_var is not None and start_entry is None:
            start_entry = tk.Entry(video_frame, textvariable=start_var, width=6)
            self.field_widgets["start"] = start_entry
        if start_entry is not None:
            start_entry.grid(row=0, column=5, sticky="we", padx=4, pady=2)

        tk.Label(video_frame, text="End (s)").grid(row=0, column=6, sticky="e", padx=4, pady=2)
        end_var = self.field_vars.get("end")
        end_entry = self.field_widgets.get("end")
        if end_var is not None and end_entry is None:
            end_entry = tk.Entry(video_frame, textvariable=end_var, width=6)
            self.field_widgets["end"] = end_entry
        if end_entry is not None:
            end_entry.grid(row=0, column=7, sticky="we", padx=4, pady=2)

        tk.Label(video_frame, text="CSV (selected frames)").grid(row=1, column=0, sticky="e", padx=4, pady=2)
        self.preview_csv_entry = tk.Entry(video_frame, textvariable=self.preview_csv_var, width=28)
        self.preview_csv_entry.grid(row=1, column=1, columnspan=4, sticky="we", padx=4, pady=2)
        self.preview_csv_button = tk.Button(
            video_frame,
            text="Browse...",
            command=self._browse_preview_csv,
            width=10,
        )
        self.preview_csv_button.grid(row=1, column=5, columnspan=2, sticky="w", padx=4, pady=2)
        try:
            self.preview_csv_var.trace_add("write", lambda *_args: self._update_preview_csv_state())
        except Exception:
            pass
        self._update_preview_csv_state()

        ffmpeg_frame = tk.LabelFrame(self.right_inner, text="ffmpeg Jobs")
        ffmpeg_frame.pack(fill="x", pady=(4, 4))
        ffmpeg_frame.columnconfigure(1, weight=1)

        jobs_label = tk.Label(ffmpeg_frame, text="Parallel jobs")
        jobs_label.grid(row=0, column=0, padx=4, pady=2, sticky="e")
        self._bind_help(jobs_label, "jobs")
        jobs_entry = tk.Entry(ffmpeg_frame, textvariable=self.jobs_var, width=10)
        jobs_entry.grid(row=0, column=1, padx=4, pady=2, sticky="we")
        self._bind_help(jobs_entry, "jobs")

        self.preview_inspect_button = tk.Button(
            ffmpeg_frame,
            text="Inspect video",
            command=self._inspect_preview_video_metadata,
            state="disabled",
            width=14,
        )
        self.preview_inspect_button.grid(row=0, column=2, padx=4, pady=2, sticky="e")

        buttons_frame = tk.LabelFrame(self.right_inner, text="Actions")
        buttons_frame.pack(fill="x", pady=(8, 12), ipady=6)

        buttons_inner = tk.Frame(buttons_frame)
        buttons_inner.pack(fill="both", expand=True, padx=14, pady=(10, 12))
        buttons_inner.columnconfigure(0, weight=1)
        buttons_inner.columnconfigure(1, weight=1)
        buttons_inner.columnconfigure(2, weight=0)
        buttons_inner.grid_rowconfigure(0, weight=1)
        self.update_button = tk.Button(
            buttons_inner,
            text="Refresh preview",
            command=self.on_update,
        )
        self.update_button.grid(row=0, column=0, padx=(0, 12), pady=0, sticky="nsew", ipadx=10, ipady=8)
        self.execute_button = tk.Button(
            buttons_inner,
            text="Run Export",
            command=self.on_execute,
        )
        self.execute_button.grid(row=0, column=1, padx=(12, 0), pady=0, sticky="nsew", ipadx=10, ipady=8)
        self.preview_stop_button = tk.Button(
            buttons_inner,
            text="Stop",
            command=self._stop_preview_execution,
        )
        self.preview_stop_button.grid(row=0, column=2, padx=(12, 0), pady=0, sticky="nsew", ipadx=10, ipady=8)
        self.preview_stop_button.configure(state="disabled")

        log_frame = tk.LabelFrame(main, text="Log")
        log_frame.pack(fill="both", expand=False, padx=8, pady=(0, 4))
        self.log_text = tk.Text(log_frame, wrap="word", cursor="arrow", height=14)
        self.log_text.pack(fill="both", expand=True, padx=6, pady=4)
        self.log_text.bind("<Key>", self._block_text_edit)
        self.log_text.bind("<Button-1>", lambda event: self.log_text.focus_set())
        self.set_log_text("")

        help_frame = tk.LabelFrame(main, text="Help")
        help_frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.help_text = tk.Text(help_frame, wrap="word", height=8, cursor="arrow")
        self.help_text.pack(fill="both", expand=True, padx=6, pady=4)
        self.help_text.insert("1.0", HELP_TEXT)
        self.help_text.bind("<Key>", self._block_text_edit)
        self.help_text.bind("<Button-1>", lambda event: self.help_text.focus_set())
        self._update_jpeg_quality_state()
        self._update_preview_inspect_state()
        self._sync_panel_heights()

    def _build_config_tab(self, parent: tk.Widget) -> None:
        container = tk.Frame(parent)
        container.pack(fill="both", expand=True)

        params = tk.LabelFrame(container, text="ffmpeg")
        params.pack(fill="x", padx=8, pady=8)
        params.columnconfigure(1, weight=1)

        ffmpeg_label = tk.Label(params, text="ffmpeg path")
        ffmpeg_label.grid(row=0, column=0, padx=4, pady=4, sticky="e")
        self._bind_help(ffmpeg_label, "ffmpeg")
        ffmpeg_entry = tk.Entry(params, textvariable=self.ffmpeg_path_var, width=52)
        ffmpeg_entry.grid(row=0, column=1, padx=4, pady=4, sticky="we")
        self._bind_help(ffmpeg_entry, "ffmpeg")
        browse_btn = tk.Button(params, text="Browse...", command=self.browse_ffmpeg)
        browse_btn.grid(row=0, column=2, padx=(4, 0), pady=4)
        self._bind_help(browse_btn, "ffmpeg")
        reset_btn = tk.Button(params, text="Reset", command=self.reset_ffmpeg_path)
        reset_btn.grid(row=0, column=3, padx=(4, 4), pady=4)
        self._bind_help(reset_btn, "ffmpeg")

    def _set_text_widget(self, widget: Optional[tk.Text], text: str) -> None:
        if widget is None:
            return
        widget.configure(state="normal")
        widget.delete("1.0", tk.END)
        widget.insert("1.0", text)
        widget.see("1.0")

    def _append_text_widget(self, widget: Optional[tk.Text], line: str) -> None:
        if widget is None:
            return
        widget.configure(state="normal")
        current = widget.get("1.0", tk.END).strip()
        prefix = "\n" if current else ""
        widget.insert(tk.END, prefix + line)
        widget.see(tk.END)

    def _run_cli_command(
        self,
        cmd: Sequence[str],
        log_widget: Optional[tk.Text],
        run_button: Optional[tk.Button],
        process_key: str,
        stop_button: Optional[tk.Button] = None,
        cwd: Optional[Path] = None,
        clear_log: bool = True,
    ) -> None:
        if log_widget is None:
            return
        if not process_key:
            raise ValueError("process_key is required for CLI commands")

        with self._process_lock:
            if process_key in self._active_processes:
                self._append_text_widget(log_widget, "[warn] Process already running. Stop it before launching a new one.")
                return

        command_text = "CLI> " + " ".join(shlex.quote(str(token)) for token in cmd)
        if clear_log:
            self._set_text_widget(log_widget, command_text)
        else:
            self._append_text_widget(log_widget, command_text)

        if run_button is not None:
            run_button.configure(state="disabled")
        if stop_button is not None:
            stop_button.configure(state="disabled")
            stop_button.configure(command=lambda key=process_key: self._stop_cli_process(key))

        def worker() -> None:
            try:
                process = subprocess.Popen(
                    list(cmd),
                    cwd=str(cwd) if cwd else None,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
            except Exception as exc:
                self.root.after(0, lambda err=exc: self._append_text_widget(log_widget, f"[ERR] {err}"))
                if run_button is not None:
                    self.root.after(0, lambda: run_button.configure(state="normal"))
                if stop_button is not None:
                    self.root.after(0, lambda: stop_button.configure(state="disabled"))
                return

            def reader() -> None:
                assert process.stdout is not None
                for raw in process.stdout:
                    line = raw.rstrip("\n")
                    self.root.after(0, lambda l=line: self._append_text_widget(log_widget, l))

            start_time = time.perf_counter()
            with self._process_lock:
                self._active_processes[process_key] = process
                self._process_ui[process_key] = {
                    "run": run_button,
                    "stop": stop_button,
                    "log": log_widget,
                }
                self._process_start_times[process_key] = start_time

            if stop_button is not None:
                self.root.after(0, lambda: stop_button.configure(state="normal"))

            stdout_thread = threading.Thread(target=reader, name=f"{process_key}-stdout", daemon=True)
            stdout_thread.start()

            rc = process.wait()
            stdout_thread.join(timeout=0.5)
            self.root.after(0, lambda rc=rc: self._on_cli_completed(process_key, rc if rc is not None else 0, stopped=False))

        threading.Thread(target=worker, name=f"{process_key}-runner", daemon=True).start()

    def _on_cli_completed(self, key: str, rc: int, stopped: bool) -> None:
        with self._process_lock:
            process = self._active_processes.pop(key, None)
            info = self._process_ui.pop(key, {})
            start_time = self._process_start_times.pop(key, None)
        run_button = info.get("run")
        stop_button = info.get("stop")
        log_widget = info.get("log")
        queue = self._queued_cli_commands.get(key)
        should_continue = bool(queue) and not stopped and rc == 0
        if not should_continue and run_button is not None:
            run_button.configure(state="normal")
        if stop_button is not None:
            stop_button.configure(state="disabled")
        if log_widget is not None:
            message = "[stop] terminated" if stopped else f"[exit] return code {rc}"
            self._append_text_widget(log_widget, message)
        if process is not None and process.poll() is None:
            try:
                process.terminate()
            except Exception:
                pass
        if should_continue and queue is not None:
            next_cmd, next_cwd, clear_log, label = queue.pop(0)
            if not queue:
                self._queued_cli_commands.pop(key, None)
            if run_button is not None:
                run_button.configure(state="disabled")
            if stop_button is not None:
                stop_button.configure(state="disabled")
            if log_widget is not None and label:
                self._append_text_widget(log_widget, label)
            self._run_cli_command(
                next_cmd,
                log_widget,
                run_button,
                process_key=key,
                stop_button=stop_button,
                cwd=next_cwd,
                clear_log=clear_log,
            )
            return
        else:
            self._queued_cli_commands.pop(key, None)
        if key == "selector":
            duration_message = None
            if start_time is not None:
                elapsed = max(0.0, time.perf_counter() - start_time)
                duration_message = f"[time] FrameSelector elapsed: {elapsed:.2f}s"
            pending = self.selector_auto_fetch_pending
            self.selector_auto_fetch_pending = False
            mode_now = "none"
            if self.selector_vars:
                mode_var = self.selector_vars.get("csv_mode")
                if mode_var is not None:
                    mode_now = mode_var.get().strip()
            if pending and not stopped and rc == 0 and mode_now in {"write", "apply", "reselect"}:
                if duration_message:
                    self._selector_duration_message = duration_message
                    self._selector_duration_pending = True
                if mode_now.lower() == "write" and self.selector_vars:
                    csv_mode_var = self.selector_vars.get("csv_mode")
                    if csv_mode_var is not None:
                        csv_mode_var.set("reselect")
                        self._on_selector_csv_mode_changed()
                self.root.after(100, self._show_selector_scores)
            elif duration_message and log_widget is not None:
                self._append_text_widget(log_widget, duration_message)
        elif key == "ply":
            if start_time is not None and log_widget is not None:
                elapsed = max(0.0, time.perf_counter() - start_time)
                self._append_text_widget(
                    log_widget,
                    f"[time] PlyOptimizer elapsed: {elapsed:.2f}s",
                )
            if not stopped and rc == 0:
                self.root.after(100, self._auto_show_ply_output_after_run)

    def _auto_show_ply_output_after_run(self) -> None:
        path = self._resolve_ply_display_path()
        if path is None:
            return
        if not path.exists():
            return
        self._show_ply_from_path(path, "output")

    def _stop_cli_process(self, key: str) -> None:
        with self._process_lock:
            process = self._active_processes.get(key)
            info = self._process_ui.get(key, {})
        if process is None:
            return
        stop_button = info.get("stop")
        log_widget = info.get("log")
        if stop_button is not None:
            stop_button.configure(state="disabled")
        if log_widget is not None:
            self._append_text_widget(log_widget, "[stop] Terminating process...")

        def worker() -> None:
            try:
                if process.poll() is None:
                    terminated = False
                    if sys.platform.startswith("win"):
                        pid = process.pid
                        if pid:
                            try:
                                subprocess.run(
                                    ["taskkill", "/T", "/F", "/PID", str(pid)],
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL,
                                    check=False,
                                )
                                terminated = True
                            except Exception as exc:
                                if log_widget is not None:
                                    self.root.after(
                                        0,
                                        lambda e=exc: self._append_text_widget(
                                            log_widget,
                                            f"[stop] taskkill failed: {e}",
                                        ),
                                    )
                    if not terminated:
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
            except Exception as exc:
                if log_widget is not None:
                    self.root.after(0, lambda e=exc: self._append_text_widget(log_widget, f"[ERR] stop failed: {e}"))
            finally:
                rc = process.poll()
                self.root.after(0, lambda: self._on_cli_completed(key, rc if rc is not None else -1, stopped=True))

        threading.Thread(target=worker, name=f"{key}-stopper", daemon=True).start()

    def _stop_preview_execution(self) -> None:
        if not self.is_executing:
            return
        cutter.stop_event.set()
        self._output_monitor_stop.set()
        if self.preview_stop_button is not None:
            self.preview_stop_button.configure(state="disabled")
        self.append_log_line("[EXEC] Stop requested...")

    def _run_human_mask_tool(self) -> None:
        if not self.human_vars:
            return
        input_dir = self.human_vars["input"].get().strip()
        if not input_dir:
            messagebox.showerror(
                "gs360_SegmentationMaskTool",
                "Input folder is required.",
            )
            return
        input_path = Path(input_dir).expanduser()
        if not input_path.exists() or not input_path.is_dir():
            messagebox.showerror(
                "gs360_SegmentationMaskTool",
                f"Input folder not found:\n{input_dir}",
            )
            return

        cmd: List[str] = [
            sys.executable,
            str(self.cli_tools_dir / "gs360_SegmentationMaskTool.py"),
            "-i",
            str(input_path),
        ]

        output_dir = self.human_vars["output"].get().strip()
        if output_dir:
            cmd.extend(["-o", output_dir])

        mode_value = self.human_vars["mode"].get().strip()
        if mode_value and mode_value in HUMAN_MODE_CHOICES:
            cmd.extend(["--mode", mode_value])

        selected_targets: List[str] = []
        for target_name in HUMAN_TARGET_CHOICES:
            key = f"target_{target_name}"
            if bool(self.human_vars[key].get()):
                selected_targets.append(target_name)

        if not selected_targets:
            messagebox.showerror(
                "gs360_SegmentationMaskTool",
                "Select at least one target (Person, Bicycle, Car, Animal).",
            )
            return
        for target_name in selected_targets:
            cmd.extend(["--target", target_name])

        if bool(self.human_vars["cpu"].get()):
            cmd.append("--cpu")
        if bool(self.human_vars["include_shadow"].get()):
            cmd.append("--include_shadow")

        self._run_cli_command(
            cmd,
            self.human_log,
            self.human_run_button,
            process_key="human",
            stop_button=self.human_stop_button,
            cwd=self.cli_tools_dir,
        )

    def _run_msxml_tool(self) -> None:
        if not self.msxml_vars:
            return
        xml_text = self.msxml_vars["xml"].get().strip()
        if not xml_text:
            messagebox.showerror("gs360_MS360xmlToPersCams", "Input XML is required.")
            return
        xml_path = Path(xml_text).expanduser()
        if not xml_path.exists():
            messagebox.showerror(
                "gs360_MS360xmlToPersCams",
                f"XML file not found:\n{xml_text}",
            )
            return

        cmd: List[str] = [
            sys.executable,
            str(self.cli_tools_dir / "gs360_MS360xmlToPersCams.py"),
            str(xml_path),
        ]

        output_dir = self.msxml_vars["output"].get().strip()
        if output_dir:
            cmd.extend(["-o", output_dir])

        format_display = self.msxml_vars["format"].get().strip()
        format_value = MSXML_FORMAT_TO_CLI.get(
            format_display, format_display.lower()
        )
        if format_value:
            cmd.extend(["--format", format_value])

        preset_value = self.msxml_vars["preset"].get().strip()
        if format_value == "metashape-multi-camera-system":
            preset_value = "fisheyelike"
            self.msxml_vars["preset"].set(preset_value)
        if preset_value:
            cmd.extend(["--preset", preset_value])

        if format_value == "metashape-multi-camera-system":
            warn_lines = [
                "Metashape-Multi-Camera-System output is experimental and "
                "requires Metashape Pro.",
            ]
            warn_lines.append("Continue conversion?")
            proceed = messagebox.askokcancel(
                "gs360_MS360xmlToPersCams",
                "\n".join(warn_lines),
            )
            if not proceed:
                return

        ext_value = self.msxml_vars["ext"].get().strip()
        if ext_value:
            cmd.extend(["--ext", ext_value])

        scale_text = self.msxml_vars["scale"].get().strip()
        if scale_text:
            try:
                float(scale_text)
            except ValueError:
                messagebox.showerror(
                    "gs360_MS360xmlToPersCams", "Scale factor must be numeric."
                )
                return
            cmd.extend(["--scale", scale_text])

        axis_text = self.msxml_vars["world_axis"].get().strip()
        if axis_text:
            parts = axis_text.replace(",", " ").split()
            if len(parts) != 3:
                messagebox.showerror(
                    "gs360_MS360xmlToPersCams",
                    "World axis must have 3 values (x y z).",
                )
                return
            try:
                [float(p) for p in parts]
            except ValueError:
                messagebox.showerror(
                    "gs360_MS360xmlToPersCams",
                    "World axis must be numeric.",
                )
                return
            cmd.extend(["--world-rot-axis", axis_text])

        world_deg = self.msxml_vars["world_deg"].get().strip()
        if world_deg:
            try:
                float(world_deg)
            except ValueError:
                messagebox.showerror(
                    "gs360_MS360xmlToPersCams", "World deg must be numeric."
                )
                return
            cmd.extend(["--world-rot-deg", world_deg])

        if bool(self.msxml_vars["cut"].get()):
            cmd.append("--persp-cut")
            cut_input = self.msxml_vars["cut_input"].get().strip()
            if cut_input:
                cmd.extend(["--cut-input", cut_input])
            cut_out = self.msxml_vars["cut_out"].get().strip()
            if cut_out:
                cmd.extend(["--cut-out", cut_out])

        if format_value in {"colmap", "all"}:
            points_ply = self.msxml_vars["points_ply"].get().strip()
            if not points_ply:
                messagebox.showerror(
                    "gs360_MS360xmlToPersCams",
                    "Points PLY is required for COLMAP output.",
                )
                return
            points_path = Path(points_ply).expanduser()
            if not points_path.exists():
                messagebox.showerror(
                    "gs360_MS360xmlToPersCams",
                    f"Points PLY not found:\n{points_ply}",
                )
                return
            cmd.extend(["--points-ply", points_ply])
            if bool(self.msxml_vars["pc_rotate_x"].get()):
                cmd.append("--pc-rotate-x-plus180")
        elif format_value == "transforms":
            points_ply = self.msxml_vars["points_ply"].get().strip()
            if points_ply:
                points_path = Path(points_ply).expanduser()
                if not points_path.exists():
                    messagebox.showerror(
                        "gs360_MS360xmlToPersCams",
                        f"Points PLY not found:\n{points_ply}",
                    )
                    return
                cmd.extend(["--points-ply", points_ply])
                if bool(self.msxml_vars["pc_rotate_x"].get()):
                    cmd.append("--pc-rotate-x-plus180")

        self._run_cli_command(
            cmd,
            self.msxml_log,
            self.msxml_run_button,
            process_key="msxml",
            stop_button=self.msxml_stop_button,
            cwd=self.cli_tools_dir,
        )

    def _run_video_tool(self) -> None:
        if not self.video_vars:
            return
        video_path = self.video_vars["video"].get().strip()
        if not video_path:
            messagebox.showerror("gs360_Video2Frames", "Input video is required.")
            return

        fps_value = self.video_vars["fps"].get().strip()
        try:
            fps_float = float(fps_value)
            if fps_float <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("gs360_Video2Frames", "FPS must be a positive number.")
            return

        fps_formatted = self._format_fps_for_output(fps_value) or f"{fps_float}"

        base_cmd: List[str] = [
            sys.executable,
            str(self.cli_tools_dir / "gs360_Video2Frames.py"),
            "-i",
            video_path,
            "-f",
            fps_formatted,
        ]

        output_dir = self.video_vars["output"].get().strip()
        if output_dir:
            base_cmd.extend(["-o", output_dir])

        ext_value = self.video_vars["ext"].get().strip()
        if ext_value:
            base_cmd.extend(["--ext", ext_value])

        prefix_value = self.video_vars.get("prefix")
        if prefix_value is not None:
            prefix_text = prefix_value.get().strip()
            if prefix_text:
                base_cmd.extend(["--prefix", prefix_text])

        start_value = self.video_vars["start"].get().strip()
        if start_value:
            try:
                float(start_value)
            except ValueError:
                messagebox.showerror("gs360_Video2Frames", "Start time must be numeric.")
                return
            base_cmd.extend(["--start", start_value])

        end_value = self.video_vars["end"].get().strip()
        if end_value:
            try:
                float(end_value)
            except ValueError:
                messagebox.showerror("gs360_Video2Frames", "End time must be numeric.")
                return
            base_cmd.extend(["--end", end_value])

        convert_to_srgb = bool(self.video_vars["keep_rec709"].get())
        if not convert_to_srgb:
            base_cmd.append("--keep-rec709")
        if bool(self.video_vars["overwrite"].get()):
            base_cmd.append("--overwrite")

        fisheye_enabled = bool(self.video_vars["fisheye_experimental"].get())
        dual_perspective_enabled = fisheye_enabled and bool(
            self.video_vars["fisheye_perspective"].get()
        )
        dual_focal_value: Optional[float] = None
        dual_fov_value: Optional[float] = None
        dual_size_value: Optional[int] = None
        if dual_perspective_enabled:
            fov_text = self.video_vars["fisheye_input_fov"].get().strip()
            if not fov_text:
                messagebox.showerror(
                    "gs360_Video2Frames",
                    "Enter an input FOV (degrees) for the dual fisheye perspective option.",
                )
                return
            try:
                dual_fov_value = float(fov_text)
            except ValueError:
                messagebox.showerror(
                    "gs360_Video2Frames",
                    "Input FOV must be numeric (degrees).",
                )
                return
            if dual_fov_value <= 0.0:
                messagebox.showerror(
                    "gs360_Video2Frames",
                    "Input FOV must be greater than zero.",
                )
                return
            focal_text = self.video_vars["fisheye_persp_focal"].get().strip()
            if not focal_text:
                messagebox.showerror(
                    "gs360_Video2Frames",
                    "Enter an output focal length (mm, approximate) for the dual fisheye perspective option.",
                )
                return
            try:
                dual_focal_value = float(focal_text)
            except ValueError:
                messagebox.showerror(
                    "gs360_Video2Frames",
                    "Output focal must be numeric (mm, approximate value).",
                )
                return
            if dual_focal_value <= 0.0:
                messagebox.showerror(
                    "gs360_Video2Frames",
                    "Output focal must be greater than zero.",
                )
                return
            size_text = self.video_vars["fisheye_persp_size"].get().strip()
            if not size_text:
                messagebox.showerror(
                    "gs360_Video2Frames",
                    "Enter an output size (pixels) for the dual fisheye perspective option.",
                )
                return
            try:
                dual_size_value = int(size_text)
            except ValueError:
                messagebox.showerror(
                    "gs360_Video2Frames",
                    "Output size must be an integer (pixels).",
                )
                return
            if dual_size_value <= 0:
                messagebox.showerror(
                    "gs360_Video2Frames",
                    "Output size must be greater than zero.",
                )
                return
            projection_value = self.video_vars["fisheye_projection"].get().strip().lower()
            if projection_value not in {"equidistant", "equisolid"}:
                messagebox.showerror(
                    "gs360_Video2Frames",
                    "Projection must be either 'equidistant' or 'equisolid'.",
                )
                return
            base_cmd.extend(["--fisheye-perspective"])
            base_cmd.extend(["--fisheye-input-fov", str(dual_fov_value)])
            base_cmd.extend(["--fisheye-focal-mm", str(dual_focal_value)])
            base_cmd.extend(["--fisheye-size", str(dual_size_value)])
            base_cmd.extend(["--fisheye-projection", projection_value])

        ffmpeg_path = self.ffmpeg_path_var.get().strip()
        if ffmpeg_path:
            base_cmd.extend(["--ffmpeg", ffmpeg_path])

        if fisheye_enabled:
            cmd_primary = list(base_cmd)
            cmd_secondary = list(base_cmd)
            cmd_primary.extend(
                ["--map-stream", "0:v:0", "--name-suffix", "_X"]
            )
            cmd_secondary.extend(
                ["--map-stream", "0:v:1", "--name-suffix", "_Y"]
            )
            self._queued_cli_commands.pop("video", None)
            self._queued_cli_commands["video"] = [
                (
                    cmd_secondary,
                    self.cli_tools_dir,
                    False,
                    "[next] Experimental dual fisheye (map 0:v:1)" if dual_perspective_enabled else "[next] Experimental fisheye (map 0:v:1)",
                ),
            ]
            self._run_cli_command(
                cmd_primary,
                self.video_log,
                self.video_run_button,
                process_key="video",
                stop_button=self.video_stop_button,
                cwd=self.cli_tools_dir,
                clear_log=True,
            )
            return

        self._run_cli_command(
            base_cmd,
            self.video_log,
            self.video_run_button,
            process_key="video",
            stop_button=self.video_stop_button,
            cwd=self.cli_tools_dir,
        )

    def _run_frame_selector(self) -> None:
        if not self.selector_vars:
            return

        self._selector_duration_message = None
        self._selector_duration_pending = False
        self.selector_auto_fetch_pending = False
        in_dir = self.selector_vars["in_dir"].get().strip()
        if not in_dir:
            messagebox.showerror("gs360_FrameSelector", "Input folder is required.")
            return

        cmd: List[str] = [
            sys.executable,
            str(self.cli_tools_dir / "gs360_FrameSelector.py"),
            "-i",
            in_dir,
        ]

        segment_text = self.selector_vars["segment_size"].get().strip()
        if not segment_text:
            messagebox.showerror(
                "gs360_FrameSelector",
                "Segment size is required. Enter a positive integer or 0/1 for per-frame mode.",
            )
            return
        try:
            segment_value = int(segment_text)
        except ValueError:
            messagebox.showerror("gs360_FrameSelector", "Segment size must be an integer.")
            return
        if segment_value < 0:
            messagebox.showerror("gs360_FrameSelector", "Segment size must be zero or greater.")
            return
        cmd.extend(["-n", str(segment_value)])

        dry_run_required = bool(self.selector_vars["dry_run"].get())
        augment_motion_enabled = bool(self.selector_vars["augment_motion"].get())

        workers_text = self.selector_vars["workers"].get().strip()
        auto_workers = self._auto_frame_selector_workers()
        max_workers = max(1, auto_workers * 2)
        if workers_text:
            if workers_text.lower() != "auto":
                try:
                    worker_value = int(workers_text)
                except ValueError:
                    messagebox.showerror("gs360_FrameSelector", "Workers must be an integer or 'auto'.")
                    return
                if worker_value <= 0:
                    messagebox.showerror("gs360_FrameSelector", "Workers must be a positive integer.")
                    return
                if worker_value > max_workers and worker_value <= 32:
                    messagebox.showerror(
                        "gs360_FrameSelector",
                        f"Workers must be <= {max_workers} (auto={auto_workers}).",
                    )
                    return
                if worker_value > 32:
                    proceed = messagebox.askokcancel(
                        "gs360_FrameSelector",
                        f"Workers is set to {worker_value} (recommended max {max_workers}). Continue?",
                    )
                    if not proceed:
                        return
                cmd.extend(["-w", str(worker_value)])

        ext_choice = self.selector_vars["ext"].get().strip()
        if ext_choice:
            cmd.extend(["-e", ext_choice])

        sort_choice = self.selector_vars["sort"].get().strip()
        if sort_choice:
            cmd.extend(["-s", sort_choice])

        score_backend = self.selector_vars["score_backend"].get().strip()
        if score_backend:
            cmd.extend(["--score_backend", score_backend])

        csv_mode = self.selector_vars["csv_mode"].get().strip()
        csv_path = self.selector_vars["csv_path"].get().strip()
        flow_recompute_needed = False
        if augment_motion_enabled and csv_mode in {"apply", "reselect"} and csv_path:
            flow_zero = self._csv_flow_motion_all_zero(csv_path, base_dir=in_dir)
            if flow_zero:
                flow_recompute_needed = True
                csv_mode = "write"
                self._append_text_widget(
                    self.selector_log,
                    "[info] flow_motion values are zero; recomputing metrics with motion before selection.",
                )
        if csv_mode and csv_mode != "none":
            if not csv_path:
                messagebox.showerror("gs360_FrameSelector", "CSV path is required for the selected mode.")
                return
            if csv_mode == "write":
                try:
                    csv_path_obj = Path(csv_path).expanduser()
                    if not csv_path_obj.is_absolute():
                        csv_path_obj = Path(in_dir).expanduser() / csv_path_obj
                except Exception:
                    csv_path_obj = None
                if csv_path_obj is not None and csv_path_obj.exists():
                    proceed = messagebox.askokcancel(
                        "FrameSelector CSV",
                        f"CSV already exists:\n{csv_path_obj}\nOverwrite it?",
                    )
                    if not proceed:
                        return
            if csv_mode == "write":
                cmd.extend(["-c", csv_path])
            elif csv_mode == "apply":
                cmd.extend(["-a", csv_path])
            elif csv_mode == "reselect":
                cmd.extend(["-r", csv_path])
                dry_run_required = True
        self.selector_auto_fetch_pending = csv_mode in {"write", "apply", "reselect"}

        if dry_run_required:
            cmd.append("--dry_run")

        crop_ratio = self.selector_vars["crop_ratio"].get().strip()
        if crop_ratio:
            try:
                float(crop_ratio)
            except ValueError:
                messagebox.showerror("gs360_FrameSelector", "Crop ratio must be numeric.")
                return
            cmd.extend(["--score_crop_ratio", crop_ratio])

        min_spacing = self.selector_vars["min_spacing_frames"].get().strip()
        if min_spacing.lower() == "auto":
            min_spacing = ""
        if min_spacing:
            try:
                int(min_spacing)
            except ValueError:
                messagebox.showerror("gs360_FrameSelector", "Min spacing must be an integer.")
                return
            cmd.extend(["--min_spacing_frames", min_spacing])

        if bool(self.selector_vars["augment_gap"].get()):
            cmd.append("--augment_gaps")
        else:
            cmd.append("--no_augment_gaps")
        if bool(self.selector_vars["augment_lowlight"].get()):
            cmd.append("--augment_lowlight")
        if augment_motion_enabled:
            cmd.append("--augment_motion")
        if not bool(self.selector_vars["segment_boundary_reopt"].get()):
            cmd.append("--no-segment-boundary-reopt")
        if not bool(self.selector_vars["ignore_highlights"].get()):
            cmd.append("--no-ignore-highlights")

        blur_percent_text = self.selector_count_var.get().strip() if self.selector_count_var is not None else ""
        blur_percent_text = blur_percent_text.rstrip("%")
        try:
            blur_percent_value = float(blur_percent_text) if blur_percent_text else 1.0
        except ValueError:
            messagebox.showerror("gs360_FrameSelector", "Check Selection percent must be a numeric percentage.")
            return
        blur_percent_value = max(0.0, min(blur_percent_value, 100.0))
        cmd.extend(["--blur-percent", f"{blur_percent_value}"])

        self._run_cli_command(
            cmd,
            self.selector_log,
            self.selector_run_button,
            process_key="selector",
            stop_button=self.selector_stop_button,
            cwd=self.cli_tools_dir,
        )

    def _update_ply_adaptive_state(self, *_args) -> None:
        method_var = self.ply_vars.get("downsample_method")
        method_label = method_var.get() if method_var is not None else "Voxel"
        method_key = self._ply_downsample_method_key_map.get(
            method_label, "voxel"
        )
        active = method_key == "adaptive"
        entry = self.ply_adaptive_weight_entry
        if entry is not None:
            state = "normal" if active else "disabled"
            try:
                entry.configure(state=state)
            except tk.TclError:
                pass
        menu = self.ply_keep_menu
        if menu is not None:
            try:
                menu.configure(state="readonly")
            except tk.TclError:
                pass

    @staticmethod
    def _auto_frame_selector_workers() -> int:
        cpu = os.cpu_count()
        if cpu is None or cpu <= 0:
            cpu = 4
        return max(1, cpu // 2)

    def _csv_flow_motion_all_zero(self, csv_path: str, base_dir: Optional[str] = None) -> Optional[bool]:
        """Return True when flow_motion exists and all numeric values are zero."""
        try:
            path_obj = Path(csv_path).expanduser()
            if not path_obj.is_absolute() and base_dir:
                path_obj = Path(base_dir).expanduser() / path_obj
            if not path_obj.exists():
                return None
            with path_obj.open("r", newline="") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    return None
                field_map = {name.lower(): name for name in reader.fieldnames}
                flow_key = field_map.get("flow_motion")
                if not flow_key:
                    return None
                any_values = False
                for row in reader:
                    raw = row.get(flow_key)
                    if raw is None:
                        continue
                    text = str(raw).strip()
                    if not text:
                        continue
                    any_values = True
                    try:
                        if float(text) != 0.0:
                            return False
                    except ValueError:
                        return False
                if any_values:
                    return True
                return None
        except Exception:
            return None

    def _load_selected_frames_from_csv(self, csv_path: Path) -> Tuple[List[int], int]:
        """Load selected frame indices and total rows from CSV."""
        indices: List[int] = []
        total_rows = 0
        path_obj = csv_path.expanduser()
        if not path_obj.exists():
            raise FileNotFoundError(f"CSV not found: {path_obj}")
        with path_obj.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
            field_map = {name.lower(): name for name in headers if name}
            selected_key = field_map.get("selected(1=keep)") or field_map.get("selected")
            index_key = field_map.get("index")
            filename_key = field_map.get("filename")
            if not selected_key:
                raise ValueError("CSV must contain 'selected(1=keep)' or 'selected' column.")
            for row_idx, row in enumerate(reader):
                total_rows += 1
                flag_text = str(row.get(selected_key, "")).strip().lower()
                selected = flag_text in {"1", "true", "yes", "keep"}
                if not selected:
                    continue
                if index_key and row.get(index_key) not in (None, ""):
                    try:
                        idx = int(row[index_key])
                    except (TypeError, ValueError):
                        idx = row_idx
                else:
                    idx = row_idx
                indices.append(idx)
        # Ensure stable ordering and uniqueness
        seen: Set[int] = set()
        unique_indices: List[int] = []
        for idx in indices:
            if idx in seen:
                continue
            seen.add(idx)
            unique_indices.append(idx)
        unique_indices.sort()
        return unique_indices, total_rows

    def _count_selected_frames_quiet(self, csv_path: Path) -> Optional[int]:
        """Return selected count from CSV without showing message boxes."""
        try:
            indices, _total = self._load_selected_frames_from_csv(csv_path)
            return len(indices)
        except Exception:
            return None

    @staticmethod
    def _auto_preview_jobs() -> int:
        cores = os.cpu_count()
        if cores is None or cores <= 0:
            cores = 1
        return max(1, cores // 2)

    def _effective_jobs(self) -> int:
        """Return worker count, halving auto when exporting directly from video."""
        requested = getattr(self.current_args, "jobs", "auto")
        base = cutter.parse_jobs(requested)
        if self.source_is_video and str(requested).lower() == "auto":
            return max(1, base // 2)
        return base

    def _update_ply_target_value_widgets(self) -> None:
        mode_label = self.ply_target_mode_var.get()
        mode_key = self._ply_mode_key_map.get(mode_label, "points")
        target_var = self._ply_target_var_map.get(mode_key)
        if target_var is None or self._ply_target_value_entry is None:
            return
        label_map = {
            "points": "Points",
            "percent": "Percent (%)",
            "voxel": "Voxel size",
        }
        if self._ply_target_value_label is not None:
            self._ply_target_value_label.configure(text=label_map.get(mode_key, "Value"))
        try:
            self._ply_target_value_entry.configure(textvariable=target_var)
        except tk.TclError:
            pass

    def _on_ply_target_mode_changed(self, _event=None) -> None:
        self._update_ply_target_value_widgets()

    def _on_ply_input_changed(self, *_args) -> None:
        self._update_ply_default_output()
        input_var = self.ply_vars.get("input")
        if input_var is None:
            return
        input_text = input_var.get().strip()
        if not input_text:
            return
        try:
            input_path = Path(input_text).expanduser()
        except Exception:
            return
        if input_path.exists():
            self._update_ply_high_max_default_from_path(input_path)

    def _on_ply_output_changed(self, *_args) -> None:
        if self._ply_output_updating:
            return
        self._ply_output_auto = False

    def _ply_default_output_for_input(self, input_text: str) -> Optional[str]:
        text = input_text.strip()
        if not text:
            return None
        try:
            path_obj = Path(text).expanduser()
        except Exception:
            return None
        suffix = path_obj.suffix if path_obj.suffix else ".ply"
        try:
            return str(path_obj.with_name(f"{path_obj.stem}_output{suffix}"))
        except Exception:
            return None

    def _update_ply_default_output(self, input_text: Optional[str] = None) -> None:
        if not self.ply_vars:
            return
        if input_text is None:
            input_text = self.ply_vars["input"].get()
        default_out = self._ply_default_output_for_input(input_text)
        if not default_out:
            return
        current_output = self.ply_vars["output"].get().strip()
        should_update = (
            self._ply_output_auto
            or not current_output
            or current_output == self._ply_last_auto_output
        )
        if not should_update:
            return
        self._ply_output_auto = True
        self._ply_last_auto_output = default_out
        self._ply_output_updating = True
        try:
            self.ply_vars["output"].set(default_out)
        finally:
            self._ply_output_updating = False

    def _on_ply_input_selected(self, selected_path: str) -> None:
        self._update_ply_default_output(selected_path)

    @staticmethod
    def _parse_ply_append_items(raw_text: str) -> List[str]:
        items: List[str] = []
        if not raw_text:
            return items
        for chunk in re.split(r"[;\r\n]+", raw_text):
            candidate = chunk.strip().strip('"').strip("'")
            if candidate:
                items.append(candidate)
        return items

    def _browse_ply_append_files(self) -> None:
        if not self.ply_vars:
            return
        append_var = self.ply_vars.get("append_ply")
        if append_var is None:
            return
        current_items = self._parse_ply_append_items(append_var.get().strip())
        initial_dir = self.base_dir
        if current_items:
            try:
                first_path = Path(current_items[0]).expanduser()
                initial_dir = first_path.parent if first_path.parent.exists() else self.base_dir
            except Exception:
                initial_dir = self.base_dir
        selected = filedialog.askopenfilenames(
            title="Select append PLY files",
            initialdir=str(initial_dir),
            filetypes=[("PLY files", "*.ply"), ("All files", "*.*")],
        )
        if not selected:
            return
        merged: List[str] = []
        for item in current_items + list(selected):
            text = str(item).strip()
            if text and text not in merged:
                merged.append(text)
        append_var.set("; ".join(merged))

    def _run_ply_optimizer(self) -> None:
        if not self.ply_vars:
            return
        input_path = self.ply_vars["input"].get().strip()
        if not input_path:
            messagebox.showerror("gs360_PlyOptimizer", "Input PLY is required.")
            return

        cmd: List[str] = [
            sys.executable,
            str(self.cli_tools_dir / "gs360_PlyOptimizer.py"),
            "-i",
            input_path,
        ]

        output_path = self.ply_vars["output"].get().strip()
        if output_path:
            cmd.extend(["-o", output_path])
        try:
            resolved_output = Path(output_path) if output_path else Path(input_path)
            self._last_ply_output_path = resolved_output.expanduser()
        except Exception:
            self._last_ply_output_path = None

        mode_label = self.ply_target_mode_var.get()
        mode_key = self._ply_mode_key_map.get(mode_label, "points")
        target_var = self._ply_target_var_map.get(mode_key)
        target_value = target_var.get().strip() if target_var is not None else ""

        if mode_key == "points" and target_value:
            try:
                int(target_value)
            except ValueError:
                messagebox.showerror("gs360_PlyOptimizer", "Target points must be an integer.")
                return
            cmd.extend(["-t", target_value])
        elif mode_key == "percent" and target_value:
            try:
                float(target_value)
            except ValueError:
                messagebox.showerror("gs360_PlyOptimizer", "Target percent must be numeric.")
                return
            cmd.extend(["-r", target_value])
        elif mode_key == "voxel" and target_value:
            try:
                float(target_value)
            except ValueError:
                messagebox.showerror("gs360_PlyOptimizer", "Voxel size must be numeric.")
                return
            cmd.extend(["-v", target_value])

        method_var = self.ply_vars.get("downsample_method")
        method_label = method_var.get() if method_var is not None else "Voxel"
        method_key = self._ply_downsample_method_key_map.get(
            method_label, "voxel"
        )
        cmd.extend(["--downsample-method", method_key])
        if method_key == "adaptive":
            weight_text = self.ply_vars["adaptive_weight"].get().strip()
            if weight_text:
                try:
                    float(weight_text)
                except ValueError:
                    messagebox.showerror("gs360_PlyOptimizer", "Adaptive weight must be numeric.")
                    return
                cmd.extend(["--adaptive-weight", weight_text])

        keep_strategy = self.ply_vars["keep_strategy"].get().strip()
        if keep_strategy:
            cmd.extend(["-k", keep_strategy])

        append_items: List[str] = []
        append_var = self.ply_vars.get("append_ply")
        if append_var is not None:
            append_items = self._parse_ply_append_items(append_var.get().strip())
        for item in append_items:
            cmd.extend(["-a", item])

        self._run_cli_command(
            cmd,
            self.ply_log,
            self.ply_run_button,
            process_key="ply",
            stop_button=self.ply_stop_button,
            cwd=self.cli_tools_dir,
        )

    def _resolve_ply_display_path(self) -> Optional[Path]:
        candidates: List[Path] = []
        if self._last_ply_output_path is not None:
            candidates.append(self._last_ply_output_path)
        for key in ("output", "input"):
            var = self.ply_vars.get(key)
            if var is None:
                continue
            text = var.get().strip()
            if not text:
                continue
            try:
                candidates.append(Path(text).expanduser())
            except Exception:
                continue
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0] if candidates else None

    def _get_ply_var_path(self, key: str) -> Optional[Path]:
        var = self.ply_vars.get(key)
        if var is None:
            return None
        text = var.get().strip()
        if not text:
            return None
        try:
            return Path(text).expanduser()
        except Exception:
            return None

    def _on_show_ply(self) -> None:
        path = self._resolve_ply_display_path()
        if path is None:
            messagebox.showerror("Show PLY", "No PLY path is configured yet.")
            return
        self._show_ply_from_path(path, "output")

    def _on_show_input_ply(self) -> None:
        path = self._get_ply_var_path("input")
        if path is None:
            messagebox.showerror("Show PLY", "Input PLY path is not set.")
            return
        self._show_ply_from_path(path, "input")

    def _on_clear_ply_view(self) -> None:
        if self._ply_interaction_after_id is not None:
            try:
                self.root.after_cancel(self._ply_interaction_after_id)
            except Exception:
                pass
            self._ply_interaction_after_id = None
        self._ply_is_interacting = False
        self._ply_view_info_var.set("PLY viewer is idle")
        self._ply_view_points = None
        self._ply_view_points_centered = None
        self._ply_view_colors = None
        self._ply_view_center = np.zeros(3, dtype=np.float32)
        self._ply_sky_points = None
        self._ply_sky_colors = None
        self._ply_loaded_points = None
        self._ply_loaded_colors = None
        self._ply_loaded_total_points = 0
        self._ply_loaded_sample_step = 1
        self._ply_view_total_points = 0
        self._ply_view_sample_step = 1
        self._ply_view_source_label = "PLY"
        self._ply_current_file_path = None
        self._ply_sky_save_path_var.set("")
        self._ply_view_rgb_mean = None
        remove_color = self._parse_color_to_rgb(self._ply_sky_color_var.get())
        if remove_color is not None:
            sky_hex = self._ply_sky_color_var.get()
            self._ply_remove_color_var.set(sky_hex)
            self._update_remove_color_display(remove_color, sky_hex)
        self._reset_ply_view_transform()
        self._redraw_ply_canvas()

    def _show_ply_from_path(self, path: Path, label: str) -> None:
        if not path.exists():
            messagebox.showerror("Show PLY", f"{path} was not found.")
            return
        self._update_ply_high_max_default_from_path(path)
        self._ply_view_source_label = label
        canvas = self._ensure_ply_viewer_window()
        if canvas is None:
            messagebox.showerror("Show PLY", "Ply Viewer canvas is not available.")
            return
        self._ply_view_info_var.set(f"Loading {label} ({path.name}) ...")
        self._redraw_ply_canvas()
        self._start_ply_async_load(path)

    def _ensure_ply_viewer_window(self) -> Optional[tk.Canvas]:
        canvas = self._ply_view_canvas
        if canvas is None or not canvas.winfo_exists():
            return None
        return canvas

    def _set_ply_view_loading_state(self, loading: bool) -> None:
        self._ply_view_is_loading = loading
        buttons = [
            self.ply_view_button,
            self.ply_input_view_button,
            self.ply_clear_view_button,
        ]
        for button in buttons:
            if button is None:
                continue
            try:
                button.configure(state="disabled" if loading else "normal")
            except tk.TclError:
                pass
        if self._ply_interactive_max_points_entry is not None:
            try:
                self._ply_interactive_max_points_entry.configure(
                    state="disabled" if loading else "normal"
                )
            except tk.TclError:
                pass
        if self._ply_high_max_points_entry is not None:
            try:
                self._ply_high_max_points_entry.configure(
                    state="disabled" if loading else "normal"
                )
            except tk.TclError:
                pass

    def _get_ply_view_interactive_max_points(
        self, show_error: bool = False
    ) -> Optional[int]:
        text = self._ply_view_interactive_max_points_var.get().strip()
        if not text:
            return PLY_VIEW_INTERACTIVE_MAX_POINTS
        try:
            value = int(float(text))
        except ValueError:
            if show_error:
                messagebox.showerror(
                    "PLY Viewer",
                    "Low quality pts must be numeric.",
                )
            return None
        if value <= 0:
            if show_error:
                messagebox.showerror(
                    "PLY Viewer",
                    "Low quality pts must be greater than zero.",
                )
            return None
        return value

    def _on_ply_high_max_points_var_changed(self, *_args) -> None:
        if self._ply_high_max_points_updating:
            return
        self._ply_high_max_points_auto = False

    def _set_ply_high_max_points_auto_value(self, value: int) -> None:
        if value <= 0:
            return
        value_text = str(int(value))
        self._ply_high_max_points_updating = True
        try:
            self._ply_view_high_max_points_var.set(value_text)
        finally:
            self._ply_high_max_points_updating = False
        self._ply_high_max_points_auto = True
        self._ply_last_auto_high_max_points = value_text

    def _read_ply_vertex_count_from_header(self, path: Path) -> Optional[int]:
        vertex_count: Optional[int] = None
        try:
            with path.open("rb") as fh:
                while True:
                    raw = fh.readline()
                    if not raw:
                        break
                    line = raw.decode("ascii", errors="ignore").strip()
                    if not line:
                        continue
                    if line.startswith("element"):
                        parts = line.split()
                        if len(parts) >= 3 and parts[1].lower() == "vertex":
                            try:
                                vertex_count = int(parts[2])
                            except ValueError:
                                vertex_count = None
                    if line == "end_header":
                        break
        except Exception:
            return None
        if vertex_count is None or vertex_count <= 0:
            return None
        return vertex_count

    def _update_ply_high_max_default_from_path(self, path: Path) -> None:
        vertex_count = self._read_ply_vertex_count_from_header(path)
        if vertex_count is None:
            return
        current = self._ply_view_high_max_points_var.get().strip()
        should_update = (
            self._ply_high_max_points_auto
            or not current
            or current == self._ply_last_auto_high_max_points
        )
        if not should_update:
            return
        self._set_ply_high_max_points_auto_value(vertex_count)

    def _get_ply_view_high_max_points(
        self, show_error: bool = False
    ) -> Optional[int]:
        text = self._ply_view_high_max_points_var.get().strip()
        if not text:
            return PLY_VIEW_MAX_POINTS
        try:
            value = int(float(text))
        except ValueError:
            if show_error:
                messagebox.showerror(
                    "PLY Viewer",
                    "High quality pts must be numeric.",
                )
            return None
        if value <= 0:
            if show_error:
                messagebox.showerror(
                    "PLY Viewer",
                    "High quality pts must be greater than zero.",
                )
            return None
        return value

    def _on_ply_interactive_max_points_commit(self, _event=None) -> Optional[str]:
        max_points = self._get_ply_view_interactive_max_points(show_error=True)
        if max_points is None:
            return "break"
        self._redraw_ply_canvas()
        return "break"

    def _on_ply_high_max_points_commit(self, _event=None) -> Optional[str]:
        max_points = self._get_ply_view_high_max_points(show_error=True)
        if max_points is None:
            return "break"
        if self._ply_view_is_loading:
            return "break"
        path = self._ply_current_file_path
        points = self._ply_view_points
        loaded_count = int(points.shape[0]) if isinstance(points, np.ndarray) else 0
        original_count = max(self._ply_view_total_points, loaded_count)
        target_count = min(max_points, original_count) if original_count > 0 else max_points
        should_reload = (
            path is not None
            and path.exists()
            and loaded_count > 0
            and loaded_count < target_count
        )
        if should_reload:
            self._ply_view_info_var.set(
                f"Reloading ({path.name}) for max pts={max_points:,} ..."
            )
            self._redraw_ply_canvas()
            self._start_ply_async_load(path)
        else:
            self._redraw_ply_canvas()
        return "break"

    def _start_ply_async_load(self, path: Path) -> None:
        max_points = self._get_ply_view_high_max_points(show_error=True)
        if max_points is None:
            return
        load_token: object = object()
        self._ply_view_load_token = load_token
        self._set_ply_view_loading_state(True)
        thread = threading.Thread(
            target=self._load_ply_async,
            args=(path, load_token, max_points),
            daemon=True,
        )
        self._ply_loader_thread = thread
        thread.start()

    def _load_ply_async(
        self, path: Path, token: object, max_points: int
    ) -> None:
        try:
            points, colors, original_count, sample_step = self._load_binary_ply_points(
                path,
                max_points=max_points,
            )
        except Exception as exc:  # pragma: no cover - UI thread handles dialog
            self.root.after(0, lambda err=exc: self._on_ply_load_error(token, err))
            return
        self.root.after(
            0,
            lambda: self._on_ply_load_success(
                token,
                path,
                points,
                colors,
                original_count,
                sample_step,
            ),
        )

    def _on_ply_load_success(
        self,
        token: object,
        path: Path,
        points: np.ndarray,
        colors: np.ndarray,
        original_count: int,
        sample_step: int,
    ) -> None:
        if token is not self._ply_view_load_token:
            return
        if self._ply_interaction_after_id is not None:
            try:
                self.root.after_cancel(self._ply_interaction_after_id)
            except Exception:
                pass
            self._ply_interaction_after_id = None
        self._ply_is_interacting = False
        self._set_ply_view_loading_state(False)
        point_count = int(points.shape[0]) if points.size else 0
        if point_count == 0:
            self._ply_view_rgb_mean = None
            self._ply_view_info_var.set(f"{path.name}: no points")
            self._redraw_ply_canvas()
            messagebox.showinfo("Show PLY", "No points were available for display.")
            return
        rgb_mean = colors.mean(axis=0)
        self._ply_view_rgb_mean = (
            float(rgb_mean[0]),
            float(rgb_mean[1]),
            float(rgb_mean[2]),
        )
        center = np.zeros(3, dtype=np.float32)
        spans = np.maximum(points.max(axis=0) - points.min(axis=0), 1e-6)
        max_extent = float(np.max(spans))
        centered_points = (points - center).astype(np.float32, copy=False)
        self._ply_view_points = points
        self._ply_view_points_centered = centered_points
        self._ply_view_colors = colors.astype(np.uint8, copy=False)
        self._ply_view_max_extent = max_extent
        self._ply_view_depth_offset = max_extent * 2.5
        self._ply_view_center = center.astype(np.float32, copy=False)
        self._ply_current_file_path = path
        self._ply_view_total_points = original_count
        self._ply_view_sample_step = max(1, sample_step)
        self._ply_loaded_points = points.astype(np.float32, copy=True)
        self._ply_loaded_colors = colors.astype(np.uint8, copy=True)
        self._ply_loaded_total_points = original_count
        self._ply_loaded_sample_step = max(1, sample_step)
        self._reset_ply_view_transform()
        label = self._ply_view_source_label or "PLY"
        if self._ply_view_sample_step > 1 and original_count > 0:
            info = (
                f"{label}: {path.name}  "
                f"({point_count:,} / {original_count:,} pts, step {self._ply_view_sample_step})"
            )
        else:
            info = f"{label}: {path.name}  ({point_count:,} pts)"
        self._ply_view_info_var.set(info)
        self._update_sky_save_default(path)
        default_sky_count = max(1, int(round(original_count * 0.05)))
        self._ply_sky_count_var.set(str(default_sky_count))
        remove_color = self._parse_color_to_rgb(self._ply_sky_color_var.get())
        if remove_color is not None:
            sky_hex = self._ply_sky_color_var.get()
            self._ply_remove_color_var.set(sky_hex)
            self._update_remove_color_display(remove_color, sky_hex)
        canvas = self._ensure_ply_viewer_window()
        if canvas is not None:
            self._render_ply_points(canvas)

    def _on_ply_load_error(self, token: object, exc: Exception) -> None:
        if token is not self._ply_view_load_token:
            return
        if self._ply_interaction_after_id is not None:
            try:
                self.root.after_cancel(self._ply_interaction_after_id)
            except Exception:
                pass
            self._ply_interaction_after_id = None
        self._ply_is_interacting = False
        self._set_ply_view_loading_state(False)
        self._ply_view_rgb_mean = None
        self._ply_view_info_var.set("PLY viewer is idle")
        self._redraw_ply_canvas()
        messagebox.showerror("Show PLY", f"Failed to load the PLY file.\n{exc}")

    def _reset_ply_view_transform(self) -> None:
        yaw = math.radians(35.0)
        pitch = math.radians(25.0)
        q_yaw = self._quat_from_axis_angle((0.0, 1.0, 0.0), yaw)
        q_pitch = self._quat_from_axis_angle((1.0, 0.0, 0.0), pitch)
        self._ply_view_quat = self._quat_normalize(
            self._quat_multiply(q_pitch, q_yaw)
        )
        self._ply_view_zoom = 1.0
        self._ply_drag_last = None
        self._ply_pan_last = None
        self._ply_view_pan = (0.0, 0.0)

    def _redraw_ply_canvas(self) -> None:
        canvas = self._ply_view_canvas
        if canvas is None or not canvas.winfo_exists():
            return
        self._render_ply_points(canvas)

    def _begin_ply_interaction(self) -> None:
        self._ply_is_interacting = True
        if self._ply_interaction_after_id is not None:
            try:
                self.root.after_cancel(self._ply_interaction_after_id)
            except Exception:
                pass
            self._ply_interaction_after_id = None

    def _schedule_end_ply_interaction(
        self, delay_ms: int = PLY_INTERACTION_SETTLE_DELAY_MS
    ) -> None:
        if self._ply_interaction_after_id is not None:
            try:
                self.root.after_cancel(self._ply_interaction_after_id)
            except Exception:
                pass
            self._ply_interaction_after_id = None
        self._ply_interaction_after_id = self.root.after(
            delay_ms, self._finish_ply_interaction
        )

    def _finish_ply_interaction(self) -> None:
        if self._ply_interaction_after_id is not None:
            try:
                self.root.after_cancel(self._ply_interaction_after_id)
            except Exception:
                pass
            self._ply_interaction_after_id = None
        was_interacting = self._ply_is_interacting
        self._ply_is_interacting = False
        if was_interacting:
            self._redraw_ply_canvas()

    @staticmethod
    def _compute_sample_step(total_count: int, max_count: int) -> int:
        if total_count <= 0 or max_count <= 0:
            return 1
        if total_count <= max_count:
            return 1
        return max(1, int(math.ceil(float(total_count) / float(max_count))))

    def _on_ply_drag_start(self, event: tk.Event) -> None:
        if self._ply_view_points_centered is None:
            return
        self._begin_ply_interaction()
        self._ply_drag_last = (event.x, event.y)

    def _on_ply_drag_move(self, event: tk.Event) -> None:
        if self._ply_drag_last is None or self._ply_view_points_centered is None:
            return
        last_x, last_y = self._ply_drag_last
        dx = event.x - last_x
        dy = event.y - last_y
        self._ply_drag_last = (event.x, event.y)
        angle_y = math.radians(dx * 0.35)
        angle_x = math.radians(dy * 0.35)
        q_y = self._quat_from_axis_angle((0.0, 1.0, 0.0), angle_y)
        q_x = self._quat_from_axis_angle((1.0, 0.0, 0.0), angle_x)
        q_inc = self._quat_multiply(q_x, q_y)
        self._ply_view_quat = self._quat_normalize(
            self._quat_multiply(q_inc, self._ply_view_quat)
        )
        self._redraw_ply_canvas()

    def _on_ply_drag_end(self, _event=None) -> None:
        self._ply_drag_last = None
        self._ply_pan_last = None
        self._schedule_end_ply_interaction()

    def _on_ply_pan_start(self, event: tk.Event) -> None:
        if self._ply_view_points_centered is None:
            return
        self._begin_ply_interaction()
        self._ply_pan_last = (event.x, event.y)

    def _on_ply_pan_move(self, event: tk.Event) -> None:
        if self._ply_pan_last is None or self._ply_view_points_centered is None:
            return
        last_x, last_y = self._ply_pan_last
        dx = event.x - last_x
        dy = event.y - last_y
        pan_x, pan_y = self._ply_view_pan
        self._ply_view_pan = (pan_x + dx, pan_y + dy)
        self._ply_pan_last = (event.x, event.y)
        self._redraw_ply_canvas()

    def _on_ply_pan_end(self, _event=None) -> None:
        self._ply_pan_last = None
        self._schedule_end_ply_interaction()

    def _on_ply_zoom(self, event: tk.Event) -> Optional[str]:
        if self._ply_view_points_centered is None:
            return None
        delta = getattr(event, "delta", 0)
        if delta == 0:
            num = getattr(event, "num", None)
            if num == 4:
                delta = 120
            elif num == 5:
                delta = -120
        if delta == 0:
            return None
        factor = 1.0 + (0.12 if delta > 0 else -0.12)
        new_zoom = self._ply_view_zoom * factor
        self._ply_view_zoom = max(0.1, min(PLY_VIEW_MAX_ZOOM, new_zoom))
        self._begin_ply_interaction()
        self._redraw_ply_canvas()
        self._schedule_end_ply_interaction()
        return "break"

    @staticmethod
    def _quat_normalize(q: np.ndarray) -> np.ndarray:
        q_arr = np.asarray(q, dtype=np.float32)
        norm = float(np.linalg.norm(q_arr))
        if norm <= 1e-9:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return (q_arr / norm).astype(np.float32, copy=False)

    @staticmethod
    def _quat_from_axis_angle(
        axis: Tuple[float, float, float], angle_rad: float
    ) -> np.ndarray:
        axis_arr = np.asarray(axis, dtype=np.float32)
        norm = float(np.linalg.norm(axis_arr))
        if norm <= 1e-9:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        axis_n = axis_arr / norm
        half = angle_rad * 0.5
        s = math.sin(half)
        c = math.cos(half)
        return np.array(
            [c, axis_n[0] * s, axis_n[1] * s, axis_n[2] * s],
            dtype=np.float32,
        )

    @staticmethod
    def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        w1, x1, y1, z1 = np.asarray(q1, dtype=np.float32)
        w2, x2, y2, z2 = np.asarray(q2, dtype=np.float32)
        return np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
        w, x, y, z = np.asarray(q, dtype=np.float32)
        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z
        wx = w * x
        wy = w * y
        wz = w * z
        return np.array(
            [
                [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
                [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
                [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
            ],
            dtype=np.float32,
        )

    def _on_ply_projection_changed(self, _event=None) -> None:
        self._redraw_ply_canvas()

    @staticmethod
    def _axis_direction(axis_label: str) -> Optional[np.ndarray]:
        direction_map = {
            "+X": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "-X": np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            "+Y": np.array([0.0, 1.0, 0.0], dtype=np.float32),
            "-Y": np.array([0.0, -1.0, 0.0], dtype=np.float32),
            "+Z": np.array([0.0, 0.0, 1.0], dtype=np.float32),
            "-Z": np.array([0.0, 0.0, -1.0], dtype=np.float32),
        }
        direction = direction_map.get((axis_label or "").upper())
        if direction is None:
            return None
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm
        return direction

    def _generate_sky_points(
        self,
        axis_label: str,
        scale: float,
        count: int,
        color_rgb: Tuple[int, int, int],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        direction = self._axis_direction(axis_label)
        if direction is None:
            return None, None
        count = max(1000, min(20000, int(count)))
        indices = np.arange(count, dtype=np.float32)
        phi = math.pi * (3.0 - math.sqrt(5.0))
        z = 1.0 - indices / count
        radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        x = np.cos(phi * indices) * radius
        y = np.sin(phi * indices) * radius
        points = np.stack((x, y, z), axis=1)
        points *= float(scale)
        rotation = self._rotation_matrix_from_vectors(np.array([0.0, 0.0, 1.0], dtype=np.float32), direction)
        rotated = points @ rotation.T
        colors = np.tile(np.array(color_rgb, dtype=np.uint8), (rotated.shape[0], 1))
        return rotated.astype(np.float32, copy=False), colors

    def _parse_color_to_rgb(self, value: str) -> Optional[Tuple[int, int, int]]:
        text = value.strip()
        if not text:
            return None
        try:
            r, g, b = self.root.winfo_rgb(text)
            return (
                min(255, max(0, r // 256)),
                min(255, max(0, g // 256)),
                min(255, max(0, b // 256)),
            )
        except tk.TclError:
            pass
        raw = text[1:] if text.startswith("#") else text
        if len(raw) == 3:
            raw = "".join(ch * 2 for ch in raw)
        if len(raw) != 6:
            return None
        try:
            r = int(raw[0:2], 16)
            g = int(raw[2:4], 16)
            b = int(raw[4:6], 16)
        except ValueError:
            return None
        return (r, g, b)

    def _update_sky_color_display(self, rgb: Tuple[int, int, int], hex_value: str) -> None:
        self._ply_sky_color_rgb_var.set(f"RGB({rgb[0]}, {rgb[1]}, {rgb[2]})")
        label = self._ply_sky_color_label
        if label is not None:
            hex_clean = hex_value if hex_value.startswith("#") else f"#{hex_value}"
            try:
                label.configure(fg=hex_clean)
            except tk.TclError:
                pass

    def _update_remove_color_display(
        self, rgb: Tuple[int, int, int], hex_value: str
    ) -> None:
        self._ply_remove_color_rgb_var.set(
            f"RGB({rgb[0]}, {rgb[1]}, {rgb[2]})"
        )
        label = self._ply_remove_color_label
        if label is not None:
            hex_clean = hex_value if hex_value.startswith("#") else f"#{hex_value}"
            try:
                label.configure(fg=hex_clean)
            except tk.TclError:
                pass

    def _sample_auto_sky_color(self) -> Optional[Tuple[int, int, int]]:
        if self._ply_view_points_centered is None or self._ply_view_colors is None:
            return None
        direction = self._axis_direction(self._ply_sky_axis_var.get() or "+Z")
        if direction is None:
            return None
        points = self._ply_view_points_centered
        colors = self._ply_view_colors
        if points.size == 0 or colors.size == 0:
            return None
        norms = np.linalg.norm(points, axis=1)
        dots = points @ direction
        cos_vals = np.zeros_like(dots)
        valid = norms > 1e-9
        cos_vals[valid] = dots[valid] / norms[valid]
        mask = (cos_vals >= math.cos(math.radians(45.0))) & valid
        candidate_indices = np.where(mask)[0]
        if candidate_indices.size < 10:
            candidate_indices = np.where(dots > 0)[0]
        if candidate_indices.size == 0:
            return None
        sample = min(200, candidate_indices.size)
        farthest = candidate_indices[np.argsort(norms[candidate_indices])]
        chosen = farthest[-sample:]
        avg_color = colors[chosen].mean(axis=0)
        rgb = tuple(int(round(val)) for val in avg_color)
        return rgb

    def _get_remove_color_tolerance(
        self, show_error: bool = False
    ) -> Optional[float]:
        text = self._ply_remove_color_tol_var.get().strip()
        if not text:
            return 0.0
        try:
            value = float(text)
        except ValueError:
            if show_error:
                messagebox.showerror(
                    "PLY Viewer",
                    "Remove color tolerance must be numeric.",
                )
            return None
        if value < 0:
            if show_error:
                messagebox.showerror(
                    "PLY Viewer",
                    "Remove color tolerance must be zero or greater.",
                )
            return None
        return value

    def _refresh_ply_view_rgb_mean(self) -> None:
        if self._ply_view_colors is None or self._ply_view_colors.size == 0:
            self._ply_view_rgb_mean = None
            return
        rgb_mean = self._ply_view_colors.mean(axis=0)
        self._ply_view_rgb_mean = (
            float(rgb_mean[0]),
            float(rgb_mean[1]),
            float(rgb_mean[2]),
        )

    def _build_ply_info_text(
        self, point_count: int, original_count: int, sample_step: int
    ) -> str:
        label = self._ply_view_source_label or "PLY"
        name = (
            self._ply_current_file_path.name
            if self._ply_current_file_path is not None
            else "PLY"
        )
        if sample_step > 1 and original_count > 0:
            return (
                f"{label}: {name}  "
                f"({point_count:,} / {original_count:,} pts, step {sample_step})"
            )
        return f"{label}: {name}  ({point_count:,} pts)"

    @staticmethod
    def _rotation_matrix_from_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
        src = source / max(np.linalg.norm(source), 1e-6)
        tgt = target / max(np.linalg.norm(target), 1e-6)
        c = np.dot(src, tgt)
        if c > 0.9999:
            return np.eye(3, dtype=np.float32)
        if c < -0.9999:
            axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            if abs(src[0]) > 0.9:
                axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            axis -= axis.dot(src) * src
            axis /= max(np.linalg.norm(axis), 1e-6)
            return PreviewApp._rotation_matrix_from_axis(axis, math.pi)
        v = np.cross(src, tgt)
        s = np.linalg.norm(v)
        vx = np.array(
            [
                [0.0, -v[2], v[1]],
                [v[2], 0.0, -v[0]],
                [-v[1], v[0], 0.0],
            ],
            dtype=np.float32,
        )
        r = np.eye(3, dtype=np.float32) + vx + (vx @ vx) * ((1.0 - c) / (s * s))
        return r.astype(np.float32)

    @staticmethod
    def _rotation_matrix_from_axis(axis: np.ndarray, angle: float) -> np.ndarray:
        axis = axis / max(np.linalg.norm(axis), 1e-6)
        x, y, z = axis
        c = math.cos(angle)
        s = math.sin(angle)
        C = 1.0 - c
        return np.array(
            [
                [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
            ],
            dtype=np.float32,
        )

    def _on_add_sky_points(self) -> None:
        if self._ply_view_points_centered is None:
            messagebox.showerror("Sky PointCloud", "Load a PLY before adding sky points.")
            return
        axis_label = (self._ply_sky_axis_var.get() or "+Z").upper()
        scale_text = self._ply_sky_scale_var.get().strip() or "100"
        try:
            scale_value = float(scale_text)
        except ValueError:
            messagebox.showerror("Sky PointCloud", "Sky Scale must be numeric.")
            return
        if scale_value <= 0:
            messagebox.showerror("Sky PointCloud", "Sky Scale must be greater than zero.")
            return
        count_text = self._ply_sky_count_var.get().strip() or "4000"
        try:
            count_value = int(float(count_text))
        except ValueError:
            messagebox.showerror("Sky PointCloud", "Sky points must be numeric.")
            return
        if count_value <= 0:
            messagebox.showerror("Sky PointCloud", "Sky points must be greater than zero.")
            return
        color_text = self._ply_sky_color_var.get().strip() or "#87cefa"
        color_rgb = self._parse_color_to_rgb(color_text)
        if color_rgb is None:
            messagebox.showerror("Sky PointCloud", "Sky color must be a valid color (e.g., #87cefa).")
            return
        self._update_sky_color_display(color_rgb, color_text)
        points, colors = self._generate_sky_points(axis_label, scale_value, count_value, color_rgb)
        if points is None or colors is None:
            messagebox.showerror("Sky PointCloud", "Failed to generate sky points. Check axis selection.")
            return
        self._ply_sky_points = points
        self._ply_sky_colors = colors
        self._redraw_ply_canvas()

    def _on_clear_sky_points(self) -> None:
        if self._ply_sky_points is None and self._ply_sky_colors is None:
            return
        self._ply_sky_points = None
        self._ply_sky_colors = None
        self._redraw_ply_canvas()

    def _on_pick_sky_color(self) -> None:
        initial = self._ply_sky_color_var.get().strip() or "#87cefa"
        try:
            _, initial_hex = colorchooser.askcolor(color=initial, title="Sky Color")
        except tk.TclError:
            initial_hex = None
        if initial_hex:
            self._ply_sky_color_var.set(initial_hex)
            rgb = self._parse_color_to_rgb(initial_hex)
            if rgb is not None:
                self._update_sky_color_display(rgb, initial_hex)

    def _on_auto_pick_sky_color(self) -> None:
        if self._ply_view_points_centered is None or self._ply_view_colors is None:
            messagebox.showerror("Sky PointCloud", "Load a PLY before auto picking a color.")
            return
        rgb = self._sample_auto_sky_color()
        if rgb is None:
            messagebox.showerror(
                "Sky PointCloud",
                "Failed to auto pick sky color from current points/axis.",
            )
            return
        hex_value = "#{:02x}{:02x}{:02x}".format(*rgb)
        self._ply_sky_color_var.set(hex_value)
        self._update_sky_color_display(rgb, hex_value)
        self._ply_remove_color_var.set(hex_value)
        self._update_remove_color_display(rgb, hex_value)
        if self._ply_sky_colors is not None:
            self._ply_sky_colors[:] = np.array(rgb, dtype=np.uint8)
            self._redraw_ply_canvas()

    def _on_pick_remove_color(self) -> None:
        initial = self._ply_remove_color_var.get().strip() or "#87cefa"
        try:
            _, picked_hex = colorchooser.askcolor(
                color=initial,
                title="Remove Color",
            )
        except tk.TclError:
            picked_hex = None
        if picked_hex:
            self._ply_remove_color_var.set(picked_hex)
            rgb = self._parse_color_to_rgb(picked_hex)
            if rgb is not None:
                self._update_remove_color_display(rgb, picked_hex)

    def _on_remove_color_points(self) -> None:
        if self._ply_view_points is None or self._ply_view_colors is None:
            messagebox.showerror("PLY Viewer", "Load a PLY before removing colors.")
            return
        target_text = self._ply_remove_color_var.get().strip()
        target_rgb = self._parse_color_to_rgb(target_text)
        if target_rgb is None:
            messagebox.showerror(
                "PLY Viewer",
                "Remove color must be a valid color (e.g., #87cefa).",
            )
            return
        tol = self._get_remove_color_tolerance(show_error=True)
        if tol is None:
            return
        self._update_remove_color_display(target_rgb, target_text)
        target = np.array(target_rgb, dtype=np.int32).reshape(1, 3)
        tol2 = float(tol) * float(tol)
        base_colors = self._ply_view_colors.astype(np.int32, copy=False)
        diff = base_colors - target
        diff64 = diff.astype(np.int64, copy=False)
        dist2 = (diff64 * diff64).sum(axis=1)
        keep_mask = dist2 > tol2
        removed_base = int(np.count_nonzero(~keep_mask))
        if removed_base <= 0:
            self._ply_view_info_var.set(
                "No points removed "
                f"(color={target_text}, tol={tol:.1f})."
            )
            self._redraw_ply_canvas()
            return
        self._ply_view_points = self._ply_view_points[keep_mask].astype(
            np.float32, copy=False
        )
        self._ply_view_colors = self._ply_view_colors[keep_mask].astype(
            np.uint8, copy=False
        )
        self._ply_view_points_centered = self._ply_view_points.astype(
            np.float32, copy=False
        )
        self._ply_view_total_points = int(self._ply_view_points.shape[0])
        self._ply_view_sample_step = 1
        removed_sky = 0
        if self._ply_sky_points is not None and self._ply_sky_colors is not None:
            sky_diff = (
                self._ply_sky_colors.astype(np.int32, copy=False) - target
            )
            sky_diff64 = sky_diff.astype(np.int64, copy=False)
            sky_dist2 = (sky_diff64 * sky_diff64).sum(axis=1)
            keep_sky_mask = sky_dist2 > tol2
            removed_sky = int(np.count_nonzero(~keep_sky_mask))
            self._ply_sky_points = self._ply_sky_points[keep_sky_mask].astype(
                np.float32, copy=False
            )
            self._ply_sky_colors = self._ply_sky_colors[keep_sky_mask].astype(
                np.uint8, copy=False
            )
            if self._ply_sky_points.size == 0:
                self._ply_sky_points = None
                self._ply_sky_colors = None
        self._refresh_ply_view_rgb_mean()
        remaining = int(self._ply_view_points.shape[0])
        total_removed = removed_base + removed_sky
        self._ply_view_info_var.set(
            f"Removed {total_removed:,} pts by color "
            f"({target_text}, tol={tol:.1f}) -> {remaining:,} pts"
        )
        self._redraw_ply_canvas()

    def _on_reset_ply_view_state(self) -> None:
        if self._ply_loaded_points is None or self._ply_loaded_colors is None:
            return
        if self._ply_interaction_after_id is not None:
            try:
                self.root.after_cancel(self._ply_interaction_after_id)
            except Exception:
                pass
            self._ply_interaction_after_id = None
        self._ply_is_interacting = False
        self._ply_sky_points = None
        self._ply_sky_colors = None
        self._ply_view_points = self._ply_loaded_points.astype(
            np.float32, copy=True
        )
        self._ply_view_colors = self._ply_loaded_colors.astype(
            np.uint8, copy=True
        )
        self._ply_view_points_centered = self._ply_view_points.astype(
            np.float32, copy=False
        )
        self._ply_view_total_points = int(
            max(self._ply_loaded_total_points, self._ply_view_points.shape[0])
        )
        self._ply_view_sample_step = max(1, int(self._ply_loaded_sample_step))
        self._refresh_ply_view_rgb_mean()
        point_count = int(self._ply_view_points.shape[0])
        info = self._build_ply_info_text(
            point_count,
            self._ply_view_total_points,
            self._ply_view_sample_step,
        )
        self._ply_view_info_var.set(info)
        self._redraw_ply_canvas()

    def _update_sky_save_default(self, source_path: Path) -> None:
        suffix = source_path.suffix or ".ply"
        candidate = source_path.with_name(f"{source_path.stem}_viewed{suffix}")
        self._ply_sky_save_path_var.set(str(candidate))

    def _on_browse_sky_save_path(self) -> None:
        initial = self._ply_sky_save_path_var.get().strip()
        if initial:
            try:
                initial_dir = Path(initial).expanduser().parent
            except Exception:
                initial_dir = self.base_dir
        else:
            initial_dir = (
                self._ply_current_file_path.parent
                if self._ply_current_file_path is not None
                else self.base_dir
            )
        try:
            if not initial_dir.exists():
                initial_dir = self.base_dir
        except Exception:
            initial_dir = self.base_dir
        path = filedialog.asksaveasfilename(
            title="Save viewed PLY",
            initialdir=str(initial_dir),
            defaultextension=".ply",
            filetypes=[("PLY files", "*.ply"), ("All files", "*.*")],
        )
        if path:
            self._ply_sky_save_path_var.set(path)

    def _on_save_sky_points(self) -> None:
        if self._ply_view_points is None or self._ply_view_colors is None:
            messagebox.showerror("PLY Viewer", "Load a PLY before saving.")
            return
        dest_text = self._ply_sky_save_path_var.get().strip()
        if not dest_text:
            messagebox.showerror("PLY Viewer", "Specify a save path first.")
            return
        dest_path = Path(dest_text).expanduser()
        try:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            messagebox.showerror("PLY Viewer", f"Failed to prepare destination.\n{exc}")
            return
        try:
            self._write_binary_ply_from_view(dest_path)
        except Exception as exc:
            messagebox.showerror("PLY Viewer", f"Failed to save PLY.\n{exc}")
            return
        messagebox.showinfo("PLY Viewer", f"Saved viewed PLY:\n{dest_path}")

    def _write_binary_ply_from_view(self, path: Path) -> None:
        base_points = self._ply_view_points
        base_colors = self._ply_view_colors
        sky_points = self._ply_sky_points
        sky_colors = self._ply_sky_colors
        if base_points is None or base_colors is None:
            raise ValueError("Missing loaded point cloud data.")
        points_list: List[np.ndarray] = [base_points.astype(np.float32, copy=False)]
        colors_list: List[np.ndarray] = [base_colors.astype(np.uint8, copy=False)]
        if sky_points is not None and sky_colors is not None:
            center = self._ply_view_center
            if not isinstance(center, np.ndarray):
                center = np.zeros(3, dtype=np.float32)
            sky_world = (sky_points + center).astype(np.float32, copy=False)
            points_list.append(sky_world)
            colors_list.append(sky_colors.astype(np.uint8, copy=False))
        xyz = np.concatenate(points_list, axis=0).astype(np.float32, copy=False)
        rgb = np.concatenate(colors_list, axis=0).astype(np.uint8, copy=False)
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {xyz.shape[0]}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        )
        data = np.empty(
            xyz.shape[0],
            dtype=[
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
            ],
        )
        data["x"] = xyz[:, 0]
        data["y"] = xyz[:, 1]
        data["z"] = xyz[:, 2]
        data["red"] = rgb[:, 0]
        data["green"] = rgb[:, 1]
        data["blue"] = rgb[:, 2]
        with path.open("wb") as fh:
            fh.write(header.encode("ascii"))
            fh.write(data.tobytes())


    def _load_binary_ply_points(
        self,
        path: Path,
        *,
        max_points: Optional[int] = PLY_VIEW_MAX_POINTS,
    ) -> Tuple[np.ndarray, np.ndarray, int, int]:
        if max_points is not None and max_points < 0:
            raise ValueError("max_points must be >= 0")
        vertex_count: Optional[int] = None
        format_token = ""
        property_defs: List[Tuple[str, str]] = []
        reading_vertex = False
        with path.open("rb") as fh:
            while True:
                raw = fh.readline()
                if not raw:
                    raise ValueError("Unexpected EOF while reading the PLY header.")
                try:
                    line = raw.decode("ascii", errors="ignore").strip()
                except UnicodeDecodeError:
                    line = raw.decode("latin-1", errors="ignore").strip()
                if not line:
                    continue
                if line.startswith("comment"):
                    continue
                if line.startswith("format"):
                    parts = line.split()
                    if len(parts) >= 2:
                        format_token = parts[1].lower()
                    continue
                if line.startswith("element"):
                    parts = line.split()
                    if len(parts) >= 3 and parts[1].lower() == "vertex":
                        vertex_count = int(parts[2])
                        reading_vertex = True
                    else:
                        reading_vertex = False
                    continue
                if line.startswith("property") and reading_vertex:
                    parts = line.split()
                    if len(parts) >= 3:
                        property_defs.append((parts[1].lower(), parts[2]))
                    continue
                if line == "end_header":
                    break
            if format_token != "binary_little_endian":
                raise ValueError("Only binary_little_endian PLY format is supported.")
            if vertex_count is None or vertex_count <= 0:
                raise ValueError("No valid vertex element was found.")
            if not property_defs:
                raise ValueError("No vertex properties are defined.")
            dtype_fields: List[Tuple[str, str]] = []
            available_props: Set[str] = set()
            for ptype, pname in property_defs:
                info = PLY_PROPERTY_TYPES.get(ptype)
                if info is None:
                    raise ValueError(f"Unsupported property type: {ptype}")
                dtype_fields.append((pname, info[2]))
                available_props.add(pname)
            required_axes = [axis for axis in ("x", "y", "z") if axis not in available_props]
            if required_axes:
                raise ValueError(f"Missing coordinate properties: {', '.join(required_axes)}")
            dtype = np.dtype(dtype_fields)
            stride = dtype.itemsize
            if stride <= 0:
                raise ValueError("Invalid PLY vertex stride.")
            vertex_bytes = stride * vertex_count
            vertex_blob = fh.read(vertex_bytes)
            if len(vertex_blob) < vertex_bytes:
                raise ValueError("Unexpected EOF while reading PLY vertex data.")
            data = np.frombuffer(vertex_blob, dtype=dtype).copy()
        original_count = int(data.shape[0])
        sample_step = 1
        if max_points:
            if original_count > max_points:
                sample_step = max(1, int(math.ceil(original_count / max_points)))
                data = data[::sample_step]
        xyz = np.stack((data["x"], data["y"], data["z"]), axis=1).astype(np.float32, copy=False)
        if all(channel in data.dtype.names for channel in ("red", "green", "blue")):
            colors = np.stack((data["red"], data["green"], data["blue"]), axis=1)
            if np.issubdtype(colors.dtype, np.floating):
                colors = self._float_colors_to_u8(colors)
            else:
                colors = colors.astype(np.uint8, copy=False)
        else:
            colors = np.full((xyz.shape[0], 3), 220, dtype=np.uint8)
        return xyz, colors, original_count, sample_step

    @staticmethod
    def _float_colors_to_u8(values: np.ndarray) -> np.ndarray:
        """Convert float RGB values to uint8 while preserving sRGB semantics."""
        values32 = values.astype(np.float32, copy=False)
        finite = values32[np.isfinite(values32)]
        if finite.size == 0:
            return np.zeros(values32.shape, dtype=np.uint8)
        max_value = float(finite.max())
        if max_value <= 1.0 + 1e-6:
            scaled = np.clip(values32, 0.0, 1.0) * 255.0
        else:
            scaled = np.clip(values32, 0.0, 255.0)
        return np.clip(np.rint(scaled), 0, 255).astype(np.uint8)

    def _render_ply_points(
        self,
        canvas: tk.Canvas,
    ) -> None:
        if self._ply_view_points_centered is None or self._ply_view_colors is None:
            self._render_ply_idle_canvas(canvas)
            return
        canvas.update_idletasks()
        width = max(1, canvas.winfo_width())
        height = max(1, canvas.winfo_height())
        if width < 10 or height < 10:
            width = PLY_VIEW_CANVAS_WIDTH
            height = PLY_VIEW_CANVAS_HEIGHT
        points = self._ply_view_points_centered
        colors = self._ply_view_colors
        if self._ply_sky_points is not None and self._ply_sky_colors is not None:
            points = np.concatenate((points, self._ply_sky_points), axis=0)
            colors = np.concatenate((colors, self._ply_sky_colors), axis=0)
        if points.size == 0 or colors.size == 0:
            self._render_ply_idle_canvas(canvas)
            return
        high_quality_max = self._get_ply_view_high_max_points(show_error=False)
        if high_quality_max is None:
            high_quality_max = PLY_VIEW_MAX_POINTS
        render_max = high_quality_max
        if self._ply_is_interacting:
            low_quality_max = self._get_ply_view_interactive_max_points(
                show_error=False
            )
            if low_quality_max is None:
                low_quality_max = PLY_VIEW_INTERACTIVE_MAX_POINTS
            render_max = min(render_max, low_quality_max)
        sample_step = self._compute_sample_step(points.shape[0], render_max)
        if sample_step > 1:
            points = points[::sample_step]
            colors = colors[::sample_step]
        depth_offset = max(self._ply_view_depth_offset, 1e-6)
        rotation = self._quat_to_matrix(self._ply_view_quat)
        rotated = points @ rotation.T
        proj_scale = min(width, height) * 0.9 * self._ply_view_zoom
        ortho_scale = self._compute_ortho_scale(width, height)
        projection_mode = (self._ply_projection_mode.get() or "").lower()
        is_orthographic = projection_mode.startswith("ortho")
        pan_x, pan_y = self._ply_view_pan
        x1 = rotated[:, 0]
        y1 = rotated[:, 1]
        z2 = rotated[:, 2]
        depth = z2 + depth_offset
        if is_orthographic:
            valid = np.ones_like(depth, dtype=bool)
            sx = width / 2.0 + x1 * ortho_scale + pan_x
            sy = height / 2.0 - y1 * ortho_scale + pan_y
        else:
            valid = depth > 1e-4
            scale = np.zeros_like(depth)
            scale[valid] = proj_scale / depth[valid]
            sx = width / 2.0 + x1 * scale + pan_x
            sy = height / 2.0 - y1 * scale + pan_y
        ix = np.rint(sx).astype(np.int32)
        iy = np.rint(sy).astype(np.int32)
        valid &= (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
        if not np.any(valid):
            messagebox.showinfo("Show PLY", "No visible points remained after projection.")
            self._render_ply_idle_canvas(canvas)
            return
        depth_valid = depth[valid]
        ix_valid = ix[valid]
        iy_valid = iy[valid]
        colors_valid = colors[valid]
        if bool(self._ply_monochrome_var.get()):
            colors_valid = np.full((colors_valid.shape[0], 3), 255, dtype=np.uint8)
        pixel_index = (iy_valid * width + ix_valid).astype(np.int64, copy=False)
        order = np.argsort(depth_valid)[::-1]
        pixel_order = pixel_index[order]
        color_order = colors_valid[order]
        pixel_count = width * height
        buf = np.full((pixel_count, 3), 0x11, dtype=np.uint8)
        buf[pixel_order, :] = color_order
        image_array = buf.reshape((height, width, 3))
        image = Image.fromarray(image_array, mode="RGB")
        self._draw_axes_overlay(
            image,
            width,
            height,
            rotation,
            depth_offset,
            proj_scale,
            ortho_scale,
            is_orthographic,
            pan_x,
            pan_y,
        )
        self._draw_ply_stats_overlay(image)
        photo = ImageTk.PhotoImage(image=image)
        self._ply_canvas_photo = photo
        if self._ply_canvas_image_id is None:
            self._ply_canvas_image_id = canvas.create_image(0, 0, anchor="nw", image=photo)
        else:
            canvas.itemconfigure(self._ply_canvas_image_id, image=photo)
        canvas.configure(scrollregion=(0, 0, width, height))

    def _render_ply_idle_canvas(self, canvas: tk.Canvas) -> None:
        canvas.update_idletasks()
        width = max(1, canvas.winfo_width())
        height = max(1, canvas.winfo_height())
        if width < 10 or height < 10:
            width = PLY_VIEW_CANVAS_WIDTH
            height = PLY_VIEW_CANVAS_HEIGHT
        image = Image.new("RGB", (width, height), (0x11, 0x11, 0x11))
        self._draw_ply_stats_overlay(image)
        photo = ImageTk.PhotoImage(image=image)
        self._ply_canvas_photo = photo
        if self._ply_canvas_image_id is None:
            self._ply_canvas_image_id = canvas.create_image(
                0, 0, anchor="nw", image=photo
            )
        else:
            canvas.itemconfigure(self._ply_canvas_image_id, image=photo)
        canvas.configure(scrollregion=(0, 0, width, height))

    def _compute_ortho_scale(self, width: int, height: int) -> float:
        max_extent = max(self._ply_view_max_extent, 1e-6)
        return max(
            1e-6,
            self._ply_view_zoom * (min(width, height) * 0.45 / max_extent),
        )

    def _draw_axes_overlay(
        self,
        image: Image.Image,
        width: int,
        height: int,
        rotation: np.ndarray,
        depth_offset: float,
        proj_scale: float,
        ortho_scale: float,
        is_orthographic: bool,
        pan_x: float,
        pan_y: float,
    ) -> None:
        axis_len = max(self._ply_view_max_extent * 0.25, 1e-3)
        origin = (0.0, 0.0, 0.0)
        axes = {
            "X": (axis_len, 0.0, 0.0),
            "Y": (0.0, axis_len, 0.0),
            "Z": (0.0, 0.0, axis_len),
        }
        screen_origin = self._project_point_to_screen(
            origin,
            width,
            height,
            rotation,
            depth_offset,
            proj_scale,
            ortho_scale,
            is_orthographic,
            pan_x,
            pan_y,
        )
        if screen_origin is None:
            return
        draw = ImageDraw.Draw(image)
        colors = {
            "X": (255, 80, 80),
            "Y": (80, 255, 120),
            "Z": (80, 160, 255),
        }
        for label, endpoint in axes.items():
            screen_point = self._project_point_to_screen(
                endpoint,
                width,
                height,
                rotation,
                depth_offset,
                proj_scale,
                ortho_scale,
                is_orthographic,
                pan_x,
                pan_y,
            )
            if screen_point is None:
                continue
            color = colors.get(label, (255, 255, 255))
            draw.line(
                [screen_origin, screen_point],
                fill=color,
                width=2,
            )
            draw.text(
                (screen_point[0] + 4, screen_point[1] - 12),
                label,
                fill=color,
            )
        r = 4
        draw.ellipse(
            [
                (screen_origin[0] - r, screen_origin[1] - r),
                (screen_origin[0] + r, screen_origin[1] + r),
            ],
            fill=(255, 255, 255),
        )
        axis_text = f"Axis scale: {axis_len:.2f}"
        draw.text(
            (12, height - 24),
            axis_text,
            fill=(255, 255, 255),
        )

    def _draw_ply_stats_overlay(self, image: Image.Image) -> None:
        draw = ImageDraw.Draw(image)
        mean_rgb = self._ply_view_rgb_mean
        if mean_rgb is not None:
            mean_text = "Mean RGB: {:.1f}, {:.1f}, {:.1f}".format(
                mean_rgb[0],
                mean_rgb[1],
                mean_rgb[2],
            )
            x0, y0 = 8, 8
            bbox = draw.textbbox((0, 0), mean_text)
            text_w = max(0, int(bbox[2] - bbox[0]))
            text_h = max(0, int(bbox[3] - bbox[1]))
            x1 = min(image.width - 8, x0 + text_w + 8)
            y1 = y0 + text_h + 8
            draw.rectangle([(x0, y0), (x1, y1)], fill=(0, 0, 0))
            draw.text((x0 + 4, y0 + 4), mean_text, fill=(255, 255, 255))
        info_text = self._ply_view_info_var.get().strip()
        if not info_text:
            return
        bbox = draw.textbbox((0, 0), info_text)
        text_w = max(0, int(bbox[2] - bbox[0]))
        text_h = max(0, int(bbox[3] - bbox[1]))
        x1 = image.width - 8
        y0 = 8
        x0 = max(8, x1 - text_w - 8)
        y1 = y0 + text_h + 8
        draw.rectangle([(x0, y0), (x1, y1)], fill=(0, 0, 0))
        draw.text((x0 + 4, y0 + 4), info_text, fill=(255, 255, 255))

    @staticmethod
    def _project_point_to_screen(
        point: Tuple[float, float, float],
        width: int,
        height: int,
        rotation: np.ndarray,
        depth_offset: float,
        proj_scale: float,
        ortho_scale: float,
        is_orthographic: bool,
        pan_x: float,
        pan_y: float,
    ) -> Optional[Tuple[float, float]]:
        px, py, pz = point
        p = np.array([px, py, pz], dtype=np.float32)
        transformed = rotation @ p
        x1 = float(transformed[0])
        y1 = float(transformed[1])
        z2 = float(transformed[2])
        depth = z2 + depth_offset
        if is_orthographic:
            scale = ortho_scale
        else:
            if depth <= 1e-4:
                return None
            scale = proj_scale / depth
        sx = width / 2.0 + x1 * scale + pan_x
        sy = height / 2.0 - y1 * scale + pan_y
        return sx, sy

    def _select_file(
        self,
        var: tk.Variable,
        title: str = "Select file",
        filetypes: Optional[Sequence[Tuple[str, str]]] = None,
        on_select: Optional[Callable[[str], None]] = None,
    ) -> None:
        current = var.get().strip() if hasattr(var, "get") else ""
        initialdir = Path(current).parent if current else self.base_dir
        try:
            initialdir = initialdir if initialdir.exists() else self.base_dir
        except Exception:
            initialdir = self.base_dir
        path = filedialog.askopenfilename(
            title=title,
            initialdir=str(initialdir),
            filetypes=filetypes or [("All files", "*.*")],
        )
        if path:
            var.set(path)
            if on_select is not None:
                try:
                    on_select(path)
                except Exception:
                    pass

    def _on_selector_csv_mode_changed(self, *_args) -> None:
        mode = self.selector_vars.get("csv_mode")
        if mode is None:
            return
        mode_value = mode.get()
        if self.selector_csv_entry is not None:
            if mode_value == "none":
                self.selector_csv_entry.configure(state="disabled")
                self._set_selector_csv_path_auto("")
            else:
                self.selector_csv_entry.configure(state="normal")
        if self.selector_csv_button is not None:
            if mode_value == "none":
                self.selector_csv_button.configure(state="disabled")
            else:
                self.selector_csv_button.configure(state="normal")
        if self.selector_dry_run_check is not None:
            if mode_value == "reselect":
                self.selector_vars["dry_run"].set(True)
                self.selector_dry_run_check.configure(state="disabled")
            elif mode_value == "apply":
                self.selector_vars["dry_run"].set(False)
                self.selector_dry_run_check.configure(state="disabled")
            else:
                if mode_value == "write":
                    self.selector_vars["dry_run"].set(True)
                self.selector_dry_run_check.configure(state="normal")
        csv_var = self.selector_vars.get("csv_path")
        in_dir_var = self.selector_vars.get("in_dir")
        if mode_value == "none":
            self.selector_csv_auto = True
            self.selector_csv_auto_value = ""
        elif (
            self.selector_csv_auto
            and csv_var is not None
            and not csv_var.get().strip()
            and in_dir_var is not None
        ):
            in_dir_value = in_dir_var.get().strip()
            if in_dir_value:
                self._update_selector_csv_default(in_dir_value)

    def _browse_selector_csv(self) -> None:
        if not self.selector_vars:
            return
        mode_value = self.selector_vars.get("csv_mode", tk.StringVar(value="none")).get().strip()
        if mode_value == "none":
            messagebox.showinfo("FrameSelector CSV", "Select a CSV mode before browsing for a file.")
            return
        current_path = self.selector_vars.get("csv_path", tk.StringVar()).get().strip()
        if current_path:
            initial_dir = Path(current_path).expanduser().parent
        else:
            in_dir = self.selector_vars.get("in_dir", tk.StringVar()).get().strip()
            initial_dir = Path(in_dir).expanduser() if in_dir else Path.cwd()
        try:
            if not initial_dir.exists():
                initial_dir = Path.cwd()
        except Exception:
            initial_dir = Path.cwd()

        path: Optional[str]
        if mode_value == "write":
            path = filedialog.asksaveasfilename(
                title="Select CSV output path",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                initialdir=str(initial_dir),
            )
        else:
            path = filedialog.askopenfilename(
                title="Select CSV file",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                initialdir=str(initial_dir),
            )
        if path:
            self._set_selector_csv_path_manual(path)

    def _set_selector_csv_path_auto(self, value: str) -> None:
        if not self.selector_vars:
            return
        csv_var = self.selector_vars.get("csv_path")
        if csv_var is None:
            return
        value_str = str(value) if value is not None else ""
        self.selector_csv_auto_value = value_str
        self._selector_csv_updating = True
        try:
            csv_var.set(value_str)
        finally:
            self._selector_csv_updating = False
        self.selector_csv_auto = True

    def _set_selector_csv_path_manual(self, value: str) -> None:
        if not self.selector_vars:
            return
        csv_var = self.selector_vars.get("csv_path")
        if csv_var is None:
            return
        value_str = str(value) if value is not None else ""
        self.selector_csv_auto_value = value_str
        self._selector_csv_updating = True
        try:
            csv_var.set(value_str)
        finally:
            self._selector_csv_updating = False
        self.selector_csv_auto = False

    def _update_selector_csv_default(self, input_dir: str) -> None:
        if not self.selector_vars:
            return
        mode_var = self.selector_vars.get("csv_mode")
        if mode_var is not None and mode_var.get().strip() == "none":
            return
        path_text = str(input_dir).strip() if input_dir is not None else ""
        if not path_text:
            self._set_selector_csv_path_auto("")
            return
        try:
            folder = Path(path_text).expanduser()
        except Exception:
            return
        default_path = folder / DEFAULT_SELECTOR_CSV_NAME
        self._set_selector_csv_path_auto(str(default_path))

    def _on_selector_in_dir_changed(self, *_args) -> None:
        if not self.selector_vars or not self.selector_csv_auto:
            return
        if self._selector_csv_updating:
            return
        in_var = self.selector_vars.get("in_dir")
        if in_var is None:
            return
        value = in_var.get().strip()
        if not value:
            self._set_selector_csv_path_auto("")
            return
        self._update_selector_csv_default(value)

    def _on_selector_csv_path_changed(self, *_args) -> None:
        if self._selector_csv_updating:
            return
        if not self.selector_vars:
            return
        csv_var = self.selector_vars.get("csv_path")
        if csv_var is None:
            return
        current = csv_var.get().strip()
        if current == self.selector_csv_auto_value.strip():
            return
        self.selector_csv_auto = False
        self.selector_csv_auto_value = current

    def _on_selector_input_selected(self, selected_path: str) -> None:
        self.selector_csv_auto = True
        self._update_selector_csv_default(selected_path)

    def _selector_suspect_percent(self) -> float:
        """Return suspect display percent from the text box."""
        try:
            limit_source = self.selector_count_var.get() if self.selector_count_var is not None else ""
            limit_text = str(limit_source).strip()
            percent_text = limit_text.rstrip("%")
            limit_percent = float(percent_text) if percent_text else 5.0
        except (TypeError, ValueError):
            limit_percent = 5.0
        return max(0.1, min(limit_percent, 100.0))

    def _clear_selector_score_state(self) -> None:
        """Clear cached overview/manual-selection state."""
        self._close_selector_preview_panel()
        self.selector_last_scores = []
        self.selector_score_entries = []
        self.selector_score_suspect_positions = set()
        self.selector_score_csv_path_loaded = None
        self.selector_score_csv_fieldnames = []
        self.selector_score_selected_key = None
        self.selector_score_value_min = 0.0
        self.selector_score_value_range = 0.0
        self._set_selector_manual_buttons_state()

    def _selector_manual_edit_count(self) -> int:
        """Return the number of rows changed manually in the overview."""
        count = 0
        for entry in self.selector_score_entries:
            if bool(entry.get("selected_current")) != bool(entry.get("selected_original")):
                count += 1
        return count

    def _set_selector_manual_buttons_state(self) -> None:
        """Enable/disable manual apply/reset buttons based on current state."""
        modified = self._selector_manual_edit_count()
        has_data = bool(self.selector_score_entries)
        can_apply = bool(
            has_data
            and modified > 0
            and self.selector_score_csv_path_loaded is not None
            and self.selector_score_selected_key
        )
        if self.selector_manual_apply_button is not None:
            self.selector_manual_apply_button.configure(
                state="normal" if can_apply else "disabled"
            )
        if self.selector_manual_reset_button is not None:
            self.selector_manual_reset_button.configure(
                state="normal" if modified > 0 else "disabled"
            )

    def _sync_selector_last_scores_from_entries(self) -> None:
        """Update legacy tuple cache used by the overview renderer."""
        self.selector_last_scores = [
            (
                int(entry.get("frame_idx", idx)),
                bool(entry.get("selected_current", False)),
                entry.get("score"),
            )
            for idx, entry in enumerate(self.selector_score_entries)
        ]

    def _refresh_selector_score_view(self, reset_view: bool = False) -> None:
        """Re-render the sharpness overview from cached entries."""
        self._sync_selector_last_scores_from_entries()
        self._set_selector_manual_buttons_state()
        self._update_selector_score_view(
            self.selector_last_scores,
            self.selector_score_suspect_positions,
            reset_view=reset_view,
        )

    def _update_selector_score_summary(
        self,
        rows: List[Tuple[int, bool, Optional[float]]],
        suspect_positions: Set[int],
    ) -> None:
        """Update the Sharpness OverView summary label only."""
        if self.selector_summary_label is None:
            return
        if not rows:
            self.selector_summary_label.configure(text="No CSV loaded.")
            return

        total = len(rows)
        selected_count = sum(1 for _, selected, _ in rows if selected)
        suspect_count = len(suspect_positions)
        manual_edit_count = self._selector_manual_edit_count()
        selected_scores = [
            score for _, selected, score in rows if selected and score is not None
        ]
        if selected_scores:
            average_score = sum(selected_scores) / len(selected_scores)
            summary = (
                f"Frames: {total} | Selected: {selected_count} "
                f"| Suspects: {suspect_count} | Avg score: {average_score:.4f}"
            )
        else:
            summary = (
                f"Frames: {total} | Selected: {selected_count} | Suspects: {suspect_count}"
            )
        if manual_edit_count > 0:
            summary += f" | Manual edits: {manual_edit_count}"
        self.selector_summary_label.configure(text=summary)

    @staticmethod
    def _selector_score_bar_tag(idx: int) -> str:
        """Return canvas tag for one score bar group."""
        return "selector_score_bar_{}".format(idx)

    def _draw_selector_score_bar(
        self,
        canvas: tk.Canvas,
        rows: List[Tuple[int, bool, Optional[float]]],
        idx: int,
    ) -> None:
        """Draw or redraw one bar in the Sharpness OverView canvas."""
        if idx < 0 or idx >= len(rows):
            return
        tag = self._selector_score_bar_tag(idx)
        canvas.delete(tag)

        if self.selector_score_bar_width <= 0.0 or self.selector_score_bar_area_height <= 0:
            return

        _frame_idx, selected_flag, score_val = rows[idx]
        bar_width = float(self.selector_score_bar_width)
        bar_area_height = float(self.selector_score_bar_area_height)
        x0 = idx * bar_width
        x1 = x0 + bar_width

        color = "#4ecdc4" if selected_flag else "#d0d0d0"
        is_suspect = idx in self.selector_score_suspect_positions
        is_preview_open = idx in self.selector_preview_items
        is_preview_current = idx == self.selector_preview_panel_active_idx
        is_manual_changed = False
        if idx < len(self.selector_score_entries):
            entry = self.selector_score_entries[idx]
            is_manual_changed = bool(entry.get("selected_current", False)) != bool(
                entry.get("selected_original", False)
            )

        if score_val is not None and math.isfinite(score_val):
            score_range = float(self.selector_score_value_range)
            if score_range > 1e-9:
                norm = (float(score_val) - float(self.selector_score_value_min)) / score_range
            else:
                norm = 1.0
        else:
            norm = 0.0
        norm = max(0.0, min(1.0, norm))
        rect_height = max(1.0, norm * bar_area_height)
        y0 = bar_area_height - rect_height
        rect_w = x1 - x0
        rect_h = bar_area_height - y0

        canvas.create_rectangle(
            x0,
            y0,
            x1,
            bar_area_height,
            fill=color,
            outline="",
            width=0,
            tags=(tag,),
        )
        if is_suspect:
            canvas.create_rectangle(
                x0 + 0.5,
                y0 + 0.5,
                x1 - 0.5,
                bar_area_height - 0.5,
                fill="",
                outline="#ff6b6b",
                width=1,
                tags=(tag,),
            )
        if is_preview_open:
            canvas.create_rectangle(
                x0 + 1.5,
                y0 + 1.5,
                x1 - 1.5,
                bar_area_height - 1.5,
                fill="",
                outline="#3a86ff",
                width=2,
                tags=(tag,),
            )
        if is_preview_current and rect_w > 10 and rect_h > 10:
            canvas.create_rectangle(
                x0 + 4.5,
                y0 + 4.5,
                x1 - 4.5,
                bar_area_height - 4.5,
                fill="",
                outline="#1d4ed8",
                width=2,
                tags=(tag,),
            )
        if is_manual_changed and rect_w > 14 and rect_h > 14:
            canvas.create_rectangle(
                x0 + 7.0,
                y0 + 7.0,
                x1 - 7.0,
                bar_area_height - 7.0,
                fill="",
                outline="#f4a261",
                width=2,
                tags=(tag,),
            )

    def _refresh_selector_score_bars(
        self,
        indices: Iterable[int],
        update_summary: bool = False,
    ) -> None:
        """Redraw only specified bars in the overview canvas."""
        canvas = self.selector_score_canvas
        if canvas is None or not self.selector_last_scores:
            return
        if update_summary:
            self._update_selector_score_summary(
                self.selector_last_scores,
                self.selector_score_suspect_positions,
            )
        seen: Set[int] = set()
        for idx in indices:
            try:
                idx_int = int(idx)
            except Exception:
                continue
            if idx_int in seen:
                continue
            seen.add(idx_int)
            self._draw_selector_score_bar(canvas, self.selector_last_scores, idx_int)

    def _selector_bar_index_from_event(self, event: tk.Event) -> Optional[int]:
        """Map a mouse event on the overview canvas to a bar index."""
        canvas = self.selector_score_canvas
        if canvas is None or not self.selector_score_entries:
            return None
        if self.selector_score_bar_width <= 0.0:
            return None
        if int(getattr(event, "y", 0)) > int(self.selector_score_bar_area_height):
            return None
        x_world = float(canvas.canvasx(getattr(event, "x", 0)))
        idx = int(x_world // self.selector_score_bar_width)
        if idx < 0 or idx >= len(self.selector_score_entries):
            return None
        return idx

    @staticmethod
    def _selector_entry_name_stem(entry: Dict[str, Any]) -> str:
        """Return a display-friendly filename stem for an overview entry."""
        raw_name = str(entry.get("filename", "") or "").strip()
        if not raw_name:
            return ""
        try:
            return Path(raw_name).stem
        except Exception:
            return raw_name

    def _selector_entry_label_text(self, entry: Dict[str, Any]) -> str:
        """Build a concise label for logs/tooling from an overview entry."""
        frame_idx = entry.get("frame_idx", "")
        name_stem = self._selector_entry_name_stem(entry)
        if name_stem:
            return "index {} ({})".format(frame_idx, name_stem)
        return "index {}".format(frame_idx)

    def _selector_image_path_for_entry(self, entry: Dict[str, Any]) -> Optional[Path]:
        """Resolve the image file path for a CSV entry using the selector input dir."""
        raw_name = str(entry.get("filename", "") or "").strip()
        if not raw_name:
            return None
        try:
            candidate = Path(raw_name).expanduser()
        except Exception:
            return None
        if candidate.is_absolute():
            return candidate if candidate.exists() else None
        base_dir = None
        if self.selector_vars:
            in_var = self.selector_vars.get("in_dir")
            if in_var is not None:
                raw_base = str(in_var.get()).strip()
                if raw_base:
                    try:
                        base_dir = Path(raw_base).expanduser()
                    except Exception:
                        base_dir = None
        if base_dir is None:
            return None
        path_obj = base_dir / candidate
        return path_obj if path_obj.exists() else None

    def _close_selector_preview_panel(self) -> None:
        """Close the consolidated preview panel and clear preview state."""
        win = self.selector_preview_panel_window
        self.selector_preview_panel_window = None
        self.selector_preview_panel_image_label = None
        self.selector_preview_panel_image_canvas = None
        self.selector_preview_panel_canvas_image_id = None
        self.selector_preview_panel_info_label = None
        self.selector_preview_panel_status_label = None
        self.selector_preview_panel_index_label = None
        self.selector_preview_panel_slider = None
        self.selector_preview_panel_slider_var = None
        self.selector_preview_panel_zoom_reset_button = None
        self.selector_preview_panel_zoom_25_button = None
        self.selector_preview_panel_zoom_50_button = None
        self.selector_preview_panel_zoom_100_button = None
        self.selector_preview_panel_close_current_button = None
        self.selector_preview_panel_close_all_button = None
        self.selector_preview_items.clear()
        self.selector_preview_panel_active_idx = None
        self.selector_preview_panel_zoom_ratio = 1.0
        if win is not None:
            try:
                if win.winfo_exists():
                    win.destroy()
            except Exception:
                pass

    def _ensure_selector_preview_panel(self) -> Optional[tk.Toplevel]:
        """Create or return the consolidated preview panel window."""
        win = self.selector_preview_panel_window
        if win is not None:
            try:
                if win.winfo_exists() and self.selector_preview_panel_image_canvas is not None:
                    return win
            except Exception:
                pass
        self.selector_preview_panel_window = None
        self.selector_preview_panel_image_label = None
        self.selector_preview_panel_image_canvas = None
        self.selector_preview_panel_canvas_image_id = None
        self.selector_preview_panel_info_label = None
        self.selector_preview_panel_status_label = None
        self.selector_preview_panel_index_label = None
        self.selector_preview_panel_slider = None
        self.selector_preview_panel_slider_var = None
        self.selector_preview_panel_zoom_reset_button = None
        self.selector_preview_panel_zoom_25_button = None
        self.selector_preview_panel_zoom_50_button = None
        self.selector_preview_panel_zoom_100_button = None
        self.selector_preview_panel_close_current_button = None
        self.selector_preview_panel_close_all_button = None

        top = tk.Toplevel(self.root)
        top.title("FrameSelector Preview Panel")
        top.geometry("1000x560")

        outer = tk.Frame(top)
        outer.pack(fill="both", expand=True)

        image_frame = tk.Frame(outer, bg="#202020")
        image_frame.pack(fill="both", expand=True, padx=8, pady=(8, 6))
        image_canvas = tk.Canvas(
            image_frame,
            bg="#202020",
            highlightthickness=0,
            xscrollincrement=1,
            yscrollincrement=1,
        )
        xbar = tk.Scrollbar(image_frame, orient=tk.HORIZONTAL, command=image_canvas.xview)
        ybar = tk.Scrollbar(image_frame, orient=tk.VERTICAL, command=image_canvas.yview)
        image_canvas.configure(xscrollcommand=xbar.set, yscrollcommand=ybar.set)
        ybar.pack(side=tk.RIGHT, fill="y")
        xbar.pack(side=tk.BOTTOM, fill="x")
        image_canvas.pack(side=tk.LEFT, fill="both", expand=True)
        image_canvas.create_text(
            20,
            20,
            text="No Preview Image",
            fill="#dddddd",
            anchor="nw",
            tags=("preview_placeholder",),
        )

        info_frame = tk.Frame(outer)
        info_frame.pack(fill="x", padx=8, pady=(0, 4))
        info_label = tk.Label(info_frame, text="", anchor="w")
        info_label.pack(side=tk.LEFT, fill="x", expand=True)
        status_label = tk.Label(info_frame, text="", anchor="e")
        status_label.pack(side=tk.RIGHT, padx=(8, 0))

        slider_frame = tk.Frame(outer)
        slider_frame.pack(fill="x", padx=8, pady=(0, 8))
        tk.Label(slider_frame, text="Image").pack(side=tk.LEFT, padx=(0, 4))
        slider_var = tk.IntVar(value=1)
        slider = tk.Scale(
            slider_frame,
            from_=1,
            to=1,
            orient=tk.HORIZONTAL,
            variable=slider_var,
            showvalue=False,
            resolution=1,
            command=self._on_selector_preview_slider_changed,
        )
        slider.pack(side=tk.LEFT, fill="x", expand=True)
        zoom_reset_button = tk.Button(
            slider_frame,
            text="Zoom Reset",
            width=10,
            command=self._selector_preview_zoom_reset,
        )
        zoom_reset_button.pack(side=tk.LEFT, padx=(8, 0))
        zoom_25_button = tk.Button(
            slider_frame,
            text="25%",
            width=6,
            command=lambda: self._selector_preview_zoom_set_absolute(0.25),
        )
        zoom_25_button.pack(side=tk.LEFT, padx=(8, 0))
        zoom_50_button = tk.Button(
            slider_frame,
            text="50%",
            width=6,
            command=lambda: self._selector_preview_zoom_set_absolute(0.5),
        )
        zoom_50_button.pack(side=tk.LEFT, padx=(4, 0))
        zoom_100_button = tk.Button(
            slider_frame,
            text="100%",
            width=6,
            command=lambda: self._selector_preview_zoom_set_absolute(1.0),
        )
        zoom_100_button.pack(side=tk.LEFT, padx=(4, 0))
        close_current_button = tk.Button(
            slider_frame,
            text="Close Current",
            width=12,
            command=self._selector_preview_close_current,
        )
        close_current_button.pack(side=tk.LEFT, padx=(8, 0))
        close_all_button = tk.Button(
            slider_frame,
            text="Close All",
            width=10,
            command=self._selector_preview_close_all,
        )
        close_all_button.pack(side=tk.LEFT, padx=(4, 0))
        index_label = tk.Label(slider_frame, text="0/0", width=8, anchor="e")
        index_label.pack(side=tk.LEFT, padx=(8, 0))

        image_canvas.bind("<MouseWheel>", self._on_selector_preview_mousewheel)
        image_canvas.bind("<Button-4>", self._on_selector_preview_mousewheel)
        image_canvas.bind("<Button-5>", self._on_selector_preview_mousewheel)
        image_canvas.bind("<ButtonPress-3>", self._on_selector_preview_pan_start)
        image_canvas.bind("<B3-Motion>", self._on_selector_preview_pan_drag)

        def _on_close() -> None:
            changed_indices = list(self.selector_preview_items.keys())
            self._close_selector_preview_panel()
            self._refresh_selector_score_bars(changed_indices)

        def _on_resize(_event=None) -> None:
            self._render_selector_preview_panel_image()

        image_canvas.bind("<Configure>", _on_resize, add="+")
        top.protocol("WM_DELETE_WINDOW", _on_close)

        self.selector_preview_panel_window = top
        self.selector_preview_panel_image_label = None
        self.selector_preview_panel_image_canvas = image_canvas
        self.selector_preview_panel_canvas_image_id = None
        self.selector_preview_panel_info_label = info_label
        self.selector_preview_panel_status_label = status_label
        self.selector_preview_panel_slider = slider
        self.selector_preview_panel_slider_var = slider_var
        self.selector_preview_panel_zoom_reset_button = zoom_reset_button
        self.selector_preview_panel_zoom_25_button = zoom_25_button
        self.selector_preview_panel_zoom_50_button = zoom_50_button
        self.selector_preview_panel_zoom_100_button = zoom_100_button
        self.selector_preview_panel_close_current_button = close_current_button
        self.selector_preview_panel_close_all_button = close_all_button
        self.selector_preview_panel_index_label = index_label
        return top

    def _remove_selector_preview_item(self, idx: int) -> bool:
        """Remove one image from the preview set."""
        old_active = self.selector_preview_panel_active_idx
        item = self.selector_preview_items.pop(idx, None)
        if item is None:
            return False
        if self.selector_preview_panel_active_idx == idx:
            self.selector_preview_panel_active_idx = None
        self._sync_selector_preview_panel_controls(preserve_zoom=True)
        changed = [idx]
        if old_active is not None:
            changed.append(old_active)
        if self.selector_preview_panel_active_idx is not None:
            changed.append(self.selector_preview_panel_active_idx)
        self._refresh_selector_score_bars(changed)
        return True

    def _selector_preview_sorted_indices(self) -> List[int]:
        """Return preview-set indices sorted by frame index, then position."""
        def _key(i: int) -> Tuple[int, int]:
            if 0 <= i < len(self.selector_score_entries):
                frame_idx = int(self.selector_score_entries[i].get("frame_idx", i))
            else:
                frame_idx = i
            return frame_idx, i

        return sorted(self.selector_preview_items.keys(), key=_key)

    def _fit_selector_preview_zoom_ratio(self, idx: int) -> float:
        """Compute a fit-to-panel zoom ratio for the given preview item."""
        item = self.selector_preview_items.get(idx)
        if not item:
            return 1.0
        image = item.get("image")
        if image is None:
            return 1.0
        orig_w, orig_h = image.size
        if orig_w <= 0 or orig_h <= 0:
            return 1.0
        canvas = self.selector_preview_panel_image_canvas
        avail_w = 1000
        avail_h = 650
        if canvas is not None:
            try:
                w = int(canvas.winfo_width())
                h = int(canvas.winfo_height())
                if w > 10:
                    avail_w = w
                if h > 10:
                    avail_h = h
            except Exception:
                pass
        fit_ratio = min(float(avail_w) / float(orig_w), float(avail_h) / float(orig_h))
        fit_ratio = max(0.05, min(16.0, fit_ratio))
        return fit_ratio

    def _sync_selector_preview_panel_controls(self, preserve_zoom: bool = False) -> None:
        """Sync slider/current selection and redraw preview panel."""
        if not self.selector_preview_items:
            if self._ensure_selector_preview_panel() is None:
                return
            if self.selector_preview_panel_slider is not None:
                try:
                    self.selector_preview_panel_slider.configure(from_=1, to=1, state="disabled")
                except Exception:
                    pass
            if self.selector_preview_panel_slider_var is not None:
                self.selector_preview_panel_slider_var.set(1)
            if self.selector_preview_panel_index_label is not None:
                self.selector_preview_panel_index_label.configure(text="0/0")
            if self.selector_preview_panel_info_label is not None:
                self.selector_preview_panel_info_label.configure(text="")
            if self.selector_preview_panel_status_label is not None:
                self.selector_preview_panel_status_label.configure(text="Zoom: -")
            self.selector_preview_panel_active_idx = None
            self.selector_preview_panel_zoom_ratio = 1.0
            self._render_selector_preview_panel_image()
            return
        if self._ensure_selector_preview_panel() is None:
            return

        ordered = self._selector_preview_sorted_indices()
        active_idx = self.selector_preview_panel_active_idx
        if active_idx not in self.selector_preview_items:
            active_idx = ordered[0]
            self.selector_preview_panel_active_idx = active_idx
            preserve_zoom = False

        slider = self.selector_preview_panel_slider
        slider_var = self.selector_preview_panel_slider_var
        index_label = self.selector_preview_panel_index_label
        if slider is not None:
            try:
                slider.configure(from_=1, to=max(1, len(ordered)), state="normal")
            except Exception:
                pass
        if slider_var is not None and active_idx in ordered:
            pos = ordered.index(active_idx) + 1
            slider_var.set(pos)
            if index_label is not None:
                index_label.configure(text="{}/{}".format(pos, len(ordered)))

        if not preserve_zoom:
            self.selector_preview_panel_zoom_ratio = self._fit_selector_preview_zoom_ratio(active_idx)
        self._render_selector_preview_panel_image()

    def _selector_preview_close_current(self) -> None:
        """Close the currently displayed preview image from the preview set."""
        idx = self.selector_preview_panel_active_idx
        if idx is None:
            return
        self._remove_selector_preview_item(idx)

    def _selector_preview_close_all(self) -> None:
        """Close all preview images while keeping the preview panel window open."""
        if not self.selector_preview_items:
            self._sync_selector_preview_panel_controls(preserve_zoom=True)
            return
        changed = list(self.selector_preview_items.keys())
        self.selector_preview_items.clear()
        self.selector_preview_panel_active_idx = None
        self._sync_selector_preview_panel_controls(preserve_zoom=True)
        self._refresh_selector_score_bars(changed)

    def _render_selector_preview_panel_image(self) -> None:
        """Render the currently active preview image with current zoom ratio."""
        canvas = self.selector_preview_panel_image_canvas
        if canvas is None:
            return
        idx = self.selector_preview_panel_active_idx
        item = self.selector_preview_items.get(idx) if idx is not None else None
        if item is None:
            try:
                canvas.delete("all")
                canvas.create_text(
                    20,
                    20,
                    text="No Preview Image",
                    fill="#dddddd",
                    anchor="nw",
                )
                canvas.configure(scrollregion=(0, 0, max(1, canvas.winfo_width()), max(1, canvas.winfo_height())))
            except Exception:
                pass
            return

        image = item.get("image")
        image_path = item.get("path")
        if image is None:
            return
        orig_w, orig_h = image.size
        zoom = max(0.05, min(16.0, float(self.selector_preview_panel_zoom_ratio)))
        self.selector_preview_panel_zoom_ratio = zoom
        disp_w = max(1, int(round(orig_w * zoom)))
        disp_h = max(1, int(round(orig_h * zoom)))
        try:
            resized = image.resize((disp_w, disp_h), Image.LANCZOS)
        except Exception:
            resized = image.copy()
            disp_w, disp_h = resized.size
        photo = ImageTk.PhotoImage(resized)
        try:
            canvas.update_idletasks()
        except Exception:
            pass
        try:
            canvas_w = max(1, int(canvas.winfo_width() or 1))
            canvas_h = max(1, int(canvas.winfo_height() or 1))
        except Exception:
            canvas_w = disp_w
            canvas_h = disp_h
        offset_x = max(0, (canvas_w - disp_w) // 2)
        offset_y = max(0, (canvas_h - disp_h) // 2)
        canvas.delete("all")
        image_id = canvas.create_image(offset_x, offset_y, image=photo, anchor="nw")
        canvas.image = photo
        self.selector_preview_panel_canvas_image_id = image_id
        canvas.configure(scrollregion=(0, 0, max(disp_w, canvas_w), max(disp_h, canvas_h)))

        info_label = self.selector_preview_panel_info_label
        status_label = self.selector_preview_panel_status_label
        if info_label is not None:
            name = image_path.name if isinstance(image_path, Path) else str(image_path)
            info_label.configure(text="{} | orig {}x{} | view {}x{}".format(name, orig_w, orig_h, disp_w, disp_h))
        if status_label is not None:
            status_label.configure(text="Zoom: {:.1f}%".format(zoom * 100.0))

    def _center_selector_preview_view_on_image_center(self) -> None:
        """Center the preview canvas viewport on the current image center."""
        canvas = self.selector_preview_panel_image_canvas
        idx = self.selector_preview_panel_active_idx
        if canvas is None or idx is None:
            return
        item = self.selector_preview_items.get(idx)
        if not item:
            return
        image = item.get("image")
        if image is None:
            return
        try:
            canvas.update_idletasks()
        except Exception:
            pass
        try:
            canvas_w = max(1.0, float(canvas.winfo_width() or 1.0))
            canvas_h = max(1.0, float(canvas.winfo_height() or 1.0))
        except Exception:
            canvas_w = 1.0
            canvas_h = 1.0
        try:
            orig_w, orig_h = image.size
        except Exception:
            return
        zoom = max(0.05, min(16.0, float(self.selector_preview_panel_zoom_ratio)))
        disp_w = max(1.0, float(int(round(orig_w * zoom))))
        disp_h = max(1.0, float(int(round(orig_h * zoom))))
        total_w = max(disp_w, canvas_w)
        total_h = max(disp_h, canvas_h)
        left = (disp_w * 0.5) - (canvas_w * 0.5)
        top = (disp_h * 0.5) - (canvas_h * 0.5)
        left = max(0.0, min(max(0.0, total_w - canvas_w), left))
        top = max(0.0, min(max(0.0, total_h - canvas_h), top))
        try:
            if total_w > 0.0:
                canvas.xview_moveto(left / total_w)
            if total_h > 0.0:
                canvas.yview_moveto(top / total_h)
        except Exception:
            pass

    def _selector_preview_zoom_reset(self) -> None:
        """Reset preview zoom to fit the active image in the panel."""
        idx = self.selector_preview_panel_active_idx
        if idx is None or idx not in self.selector_preview_items:
            return
        self.selector_preview_panel_zoom_ratio = self._fit_selector_preview_zoom_ratio(idx)
        self._render_selector_preview_panel_image()
        self._center_selector_preview_view_on_image_center()

    def _selector_preview_zoom_set_absolute(self, ratio: float) -> None:
        """Set preview zoom ratio to an absolute scale (e.g. 25%% or 100%%)."""
        if self.selector_preview_panel_active_idx is None:
            return
        self.selector_preview_panel_zoom_ratio = max(0.05, min(16.0, float(ratio)))
        self._render_selector_preview_panel_image()
        self._center_selector_preview_view_on_image_center()

    def _on_selector_preview_slider_changed(self, value: str) -> None:
        """Switch active preview image using the slider."""
        if not self.selector_preview_items:
            return
        old_active = self.selector_preview_panel_active_idx
        ordered = self._selector_preview_sorted_indices()
        if not ordered:
            return
        try:
            pos = int(float(value))
        except Exception:
            pos = 1
        pos = max(1, min(len(ordered), pos))
        self.selector_preview_panel_active_idx = ordered[pos - 1]
        if self.selector_preview_panel_index_label is not None:
            self.selector_preview_panel_index_label.configure(text="{}/{}".format(pos, len(ordered)))
        self._render_selector_preview_panel_image()
        changed = [ordered[pos - 1]]
        if old_active is not None:
            changed.append(old_active)
        self._refresh_selector_score_bars(changed)

    def _on_selector_preview_mousewheel(self, event: tk.Event) -> str:
        """Zoom preview image in/out with the mouse wheel."""
        if not self.selector_preview_items:
            return "break"
        delta = 0
        num = getattr(event, "num", None)
        if num == 4:
            delta = 120
        elif num == 5:
            delta = -120
        else:
            try:
                delta = int(getattr(event, "delta", 0))
            except Exception:
                delta = 0
        if delta == 0:
            return "break"
        step = 1.10 if delta > 0 else (1.0 / 1.10)
        self.selector_preview_panel_zoom_ratio = max(
            0.05,
            min(16.0, float(self.selector_preview_panel_zoom_ratio) * step),
        )
        self._render_selector_preview_panel_image()
        return "break"

    def _on_selector_preview_pan_start(self, event: tk.Event) -> str:
        """Start panning the preview image with right-drag."""
        canvas = self.selector_preview_panel_image_canvas
        if canvas is None:
            return "break"
        try:
            canvas.scan_mark(int(event.x), int(event.y))
        except Exception:
            pass
        return "break"

    def _on_selector_preview_pan_drag(self, event: tk.Event) -> str:
        """Pan the preview image with right mouse drag."""
        canvas = self.selector_preview_panel_image_canvas
        if canvas is None:
            return "break"
        try:
            canvas.scan_dragto(int(event.x), int(event.y), gain=1)
        except Exception:
            pass
        return "break"

    def _on_selector_score_canvas_click(self, event: tk.Event) -> None:
        """Toggle selected/non-selected flag by clicking a bar."""
        idx = self._selector_bar_index_from_event(event)
        if idx is None:
            return
        if idx < 0 or idx >= len(self.selector_score_entries):
            return
        entry = self.selector_score_entries[idx]
        entry["selected_current"] = not bool(entry.get("selected_current", False))
        state_text = "ON" if bool(entry.get("selected_current", False)) else "OFF"
        self._append_text_widget(
            self.selector_log,
            "[manual] {} {}".format(state_text, self._selector_entry_label_text(entry)),
        )
        self._sync_selector_last_scores_from_entries()
        self._set_selector_manual_buttons_state()
        self._refresh_selector_score_bars([idx], update_summary=True)

    def _on_selector_score_canvas_right_click(self, event: tk.Event) -> str:
        """Toggle preview window for the clicked overview bar."""
        idx = self._selector_bar_index_from_event(event)
        if idx is None:
            return "break"
        self._toggle_selector_preview_window(idx)
        return "break"

    def _add_selector_preview_item_by_index(
        self,
        idx: int,
        show_errors: bool = True,
    ) -> bool:
        """Load one frame image and add it to the preview set."""
        if idx < 0 or idx >= len(self.selector_score_entries):
            return False
        entry = self.selector_score_entries[idx]
        image_path = self._selector_image_path_for_entry(entry)
        if image_path is None:
            if show_errors:
                messagebox.showerror(
                    "FrameSelector",
                    "Could not resolve image path for {}.".format(
                        self._selector_entry_label_text(entry)
                    ),
                )
            return False
        try:
            with Image.open(image_path) as img:
                image = img.convert("RGB").copy()
        except Exception as exc:
            if show_errors:
                messagebox.showerror(
                    "FrameSelector",
                    "Failed to open image:\n{}\n\n{}".format(image_path, exc),
                )
            return False
        self.selector_preview_items[idx] = {
            "image": image,
            "path": image_path,
        }
        return True

    def _toggle_selector_preview_window(self, idx: int) -> None:
        """Open/close one image inside the consolidated preview panel."""
        if self._remove_selector_preview_item(idx):
            return
        old_active = self.selector_preview_panel_active_idx
        if not self._add_selector_preview_item_by_index(idx, show_errors=True):
            return
        self.selector_preview_panel_active_idx = idx
        preserve_zoom = old_active is not None
        self._sync_selector_preview_panel_controls(preserve_zoom=preserve_zoom)
        self._center_selector_preview_view_on_image_center()
        changed = [idx]
        if old_active is not None:
            changed.append(old_active)
        self._refresh_selector_score_bars(changed)

    def _open_selector_suspects_preview(self) -> None:
        """Confirm and open all current suspect frames in the preview panel."""
        if not self.selector_score_entries:
            messagebox.showinfo(
                "FrameSelector",
                "Load a Sharpness OverView CSV and run Check Selection first.",
            )
            return
        suspect_indices = sorted(
            idx for idx in self.selector_score_suspect_positions
            if 0 <= idx < len(self.selector_score_entries)
        )
        if not suspect_indices:
            messagebox.showinfo(
                "FrameSelector",
                "No suspects are currently marked. Run Check Selection first.",
            )
            return
        proceed = messagebox.askokcancel(
            "FrameSelector",
            "Open {} suspect image(s) in FrameSelector Preview Panel?".format(
                len(suspect_indices)
            ),
        )
        if not proceed:
            return

        old_active = self.selector_preview_panel_active_idx
        had_preview_items = bool(self.selector_preview_items)
        opened = 0
        failed = 0
        changed = set()

        for idx in suspect_indices:
            already_open = idx in self.selector_preview_items
            ok = self._add_selector_preview_item_by_index(idx, show_errors=False)
            if ok:
                if not already_open:
                    opened += 1
                changed.add(idx)
            else:
                failed += 1

        if not self.selector_preview_items:
            if failed > 0:
                messagebox.showerror(
                    "FrameSelector",
                    "Failed to open all suspect images.",
                )
            return

        active_idx = None
        for idx in suspect_indices:
            if idx in self.selector_preview_items:
                active_idx = idx
                break
        if active_idx is None and self.selector_preview_panel_active_idx in self.selector_preview_items:
            active_idx = self.selector_preview_panel_active_idx
        if active_idx is None:
            active_idx = sorted(self.selector_preview_items.keys())[0]

        self.selector_preview_panel_active_idx = active_idx
        preserve_zoom = had_preview_items or (old_active is not None)
        self._sync_selector_preview_panel_controls(preserve_zoom=preserve_zoom)
        self._center_selector_preview_view_on_image_center()
        if old_active is not None:
            changed.add(old_active)
        if active_idx is not None:
            changed.add(active_idx)
        self._refresh_selector_score_bars(changed)

        self._append_text_widget(
            self.selector_log,
            "[preview] Opened {} suspect image(s){}.".format(
                opened,
                "" if failed <= 0 else " ({} failed)".format(failed),
            ),
        )

    def _on_selector_score_canvas_mousewheel(self, event: tk.Event) -> str:
        """Zoom the overview X-axis (bar width) using the mouse wheel."""
        canvas = self.selector_score_canvas
        if canvas is None or not self.selector_score_entries:
            return "break"

        delta = 0
        num = getattr(event, "num", None)
        if num == 4:
            delta = 120
        elif num == 5:
            delta = -120
        else:
            raw_delta = getattr(event, "delta", 0)
            try:
                delta = int(raw_delta)
            except Exception:
                delta = 0
        if delta == 0:
            return "break"

        old_zoom = float(self.selector_score_zoom)
        zoom_step = 1.15 if delta > 0 else (1.0 / 1.15)
        new_zoom = max(0.25, min(100.0, old_zoom * zoom_step))
        if abs(new_zoom - old_zoom) < 1e-9:
            return "break"

        x_world_before = float(canvas.canvasx(getattr(event, "x", 0)))
        total_width_before = max(1.0, float(self.selector_score_total_width or 1.0))
        anchor_ratio = max(0.0, min(1.0, x_world_before / total_width_before))

        self.selector_score_zoom = new_zoom
        self._refresh_selector_score_view(reset_view=False)

        canvas.update_idletasks()
        total_width_after = max(1.0, float(self.selector_score_total_width or 1.0))
        view_width = max(1.0, float(canvas.winfo_width() or 1.0))
        target_x_world = anchor_ratio * total_width_after
        left = target_x_world - float(getattr(event, "x", 0))
        max_left = max(0.0, total_width_after - view_width)
        left = max(0.0, min(max_left, left))
        if total_width_after > 0.0:
            canvas.xview_moveto(left / total_width_after)
        return "break"

    def _reset_selector_manual_selection(self) -> None:
        """Reset manual selection edits to the CSV-loaded state."""
        if not self.selector_score_entries:
            return
        changed = False
        for entry in self.selector_score_entries:
            original = bool(entry.get("selected_original", False))
            if bool(entry.get("selected_current", False)) != original:
                entry["selected_current"] = original
                changed = True
        if changed:
            self._refresh_selector_score_view(reset_view=False)
            self._append_text_widget(
                self.selector_log,
                "[manual] Reset manual selection edits.",
            )

    def _confirm_selector_manual_selection(self) -> None:
        """Write manual selection edits back to the loaded CSV."""
        csv_path = self.selector_score_csv_path_loaded
        selected_key = self.selector_score_selected_key
        if csv_path is None or not selected_key or not self.selector_score_entries:
            messagebox.showerror(
                "FrameSelector",
                "Load a selection CSV in Sharpness OverView before applying manual edits.",
            )
            return

        modified = self._selector_manual_edit_count()
        if modified <= 0:
            self._append_text_widget(self.selector_log, "[manual] No manual edits to apply.")
            self._set_selector_manual_buttons_state()
            return

        fieldnames = list(self.selector_score_csv_fieldnames)
        if not fieldnames:
            first_row = self.selector_score_entries[0].get("row", {})
            if isinstance(first_row, dict):
                fieldnames = list(first_row.keys())
        if not fieldnames:
            messagebox.showerror("FrameSelector", "CSV header information is missing.")
            return

        try:
            with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for entry in self.selector_score_entries:
                    row_obj = entry.get("row")
                    row_dict = dict(row_obj) if isinstance(row_obj, dict) else {}
                    row_dict[selected_key] = "1" if bool(entry.get("selected_current", False)) else "0"
                    writer.writerow(row_dict)
                    entry["row"] = row_dict
                    entry["selected_original"] = bool(entry.get("selected_current", False))
        except Exception as exc:
            messagebox.showerror(
                "FrameSelector",
                f"Failed to update selection CSV:\n{exc}",
            )
            return

        self._refresh_selector_score_view(reset_view=False)
        self._append_text_widget(
            self.selector_log,
            f"[manual] Applied {modified} manual selection edit(s) to {csv_path.name}",
        )

    def _show_selector_scores(self) -> None:
        duration_message = None
        if self._selector_duration_pending:
            duration_message = self._selector_duration_message
            self._selector_duration_message = None
            self._selector_duration_pending = False
        try:
            if not self.selector_vars:
                return
            csv_var = self.selector_vars.get("csv_path")
            csv_path_raw = csv_var.get().strip() if csv_var is not None else ""
            if not csv_path_raw:
                messagebox.showerror("FrameSelector", "Select a CSV file in the Path field.")
                return
            csv_path = Path(csv_path_raw).expanduser()
            try:
                with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
                    reader = csv.DictReader(f)
                    headers = reader.fieldnames or []
                    field_map = {name.lower(): name for name in headers if name}
                    selected_key = field_map.get("selected(1=keep)") or field_map.get("selected")
                    score_key = field_map.get("score")
                    filename_key = field_map.get("filename")
                    index_key = field_map.get("index")
                    if not selected_key or not score_key:
                        raise ValueError("CSV must contain 'selected(1=keep)' (or 'selected') and 'score' columns.")
                    parsed_entries: List[Dict[str, Any]] = []
                    row_counter = 0
                    for row in reader:
                        if not row:
                            continue
                        flag = str(row.get(selected_key, "")).strip().lower()
                        if flag not in {"1", "true", "yes", "keep"}:
                            selected_flag = False
                        else:
                            selected_flag = True
                        score_raw = row.get(score_key)
                        try:
                            score_val = float(score_raw)
                        except (TypeError, ValueError):
                            score_val = None
                        if score_val is not None and not math.isfinite(score_val):
                            score_val = None
                        if index_key:
                            idx_raw = row.get(index_key, "")
                        else:
                            idx_raw = ""
                        try:
                            frame_idx = int(idx_raw)
                        except (TypeError, ValueError):
                            frame_idx = row_counter
                        fname = row.get(filename_key, "") if filename_key else ""
                        idx_text = row.get(index_key, "") if index_key else ""
                        parsed_entries.append(
                            {
                                "row_counter": row_counter,
                                "frame_idx": frame_idx,
                                "selected_original": selected_flag,
                                "selected_current": selected_flag,
                                "score": score_val,
                                "filename": fname,
                                "index_text": idx_text,
                                "row": dict(row),
                            }
                        )
                        row_counter += 1
            except FileNotFoundError:
                messagebox.showerror("FrameSelector", f"CSV file not found:\n{csv_path}")
                self._clear_selector_score_state()
                self._update_selector_score_view([], set(), reset_view=True)
                return
            except ValueError as exc:
                messagebox.showerror("FrameSelector", str(exc))
                self._clear_selector_score_state()
                self._update_selector_score_view([], set(), reset_view=True)
                return
            except Exception as exc:  # pragma: no cover - unexpected CSV error
                messagebox.showerror("FrameSelector", f"Failed to read CSV:\n{exc}")
                self._clear_selector_score_state()
                self._update_selector_score_view([], set(), reset_view=True)
                return

            parsed_entries.sort(key=lambda item: int(item.get("frame_idx", 0)))
            selected_entries: List[Tuple[float, str, str, int]] = []
            for pos, entry in enumerate(parsed_entries):
                if not bool(entry.get("selected_current", False)):
                    continue
                score_val = entry.get("score")
                if score_val is None:
                    continue
                selected_entries.append(
                    (
                        float(score_val),
                        str(entry.get("filename", "")),
                        str(entry.get("index_text", "")),
                        pos,
                    )
                )

            selected_entries.sort(key=lambda item: item[0])
            selected_flag_total = sum(1 for entry in parsed_entries if bool(entry.get("selected_current", False)))
            limit_percent = self._selector_suspect_percent()
            if selected_entries:
                max_lines = max(1, min(200, math.ceil((limit_percent / 100.0) * len(selected_entries))))
                suspects = selected_entries[:max_lines]
            else:
                max_lines = 0
                suspects = []
            suspect_positions = {entry[3] for entry in suspects}

            self.selector_score_entries = parsed_entries
            self.selector_score_suspect_positions = suspect_positions
            self.selector_score_csv_path_loaded = csv_path
            self.selector_score_csv_fieldnames = list(headers)
            self.selector_score_selected_key = selected_key
            self._refresh_selector_score_view(reset_view=True)

            self._append_text_widget(
                self.selector_log,
                (
                    f"[score] {csv_path.name}: selected={selected_flag_total} "
                    f"showing lowest {len(suspects)} scores ({limit_percent:.1f}% of selected, limit={max_lines})"
                ),
            )
            if not suspects:
                self._append_text_widget(self.selector_log, "[score] No low-score selections detected.")
                return

            for score, fname, idx_text, _pos in suspects:
                label = fname or (f"index {idx_text}" if idx_text else "(unknown)")
                self._append_text_widget(
                    self.selector_log,
                    f"  - {label} (score={score:.4f})",
                )
        finally:
            if duration_message:
                self._append_text_widget(self.selector_log, duration_message)

    def _update_selector_score_view(
        self,
        rows: List[Tuple[int, bool, Optional[float]]],
        suspect_positions: Set[int],
        reset_view: bool = True,
    ) -> None:
        self._update_selector_score_summary(rows, suspect_positions)

        canvas = self.selector_score_canvas
        if canvas is None:
            return
        canvas.delete("all")
        if not rows:
            canvas.create_text(
                int(canvas.winfo_width() or 200) / 2,
                int(canvas.cget("height") or 36) / 2,
                text="No data",
                fill="#666666",
            )
            return

        canvas.update_idletasks()
        view_width = canvas.winfo_width()
        if view_width <= 1:
            view_width = int(canvas.cget("width") or 600)
            if view_width <= 1:
                view_width = 600
            canvas.configure(width=view_width)
        total = len(rows)
        height = int(canvas.cget("height") or 64)
        label_height = 34
        bar_area_height = max(20, height - label_height)
        approx_width = view_width / float(total) if total > 0 else view_width
        prev_xview = canvas.xview()
        zoom = max(0.25, min(100.0, float(self.selector_score_zoom)))
        bar_width = max(2.0, min(640.0, approx_width * zoom))
        total_width = max(bar_width * total, view_width)
        self.selector_score_bar_width = bar_width
        self.selector_score_bar_area_height = bar_area_height
        self.selector_score_total_width = total_width
        canvas.configure(scrollregion=(0, 0, total_width, height))
        if reset_view:
            canvas.xview_moveto(0.0)
        else:
            left_frac = prev_xview[0] if prev_xview else 0.0
            canvas.xview_moveto(max(0.0, min(1.0, left_frac)))
        suspect_set = set(suspect_positions)
        score_values = [
            score
            for _, _, score in rows
            if score is not None and math.isfinite(score)
        ]
        if score_values:
            score_min = min(score_values)
            score_max = max(score_values)
        else:
            score_min = 0.0
            score_max = 0.0
        self.selector_score_value_min = float(score_min)
        self.selector_score_value_range = float(score_max - score_min)
        for idx in range(len(rows)):
            self._draw_selector_score_bar(canvas, rows, idx)
        canvas.create_rectangle(0, 0, bar_width * total, bar_area_height, outline="#808080")

        tick_total = 1 if total <= 1 else max(2, min(total, int(total_width / 140.0)))
        used_indices: List[int] = []
        if total <= tick_total:
            used_indices = list(range(total))
        else:
            steps = tick_total - 1 if tick_total > 1 else 1
            seen = set()
            for i in range(tick_total):
                position = int(round((total - 1) * (i / steps)))
                if position in seen:
                    continue
                seen.add(position)
                used_indices.append(position)
        tick_y0 = bar_area_height
        text_y = bar_area_height + (label_height / 2)
        for idx in used_indices:
            frame_idx, _, _ = rows[idx]
            x = idx * bar_width + (bar_width / 2.0)
            canvas.create_line(x, tick_y0, x, tick_y0 + 4, fill="#808080")
            entry_name = ""
            if idx < len(self.selector_score_entries):
                entry_name = self._selector_entry_name_stem(self.selector_score_entries[idx])
            if entry_name and len(entry_name) > 14:
                entry_name = entry_name[:11] + "..."
            label_text = str(frame_idx) if not entry_name else "{}\n{}".format(frame_idx, entry_name)
            canvas.create_text(
                x,
                text_y,
                text=label_text,
                fill="#333333",
                font=("TkDefaultFont", 7),
            )

    def _select_directory(
        self,
        var: tk.Variable,
        title: str = "Select folder",
        on_select: Optional[Callable[[str], None]] = None,
    ) -> None:
        current = var.get().strip() if hasattr(var, "get") else ""
        initialdir = Path(current) if current else self.base_dir
        try:
            initialdir = initialdir if initialdir.exists() else self.base_dir
        except Exception:
            initialdir = self.base_dir
        path = filedialog.askdirectory(
            title=title,
            initialdir=str(initialdir),
        )
        if path:
            var.set(path)
            if on_select is not None:
                try:
                    on_select(path)
                except Exception:
                    pass

    def _select_save_file(
        self,
        var: tk.Variable,
        title: str,
        defaultextension: str,
        filetypes: Sequence[Tuple[str, str]],
    ) -> None:
        current = var.get().strip() if hasattr(var, "get") else ""
        initialdir = Path(current).parent if current else self.base_dir
        try:
            initialdir = initialdir if initialdir.exists() else self.base_dir
        except Exception:
            initialdir = self.base_dir
        path = filedialog.asksaveasfilename(
            title=title,
            initialdir=str(initialdir),
            defaultextension=defaultextension,
            filetypes=filetypes,
        )
        if path:
            var.set(path)

    def _set_video_mode_controls(self, enabled: bool) -> None:
        for name in self.video_only_fields:
            widget = self.field_widgets.get(name)
            if widget is not None:
                target_state = "normal"
                if isinstance(widget, ttk.Combobox):
                    target_state = "readonly"
                state = target_state if enabled else "disabled"
                try:
                    widget.configure(state=state)
                except tk.TclError:
                    pass
            var = self.field_vars.get(name)
            if isinstance(var, tk.BooleanVar):
                if not enabled:
                    persisted = bool(self.video_persist_state.get(name, False))
                    var.set(persisted)
            elif isinstance(var, tk.StringVar):
                if not enabled:
                    persisted = self.video_persist_state.get(name)
                    var.set("" if persisted in (None, "") else str(persisted))
        if not enabled:
            fps_text = self.field_vars["fps"].get().strip() if "fps" in self.field_vars else ""
            try:
                fps_value = float(fps_text) if fps_text else None
            except ValueError:
                fps_value = None
            if fps_value is not None and fps_value > 0:
                self.video_persist_state["fps"] = fps_value
            self.video_persist_state["keep_rec709"] = bool(self.field_vars["keep_rec709"].get())
            start_text = self.field_vars["start"].get().strip() if "start" in self.field_vars else ""
            end_text = self.field_vars["end"].get().strip() if "end" in self.field_vars else ""
            try:
                start_value = float(start_text) if start_text else None
            except ValueError:
                start_value = None
            try:
                end_value = float(end_text) if end_text else None
            except ValueError:
                end_value = None
            self.video_persist_state["start"] = start_value
            self.video_persist_state["end"] = end_value
            setattr(self.current_args, "fps", None)
            setattr(self.current_args, "keep_rec709", False)
            setattr(self.current_args, "start", None)
            setattr(self.current_args, "end", None)
            self._video_preview_signature = None
        self._update_preview_inspect_state()
        self._update_preview_csv_state()

    def _on_show_seam_toggle(self, var: tk.BooleanVar) -> None:
        value = bool(var.get())
        setattr(self.current_args, "show_seam_overlay", value)
        self.refresh_overlays()

    def _generate_video_preview_image(self, video_path: pathlib.Path) -> Optional[Image.Image]:
        ffmpeg_path = self.ffmpeg_path_var.get().strip() or getattr(self.defaults, "ffmpeg", "ffmpeg")
        if not ffmpeg_path:
            messagebox.showerror("Video Preview", "ffmpeg path is not configured.")
            return None
        keep_rec709 = bool(getattr(self.current_args, "keep_rec709", False))
        colorspace_filter = "colorspace=iall=bt709:all=smpte170m"
        if not keep_rec709:
            colorspace_filter += ":trc=iec61966-2-1"
        colorspace_filter += ":format=yuv444p"
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
                tmp_path = Path(tmp_file.name)
        except Exception as exc:
            messagebox.showerror("Video Preview", f"Failed to allocate temporary file.\n{exc}")
            return None
        cmd = [
            ffmpeg_path,
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-i", str(video_path),
            "-vf", colorspace_filter,
            "-frames:v", "1",
            str(tmp_path),
        ]
        try:
            subprocess.run(cmd, check=True)
        except FileNotFoundError:
            messagebox.showerror("Video Preview", f"ffmpeg not found: {ffmpeg_path}")
            try:
                tmp_path.unlink()
            except OSError:
                pass
            return None
        except subprocess.CalledProcessError as exc:
            messagebox.showerror(
                "Video Preview",
                f"ffmpeg failed to extract preview frame.\nCommand: {' '.join(cmd)}\n{exc}",
            )
            try:
                tmp_path.unlink()
            except OSError:
                pass
            return None
        try:
            with Image.open(tmp_path) as img:
                return img.convert("RGB")
        except Exception as exc:
            messagebox.showerror("Video Preview", f"Failed to load preview frame.\n{exc}")
            return None
        finally:
            try:
                tmp_path.unlink()
            except OSError:
                pass

    def _prepare_video_preview(self, video_path: pathlib.Path) -> bool:
        resolved_path = video_path
        try:
            resolved_path = video_path.resolve()
        except FileNotFoundError:
            pass
        keep_rec709 = bool(getattr(self.current_args, "keep_rec709", False))
        signature = (resolved_path, keep_rec709)
        if self._video_preview_signature == signature and self.display_image is not None:
            return True
        preview_image = self._generate_video_preview_image(video_path)
        if preview_image is None:
            return False
        self.pano_image = preview_image
        self.pano_width, self.pano_height = preview_image.size
        try:
            self.scale = compute_scale(
                self.pano_width,
                self.pano_height,
                self.scale_override,
                self.view_max_width,
                self.view_max_height,
            )
        except ValueError as exc:
            messagebox.showerror("Video Preview", str(exc))
            self.scale = 1.0
        self.display_width = int(round(self.pano_width * self.scale))
        self.display_height = int(round(self.pano_height * self.scale))
        if self.scale != 1.0:
            self.display_image = preview_image.resize(
                (self.display_width, self.display_height),
                Image.LANCZOS,
            )
        else:
            self.display_image = preview_image.copy()
        self.update_canvas_image()
        self._video_preview_signature = signature
        self.image_path = video_path
        return True

    def _default_output_path(self) -> Optional[pathlib.Path]:
        if not self.in_dir:
            return None
        in_path = pathlib.Path(self.in_dir)
        try:
            if self.source_is_video or in_path.is_file():
                base_dir = in_path.parent
            else:
                base_dir = in_path
        except Exception:
            base_dir = pathlib.Path.cwd()
        return base_dir / "_geometry"

    def set_form_values(self) -> None:
        self.form_snapshot.clear()
        for definition in self.FIELD_DEFS:
            name = definition["name"]
            field_type = definition["type"]
            if definition.get("video_only") and not self.source_is_video:
                if field_type == "bool":
                    var = self.field_vars[name]
                    assert isinstance(var, tk.BooleanVar)
                    var.set(False)
                    self.form_snapshot[name] = "0"
                else:
                    var = self.field_vars[name]
                    assert isinstance(var, tk.StringVar)
                    var.set("")
                    self.form_snapshot[name] = ""
                continue
            value = getattr(self.current_args, name)
            if field_type == "bool":
                var = self.field_vars[name]
                assert isinstance(var, tk.BooleanVar)
                flag_value = bool(value)
                if name == "keep_rec709":
                    flag_value = not flag_value
                var.set(flag_value)
                self.form_snapshot[name] = str(int(flag_value))
                continue
            if field_type == "choice":
                var = self.field_vars[name]
                assert isinstance(var, tk.StringVar)
                choice = str(value) if value else definition["choices"][0]
                if choice not in definition["choices"]:
                    choice = definition["choices"][0]
                var.set(choice)
                self.form_snapshot[name] = choice
                continue
            var = self.field_vars[name]
            assert isinstance(var, tk.StringVar)
            display = "" if value is None else str(value)
            var.set(display)
            self.form_snapshot[name] = display
        self.ffmpeg_path_var.set(str(getattr(self.current_args, "ffmpeg", getattr(self.defaults, "ffmpeg", "ffmpeg"))))
        self.jobs_var.set(str(getattr(self.current_args, "jobs", getattr(self.defaults, "jobs", "auto"))))
        if self.out_dir:
            self.output_path_var.set(str(self.out_dir))
        elif self.in_dir:
            default_out = self._default_output_path()
            self.output_path_var.set(str(default_out) if default_out is not None else "")
        else:
            self.output_path_var.set("")
        self._update_jpeg_quality_state()
        self._update_preview_inspect_state()
        self._update_preview_csv_state()


    def _bind_help(self, widget: tk.Widget, key: str) -> None:
        text = FIELD_HELP_TEXT.get(key)
        if not text:
            return
        tooltip = ToolTip(widget, text)
        self._tooltips.append(tooltip)

    def _update_jpeg_quality_state(self) -> None:
        checkbox = self.field_widgets.get("jpeg_quality_95")
        ext_var = self.field_vars.get("ext")
        quality_var = self.field_vars.get("jpeg_quality_95")
        if checkbox is None or ext_var is None or quality_var is None:
            return
        ext_value = ext_var.get().strip().lower()
        enabled = ext_value == "jpg"
        state = "normal" if enabled else "disabled"
        try:
            checkbox.configure(state=state)
        except tk.TclError:
            return
        if not enabled:
            quality_var.set(False)

    def _update_preview_csv_state(self) -> None:
        enabled = bool(self.source_is_video)
        entry = self.preview_csv_entry
        button = self.preview_csv_button
        state = "normal" if enabled else "disabled"
        csv_filled = bool(self.preview_csv_var.get().strip()) if self.preview_csv_var is not None else False
        try:
            if entry is not None:
                entry.configure(state=state)
            if button is not None:
                button.configure(state=state)
        except tk.TclError:
            pass
        fps_widget = self.field_widgets.get("fps")
        if fps_widget is not None:
            fps_state = "disabled"
            if self.source_is_video and not csv_filled:
                fps_state = "normal"
            try:
                fps_widget.configure(state=fps_state)
            except tk.TclError:
                pass
        range_state = "disabled"
        if self.source_is_video and not csv_filled:
            range_state = "normal"
        for name in ("start", "end"):
            widget = self.field_widgets.get(name)
            if widget is None:
                continue
            try:
                widget.configure(state=range_state)
            except tk.TclError:
                pass

    def _browse_preview_csv(self) -> None:
        if not self.source_is_video:
            return
        current = self.preview_csv_var.get().strip()
        initial_dir = Path(current).expanduser().parent if current else (self.in_dir.parent if self.in_dir else self.base_dir)
        try:
            if not initial_dir.exists():
                initial_dir = self.base_dir
        except Exception:
            initial_dir = self.base_dir
        path = filedialog.askopenfilename(
            title="Select selection CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir=str(initial_dir),
        )
        if path:
            self.preview_csv_var.set(path)

    def collect_updated_args(self) -> Optional[argparse.Namespace]:
        updated = clone_namespace(self.current_args)
        ensure_explicit_flags(updated)

        def previous_flag(attr: str) -> bool:
            return bool(getattr(self.current_args, f"{attr}_explicit", False))

        for definition in self.FIELD_DEFS:
            name = definition["name"]
            field_type = definition["type"]
            default_value = getattr(self.defaults, name, None)
            if definition.get("video_only") and not self.source_is_video:
                if field_type == "bool":
                    setattr(updated, name, False)
                else:
                    setattr(updated, name, None)
                continue
            if field_type == "bool":
                var = self.field_vars[name]
                assert isinstance(var, tk.BooleanVar)
                value = bool(var.get())
                if name == "keep_rec709":
                    value = not value
                setattr(updated, name, value)
                continue
            if field_type == "choice":
                var = self.field_vars[name]
                assert isinstance(var, tk.StringVar)
                choice = var.get().strip() or definition["choices"][0]
                if choice not in definition["choices"]:
                    messagebox.showerror(
                        "Input Error",
                        f"{definition['label']}: invalid value '{choice}'",
                    )
                    return None
                setattr(updated, name, choice)
                self.form_snapshot[name] = choice
                continue
            var = self.field_vars[name]
            assert isinstance(var, tk.StringVar)
            raw = var.get().strip()
            snapshot = self.form_snapshot.get(name, "")
            try:
                if field_type == "int":
                    if not raw:
                        if default_value is None:
                            raise ValueError("Please enter a value")
                        value = int(default_value)
                    else:
                        value = int(raw)
                    if name == "count" and value <= 0:
                        raise ValueError("count must be >= 1")
                    setattr(updated, name, value)
                    self.form_snapshot[name] = str(value)
                elif field_type == "float":
                    if not raw:
                        if default_value is None:
                            raise ValueError("Please enter a value")
                        value = float(default_value)
                    else:
                        value = float(raw)
                    setattr(updated, name, value)
                    self.form_snapshot[name] = str(value)
                elif field_type == "float_optional":
                    if raw:
                        value = float(raw)
                        if name == "fps" and value <= 0:
                            raise ValueError("must be > 0")
                        if name in {"start", "end"} and value < 0:
                            raise ValueError("must be >= 0")
                        setattr(updated, name, value)
                        setattr(
                            updated,
                            f"{name}_explicit",
                            previous_flag(name) if raw == snapshot else True,
                        )
                        self.form_snapshot[name] = raw
                    else:
                        if definition.get("required_if_video") and self.source_is_video:
                            raise ValueError("Please enter a value")
                        setattr(updated, name, default_value)
                        setattr(updated, f"{name}_explicit", False)
                        self.form_snapshot[name] = "" if default_value is None else str(default_value)
                else:
                    if not raw:
                        value = "" if default_value is None else str(default_value)
                        setattr(updated, name, value)
                        self.form_snapshot[name] = value
                    else:
                        setattr(updated, name, raw)
                        self.form_snapshot[name] = raw
                if name in {"addcam", "delcam"}:
                    setattr(updated, f"{name}_explicit", True)
            except ValueError as exc:
                messagebox.showerror("Input Error", f"{definition['label']}: {exc}")
                return None

            if field_type in {"int", "float"} and name in self.EXPLICIT_FIELDS:
                setattr(
                    updated,
                    f"{name}_explicit",
                    previous_flag(name) if raw == snapshot and raw else bool(raw),
                )

        start_val = getattr(updated, "start", None)
        end_val = getattr(updated, "end", None)
        if start_val is not None and end_val is not None and end_val <= start_val:
            messagebox.showerror("Input Error", "End (s) must be greater than Start (s).")
            return None
        if self.source_is_video and bool(self.preview_csv_var.get().strip()):
            updated.start = None
            updated.end = None

        ffmpeg_path = self.ffmpeg_path_var.get().strip()
        if not ffmpeg_path:
            ffmpeg_path = getattr(self.defaults, "ffmpeg", "ffmpeg")
        updated.ffmpeg = ffmpeg_path
        self.ffmpeg_path_var.set(ffmpeg_path)

        jobs_value = self.jobs_var.get().strip()
        if not jobs_value:
            jobs_value = str(getattr(self.defaults, "jobs", "auto"))
        jobs_lower = jobs_value.lower()
        if jobs_lower != "auto":
            try:
                jobs_int = int(jobs_value)
            except ValueError:
                messagebox.showerror("Input Error", "Parallel jobs must be an integer or 'auto'.")
                return None
            if jobs_int <= 0:
                messagebox.showerror("Input Error", "Parallel jobs must be a positive integer.")
                return None
            auto_jobs = self._auto_preview_jobs()
            max_jobs = max(1, auto_jobs * 3)
            if jobs_int > max_jobs:
                messagebox.showerror(
                    "Input Error",
                    f"Parallel jobs must be <= {max_jobs} (auto={auto_jobs}).",
                )
                return None
            jobs_value = str(jobs_int)
        updated.jobs = jobs_value
        self.jobs_var.set(jobs_value)

        default_out = self._default_output_path()
        out_dir_text = self.output_path_var.get().strip()
        if out_dir_text:
            out_dir_path = Path(out_dir_text).expanduser()
            auto_match = False
            if default_out is not None:
                try:
                    auto_match = out_dir_path.resolve() == default_out.resolve()
                except Exception:
                    auto_match = False
            self.out_dir = out_dir_path
            normalized = str(out_dir_path)
            updated.out_dir = normalized
            self.output_path_var.set(normalized)
            self._out_dir_custom = self._out_dir_custom or not auto_match
        else:
            self.out_dir = None
            updated.out_dir = None
            self.output_path_var.set("")
            self._out_dir_custom = False

        return updated

    def on_ext_changed(self, selection: str) -> None:
        var = self.field_vars.get("ext")
        if var is not None:
            var.set(selection)
        self._update_jpeg_quality_state()

    def _apply_preset_defaults(self, preset_value: str) -> None:
        preset_defaults: Dict[str, Dict[str, Any]] = {
            "fisheyelike": {"count": 10, "focal_mm": 17.0, "delcam": "C,D,H,I", "addcam": "A,F"},
            "full360coverage": {"count": 8, "focal_mm": 14.0, "delcam": "B,D,F,H", "addcam": "B,D,F,H"},
            "2views": {"size": 3600, "focal_mm": 6.0, "delcam": "B,C,D,F,G,H"},
            "evenMinus30": {"setcam": "B:D30,D:D30,F:D30,H:D30"},
            "evenPlus30": {"setcam": "B:U30,D:U30,F:U30,H:U30"},
            "fisheyeXY": {"count": 8, "size": 3600, "hfov": 180.0},
        }
        values = preset_defaults.get(preset_value)
        if not values:
            return
        # Reset HFOV to allow focal-based presets to take effect.
        self.current_args.hfov = None
        if hasattr(self.current_args, "hfov_explicit"):
            setattr(self.current_args, "hfov_explicit", False)
        for field_name, field_value in values.items():
            setattr(self.current_args, field_name, field_value)
            explicit_flag = f"{field_name}_explicit"
            if hasattr(self.current_args, explicit_flag):
                setattr(self.current_args, explicit_flag, False)

    def on_preset_changed(self, selection: str) -> None:
        preset_value = selection or self.field_vars["preset"].get()
        current_ffmpeg = self.ffmpeg_path_var.get().strip() or getattr(self.defaults, "ffmpeg", "ffmpeg")
        current_jobs = self.jobs_var.get().strip() or str(getattr(self.defaults, "jobs", "auto"))
        output_path_text = self.output_path_var.get()
        fps_text = self.field_vars["fps"].get() if "fps" in self.field_vars else ""
        start_text = self.field_vars["start"].get() if "start" in self.field_vars else ""
        end_text = self.field_vars["end"].get() if "end" in self.field_vars else ""
        keep_ui_var = self.field_vars.get("keep_rec709")
        keep_rec709_ui = bool(keep_ui_var.get()) if isinstance(keep_ui_var, tk.BooleanVar) else None

        def _parse_float(text: str) -> Optional[float]:
            cleaned = (text or "").strip()
            if not cleaned:
                return None
            try:
                return float(cleaned)
            except ValueError:
                return None

        video_prev_values = {
            "fps": _parse_float(fps_text),
            "start": _parse_float(start_text),
            "end": _parse_float(end_text),
        }
        if video_prev_values["fps"] is None:
            video_prev_values["fps"] = getattr(self.current_args, "fps", None)
        if video_prev_values["start"] is None:
            video_prev_values["start"] = getattr(self.current_args, "start", None)
        if video_prev_values["end"] is None:
            video_prev_values["end"] = getattr(self.current_args, "end", None)
        if keep_rec709_ui is None:
            video_prev_values["keep_rec709"] = getattr(self.current_args, "keep_rec709", False)
        else:
            video_prev_values["keep_rec709"] = not keep_rec709_ui
        video_ui_values = {
            "fps": fps_text,
            "start": start_text,
            "end": end_text,
            "keep_rec709": keep_rec709_ui,
        }
        seam_prev = getattr(self.current_args, "show_seam_overlay", False)
        self.current_args = clone_namespace(self.defaults)
        ensure_explicit_flags(self.current_args)
        self.current_args.preset = preset_value
        self.current_args.ffmpeg = current_ffmpeg
        self.current_args.jobs = current_jobs
        self.current_args.show_seam_overlay = seam_prev
        # Keep video flags when a video source is loaded.
        if self.source_is_video:
            setattr(self.current_args, "input_is_video", True)
            if hasattr(self.current_args, "video_bit_depth"):
                setattr(self.current_args, "video_bit_depth", getattr(self.current_args, "video_bit_depth"))
        self._apply_preset_defaults(preset_value)
        if getattr(self.current_args, "input_is_video", False) or self.source_is_video:
            fps_value = video_prev_values.get("fps")
            if fps_value is None or fps_value <= 0:
                fps_value = self.video_persist_state.get("fps", 1.0)
            self.current_args.fps = fps_value
            setattr(self.current_args, "fps_explicit", True)
            keep_val = bool(video_prev_values.get("keep_rec709", False))
            self.current_args.keep_rec709 = keep_val
            self.current_args.start = video_prev_values.get("start")
            self.current_args.end = video_prev_values.get("end")
        self.set_form_values()
        self.field_vars["preset"].set(preset_value)
        if getattr(self.current_args, "input_is_video", False) or self.source_is_video:
            fps_var = self.field_vars.get("fps")
            start_var = self.field_vars.get("start")
            end_var = self.field_vars.get("end")
            if isinstance(fps_var, tk.StringVar):
                fps_var.set(video_ui_values.get("fps") or "")
            if isinstance(start_var, tk.StringVar):
                start_var.set(video_ui_values.get("start") or "")
            if isinstance(end_var, tk.StringVar):
                end_var.set(video_ui_values.get("end") or "")
            if isinstance(keep_ui_var, tk.BooleanVar) and video_ui_values.get("keep_rec709") is not None:
                keep_ui_var.set(bool(video_ui_values.get("keep_rec709")))
        self.output_path_var.set(output_path_text)
        updated = self.collect_updated_args()
        if updated is None:
            return
        self.current_args = updated
        ensure_explicit_flags(self.current_args)
        self.refresh_overlays()

    def load_directory_from_entry(self) -> None:
        path_value = self.folder_path_var.get().strip()
        if not path_value:
            messagebox.showerror("Input Source", "Please enter a folder or video file path, or use Browse.")
            return
        if self.try_load_directory(path_value):
            self.refresh_overlays(initial=True)
            if self.out_dir:
                self.output_path_var.set(str(self.out_dir))
            elif self.in_dir:
                self._update_default_output_path()

    def _browse_image_input(self) -> None:
        current = self.folder_path_var.get().strip()
        initial_dir = None
        if current:
            try:
                path_obj = Path(current).expanduser()
                initial_dir = path_obj if path_obj.is_dir() else path_obj.parent
            except Exception:
                initial_dir = None
        if initial_dir is None:
            initial_dir = self.in_dir if self.in_dir and self.in_dir.is_dir() else self.base_dir
        selected = filedialog.askdirectory(parent=self.root, initialdir=str(initial_dir))
        if selected:
            if self.try_load_directory(selected):
                self.folder_path_var.set(str(self.in_dir))
                self.refresh_overlays(initial=True)

    def _browse_video_input(self) -> None:
        current = self.folder_path_var.get().strip()
        initial_dir = None
        if current:
            try:
                path_obj = Path(current).expanduser()
                initial_dir = path_obj.parent if path_obj.is_file() else path_obj
            except Exception:
                initial_dir = None
        if initial_dir is None:
            initial_dir = self.in_dir.parent if self.in_dir and self.in_dir.is_file() else self.base_dir
        selected = filedialog.askopenfilename(
            parent=self.root,
            initialdir=str(initial_dir),
            title="Select input video",
            filetypes=[
                ("Video files", "*.mp4 *.mov *.m4v *.avi *.mkv *.wmv *.mpg *.mpeg *.mxf"),
                ("All files", "*.*"),
            ],
        )
        if selected:
            if self.try_load_directory(selected):
                self.folder_path_var.set(str(self.in_dir))
                self.refresh_overlays(initial=True)

    def browse_ffmpeg(self) -> None:
        filename = filedialog.askopenfilename(
            title="Select ffmpeg executable",
            filetypes=[("ffmpeg", "ffmpeg.exe"), ("All files", "*.*")]
        )
        if filename:
            self.ffmpeg_path_var.set(filename)

    def reset_ffmpeg_path(self) -> None:
        default_value = getattr(self.defaults, "ffmpeg", "ffmpeg")
        self.ffmpeg_path_var.set(default_value)

    def browse_output_directory(self) -> None:
        if self.out_dir:
            initial_path = self.out_dir
        elif self.in_dir:
            initial_path = self.in_dir.parent if self.source_is_video else self.in_dir
        else:
            initial_path = Path.cwd()
        initial = str(initial_path)
        selected = filedialog.askdirectory(parent=self.root, initialdir=initial)
        if selected:
            path_obj = Path(selected).expanduser()
            self.out_dir = path_obj
            self.current_args.out_dir = str(path_obj)
            self.output_path_var.set(str(path_obj))
            self._out_dir_custom = True

    def _update_default_output_path(self) -> None:
        default_out = self._default_output_path()
        if self._out_dir_custom:
            if self.out_dir:
                self.output_path_var.set(str(self.out_dir))
            elif default_out is not None:
                self.output_path_var.set(str(default_out))
            return
        self.out_dir = default_out
        if default_out is not None:
            self.current_args.out_dir = str(default_out)
            self.output_path_var.set(str(default_out))
        else:
            self.current_args.out_dir = None
            self.output_path_var.set("")

    def prompt_for_directory(self, initial: bool = False) -> None:
        initial_path = self.in_dir
        if initial_path is None:
            initial_path = pathlib.Path.cwd()
        else:
            try:
                initial_path = initial_path if initial_path.is_dir() else initial_path.parent
            except Exception:
                initial_path = pathlib.Path.cwd()
        selected_dir = filedialog.askdirectory(parent=self.root, initialdir=str(initial_path))
        if selected_dir:
            if self.try_load_directory(selected_dir):
                self.folder_path_var.set(str(self.in_dir))
                self.refresh_overlays(initial=True)
            return
        video_file = filedialog.askopenfilename(
            parent=self.root,
            initialdir=str(initial_path),
            title="Select input video",
            filetypes=[
                ("Video files", "*.mp4 *.mov *.m4v *.avi *.mkv *.wmv *.mpg *.mpeg *.mxf"),
                ("All files", "*.*"),
            ],
        )
        if not video_file:
            if initial and not self.files:
                self.set_log_text("No folder or video selected.")
            return
        if self.try_load_directory(video_file):
            self.folder_path_var.set(str(self.in_dir))
            self.refresh_overlays(initial=True)

    def _load_video_source(self, source: pathlib.Path) -> bool:
        video_path = pathlib.Path(source).expanduser()
        try:
            video_path = video_path.resolve()
        except FileNotFoundError:
            messagebox.showerror("Error", f"File not found: {source}")
            return False
        if not video_path.is_file():
            messagebox.showerror("Error", f"File not found: {video_path}")
            return False
        self.in_dir = video_path
        self.files = [video_path]
        self.current_args.input_dir = str(video_path)
        setattr(self.current_args, "input_is_video", True)
        setattr(
            self.current_args,
            "video_bit_depth",
            cutter.detect_input_bit_depth(video_path),
        )
        default_out = self._default_output_path()
        if self._out_dir_custom and getattr(self.current_args, "out_dir", None):
            try:
                self.out_dir = pathlib.Path(self.current_args.out_dir).expanduser().resolve()
            except Exception:
                self.out_dir = default_out
                if self.out_dir is not None:
                    self.current_args.out_dir = str(self.out_dir)
        else:
            self.out_dir = default_out
            if self.out_dir is not None:
                self.current_args.out_dir = str(self.out_dir)
            else:
                self.current_args.out_dir = None
            self._out_dir_custom = False
        fps_current = getattr(self.current_args, "fps", None)
        if fps_current is None or fps_current <= 0:
            self.current_args.fps = 1.0
            setattr(self.current_args, "fps_explicit", True)
        if not self._prepare_video_preview(video_path):
            self.source_is_video = False
            self.files = []
            self.image_path = None
            self.pano_image = None
            self.display_image = None
            self._update_preview_inspect_state()
            return False
        self.source_is_video = True
        self.folder_path_var.set(str(video_path))
        self._set_video_mode_controls(True)
        if self.out_dir:
            self.output_path_var.set(str(self.out_dir))
        else:
            self._update_default_output_path()
        self.set_form_values()
        return True

    def try_load_directory(
        self,
        directory: pathlib.Path,
        image_hint: Optional[str] = None,
    ) -> bool:
        path = pathlib.Path(directory).expanduser()
        try:
            path = path.resolve()
        except FileNotFoundError:
            messagebox.showerror("Error", f"Path not found: {directory}")
            return False
        if path.is_file():
            return self._load_video_source(path)
        if not path.is_dir():
            messagebox.showerror("Error", f"Folder not found: {path}")
            return False

        files = [
            p
            for p in sorted(path.iterdir())
            if p.is_file() and p.suffix.lower() in cutter.EXTS
        ]
        if not files:
            messagebox.showerror("Error", "No panorama images found (tif/jpg/png).")
            return False

        self.source_is_video = False
        self.in_dir = path
        self.files = files
        self.current_args.input_dir = str(path)
        setattr(self.current_args, "input_is_video", False)
        setattr(self.current_args, "video_bit_depth", 8)

        default_out = self._default_output_path()
        if self._out_dir_custom and getattr(self.current_args, "out_dir", None):
            try:
                self.out_dir = pathlib.Path(self.current_args.out_dir).expanduser().resolve()
            except Exception:
                self.out_dir = default_out
                if self.out_dir is not None:
                    self.current_args.out_dir = str(self.out_dir)
        else:
            self.out_dir = default_out
            if self.out_dir is not None:
                self.current_args.out_dir = str(self.out_dir)
            else:
                self.current_args.out_dir = None
            self._out_dir_custom = False

        if image_hint:
            hint_path = path / pathlib.Path(image_hint).name
            self.image_path = hint_path if hint_path in files else files[0]
        else:
            self.image_path = files[0]

        self.folder_path_var.set(str(self.in_dir))
        self._set_video_mode_controls(False)
        self.load_image()
        if self.out_dir:
            self.output_path_var.set(str(self.out_dir))
        else:
            self._update_default_output_path()
        self.set_form_values()
        return True

    def load_image(self) -> None:
        if self.image_path is None:
            return
        try:
            with Image.open(self.image_path) as img:
                pano = img.copy()
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to open image:\n{self.image_path}\n{exc}")
            return
        if pano.mode not in {"RGB", "RGBA", "L"}:
            try:
                pano = pano.convert("RGB")
            except Exception as exc:
                messagebox.showerror("Error", f"Unsupported image mode '{pano.mode}' for {self.image_path.name}:\n{exc}")
                return
        elif pano.mode == "L":
            # Ensure consistent 8-bit grayscale data
            pano = pano.convert("L")

        self.pano_image = pano
        self.pano_width, self.pano_height = self.pano_image.size

        try:
            self.scale = compute_scale(
                self.pano_width,
                self.pano_height,
                self.scale_override,
                self.view_max_width,
                self.view_max_height,
            )
        except ValueError as exc:
            messagebox.showerror("Error", str(exc))
            self.scale = 1.0

        self.display_width = int(round(self.pano_width * self.scale))
        self.display_height = int(round(self.pano_height * self.scale))
        if self.scale != 1.0:
            self.display_image = self.pano_image.resize(
                (self.display_width, self.display_height),
                Image.LANCZOS,
            )
        else:
            self.display_image = self.pano_image.copy()
        self.update_canvas_image()
        self._sync_panel_heights()

    def update_canvas_image(self) -> None:
        if self.canvas is None or self.display_image is None:
            return
        self.canvas.delete("all")
        self.photo = ImageTk.PhotoImage(self.display_image, master=self.root)
        self.canvas_offset_x = 0.0
        self.canvas_offset_y = 0.0
        self.canvas.configure(width=self.display_width, height=self.display_height)
        self.canvas_image_id = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.canvas.image = self.photo
        self._sync_panel_heights()

    def build_cli_command_line(self) -> str:
        parts = ["python", "cli_tools/gs360_360PerspCut.py"]
        if self.in_dir:
            parts.extend(["-i", str(self.in_dir)])
        if self.out_dir:
            default_out = self._default_output_path()
            if self.current_args.out_dir:
                parts.extend(["-o", str(self.out_dir)])
            elif default_out is not None and self.out_dir != default_out:
                parts.extend(["-o", str(self.out_dir)])

        ffmpeg_value = getattr(self.current_args, "ffmpeg", "")
        ffmpeg_default = getattr(self.defaults, "ffmpeg", "ffmpeg")
        if ffmpeg_value and ffmpeg_value != ffmpeg_default:
            parts.extend(["--ffmpeg", ffmpeg_value])

        csv_selected = bool(self.preview_csv_var.get().strip())
        defaults = self.defaults
        option_map = [
            ("--preset", "preset"),
            ("--count", "count"),
            ("--addcam", "addcam"),
            ("--addcam-deg", "addcam_deg"),
            ("--delcam", "delcam"),
            ("--setcam", "setcam"),
            ("--size", "size"),
            ("--sensor-mm", "sensor_mm"),
            ("--ext", "ext"),
            ("-f", "fps"),
            ("--start", "start"),
            ("--end", "end"),
            ("--jobs", "jobs"),
            ("--print-cmd", "print_cmd"),
        ]
        for flag, name in option_map:
            value = getattr(self.current_args, name, None)
            if value in (None, ""):
                continue
            if name in {"start", "end"} and csv_selected:
                continue
            if name == "addcam" and not str(value).strip():
                continue
            if name in {"delcam", "setcam"} and not str(value).strip():
                continue
            if name == "jobs" and str(value).lower() == "auto":
                continue
            if name == "print_cmd" and value == "once":
                continue
            if name == "ext" and str(value).lower() in {"jpg", "jpeg"}:
                continue
            default_value = getattr(defaults, name, None)
            if name in self.EXPLICIT_FIELDS:
                if not getattr(self.current_args, f"{name}_explicit", False) and value == default_value:
                    continue
            else:
                if value == default_value:
                    continue
            parts.extend([flag, str(value)])

        if getattr(self.current_args, "hfov", None) is not None:
            parts.extend(["--hfov", str(self.current_args.hfov)])
        else:
            focal_default = getattr(self.original_args, "focal_mm", None)
            if getattr(self.current_args, "focal_mm_explicit", False) or self.current_args.focal_mm != focal_default:
                parts.extend(["--focal-mm", str(self.current_args.focal_mm)])

        if getattr(self.current_args, "add_top", False):
            parts.append("--add-top")
        if getattr(self.current_args, "add_bottom", False):
            parts.append("--add-bottom")
        if getattr(self.current_args, "jpeg_quality_95", False):
            parts.append("--jpeg-quality-95")
        if self.source_is_video and getattr(self.current_args, "keep_rec709", False):
            parts.append("--keep-rec709")
        if getattr(self.current_args, "dry_run", False):
            parts.append("--dry-run")

        return "CLI> " + " ".join(shlex.quote(str(token)) for token in parts)

    def refresh_overlays(self, initial: bool = False) -> None:
        if not self.in_dir or not self.files:
            return
        cutter.stop_event.clear()
        source_is_video = bool(self.source_is_video)
        if source_is_video:
            video_path = self.files[0]
            if not self._prepare_video_preview(video_path):
                return
            fps_value = getattr(self.current_args, "fps", None)
            if fps_value is None or fps_value <= 0:
                if self.canvas is not None:
                    self.canvas.delete("overlay")
                self.set_log_text("Please enter a positive FPS for the video source and press Update.")
                return
            try:
                target_path = video_path.resolve()
            except FileNotFoundError:
                target_path = video_path
        else:
            if self.image_path is None:
                return
            if self.display_image is None:
                self.load_image()
                if self.display_image is None:
                    return
            target_path = self.image_path.resolve()

        args_for_build = clone_namespace(self.current_args)
        ensure_explicit_flags(args_for_build)
        args_for_build.input_dir = str(self.in_dir)
        args_for_build.out_dir = str(self.out_dir) if self.out_dir else None
        setattr(args_for_build, "input_is_video", source_is_video)
        setattr(
            args_for_build,
            "video_bit_depth",
            getattr(self.current_args, "video_bit_depth", 8),
        )

        default_out = self._default_output_path()
        output_root = self.out_dir or default_out or self.in_dir

        self.result = cutter.build_view_jobs(
            args_for_build,
            self.files,
            output_root,
        )
        self.matched_specs = [
            spec
            for spec in self.result.view_specs
            if spec.source_path.resolve() == target_path
        ]

        if self.canvas is not None:
            self._reposition_canvas_items()
            self.canvas.delete("overlay")
            if self.display_image is not None:
                offset_x = self.canvas_offset_x
                offset_y = self.canvas_offset_y
                draw_items: List[Tuple[List[List[Tuple[float, float]]], Tuple[float, float], str, str]] = []
                for spec, color in zip(self.matched_specs, itertools.cycle(COLOR_CYCLE)):
                    segments, centre = sample_view_segments(
                        spec,
                        self.pano_width,
                        self.pano_height,
                        self.samples,
                    )
                    draw_items.append((segments, centre, color, spec.view_id))
                seam_line_specs: Optional[Tuple[List[float], int]] = None
                if getattr(self.current_args, "show_seam_overlay", False) and self.display_width > 0:
                    def yaw_to_x(yaw_deg: float) -> float:
                        yaw_norm = ((yaw_deg + 180.0) % 360.0) - 180.0
                        return (yaw_norm + 180.0) / 360.0 * self.display_width

                    mid_yaws = [90.0, -90.0]
                    line_width = max(4, int(self.display_width * 0.01))
                    seam_line_specs = ([yaw_to_x(yaw_center) for yaw_center in mid_yaws], line_width)

                for segments, centre, color, view_id in draw_items:
                    for segment in segments:
                        if len(segment) < 2:
                            continue
                        flat_coords: List[float] = []
                        for px, py in segment:
                            flat_coords.extend([px * self.scale + offset_x, py * self.scale + offset_y])
                        self.canvas.create_line(
                            flat_coords,
                            fill=color,
                            width=2,
                            tags=("overlay",),
                        )
                    if not self.hide_labels:
                        self.canvas.create_text(
                            centre[0] * self.scale + offset_x,
                            centre[1] * self.scale + offset_y,
                            text=view_id,
                            fill=color,
                            font=("TkDefaultFont", 20, "bold"),
                            tags=("overlay",),
                        )
                if seam_line_specs:
                    seam_xs, line_width = seam_line_specs
                    for x in seam_xs:
                        self.canvas.create_line(
                            x + offset_x,
                            offset_y,
                            x + offset_x,
                            offset_y + self.display_height,
                            fill="#000000",
                            width=line_width,
                            tags=("overlay", "seam_line"),
                        )

        info_lines: List[str] = [self.build_cli_command_line()]

        try:
            jobs_parallel = self._effective_jobs()
        except Exception:
            jobs_parallel = self.current_args.jobs
        total_units = len(self.result.jobs)
        if self.source_is_video:
            selected_count: Optional[int] = None
            if self.preview_csv_var is not None:
                csv_text = self.preview_csv_var.get().strip()
                if csv_text:
                    selected_count = self._count_selected_frames_quiet(Path(csv_text))
            frames_per_job = self._estimate_frames_per_job(selected_count)
            if frames_per_job is None or frames_per_job <= 0:
                frames_per_job = 1
            total_units = frames_per_job * len(self.result.jobs)
        info_lines.append(
            f"[INFO] parallel jobs: {jobs_parallel} / total outputs: {total_units}"
        )
        if self.result.preview_views_line:
            info_lines.append(self.result.preview_views_line)
        if self.result.sensor_line:
            info_lines.append(self.result.sensor_line)
        if self.result.realityscan_line:
            info_lines.append(self.result.realityscan_line)
        if self.result.metashape_line:
            info_lines.append(self.result.metashape_line)

        if self.matched_specs:
            info_lines.append("")
            info_lines.append(f"[{target_path.name}] view list:")
            info_lines.extend(build_info_lines(self.matched_specs))
        else:
            info_lines.append("")
            info_lines.append("No views defined for the current source with these settings.")

        self.set_log_text("\n".join(info_lines))

        if not initial:
            self.set_form_values()

        self._sync_panel_heights()

    def on_update(self) -> None:
        updated = self.collect_updated_args()
        if updated is None:
            return
        self.current_args = updated
        ensure_explicit_flags(self.current_args)
        self.refresh_overlays()

    def on_execute(self) -> None:
        if self.is_executing:
            return
        updated = self.collect_updated_args()
        if updated is None:
            return
        self.current_args = updated
        ensure_explicit_flags(self.current_args)
        self._selected_frame_indices = None
        if self.source_is_video:
            csv_path_text = self.preview_csv_var.get().strip()
            if csv_path_text:
                try:
                    indices, total_rows = self._load_selected_frames_from_csv(Path(csv_path_text))
                except FileNotFoundError as exc:
                    messagebox.showerror("Selection CSV", str(exc))
                    return
                except Exception as exc:
                    messagebox.showerror("Selection CSV", f"Failed to read CSV:\n{exc}")
                    return
                if total_rows <= 0:
                    messagebox.showerror("Selection CSV", "CSV has no rows.")
                    return
                if not indices:
                    messagebox.showerror("Selection CSV", "No rows with selected=1 were found.")
                    return
                video_path = Path(self.in_dir) if self.in_dir else None
                estimate = None
                estimate_fps = None
                if video_path is not None:
                    estimate, estimate_fps = self._get_estimated_frames_info(video_path)
                fps_out = getattr(self.current_args, "fps", None)
                if estimate is not None:
                    expected_output = estimate
                    if fps_out and estimate_fps and estimate_fps > 0:
                        expected_output = max(1, int(round(estimate * (float(fps_out) / float(estimate_fps)))))
                    tol_output = max(1, int(expected_output * 0.05))
                    delta_output = abs(total_rows - expected_output)
                    if delta_output <= tol_output:
                        if delta_output > 0:
                            self.append_log_line(
                                f"[warn] CSV rows ({total_rows}) differ from estimated output frames ({expected_output}) by {delta_output}; proceeding."
                            )
                    else:
                        tol_input = max(1, int(estimate * 0.05))
                        delta_input = abs(total_rows - estimate)
                        fps_out_val = float(fps_out) if fps_out else None
                        fps_reduced = fps_out_val is not None and estimate_fps and fps_out_val < estimate_fps
                        if delta_input <= tol_input and fps_reduced:
                            self.append_log_line(
                                f"[info] CSV rows ({total_rows}) match source frames ({estimate}) "
                                f"but output fps={fps_out_val} < input fps={estimate_fps}; accepting CSV row count."
                            )
                        else:
                            messagebox.showerror(
                                "Selection CSV",
                                f"CSV rows ({total_rows}) do not match estimated output frames ({expected_output}).\n"
                                f"(Estimated input frames: {estimate}, input fps: {estimate_fps}, output fps: {fps_out})",
                            )
                            return
                else:
                    self.append_log_line("[info] Estimated frame count unavailable; proceeding without count check.")
                self._selected_frame_indices = indices
                self.append_log_line(f"[select] Using {len(indices)} selected frames from CSV ({total_rows} rows).")
        self.result = None
        if self.result is None or not self.result.jobs:
            self.refresh_overlays()
        if self.result is None or not self.result.jobs:
            messagebox.showinfo("Information", "No jobs to run with the current settings.")
            return
        selected_indices = getattr(self, "_selected_frame_indices", None)
        jobs_snapshot = list(self.result.jobs)
        if self.source_is_video and selected_indices:
            jobs_snapshot = self._apply_frame_selection_to_jobs(jobs_snapshot, selected_indices)
        planned_total = 0
        if self.source_is_video:
            frames_per_job = len(selected_indices) if selected_indices else self._estimate_frames_per_job(None)
            if frames_per_job is None or frames_per_job <= 0:
                frames_per_job = 1
            planned_total = frames_per_job * len(jobs_snapshot)
        else:
            planned_total = len(jobs_snapshot)
        planned_line = f"\nPlanned outputs: {planned_total:,} images." if planned_total > 0 else ""
        confirm_text = (
            "Run export with the current settings?\n"
            "FFmpeg will write results to the output folder."
            f"{planned_line}"
        )
        if not messagebox.askyesno("Confirm", confirm_text):
            return
        if self.out_dir:
            self.out_dir.mkdir(parents=True, exist_ok=True)
        cutter.stop_event.clear()
        self.is_executing = True
        if self.update_button is not None:
            self.update_button.configure(state="disabled")
        if self.execute_button is not None:
            self.execute_button.configure(state="disabled")
        if self.preview_stop_button is not None:
            self.preview_stop_button.configure(state="normal")
        self.append_log_line("[EXEC] Starting export...")
        thread = threading.Thread(
            target=self._run_execute_jobs,
            args=(jobs_snapshot,),
            daemon=True,
        )
        thread.start()

    def _apply_frame_selection_to_jobs(
        self,
        jobs: List[Tuple[List[str], str, str]],
        indices: Sequence[int],
    ) -> List[Tuple[List[str], str, str]]:
        if not indices:
            return jobs
        clauses = [f"eq(n\\,{idx})" for idx in indices]
        select_filter = "select='" + "+".join(clauses) + "'"
        updated: List[Tuple[List[str], str, str]] = []
        for cmd, src, dst in jobs:
            cmd = list(cmd)
            # Normalize seek args: move -ss/-to after input for CSV selection to preserve timestamps.
            input_idx = cmd.index("-i") if "-i" in cmd else None
            seek_args: List[str] = []
            def _pull_flag(flag: str) -> None:
                while flag in cmd:
                    idx_flag = cmd.index(flag)
                    cmd.pop(idx_flag)
                    if idx_flag < len(cmd):
                        seek_args.append(flag)
                        seek_args.append(cmd.pop(idx_flag))
            _pull_flag("-ss")
            _pull_flag("-to")
            if "-vf" in cmd:
                vf_idx = cmd.index("-vf")
                if vf_idx + 1 < len(cmd):
                    filters = cmd[vf_idx + 1]
                    new_filters = select_filter
                    if filters:
                        parts = filters.split(",")
                        fps_parts = [p for p in parts if p.strip().startswith("fps=")]
                        other_parts = [p for p in parts if not p.strip().startswith("fps=")]
                        # When using CSV selection, drop fps filter to avoid double sub-sampling.
                        if other_parts:
                            new_filters = f"{select_filter},{','.join(other_parts)}"
                        else:
                            new_filters = select_filter
                    cmd[vf_idx + 1] = new_filters
            # Preserve original frame numbers in filenames when using CSV selection.
            if "-frame_pts" not in cmd:
                insert_at = max(1, len(cmd) - 1)
                cmd.insert(insert_at, "1")
                cmd.insert(insert_at, "-frame_pts")
            # Remove start_number to allow frame_pts-driven numbering.
            while "-start_number" in cmd:
                idx = cmd.index("-start_number")
                cmd.pop(idx)  # flag
                if idx < len(cmd):
                    cmd.pop(idx)  # value
            if "-vsync" not in cmd:
                insert_at = max(1, len(cmd) - 1)
                cmd.insert(insert_at, "vfr")
                cmd.insert(insert_at, "-vsync")
            if "-copyts" not in cmd:
                insert_at = 1
                cmd.insert(insert_at, "-copyts")
            if input_idx is not None:
                # Recompute input index after prior modifications.
                try:
                    input_idx = cmd.index("-i")
                except ValueError:
                    input_idx = None
            if input_idx is not None and seek_args:
                insert_at = input_idx + 2  # after "-i" and path
                cmd[insert_at:insert_at] = seek_args
            updated.append((cmd, src, dst))
        return updated

    def _estimate_frames_per_job(self, selected_count: Optional[int] = None) -> Optional[int]:
        if not self.source_is_video:
            return 1
        frames_per_job: Optional[int] = selected_count
        if frames_per_job is None:
            video_path = None
            if self.files:
                try:
                    video_path = Path(self.files[0])
                except Exception:
                    video_path = None
            if video_path is not None:
                frames_est, fps_in = self._get_estimated_frames_info(video_path)
                fps_out = getattr(self.current_args, "fps", None)
                if frames_est is not None:
                    frames_effective = frames_est
                    if fps_in and fps_in > 0:
                        duration_est = frames_est / float(fps_in)
                        start_val = getattr(self.current_args, "start", None)
                        end_val = getattr(self.current_args, "end", None)
                        start_sec = max(0.0, float(start_val)) if start_val is not None else 0.0
                        end_limit = duration_est
                        if end_val is not None:
                            try:
                                end_limit = max(0.0, min(float(end_val), duration_est))
                            except Exception:
                                end_limit = max(0.0, duration_est)
                        trimmed = max(end_limit - start_sec, 0.0)
                        frames_effective = int(round(trimmed * float(fps_in)))
                    frames_effective = max(frames_effective, 1)
                    if fps_out and fps_in and fps_in > 0:
                        ratio = float(fps_out) / float(fps_in)
                        frames_per_job = max(1, int(round(frames_effective * ratio)))
                    else:
                        frames_per_job = frames_effective
        return frames_per_job

    def _count_output_matches(self, directory: Path, patterns: Sequence[str]) -> int:
        total = 0
        for pattern in patterns:
            try:
                total += len(list(directory.glob(pattern)))
            except Exception:
                continue
        return total

    def _output_monitor_loop(
        self,
        out_dir: Path,
        patterns: Sequence[str],
        initial_count: int,
        total_units: int,
        label: str,
        interval_sec: float = 10.0,
    ) -> None:
        if total_units <= 0:
            return
        last_pct = -1
        last_seen = -1
        while not self._output_monitor_stop.is_set():
            current = self._count_output_matches(out_dir, patterns)
            done = max(0, current - initial_count)
            if total_units > 0:
                done = min(total_units, done)
                pct = int((done * 100) / total_units)
            else:
                pct = 100
            if done != last_seen:
                if pct == 100 or last_pct < 0 or (pct - last_pct) >= cutter.PROGRESS_INTERVAL:
                    last_pct = pct
                    self.root.after(0, self._log_progress, pct, done, total_units, f"{label} (files)")
            last_seen = done
            if done >= total_units:
                break
            self._output_monitor_stop.wait(interval_sec)
        self._output_monitor_stop.set()

    def _start_output_monitor(
        self,
        jobs: Sequence[Tuple[List[str], str, str]],
        total_units: int,
        label: str,
    ) -> bool:
        if not self.source_is_video or total_units <= 0 or not jobs:
            return False
        # Prefer configured output folder; fall back to job destination parent.
        out_dir = None
        if self.out_dir:
            try:
                out_dir = Path(self.out_dir).expanduser()
            except Exception:
                out_dir = None
        if out_dir is None:
            try:
                out_dir = Path(jobs[0][2]).expanduser().parent
            except Exception:
                out_dir = None
        if out_dir is None or not out_dir.exists():
            return False
        patterns: Set[str] = set()
        for _cmd, _src, dst in jobs:
            name = Path(dst).name
            if "%07d" in name:
                name = name.replace("%07d", "*")
            patterns.add(name)
        if not patterns:
            return False
        initial_count = self._count_output_matches(out_dir, patterns)
        self._output_monitor_stop.clear()
        monitor = threading.Thread(
            target=self._output_monitor_loop,
            args=(out_dir, sorted(patterns), initial_count, total_units, label),
            daemon=True,
            name="output-monitor",
        )
        self._output_monitor_thread = monitor
        monitor.start()
        return True

    def _run_execute_jobs(self, jobs: List[Tuple[List[str], str, str]]) -> None:
        try:
            workers = self._effective_jobs()
        except Exception:
            workers = cutter.parse_jobs(getattr(self.current_args, "jobs", "auto"))
        total_jobs = len(jobs)
        if total_jobs == 0:
            self.root.after(0, self._on_execute_finished, 0, 0, 0, [])
            return
        # For video direct export, scale progress by estimated frame count (or selection count) per view.
        if self.source_is_video:
            selected = getattr(self, "_selected_frame_indices", None)
            frames_per_job = len(selected) if selected else self._estimate_frames_per_job(None)
            if frames_per_job is None or frames_per_job <= 0:
                frames_per_job = 1
            progress_units_per_job = frames_per_job
            progress_total_units = progress_units_per_job * total_jobs
            progress_label = "images"
        else:
            progress_units_per_job = 1
            progress_total_units = total_jobs
            progress_label = "images"
        monitor_active = self._start_output_monitor(jobs, progress_total_units, progress_label)
        ok_units = 0
        fail_units = 0
        errors: List[str] = []
        done_units = 0
        last_pct = -1
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(cutter.run_one, cmd): (src, dst)
                for cmd, src, dst in jobs
            }
            for future in as_completed(future_map):
                rc, err = future.result()
                done_units = min(progress_total_units, done_units + progress_units_per_job)
                if rc == 0:
                    ok_units += progress_units_per_job
                else:
                    fail_units += progress_units_per_job
                    err_text = (err or "").strip()
                    if err_text:
                        errors.append(err_text)
                if not monitor_active:
                    pct = int((done_units * 100) / progress_total_units)
                    if pct == 100 or last_pct < 0 or (pct - last_pct) >= cutter.PROGRESS_INTERVAL:
                        last_pct = pct
                        self.root.after(0, self._log_progress, pct, done_units, progress_total_units, progress_label)
        self.root.after(0, self._on_execute_finished, ok_units, fail_units, progress_total_units, errors, progress_label)

    def _log_progress(self, pct: int, done: int, total: int, label: str = "jobs") -> None:
        suffix = f" {label}" if label else ""
        self.append_log_line(f"Progress... {pct:3d}% ({done}/{total}{suffix})")

    def _on_execute_finished(
        self,
        ok: int,
        fail: int,
        total: int,
        errors: List[str],
        label: str = "jobs",
    ) -> None:
        self._output_monitor_stop.set()
        self.is_executing = False
        if self.update_button is not None:
            self.update_button.configure(state="normal")
        if self.execute_button is not None:
            self.execute_button.configure(state="normal")
        if self.preview_stop_button is not None:
            self.preview_stop_button.configure(state="disabled")
        cutter.stop_event.clear()
        units_label = label if label else "items"
        summary = f"[EXEC] Done: succeeded={ok}, failed={fail}, total={total} ({units_label})"
        self.append_log_line(summary)
        for line in errors:
            self.append_log_line(line)
        if errors:
            messagebox.showwarning("Complete", summary + "\nCheck the log for error details.")
        else:
            messagebox.showinfo("Complete", summary)


    def _ensure_window_min_dimensions(self) -> None:
        if self.root is None:
            return
        self.root.update_idletasks()
        if self.controls_frame is not None:
            controls_width = max(
                self.controls_frame.winfo_width(),
                self.controls_frame.winfo_reqwidth(),
                460,
            )
        else:
            controls_width = 460
        total_width = int(self.display_width + controls_width + 80)
        total_height = int(self.display_height + 320)
        current_w = self.root.winfo_width()
        current_h = self.root.winfo_height()
        new_w = max(current_w, total_width)
        self.root.minsize(total_width, total_height)
        if new_w != current_w or total_height > current_h:
            new_h = max(current_h, total_height)
            self.root.geometry(f"{new_w}x{new_h}")

    def _sync_panel_heights(self) -> None:
        if self.left_frame is None or self.right_inner is None:
            return
        self.root.update_idletasks()
        right_height = max(self.right_inner.winfo_height(), self.right_inner.winfo_reqheight())
        desired_inner_height = max(self.display_height, right_height)
        self.left_frame.configure(height=desired_inner_height, width=self.display_width)
        if self.canvas is not None:
            target_canvas_height = max(desired_inner_height, self.display_height)
            self.canvas.configure(height=target_canvas_height, width=self.display_width)
        if self.preview_frame is not None:
            self.preview_frame.configure(width=self.display_width + 16)
        if self.left_frame is not None:
            self.left_frame.configure(width=self.display_width)
        if self.canvas is not None:
            self.canvas.configure(width=self.display_width)
        if self.controls_frame is not None:
            frame_height = max(self.controls_frame.winfo_height(), self.controls_frame.winfo_reqheight())
            inner_height = max(self.right_inner.winfo_height(), self.right_inner.winfo_reqheight())
            self._controls_frame_padding = max(0, frame_height - inner_height)
            controls_height = desired_inner_height + self._controls_frame_padding
            self.controls_frame.configure(height=controls_height)

        if self.preview_frame is not None:
            frame_height = max(self.preview_frame.winfo_height(), self.preview_frame.winfo_reqheight())
            inner_height = max(self.left_frame.winfo_height(), self.left_frame.winfo_reqheight())
            self._preview_frame_padding = max(0, frame_height - inner_height)
            preview_height = desired_inner_height + self._preview_frame_padding
            self.preview_frame.configure(height=preview_height)
        self._reposition_canvas_items()
        self._ensure_window_min_dimensions()

    def _reposition_canvas_items(self) -> None:
        if self.canvas is None:
            return
        self.canvas.update_idletasks()
        canvas_width = max(int(self.canvas.winfo_width() or 0), self.display_width)
        canvas_height = max(int(self.canvas.winfo_height() or 0), self.display_height)
        offset_x = max(0.0, (canvas_width - self.display_width) / 2.0)
        offset_y = max(0.0, (canvas_height - self.display_height) / 2.0)
        delta_x = offset_x - self.canvas_offset_x
        delta_y = offset_y - self.canvas_offset_y
        self.canvas_offset_x = offset_x
        self.canvas_offset_y = offset_y
        if self.canvas_image_id is not None:
            self.canvas.coords(self.canvas_image_id, offset_x, offset_y)
        if delta_x or delta_y:
            self.canvas.move("overlay", delta_x, delta_y)
        if self.canvas.find_withtag("placeholder"):
            self.canvas.coords(
                "placeholder",
                offset_x + self.display_width / 2,
                offset_y + self.display_height / 2,
            )

    def set_log_text(self, text: str) -> None:
        if self.log_text is None:
            return
        self.log_text.delete("1.0", tk.END)
        self.log_text.insert("1.0", text)
        self.log_text.see("1.0")

    def append_log_line(self, line: str) -> None:
        if self.log_text is None:
            return
        current = self.log_text.get("1.0", tk.END).strip()
        prefix = "\n" if current else ""
        self.log_text.insert(tk.END, prefix + line)
        self.log_text.see(tk.END)

    @staticmethod
    def _block_text_edit(event) -> Optional[str]:
        if getattr(event, "state", 0) & 0x4 and event.keysym.lower() == "c":
            return None
        return "break"

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    args = parse_arguments()
    ensure_explicit_flags(args)
    app = PreviewApp(args)
    app.run()


if __name__ == "__main__":
    main()

 
