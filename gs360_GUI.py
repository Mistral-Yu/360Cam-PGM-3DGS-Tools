#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI preview and execution tool for gs360_360PerspCut."""

import argparse
import ctypes
import csv
import copy
import importlib
import itertools
import json
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

try:
    import cv2
except ImportError as exc:  # pragma: no cover - environment guard
    print("[ERR] OpenCV (cv2) is required: pip install opencv-python", file=sys.stderr)
    raise SystemExit(1) from exc

SCRIPT_DIR = Path(__file__).resolve().parent
CLI_TOOLS_DIR = SCRIPT_DIR / "cli_tools"
GUI_SETTINGS_PATH = CLI_TOOLS_DIR / "gs360_gui_settings.json"
if str(CLI_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(CLI_TOOLS_DIR))

cutter = importlib.import_module("gs360_360PerspCut")
pointcloud_optimizer = importlib.import_module("gs360_PlyOptimizer")
camera_pose_scene = importlib.import_module("gs360_CameraPoseScene")

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
HUMAN_TARGET_CHOICES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "animal",
]
HUMAN_MASK_EXPAND_MODE_CHOICES = ["pixels", "percent"]
HUMAN_PREVIEW_IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
}
HUMAN_PREVIEW_SIZE_CHOICES = ["320", "800", "Original", "Frame Fit"]
HUMAN_PREVIEW_DEFAULT_SIZE = "Frame Fit"
HUMAN_PREVIEW_DELAY_MS = 350
HUMAN_PREVIEW_MARGIN = 12
HUMAN_PREVIEW_MAX_IMAGES = 24

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
    "Metashape XML",
    "Metashape-Multi-Camera-System XML",
    "transforms.json",
    "COLMAP",
    "RealityScan XMP",
    "all",
]
MSXML_FORMAT_TO_CLI = {
    "Metashape XML": "metashape",
    "Metashape-Multi-Camera-System XML": "metashape-multi-camera-system",
    "transforms.json": "transforms",
    "COLMAP": "colmap",
    "RealityScan XMP": "realityscan",
    "all": "all",
}

DEFAULT_SELECTOR_CSV_NAME = "selected_image_list.csv"
SELECTOR_INPUT_MODE_LABEL_TO_CLI = {
    "Auto": "auto",
    "Single 360 Image": "single",
    "Dual Fisheye Image": "pair",
}
SELECTOR_INPUT_MODE_CLI_TO_LABEL = {
    value: key for key, value in SELECTOR_INPUT_MODE_LABEL_TO_CLI.items()
}
SELECTOR_GAP_MODE_LABEL_TO_CLI = {
    "Off": "off",
    "Single insert": "single",
    "Strict": "strict",
}
SELECTOR_GAP_MODE_CLI_TO_LABEL = {
    value: key for key, value in SELECTOR_GAP_MODE_LABEL_TO_CLI.items()
}

PLY_VIEW_CANVAS_WIDTH = 840
PLY_VIEW_CANVAS_HEIGHT = 840
PLY_VIEW_MAX_POINTS = 5_000_000
PLY_VIEW_INTERACTIVE_MAX_POINTS = 100_000
CAMERA_SCENE_INTERACTIVE_MAX_CAMERAS = 300
CAMERA_SCENE_GRID_MAX_HALF_LINES = 60
PLY_INTERACTION_SETTLE_DELAY_MS = 350
PLY_VIEW_MAX_ZOOM = 640.0
CAMERA_SCENE_SOURCE_CHOICES = (
    "COLMAP text model",
    "transforms.json + PLY",
    "RealityScan CSV + PLY",
    "RealityScan XMP",
    "Metashape XML",
)
SELECTOR_OVERVIEW_LOG_SCALE = 99.0
SELECTOR_OVERVIEW_X_ZOOM_MIN = 0.25
SELECTOR_OVERVIEW_X_ZOOM_MAX = 150.0
SELECTOR_OVERVIEW_BAR_WIDTH_MIN = 0.25
SELECTOR_OVERVIEW_BAR_WIDTH_MAX = 1024.0
SELECTOR_OVERVIEW_PRESET_VISIBLE_BARS_MAX = 50
SELECTOR_OVERVIEW_PRESET_VISIBLE_BARS_HALF = 500
SELECTOR_SUSPECT_BRIGHTNESS_BINS = 5
SELECTOR_PREVIEW_DEFAULT_OPEN_ZOOM_RATIO = 0.5
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

    APP_BG = "#f4f3f1"
    HEADER_BG = "#ebe9e6"
    SURFACE_BG = "#f4f3f1"
    TEXT_FG = "#0f172a"
    MUTED_FG = "#6b7280"
    NOTEBOOK_STYLE = "Gs360Workbench.TNotebook"
    NOTEBOOK_TAB_STYLE = "Gs360Workbench.TNotebook.Tab"
    NOTEBOOK_COLORS = {
        "bar_bg": "#ece9e5",
        "tab_idle_bg": "#ece9e5",
        "tab_hover_bg": "#f6f4f1",
        "tab_selected_bg": "#ffffff",
        "tab_idle_fg": "#4f5358",
        "tab_selected_fg": "#0f172a",
        "tab_border": "#d1ccc5",
        "tab_hover_border": "#c2bbb2",
        "tab_selected_border": "#0078d4",
    }
    UI_THEMES = {
        "Default": {
            "app_bg": "#f4f3f1",
            "header_bg": "#ebe9e6",
            "surface_bg": "#f4f3f1",
            "text_fg": "#0f172a",
            "muted_fg": "#6b7280",
            "notebook_colors": {
                "bar_bg": "#ece9e5",
                "tab_idle_bg": "#ece9e5",
                "tab_hover_bg": "#f6f4f1",
                "tab_selected_bg": "#ffffff",
                "tab_idle_fg": "#4f5358",
                "tab_selected_fg": "#0f172a",
                "tab_border": "#d1ccc5",
                "tab_hover_border": "#c2bbb2",
                "tab_selected_border": "#0078d4",
            },
        },
        "Dark Gray": {
            "app_bg": "#1e1e1e",
            "header_bg": "#252526",
            "surface_bg": "#1e1e1e",
            "text_fg": "#f3f4f6",
            "muted_fg": "#c8c8c8",
            "notebook_colors": {
                "bar_bg": "#252526",
                "tab_idle_bg": "#2d2d30",
                "tab_hover_bg": "#37373d",
                "tab_selected_bg": "#1e1e1e",
                "tab_idle_fg": "#d4d4d4",
                "tab_selected_fg": "#ffffff",
                "tab_border": "#3f3f46",
                "tab_hover_border": "#4c4c54",
                "tab_selected_border": "#007acc",
            },
        },
        "Light Gray": {
            "app_bg": "#ececec",
            "header_bg": "#e1e1e1",
            "surface_bg": "#ececec",
            "text_fg": "#111827",
            "muted_fg": "#6b7280",
            "notebook_colors": {
                "bar_bg": "#e3e3e3",
                "tab_idle_bg": "#e3e3e3",
                "tab_hover_bg": "#f3f3f3",
                "tab_selected_bg": "#fbfbfb",
                "tab_idle_fg": "#4b5563",
                "tab_selected_fg": "#111827",
                "tab_border": "#c8c8c8",
                "tab_hover_border": "#bababa",
                "tab_selected_border": "#0078d4",
            },
        },
    }
    DARK_ENTRY_BG = "#252526"
    DARK_BUTTON_BG = "#2d2d30"
    DARK_BUTTON_ACTIVE_BG = "#3e3e42"
    LIGHT_ENTRY_BG = "#ffffff"
    LIGHT_BUTTON_BG = "#f3f2f1"
    LIGHT_BUTTON_ACTIVE_BG = "#e8e6e3"
    LIGHT_DISABLED_FG = "#8a8886"
    DEFAULT_WINDOW_WIDTH_PX = 1360
    DEFAULT_WINDOW_HEIGHT_PX = 980
    DEFAULT_WINDOW_MIN_WIDTH_PX = 1120
    DEFAULT_WINDOW_MIN_HEIGHT_PX = 760

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
        self.gui_settings_path = GUI_SETTINGS_PATH
        self._config_persist_suspend = True
        self._persisted_gui_settings = self._load_gui_settings()
        saved_ui_style = self._sanitize_ui_theme_name(self._persisted_gui_settings.get("ui_style"))
        saved_ffmpeg_path = self._normalize_saved_ffmpeg_path(
            self._persisted_gui_settings.get("ffmpeg_path"),
            str(getattr(self.current_args, "ffmpeg", "ffmpeg")),
        )
        self._set_theme_palette(saved_ui_style)
        self.current_args.ffmpeg = saved_ffmpeg_path
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
        self.display_width = 760
        self.display_height = 300

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
        self._ui_theme_name_var = tk.StringVar(value=saved_ui_style)
        self.root.option_add("*Frame.Background", self.APP_BG)
        self.root.option_add("*Label.Background", self.APP_BG)
        self.root.option_add("*LabelFrame.Background", self.APP_BG)
        self.root.option_add("*Checkbutton.Background", self.APP_BG)
        self.root.option_add("*Radiobutton.Background", self.APP_BG)
        self.root.option_add("*Label.Foreground", self.TEXT_FG)
        self.root.option_add("*Checkbutton.Foreground", self.TEXT_FG)
        self.root.option_add("*Radiobutton.Foreground", self.TEXT_FG)
        self.root.configure(bg=self.APP_BG)
        self._configure_notebook_style()

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
        self.ffmpeg_path_var = tk.StringVar(value=saved_ffmpeg_path)
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
        self.preview_inspect_button: Optional[tk.Button] = None

        self.selector_vars: Dict[str, tk.Variable] = {}
        self.selector_log: Optional[tk.Text] = None
        self.selector_run_button: Optional[tk.Button] = None
        self.selector_show_score_button: Optional[tk.Button] = None
        self.selector_jump_next_suspect_button: Optional[tk.Button] = None
        self.selector_open_suspects_button: Optional[tk.Button] = None
        self.selector_manual_apply_button: Optional[tk.Button] = None
        self.selector_manual_reset_button: Optional[tk.Button] = None
        self.selector_xzoom_max_button: Optional[tk.Button] = None
        self.selector_xzoom_half_button: Optional[tk.Button] = None
        self.selector_xzoom_fit_button: Optional[tk.Button] = None
        self.selector_count_var: Optional[tk.StringVar] = None
        self.selector_summary_label: Optional[tk.Label] = None
        self.selector_score_canvas: Optional[tk.Canvas] = None
        self.selector_last_scores: List[Tuple[int, bool, Optional[float]]] = []
        self.selector_score_entries: List[Dict[str, Any]] = []
        self.selector_score_suspect_positions: Set[int] = set()
        self.selector_motion_suspect_positions: Set[int] = set()
        self.selector_score_csv_path_loaded: Optional[Path] = None
        self.selector_score_csv_fieldnames: List[str] = []
        self.selector_score_selected_key: Optional[str] = None
        self.selector_score_zoom = 1.0
        self.selector_score_bar_width = 4.0
        self.selector_score_bar_area_height = 0
        self.selector_score_total_width = 0.0
        self.selector_score_value_min = 0.0
        self.selector_score_value_range = 0.0
        self.selector_optical_flow_threshold_entry: Optional[tk.Entry] = None
        self.selector_last_suspect_jump_idx: Optional[int] = None
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
        self.selector_preview_panel_select_toggle_button: Optional[tk.Button] = None
        self.selector_preview_panel_jump_current_button: Optional[tk.Button] = None
        self.selector_preview_items: Dict[int, Dict[str, Any]] = {}
        self.selector_preview_panel_active_idx: Optional[int] = None
        self.selector_preview_panel_zoom_ratio = 1.0
        self.selector_auto_fetch_pending = False
        self.selector_csv_auto = True
        self.selector_csv_auto_value = ""
        self._selector_csv_updating = False

        self.human_vars: Dict[str, tk.Variable] = {}
        self.human_log: Optional[tk.Text] = None
        self.human_preview_button: Optional[tk.Button] = None
        self.human_run_button: Optional[tk.Button] = None
        self.human_expand_mode_combo: Optional[ttk.Combobox] = None
        self.human_expand_pixels_entry: Optional[tk.Entry] = None
        self.human_expand_percent_entry: Optional[tk.Entry] = None
        self.human_edge_fuse_check: Optional[tk.Checkbutton] = None
        self.human_edge_fuse_entry: Optional[tk.Entry] = None
        self.human_preview_expand_mode_combo: Optional[ttk.Combobox] = None
        self.human_preview_expand_scale: Optional[tk.Scale] = None
        self.human_preview_expand_pixels_entry: Optional[tk.Entry] = None
        self.human_preview_expand_percent_entry: Optional[tk.Entry] = None
        self.human_preview_edge_fuse_check: Optional[tk.Checkbutton] = None
        self.human_preview_edge_fuse_entry: Optional[tk.Entry] = None
        self.human_preview_size_combo: Optional[ttk.Combobox] = None
        self.human_preview_update_button: Optional[tk.Button] = None
        self.human_preview_reset_button: Optional[tk.Button] = None
        self.human_preview_canvas: Optional[tk.Canvas] = None
        self.human_preview_status_var = tk.StringVar(
            value="Preview will show the first image group in the selected folder."
        )
        self.human_gpu_status_var = tk.StringVar(
            value="GPU status: Select this tab to check."
        )
        self.human_gpu_fix_var = tk.StringVar(
            value="Select SegmentationMaskTool tab to check GPU status."
        )
        self.human_preview_slider_var = tk.DoubleVar(value=15.0)
        self.human_preview_size_var = tk.StringVar(
            value=HUMAN_PREVIEW_DEFAULT_SIZE
        )
        self._human_preview_photo: Optional[ImageTk.PhotoImage] = None
        self._human_gpu_status_label: Optional[tk.Label] = None
        self._human_gpu_fix_text: Optional[tk.Text] = None
        self._human_gpu_fix_after_id: Optional[str] = None
        self._human_preview_busy = False
        self._human_segmentation_module: Optional[Any] = None
        self._human_segmentation_model: Optional[Any] = None
        self._human_segmentation_device: Optional[Any] = None
        self._human_segmentation_device_type: Optional[str] = None
        self._human_preview_after_id: Optional[str] = None
        self._human_preview_pending = False
        self._human_preview_rendered_items: List[
            Tuple[str, Image.Image, int]
        ] = []
        self._human_preview_cache_items: List[
            Tuple[str, Image.Image, Optional[np.ndarray]]
        ] = []
        self._human_preview_cache_signature: Optional[Tuple[Any, ...]] = None
        self._human_preview_original_rendered_items: List[
            Tuple[str, Image.Image, int]
        ] = []
        self._human_preview_original_cache_items: List[
            Tuple[str, Image.Image, Optional[np.ndarray]]
        ] = []
        self._human_preview_original_signature: Optional[
            Tuple[Any, ...]
        ] = None
        self._human_preview_original_settings: Dict[str, Any] = {}
        self._human_preview_original_status_text = ""
        self._human_preview_group_name = ""
        self._human_preview_group_total_count = 0
        self._human_preview_hit_regions: List[
            Tuple[int, int, int, int, str]
        ] = []
        self._human_preview_marked_names: Set[str] = set()
        self._human_preview_manual_masks: Dict[str, np.ndarray] = {}
        self._human_manual_mask_temp_dir: Optional[Path] = None
        self._human_mask_editor_window: Optional[tk.Toplevel] = None
        self._human_mask_editor_canvas: Optional[tk.Canvas] = None
        self._human_mask_editor_photo: Optional[ImageTk.PhotoImage] = None
        self._human_mask_editor_name = ""
        self._human_mask_editor_image: Optional[Image.Image] = None
        self._human_mask_editor_base_mask: Optional[np.ndarray] = None
        self._human_mask_editor_mask: Optional[np.ndarray] = None
        self._human_mask_editor_initial_mask: Optional[np.ndarray] = None
        self._human_mask_editor_last_point: Optional[Tuple[int, int]] = None
        self._human_mask_editor_manual_color = "#00c8ff"
        self._human_mask_editor_color_swatch: Optional[tk.Label] = None
        self._human_mask_editor_zoom = 1.0
        self._human_mask_editor_tool_var = tk.StringVar(value="add")
        self._human_mask_editor_brush_var = tk.IntVar(value=60)
        self._human_expand_syncing = False

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

        self.dualfisheye_vars: Dict[str, tk.Variable] = {}
        self.dualfisheye_log: Optional[tk.Text] = None
        self.dualfisheye_inspect_button: Optional[tk.Button] = None
        self.dualfisheye_set_fps_var = tk.BooleanVar(value=True)
        self.dualfisheye_extract_run_button: Optional[tk.Button] = None
        self.dualfisheye_extract_stop_button: Optional[tk.Button] = None
        self.dualfisheye_calibration_run_button: Optional[tk.Button] = None
        self.dualfisheye_calibration_stop_button: Optional[tk.Button] = None
        self.dualfisheye_fisheye_output_entry: Optional[tk.Entry] = None
        self.dualfisheye_fisheye_output_button: Optional[tk.Button] = None
        self.dualfisheye_perspective_output_entry: Optional[tk.Entry] = None
        self.dualfisheye_perspective_output_button: Optional[tk.Button] = None
        self.dualfisheye_color_output_entry: Optional[tk.Entry] = None
        self.dualfisheye_color_output_button: Optional[tk.Button] = None

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
        self._ply_source_kind = "ply"
        self._ply_colmap_model = None
        self._ply_viewer_root: Optional[tk.Widget] = None
        self._ply_view_canvas: Optional[tk.Canvas] = None
        self._ply_canvas_image_id: Optional[int] = None
        self._ply_canvas_photo: Optional[ImageTk.PhotoImage] = None
        self._ply_view_info_var = tk.StringVar(value="Point cloud viewer is idle")
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
        self._ply_display_up_axis_var = tk.StringVar(value="Z-up")
        self._ply_view_interactive_max_points_var = tk.StringVar(
            value=str(PLY_VIEW_INTERACTIVE_MAX_POINTS)
        )
        self._ply_view_high_max_points_var = tk.StringVar(
            value=str(PLY_VIEW_MAX_POINTS)
        )
        self._ply_point_size_var = tk.StringVar(value="2")
        self._ply_grid_step_var = tk.StringVar(value="1.0")
        self._ply_grid_span_var = tk.StringVar(value="auto")
        self._ply_draw_points_var = tk.BooleanVar(value=True)
        self._ply_front_occlusion_var = tk.BooleanVar(value=True)
        self._ply_front_occlusion_checkbutton: Optional[tk.Checkbutton] = None
        self._ply_show_world_axes_var = tk.BooleanVar(value=True)
        self._ply_show_grid_var = tk.BooleanVar(value=True)
        self._ply_projection_combo: Optional[ttk.Combobox] = None
        self._ply_interactive_max_points_entry: Optional[tk.Entry] = None
        self._ply_high_max_points_entry: Optional[tk.Entry] = None
        self._ply_initial_view_state: Optional[Dict[str, Any]] = None
        self._ply_rendered_grid_span = 0.0
        self._ply_render_sample_step = 1
        self._ply_rendered_point_count = 0
        self._ply_sky_axis_var = tk.StringVar(value="+Z")
        self._ply_sky_scale_var = tk.StringVar(value="100")
        self._ply_sky_count_var = tk.StringVar(value="4000")
        self._ply_sky_percent_var = tk.StringVar(value="50")
        self._ply_sky_color_var = tk.StringVar(value="#87cefa")
        self._ply_sky_color_rgb_var = tk.StringVar(value="RGB(135, 206, 250)")
        self._ply_sky_color_label: Optional[tk.Label] = None
        self._ply_remove_color_var = tk.StringVar(
            value=self._ply_sky_color_var.get()
        )
        self._ply_remove_color_rgb_var = tk.StringVar(
            value=self._ply_sky_color_rgb_var.get()
        )
        self._ply_remove_color_tol_var = tk.StringVar(value="125")
        self._ply_remove_color_label: Optional[tk.Label] = None
        self._ply_sky_save_path_var = tk.StringVar()
        self._ply_sky_points: Optional[np.ndarray] = None
        self._ply_sky_colors: Optional[np.ndarray] = None
        self._ply_exp_bbox_center_x_var = tk.StringVar(value="0")
        self._ply_exp_bbox_center_y_var = tk.StringVar(value="0")
        self._ply_exp_bbox_center_z_var = tk.StringVar(value="0")
        self._ply_exp_bbox_size_x_var = tk.StringVar(value="1")
        self._ply_exp_bbox_size_y_var = tk.StringVar(value="1")
        self._ply_exp_bbox_size_z_var = tk.StringVar(value="1")
        self._ply_exp_point_count_var = tk.StringVar(value="5000")
        self._ply_exp_mode_var = tk.StringVar(value="Outside")
        self._ply_exp_outer_mult_var = tk.StringVar(value="10")
        self._ply_exp_color_mode_var = tk.StringVar(value="Edge Sample")
        self._ply_exp_color_count_var = tk.StringVar(value="4")
        self._ply_exp_bbox_active_var = tk.BooleanVar(value=False)
        self._ply_exp_edit_mode_var = tk.StringVar(value="Move")
        self._ply_exp_bbox_center = np.zeros(3, dtype=np.float32)
        self._ply_exp_bbox_size = np.ones(3, dtype=np.float32)
        self._ply_exp_bbox_rotation = np.eye(3, dtype=np.float32)
        self._ply_exp_points: Optional[np.ndarray] = None
        self._ply_exp_colors: Optional[np.ndarray] = None
        self._ply_exp_drag_kind: Optional[str] = None
        self._ply_exp_drag_axis: Optional[int] = None
        self._ply_exp_drag_last: Optional[Tuple[int, int]] = None
        self._ply_exp_drag_center_start = np.zeros(3, dtype=np.float32)
        self._ply_exp_drag_size_start = np.ones(3, dtype=np.float32)
        self._ply_exp_drag_axis_world = np.zeros(3, dtype=np.float32)
        self._ply_exp_drag_screen_dir = np.zeros(2, dtype=np.float32)
        self._ply_exp_drag_pixels_per_world = 0.0
        self._ply_view_point_ids: Optional[np.ndarray] = None
        self._ply_view_rgb_mean: Optional[Tuple[float, float, float]] = None
        self._ply_loaded_points: Optional[np.ndarray] = None
        self._ply_loaded_colors: Optional[np.ndarray] = None
        self._ply_loaded_point_ids: Optional[np.ndarray] = None
        self._ply_loaded_total_points = 0
        self._ply_loaded_sample_step = 1
        self._ply_pre_append_points: Optional[np.ndarray] = None
        self._ply_pre_append_colors: Optional[np.ndarray] = None
        self._ply_pre_append_point_ids: Optional[np.ndarray] = None
        self._ply_pre_append_total_points = 0
        self._ply_pre_append_sample_step = 1
        self._ply_pre_remove_points: Optional[np.ndarray] = None
        self._ply_pre_remove_colors: Optional[np.ndarray] = None
        self._ply_pre_remove_point_ids: Optional[np.ndarray] = None
        self._ply_pre_remove_total_points = 0
        self._ply_pre_remove_sample_step = 1
        self._ply_pre_remove_sky_points: Optional[np.ndarray] = None
        self._ply_pre_remove_sky_colors: Optional[np.ndarray] = None
        self._ply_is_interacting = False
        self._ply_interaction_after_id: Optional[str] = None
        self._ply_redraw_pending = False
        self._ply_high_max_points_auto = True
        self._ply_high_max_points_updating = False
        self._ply_last_auto_high_max_points = str(PLY_VIEW_MAX_POINTS)
        self.camera_scene_vars: Dict[str, tk.Variable] = {}
        self.camera_scene_load_button: Optional[tk.Button] = None
        self.camera_scene_clear_button: Optional[tk.Button] = None
        self._camera_scene_input_groups: Dict[str, tk.Widget] = {}
        self._camera_scene_colmap_entry: Optional[tk.Entry] = None
        self._camera_scene_transforms_entry: Optional[tk.Entry] = None
        self._camera_scene_transforms_ply_entry: Optional[tk.Entry] = None
        self._camera_scene_csv_entry: Optional[tk.Entry] = None
        self._camera_scene_csv_ply_entry: Optional[tk.Entry] = None
        self._camera_scene_projection_combo: Optional[ttk.Combobox] = None
        self._camera_scene_canvas: Optional[tk.Canvas] = None
        self._camera_scene_canvas_image_id: Optional[int] = None
        self._camera_scene_canvas_photo: Optional[ImageTk.PhotoImage] = None
        self._camera_scene_info_var = tk.StringVar(
            value="Camera format viewer is idle"
        )
        self._camera_scene_points: Optional[np.ndarray] = None
        self._camera_scene_points_centered: Optional[np.ndarray] = None
        self._camera_scene_colors: Optional[np.ndarray] = None
        self._camera_scene_camera_items: List[Dict[str, Any]] = []
        self._camera_scene_center = np.zeros(3, dtype=np.float32)
        self._camera_scene_view_center = np.zeros(3, dtype=np.float32)
        self._camera_scene_origin_centered = np.zeros(3, dtype=np.float32)
        self._camera_scene_total_points = 0
        self._camera_scene_total_cameras = 0
        self._camera_scene_sample_step = 1
        self._camera_scene_camera_sample_step = 1
        self._camera_scene_rendered_camera_count = 0
        self._camera_scene_rendered_grid_span = 0.0
        self._camera_scene_source_label = "CameraPoseScene"
        self._camera_scene_quat = np.array(
            [1.0, 0.0, 0.0, 0.0], dtype=np.float32
        )
        self._camera_scene_zoom = 1.0
        self._camera_scene_pan: Tuple[float, float] = (0.0, 0.0)
        self._camera_scene_drag_last: Optional[Tuple[int, int]] = None
        self._camera_scene_pan_last: Optional[Tuple[int, int]] = None
        self._camera_scene_depth_offset = 1.0
        self._camera_scene_max_extent = 1.0
        self._camera_scene_is_interacting = False
        self._camera_scene_interaction_after_id: Optional[str] = None
        self._camera_scene_redraw_pending = False
        self._camera_scene_projection_mode = tk.StringVar(value="Orthographic")
        self._camera_scene_point_cap_var = tk.StringVar(
            value=str(PLY_VIEW_MAX_POINTS)
        )
        self._camera_scene_interactive_point_cap_var = tk.StringVar(
            value=str(PLY_VIEW_INTERACTIVE_MAX_POINTS)
        )
        self._camera_scene_point_size_var = tk.StringVar(value="2")
        self._camera_scene_camera_scale_var = tk.StringVar(value="1")
        self._camera_scene_camera_stride_var = tk.StringVar(value="1")
        self._camera_scene_grid_step_var = tk.StringVar(value="1.0")
        self._camera_scene_grid_span_var = tk.StringVar(value="auto")
        self._camera_scene_display_up_axis_var = tk.StringVar(value="Z-up")
        self._camera_scene_show_labels_var = tk.BooleanVar(value=False)
        self._camera_scene_show_camera_axes_var = tk.BooleanVar(value=False)
        self._camera_scene_show_world_axes_var = tk.BooleanVar(value=True)
        self._camera_scene_show_grid_var = tk.BooleanVar(value=True)
        self._camera_scene_monochrome_points_var = tk.BooleanVar(value=False)
        self._camera_scene_front_occlusion_var = tk.BooleanVar(value=True)
        self._camera_scene_front_occlusion_checkbutton: Optional[tk.Checkbutton] = None
        self._camera_scene_draw_points_var = tk.BooleanVar(value=True)
        self._camera_scene_draw_cameras_var = tk.BooleanVar(value=True)
        self._camera_scene_initial_view_state: Optional[Dict[str, Any]] = None
        self._camera_scene_base_points = np.zeros((0, 3), dtype=np.float32)
        self._camera_scene_base_colors = np.zeros((0, 3), dtype=np.uint8)
        self._camera_scene_base_camera_items: List[Dict[str, Any]] = []
        self._camera_scene_base_info_text = "Camera format viewer is idle"
        self._camera_scene_base_source_label = "CameraPoseScene"
        self._camera_scene_ground_y = 0.0
        self.camera_converter_vars: Dict[str, tk.Variable] = {}
        self.camera_converter_run_button: Optional[tk.Button] = None
        self.camera_converter_stop_button: Optional[tk.Button] = None
        self.camera_converter_log: Optional[tk.Text] = None
        self.camera_scene_preview_apply_button: Optional[tk.Button] = None
        self._camera_scene_point_transform_widgets: List[tk.Widget] = []
        self._human_tab_widget: Optional[tk.Widget] = None
        self._ply_tab_widget: Optional[tk.Widget] = None
        self._camera_scene_tab_widget: Optional[tk.Widget] = None

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
        self._dualfisheye_output_auto: Dict[str, bool] = {
            "pairs_output": True,
            "fisheye_output": True,
            "perspective_output": True,
            "color_output": True,
        }
        self._dualfisheye_last_auto_output: Dict[str, str] = {}
        self._dualfisheye_output_updating: Set[str] = set()
        self._dualfisheye_prefix_auto = True
        self._dualfisheye_last_auto_prefix = "out"
        self._dualfisheye_prefix_updating = False
        self._dualfisheye_pair_input_auto = True
        self._dualfisheye_last_auto_pair_input = ""
        self._dualfisheye_pair_input_updating = False
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
        self._apply_ui_theme(saved_ui_style)
        self._config_persist_suspend = False
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

    def _sanitize_ui_theme_name(self, theme_name: Any) -> str:
        raw = str(theme_name or "").strip()
        if raw in self.UI_THEMES:
            return raw
        return "Default"

    def _normalize_saved_ffmpeg_path(self, value: Any, fallback: str) -> str:
        raw = str(value or "").strip()
        if raw:
            return raw
        fallback_text = str(fallback or "").strip()
        if fallback_text:
            return fallback_text
        return "ffmpeg"

    def _set_theme_palette(self, theme_name: str) -> None:
        theme = self.UI_THEMES.get(theme_name, self.UI_THEMES["Default"])
        self._current_ui_theme_name = self._sanitize_ui_theme_name(theme_name)
        self.APP_BG = str(theme["app_bg"])
        self.HEADER_BG = str(theme["header_bg"])
        self.SURFACE_BG = str(theme["surface_bg"])
        self.TEXT_FG = str(theme["text_fg"])
        self.MUTED_FG = str(theme["muted_fg"])
        self.NOTEBOOK_COLORS = dict(theme["notebook_colors"])

    def _load_gui_settings(self) -> Dict[str, str]:
        if not self.gui_settings_path.exists():
            return {}
        try:
            with self.gui_settings_path.open("r", encoding="utf-8-sig") as handle:
                data = json.load(handle)
        except (OSError, ValueError) as exc:
            print(f"[WARN] Failed to read GUI settings: {exc}", file=sys.stderr)
            return {}
        if not isinstance(data, dict):
            return {}
        result: Dict[str, str] = {}
        ui_style = data.get("ui_style")
        ffmpeg_path = data.get("ffmpeg_path")
        if isinstance(ui_style, str):
            result["ui_style"] = ui_style
        if isinstance(ffmpeg_path, str):
            result["ffmpeg_path"] = ffmpeg_path
        return result

    def _collect_gui_settings(self) -> Dict[str, str]:
        theme_name = self._sanitize_ui_theme_name(self._ui_theme_name_var.get().strip())
        ffmpeg_path = self.ffmpeg_path_var.get().strip() or str(getattr(self.defaults, "ffmpeg", "ffmpeg"))
        return {
            "ui_style": theme_name,
            "ffmpeg_path": ffmpeg_path,
        }

    def _save_gui_settings(self) -> None:
        if self._config_persist_suspend:
            return
        payload = self._collect_gui_settings()
        try:
            self.gui_settings_path.parent.mkdir(parents=True, exist_ok=True)
            with self.gui_settings_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
        except OSError as exc:
            print(f"[WARN] Failed to write GUI settings: {exc}", file=sys.stderr)

    def _configure_notebook_style(self) -> None:
        """Apply a higher-contrast tab style that remains readable with long labels."""
        style = ttk.Style(self.root)
        try:
            if style.theme_use() != "clam":
                # Native Windows themes often ignore custom tab colors.
                style.theme_use("clam")
        except tk.TclError:
            pass

        colors = self.NOTEBOOK_COLORS
        style.configure(
            self.NOTEBOOK_STYLE,
            background=colors["bar_bg"],
            borderwidth=0,
            tabmargins=(10, 8, 10, 0),
        )
        style.configure(
            self.NOTEBOOK_TAB_STYLE,
            background=colors["tab_idle_bg"],
            foreground=colors["tab_idle_fg"],
            borderwidth=1,
            relief="flat",
            padding=(12, 8, 12, 7),
            focuscolor=colors["bar_bg"],
        )
        style.map(
            self.NOTEBOOK_TAB_STYLE,
            background=[
                ("selected", colors["tab_selected_bg"]),
                ("active", colors["tab_hover_bg"]),
                ("!disabled", colors["tab_idle_bg"]),
            ],
            foreground=[
                ("selected", colors["tab_selected_fg"]),
                ("active", colors["tab_selected_fg"]),
                ("!disabled", colors["tab_idle_fg"]),
            ],
            bordercolor=[
                ("selected", colors["tab_selected_border"]),
                ("active", colors["tab_hover_border"]),
                ("!disabled", colors["tab_border"]),
            ],
            lightcolor=[
                ("selected", colors["tab_selected_border"]),
                ("active", colors["tab_hover_border"]),
                ("!disabled", colors["tab_border"]),
            ],
            darkcolor=[
                ("selected", colors["tab_selected_border"]),
                ("active", colors["tab_hover_border"]),
                ("!disabled", colors["tab_border"]),
            ],
        )

    def _get_window_scale_factor(self) -> float:
        if self.root is None:
            return 1.0
        try:
            pixels_per_inch = float(self.root.winfo_fpixels("1i"))
        except Exception:
            return 1.0
        if pixels_per_inch <= 0.0:
            return 1.0
        return max(1.0, pixels_per_inch / 96.0)

    def _physical_px_to_logical(self, width_px: int, height_px: int) -> Tuple[int, int]:
        scale = self._get_window_scale_factor()
        logical_w = max(1, int(round(float(width_px) / scale)))
        logical_h = max(1, int(round(float(height_px) / scale)))
        return logical_w, logical_h


    def build_ui(self) -> None:
        self.notebook = ttk.Notebook(self.root, style=self.NOTEBOOK_STYLE)
        self.notebook.pack(fill="both", expand=True)
        self.notebook.bind("<<NotebookTabChanged>>", self._on_notebook_tab_changed, add="+")
        ttk.Separator(self.root, orient="horizontal").pack(fill="x")

        tab_bg = self.APP_BG
        video_tab = tk.Frame(self.notebook, bg=tab_bg)
        selector_tab = tk.Frame(self.notebook, bg=tab_bg)
        preview_tab = tk.Frame(self.notebook, bg=tab_bg)
        human_tab = tk.Frame(self.notebook, bg=tab_bg)
        ply_tab = tk.Frame(self.notebook, bg=tab_bg)
        msxml_tab = tk.Frame(self.notebook, bg=tab_bg)
        dualfisheye_tab = tk.Frame(self.notebook, bg=tab_bg)
        camera_scene_tab = tk.Frame(self.notebook, bg=tab_bg)
        config_tab = tk.Frame(self.notebook, bg=tab_bg)
        self._human_tab_widget = human_tab
        self._ply_tab_widget = ply_tab
        self._camera_scene_tab_widget = camera_scene_tab

        self.notebook.add(video_tab, text="Video2Frames")
        self.notebook.add(selector_tab, text="FrameSelector")
        self.notebook.add(preview_tab, text="360PerspCut")
        self.notebook.add(human_tab, text="SegmentationMaskTool")
        self.notebook.add(ply_tab, text="PointCloudOptimizer")
        self.notebook.add(msxml_tab, text="MS360xmlToPerspCams")
        self.notebook.add(
            dualfisheye_tab,
            text="DualFisheyePipeline (Experimental)",
        )
        self.notebook.add(
            camera_scene_tab,
            text="Camera Optimization (Experimental)",
        )
        self.notebook.add(config_tab, text="Config")

        self._build_video_tab(video_tab)
        self._build_frame_selector_tab(selector_tab)
        self._build_preview_tab(preview_tab)
        self._build_human_mask_tab(human_tab)
        self._build_ply_tab(ply_tab)
        self._build_msxml_tab(msxml_tab)
        self._build_dualfisheye_tab(dualfisheye_tab)
        self._build_camera_scene_tab(camera_scene_tab)
        self._build_config_tab(config_tab)

        self.notebook.select(preview_tab)
        self.root.after_idle(self._refresh_tab_activity_state)

    def _is_notebook_tab_selected(self, tab_widget: Optional[tk.Widget]) -> bool:
        """Return True when the given notebook tab is currently selected."""
        if self.notebook is None or tab_widget is None:
            return False
        try:
            selected = self.notebook.select()
            if not selected:
                return False
            return str(self.notebook.nametowidget(selected)) == str(tab_widget)
        except Exception:
            return False

    def _on_notebook_tab_changed(self, _event=None) -> None:
        """Update background activity when the visible tab changes."""
        self._refresh_tab_activity_state()

    def _refresh_tab_activity_state(self) -> None:
        """Enable background refresh only for currently visible tabs."""
        if self._is_notebook_tab_selected(self._human_tab_widget):
            self._update_human_gpu_status_ui()
        else:
            self._cancel_human_gpu_status_refresh()

        if self._is_notebook_tab_selected(self._ply_tab_widget):
            if self._ply_redraw_pending:
                self._redraw_ply_canvas(force=True)
        else:
            self._ply_redraw_pending = True

        if self._is_notebook_tab_selected(self._camera_scene_tab_widget):
            if self._camera_scene_redraw_pending:
                self._redraw_camera_scene_canvas(force=True)
        else:
            self._camera_scene_redraw_pending = True

    def _cancel_human_gpu_status_refresh(self) -> None:
        """Stop any pending GPU status refresh timer."""
        if self._human_gpu_fix_after_id is not None:
            try:
                self.root.after_cancel(self._human_gpu_fix_after_id)
            except Exception:
                pass
            self._human_gpu_fix_after_id = None

    def _is_human_tab_active(self) -> bool:
        """Return True while SegmentationMaskTool tab is selected."""
        return self._is_notebook_tab_selected(self._human_tab_widget)

    def _is_ply_tab_active(self) -> bool:
        """Return True while PointCloudOptimizer tab is selected."""
        return self._is_notebook_tab_selected(self._ply_tab_widget)

    def _is_camera_scene_tab_active(self) -> bool:
        """Return True while CameraFormatConverter tab is selected."""
        return self._is_notebook_tab_selected(self._camera_scene_tab_widget)

    def _create_tab_shell(
        self,
        parent: tk.Widget,
        title: str,
        subtitle: str,
    ) -> tk.Frame:
        container = tk.Frame(parent, bg=self.APP_BG)
        container.pack(fill="both", expand=True)
        header = tk.Frame(container, bg=self.HEADER_BG, bd=1, relief=tk.FLAT)
        header.pack(fill="x", padx=6, pady=(6, 6))
        tk.Label(
            header,
            text=title,
            bg=self.HEADER_BG,
            fg=self.TEXT_FG,
            font=("TkDefaultFont", 12, "bold"),
        ).pack(side=tk.LEFT, padx=(8, 10), pady=8)
        tk.Label(
            header,
            text=subtitle,
            bg=self.HEADER_BG,
            fg=self.MUTED_FG,
        ).pack(side=tk.LEFT, padx=(0, 10), pady=8)
        content = tk.Frame(container, bg=self.APP_BG)
        content.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        return content

    def _apply_standard_tab_palette(self) -> None:
        if self.root is None:
            return
        legacy_surface_colors = {
            "#eef3f8",
            "#f8fafc",
            "#edf2f7",
            "#e2e8f0",
            "#d9e3ee",
        }
        app_bg_colors = set(legacy_surface_colors)
        header_bg_colors = set()
        text_fg_colors = {"#0f172a", "#111827"}
        muted_fg_colors = {"#6b7280", "#475569", "#64748b", "#666666"}
        for theme in self.UI_THEMES.values():
            app_bg_colors.add(str(theme["app_bg"]).lower())
            app_bg_colors.add(str(theme["surface_bg"]).lower())
            header_bg_colors.add(str(theme["header_bg"]).lower())
            text_fg_colors.add(str(theme["text_fg"]).lower())
            muted_fg_colors.add(str(theme["muted_fg"]).lower())
        preserve_colors = {
            "#101010",
            "#202020",
            "#ffffff",
            "white",
        }
        default_like = {
            "systembuttonface",
            "systemwindow",
            "systemwindowframe",
            "system3dface",
        }
        is_dark_theme = getattr(self, "_current_ui_theme_name", "Default") == "Dark Gray"

        def _normalize_color(value: Any) -> str:
            try:
                return str(value).strip().lower()
            except Exception:
                return ""

        def _walk(widget: tk.Widget, inherited_bg: str) -> None:
            try:
                widget_class = widget.winfo_class()
            except Exception:
                return
            current_bg = ""
            can_set_bg = False
            try:
                current_bg = _normalize_color(widget.cget("bg"))
                can_set_bg = True
            except Exception:
                pass
            if current_bg in header_bg_colors:
                target_bg = self.HEADER_BG
            elif widget_class in {"Frame", "Labelframe", "LabelFrame", "Panedwindow"}:
                target_bg = self.APP_BG
            elif widget_class in {"Label", "Checkbutton", "Radiobutton"}:
                target_bg = inherited_bg
            else:
                target_bg = inherited_bg
            if can_set_bg and current_bg not in preserve_colors:
                if current_bg in default_like or current_bg.startswith("sys") or current_bg == "":
                    try:
                        widget.configure(bg=target_bg)
                    except Exception:
                        pass
                elif current_bg in app_bg_colors or current_bg in header_bg_colors:
                    try:
                        widget.configure(bg=target_bg)
                    except Exception:
                        pass
                elif widget_class in {"Label", "Checkbutton", "Radiobutton"} and (
                    current_bg in app_bg_colors or current_bg in header_bg_colors
                ):
                    try:
                        widget.configure(bg=target_bg)
                    except Exception:
                        pass
            if widget_class in {"Labelframe", "LabelFrame"}:
                try:
                    widget.configure(fg=self.TEXT_FG)
                except Exception:
                    pass
            try:
                current_fg = _normalize_color(widget.cget("fg"))
            except Exception:
                current_fg = ""
            if current_fg in text_fg_colors:
                try:
                    widget.configure(fg=self.TEXT_FG)
                except Exception:
                    pass
            elif current_fg in muted_fg_colors:
                try:
                    widget.configure(fg=self.MUTED_FG)
                except Exception:
                    pass
            if is_dark_theme:
                if widget_class == "Button":
                    try:
                        widget.configure(
                            bg=self.DARK_BUTTON_BG,
                            fg=self.TEXT_FG,
                            activebackground=self.DARK_BUTTON_ACTIVE_BG,
                            activeforeground=self.TEXT_FG,
                            disabledforeground=self.MUTED_FG,
                            highlightbackground=self.APP_BG,
                        )
                    except Exception:
                        pass
                elif widget_class == "Entry":
                    try:
                        widget.configure(
                            bg=self.DARK_ENTRY_BG,
                            fg=self.TEXT_FG,
                            insertbackground=self.TEXT_FG,
                            disabledforeground=self.MUTED_FG,
                            disabledbackground=self.DARK_ENTRY_BG,
                            readonlybackground=self.DARK_ENTRY_BG,
                        )
                    except Exception:
                        pass
                elif widget_class in {"Checkbutton", "Radiobutton"}:
                    try:
                        widget.configure(
                            activebackground=self.APP_BG,
                            activeforeground=self.TEXT_FG,
                            disabledforeground=self.MUTED_FG,
                            highlightbackground=self.APP_BG,
                            highlightcolor=self.APP_BG,
                            selectcolor=self.DARK_BUTTON_ACTIVE_BG,
                        )
                    except Exception:
                        pass
                elif widget_class == "Text":
                    try:
                        widget.configure(
                            bg=self.DARK_ENTRY_BG,
                            fg=self.TEXT_FG,
                            insertbackground=self.TEXT_FG,
                        )
                    except Exception:
                        pass
            else:
                if widget_class == "Button":
                    try:
                        widget.configure(
                            bg=self.LIGHT_BUTTON_BG,
                            fg=self.TEXT_FG,
                            activebackground=self.LIGHT_BUTTON_ACTIVE_BG,
                            activeforeground=self.TEXT_FG,
                            disabledforeground=self.LIGHT_DISABLED_FG,
                            highlightbackground=self.APP_BG,
                        )
                    except Exception:
                        pass
                elif widget_class == "Entry":
                    try:
                        widget.configure(
                            bg=self.LIGHT_ENTRY_BG,
                            fg=self.TEXT_FG,
                            insertbackground=self.TEXT_FG,
                            disabledforeground=self.MUTED_FG,
                            disabledbackground=self.LIGHT_ENTRY_BG,
                            readonlybackground=self.LIGHT_ENTRY_BG,
                        )
                    except Exception:
                        pass
                elif widget_class in {"Checkbutton", "Radiobutton"}:
                    try:
                        widget.configure(
                            activebackground=self.APP_BG,
                            activeforeground=self.TEXT_FG,
                            disabledforeground=self.LIGHT_DISABLED_FG,
                            highlightbackground=self.APP_BG,
                            highlightcolor=self.APP_BG,
                            selectcolor=self.LIGHT_ENTRY_BG,
                        )
                    except Exception:
                        pass
                elif widget_class == "Text":
                    try:
                        widget.configure(
                            bg=self.LIGHT_ENTRY_BG,
                            fg=self.TEXT_FG,
                            insertbackground=self.TEXT_FG,
                        )
                    except Exception:
                        pass
            child_bg = target_bg
            try:
                child_bg = str(widget.cget("bg"))
            except Exception:
                pass
            for child in widget.winfo_children():
                _walk(child, child_bg)

        _walk(self.root, self.APP_BG)

    def _apply_ui_theme(self, theme_name: str) -> None:
        theme_name = self._sanitize_ui_theme_name(theme_name)
        self._set_theme_palette(theme_name)
        if hasattr(self, "_ui_theme_name_var"):
            self._ui_theme_name_var.set(theme_name)
        if self.root is None:
            return
        self.root.option_add("*Frame.Background", self.APP_BG)
        self.root.option_add("*Label.Background", self.APP_BG)
        self.root.option_add("*LabelFrame.Background", self.APP_BG)
        self.root.option_add("*Checkbutton.Background", self.APP_BG)
        self.root.option_add("*Radiobutton.Background", self.APP_BG)
        self.root.option_add("*Label.Foreground", self.TEXT_FG)
        self.root.option_add("*Checkbutton.Foreground", self.TEXT_FG)
        self.root.option_add("*Radiobutton.Foreground", self.TEXT_FG)
        self.root.configure(bg=self.APP_BG)
        self._configure_notebook_style()
        self._apply_standard_tab_palette()

    def _on_ui_theme_selected(self, *_args: Any) -> None:
        self._apply_ui_theme(self._ui_theme_name_var.get().strip() or "Default")

    def _save_config_settings(self) -> None:
        value = self.ffmpeg_path_var.get().strip()
        self.current_args.ffmpeg = value or str(getattr(self.defaults, "ffmpeg", "ffmpeg"))
        self._save_gui_settings()
        messagebox.showinfo(
            "Config Saved",
            f"Saved Config settings to:\n{self.gui_settings_path}",
        )

    def _build_video_tab(self, parent: tk.Widget) -> None:
        container = self._create_tab_shell(
            parent,
            "Video2Frames",
            "Extract frames from video clips with ffmpeg controls and output presets.",
        )

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
            "experimental_dualfisheye": tk.BooleanVar(value=False),
        }

        self.video_vars["video"].trace_add("write", self._on_video_input_changed)
        self.video_vars["output"].trace_add("write", self._on_video_output_changed)
        self.video_vars["fps"].trace_add("write", self._on_video_fps_changed)
        self.video_vars["prefix"].trace_add("write", self._on_video_prefix_changed)
        self.video_vars["experimental_dualfisheye"].trace_add(
            "write", self._on_video_experimental_dualfisheye_changed
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
        dualfisheye_frame = tk.Frame(params)
        dualfisheye_frame.grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(0, 4)
        )
        tk.Checkbutton(
            dualfisheye_frame,
            text="Expermental DualFisheye",
            variable=self.video_vars["experimental_dualfisheye"],
        ).pack(side=tk.LEFT, padx=(0, 12))

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

    def _on_video_experimental_dualfisheye_changed(self, *_args) -> None:
        if not self.video_vars:
            return
        if bool(self.video_vars["experimental_dualfisheye"].get()):
            self.video_vars["keep_rec709"].set(False)

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

    def _on_dualfisheye_video_changed(self, *_args) -> None:
        self._update_dualfisheye_default_paths(force=False)

    def _on_dualfisheye_pair_input_changed(self, *_args) -> None:
        if self._dualfisheye_pair_input_updating:
            return
        self._dualfisheye_pair_input_auto = False
        self._update_dualfisheye_outputs_from_pair_input(force=False)

    def _on_dualfisheye_prefix_changed(self, *_args) -> None:
        if self._dualfisheye_prefix_updating:
            return
        self._dualfisheye_prefix_auto = False

    def _on_dualfisheye_output_changed(self, key: str, *_args) -> None:
        if key in self._dualfisheye_output_updating:
            return
        self._dualfisheye_output_auto[key] = False
        if key == "pairs_output":
            self._sync_dualfisheye_pair_input_from_extract()
            return
        if key == "perspective_output":
            self._update_dualfisheye_derived_output_paths()

    def _on_dualfisheye_output_toggle_changed(self, *_args) -> None:
        self._update_dualfisheye_output_controls_state()
        self._update_dualfisheye_derived_output_paths()

    def _compute_dualfisheye_metashape_f_text(self) -> str:
        if not self.dualfisheye_vars:
            return "Metashape f: -"
        try:
            output_size = int(
                self.dualfisheye_vars["perspective_size"].get().strip()
            )
            focal_mm = float(
                self.dualfisheye_vars["perspective_focal_mm"].get().strip()
            )
        except (KeyError, ValueError):
            return "Metashape f: -"
        if output_size <= 0 or focal_mm <= 0.0:
            return "Metashape f: -"
        pixel_size_mm = 36.0 / float(output_size)
        if pixel_size_mm <= 0.0:
            return "Metashape f: -"
        focal_px = focal_mm / pixel_size_mm
        return "Metashape f: {:.5f}px".format(focal_px)

    def _compute_dualfisheye_xml_output_path(self) -> str:
        root_path = self._compute_dualfisheye_perspective_root_path()
        if root_path is None:
            return ""
        return str(root_path / "perspective_cams.xml")

    def _compute_dualfisheye_perspective_root_path(self) -> Optional[Path]:
        if not self.dualfisheye_vars:
            return None
        root_text = self.dualfisheye_vars["perspective_output"].get().strip()
        if root_text:
            return Path(root_text).expanduser()
        if not bool(self.dualfisheye_vars["metadata_only"].get()):
            return None
        extrinsics_text = self.dualfisheye_vars[
            "camera_extrinsics_xml"
        ].get().strip()
        if not extrinsics_text:
            return None
        extrinsics_path = Path(extrinsics_text).expanduser()
        if not extrinsics_path.is_absolute():
            extrinsics_path = (self.cli_tools_dir.parent / extrinsics_text).resolve()
        else:
            extrinsics_path = extrinsics_path.resolve()
        return extrinsics_path.with_name(
            extrinsics_path.stem + "_perspective_colmap"
        )

    def _compute_dualfisheye_colmap_images_path(self) -> str:
        root_path = self._compute_dualfisheye_perspective_root_path()
        if root_path is None:
            return ""
        return str(root_path / "Images")

    def _compute_dualfisheye_colmap_masks_path(self) -> str:
        root_path = self._compute_dualfisheye_perspective_root_path()
        if root_path is None:
            return ""
        return str(root_path / "Masks")

    def _compute_dualfisheye_colmap_sparse_path(self) -> str:
        root_path = self._compute_dualfisheye_perspective_root_path()
        if root_path is None:
            return ""
        return str(root_path / "Sparse" / "0")

    def _update_dualfisheye_metashape_f_display(self, *_args) -> None:
        if hasattr(self, "dualfisheye_metashape_f_var"):
            self.dualfisheye_metashape_f_var.set(
                self._compute_dualfisheye_metashape_f_text()
            )
        self._update_dualfisheye_derived_output_paths()

    def _update_dualfisheye_derived_output_paths(self, *_args) -> None:
        if hasattr(self, "dualfisheye_xml_output_var"):
            self.dualfisheye_xml_output_var.set(
                self._compute_dualfisheye_xml_output_path()
            )
        if hasattr(self, "dualfisheye_colmap_images_var"):
            self.dualfisheye_colmap_images_var.set(
                self._compute_dualfisheye_colmap_images_path()
            )
        if hasattr(self, "dualfisheye_colmap_masks_var"):
            self.dualfisheye_colmap_masks_var.set(
                self._compute_dualfisheye_colmap_masks_path()
            )
        if hasattr(self, "dualfisheye_colmap_sparse_var"):
            self.dualfisheye_colmap_sparse_var.set(
                self._compute_dualfisheye_colmap_sparse_path()
            )

    def _set_dualfisheye_path_auto(self, key: str, value: str) -> None:
        var = self.dualfisheye_vars.get(key)
        if var is None:
            return
        self._dualfisheye_output_auto[key] = True
        self._dualfisheye_last_auto_output[key] = value
        self._dualfisheye_output_updating.add(key)
        try:
            var.set(value)
        finally:
            self._dualfisheye_output_updating.discard(key)

    def _update_dualfisheye_default_paths(self, force: bool = False) -> None:
        if not self.dualfisheye_vars:
            return
        video_var = self.dualfisheye_vars.get("video")
        if video_var is None:
            return
        video_text = video_var.get().strip()
        self._update_dualfisheye_inspect_state()
        if not video_text:
            return
        path = Path(video_text).expanduser()
        try:
            if not path.is_absolute():
                path = path.resolve()
        except FileNotFoundError:
            return
        if not path.suffix:
            return

        parent = path.parent if path.parent != Path("") else Path.cwd()
        base_name = re.sub(r"\s+", "_", path.stem) if path.stem else "dualfisheye"
        if not base_name:
            base_name = "dualfisheye"

        prefix_var = self.dualfisheye_vars.get("prefix")
        if prefix_var is not None:
            current_prefix = prefix_var.get().strip()
            should_update_prefix = (
                force
                or self._dualfisheye_prefix_auto
                or not current_prefix
                or current_prefix == self._dualfisheye_last_auto_prefix
            )
            if should_update_prefix:
                self._dualfisheye_prefix_auto = True
                self._dualfisheye_last_auto_prefix = base_name
                self._dualfisheye_prefix_updating = True
                try:
                    prefix_var.set(base_name)
                finally:
                    self._dualfisheye_prefix_updating = False

        pairs_output = str(parent / f"{base_name}_dualfisheye_pairs")
        fisheye_output = f"{pairs_output}_undistorted"
        perspective_output = str(parent / f"{base_name}_perspective_colmap")
        color_output = f"{fisheye_output}_colorcorrected"
        defaults = {
            "pairs_output": pairs_output,
            "fisheye_output": fisheye_output,
            "perspective_output": perspective_output,
            "color_output": color_output,
        }
        for key, default_value in defaults.items():
            current_var = self.dualfisheye_vars.get(key)
            if current_var is None:
                continue
            current_value = current_var.get().strip()
            auto_enabled = self._dualfisheye_output_auto.get(key, True)
            last_auto = self._dualfisheye_last_auto_output.get(key, "")
            should_update = (
                force
                or auto_enabled
                or not current_value
                or current_value == last_auto
            )
            if should_update:
                self._set_dualfisheye_path_auto(key, default_value)
        self._sync_dualfisheye_pair_input_from_extract(force=force)
        self._update_dualfisheye_derived_output_paths()

    def _sync_dualfisheye_pair_input_from_extract(
        self, force: bool = False
    ) -> None:
        if not self.dualfisheye_vars:
            return
        source_var = self.dualfisheye_vars.get("pairs_output")
        pair_var = self.dualfisheye_vars.get("pair_input")
        if source_var is None or pair_var is None:
            return
        source_value = source_var.get().strip()
        current_value = pair_var.get().strip()
        should_update = (
            force
            or self._dualfisheye_pair_input_auto
            or not current_value
            or current_value == self._dualfisheye_last_auto_pair_input
        )
        if not should_update:
            return
        self._dualfisheye_pair_input_auto = True
        self._dualfisheye_last_auto_pair_input = source_value
        self._dualfisheye_pair_input_updating = True
        try:
            pair_var.set(source_value)
        finally:
            self._dualfisheye_pair_input_updating = False
        self._update_dualfisheye_outputs_from_pair_input(force=force)

    def _update_dualfisheye_outputs_from_pair_input(
        self, force: bool = False
    ) -> None:
        if not self.dualfisheye_vars:
            return
        pair_var = self.dualfisheye_vars.get("pair_input")
        if pair_var is None:
            return
        pair_text = pair_var.get().strip()
        if not pair_text:
            return
        pair_path = Path(pair_text).expanduser()
        try:
            if not pair_path.is_absolute():
                pair_path = pair_path.resolve()
        except FileNotFoundError:
            return
        base_dir = pair_path.parent if pair_path.parent != Path("") else Path.cwd()
        base_name = pair_path.name
        defaults = {
            "fisheye_output": str(base_dir / f"{base_name}_undistorted"),
            "perspective_output": str(base_dir / f"{base_name}_perspective_colmap"),
            "color_output": str(base_dir / f"{base_name}_colorcorrected"),
        }
        for key, default_value in defaults.items():
            current_var = self.dualfisheye_vars.get(key)
            if current_var is None:
                continue
            current_value = current_var.get().strip()
            auto_enabled = self._dualfisheye_output_auto.get(key, True)
            last_auto = self._dualfisheye_last_auto_output.get(key, "")
            should_update = (
                force
                or auto_enabled
                or not current_value
                or current_value == last_auto
            )
            if should_update:
                self._set_dualfisheye_path_auto(key, default_value)
        self._update_dualfisheye_derived_output_paths()
        self._update_dualfisheye_output_controls_state()

    def _update_dualfisheye_output_controls_state(self) -> None:
        if not self.dualfisheye_vars:
            return

        control_specs = [
            (
                self.dualfisheye_color_output_entry,
                self.dualfisheye_color_output_button,
                (
                    bool(self.dualfisheye_vars["save_color_corrected_output"].get())
                    and not bool(self.dualfisheye_vars["metadata_only"].get())
                ),
            ),
            (
                self.dualfisheye_fisheye_output_entry,
                self.dualfisheye_fisheye_output_button,
                (
                    bool(self.dualfisheye_vars["save_fisheye_output"].get())
                    and not bool(self.dualfisheye_vars["metadata_only"].get())
                ),
            ),
            (
                self.dualfisheye_perspective_output_entry,
                self.dualfisheye_perspective_output_button,
                (
                    bool(self.dualfisheye_vars["metadata_only"].get())
                    or not bool(self.dualfisheye_vars["no_perspective"].get())
                ),
            ),
        ]
        for entry, button, enabled in control_specs:
            if entry is not None:
                entry.configure(state="normal" if enabled else "readonly")
            if button is not None:
                button.configure(state="normal" if enabled else "disabled")

    def _update_dualfisheye_inspect_state(self) -> None:
        button = self.dualfisheye_inspect_button
        if button is None:
            return
        video_var = self.dualfisheye_vars.get("video")
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

    def _inspect_dualfisheye_video_metadata(self) -> None:
        if not self.dualfisheye_vars:
            return
        video_value = self.dualfisheye_vars.get("video")
        if video_value is None:
            return
        video_path_text = video_value.get().strip()
        if not video_path_text:
            messagebox.showerror(
                "Video metadata", "Select an input raw video first."
            )
            return
        try:
            video_path = Path(video_path_text).expanduser().resolve(strict=True)
        except FileNotFoundError:
            messagebox.showerror(
                "Video metadata",
                f"Video file not found:\n{video_path_text}",
            )
            return
        except Exception as exc:
            messagebox.showerror(
                "Video metadata",
                f"Failed to resolve video path:\n{exc}",
            )
            return

        result = self._collect_video_metadata_lines(video_path, "Video metadata")
        if not result:
            return
        lines, metadata = result
        self._set_text_widget(self.dualfisheye_log, "")
        for line in lines:
            self._append_text_widget(self.dualfisheye_log, line)

        fps_value = metadata.get("frame_rate") if metadata else None
        fps_text_raw = metadata.get("frame_rate_text") if metadata else None
        detected = None
        if fps_value is not None:
            try:
                detected = float(fps_value)
            except (TypeError, ValueError):
                detected = None
        if detected is None and fps_text_raw:
            detected = self._parse_fps_from_stream(str(fps_text_raw))
        if detected is None or detected <= 0.0:
            return

        formatted = self._format_fps_for_output(f"{detected}")
        if not formatted:
            return
        should_apply = bool(self.dualfisheye_set_fps_var.get())
        if should_apply:
            self.dualfisheye_vars["fps"].set(formatted)
            self._append_text_widget(
                self.dualfisheye_log,
                f"[fps] Updated FPS to detected value ({formatted})",
            )
            return
        self._append_text_widget(
            self.dualfisheye_log,
            (
                "[fps] Detected FPS {} (Set FPS unchecked; "
                "entry left unchanged)"
            ).format(formatted),
        )

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
        container = self._create_tab_shell(
            parent,
            "FrameSelector",
            "Score image sequences and select representative frames for downstream processing.",
        )

        params = tk.LabelFrame(container, text="Parameters")
        params.pack(fill="x", padx=8, pady=8)

        self.selector_vars = {
            "in_dir": tk.StringVar(),
            "input_mode": tk.StringVar(
                value=SELECTOR_INPUT_MODE_CLI_TO_LABEL["auto"]
            ),
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
            "augment_gap_mode": tk.StringVar(
                value=SELECTOR_GAP_MODE_CLI_TO_LABEL["single"]
            ),
            "augment_lowlight": tk.BooleanVar(value=False),
            "compute_optical_flow": tk.BooleanVar(value=False),
            "optical_flow_low_threshold": tk.StringVar(value="3"),
            "augment_motion": tk.BooleanVar(value=False),
            "segment_boundary_reopt": tk.BooleanVar(value=True),
            "ignore_highlights": tk.BooleanVar(value=True),
        }

        self.selector_vars["csv_mode"].trace_add("write", self._on_selector_csv_mode_changed)
        self.selector_vars["in_dir"].trace_add("write", self._on_selector_in_dir_changed)
        self.selector_vars["csv_path"].trace_add("write", self._on_selector_csv_path_changed)
        self.selector_vars["compute_optical_flow"].trace_add(
            "write",
            self._update_selector_optical_flow_state,
        )

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
        tk.Label(segment_frame, text="Input mode").pack(side=tk.LEFT, padx=(0, 4))
        selector_input_mode_combo = ttk.Combobox(
            segment_frame,
            textvariable=self.selector_vars["input_mode"],
            values=tuple(SELECTOR_INPUT_MODE_LABEL_TO_CLI.keys()),
            state="readonly",
            width=16,
        )
        selector_input_mode_combo.pack(side=tk.LEFT, padx=(0, 12))
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
        spacing_frame = tk.Frame(params)
        spacing_frame.grid(row=row, column=0, columnspan=3, sticky="w", pady=4)
        tk.Label(spacing_frame, text="Min spacing").pack(side=tk.LEFT, padx=(0, 4))
        tk.Entry(
            spacing_frame,
            textvariable=self.selector_vars["min_spacing_frames"],
            width=10,
        ).pack(side=tk.LEFT, padx=(0, 12))
        tk.Label(spacing_frame, text="Augment gap").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Combobox(
            spacing_frame,
            textvariable=self.selector_vars["augment_gap_mode"],
            values=tuple(SELECTOR_GAP_MODE_LABEL_TO_CLI.keys()),
            state="readonly",
            width=14,
        ).pack(side=tk.LEFT, padx=(0, 12))
        tk.Checkbutton(
            spacing_frame,
            text="Segment Top-K + Boundary Local Reopt",
            variable=self.selector_vars["segment_boundary_reopt"],
        ).pack(side=tk.LEFT, padx=(0, 12))

        row += 1
        augment_frame = tk.Frame(params)
        augment_frame.grid(row=row, column=0, columnspan=3, sticky="w", pady=4)
        tk.Checkbutton(
            augment_frame,
            text="Compute optical flow",
            variable=self.selector_vars["compute_optical_flow"],
        ).pack(side=tk.LEFT, padx=(0, 12))
        tk.Label(augment_frame, text="Low motion span <=").pack(side=tk.LEFT, padx=(0, 4))
        self.selector_optical_flow_threshold_entry = tk.Entry(
            augment_frame,
            textvariable=self.selector_vars["optical_flow_low_threshold"],
            width=6,
        )
        self.selector_optical_flow_threshold_entry.pack(side=tk.LEFT, padx=(0, 12))
        tk.Checkbutton(
            augment_frame,
            text="Augment motion (Experimental)",
            variable=self.selector_vars["augment_motion"],
        ).pack(side=tk.LEFT, padx=(0, 12))
        tk.Checkbutton(
            augment_frame,
            text="Augment lowlight (Experimental)",
            variable=self.selector_vars["augment_lowlight"],
        ).pack(side=tk.LEFT, padx=(0, 12))

        row += 1
        final_frame = tk.Frame(params)
        final_frame.grid(row=row, column=0, columnspan=3, sticky="we", pady=4)
        tk.Label(final_frame, text="Score crop ratio (single pano only; pair mode uses circular center mask)").pack(side=tk.LEFT, padx=(0, 4))
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
        self.selector_jump_next_suspect_button = tk.Button(
            overview_controls,
            text="Jump to Next Suspect",
            command=self._jump_to_next_selector_suspect,
        )
        self.selector_jump_next_suspect_button.pack(side=tk.LEFT, padx=(0, 8))
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
        self.selector_xzoom_max_button = tk.Button(
            overview_controls,
            text="X Zoom 50",
            command=self._selector_overview_zoom_max,
        )
        self.selector_xzoom_max_button.pack(side=tk.LEFT, padx=(4, 4))
        self.selector_xzoom_half_button = tk.Button(
            overview_controls,
            text="X Zoom 500",
            command=self._selector_overview_zoom_half,
        )
        self.selector_xzoom_half_button.pack(side=tk.LEFT, padx=(0, 4))
        self.selector_xzoom_fit_button = tk.Button(
            overview_controls,
            text="X Zoom Fit",
            command=self._selector_overview_zoom_min,
        )
        self.selector_xzoom_fit_button.pack(side=tk.LEFT, padx=(0, 4))
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
            height=121,
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
            text="Legend: sharpness suspect=red outline, motion suspect=gold outline, selected=teal, unselected=gray, preview set=blue outline, preview current=deep blue outline, manual edit=orange outline, left-click=toggle, right-click=preview toggle, wheel=zoom X",
            anchor="w",
        ).pack(fill="x", padx=6, pady=(0, 4))
        self._update_selector_optical_flow_state()

        log_frame = tk.LabelFrame(container, text="Log")
        log_frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.selector_log = tk.Text(log_frame, wrap="word", height=18, cursor="arrow")
        self.selector_log.pack(fill="both", expand=True, padx=6, pady=4)
        self.selector_log.bind("<Key>", self._block_text_edit)
        self.selector_log.bind("<Button-1>", lambda event: self.selector_log.focus_set())
        self._set_text_widget(self.selector_log, "")

        self._on_selector_csv_mode_changed()

    def _build_human_mask_tab(self, parent: tk.Widget) -> None:
        container = self._create_tab_shell(
            parent,
            "SegmentationMaskTool",
            "Generate and refine human or object masks with GPU-aware segmentation options.",
        )

        params = tk.LabelFrame(container, text="Parameters")
        params.pack(fill="x", padx=8, pady=8)

        default_cpu_workers = str(max(1, os.cpu_count() or 1))

        self.human_vars = {
            "input": tk.StringVar(),
            "output": tk.StringVar(),
            "mode": tk.StringVar(value="mask"),
            "expand_mode": tk.StringVar(value="pixels"),
            "expand_pixels": tk.StringVar(value="15"),
            "expand_percent": tk.StringVar(value="1"),
            "edge_fuse_enabled": tk.BooleanVar(value=True),
            "edge_fuse_pixels": tk.StringVar(value="25"),
            "cpu": tk.BooleanVar(value=False),
            "cpu_workers": tk.StringVar(value=default_cpu_workers),
            "include_shadow": tk.BooleanVar(value=False),
            "target_person": tk.BooleanVar(value=True),
            "target_bicycle": tk.BooleanVar(value=False),
            "target_car": tk.BooleanVar(value=False),
            "target_motorcycle": tk.BooleanVar(value=False),
            "target_bus": tk.BooleanVar(value=False),
            "target_truck": tk.BooleanVar(value=False),
            "target_animal": tk.BooleanVar(value=False),
            "custom_targets": tk.StringVar(),
        }
        self.human_vars["input"].trace_add("write", self._on_human_input_changed)
        self.human_vars["output"].trace_add("write", self._on_human_output_changed)
        self.human_vars["expand_mode"].trace_add(
            "write", self._update_human_expand_state
        )
        self.human_vars["expand_pixels"].trace_add(
            "write", self._on_human_expand_value_changed
        )
        self.human_vars["expand_percent"].trace_add(
            "write", self._on_human_expand_value_changed
        )
        self.human_vars["edge_fuse_pixels"].trace_add(
            "write", self._on_human_expand_value_changed
        )
        self.human_vars["edge_fuse_enabled"].trace_add(
            "write", self._on_human_expand_value_changed
        )
        self.human_vars["cpu"].trace_add(
            "write", lambda *_args: self._update_human_gpu_status_ui()
        )

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
        self.human_expand_mode_combo = ttk.Combobox(
            mode_frame,
            textvariable=self.human_vars["expand_mode"],
            values=HUMAN_MASK_EXPAND_MODE_CHOICES,
            state="readonly",
            width=10,
        )
        tk.Label(mode_frame, text="Mask Expand").pack(side=tk.LEFT, padx=(12, 4))
        self.human_expand_mode_combo.pack(side=tk.LEFT, padx=(0, 12))
        self.human_expand_mode_combo.set(self.human_vars["expand_mode"].get())
        tk.Label(mode_frame, text="Pixels").pack(side=tk.LEFT, padx=(0, 4))
        self.human_expand_pixels_entry = tk.Entry(
            mode_frame,
            textvariable=self.human_vars["expand_pixels"],
            width=8,
        )
        self.human_expand_pixels_entry.pack(side=tk.LEFT, padx=(0, 12))
        tk.Label(mode_frame, text="Percent").pack(side=tk.LEFT, padx=(0, 4))
        self.human_expand_percent_entry = tk.Entry(
            mode_frame,
            textvariable=self.human_vars["expand_percent"],
            width=8,
        )
        self.human_expand_percent_entry.pack(side=tk.LEFT, padx=(0, 4))
        tk.Label(mode_frame, text="Edge Fuse").pack(side=tk.LEFT, padx=(12, 4))
        self.human_edge_fuse_check = tk.Checkbutton(
            mode_frame,
            text="",
            variable=self.human_vars["edge_fuse_enabled"],
        )
        self.human_edge_fuse_check.pack(side=tk.LEFT, padx=(0, 4))
        self.human_edge_fuse_entry = tk.Entry(
            mode_frame,
            textvariable=self.human_vars["edge_fuse_pixels"],
            width=8,
        )
        self.human_edge_fuse_entry.pack(side=tk.LEFT, padx=(0, 4))
        tk.Label(mode_frame, text="Pixels").pack(side=tk.LEFT, padx=(0, 4))
        tk.Checkbutton(
            mode_frame,
            text="Include shadow",
            variable=self.human_vars["include_shadow"],
        ).pack(side=tk.LEFT, padx=(12, 0))

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
            text="Motorcycle",
            variable=self.human_vars["target_motorcycle"],
        ).pack(side=tk.LEFT, padx=(0, 8))
        tk.Checkbutton(
            target_frame,
            text="Bus",
            variable=self.human_vars["target_bus"],
        ).pack(side=tk.LEFT, padx=(0, 8))
        tk.Checkbutton(
            target_frame,
            text="Truck",
            variable=self.human_vars["target_truck"],
        ).pack(side=tk.LEFT, padx=(0, 8))
        tk.Checkbutton(
            target_frame,
            text="Animal (Bird/Cat/Dog)",
            variable=self.human_vars["target_animal"],
        ).pack(side=tk.LEFT, padx=(0, 8))

        row += 1
        custom_target_frame = tk.Frame(params)
        custom_target_frame.grid(row=row, column=0, columnspan=3, sticky="we", pady=4)
        tk.Label(custom_target_frame, text="Custom Targets(comma separated)").pack(
            side=tk.LEFT, padx=(0, 8)
        )
        tk.Entry(
            custom_target_frame,
            textvariable=self.human_vars["custom_targets"],
            width=72,
        ).pack(side=tk.LEFT, fill="x", expand=True, padx=(0, 4))

        row += 1
        flags_frame = tk.Frame(params)
        flags_frame.grid(row=row, column=0, columnspan=3, sticky="w", pady=4)
        tk.Checkbutton(flags_frame, text="Force CPU", variable=self.human_vars["cpu"]).pack(side=tk.LEFT, padx=(0, 12))
        tk.Label(flags_frame, text="CPU workers (all modes)").pack(side=tk.LEFT, padx=(0, 4))
        tk.Entry(
            flags_frame,
            textvariable=self.human_vars["cpu_workers"],
            width=8,
        ).pack(side=tk.LEFT, padx=(0, 4))
        tk.Label(flags_frame, text="GPU status").pack(side=tk.LEFT, padx=(12, 4))
        self._human_gpu_status_label = tk.Label(
            flags_frame,
            textvariable=self.human_gpu_status_var,
            anchor="w",
            justify="left",
        )
        self._human_gpu_status_label.pack(side=tk.LEFT, padx=(0, 4))

        row += 1
        gpu_fix_frame = tk.Frame(params)
        gpu_fix_frame.grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 4))
        tk.Label(gpu_fix_frame, text="How to fix").pack(
            side=tk.LEFT,
            padx=(0, 8),
        )
        self._human_gpu_fix_text = tk.Text(
            gpu_fix_frame,
            width=118,
            height=1,
            wrap="none",
        )
        self._human_gpu_fix_text.pack(side=tk.LEFT, fill="x", expand=True, padx=(0, 4))
        self._human_gpu_fix_text.bind("<Key>", lambda _event: "break")
        self._human_gpu_fix_text.bind("<Control-a>", self._select_all_human_gpu_fix_text)

        for col in range(3):
            params.grid_columnconfigure(col, weight=1 if col == 1 else 0)

        self._update_human_gpu_status_ui()

        actions = tk.Frame(container)
        actions.pack(fill="x", padx=8, pady=(0, 8))
        self.human_preview_button = tk.Button(
            actions,
            text="Preview masks",
            command=self._preview_human_masks,
        )
        self.human_preview_button.pack(side=tk.LEFT, padx=4, pady=4)
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

        body_pane = ttk.PanedWindow(container, orient=tk.VERTICAL)
        body_pane.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        preview_frame = tk.LabelFrame(body_pane, text="Preview")
        tk.Label(
            preview_frame,
            textvariable=self.human_preview_status_var,
            anchor="w",
            justify="left",
        ).pack(fill="x", padx=6, pady=(4, 2))
        preview_controls = tk.Frame(preview_frame)
        preview_controls.pack(fill="x", padx=6, pady=(0, 6))
        tk.Label(preview_controls, text="Mask Expand").pack(
            side=tk.LEFT, padx=(0, 4)
        )
        self.human_preview_expand_mode_combo = ttk.Combobox(
            preview_controls,
            textvariable=self.human_vars["expand_mode"],
            values=HUMAN_MASK_EXPAND_MODE_CHOICES,
            state="readonly",
            width=10,
        )
        self.human_preview_expand_mode_combo.pack(
            side=tk.LEFT, padx=(0, 8)
        )
        self.human_preview_expand_scale = tk.Scale(
            preview_controls,
            orient="horizontal",
            from_=0,
            to=200,
            resolution=1,
            showvalue=False,
            variable=self.human_preview_slider_var,
            command=self._on_human_preview_scale_changed,
            length=195,
        )
        self.human_preview_expand_scale.pack(
            side=tk.LEFT, padx=(0, 8), fill="x", expand=False
        )
        tk.Label(preview_controls, text="Pixels").pack(
            side=tk.LEFT, padx=(0, 4)
        )
        self.human_preview_expand_pixels_entry = tk.Entry(
            preview_controls,
            textvariable=self.human_vars["expand_pixels"],
            width=8,
        )
        self.human_preview_expand_pixels_entry.pack(
            side=tk.LEFT, padx=(0, 8)
        )
        tk.Label(preview_controls, text="Percent").pack(
            side=tk.LEFT, padx=(0, 4)
        )
        self.human_preview_expand_percent_entry = tk.Entry(
            preview_controls,
            textvariable=self.human_vars["expand_percent"],
            width=8,
        )
        self.human_preview_expand_percent_entry.pack(
            side=tk.LEFT, padx=(0, 8)
        )
        tk.Label(preview_controls, text="Edge Fuse").pack(
            side=tk.LEFT, padx=(0, 4)
        )
        self.human_preview_edge_fuse_check = tk.Checkbutton(
            preview_controls,
            text="",
            variable=self.human_vars["edge_fuse_enabled"],
        )
        self.human_preview_edge_fuse_check.pack(
            side=tk.LEFT, padx=(0, 2)
        )
        self.human_preview_edge_fuse_entry = tk.Entry(
            preview_controls,
            textvariable=self.human_vars["edge_fuse_pixels"],
            width=8,
        )
        self.human_preview_edge_fuse_entry.pack(
            side=tk.LEFT, padx=(0, 8)
        )
        tk.Label(preview_controls, text="Pixels").pack(
            side=tk.LEFT, padx=(0, 8)
        )
        tk.Label(preview_controls, text="Preview Size").pack(
            side=tk.LEFT, padx=(12, 4)
        )
        self.human_preview_size_combo = ttk.Combobox(
            preview_controls,
            textvariable=self.human_preview_size_var,
            values=HUMAN_PREVIEW_SIZE_CHOICES,
            state="readonly",
            width=10,
        )
        self.human_preview_size_combo.pack(side=tk.LEFT, padx=(0, 4))
        self.human_preview_size_combo.bind(
            "<<ComboboxSelected>>",
            self._on_human_preview_size_changed,
        )
        self.human_preview_update_button = tk.Button(
            preview_controls,
            text="Update hidden",
            command=self._apply_human_preview_marked_removal,
            state="disabled",
        )
        self.human_preview_update_button.pack(side=tk.LEFT, padx=(12, 4))
        self.human_preview_reset_button = tk.Button(
            preview_controls,
            text="Reset preview",
            command=self._reset_human_preview_state,
            state="disabled",
        )
        self.human_preview_reset_button.pack(side=tk.LEFT, padx=(0, 4))
        tk.Label(
            preview_frame,
            text=(
                "Preview controls: Left-click=mark hide, Update hidden=hide marked, "
                "Reset preview=restore, Right-click=Mask Editor, Mouse wheel=scroll."
            ),
            anchor="w",
            justify="left",
        ).pack(fill="x", padx=6, pady=(0, 6))
        preview_wrap = tk.Frame(preview_frame)
        preview_wrap.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        preview_x = tk.Scrollbar(preview_wrap, orient="horizontal")
        preview_x.pack(side=tk.BOTTOM, fill="x")
        preview_y = tk.Scrollbar(preview_wrap, orient="vertical")
        preview_y.pack(side=tk.RIGHT, fill="y")
        self.human_preview_canvas = tk.Canvas(
            preview_wrap,
            height=380,
            background="#f4f4f4",
            xscrollcommand=preview_x.set,
            yscrollcommand=preview_y.set,
            highlightthickness=1,
            highlightbackground="#d0d0d0",
        )
        self.human_preview_canvas.pack(side=tk.LEFT, fill="both", expand=True)
        preview_x.configure(command=self.human_preview_canvas.xview)
        preview_y.configure(command=self.human_preview_canvas.yview)
        self.human_preview_canvas.bind(
            "<Configure>",
            self._on_human_preview_canvas_configure,
        )
        self.human_preview_canvas.bind(
            "<Button-1>",
            self._on_human_preview_canvas_click,
        )
        self.human_preview_canvas.bind(
            "<Button-3>",
            self._on_human_preview_canvas_right_click,
        )
        self.human_preview_canvas.bind(
            "<MouseWheel>",
            self._on_human_preview_mousewheel,
        )
        self.human_preview_canvas.bind(
            "<Button-4>",
            self._on_human_preview_mousewheel,
        )
        self.human_preview_canvas.bind(
            "<Button-5>",
            self._on_human_preview_mousewheel,
        )

        log_frame = tk.LabelFrame(body_pane, text="Log")
        self.human_log = tk.Text(log_frame, wrap="word", height=9, cursor="arrow")
        self.human_log.pack(fill="both", expand=True, padx=6, pady=4)
        self.human_log.bind("<Key>", self._block_text_edit)
        self.human_log.bind("<Button-1>", lambda event: self.human_log.focus_set())
        self._set_text_widget(self.human_log, "")
        body_pane.add(preview_frame, weight=5)
        body_pane.add(log_frame, weight=1)
        self._update_human_expand_state()
        self._render_human_preview_image(None)

    def _on_human_input_changed(self, *_args) -> None:
        self._update_human_default_output()

    def _on_human_output_changed(self, *_args) -> None:
        if self._human_output_updating:
            return
        self._human_output_auto = False

    def _update_human_expand_state(self, *_args) -> None:
        if not self.human_vars:
            return
        mode = self.human_vars["expand_mode"].get().strip().lower()
        manual_locked = bool(self._human_preview_manual_masks)
        pixels_state = (
            "normal" if mode == "pixels" and not manual_locked else "disabled"
        )
        percent_state = (
            "normal" if mode == "percent" and not manual_locked else "disabled"
        )
        combo_state = "readonly" if not manual_locked else "disabled"
        for combo in (
            self.human_expand_mode_combo,
            self.human_preview_expand_mode_combo,
        ):
            if combo is None:
                continue
            try:
                combo.configure(state=combo_state)
            except tk.TclError:
                pass
        for widget, state in (
            (self.human_expand_pixels_entry, pixels_state),
            (self.human_expand_percent_entry, percent_state),
            (self.human_preview_expand_pixels_entry, pixels_state),
            (self.human_preview_expand_percent_entry, percent_state),
        ):
            if widget is None:
                continue
            try:
                widget.configure(state=state)
            except tk.TclError:
                pass
        for widget in (
            self.human_edge_fuse_entry,
            self.human_preview_edge_fuse_entry,
        ):
            if widget is None:
                continue
            try:
                widget.configure(
                    state=(
                        "normal" if bool(
                            self.human_vars["edge_fuse_enabled"].get()
                        ) else "disabled"
                    )
                )
            except tk.TclError:
                pass
        scale = self.human_preview_expand_scale
        if scale is not None:
            if mode == "pixels":
                scale.configure(from_=0, to=150, resolution=1)
            else:
                scale.configure(from_=0, to=10, resolution=0.5)
            scale.configure(state="normal" if not manual_locked else "disabled")
        self._sync_human_preview_slider_from_fields()
        if self._human_preview_rendered_items:
            self._schedule_human_preview_refresh()

    def _on_human_expand_value_changed(self, *_args) -> None:
        if self._human_expand_syncing:
            return
        self._sync_human_preview_slider_from_fields()
        if self._human_preview_cache_items and not self._human_preview_busy:
            self._schedule_human_preview_refresh()

    def _sync_human_preview_slider_from_fields(self) -> None:
        if not self.human_vars:
            return
        mode = self.human_vars["expand_mode"].get().strip().lower()
        try:
            if mode == "percent":
                value = float(self.human_vars["expand_percent"].get().strip())
            else:
                value = float(self.human_vars["expand_pixels"].get().strip())
        except ValueError:
            return
        if value < 0:
            value = 0.0
        scale = self.human_preview_expand_scale
        if scale is not None:
            try:
                upper = float(scale.cget("to"))
            except Exception:
                upper = value
            value = min(value, upper)
        if abs(self.human_preview_slider_var.get() - value) > 1e-6:
            self.human_preview_slider_var.set(value)
        self._update_human_preview_slider_label()

    def _format_human_expand_value(self, value: float, mode: str) -> str:
        if mode == "percent":
            rounded = round(float(value), 1)
            if abs(rounded - int(round(rounded))) < 1e-6:
                return str(int(round(rounded)))
            return "{:.1f}".format(rounded)
        return str(int(round(float(value))))

    def _update_human_preview_slider_label(self) -> None:
        return

    def _update_human_preview_update_button_state(self) -> None:
        if self.human_preview_update_button is None:
            update_state = "disabled"
        else:
            update_state = (
                "normal" if self._human_preview_marked_names else "disabled"
            )
            self.human_preview_update_button.configure(state=update_state)
        if self.human_preview_reset_button is not None:
            reset_state = (
                "normal"
                if self._human_preview_original_cache_items else
                "disabled"
            )
            self.human_preview_reset_button.configure(state=reset_state)

    def _capture_human_preview_reset_settings(self) -> Dict[str, Any]:
        snapshot: Dict[str, Any] = {}
        for key in (
            "mode",
            "expand_mode",
            "expand_pixels",
            "expand_percent",
            "edge_fuse_enabled",
            "edge_fuse_pixels",
            "cpu",
            "include_shadow",
            "target_person",
            "target_bicycle",
            "target_car",
            "target_animal",
        ):
            if key not in self.human_vars:
                continue
            snapshot[key] = self.human_vars[key].get()
        return snapshot

    def _restore_human_preview_reset_settings(
        self,
        snapshot: Dict[str, Any],
    ) -> None:
        if not snapshot:
            return
        self._human_expand_syncing = True
        try:
            for key, value in snapshot.items():
                if key in self.human_vars:
                    self.human_vars[key].set(value)
        finally:
            self._human_expand_syncing = False
        self._update_human_expand_state()
        self._sync_human_preview_slider_from_fields()

    def _apply_human_preview_slider_value(self) -> None:
        if not self.human_vars:
            return
        mode = self.human_vars["expand_mode"].get().strip().lower()
        value = float(self.human_preview_slider_var.get())
        formatted = self._format_human_expand_value(value, mode)
        self._human_expand_syncing = True
        try:
            if mode == "percent":
                if self.human_vars["expand_percent"].get().strip() != formatted:
                    self.human_vars["expand_percent"].set(formatted)
            else:
                if self.human_vars["expand_pixels"].get().strip() != formatted:
                    self.human_vars["expand_pixels"].set(formatted)
        finally:
            self._human_expand_syncing = False
        self._update_human_preview_slider_label()

    def _schedule_human_preview_refresh(self, delay_ms: int = HUMAN_PREVIEW_DELAY_MS) -> None:
        if self.root is None:
            return
        if self._human_preview_after_id is not None:
            try:
                self.root.after_cancel(self._human_preview_after_id)
            except Exception:
                pass
            self._human_preview_after_id = None
        self._human_preview_after_id = self.root.after(
            max(0, int(delay_ms)),
            self._trigger_human_preview_refresh,
        )

    def _trigger_human_preview_refresh(self) -> None:
        self._human_preview_after_id = None
        self._apply_human_preview_slider_value()
        if self._human_preview_busy:
            self._human_preview_pending = True
            return
        self._rebuild_human_preview_from_cache()

    def _on_human_preview_scale_changed(self, _value: str) -> None:
        self._update_human_preview_slider_label()
        self._schedule_human_preview_refresh()

    def _on_human_preview_size_changed(self, _event=None) -> None:
        if self._human_preview_rendered_items:
            self._render_human_preview_sheet()

    def _on_human_preview_canvas_configure(self, _event=None) -> None:
        if self.human_preview_size_var.get().strip().lower() == "frame fit":
            if self._human_preview_rendered_items:
                self._render_human_preview_sheet()

    def _on_human_preview_canvas_click(self, event) -> str:
        canvas = self.human_preview_canvas
        if canvas is None or not self._human_preview_hit_regions:
            return "break"
        x_pos = int(canvas.canvasx(event.x))
        y_pos = int(canvas.canvasy(event.y))
        for left, top, right, bottom, name in self._human_preview_hit_regions:
            if left <= x_pos <= right and top <= y_pos <= bottom:
                if name in self._human_preview_marked_names:
                    self._human_preview_marked_names.remove(name)
                else:
                    self._human_preview_marked_names.add(name)
                self._update_human_preview_update_button_state()
                self._render_human_preview_sheet()
                break
        return "break"

    def _on_human_preview_canvas_right_click(self, event) -> str:
        canvas = self.human_preview_canvas
        if canvas is None or not self._human_preview_hit_regions:
            return "break"
        x_pos = int(canvas.canvasx(event.x))
        y_pos = int(canvas.canvasy(event.y))
        item_name = self._human_preview_item_name_at(x_pos, y_pos)
        if item_name:
            self._open_human_mask_editor(item_name)
        return "break"

    def _on_human_preview_mousewheel(self, event) -> str:
        canvas = self.human_preview_canvas
        if canvas is None:
            return "break"
        delta = 0
        if getattr(event, "num", None) == 4:
            delta = -1
        elif getattr(event, "num", None) == 5:
            delta = 1
        elif getattr(event, "delta", 0):
            delta = -1 if event.delta > 0 else 1
        if delta != 0:
            canvas.yview_scroll(delta, "units")
        return "break"

    def _collect_human_mask_settings(
        self,
        show_errors: bool = True,
    ) -> Optional[Dict[str, Any]]:
        if not self.human_vars:
            return None

        input_dir = self.human_vars["input"].get().strip()
        if not input_dir:
            if show_errors:
                messagebox.showerror(
                    "gs360_SegmentationMaskTool",
                    "Input folder is required.",
                )
            return None
        input_path = Path(input_dir).expanduser()
        if not input_path.exists() or not input_path.is_dir():
            if show_errors:
                messagebox.showerror(
                    "gs360_SegmentationMaskTool",
                    "Input folder not found:\n{}".format(input_dir),
                )
            return None

        mode_value = self.human_vars["mode"].get().strip()
        if mode_value not in HUMAN_MODE_CHOICES:
            if show_errors:
                messagebox.showerror(
                    "gs360_SegmentationMaskTool",
                    "Mode must be one of: {}.".format(
                        ", ".join(HUMAN_MODE_CHOICES)
                    ),
                )
            return None

        selected_targets: List[str] = []
        for target_name in HUMAN_TARGET_CHOICES:
            key = "target_{}".format(target_name)
            if bool(self.human_vars[key].get()):
                selected_targets.append(target_name)
        custom_targets_text = self.human_vars["custom_targets"].get().strip()
        raw_targets = list(selected_targets)
        if custom_targets_text:
            raw_targets.append(custom_targets_text)
        if not raw_targets:
            if show_errors:
                messagebox.showerror(
                    "gs360_SegmentationMaskTool",
                    (
                        "Select at least one target or enter a custom target "
                        "name."
                    ),
                )
            return None
        try:
            module = importlib.import_module("gs360_SegmentationMaskTool")
            resolved_targets, invalid_targets = module.resolve_target_names(
                raw_targets
            )
            available_targets = module.list_available_target_names()
        except Exception:
            resolved_targets = raw_targets
            invalid_targets = []
            available_targets = []
        if invalid_targets:
            if show_errors:
                messagebox.showerror(
                    "gs360_SegmentationMaskTool",
                    "Unsupported target name(s): {}\n\nAvailable targets:\n{}".format(
                        ", ".join(invalid_targets),
                        ", ".join(available_targets),
                    ),
                )
            return None

        expand_mode = self.human_vars["expand_mode"].get().strip().lower()
        if expand_mode not in HUMAN_MASK_EXPAND_MODE_CHOICES:
            if show_errors:
                messagebox.showerror(
                    "gs360_SegmentationMaskTool",
                    "Mask expand mode must be 'pixels' or 'percent'.",
                )
            return None

        expand_pixels = 0
        expand_percent = 0.0
        edge_fuse_pixels = 0
        edge_fuse_enabled = bool(self.human_vars["edge_fuse_enabled"].get())
        if expand_mode == "pixels":
            pixels_text = self.human_vars["expand_pixels"].get().strip()
            try:
                expand_pixels = int(pixels_text)
            except ValueError:
                if show_errors:
                    messagebox.showerror(
                        "gs360_SegmentationMaskTool",
                        "Mask expand pixels must be an integer.",
                    )
                return None
            if expand_pixels < 0:
                if show_errors:
                    messagebox.showerror(
                        "gs360_SegmentationMaskTool",
                        "Mask expand pixels must be 0 or greater.",
                    )
                return None
        else:
            percent_text = self.human_vars["expand_percent"].get().strip()
            try:
                expand_percent = float(percent_text)
            except ValueError:
                if show_errors:
                    messagebox.showerror(
                        "gs360_SegmentationMaskTool",
                        "Mask expand percent must be a number.",
                    )
                return None
            if expand_percent < 0:
                if show_errors:
                    messagebox.showerror(
                        "gs360_SegmentationMaskTool",
                        "Mask expand percent must be 0 or greater.",
                    )
                return None
        edge_fuse_text = self.human_vars["edge_fuse_pixels"].get().strip()
        try:
            edge_fuse_pixels = int(edge_fuse_text)
        except ValueError:
            if show_errors:
                messagebox.showerror(
                    "gs360_SegmentationMaskTool",
                    "Edge fuse pixels must be an integer.",
                )
            return None
        if edge_fuse_pixels < 0:
            if show_errors:
                messagebox.showerror(
                    "gs360_SegmentationMaskTool",
                    "Edge fuse pixels must be 0 or greater.",
                )
            return None
        if not edge_fuse_enabled:
            edge_fuse_pixels = 0

        cpu_workers_text = self.human_vars["cpu_workers"].get().strip()
        try:
            cpu_workers = int(cpu_workers_text)
        except ValueError:
            if show_errors:
                messagebox.showerror(
                    "gs360_SegmentationMaskTool",
                    "CPU workers (all modes) must be an integer.",
                )
            return None
        if cpu_workers <= 0:
            if show_errors:
                messagebox.showerror(
                    "gs360_SegmentationMaskTool",
                    "CPU workers (all modes) must be 1 or greater.",
                )
            return None

        return {
            "input_path": input_path,
            "output_dir": self.human_vars["output"].get().strip(),
            "mode": mode_value,
            "targets": resolved_targets,
            "cpu": bool(self.human_vars["cpu"].get()),
            "cpu_workers": cpu_workers,
            "include_shadow": bool(self.human_vars["include_shadow"].get()),
            "expand_mode": expand_mode,
            "expand_pixels": expand_pixels,
            "expand_percent": expand_percent,
            "edge_fuse_enabled": edge_fuse_enabled,
            "edge_fuse_pixels": edge_fuse_pixels,
        }

    def _human_preview_group_key(self, stem: str) -> str:
        view_id = self._extract_multicam_view_id(stem)
        if not view_id:
            return stem
        suffix = "_{}".format(view_id)
        if len(stem) > len(suffix) and stem.upper().endswith(suffix.upper()):
            return stem[:-len(suffix)]
        return stem

    def _collect_human_preview_group(
        self, input_path: Path
    ) -> Tuple[str, List[Path]]:
        image_paths = sorted(
            path for path in input_path.iterdir()
            if path.is_file() and path.suffix.lower() in HUMAN_PREVIEW_IMAGE_EXTS
        )
        if not image_paths:
            raise ValueError("No supported images found in:\n{}".format(input_path))
        first_key = self._human_preview_group_key(image_paths[0].stem)
        grouped = [
            path for path in image_paths
            if self._human_preview_group_key(path.stem) == first_key
        ]
        return first_key, grouped

    def _human_preview_signature(
        self,
        settings: Dict[str, Any],
    ) -> Tuple[Any, ...]:
        return (
            str(settings["input_path"]),
            tuple(settings["targets"]),
            bool(settings["cpu"]),
            bool(settings["include_shadow"]),
        )

    def _should_confirm_human_preview_group(
        self,
        group_paths: Sequence[Path],
    ) -> Optional[str]:
        if not group_paths:
            return None
        try:
            with Image.open(str(group_paths[0])) as img:
                width, height = img.size
        except Exception:
            return None
        ratio = (float(width) / float(height)) if height else 0.0
        reasons: List[str] = []
        if len(group_paths) == 1:
            reasons.append(
                "Input looks like a single image rather than a multi-image set."
            )
        if height > 0 and width >= 2048 and height >= 1024:
            if 1.95 <= ratio <= 2.05:
                reasons.append(
                    "The first image looks like a high-resolution 2:1 panorama."
                )
        if not reasons:
            return None
        return "\n".join(reasons)

    @staticmethod
    def _human_expand_label(settings: Dict[str, Any]) -> str:
        if settings["expand_mode"] == "pixels":
            return "{} px".format(settings["expand_pixels"])
        return "{} %".format(settings["expand_percent"])

    @staticmethod
    def _human_edge_fuse_label(settings: Dict[str, Any]) -> str:
        if not settings.get("edge_fuse_enabled", False):
            return "off"
        return "{} px".format(settings["edge_fuse_pixels"])

    @staticmethod
    def _format_human_vram_text(
        free_bytes: Optional[int],
        total_bytes: Optional[int],
    ) -> str:
        """Format used/total VRAM in decimal GB."""
        if free_bytes is None or total_bytes is None or total_bytes <= 0:
            return ""
        used_gb = float(max(0, int(total_bytes) - int(free_bytes))) / 1e9
        total_gb = float(total_bytes) / 1e9
        return " | VRAM used/total: {:.1f}/{:.1f} GB".format(
            used_gb,
            total_gb,
        )

    @staticmethod
    def _build_human_gpu_fix_command(
        torch_version: str,
        torchvision_version: str,
    ) -> str:
        """Build a copyable pip install command for the current environment."""
        torch_pkg = "torch=={}".format(torch_version)
        tv_pkg = "torchvision=={}".format(torchvision_version)
        return (
            "pip install {} {} -f "
            "https://download.pytorch.org/whl/torch_stable.html"
        ).format(torch_pkg, tv_pkg)

    def _get_human_gpu_status_info(self) -> Dict[str, str]:
        """Build a short GPU availability message for tool users."""
        force_cpu = False
        if self.human_vars:
            try:
                force_cpu = bool(self.human_vars["cpu"].get())
            except Exception:
                force_cpu = False

        try:
            module = importlib.import_module("gs360_SegmentationMaskTool")
        except Exception:
            return {
                "status": "GPU status: Unavailable",
                "fix": "Open this GUI in a CUDA-enabled PyTorch environment.",
                "color": "#b00020",
            }

        torch_version = str(getattr(module.torch, "__version__", "torch"))
        try:
            torchvision_module = importlib.import_module("torchvision")
            torchvision_version = str(
                getattr(torchvision_module, "__version__", "torchvision")
            )
        except Exception:
            torchvision_version = "torchvision"
        fix_command = self._build_human_gpu_fix_command(
            torch_version,
            torchvision_version,
        )

        try:
            cuda_available = bool(module.torch.cuda.is_available())
        except Exception:
            cuda_available = False

        gpu_name = ""
        free_bytes = None
        total_bytes = None
        if cuda_available:
            try:
                gpu_name = str(module.torch.cuda.get_device_name(0))
            except Exception:
                gpu_name = "CUDA GPU"
            try:
                free_bytes, total_bytes = module.torch.cuda.mem_get_info()
            except Exception:
                try:
                    props = module.torch.cuda.get_device_properties(0)
                    total_bytes = int(getattr(props, "total_memory", 0))
                except Exception:
                    total_bytes = None
                free_bytes = None
        vram_text = self._format_human_vram_text(free_bytes, total_bytes)

        if cuda_available and force_cpu:
            return {
                "status": "GPU status: Available ({}){}".format(
                    gpu_name,
                    vram_text,
                ) + " | Force CPU enabled",
                "fix": "Turn off Force CPU to use GPU.",
                "color": "#9a6700",
            }
        if cuda_available:
            return {
                "status": "GPU status: Available ({}){}".format(
                    gpu_name,
                    vram_text,
                ),
                "fix": "GPU can be used. No fix needed.",
                "color": "#0b6e4f",
            }
        return {
            "status": "GPU status: Unavailable",
            "fix": fix_command,
            "color": "#b00020",
        }

    def _update_human_gpu_status_ui(self) -> None:
        """Refresh the user-facing GPU availability text."""
        if not self._is_human_tab_active():
            self._cancel_human_gpu_status_refresh()
            return
        info = self._get_human_gpu_status_info()
        self.human_gpu_status_var.set(info["status"])
        self.human_gpu_fix_var.set(info["fix"])
        label = self._human_gpu_status_label
        if label is not None:
            try:
                label.configure(fg=info.get("color", "#202020"))
            except tk.TclError:
                pass
        text_widget = self._human_gpu_fix_text
        if text_widget is not None:
            try:
                text_widget.delete("1.0", tk.END)
                text_widget.insert("1.0", info["fix"])
            except tk.TclError:
                pass
        self._schedule_human_gpu_status_refresh()

    def _schedule_human_gpu_status_refresh(self) -> None:
        """Refresh GPU status periodically without updating too frequently."""
        self._cancel_human_gpu_status_refresh()
        if not self._is_human_tab_active():
            return
        self._human_gpu_fix_after_id = self.root.after(
            5000,
            self._update_human_gpu_status_ui,
        )

    def _select_all_human_gpu_fix_text(self, _event=None) -> str:
        """Select all text in the How to fix text box."""
        text_widget = self._human_gpu_fix_text
        if text_widget is None:
            return "break"
        try:
            text_widget.tag_add("sel", "1.0", "end-1c")
            text_widget.mark_set("insert", "1.0")
            text_widget.see("insert")
        except tk.TclError:
            pass
        return "break"

    def _load_human_segmentation_runtime(
        self, force_cpu: bool
    ) -> Tuple[Any, Any, Any]:
        if self._human_segmentation_module is None:
            self._human_segmentation_module = importlib.import_module(
                "gs360_SegmentationMaskTool"
            )
        module = self._human_segmentation_module
        device_type = "cpu"
        if not force_cpu and module.torch.cuda.is_available():
            device_type = "cuda"
        if (
            self._human_segmentation_model is None
            or self._human_segmentation_device is None
            or self._human_segmentation_device_type != device_type
        ):
            device = module.torch.device(device_type)
            if device_type == "cuda":
                module.torch.backends.cudnn.benchmark = True
            model = module.load_model(device, conf_thres=module.SCORE_THRESH)
            self._human_segmentation_model = model
            self._human_segmentation_device = device
            self._human_segmentation_device_type = device_type
        return (
            module,
            self._human_segmentation_model,
            self._human_segmentation_device,
        )

    def _release_human_segmentation_runtime(self) -> None:
        module = self._human_segmentation_module
        model = self._human_segmentation_model
        device_type = self._human_segmentation_device_type
        if module is not None and model is not None and device_type == "cuda":
            try:
                model.to(module.torch.device("cpu"))
            except Exception:
                pass
        self._human_segmentation_model = None
        self._human_segmentation_device = None
        self._human_segmentation_device_type = None
        if module is not None and device_type == "cuda":
            try:
                module.torch.cuda.empty_cache()
            except Exception:
                pass

    def _generate_human_preview_base_mask(
        self,
        module: Any,
        model: Any,
        device: Any,
        image: Image.Image,
        settings: Dict[str, Any],
    ) -> Optional[np.ndarray]:
        image_rgb = np.array(image.convert("RGB"))
        pred = module.run_inference(model, device, image_rgb)

        image_shape = (image.size[1], image.size[0])
        mask = module.target_mask_from_prediction(
            pred,
            settings["targets"],
            module.SCORE_THRESH,
            module.MASK_THRESH,
        )
        mask = module.refine_mask(
            mask,
            close=module.CLOSE_KERNEL,
            expand_mode="pixels",
            expand_pixels=0,
            expand_percent=0.0,
            image_shape=image_shape,
        )
        if settings["include_shadow"]:
            shadow = module.estimate_shadow_mask(
                image_rgb,
                mask,
                t=module.SHADOW_T,
                sigma=module.SHADOW_SIGMA,
                near_px=module.SHADOW_NEAR,
                min_area=module.SHADOW_MIN_AREA,
            )
            if shadow is not None:
                base = np.zeros_like(shadow) if mask is None else mask
                mask = np.maximum(base, shadow)
        return mask

    def _expand_human_preview_mask(
        self,
        module: Any,
        base_mask: Optional[np.ndarray],
        image: Image.Image,
        settings: Dict[str, Any],
    ) -> Optional[np.ndarray]:
        if base_mask is None:
            return None
        return module.expand_mask(
            base_mask.copy(),
            expand_mode=settings["expand_mode"],
            expand_pixels=settings["expand_pixels"],
            expand_percent=settings["expand_percent"],
            image_shape=(image.size[1], image.size[0]),
        )

    def _compose_human_preview_overlay(
        self,
        image: Image.Image,
        mask: Optional[np.ndarray],
    ) -> Image.Image:
        if mask is None or not np.any(mask):
            return image.convert("RGB")

        rgb = np.array(image.convert("RGB"), dtype=np.uint8)
        active = mask > 0

        darkened = rgb.astype(np.float32)
        darkened[active] *= 0.45
        rgb = np.clip(darkened, 0, 255).astype(np.uint8)
        return Image.fromarray(rgb, "RGB")

    def _compose_human_mask_editor_overlay(
        self,
        image: Image.Image,
        base_mask: Optional[np.ndarray],
        manual_mask: Optional[np.ndarray],
    ) -> Image.Image:
        rgb = np.array(image.convert("RGB"), dtype=np.uint8)
        base_active = (
            np.zeros(rgb.shape[:2], dtype=bool)
            if base_mask is None else
            (base_mask > 0)
        )
        manual_active = (
            np.zeros(rgb.shape[:2], dtype=bool)
            if manual_mask is None else
            (manual_mask > 0)
        )
        if not np.any(base_active) and not np.any(manual_active):
            return Image.fromarray(rgb, "RGB")

        composed = rgb.astype(np.float32)
        if np.any(base_active):
            composed[base_active] *= 0.45
        if np.any(manual_active):
            manual_color = np.array(
                self._hex_to_rgb(self._human_mask_editor_manual_color),
                dtype=np.float32,
            )
            composed[manual_active] = (
                (composed[manual_active] * 0.35) +
                (manual_color * 0.65)
            )
        composed = np.clip(composed, 0, 255).astype(np.uint8)
        return Image.fromarray(composed, "RGB")

    @staticmethod
    def _hex_to_rgb(color_text: str) -> Tuple[int, int, int]:
        text = color_text.strip()
        if len(text) == 7 and text.startswith("#"):
            try:
                return (
                    int(text[1:3], 16),
                    int(text[3:5], 16),
                    int(text[5:7], 16),
                )
            except ValueError:
                pass
        return (0, 200, 255)

    def _human_preview_item_name_at(
        self,
        x_pos: int,
        y_pos: int,
    ) -> Optional[str]:
        for left, top, right, bottom, name in self._human_preview_hit_regions:
            if left <= x_pos <= right and top <= y_pos <= bottom:
                return name
        return None

    def _find_human_preview_cache_item(
        self,
        name: str,
    ) -> Optional[Tuple[str, Image.Image, Optional[np.ndarray]]]:
        for item in self._human_preview_cache_items:
            if item[0] == name:
                return item
        return None

    def _human_manual_mask_key(self, name: str) -> str:
        stem = Path(name).stem
        view_id = self._extract_multicam_view_id(stem)
        if view_id:
            return "view__{}".format(view_id)
        return "file__{}".format(stem)

    @staticmethod
    def _normalize_binary_mask(
        mask: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        if mask is None:
            return None
        return np.where(mask > 0, 255, 0).astype(np.uint8)

    def _apply_human_manual_mask_layers(
        self,
        base_mask: Optional[np.ndarray],
        name: str,
        image_shape: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        add_mask = self._human_preview_manual_masks.get(
            self._human_manual_mask_key(name)
        )
        mask = self._normalize_binary_mask(base_mask)
        if mask is None:
            if add_mask is None:
                return None
            mask = np.zeros(image_shape, dtype=np.uint8)
        if add_mask is not None:
            add_bool = add_mask > 0
            mask[add_bool] = 255
        if not np.any(mask):
            return None
        return mask

    def _apply_human_edge_fuse(
        self,
        module: Any,
        mask: Optional[np.ndarray],
        settings: Dict[str, Any],
    ) -> Optional[np.ndarray]:
        if mask is None:
            return None
        return module.fuse_mask_to_edges(
            mask,
            edge_fuse_pixels=settings["edge_fuse_pixels"],
        )

    def _resolve_human_preview_mask(
        self,
        name: str,
        image: Image.Image,
        base_mask: Optional[np.ndarray],
        settings: Dict[str, Any],
    ) -> Optional[np.ndarray]:
        auto_mask = None
        if self._human_segmentation_module is None:
            auto_mask = None
        else:
            auto_mask = self._expand_human_preview_mask(
                self._human_segmentation_module,
                base_mask,
                image,
                settings,
            )
            auto_mask = self._apply_human_edge_fuse(
                self._human_segmentation_module,
                auto_mask,
                settings,
            )
        return self._apply_human_manual_mask_layers(
            auto_mask,
            name,
            (image.size[1], image.size[0]),
        )

    def _build_human_preview_sheet(
        self,
        rendered_items: Sequence[Tuple[str, Image.Image, int]],
    ) -> Image.Image:
        margin = HUMAN_PREVIEW_MARGIN
        self._human_preview_hit_regions = []
        size_value = self.human_preview_size_var.get().strip()
        if size_value == "800":
            thumb_w, thumb_h = (800, 800)
            cols = 2
        elif size_value.lower() == "original":
            max_w = max(item[1].width for item in rendered_items)
            max_h = max(item[1].height for item in rendered_items)
            thumb_w = max(1, max_w)
            thumb_h = max(1, max_h)
            cols = 1
        elif size_value.lower() == "frame fit":
            canvas_width = 1280
            if self.human_preview_canvas is not None:
                try:
                    canvas_width = max(
                        480,
                        int(self.human_preview_canvas.winfo_width()),
                    )
                except Exception:
                    canvas_width = 1280
            cols = max(1, min(4, len(rendered_items)))
            available_w = max(
                120,
                canvas_width - ((cols + 1) * margin),
            )
            thumb_w = max(120, int(available_w / float(cols)))
            thumb_h = thumb_w
        else:
            thumb_w, thumb_h = (320, 320)
            cols = 4
        text_height = 42
        rows = max(1, int(math.ceil(len(rendered_items) / float(cols))))
        sheet_w = (cols * thumb_w) + ((cols + 1) * margin)
        sheet_h = (rows * (thumb_h + text_height)) + ((rows + 1) * margin)
        sheet = Image.new("RGB", (sheet_w, sheet_h), "#f4f4f4")
        draw = ImageDraw.Draw(sheet)

        for idx, (name, image, active_pixels) in enumerate(rendered_items):
            col = idx % cols
            row = idx // cols
            x = margin + (col * (thumb_w + margin))
            y = margin + (row * (thumb_h + text_height + margin))
            draw.rectangle(
                [x - 1, y - 1, x + thumb_w + 1, y + thumb_h + text_height + 1],
                outline="#cfcfcf",
                width=1,
            )
            if size_value.lower() == "original":
                thumb = image.copy()
            else:
                thumb = image.copy()
                thumb.thumbnail((thumb_w, thumb_h), Image.LANCZOS)
            is_marked = name in self._human_preview_marked_names
            if is_marked:
                thumb = Image.blend(
                    thumb.convert("RGB"),
                    Image.new("RGB", thumb.size, "black"),
                    0.68,
                )
            offset_x = x + max(0, (thumb_w - thumb.width) // 2)
            offset_y = y + max(0, (thumb_h - thumb.height) // 2)
            sheet.paste(thumb, (offset_x, offset_y))
            self._human_preview_hit_regions.append(
                (
                    x,
                    y,
                    x + thumb_w,
                    y + thumb_h + text_height,
                    name,
                )
            )
            name_text = name if len(name) <= 28 else "{}...".format(name[:25])
            if is_marked:
                name_text = "{} [hide]".format(name_text)
            if self._human_manual_mask_key(name) in self._human_preview_manual_masks:
                name_text = "{} [manual]".format(name_text)
            mask_text = (
                "mask px: {:,}".format(active_pixels)
                if active_pixels > 0 else
                "mask: empty"
            )
            draw.text((x, y + thumb_h + 6), name_text, fill="#202020")
            draw.text((x, y + thumb_h + 22), mask_text, fill="#606060")
        return sheet

    def _render_human_preview_sheet(self) -> None:
        if not self._human_preview_rendered_items:
            self._render_human_preview_image(None)
            return
        self._render_human_preview_image(
            self._build_human_preview_sheet(self._human_preview_rendered_items),
            preserve_scroll=True,
        )

    def _build_human_preview_status_text(
        self,
        settings: Dict[str, Any],
        device_label: str,
    ) -> str:
        preview_count = len(self._human_preview_cache_items)
        total_count = max(preview_count, self._human_preview_group_total_count)
        if total_count > preview_count:
            count_text = "{}/{}".format(preview_count, total_count)
        else:
            count_text = str(preview_count)
        status_text = (
            "Group: {} | images: {} | targets: {} | expand: {} | edge fuse: {} | size: {} | device: {}"
        ).format(
            self._human_preview_group_name,
            count_text,
            ", ".join(settings["targets"]),
            self._human_expand_label(settings),
            self._human_edge_fuse_label(settings),
            self.human_preview_size_var.get().strip(),
            device_label,
        )
        manual_count = len(self._human_preview_manual_masks)
        if manual_count > 0:
            status_text += " | manual: {} | expand locked".format(manual_count)
        return status_text

    def _rebuild_human_preview_from_cache(self) -> bool:
        if not self._human_preview_cache_items:
            return False
        if self._human_segmentation_module is None:
            return False
        settings = self._collect_human_mask_settings(show_errors=False)
        if settings is None:
            return False
        if self._human_preview_cache_signature != self._human_preview_signature(
            settings
        ):
            return False
        rendered_items: List[Tuple[str, Image.Image, int]] = []
        for name, image, base_mask in self._human_preview_cache_items:
            expanded = self._resolve_human_preview_mask(
                name,
                image,
                base_mask,
                settings,
            )
            rendered_items.append(
                (
                    name,
                    self._compose_human_preview_overlay(image, expanded),
                    int(np.count_nonzero(expanded)) if expanded is not None else 0,
                )
            )
        self._human_preview_rendered_items = rendered_items
        self.human_preview_status_var.set(
            self._build_human_preview_status_text(
                settings,
                self._human_segmentation_device_type or "cpu",
            )
        )
        self._render_human_preview_sheet()
        return True

    def _render_human_preview_image(
        self,
        image: Optional[Image.Image],
        preserve_scroll: bool = False,
    ) -> None:
        canvas = self.human_preview_canvas
        if canvas is None:
            return
        x_view = canvas.xview()
        y_view = canvas.yview()
        canvas.delete("all")
        self._human_preview_photo = None
        if image is None:
            self._human_preview_hit_regions = []
            canvas.configure(scrollregion=(0, 0, 1, 1))
            canvas.create_text(
                16,
                16,
                anchor="nw",
                width=max(120, int(canvas.winfo_width()) - 32),
                text=self.human_preview_status_var.get(),
                fill="#555555",
            )
            return
        photo = ImageTk.PhotoImage(image=image, master=self.root)
        self._human_preview_photo = photo
        canvas.create_image(0, 0, anchor="nw", image=photo)
        canvas.configure(scrollregion=(0, 0, image.width, image.height))
        if preserve_scroll:
            try:
                canvas.xview_moveto(x_view[0] if x_view else 0.0)
                canvas.yview_moveto(y_view[0] if y_view else 0.0)
            except Exception:
                canvas.xview_moveto(0.0)
                canvas.yview_moveto(0.0)
        else:
            canvas.xview_moveto(0.0)
            canvas.yview_moveto(0.0)

    def _clear_human_preview_cache(self, clear_display: bool = False) -> None:
        self._human_preview_cache_items = []
        self._human_preview_cache_signature = None
        self._human_preview_original_rendered_items = []
        self._human_preview_original_cache_items = []
        self._human_preview_original_signature = None
        self._human_preview_original_settings = {}
        self._human_preview_original_status_text = ""
        self._human_preview_group_name = ""
        self._human_preview_group_total_count = 0
        self._human_preview_hit_regions = []
        self._human_preview_marked_names = set()
        self._human_preview_manual_masks = {}
        self._human_preview_rendered_items = []
        self._human_preview_pending = False
        self._close_human_mask_editor()
        self._update_human_expand_state()
        self._update_human_preview_update_button_state()
        if self._human_preview_after_id is not None:
            try:
                self.root.after_cancel(self._human_preview_after_id)
            except Exception:
                pass
            self._human_preview_after_id = None
        if clear_display:
            self.human_preview_status_var.set(
                "Preview cache cleared. Run Preview masks again to regenerate."
            )
            self._render_human_preview_image(None)

    def _set_human_preview_running(self, running: bool) -> None:
        self._human_preview_busy = running
        if self.human_preview_button is not None:
            self.human_preview_button.configure(
                state="disabled" if running else "normal"
            )

    def _complete_human_preview(
        self,
        rendered_items: Sequence[Tuple[str, Image.Image, int]],
        status_text: str,
        log_line: str,
    ) -> None:
        self._human_preview_rendered_items = list(rendered_items)
        self._apply_human_preview_slider_value()
        self.human_preview_status_var.set(status_text)
        if self._human_preview_rendered_items:
            self._render_human_preview_sheet()
        else:
            self._render_human_preview_image(None)
        if log_line:
            self._append_text_widget(self.human_log, log_line)
        self._set_human_preview_running(False)
        if self._human_preview_pending:
            self._human_preview_pending = False
            self._schedule_human_preview_refresh(delay_ms=0)

    def _apply_human_preview_marked_removal(self) -> None:
        if not self._human_preview_marked_names:
            return
        hidden_names = set(self._human_preview_marked_names)
        before_count = len(self._human_preview_cache_items)
        self._human_preview_cache_items = [
            item
            for item in self._human_preview_cache_items
            if item[0] not in hidden_names
        ]
        hidden_count = before_count - len(self._human_preview_cache_items)
        self._human_preview_marked_names = set()
        self._update_human_preview_update_button_state()
        if hidden_count <= 0:
            return
        self._human_preview_rendered_items = [
            item
            for item in self._human_preview_rendered_items
            if item[0] not in hidden_names
        ]
        if self._human_preview_cache_items and self._rebuild_human_preview_from_cache():
            self._append_text_widget(
                self.human_log,
                "[preview] Hid {} image(s) from preview.".format(hidden_count),
            )
            return
        if self._human_preview_rendered_items:
            current_status = self.human_preview_status_var.get().strip()
            if current_status:
                current_status = "{} | hidden: {}".format(
                    current_status,
                    hidden_count,
                )
            else:
                current_status = "Preview updated. Hidden: {} image(s).".format(
                    hidden_count,
                )
            self.human_preview_status_var.set(current_status)
            self._render_human_preview_sheet()
            self._append_text_widget(
                self.human_log,
                "[preview] Hid {} image(s) from preview.".format(hidden_count),
            )
            return
        self._human_preview_rendered_items = []
        self._human_preview_hit_regions = []
        self.human_preview_status_var.set(
            "All preview images are hidden. Run Preview masks again to regenerate."
        )
        self._render_human_preview_image(None, preserve_scroll=True)
        self._append_text_widget(
            self.human_log,
            "[preview] Hid {} image(s) from preview.".format(hidden_count),
        )

    def _reset_human_preview_state(self) -> None:
        if not self._human_preview_original_cache_items:
            return
        self._human_preview_manual_masks = {}
        self._restore_human_preview_reset_settings(
            dict(self._human_preview_original_settings)
        )
        self._human_preview_cache_items = list(
            self._human_preview_original_cache_items
        )
        self._human_preview_cache_signature = (
            self._human_preview_original_signature
        )
        self._human_preview_rendered_items = list(
            self._human_preview_original_rendered_items
        )
        self._human_preview_marked_names = set()
        self._human_preview_hit_regions = []
        self.human_preview_status_var.set(
            self._human_preview_original_status_text or
            "Preview reset to the last inferred state."
        )
        self._close_human_mask_editor()
        self._update_human_expand_state()
        self._update_human_preview_update_button_state()
        if self._human_preview_rendered_items:
            self._render_human_preview_sheet()
        else:
            self._render_human_preview_image(None, preserve_scroll=True)
        self._append_text_widget(
            self.human_log,
            "[preview] Preview reset to the last inferred state.",
        )

    def _ensure_human_mask_editor_window(self) -> Optional[tk.Toplevel]:
        if self.root is None:
            return None
        existing = self._human_mask_editor_window
        if existing is not None:
            try:
                if existing.winfo_exists():
                    return existing
            except Exception:
                pass
        top = tk.Toplevel(self.root)
        top.title("Mask Editor")
        top.geometry("980x920")
        top.protocol("WM_DELETE_WINDOW", self._close_human_mask_editor)

        controls = tk.Frame(top)
        controls.pack(fill="x", padx=8, pady=(8, 4))
        tk.Label(controls, text="Tool").pack(side=tk.LEFT, padx=(0, 4))
        tk.Radiobutton(
            controls,
            text="Add",
            variable=self._human_mask_editor_tool_var,
            value="add",
        ).pack(side=tk.LEFT, padx=(0, 8))
        tk.Radiobutton(
            controls,
            text="Erase",
            variable=self._human_mask_editor_tool_var,
            value="erase",
        ).pack(side=tk.LEFT, padx=(0, 12))
        tk.Scale(
            controls,
            orient="horizontal",
            from_=2,
            to=200,
            resolution=1,
            showvalue=True,
            variable=self._human_mask_editor_brush_var,
            length=240,
        ).pack(side=tk.LEFT, padx=(0, 12))
        tk.Button(
            controls,
            text="Manual Color...",
            command=self._choose_human_mask_editor_manual_color,
        ).pack(side=tk.LEFT, padx=(0, 6))
        self._human_mask_editor_color_swatch = tk.Label(
            controls,
            width=3,
            relief="solid",
            borderwidth=1,
            background=self._human_mask_editor_manual_color,
        )
        self._human_mask_editor_color_swatch.pack(side=tk.LEFT, padx=(0, 12))
        tk.Button(
            controls,
            text="Apply",
            command=self._apply_human_mask_editor,
        ).pack(side=tk.RIGHT, padx=(4, 0))
        tk.Button(
            controls,
            text="Close",
            command=self._close_human_mask_editor,
        ).pack(side=tk.RIGHT, padx=(4, 0))
        tk.Button(
            controls,
            text="Reset",
            command=self._reset_human_mask_editor,
        ).pack(side=tk.RIGHT, padx=(4, 0))

        note = tk.Label(
            top,
            text=(
                "Recommended for multi-360 camera rigs or a 360 camera mounted on a drone.\n"
                "Manual paint is shared per camera ID such as _A, _B, ... _X.\n"
                "Auto mask=dark overlay, manual paint=selected color. Erase removes manual paint only."
            ),
            anchor="w",
            justify="left",
            wraplength=920,
        )
        note.pack(fill="x", padx=8, pady=(0, 4))

        canvas_wrap = tk.Frame(top)
        canvas_wrap.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        canvas_x = tk.Scrollbar(canvas_wrap, orient="horizontal")
        canvas_x.pack(side=tk.BOTTOM, fill="x")
        canvas_y = tk.Scrollbar(canvas_wrap, orient="vertical")
        canvas_y.pack(side=tk.RIGHT, fill="y")
        canvas = tk.Canvas(
            canvas_wrap,
            background="#202020",
            xscrollcommand=canvas_x.set,
            yscrollcommand=canvas_y.set,
            highlightthickness=1,
            highlightbackground="#404040",
            cursor="crosshair",
        )
        canvas.pack(side=tk.LEFT, fill="both", expand=True)
        canvas_x.configure(command=canvas.xview)
        canvas_y.configure(command=canvas.yview)
        canvas.bind("<ButtonPress-1>", self._on_human_mask_editor_press)
        canvas.bind("<B1-Motion>", self._on_human_mask_editor_drag)
        canvas.bind(
            "<ButtonRelease-1>",
            self._on_human_mask_editor_release,
        )
        canvas.bind("<MouseWheel>", self._on_human_mask_editor_mousewheel)
        canvas.bind("<Button-4>", self._on_human_mask_editor_mousewheel)
        canvas.bind("<Button-5>", self._on_human_mask_editor_mousewheel)
        self._human_mask_editor_window = top
        self._human_mask_editor_canvas = canvas
        return top

    def _close_human_mask_editor(self) -> None:
        top = self._human_mask_editor_window
        self._human_mask_editor_window = None
        self._human_mask_editor_canvas = None
        self._human_mask_editor_photo = None
        self._human_mask_editor_color_swatch = None
        self._human_mask_editor_name = ""
        self._human_mask_editor_image = None
        self._human_mask_editor_base_mask = None
        self._human_mask_editor_mask = None
        self._human_mask_editor_initial_mask = None
        self._human_mask_editor_last_point = None
        if top is not None:
            try:
                if top.winfo_exists():
                    top.destroy()
            except Exception:
                pass

    def _update_human_mask_editor_color_swatch(self) -> None:
        swatch = self._human_mask_editor_color_swatch
        if swatch is None:
            return
        try:
            swatch.configure(background=self._human_mask_editor_manual_color)
        except tk.TclError:
            pass

    def _choose_human_mask_editor_manual_color(self) -> None:
        initial = self._human_mask_editor_manual_color
        try:
            _rgb, color_text = colorchooser.askcolor(
                color=initial,
                title="Select manual paint color",
                parent=self._human_mask_editor_window,
            )
        except Exception:
            color_text = None
        if not color_text:
            return
        self._human_mask_editor_manual_color = str(color_text)
        self._update_human_mask_editor_color_swatch()
        self._render_human_mask_editor()

    def _open_human_mask_editor(self, name: str) -> None:
        cache_item = self._find_human_preview_cache_item(name)
        if cache_item is None:
            return
        settings = self._collect_human_mask_settings(show_errors=False)
        if settings is None:
            return
        _, image, base_mask = cache_item
        auto_mask = None
        if self._human_segmentation_module is not None:
            auto_mask = self._expand_human_preview_mask(
                self._human_segmentation_module,
                base_mask,
                image,
                settings,
            )
            auto_mask = self._apply_human_edge_fuse(
                self._human_segmentation_module,
                auto_mask,
                settings,
            )
        manual_key = self._human_manual_mask_key(name)
        add_mask = self._human_preview_manual_masks.get(manual_key)
        if add_mask is None:
            add_mask = np.zeros((image.size[1], image.size[0]), dtype=np.uint8)
        else:
            add_mask = self._normalize_binary_mask(add_mask)
        top = self._ensure_human_mask_editor_window()
        if top is None:
            return
        self._human_mask_editor_name = name
        self._human_mask_editor_image = image.copy()
        self._human_mask_editor_base_mask = self._normalize_binary_mask(auto_mask)
        self._human_mask_editor_mask = add_mask.copy()
        self._human_mask_editor_initial_mask = add_mask.copy()
        self._human_mask_editor_last_point = None
        longest_side = max(image.size[0], image.size[1], 1)
        self._human_mask_editor_zoom = min(1.0, 800.0 / float(longest_side))
        if self._human_mask_editor_zoom <= 0:
            self._human_mask_editor_zoom = 1.0
        top.title("Mask Editor - {}".format(name))
        top.deiconify()
        top.lift()
        top.focus_force()
        self._render_human_mask_editor()

    def _render_human_mask_editor(self) -> None:
        canvas = self._human_mask_editor_canvas
        image = self._human_mask_editor_image
        manual_mask = self._human_mask_editor_mask
        if canvas is None or image is None or manual_mask is None:
            return
        display_mask = self._normalize_binary_mask(self._human_mask_editor_base_mask)
        if display_mask is None:
            display_mask = np.zeros((image.size[1], image.size[0]), dtype=np.uint8)
        else:
            display_mask = display_mask.copy()
        overlay = self._compose_human_mask_editor_overlay(
            image,
            display_mask,
            manual_mask,
        )
        src_w, src_h = overlay.size
        zoom = max(0.01, float(self._human_mask_editor_zoom))
        dst_w = max(1, int(round(src_w * zoom)))
        dst_h = max(1, int(round(src_h * zoom)))
        if (dst_w, dst_h) != overlay.size:
            overlay = overlay.resize((dst_w, dst_h), Image.LANCZOS)
        photo = ImageTk.PhotoImage(image=overlay, master=self.root)
        self._human_mask_editor_photo = photo
        canvas.delete("all")
        canvas.create_image(0, 0, anchor="nw", image=photo)
        canvas.configure(scrollregion=(0, 0, dst_w, dst_h))
        canvas.xview_moveto(0.0)
        canvas.yview_moveto(0.0)

    def _human_mask_editor_canvas_to_image(
        self,
        event,
    ) -> Optional[Tuple[int, int]]:
        canvas = self._human_mask_editor_canvas
        image = self._human_mask_editor_image
        if canvas is None or image is None:
            return None
        zoom = max(0.01, float(self._human_mask_editor_zoom))
        x_pos = int(canvas.canvasx(event.x) / zoom)
        y_pos = int(canvas.canvasy(event.y) / zoom)
        if x_pos < 0 or y_pos < 0:
            return None
        if x_pos >= image.size[0] or y_pos >= image.size[1]:
            return None
        return (x_pos, y_pos)

    def _paint_human_mask_editor_segment(
        self,
        start_point: Tuple[int, int],
        end_point: Tuple[int, int],
    ) -> None:
        mask = self._human_mask_editor_mask
        if mask is None:
            return
        brush_radius = max(
            1,
            int(round(float(self._human_mask_editor_brush_var.get()) / 2.0)),
        )
        value = 255 if self._human_mask_editor_tool_var.get() == "add" else 0
        x0 = float(start_point[0])
        y0 = float(start_point[1])
        x1 = float(end_point[0])
        y1 = float(end_point[1])
        dx = x1 - x0
        dy = y1 - y0
        distance = math.hypot(dx, dy)
        spacing = max(1.0, float(brush_radius) * 0.3)
        steps = max(1, int(math.ceil(distance / spacing)))
        for step in range(steps + 1):
            t_value = float(step) / float(steps)
            px = int(round(x0 + (dx * t_value)))
            py = int(round(y0 + (dy * t_value)))
            cv2.circle(
                mask,
                (px, py),
                brush_radius,
                int(value),
                thickness=-1,
            )

    def _on_human_mask_editor_press(self, event) -> str:
        coords = self._human_mask_editor_canvas_to_image(event)
        if coords is None:
            self._human_mask_editor_last_point = None
            return "break"
        self._human_mask_editor_last_point = coords
        self._paint_human_mask_editor_segment(coords, coords)
        self._render_human_mask_editor()
        return "break"

    def _on_human_mask_editor_drag(self, event) -> str:
        coords = self._human_mask_editor_canvas_to_image(event)
        if coords is None:
            return "break"
        last_point = self._human_mask_editor_last_point
        if last_point is None:
            last_point = coords
        self._paint_human_mask_editor_segment(last_point, coords)
        self._human_mask_editor_last_point = coords
        self._render_human_mask_editor()
        return "break"

    def _on_human_mask_editor_release(self, _event=None) -> str:
        self._human_mask_editor_last_point = None
        return "break"

    def _on_human_mask_editor_mousewheel(self, event) -> str:
        canvas = self._human_mask_editor_canvas
        if canvas is None:
            return "break"
        delta = 0
        if getattr(event, "num", None) == 4:
            delta = -1
        elif getattr(event, "num", None) == 5:
            delta = 1
        elif getattr(event, "delta", 0):
            delta = -1 if event.delta > 0 else 1
        if delta != 0:
            canvas.yview_scroll(delta, "units")
        return "break"

    def _reset_human_mask_editor(self) -> None:
        if self._human_mask_editor_initial_mask is None:
            return
        self._human_mask_editor_mask = self._human_mask_editor_initial_mask.copy()
        self._human_mask_editor_last_point = None
        self._render_human_mask_editor()

    def _apply_human_mask_editor(self) -> None:
        name = self._human_mask_editor_name
        mask = self._human_mask_editor_mask
        if not name or mask is None:
            return
        manual_key = self._human_manual_mask_key(name)
        add_mask = self._normalize_binary_mask(mask)
        if add_mask is not None and np.any(add_mask):
            self._human_preview_manual_masks[manual_key] = add_mask
        else:
            self._human_preview_manual_masks.pop(manual_key, None)
        self._human_preview_marked_names.discard(name)
        self._update_human_expand_state()
        self._update_human_preview_update_button_state()
        if self._rebuild_human_preview_from_cache():
            self._append_text_widget(
                self.human_log,
                "[manual] Applied manual mask edit to {}.".format(name),
            )
        self._close_human_mask_editor()

    def _complete_human_preview_loaded(
        self,
        rendered_items: Sequence[Tuple[str, Image.Image, int]],
        cache_items: Sequence[Tuple[str, Image.Image, Optional[np.ndarray]]],
        signature: Tuple[Any, ...],
        group_name: str,
        total_count: int,
        settings: Dict[str, Any],
        device_label: str,
        log_line: str,
    ) -> None:
        self._human_preview_cache_items = list(cache_items)
        self._human_preview_cache_signature = signature
        self._human_preview_original_rendered_items = list(rendered_items)
        self._human_preview_original_cache_items = list(cache_items)
        self._human_preview_original_signature = signature
        self._human_preview_original_settings = (
            self._capture_human_preview_reset_settings()
        )
        self._human_preview_group_name = group_name
        self._human_preview_group_total_count = int(total_count)
        self._human_preview_hit_regions = []
        self._human_preview_marked_names = set()
        self._human_preview_manual_masks = {}
        self._close_human_mask_editor()
        status_text = self._build_human_preview_status_text(
            settings,
            device_label,
        )
        self._human_preview_original_status_text = status_text
        self._update_human_expand_state()
        self._update_human_preview_update_button_state()
        self._complete_human_preview(
            rendered_items,
            status_text,
            log_line,
        )

    def _preview_human_masks(self, auto_refresh: bool = False) -> None:
        settings = self._collect_human_mask_settings(show_errors=not auto_refresh)
        if settings is None or self._human_preview_busy:
            return
        if auto_refresh and self._rebuild_human_preview_from_cache():
            return
        try:
            group_name, group_paths = self._collect_human_preview_group(
                settings["input_path"]
            )
        except Exception as exc:
            if not auto_refresh:
                messagebox.showerror(
                    "SegmentationMaskTool Preview",
                    str(exc),
                )
            return
        confirm_message = self._should_confirm_human_preview_group(group_paths)
        if confirm_message and not auto_refresh:
            proceed = messagebox.askyesno(
                "SegmentationMaskTool Preview",
                "{}\n\nContinue preview generation?".format(confirm_message),
            )
            if not proceed:
                return
        preview_paths = list(group_paths[:HUMAN_PREVIEW_MAX_IMAGES])
        self._set_human_preview_running(True)
        status = "Generating preview for the first image group..."
        self.human_preview_status_var.set(status)
        if not self._human_preview_rendered_items:
            self._render_human_preview_image(None)
        if not auto_refresh:
            self._append_text_widget(
                self.human_log,
                "[preview] Generating mask overlay preview...",
            )

        def worker() -> None:
            try:
                module, model, device = self._load_human_segmentation_runtime(
                    settings["cpu"]
                )
                cache_items: List[
                    Tuple[str, Image.Image, Optional[np.ndarray]]
                ] = []
                rendered_items: List[Tuple[str, Image.Image, int]] = []
                for path in preview_paths:
                    image = Image.open(str(path)).convert("RGB")
                    base_mask = self._generate_human_preview_base_mask(
                        module,
                        model,
                        device,
                        image,
                        settings,
                    )
                    cache_items.append((path.name, image.copy(), base_mask))
                    mask = self._expand_human_preview_mask(
                        module,
                        base_mask,
                        image,
                        settings,
                    )
                    mask = self._apply_human_edge_fuse(
                        module,
                        mask,
                        settings,
                    )
                    rendered_items.append(
                        (
                            path.name,
                            self._compose_human_preview_overlay(image, mask),
                            int(np.count_nonzero(mask)) if mask is not None else 0,
                        )
                    )
                log_line = (
                    "[preview] Updated group '{}' ({} images shown / {} total, expand={}, size={}, device={})"
                ).format(
                    group_name,
                    len(preview_paths),
                    len(group_paths),
                    self._human_expand_label(settings),
                    self.human_preview_size_var.get().strip(),
                    device.type,
                )
                if auto_refresh:
                    log_line = ""
                self.root.after(
                    0,
                    lambda items=rendered_items,
                    cache=cache_items,
                    line=log_line,
                    name=group_name,
                    total=len(group_paths),
                    cfg=dict(settings),
                    dev=device.type:
                    self._complete_human_preview_loaded(
                        items,
                        cache,
                        self._human_preview_signature(settings),
                        name,
                        total,
                        cfg,
                        dev,
                        line,
                    ),
                )
            except Exception as exc:
                error_text = "Preview failed: {}".format(exc)
                self.root.after(
                    0,
                    lambda text=error_text: self._complete_human_preview(
                        [],
                        text,
                        "[preview] {}".format(text),
                    ),
                )
            finally:
                self._release_human_segmentation_runtime()

        threading.Thread(
            target=worker,
            name="human-mask-preview",
            daemon=True,
        ).start()

    def _cleanup_human_manual_mask_temp_dir(self) -> None:
        temp_dir = self._human_manual_mask_temp_dir
        self._human_manual_mask_temp_dir = None
        if temp_dir is None:
            return
        try:
            if temp_dir.exists():
                shutil.rmtree(str(temp_dir), ignore_errors=True)
        except Exception:
            pass

    def _prepare_human_manual_mask_override_dir(self) -> Optional[Path]:
        self._cleanup_human_manual_mask_temp_dir()
        if not self._human_preview_manual_masks:
            return None
        temp_root = SCRIPT_DIR / "_temp" / "human_manual_masks"
        temp_root.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(
            tempfile.mkdtemp(
                prefix="manual_mask_",
                dir=str(temp_root),
            )
        )
        for manual_key, add_mask in self._human_preview_manual_masks.items():
            if add_mask is None:
                continue
            out_path = temp_dir / "{}__add.png".format(manual_key)
            Image.fromarray(self._normalize_binary_mask(add_mask)).save(
                str(out_path)
            )
        self._human_manual_mask_temp_dir = temp_dir
        return temp_dir

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
        ):
            if widget is None:
                continue
            try:
                widget.configure(state=state)
            except tk.TclError:
                pass
        rotate_enabled = fmt in {"transforms", "all"}
        if not rotate_enabled:
            self.msxml_vars["pc_rotate_x"].set(False)
        elif not self.msxml_vars["pc_rotate_x"].get():
            self.msxml_vars["pc_rotate_x"].set(True)
        if self.msxml_points_rotate_check is not None:
            try:
                self.msxml_points_rotate_check.configure(
                    state="normal" if rotate_enabled else "disabled"
                )
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
        container = self._create_tab_shell(
            parent,
            "MS360xmlToPerspCams",
            "Convert Metashape 360 camera XML data into perspective camera exports and related assets.",
        )

        params = tk.LabelFrame(container, text="Parameters")
        params.pack(fill="x", padx=8, pady=8)

        self.msxml_vars = {
            "xml": tk.StringVar(),
            "output": tk.StringVar(),
            "preset": tk.StringVar(value="full360coverage"),
            "format": tk.StringVar(value="Metashape XML"),
            "ext": tk.StringVar(value="jpg"),
            "cut": tk.BooleanVar(value=False),
            "cut_input": tk.StringVar(),
            "cut_out": tk.StringVar(),
            "points_ply": tk.StringVar(),
            "pc_rotate_x": tk.BooleanVar(value=False),
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
            text="Rotate pointcloud for transforms.json (X +180)",
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

    def _build_dualfisheye_tab(self, parent: tk.Widget) -> None:
        container = self._create_tab_shell(
            parent,
            "DualFisheyePipeline (Experimental)",
            "Inspect and process dual-fisheye captures into calibrated perspective outputs.",
        )

        scroll_wrap = tk.Frame(container)
        scroll_wrap.pack(fill="both", expand=True)
        scroll_canvas = tk.Canvas(scroll_wrap, highlightthickness=0)
        scroll_bar = tk.Scrollbar(
            scroll_wrap,
            orient=tk.VERTICAL,
            command=scroll_canvas.yview,
        )
        scroll_canvas.configure(yscrollcommand=scroll_bar.set)
        scroll_bar.pack(side=tk.RIGHT, fill=tk.Y)
        scroll_canvas.pack(side=tk.LEFT, fill="both", expand=True)
        content = tk.Frame(scroll_canvas)
        content_window = scroll_canvas.create_window(
            (0, 0),
            window=content,
            anchor="nw",
        )

        def _update_dualfisheye_scrollregion(_event: tk.Event) -> None:
            scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all"))

        def _resize_dualfisheye_scroll_content(event: tk.Event) -> None:
            scroll_canvas.itemconfigure(content_window, width=event.width)

        content.bind("<Configure>", _update_dualfisheye_scrollregion)
        scroll_canvas.bind("<Configure>", _resize_dualfisheye_scroll_content)

        params = tk.LabelFrame(content, text="Parameters")
        params.pack(fill="x", padx=8, pady=(8, 8))

        default_xml = (
            self.cli_tools_dir / "templates" / "Osmo360-Fisheye-Distortion.xml"
        )
        default_lut_rel = (
            "cli_tools/templates/DJI Osmo 360 D-Log M to Rec.709 V1.cube"
        )
        default_lut = str(
            (self.cli_tools_dir.parent / default_lut_rel).resolve()
        )
        default_workers = str(max(1, os.cpu_count() or 1))
        self.dualfisheye_vars = {
            "video": tk.StringVar(),
            "pairs_output": tk.StringVar(),
            "pair_input": tk.StringVar(),
            "mask_input": tk.StringVar(),
            "prefix": tk.StringVar(value="out"),
            "camera_xml": tk.StringVar(value=str(default_xml)),
            "camera_extrinsics_xml": tk.StringVar(),
            "pointcloud_ply": tk.StringVar(),
            "fps": tk.StringVar(value="1"),
            "ext": tk.StringVar(value="jpg"),
            "start": tk.StringVar(),
            "end": tk.StringVar(),
            "keep_rec709": tk.BooleanVar(value=False),
            "overwrite": tk.BooleanVar(value=False),
            "use_input_lut": tk.BooleanVar(value=True),
            "input_lut": tk.StringVar(value=default_lut),
            "lut_output_color_space": tk.StringVar(value="sRGB"),
            "save_fisheye_output": tk.BooleanVar(value=False),
            "fisheye_output": tk.StringVar(),
            "no_perspective": tk.BooleanVar(value=False),
            "metadata_only": tk.BooleanVar(value=False),
            "perspective_output": tk.StringVar(),
            "perspective_ext": tk.StringVar(value="jpg"),
            "perspective_mask_ext": tk.StringVar(value="png"),
            "perspective_size": tk.StringVar(value="1750"),
            "perspective_focal_mm": tk.StringVar(value="14.0"),
            "save_color_corrected_output": tk.BooleanVar(value=False),
            "color_output": tk.StringVar(),
            "workers": tk.StringVar(value=default_workers),
            "memory_throttle_percent": tk.StringVar(value="80"),
            "dry_run": tk.BooleanVar(value=False),
        }
        self.dualfisheye_metashape_f_var = tk.StringVar(
            value="Metashape f: -"
        )
        self.dualfisheye_xml_output_var = tk.StringVar(value="")
        self.dualfisheye_colmap_images_var = tk.StringVar(value="")
        self.dualfisheye_colmap_masks_var = tk.StringVar(value="")
        self.dualfisheye_colmap_sparse_var = tk.StringVar(value="")

        self.dualfisheye_vars["video"].trace_add(
            "write", self._on_dualfisheye_video_changed
        )
        self.dualfisheye_vars["prefix"].trace_add(
            "write", self._on_dualfisheye_prefix_changed
        )
        self.dualfisheye_vars["pair_input"].trace_add(
            "write", self._on_dualfisheye_pair_input_changed
        )
        self.dualfisheye_vars["save_color_corrected_output"].trace_add(
            "write", self._on_dualfisheye_output_toggle_changed
        )
        self.dualfisheye_vars["save_fisheye_output"].trace_add(
            "write", self._on_dualfisheye_output_toggle_changed
        )
        self.dualfisheye_vars["no_perspective"].trace_add(
            "write", self._on_dualfisheye_output_toggle_changed
        )
        self.dualfisheye_vars["metadata_only"].trace_add(
            "write", self._on_dualfisheye_output_toggle_changed
        )
        self.dualfisheye_vars["camera_extrinsics_xml"].trace_add(
            "write", self._update_dualfisheye_derived_output_paths
        )
        self.dualfisheye_vars["perspective_size"].trace_add(
            "write", self._update_dualfisheye_metashape_f_display
        )
        self.dualfisheye_vars["perspective_focal_mm"].trace_add(
            "write", self._update_dualfisheye_metashape_f_display
        )
        for key in (
            "pairs_output",
            "fisheye_output",
            "perspective_output",
            "color_output",
        ):
            self.dualfisheye_vars[key].trace_add(
                "write",
                lambda *_args, output_key=key: self._on_dualfisheye_output_changed(
                    output_key
                ),
            )

        row = 0
        tk.Label(
            params,
            text="Step 1: Extract dual-fisheye X/Y image pairs from raw video with Video2Frames.",
            anchor="w",
            justify="left",
        ).grid(row=row, column=0, columnspan=3, sticky="w", padx=8, pady=(6, 6))

        row += 1
        tk.Label(
            params,
            text="Stage 2: Use FrameSelector to review the extracted pairs and choose the frames to calibrate.",
            anchor="w",
            justify="left",
        ).grid(row=row, column=0, columnspan=3, sticky="w", padx=8, pady=(2, 6))

        row += 1
        calibration_frame = tk.LabelFrame(
            params,
            text="Stage 3: Distortion Calibration / Perspective Output",
        )
        calibration_frame.grid(
            row=row,
            column=0,
            columnspan=3,
            sticky="we",
            padx=4,
            pady=(2, 4),
        )
        calibration_frame.grid_columnconfigure(1, weight=1)

        c_row = 0
        tk.Label(calibration_frame, text="Pair Folder").grid(
            row=c_row, column=0, sticky="e", padx=4, pady=4
        )
        tk.Entry(
            calibration_frame,
            textvariable=self.dualfisheye_vars["pair_input"],
            width=52,
        ).grid(row=c_row, column=1, sticky="we", padx=4, pady=4)
        tk.Button(
            calibration_frame,
            text="Browse...",
            command=lambda: self._select_directory(
                self.dualfisheye_vars["pair_input"],
                title="Select pair input folder",
            ),
        ).grid(row=c_row, column=2, padx=4, pady=4)

        c_row += 1
        tk.Label(calibration_frame, text="Mask Folder").grid(
            row=c_row, column=0, sticky="e", padx=4, pady=4
        )
        tk.Entry(
            calibration_frame,
            textvariable=self.dualfisheye_vars["mask_input"],
            width=52,
        ).grid(row=c_row, column=1, sticky="we", padx=4, pady=4)
        tk.Button(
            calibration_frame,
            text="Browse...",
            command=lambda: self._select_directory(
                self.dualfisheye_vars["mask_input"],
                title="Select pair mask folder",
            ),
        ).grid(row=c_row, column=2, padx=4, pady=4)

        c_row += 1
        calibration_input_frame = tk.LabelFrame(
            calibration_frame,
            text="Calibration",
        )
        calibration_input_frame.grid(
            row=c_row,
            column=0,
            columnspan=3,
            sticky="we",
            padx=4,
            pady=(2, 4),
        )
        calibration_input_frame.grid_columnconfigure(1, weight=1)
        tk.Label(
            calibration_input_frame,
            text="Fisheye Distortion XML (used only when Extrinsics XML is empty)",
        ).grid(
            row=0, column=0, sticky="e", padx=4, pady=4
        )
        tk.Entry(
            calibration_input_frame,
            textvariable=self.dualfisheye_vars["camera_xml"],
            width=52,
        ).grid(row=0, column=1, sticky="we", padx=4, pady=4)
        tk.Button(
            calibration_input_frame,
            text="Browse...",
            command=lambda: self._select_file(
                self.dualfisheye_vars["camera_xml"],
                title="Select camera calibration XML",
                filetypes=[("XML files", "*.xml"), ("All files", "*.*")],
            ),
        ).grid(row=0, column=2, padx=4, pady=4)
        tk.Label(calibration_input_frame, text="Input LUT").grid(
            row=1, column=0, sticky="e", padx=4, pady=4
        )
        tk.Entry(
            calibration_input_frame,
            textvariable=self.dualfisheye_vars["input_lut"],
            width=52,
        ).grid(row=1, column=1, sticky="we", padx=4, pady=4)
        tk.Button(
            calibration_input_frame,
            text="Browse...",
            command=lambda: self._select_file(
                self.dualfisheye_vars["input_lut"],
                title="Select input LUT",
                filetypes=[("Cube LUT", "*.cube"), ("All files", "*.*")],
            ),
        ).grid(row=1, column=2, padx=4, pady=4)
        tk.Label(
            calibration_input_frame,
            text="Perspective Camera Extrinsics XML",
        ).grid(
            row=2, column=0, sticky="e", padx=4, pady=4
        )
        tk.Entry(
            calibration_input_frame,
            textvariable=self.dualfisheye_vars["camera_extrinsics_xml"],
            width=52,
        ).grid(row=2, column=1, sticky="we", padx=4, pady=4)
        tk.Button(
            calibration_input_frame,
            text="Browse...",
            command=lambda: self._select_file(
                self.dualfisheye_vars["camera_extrinsics_xml"],
                title="Select perspective camera extrinsics XML",
                filetypes=[("XML files", "*.xml"), ("All files", "*.*")],
            ),
        ).grid(row=2, column=2, padx=4, pady=4)
        tk.Label(
            calibration_input_frame,
            text="Metashape PointCloud PLY (optional)",
        ).grid(
            row=3, column=0, sticky="e", padx=4, pady=4
        )
        tk.Entry(
            calibration_input_frame,
            textvariable=self.dualfisheye_vars["pointcloud_ply"],
            width=52,
        ).grid(row=3, column=1, sticky="we", padx=4, pady=4)
        tk.Button(
            calibration_input_frame,
            text="Browse...",
            command=lambda: self._select_file(
                self.dualfisheye_vars["pointcloud_ply"],
                title="Select Metashape point cloud PLY",
                filetypes=[("PLY files", "*.ply"), ("All files", "*.*")],
            ),
        ).grid(row=3, column=2, padx=4, pady=4)
        tk.Label(
            calibration_input_frame,
            text="TODO: add Chromatic Aberration Correction here later.",
            anchor="w",
            justify="left",
        ).grid(row=4, column=0, columnspan=3, sticky="w", padx=4, pady=(0, 4))

        c_row += 1
        options_frame = tk.LabelFrame(calibration_frame, text="Options")
        options_frame.grid(
            row=c_row,
            column=0,
            columnspan=3,
            sticky="we",
            padx=4,
            pady=(2, 4),
        )
        options_frame.grid_columnconfigure(0, weight=1)
        flags_row = tk.Frame(options_frame)
        flags_row.grid(row=0, column=0, sticky="w", padx=4, pady=4)
        tk.Checkbutton(
            flags_row,
            text="Use Input LUT",
            variable=self.dualfisheye_vars["use_input_lut"],
        ).pack(side=tk.LEFT, padx=(0, 12), pady=0)
        tk.Checkbutton(
            flags_row,
            text="Save Color Correct Only Output",
            variable=self.dualfisheye_vars["save_color_corrected_output"],
        ).pack(side=tk.LEFT, padx=(0, 12), pady=0)
        tk.Checkbutton(
            flags_row,
            text="Save undistorted fisheye output",
            variable=self.dualfisheye_vars["save_fisheye_output"],
        ).pack(side=tk.LEFT, padx=(0, 12), pady=0)
        tk.Checkbutton(
            flags_row,
            text="Disable perspective output",
            variable=self.dualfisheye_vars["no_perspective"],
        ).pack(side=tk.LEFT, padx=(0, 12), pady=0)
        tk.Checkbutton(
            flags_row,
            text="COLMAP + XML only",
            variable=self.dualfisheye_vars["metadata_only"],
        ).pack(side=tk.LEFT, padx=(0, 12), pady=0)
        tk.Checkbutton(
            flags_row,
            text="Calibration dry run",
            variable=self.dualfisheye_vars["dry_run"],
        ).pack(side=tk.LEFT, padx=(0, 6), pady=0)

        perspective_row = tk.Frame(options_frame)
        perspective_row.grid(row=1, column=0, sticky="w", padx=4, pady=4)
        tk.Label(perspective_row, text="Output color space").pack(
            side=tk.LEFT, padx=(0, 4)
        )
        ttk.Combobox(
            perspective_row,
            textvariable=self.dualfisheye_vars["lut_output_color_space"],
            values=("passthrough", "sRGB"),
            state="readonly",
            width=16,
        ).pack(side=tk.LEFT, padx=(0, 10))
        tk.Label(perspective_row, text="Ext").pack(
            side=tk.LEFT, padx=(0, 4)
        )
        ttk.Combobox(
            perspective_row,
            textvariable=self.dualfisheye_vars["perspective_ext"],
            values=("jpg", "png", "tif"),
            state="readonly",
            width=8,
        ).pack(side=tk.LEFT, padx=(0, 10))
        tk.Label(perspective_row, text="Mask Ext").pack(
            side=tk.LEFT, padx=(0, 4)
        )
        ttk.Combobox(
            perspective_row,
            textvariable=self.dualfisheye_vars["perspective_mask_ext"],
            values=("png", "jpg", "tif"),
            width=8,
        ).pack(side=tk.LEFT, padx=(0, 10))
        tk.Label(perspective_row, text="Size").pack(
            side=tk.LEFT, padx=(0, 4)
        )
        tk.Entry(
            perspective_row,
            textvariable=self.dualfisheye_vars["perspective_size"],
            width=8,
        ).pack(side=tk.LEFT, padx=(0, 10))
        tk.Label(perspective_row, text="Focal mm").pack(
            side=tk.LEFT, padx=(0, 4)
        )
        tk.Entry(
            perspective_row,
            textvariable=self.dualfisheye_vars["perspective_focal_mm"],
            width=8,
        ).pack(side=tk.LEFT, padx=(0, 10))
        tk.Label(
            perspective_row,
            textvariable=self.dualfisheye_metashape_f_var,
        ).pack(side=tk.LEFT, padx=(0, 10))

        worker_row = tk.Frame(options_frame)
        worker_row.grid(row=2, column=0, sticky="w", padx=4, pady=(0, 4))
        tk.Label(worker_row, text="Workers").pack(
            side=tk.LEFT, padx=(0, 4)
        )
        tk.Entry(
            worker_row,
            textvariable=self.dualfisheye_vars["workers"],
            width=8,
        ).pack(side=tk.LEFT, padx=(0, 10))
        tk.Label(
            worker_row,
            text="Memory throttle %",
        ).pack(side=tk.LEFT, padx=(0, 4))
        tk.Entry(
            worker_row,
            textvariable=self.dualfisheye_vars["memory_throttle_percent"],
            width=8,
        ).pack(side=tk.LEFT, padx=(0, 10))
        tk.Label(
            worker_row,
            text="(default: 80)",
        ).pack(side=tk.LEFT, padx=(0, 6))

        c_row += 1
        output_frame = tk.LabelFrame(
            calibration_frame,
            text="Calibration Outputs",
        )
        output_frame.grid(
            row=c_row,
            column=0,
            columnspan=3,
            sticky="we",
            padx=4,
            pady=(2, 4),
        )
        output_frame.grid_columnconfigure(1, weight=1)

        o_row = 0
        tk.Label(output_frame, text="Perspective / COLMAP Root").grid(
            row=o_row, column=0, sticky="e", padx=4, pady=4
        )
        self.dualfisheye_perspective_output_entry = tk.Entry(
            output_frame,
            textvariable=self.dualfisheye_vars["perspective_output"],
            width=52,
        )
        self.dualfisheye_perspective_output_entry.grid(
            row=o_row, column=1, sticky="we", padx=4, pady=4
        )
        self.dualfisheye_perspective_output_button = tk.Button(
            output_frame,
            text="Browse...",
            command=lambda: self._select_directory(
                self.dualfisheye_vars["perspective_output"],
                title="Select perspective output folder",
            ),
        )
        self.dualfisheye_perspective_output_button.grid(
            row=o_row, column=2, padx=4, pady=4
        )

        o_row += 1
        tk.Label(output_frame, text="Perspective XML").grid(
            row=o_row, column=0, sticky="e", padx=4, pady=4
        )
        tk.Entry(
            output_frame,
            textvariable=self.dualfisheye_xml_output_var,
            width=52,
            state="readonly",
        ).grid(row=o_row, column=1, sticky="we", padx=4, pady=4)
        tk.Label(output_frame, text="auto").grid(
            row=o_row, column=2, sticky="w", padx=4, pady=4
        )

        o_row += 1
        tk.Label(output_frame, text="COLMAP Images").grid(
            row=o_row, column=0, sticky="e", padx=4, pady=4
        )
        tk.Entry(
            output_frame,
            textvariable=self.dualfisheye_colmap_images_var,
            width=52,
            state="readonly",
        ).grid(row=o_row, column=1, sticky="we", padx=4, pady=4)
        tk.Label(output_frame, text="auto").grid(
            row=o_row, column=2, sticky="w", padx=4, pady=4
        )

        o_row += 1
        tk.Label(output_frame, text="COLMAP Masks").grid(
            row=o_row, column=0, sticky="e", padx=4, pady=4
        )
        tk.Entry(
            output_frame,
            textvariable=self.dualfisheye_colmap_masks_var,
            width=52,
            state="readonly",
        ).grid(row=o_row, column=1, sticky="we", padx=4, pady=4)
        tk.Label(output_frame, text="auto").grid(
            row=o_row, column=2, sticky="w", padx=4, pady=4
        )

        o_row += 1
        tk.Label(output_frame, text="COLMAP Sparse\\0").grid(
            row=o_row, column=0, sticky="e", padx=4, pady=4
        )
        tk.Entry(
            output_frame,
            textvariable=self.dualfisheye_colmap_sparse_var,
            width=52,
            state="readonly",
        ).grid(row=o_row, column=1, sticky="we", padx=4, pady=4)
        tk.Label(output_frame, text="auto").grid(
            row=o_row, column=2, sticky="w", padx=4, pady=4
        )

        o_row += 1
        tk.Label(output_frame, text="Color corrected only").grid(
            row=o_row, column=0, sticky="e", padx=4, pady=4
        )
        self.dualfisheye_color_output_entry = tk.Entry(
            output_frame,
            textvariable=self.dualfisheye_vars["color_output"],
            width=52,
        )
        self.dualfisheye_color_output_entry.grid(
            row=o_row, column=1, sticky="we", padx=4, pady=4
        )
        self.dualfisheye_color_output_button = tk.Button(
            output_frame,
            text="Browse...",
            command=lambda: self._select_directory(
                self.dualfisheye_vars["color_output"],
                title="Select color-corrected output folder",
            ),
        )
        self.dualfisheye_color_output_button.grid(
            row=o_row, column=2, padx=4, pady=4
        )

        o_row += 1
        tk.Label(output_frame, text="Undistorted fisheye").grid(
            row=o_row, column=0, sticky="e", padx=4, pady=4
        )
        self.dualfisheye_fisheye_output_entry = tk.Entry(
            output_frame,
            textvariable=self.dualfisheye_vars["fisheye_output"],
            width=52,
        )
        self.dualfisheye_fisheye_output_entry.grid(
            row=o_row, column=1, sticky="we", padx=4, pady=4
        )
        self.dualfisheye_fisheye_output_button = tk.Button(
            output_frame,
            text="Browse...",
            command=lambda: self._select_directory(
                self.dualfisheye_vars["fisheye_output"],
                title="Select undistorted fisheye output folder",
            ),
        )
        self.dualfisheye_fisheye_output_button.grid(
            row=o_row, column=2, padx=4, pady=4
        )

        params.grid_columnconfigure(1, weight=1)

        actions = tk.Frame(content)
        actions.pack(fill="x", padx=8, pady=(0, 8))
        self.dualfisheye_calibration_stop_button = tk.Button(
            actions,
            text="Stop Calibration",
            command=lambda: self._stop_cli_process("dualfisheye_calibration"),
        )
        self.dualfisheye_calibration_stop_button.pack(
            side=tk.RIGHT, padx=4, pady=4
        )
        self.dualfisheye_calibration_stop_button.configure(state="disabled")
        self.dualfisheye_calibration_run_button = tk.Button(
            actions,
            text="Run Distortion Calibration",
            command=self._run_dualfisheye_calibration_tool,
        )
        self.dualfisheye_calibration_run_button.pack(
            side=tk.RIGHT, padx=4, pady=4
        )

        log_frame = tk.LabelFrame(content, text="Log")
        log_frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.dualfisheye_log = tk.Text(
            log_frame,
            wrap="word",
            height=18,
            cursor="arrow",
        )
        self.dualfisheye_log.pack(fill="both", expand=True, padx=6, pady=4)
        self.dualfisheye_log.bind("<Key>", self._block_text_edit)
        self.dualfisheye_log.bind(
            "<Button-1>",
            lambda event: self.dualfisheye_log.focus_set(),
        )
        self._set_text_widget(self.dualfisheye_log, "")
        self._update_dualfisheye_default_paths(force=False)
        self._update_dualfisheye_output_controls_state()
        self._update_dualfisheye_metashape_f_display()
        self._update_dualfisheye_derived_output_paths()

    def _build_ply_tab(self, parent: tk.Widget) -> None:
        container = tk.Frame(parent, bg=self.APP_BG)
        container.pack(fill="both", expand=True)

        header = tk.Frame(container, bg=self.HEADER_BG, bd=1, relief=tk.FLAT)
        header.pack(fill="x", padx=6, pady=(6, 6))
        tk.Label(
            header,
            text="PointCloudOptimizer",
            bg=self.HEADER_BG,
            fg="#0f172a",
            font=("TkDefaultFont", 12, "bold"),
        ).pack(side=tk.LEFT, padx=(8, 10), pady=8)
        tk.Label(
            header,
            text="Optimize and preview point clouds with the same viewer workflow used for camera scenes.",
            bg=self.HEADER_BG,
            fg="#6b7280",
        ).pack(side=tk.LEFT, padx=(0, 10), pady=8)

        body_pane = tk.PanedWindow(
            container,
            orient=tk.HORIZONTAL,
            sashrelief=tk.RAISED,
            sashwidth=6,
            bd=0,
            opaqueresize=True,
        )
        body_pane.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        left_panel = tk.Frame(body_pane, bg=self.APP_BG, width=620)
        right_panel = tk.Frame(body_pane, bg=self.APP_BG)
        body_pane.add(left_panel, minsize=560, stretch="never")
        body_pane.add(right_panel, minsize=700, stretch="always")

        params = tk.LabelFrame(left_panel, text="Parameters")
        params.pack(fill="x", padx=0, pady=(0, 8))
        params.grid_columnconfigure(1, weight=1)
        params_label_width = 16
        params_path_button_width = 11

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
        for redraw_var in (
            self._ply_point_size_var,
            self._ply_grid_step_var,
            self._ply_grid_span_var,
        ):
            redraw_var.trace_add(
                "write",
                lambda *_args: self._redraw_ply_canvas(),
            )
        self._ply_target_var_map = {
            "points": self.ply_vars["target_points"],
            "percent": self.ply_vars["target_percent"],
            "voxel": self.ply_vars["voxel_size"],
        }

        row = 0
        tk.Label(
            params,
            text="Input Point Cloud",
            width=params_label_width,
            anchor="w",
        ).grid(row=row, column=0, sticky="w", padx=(8, 6), pady=(8, 4))
        tk.Entry(params, textvariable=self.ply_vars["input"], width=52).grid(
            row=row,
            column=1,
            sticky="we",
            padx=(0, 6),
            pady=(8, 4),
        )
        tk.Button(
            params,
            text="Ply File",
            width=params_path_button_width,
            command=lambda: self._select_file(
                self.ply_vars["input"],
                title="Select input PLY",
                filetypes=[("PLY files", "*.ply"), ("All files", "*.*")],
                on_select=self._on_ply_input_selected,
            ),
        ).grid(row=row, column=2, padx=(0, 6), pady=(8, 4))
        tk.Button(
            params,
            text="Colmap Folder",
            width=params_path_button_width,
            command=lambda: self._select_directory(
                self.ply_vars["input"],
                title="Select input COLMAP folder",
                on_select=self._on_ply_input_selected,
            ),
        ).grid(row=row, column=3, padx=(0, 8), pady=(8, 4))

        row += 1
        tk.Label(
            params,
            text="Output Point Cloud",
            width=params_label_width,
            anchor="w",
        ).grid(row=row, column=0, sticky="w", padx=(8, 6), pady=4)
        tk.Entry(params, textvariable=self.ply_vars["output"], width=52).grid(
            row=row,
            column=1,
            sticky="we",
            padx=(0, 6),
            pady=4,
        )
        tk.Button(
            params,
            text="Ply File",
            width=params_path_button_width,
            command=lambda: self._select_save_file(self.ply_vars["output"], title="Select output PLY", defaultextension=".ply", filetypes=[("PLY files", "*.ply"), ("All files", "*.*")]),
        ).grid(row=row, column=2, padx=(0, 6), pady=4)
        tk.Button(
            params,
            text="Colmap Folder",
            width=params_path_button_width,
            command=lambda: self._select_directory(
                self.ply_vars["output"],
                title="Select output COLMAP folder",
            ),
        ).grid(row=row, column=3, padx=(0, 8), pady=4)

        row += 1
        downsample_frame = tk.Frame(params, bg=self.APP_BG)
        downsample_frame.grid(
            row=row,
            column=0,
            columnspan=4,
            sticky="we",
            padx=(8, 8),
            pady=(6, 2),
        )
        downsample_frame.grid_columnconfigure(1, weight=1)
        downsample_frame.grid_columnconfigure(3, weight=1)
        downsample_frame.grid_columnconfigure(5, weight=1)
        tk.Label(downsample_frame, text="Target mode").grid(
            row=0, column=0, sticky="w", padx=(0, 4), pady=(0, 4)
        )
        mode_labels = tuple(self._ply_mode_key_map.keys())
        mode_combo = ttk.Combobox(
            downsample_frame,
            textvariable=self.ply_target_mode_var,
            values=mode_labels,
            state="readonly",
            width=16,
        )
        mode_combo.grid(row=0, column=1, sticky="we", padx=(0, 12), pady=(0, 4))
        mode_combo.bind("<<ComboboxSelected>>", self._on_ply_target_mode_changed)
        self._ply_target_value_label = tk.Label(downsample_frame, text="Points")
        self._ply_target_value_label.grid(
            row=0, column=2, sticky="w", padx=(0, 4), pady=(0, 4)
        )
        current_mode_key = self._ply_mode_key_map.get(self.ply_target_mode_var.get(), "points")
        current_var = self._ply_target_var_map.get(current_mode_key, self.ply_vars["target_points"])
        self._ply_target_value_entry = tk.Entry(
            downsample_frame,
            textvariable=current_var,
            width=10,
        )
        self._ply_target_value_entry.grid(
            row=0, column=3, sticky="we", padx=(0, 12), pady=(0, 4)
        )
        tk.Label(downsample_frame, text="Downsample").grid(
            row=1, column=0, sticky="w", padx=(0, 4)
        )
        method_labels = tuple(self._ply_downsample_method_key_map.keys())
        ttk.Combobox(
            downsample_frame,
            textvariable=self.ply_downsample_method_var,
            values=method_labels,
            state="readonly",
            width=18,
        ).grid(row=1, column=1, sticky="we", padx=(0, 12))
        tk.Label(downsample_frame, text="Weight").grid(
            row=1, column=2, sticky="w", padx=(0, 4)
        )
        self.ply_adaptive_weight_entry = tk.Entry(
            downsample_frame,
            textvariable=self.ply_vars["adaptive_weight"],
            width=8,
        )
        self.ply_adaptive_weight_entry.grid(
            row=1, column=3, sticky="w", padx=(0, 12)
        )
        tk.Label(downsample_frame, text="Keep strategy").grid(
            row=1, column=4, sticky="w", padx=(0, 4)
        )
        self.ply_keep_menu = ttk.Combobox(
            downsample_frame,
            textvariable=self.ply_vars["keep_strategy"],
            values=("centroid", "center", "first", "random"),
            state="readonly",
            width=12,
        )
        self.ply_keep_menu.grid(row=1, column=5, sticky="we", padx=(0, 4))
        self._update_ply_target_value_widgets()

        row += 1
        params_actions = tk.Frame(params, bg=self.APP_BG)
        params_actions.grid(
            row=row,
            column=0,
            columnspan=4,
            sticky="e",
            padx=(8, 8),
            pady=(2, 8),
        )
        self.ply_stop_button = tk.Button(
            params_actions,
            text="Stop",
            command=lambda: self._stop_cli_process("ply"),
        )
        self.ply_stop_button.pack(side=tk.RIGHT, padx=4, pady=4)
        self.ply_stop_button.configure(state="disabled")
        self.ply_run_button = tk.Button(
            params_actions,
            text="Run PointCloudOptimizer",
            command=self._run_ply_optimizer,
        )
        self.ply_run_button.pack(side=tk.RIGHT, padx=4, pady=4)

        viewer_tools_frame = tk.LabelFrame(left_panel, text="Viewer Tools")
        viewer_tools_frame.pack(fill="x", padx=0, pady=(0, 8))
        viewer_tools_actions = tk.Frame(viewer_tools_frame, bg=self.APP_BG)
        viewer_tools_actions.pack(fill="x", padx=12, pady=(6, 0))
        tk.Button(
            viewer_tools_actions,
            text="Reset All Edits",
            command=self._on_reset_ply_view_state,
        ).pack(side=tk.RIGHT)

        viewer_frame = tk.Frame(right_panel, bg=self.APP_BG, bd=1, relief=tk.GROOVE)
        viewer_frame.pack(fill="both", expand=True, padx=0, pady=0)
        self._ply_viewer_root = viewer_frame

        viewer_header = tk.Frame(viewer_frame, bg=self.HEADER_BG)
        viewer_header.pack(fill="x", padx=0, pady=0)
        tk.Label(
            viewer_header,
            text="Viewer",
            bg=self.HEADER_BG,
            fg="#0f172a",
            font=("TkDefaultFont", 12, "bold"),
        ).pack(side=tk.LEFT, padx=(12, 10), pady=10)
        tk.Label(
            viewer_header,
            text="Left drag = orbit, Right drag = pan, Wheel = zoom",
            bg=self.HEADER_BG,
            fg="#6b7280",
        ).pack(side=tk.LEFT, padx=(0, 12), pady=10)

        viewer_actions = tk.Frame(viewer_frame, bg=self.APP_BG)
        viewer_actions.pack(fill="x", padx=12, pady=(8, 2))
        self.ply_input_view_button = tk.Button(
            viewer_actions,
            text="Show Input",
            command=self._on_show_input_ply,
        )
        self.ply_input_view_button.pack(side=tk.LEFT, padx=(0, 4), pady=0)
        self.ply_view_button = tk.Button(
            viewer_actions,
            text="Show Output",
            command=self._on_show_ply,
        )
        self.ply_view_button.pack(side=tk.LEFT, padx=4, pady=0)
        self.ply_clear_view_button = tk.Button(
            viewer_actions,
            text="Clear",
            command=self._on_clear_ply_view,
        )
        self.ply_clear_view_button.pack(side=tk.LEFT, padx=(4, 8), pady=0)
        tk.Label(viewer_actions, text="Projection", bg=self.APP_BG).pack(
            side=tk.LEFT, padx=(8, 4)
        )
        self._ply_projection_combo = ttk.Combobox(
            viewer_actions,
            textvariable=self._ply_projection_mode,
            values=("Orthographic", "Perspective"),
            state="readonly",
            width=12,
        )
        self._ply_projection_combo.pack(side=tk.LEFT, padx=(0, 4))
        self._ply_projection_combo.bind("<<ComboboxSelected>>", self._on_ply_projection_changed)
        tk.Label(viewer_actions, text="Display Up", bg=self.APP_BG).pack(
            side=tk.LEFT, padx=(10, 4)
        )
        ply_display_up_combo = ttk.Combobox(
            viewer_actions,
            textvariable=self._ply_display_up_axis_var,
            values=("Z-up", "Y-down"),
            state="readonly",
            width=8,
        )
        ply_display_up_combo.pack(side=tk.LEFT, padx=(0, 4))
        ply_display_up_combo.bind(
            "<<ComboboxSelected>>",
            lambda _event: self._redraw_ply_canvas(),
        )
        tk.Label(viewer_actions, text="Interactive Points", bg=self.APP_BG).pack(
            side=tk.LEFT, padx=(10, 4)
        )
        interactive_max_points_entry = ttk.Combobox(
            viewer_actions,
            textvariable=self._ply_view_interactive_max_points_var,
            values=("10000", "100000", "1000000"),
            state="normal",
            width=8,
        )
        interactive_max_points_entry.pack(side=tk.LEFT)
        interactive_max_points_entry.bind(
            "<Return>", self._on_ply_interactive_max_points_commit
        )
        interactive_max_points_entry.bind(
            "<FocusOut>", self._on_ply_interactive_max_points_commit
        )
        self._ply_interactive_max_points_entry = interactive_max_points_entry
        tk.Label(viewer_actions, text="Final Points", bg=self.APP_BG).pack(
            side=tk.LEFT, padx=(10, 4)
        )
        high_max_points_entry = ttk.Combobox(
            viewer_actions,
            textvariable=self._ply_view_high_max_points_var,
            values=("50000", "500000", "5000000"),
            state="normal",
            width=8,
        )
        high_max_points_entry.pack(side=tk.LEFT)
        high_max_points_entry.bind(
            "<Return>", self._on_ply_high_max_points_commit
        )
        high_max_points_entry.bind(
            "<FocusOut>", self._on_ply_high_max_points_commit
        )
        self._ply_high_max_points_entry = high_max_points_entry

        viewer_controls_middle = tk.Frame(viewer_frame, bg=self.APP_BG)
        viewer_controls_middle.pack(fill="x", padx=12, pady=(0, 2))
        tk.Label(viewer_controls_middle, text="Point size", bg=self.APP_BG).pack(
            side=tk.LEFT, padx=(8, 4)
        )
        ttk.Combobox(
            viewer_controls_middle,
            textvariable=self._ply_point_size_var,
            values=("1", "2", "3", "5"),
            state="normal",
            width=5,
        ).pack(side=tk.LEFT)
        tk.Label(viewer_controls_middle, text="Grid step", bg=self.APP_BG).pack(
            side=tk.LEFT, padx=(10, 4)
        )
        ttk.Combobox(
            viewer_controls_middle,
            textvariable=self._ply_grid_step_var,
            values=("0.01", "1", "10", "100"),
            state="normal",
            width=7,
        ).pack(side=tk.LEFT)
        tk.Label(viewer_controls_middle, text="Grid span", bg=self.APP_BG).pack(
            side=tk.LEFT, padx=(10, 4)
        )
        ttk.Combobox(
            viewer_controls_middle,
            textvariable=self._ply_grid_span_var,
            values=("auto", "5", "10", "20", "50", "100", "200", "500"),
            state="normal",
            width=7,
        ).pack(side=tk.LEFT)
        tk.Checkbutton(
            viewer_controls_middle,
            text="Ground Grid",
            variable=self._ply_show_grid_var,
            command=self._redraw_ply_canvas,
            bg=self.APP_BG,
        ).pack(side=tk.LEFT, padx=(12, 0))
        tk.Checkbutton(
            viewer_controls_middle,
            text="World XYZ Axes",
            variable=self._ply_show_world_axes_var,
            command=self._redraw_ply_canvas,
            bg=self.APP_BG,
        ).pack(side=tk.LEFT, padx=(12, 0))

        viewer_controls_bottom = tk.Frame(viewer_frame, bg=self.APP_BG)
        viewer_controls_bottom.pack(fill="x", padx=12, pady=(0, 2))
        tk.Checkbutton(
            viewer_controls_bottom,
            text="Draw PointCloud",
            variable=self._ply_draw_points_var,
            command=self._redraw_ply_canvas,
            bg=self.APP_BG,
        ).pack(side=tk.LEFT, padx=(12, 0))
        tk.Checkbutton(
            viewer_controls_bottom,
            text="Depth View",
            variable=self._ply_monochrome_var,
            command=self._on_ply_depth_view_toggle,
            bg=self.APP_BG,
        ).pack(side=tk.LEFT, padx=(12, 0))
        ply_front_occlusion_cb = tk.Checkbutton(
            viewer_controls_bottom,
            text="Depth Occlusion",
            variable=self._ply_front_occlusion_var,
            command=self._on_ply_depth_occlusion_toggle,
            bg=self.APP_BG,
        )
        ply_front_occlusion_cb.pack(side=tk.LEFT, padx=(12, 0))
        self._ply_front_occlusion_checkbutton = ply_front_occlusion_cb
        self._enforce_ply_depth_view_constraints()

        viewer_controls_actions = tk.Frame(viewer_frame, bg=self.APP_BG)
        viewer_controls_actions.pack(fill="x", padx=12, pady=(0, 4))
        tk.Button(
            viewer_controls_actions,
            text="Reset View",
            command=self._on_reset_ply_camera_view,
        ).pack(side=tk.LEFT, padx=(8, 0))

        status_band = tk.Frame(viewer_frame, bg=self.APP_BG)
        status_band.pack(fill="x")
        tk.Label(
            status_band,
            textvariable=self._ply_view_info_var,
            anchor="w",
            bg=self.APP_BG,
            fg="#0f172a",
            font=("TkDefaultFont", 10, "bold"),
        ).pack(fill="x", padx=12, pady=(0, 6))

        color_label_width = 16
        pick_color_button_width = 15
        remove_color_controls = tk.LabelFrame(viewer_tools_frame, text="Remove Points")
        remove_color_controls.pack(fill="x", padx=12, pady=(6, 0))
        remove_color_controls.grid_columnconfigure(1, weight=1)
        remove_color_controls.grid_columnconfigure(2, weight=1)
        tk.Label(
            remove_color_controls,
            text="Remove color",
            width=color_label_width,
            anchor="w",
        ).grid(row=0, column=0, sticky="w", padx=(8, 4), pady=(6, 4))
        remove_color_display = tk.Label(
            remove_color_controls,
            textvariable=self._ply_remove_color_rgb_var,
            fg="#87cefa",
            anchor="w",
        )
        remove_color_display.grid(
            row=0, column=1, sticky="w", padx=(4, 8), pady=(6, 4)
        )
        self._ply_remove_color_label = remove_color_display
        self._update_remove_color_display(
            (135, 206, 250),
            self._ply_remove_color_var.get(),
        )
        remove_actions_top = tk.Frame(remove_color_controls, bg=self.APP_BG)
        remove_actions_top.grid(
            row=0,
            column=2,
            sticky="e",
            padx=(0, 8),
            pady=(6, 4),
        )
        tk.Button(
            remove_actions_top,
            text="Pick Remove Color",
            width=pick_color_button_width,
            command=self._on_pick_remove_color,
        ).pack(side=tk.RIGHT)
        tk.Label(remove_color_controls, text="Tol (RGB)").grid(
            row=1, column=0, sticky="w", padx=(8, 4), pady=(0, 6)
        )
        tk.Entry(
            remove_color_controls,
            textvariable=self._ply_remove_color_tol_var,
            width=6,
        ).grid(row=1, column=1, sticky="w", padx=(4, 8), pady=(0, 6))
        remove_actions_bottom = tk.Frame(remove_color_controls, bg=self.APP_BG)
        remove_actions_bottom.grid(
            row=1,
            column=2,
            sticky="e",
            padx=(0, 8),
            pady=(0, 6),
        )
        tk.Button(
            remove_actions_bottom,
            text="Remove Color Points",
            command=self._on_remove_color_points,
        ).pack(side=tk.LEFT)
        tk.Button(
            remove_actions_bottom,
            text="Reset Remove",
            command=self._on_reset_ply_remove_controls,
        ).pack(side=tk.LEFT, padx=(8, 0))

        sky_controls = tk.LabelFrame(viewer_tools_frame, text="Add Sky")
        sky_controls.pack(fill="x", padx=12, pady=(6, 0))
        sky_controls.grid_columnconfigure(1, weight=1)
        sky_controls.grid_columnconfigure(3, weight=1)
        tk.Label(
            sky_controls,
            text="Sky point cloud color",
            width=color_label_width,
            anchor="w",
        ).grid(row=0, column=0, sticky="w", padx=(8, 4), pady=(6, 4))
        sky_color_display = tk.Label(
            sky_controls,
            textvariable=self._ply_sky_color_rgb_var,
            fg="#87cefa",
            anchor="w",
        )
        sky_color_display.grid(row=0, column=1, sticky="w", padx=(4, 8), pady=(6, 4))
        self._ply_sky_color_label = sky_color_display
        self._update_sky_color_display((135, 206, 250), self._ply_sky_color_var.get())
        sky_actions_top = tk.Frame(sky_controls, bg=self.APP_BG)
        sky_actions_top.grid(
            row=0,
            column=2,
            columnspan=2,
            sticky="e",
            padx=(0, 8),
            pady=(6, 4),
        )
        tk.Button(
            sky_actions_top,
            text="Pick Sky Color",
            width=pick_color_button_width,
            command=self._on_pick_sky_color,
        ).pack(side=tk.LEFT)
        tk.Button(
            sky_actions_top,
            text="Auto Pick Sky Color",
            command=self._on_auto_pick_sky_color,
        ).pack(side=tk.LEFT, padx=(8, 0))

        axis_options = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")
        tk.Label(sky_controls, text="Sky axis").grid(
            row=1, column=0, sticky="w", padx=(8, 4), pady=(0, 4)
        )
        axis_combo = ttk.Combobox(
            sky_controls,
            textvariable=self._ply_sky_axis_var,
            values=axis_options,
            state="readonly",
            width=6,
        )
        axis_combo.grid(row=1, column=1, sticky="w", padx=(0, 12), pady=(0, 4))
        tk.Label(sky_controls, text="Sky scale").grid(
            row=1, column=2, sticky="w", padx=(0, 4), pady=(0, 4)
        )
        sky_scale_entry = tk.Entry(
            sky_controls,
            textvariable=self._ply_sky_scale_var,
            width=8,
        )
        sky_scale_entry.grid(row=1, column=3, sticky="w", padx=(0, 12), pady=(0, 4))
        tk.Label(sky_controls, text="Sky points").grid(
            row=2, column=0, sticky="w", padx=(8, 4), pady=(0, 6)
        )
        sky_count_entry = tk.Entry(
            sky_controls,
            textvariable=self._ply_sky_count_var,
            width=10,
        )
        sky_count_entry.grid(row=2, column=1, sticky="w", padx=(0, 12), pady=(0, 6))
        tk.Label(sky_controls, text="Sky sphere %").grid(
            row=2, column=2, sticky="w", padx=(0, 4), pady=(0, 6)
        )
        sky_percent_entry = tk.Entry(
            sky_controls,
            textvariable=self._ply_sky_percent_var,
            width=6,
        )
        sky_percent_entry.grid(row=2, column=3, sticky="w", padx=(0, 12), pady=(0, 6))
        sky_actions_bottom = tk.Frame(sky_controls, bg=self.APP_BG)
        sky_actions_bottom.grid(
            row=3,
            column=0,
            columnspan=4,
            sticky="e",
            padx=(8, 8),
            pady=(0, 6),
        )
        tk.Button(
            sky_actions_bottom,
            text="Add Sky Points (Preview)",
            command=self._on_add_sky_points,
        ).pack(side=tk.LEFT)
        tk.Button(
            sky_actions_bottom,
            text="Clear Sky",
            command=self._on_clear_sky_points,
        ).pack(side=tk.LEFT, padx=(8, 0))

        exp_controls = tk.LabelFrame(
            viewer_tools_frame,
            text="BBox Scatter (Experimental)",
        )
        exp_controls.pack(fill="x", padx=12, pady=(6, 0))
        exp_controls.grid_columnconfigure(0, weight=1)
        exp_controls.grid_columnconfigure(1, weight=1)
        bbox_entry_width = 6
        bbox_combo_width = 15

        center_wrap = tk.Frame(exp_controls, bg=self.APP_BG)
        center_wrap.grid(row=0, column=0, sticky="we", padx=(8, 6), pady=(6, 4))
        tk.Label(center_wrap, text="Center", anchor="w").pack(anchor="w")
        center_entry_frame = tk.Frame(center_wrap, bg=self.APP_BG)
        center_entry_frame.pack(anchor="w", pady=(2, 0))
        center_vars = (
            ("X", self._ply_exp_bbox_center_x_var),
            ("Y", self._ply_exp_bbox_center_y_var),
            ("Z", self._ply_exp_bbox_center_z_var),
        )
        for col_idx, (axis_label, axis_var) in enumerate(center_vars):
            tk.Label(center_entry_frame, text=axis_label).grid(
                row=0,
                column=col_idx * 2,
                sticky="w",
                padx=(0 if col_idx == 0 else 4, 2),
            )
            entry = tk.Entry(
                center_entry_frame,
                textvariable=axis_var,
                width=bbox_entry_width,
            )
            entry.grid(row=0, column=col_idx * 2 + 1, sticky="w", padx=(0, 4))
            entry.bind("<Return>", self._on_apply_ply_exp_bbox)
            entry.bind("<FocusOut>", self._on_ply_exp_bbox_focus_out)
        size_wrap = tk.Frame(exp_controls, bg=self.APP_BG)
        size_wrap.grid(row=0, column=1, sticky="we", padx=(6, 8), pady=(6, 4))
        tk.Label(size_wrap, text="Size", anchor="w").pack(anchor="w")
        size_entry_frame = tk.Frame(size_wrap, bg=self.APP_BG)
        size_entry_frame.pack(anchor="w", pady=(2, 0))
        size_vars = (
            ("X", self._ply_exp_bbox_size_x_var),
            ("Y", self._ply_exp_bbox_size_y_var),
            ("Z", self._ply_exp_bbox_size_z_var),
        )
        for col_idx, (axis_label, axis_var) in enumerate(size_vars):
            tk.Label(size_entry_frame, text=axis_label).grid(
                row=0,
                column=col_idx * 2,
                sticky="w",
                padx=(0 if col_idx == 0 else 4, 2),
            )
            entry = tk.Entry(
                size_entry_frame,
                textvariable=axis_var,
                width=bbox_entry_width,
            )
            entry.grid(row=0, column=col_idx * 2 + 1, sticky="w", padx=(0, 4))
            entry.bind("<Return>", self._on_apply_ply_exp_bbox)
            entry.bind("<FocusOut>", self._on_ply_exp_bbox_focus_out)

        edit_wrap = tk.Frame(exp_controls, bg=self.APP_BG)
        edit_wrap.grid(row=1, column=0, sticky="we", padx=(8, 6), pady=(0, 4))
        mode_row = tk.Frame(edit_wrap, bg=self.APP_BG)
        mode_row.pack(anchor="w")
        tk.Checkbutton(
            mode_row,
            text="Active",
            variable=self._ply_exp_bbox_active_var,
            command=self._redraw_ply_canvas,
        ).pack(side=tk.LEFT)
        tk.Label(mode_row, text="Edit mode").pack(side=tk.LEFT, padx=(12, 4))
        edit_mode_combo = ttk.Combobox(
            mode_row,
            textvariable=self._ply_exp_edit_mode_var,
            values=("Move", "Scale"),
            state="readonly",
            width=9,
        )
        edit_mode_combo.pack(side=tk.LEFT)
        edit_mode_combo.bind("<<ComboboxSelected>>", self._on_ply_exp_edit_mode_selected)

        scatter_wrap = tk.Frame(exp_controls, bg=self.APP_BG)
        scatter_wrap.grid(row=1, column=1, sticky="we", padx=(6, 8), pady=(0, 4))
        scatter_row = tk.Frame(scatter_wrap, bg=self.APP_BG)
        scatter_row.pack(anchor="w")
        tk.Label(scatter_row, text="Scatter mode").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Combobox(
            scatter_row,
            textvariable=self._ply_exp_mode_var,
            values=("Inside", "Outside"),
            state="readonly",
            width=11,
        ).pack(side=tk.LEFT)
        tk.Label(scatter_row, text="Count").pack(side=tk.LEFT, padx=(12, 4))
        tk.Entry(
            scatter_row,
            textvariable=self._ply_exp_point_count_var,
            width=9,
        ).pack(side=tk.LEFT)

        outer_wrap = tk.Frame(exp_controls, bg=self.APP_BG)
        outer_wrap.grid(row=2, column=0, sticky="we", padx=(8, 6), pady=(0, 4))
        tk.Label(outer_wrap, text="Outer distance x", anchor="w").pack(anchor="w")
        tk.Entry(
            outer_wrap,
            textvariable=self._ply_exp_outer_mult_var,
            width=8,
        ).pack(anchor="w", pady=(2, 0))

        color_wrap = tk.Frame(exp_controls, bg=self.APP_BG)
        color_wrap.grid(row=2, column=1, sticky="we", padx=(6, 8), pady=(0, 4))
        color_top = tk.Frame(color_wrap, bg=self.APP_BG)
        color_top.pack(fill="x")
        tk.Label(color_top, text="Color mode", anchor="w").pack(side=tk.LEFT)
        ttk.Combobox(
            color_top,
            textvariable=self._ply_exp_color_mode_var,
            values=("Random", "Edge Sample", "Main Sample"),
            state="readonly",
            width=14,
        ).pack(side=tk.LEFT, padx=(8, 0))
        color_bottom = tk.Frame(color_wrap, bg=self.APP_BG)
        color_bottom.pack(fill="x", pady=(2, 0))
        tk.Label(color_bottom, text="Types").pack(side=tk.LEFT)
        tk.Entry(
            color_bottom,
            textvariable=self._ply_exp_color_count_var,
            width=6,
        ).pack(side=tk.LEFT, padx=(8, 0))
        button_row = tk.Frame(exp_controls, bg=self.APP_BG)
        button_row.grid(
            row=3,
            column=0,
            columnspan=2,
            sticky="e",
            padx=(8, 8),
            pady=(2, 6),
        )
        tk.Button(
            button_row,
            text="Add Scatter Points",
            command=self._on_add_ply_exp_points,
        ).pack(side=tk.LEFT)
        tk.Button(
            button_row,
            text="Reset Scatter",
            command=self._on_reset_ply_exp_points,
        ).pack(side=tk.LEFT, padx=(8, 0))
        tk.Button(
            button_row,
            text="Reset BBox",
            command=self._on_reset_ply_exp_bbox,
        ).pack(side=tk.LEFT, padx=(8, 0))

        append_controls = tk.LabelFrame(viewer_tools_frame, text="Append PLY")
        append_controls.pack(fill="x", padx=12, pady=(6, 0))
        append_controls.grid_columnconfigure(1, weight=1)
        action_button_width = 11
        browse_button_width = 10
        tk.Label(append_controls, text="Files").grid(
            row=0, column=0, sticky="w", padx=(8, 4), pady=(6, 6)
        )
        append_entry = tk.Entry(
            append_controls,
            textvariable=self.ply_vars["append_ply"],
            width=44,
        )
        append_entry.grid(row=0, column=1, sticky="we", padx=(0, 4), pady=(6, 6))
        self.ply_append_entry = append_entry
        tk.Button(
            append_controls,
            text="Browse...",
            width=browse_button_width,
            command=self._browse_ply_append_files,
        ).grid(row=0, column=2, sticky="w", padx=(0, 4), pady=(6, 6))
        tk.Button(
            append_controls,
            text="Clear",
            width=8,
            command=self._clear_appended_ply_from_viewer,
        ).grid(row=0, column=3, sticky="w", padx=(0, 4), pady=(6, 6))
        tk.Button(
            append_controls,
            text="Append PLY",
            width=action_button_width,
            command=self._append_ply_files_to_viewer,
        ).grid(row=0, column=4, sticky="w", padx=(0, 8), pady=(6, 6))

        save_top_controls = tk.LabelFrame(viewer_tools_frame, text="Save Viewed PLY")
        save_top_controls.pack(fill="x", padx=12, pady=(6, 8))
        save_top_controls.grid_columnconfigure(1, weight=1)
        tk.Label(save_top_controls, text="Path").grid(
            row=0, column=0, sticky="w", padx=(8, 4), pady=(6, 6)
        )
        sky_save_entry = tk.Entry(
            save_top_controls,
            textvariable=self._ply_sky_save_path_var,
            width=44,
        )
        sky_save_entry.grid(row=0, column=1, sticky="we", padx=(0, 4), pady=(6, 6))
        tk.Button(
            save_top_controls,
            text="Browse...",
            width=browse_button_width,
            command=self._on_browse_sky_save_path,
        ).grid(row=0, column=2, sticky="w", padx=(0, 4), pady=(6, 6))
        tk.Button(
            save_top_controls,
            text="Save PLY",
            width=action_button_width,
            command=self._on_save_sky_points,
        ).grid(row=0, column=3, sticky="w", padx=(0, 8), pady=(6, 6))

        canvas_wrap = tk.Frame(viewer_frame, bg="#0f172a")
        canvas_wrap.pack(fill="both", expand=True, padx=12, pady=12)
        canvas = tk.Canvas(
            canvas_wrap,
            width=PLY_VIEW_CANVAS_WIDTH,
            height=400,
            bg="#101010",
            highlightthickness=0,
        )
        canvas.pack(fill="both", expand=True)
        self._ply_canvas_image_id = canvas.create_image(0, 0, anchor="nw")
        self._ply_view_canvas = canvas
        canvas.bind("<Configure>", self._on_ply_canvas_configure)
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
        canvas.bind("<KeyPress-w>", self._on_ply_exp_mode_move)
        canvas.bind("<KeyPress-W>", self._on_ply_exp_mode_move)
        canvas.bind("<KeyPress-e>", self._on_ply_exp_mode_scale)
        canvas.bind("<KeyPress-E>", self._on_ply_exp_mode_scale)

        log_frame = tk.LabelFrame(left_panel, text="Log")
        log_frame.pack(fill="both", expand=True, padx=0, pady=0)
        self.ply_log = tk.Text(log_frame, wrap="word", height=8, cursor="arrow")
        self.ply_log.pack(fill="both", expand=True, padx=6, pady=4)
        self.ply_log.bind("<Key>", self._block_text_edit)
        self.ply_log.bind("<Button-1>", lambda event: self.ply_log.focus_set())
        self._set_text_widget(self.ply_log, "")

        def _init_ply_sash() -> None:
            try:
                w = max(640, body_pane.winfo_width())
                body_pane.sash_place(0, int(w * 0.39), 1)
            except Exception:
                pass

        self.root.after_idle(_init_ply_sash)
        self._redraw_ply_canvas()
        self._update_ply_adaptive_state()

    def _build_camera_scene_tab(self, parent: tk.Widget) -> None:
        container = tk.Frame(parent, bg=self.APP_BG)
        container.pack(fill="both", expand=True)

        header = tk.Frame(container, bg=self.HEADER_BG, bd=1, relief=tk.FLAT)
        header.pack(fill="x", padx=6, pady=(6, 6))
        tk.Label(
            header,
            text="Camera Optimization (Experimental)",
            bg=self.HEADER_BG,
            fg="#0f172a",
            font=("TkDefaultFont", 12, "bold"),
        ).pack(side=tk.LEFT, padx=(8, 10), pady=8)
        tk.Label(
            header,
            text="Preview camera poses and point clouds while exporting to other formats.",
            bg=self.HEADER_BG,
            fg="#6b7280",
        ).pack(side=tk.LEFT, padx=(0, 10), pady=8)

        workspace = tk.PanedWindow(
            container,
            orient=tk.HORIZONTAL,
            sashrelief=tk.RAISED,
            sashwidth=8,
            bd=0,
            opaqueresize=True,
        )
        workspace.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        left_panel = tk.Frame(workspace, bg=self.APP_BG, width=620)
        right_panel = tk.Frame(workspace, bg=self.APP_BG)
        workspace.add(left_panel, minsize=520)
        workspace.add(right_panel, minsize=720)

        params = tk.LabelFrame(left_panel, text="Scene Input", bg="#f8fafc")
        params.pack(fill="x", padx=0, pady=(0, 8))

        self.camera_scene_vars = {
            "source_type": tk.StringVar(value=CAMERA_SCENE_SOURCE_CHOICES[0]),
            "colmap_dir": tk.StringVar(),
            "transforms_json": tk.StringVar(),
            "transforms_ply": tk.StringVar(),
            "csv_path": tk.StringVar(),
            "csv_ply": tk.StringVar(),
            "xmp_dir": tk.StringVar(),
            "xmp_ply": tk.StringVar(),
            "metashape_xml": tk.StringVar(),
            "metashape_ply": tk.StringVar(),
        }
        self.camera_converter_vars = {
            "output_dir": tk.StringVar(),
            "width": tk.StringVar(),
            "height": tk.StringVar(),
            "export_colmap": tk.BooleanVar(value=False),
            "export_csv": tk.BooleanVar(value=False),
            "export_ply": tk.BooleanVar(value=False),
            "export_transforms": tk.BooleanVar(value=False),
            "export_transforms_ply": tk.BooleanVar(value=False),
            "export_xmp": tk.BooleanVar(value=False),
            "export_metashape_xml": tk.BooleanVar(value=False),
            "camera_rot_x_deg": tk.StringVar(value="0"),
            "camera_rot_y_deg": tk.StringVar(value="0"),
            "camera_rot_z_deg": tk.StringVar(value="0"),
            "camera_scale": tk.StringVar(value="1.0"),
            "pointcloud_rot_x_deg": tk.StringVar(value="0"),
            "pointcloud_rot_y_deg": tk.StringVar(value="0"),
            "pointcloud_rot_z_deg": tk.StringVar(value="0"),
            "pointcloud_scale": tk.StringVar(value="1.0"),
            "link_transform": tk.BooleanVar(value=True),
        }
        self.camera_scene_vars["source_type"].trace_add(
            "write",
            lambda *_args: self._update_camera_scene_input_state(),
        )
        self.camera_converter_vars["link_transform"].trace_add(
            "write",
            self._on_camera_scene_link_transform_changed,
        )
        for trace_key in (
            "camera_rot_x_deg",
            "camera_rot_y_deg",
            "camera_rot_z_deg",
            "camera_scale",
        ):
            self.camera_converter_vars[trace_key].trace_add(
                "write",
                self._sync_camera_scene_linked_transform_vars,
            )
        for redraw_var in (
            self._camera_scene_point_size_var,
            self._camera_scene_camera_scale_var,
            self._camera_scene_camera_stride_var,
            self._camera_scene_grid_step_var,
            self._camera_scene_grid_span_var,
        ):
            redraw_var.trace_add(
                "write",
                lambda *_args: self._redraw_camera_scene_canvas(),
            )

        top_row = tk.Frame(params, bg="#f8fafc")
        top_row.pack(fill="x", padx=4, pady=4)
        tk.Label(top_row, text="Input type").pack(side=tk.LEFT, padx=(0, 8))
        ttk.Combobox(
            top_row,
            textvariable=self.camera_scene_vars["source_type"],
            values=CAMERA_SCENE_SOURCE_CHOICES,
            state="readonly",
            width=24,
        ).pack(side=tk.LEFT)

        group_holder = tk.Frame(params, bg="#f8fafc")
        group_holder.pack(fill="x", padx=4, pady=(0, 4))
        self._camera_scene_input_groups = {}

        colmap_group = tk.Frame(group_holder, bg="#f8fafc")
        colmap_group.columnconfigure(1, weight=1)
        self._camera_scene_input_groups["colmap"] = colmap_group
        tk.Label(colmap_group, text="COLMAP folder", bg="#f8fafc").grid(
            row=0, column=0, sticky="e", padx=4, pady=4
        )
        colmap_entry = tk.Entry(
            colmap_group,
            textvariable=self.camera_scene_vars["colmap_dir"],
            width=62,
        )
        colmap_entry.grid(row=0, column=1, sticky="we", padx=4, pady=4)
        self._camera_scene_colmap_entry = colmap_entry
        tk.Button(
            colmap_group,
            text="Browse...",
            command=lambda: self._select_directory(
                self.camera_scene_vars["colmap_dir"],
                title="Select COLMAP text-model folder",
            ),
        ).grid(row=0, column=2, padx=4, pady=4)

        transforms_group = tk.Frame(group_holder, bg="#f8fafc")
        transforms_group.columnconfigure(0, weight=1, uniform="camera_scene_input_pair")
        transforms_group.columnconfigure(1, weight=1, uniform="camera_scene_input_pair")
        self._camera_scene_input_groups["transforms"] = transforms_group
        transforms_json_block = tk.Frame(transforms_group, bg="#f8fafc")
        transforms_json_block.grid(row=0, column=0, sticky="we", padx=(0, 6), pady=0)
        transforms_json_block.columnconfigure(0, weight=1)
        tk.Label(
            transforms_json_block,
            text="transforms.json",
            bg="#f8fafc",
        ).grid(row=0, column=0, sticky="w", padx=4, pady=(2, 2))
        transforms_json_row = tk.Frame(transforms_json_block, bg="#f8fafc")
        transforms_json_row.grid(row=1, column=0, sticky="we", padx=4, pady=(0, 4))
        transforms_json_row.columnconfigure(0, weight=1)
        transforms_entry = tk.Entry(
            transforms_json_row,
            textvariable=self.camera_scene_vars["transforms_json"],
            width=32,
        )
        transforms_entry.grid(row=0, column=0, sticky="we")
        self._camera_scene_transforms_entry = transforms_entry
        tk.Button(
            transforms_json_row,
            text="Browse...",
            command=lambda: self._select_file(
                self.camera_scene_vars["transforms_json"],
                title="Select transforms.json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            ),
        ).grid(row=0, column=1, padx=(6, 0), pady=0)

        transforms_ply_block = tk.Frame(transforms_group, bg="#f8fafc")
        transforms_ply_block.grid(row=0, column=1, sticky="we", padx=(6, 0), pady=0)
        transforms_ply_block.columnconfigure(0, weight=1)
        tk.Label(
            transforms_ply_block,
            text="transforms PLY",
            bg="#f8fafc",
        ).grid(row=0, column=0, sticky="w", padx=4, pady=(2, 2))
        transforms_ply_row = tk.Frame(transforms_ply_block, bg="#f8fafc")
        transforms_ply_row.grid(row=1, column=0, sticky="we", padx=4, pady=(0, 4))
        transforms_ply_row.columnconfigure(0, weight=1)
        transforms_ply_entry = tk.Entry(
            transforms_ply_row,
            textvariable=self.camera_scene_vars["transforms_ply"],
            width=32,
        )
        transforms_ply_entry.grid(row=0, column=0, sticky="we")
        self._camera_scene_transforms_ply_entry = transforms_ply_entry
        tk.Button(
            transforms_ply_row,
            text="Browse...",
            command=lambda: self._select_file(
                self.camera_scene_vars["transforms_ply"],
                title="Select transforms PLY",
                filetypes=[("PLY files", "*.ply"), ("All files", "*.*")],
            ),
        ).grid(row=0, column=1, padx=(6, 0), pady=0)

        realityscan_group = tk.Frame(group_holder, bg="#f8fafc")
        realityscan_group.columnconfigure(0, weight=1, uniform="camera_scene_input_pair")
        realityscan_group.columnconfigure(1, weight=1, uniform="camera_scene_input_pair")
        self._camera_scene_input_groups["realityscan"] = realityscan_group
        realityscan_csv_block = tk.Frame(realityscan_group, bg="#f8fafc")
        realityscan_csv_block.grid(row=0, column=0, sticky="we", padx=(0, 6), pady=0)
        realityscan_csv_block.columnconfigure(0, weight=1)
        tk.Label(
            realityscan_csv_block,
            text="RealityScan CSV",
            bg="#f8fafc",
        ).grid(row=0, column=0, sticky="w", padx=4, pady=(2, 2))
        realityscan_csv_row = tk.Frame(realityscan_csv_block, bg="#f8fafc")
        realityscan_csv_row.grid(row=1, column=0, sticky="we", padx=4, pady=(0, 4))
        realityscan_csv_row.columnconfigure(0, weight=1)
        csv_entry = tk.Entry(
            realityscan_csv_row,
            textvariable=self.camera_scene_vars["csv_path"],
            width=32,
        )
        csv_entry.grid(row=0, column=0, sticky="we")
        self._camera_scene_csv_entry = csv_entry
        tk.Button(
            realityscan_csv_row,
            text="Browse...",
            command=lambda: self._select_file(
                self.camera_scene_vars["csv_path"],
                title="Select RealityScan CSV",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            ),
        ).grid(row=0, column=1, padx=(6, 0), pady=0)

        realityscan_ply_block = tk.Frame(realityscan_group, bg="#f8fafc")
        realityscan_ply_block.grid(row=0, column=1, sticky="we", padx=(6, 0), pady=0)
        realityscan_ply_block.columnconfigure(0, weight=1)
        tk.Label(
            realityscan_ply_block,
            text="RealityScan PLY",
            bg="#f8fafc",
        ).grid(row=0, column=0, sticky="w", padx=4, pady=(2, 2))
        realityscan_ply_row = tk.Frame(realityscan_ply_block, bg="#f8fafc")
        realityscan_ply_row.grid(row=1, column=0, sticky="we", padx=4, pady=(0, 4))
        realityscan_ply_row.columnconfigure(0, weight=1)
        csv_ply_entry = tk.Entry(
            realityscan_ply_row,
            textvariable=self.camera_scene_vars["csv_ply"],
            width=32,
        )
        csv_ply_entry.grid(row=0, column=0, sticky="we")
        self._camera_scene_csv_ply_entry = csv_ply_entry
        tk.Button(
            realityscan_ply_row,
            text="Browse...",
            command=lambda: self._select_file(
                self.camera_scene_vars["csv_ply"],
                title="Select RealityScan PLY",
                filetypes=[("PLY files", "*.ply"), ("All files", "*.*")],
            ),
        ).grid(row=0, column=1, padx=(6, 0), pady=0)

        xmp_group = tk.Frame(group_holder, bg="#f8fafc")
        xmp_group.columnconfigure(0, weight=1, uniform="camera_scene_input_pair")
        xmp_group.columnconfigure(1, weight=1, uniform="camera_scene_input_pair")
        self._camera_scene_input_groups["xmp"] = xmp_group
        xmp_dir_block = tk.Frame(xmp_group, bg="#f8fafc")
        xmp_dir_block.grid(row=0, column=0, sticky="we", padx=(0, 6), pady=0)
        xmp_dir_block.columnconfigure(0, weight=1)
        tk.Label(
            xmp_dir_block,
            text="RealityScan XMP dir",
            bg="#f8fafc",
        ).grid(row=0, column=0, sticky="w", padx=4, pady=(2, 2))
        xmp_dir_row = tk.Frame(xmp_dir_block, bg="#f8fafc")
        xmp_dir_row.grid(row=1, column=0, sticky="we", padx=4, pady=(0, 4))
        xmp_dir_row.columnconfigure(0, weight=1)
        xmp_entry = tk.Entry(
            xmp_dir_row,
            textvariable=self.camera_scene_vars["xmp_dir"],
            width=32,
        )
        xmp_entry.grid(row=0, column=0, sticky="we")
        self._camera_scene_xmp_entry = xmp_entry
        tk.Button(
            xmp_dir_row,
            text="Browse...",
            command=lambda: self._select_directory(
                self.camera_scene_vars["xmp_dir"],
                title="Select RealityScan XMP directory",
            ),
        ).grid(row=0, column=1, padx=(6, 0), pady=0)

        xmp_ply_block = tk.Frame(xmp_group, bg="#f8fafc")
        xmp_ply_block.grid(row=0, column=1, sticky="we", padx=(6, 0), pady=0)
        xmp_ply_block.columnconfigure(0, weight=1)
        tk.Label(
            xmp_ply_block,
            text="Optional PLY",
            bg="#f8fafc",
        ).grid(row=0, column=0, sticky="w", padx=4, pady=(2, 2))
        xmp_ply_row = tk.Frame(xmp_ply_block, bg="#f8fafc")
        xmp_ply_row.grid(row=1, column=0, sticky="we", padx=4, pady=(0, 4))
        xmp_ply_row.columnconfigure(0, weight=1)
        xmp_ply_entry = tk.Entry(
            xmp_ply_row,
            textvariable=self.camera_scene_vars["xmp_ply"],
            width=32,
        )
        xmp_ply_entry.grid(row=0, column=0, sticky="we")
        self._camera_scene_xmp_ply_entry = xmp_ply_entry
        tk.Button(
            xmp_ply_row,
            text="Browse...",
            command=lambda: self._select_file(
                self.camera_scene_vars["xmp_ply"],
                title="Select optional PLY",
                filetypes=[("PLY files", "*.ply"), ("All files", "*.*")],
            ),
        ).grid(row=0, column=1, padx=(6, 0), pady=0)

        metashape_group = tk.Frame(group_holder, bg="#f8fafc")
        metashape_group.columnconfigure(0, weight=1, uniform="camera_scene_input_pair")
        metashape_group.columnconfigure(1, weight=1, uniform="camera_scene_input_pair")
        self._camera_scene_input_groups["metashape"] = metashape_group
        metashape_xml_block = tk.Frame(metashape_group, bg="#f8fafc")
        metashape_xml_block.grid(row=0, column=0, sticky="we", padx=(0, 6), pady=0)
        metashape_xml_block.columnconfigure(0, weight=1)
        tk.Label(
            metashape_xml_block,
            text="Metashape XML",
            bg="#f8fafc",
        ).grid(row=0, column=0, sticky="w", padx=4, pady=(2, 2))
        metashape_xml_row = tk.Frame(metashape_xml_block, bg="#f8fafc")
        metashape_xml_row.grid(row=1, column=0, sticky="we", padx=4, pady=(0, 4))
        metashape_xml_row.columnconfigure(0, weight=1)
        metashape_xml_entry = tk.Entry(
            metashape_xml_row,
            textvariable=self.camera_scene_vars["metashape_xml"],
            width=32,
        )
        metashape_xml_entry.grid(row=0, column=0, sticky="we")
        self._camera_scene_metashape_xml_entry = metashape_xml_entry
        tk.Button(
            metashape_xml_row,
            text="Browse...",
            command=lambda: self._select_file(
                self.camera_scene_vars["metashape_xml"],
                title="Select Metashape XML",
                filetypes=[("XML files", "*.xml"), ("All files", "*.*")],
            ),
        ).grid(row=0, column=1, padx=(6, 0), pady=0)

        metashape_ply_block = tk.Frame(metashape_group, bg="#f8fafc")
        metashape_ply_block.grid(row=0, column=1, sticky="we", padx=(6, 0), pady=0)
        metashape_ply_block.columnconfigure(0, weight=1)
        tk.Label(
            metashape_ply_block,
            text="Optional PLY",
            bg="#f8fafc",
        ).grid(row=0, column=0, sticky="w", padx=4, pady=(2, 2))
        metashape_ply_row = tk.Frame(metashape_ply_block, bg="#f8fafc")
        metashape_ply_row.grid(row=1, column=0, sticky="we", padx=4, pady=(0, 4))
        metashape_ply_row.columnconfigure(0, weight=1)
        metashape_ply_entry = tk.Entry(
            metashape_ply_row,
            textvariable=self.camera_scene_vars["metashape_ply"],
            width=32,
        )
        metashape_ply_entry.grid(row=0, column=0, sticky="we")
        self._camera_scene_metashape_ply_entry = metashape_ply_entry
        tk.Button(
            metashape_ply_row,
            text="Browse...",
            command=lambda: self._select_file(
                self.camera_scene_vars["metashape_ply"],
                title="Select optional PLY",
                filetypes=[("PLY files", "*.ply"), ("All files", "*.*")],
            ),
        ).grid(row=0, column=1, padx=(6, 0), pady=0)

        actions = tk.Frame(params, bg="#f8fafc")
        actions.pack(fill="x", padx=4, pady=(6, 0))
        self.camera_scene_load_button = tk.Button(
            actions,
            text="Load Camera View",
            command=self._load_camera_scene,
        )
        self.camera_scene_load_button.pack(side=tk.RIGHT, padx=4, pady=4)
        self.camera_scene_clear_button = tk.Button(
            actions,
            text="Clear",
            command=self._clear_camera_scene,
        )
        self.camera_scene_clear_button.pack(side=tk.RIGHT, padx=4, pady=4)

        converter_frame = tk.LabelFrame(left_panel, text="Converter", bg=self.APP_BG)
        converter_frame.pack(fill="x", padx=0, pady=(0, 8))
        converter_frame.columnconfigure(1, weight=1)

        c_row = 0
        tk.Label(converter_frame, text="Output root").grid(
            row=c_row, column=0, sticky="e", padx=4, pady=4
        )
        tk.Entry(
            converter_frame,
            textvariable=self.camera_converter_vars["output_dir"],
            width=64,
        ).grid(row=c_row, column=1, sticky="we", padx=4, pady=4)
        tk.Button(
            converter_frame,
            text="Browse...",
            command=lambda: self._select_directory(
                self.camera_converter_vars["output_dir"],
                title="Select converter output directory",
            ),
        ).grid(row=c_row, column=2, padx=4, pady=4)

        c_row += 1
        size_row = tk.Frame(converter_frame, bg="#f8fafc")
        size_row.grid(row=c_row, column=0, columnspan=3, sticky="we", padx=4, pady=4)
        tk.Label(size_row, text="Width", bg="#f8fafc").pack(side=tk.LEFT, padx=(0, 4))
        tk.Entry(
            size_row,
            textvariable=self.camera_converter_vars["width"],
            width=8,
        ).pack(side=tk.LEFT)
        tk.Label(size_row, text="Height", bg="#f8fafc").pack(side=tk.LEFT, padx=(12, 4))
        tk.Entry(
            size_row,
            textvariable=self.camera_converter_vars["height"],
            width=8,
        ).pack(side=tk.LEFT)
        tk.Label(
            size_row,
            text="Used by RS CSV / XMP when needed",
            bg="#f8fafc",
            fg="#666666",
        ).pack(side=tk.LEFT, padx=(12, 0))

        c_row += 1
        export_row = tk.Frame(converter_frame, bg="#f8fafc")
        export_row.grid(row=c_row, column=0, columnspan=3, sticky="we", padx=4, pady=4)
        export_row.columnconfigure(1, weight=1)
        tk.Label(export_row, text="Exports", bg="#f8fafc").grid(
            row=0, column=0, sticky="nw", padx=(0, 12), pady=(2, 0)
        )
        export_grid = tk.Frame(export_row, bg="#f8fafc")
        export_grid.grid(row=0, column=1, sticky="we")
        export_grid.columnconfigure(0, weight=1)
        export_grid.columnconfigure(1, weight=1)
        export_rows = (
            (("export_colmap", "COLMAP"),),
            (("export_csv", "RealityScan CSV"), ("export_ply", "RealityScan PLY")),
            (
                ("export_transforms", "transforms.json"),
                ("export_transforms_ply", "transforms PLY"),
            ),
            (
                ("export_metashape_xml", "Metashape XML"),
                ("export_xmp", "RealityScan XMP"),
            ),
        )
        for export_row_index, export_items in enumerate(export_rows):
            for export_col_index in range(2):
                export_grid.columnconfigure(export_col_index, weight=1)
            for export_col_index, (export_key, export_label) in enumerate(export_items):
                tk.Checkbutton(
                    export_grid,
                    text=export_label,
                    variable=self.camera_converter_vars[export_key],
                    bg="#f8fafc",
                    anchor="w",
                ).grid(
                    row=export_row_index,
                    column=export_col_index,
                    sticky="w",
                    padx=(0, 18),
                    pady=1,
                )

        c_row += 1
        link_row = tk.Frame(converter_frame, bg="#f8fafc")
        link_row.grid(row=c_row, column=0, columnspan=3, sticky="we", padx=4, pady=(2, 4))
        tk.Checkbutton(
            link_row,
            text="Link camera / pointcloud transform",
            variable=self.camera_converter_vars["link_transform"],
            bg="#f8fafc",
            anchor="w",
        ).pack(side=tk.LEFT)
        tk.Label(
            link_row,
            text="Default workflow: move both together",
            bg="#f8fafc",
            fg="#64748b",
        ).pack(side=tk.LEFT, padx=(10, 0))

        c_row += 1
        camera_rot_row = tk.Frame(converter_frame, bg="#f8fafc")
        camera_rot_row.grid(row=c_row, column=0, columnspan=3, sticky="we", padx=4, pady=4)
        tk.Label(camera_rot_row, text="Camera rot XYZ deg", bg="#f8fafc").pack(side=tk.LEFT, padx=(0, 8))
        tk.Entry(
            camera_rot_row,
            textvariable=self.camera_converter_vars["camera_rot_x_deg"],
            width=7,
        ).pack(side=tk.LEFT)
        tk.Entry(
            camera_rot_row,
            textvariable=self.camera_converter_vars["camera_rot_y_deg"],
            width=7,
        ).pack(side=tk.LEFT, padx=(4, 0))
        tk.Entry(
            camera_rot_row,
            textvariable=self.camera_converter_vars["camera_rot_z_deg"],
            width=7,
        ).pack(side=tk.LEFT, padx=(4, 0))

        c_row += 1
        camera_scale_row = tk.Frame(converter_frame, bg="#f8fafc")
        camera_scale_row.grid(row=c_row, column=0, columnspan=3, sticky="we", padx=4, pady=4)
        tk.Label(camera_scale_row, text="Camera scale", bg="#f8fafc").pack(side=tk.LEFT, padx=(0, 8))
        tk.Entry(
            camera_scale_row,
            textvariable=self.camera_converter_vars["camera_scale"],
            width=10,
        ).pack(side=tk.LEFT)
        tk.Label(
            camera_scale_row,
            text="1.0 keeps original world scale",
            bg="#f8fafc",
            fg="#64748b",
        ).pack(side=tk.LEFT, padx=(10, 0))

        c_row += 1
        point_rot_row = tk.Frame(converter_frame, bg="#f8fafc")
        point_rot_row.grid(row=c_row, column=0, columnspan=3, sticky="we", padx=4, pady=4)
        tk.Label(point_rot_row, text="PointCloud rot XYZ deg", bg="#f8fafc").pack(side=tk.LEFT, padx=(0, 8))
        point_rot_x_entry = tk.Entry(
            point_rot_row,
            textvariable=self.camera_converter_vars["pointcloud_rot_x_deg"],
            width=7,
        )
        point_rot_x_entry.pack(side=tk.LEFT)
        point_rot_y_entry = tk.Entry(
            point_rot_row,
            textvariable=self.camera_converter_vars["pointcloud_rot_y_deg"],
            width=7,
        )
        point_rot_y_entry.pack(side=tk.LEFT, padx=(4, 0))
        point_rot_z_entry = tk.Entry(
            point_rot_row,
            textvariable=self.camera_converter_vars["pointcloud_rot_z_deg"],
            width=7,
        )
        point_rot_z_entry.pack(side=tk.LEFT, padx=(4, 0))
        self._camera_scene_point_transform_widgets.extend(
            [point_rot_x_entry, point_rot_y_entry, point_rot_z_entry]
        )

        c_row += 1
        point_scale_row = tk.Frame(converter_frame, bg="#f8fafc")
        point_scale_row.grid(row=c_row, column=0, columnspan=3, sticky="we", padx=4, pady=4)
        tk.Label(point_scale_row, text="PointCloud scale", bg="#f8fafc").pack(side=tk.LEFT, padx=(0, 8))
        point_scale_entry = tk.Entry(
            point_scale_row,
            textvariable=self.camera_converter_vars["pointcloud_scale"],
            width=10,
        )
        point_scale_entry.pack(side=tk.LEFT)
        self._camera_scene_point_transform_widgets.append(point_scale_entry)

        c_row += 1
        converter_actions = tk.Frame(converter_frame, bg="#f8fafc")
        converter_actions.grid(
            row=c_row, column=0, columnspan=3, sticky="we", padx=4, pady=(4, 6)
        )
        self.camera_scene_preview_apply_button = tk.Button(
            converter_actions,
            text="Apply To Preview",
            command=self._apply_camera_scene_preview_transform,
        )
        self.camera_scene_preview_apply_button.pack(side=tk.LEFT, padx=4, pady=4)
        self.camera_converter_stop_button = tk.Button(
            converter_actions,
            text="Stop",
            command=lambda: self._stop_cli_process("camera_converter"),
        )
        self.camera_converter_stop_button.pack(side=tk.RIGHT, padx=4, pady=4)
        self.camera_converter_stop_button.configure(state="disabled")
        self.camera_converter_run_button = tk.Button(
            converter_actions,
            text="Run CameraFormatConverter",
            command=self._run_camera_format_converter,
        )
        self.camera_converter_run_button.pack(side=tk.RIGHT, padx=4, pady=4)
        self._sync_camera_scene_transform_link_state()

        log_frame = tk.LabelFrame(left_panel, text="Converter Log", bg="#f8fafc")
        log_frame.pack(fill="both", expand=True, padx=0, pady=(0, 8))
        self.camera_converter_log = tk.Text(
            log_frame,
            wrap="word",
            height=10,
            cursor="arrow",
        )
        self.camera_converter_log.pack(fill="both", expand=True, padx=6, pady=4)
        self.camera_converter_log.bind("<Key>", self._block_text_edit)
        self.camera_converter_log.bind(
            "<Button-1>",
            lambda event: self.camera_converter_log.focus_set(),
        )
        self._set_text_widget(self.camera_converter_log, "")

        viewer_frame = tk.Frame(right_panel, bg=self.APP_BG)
        viewer_frame.pack(fill="both", expand=True, padx=0, pady=0)

        viewer_header = tk.Frame(viewer_frame, bg=self.HEADER_BG, bd=1, relief=tk.FLAT)
        viewer_header.pack(fill="x")
        tk.Label(
            viewer_header,
            text="Viewer",
            bg=self.HEADER_BG,
            fg="#0f172a",
            font=("TkDefaultFont", 11, "bold"),
        ).pack(side=tk.LEFT, padx=(12, 10), pady=8)
        tk.Label(
            viewer_header,
            text="Left drag = orbit, Right drag = pan, Wheel = zoom",
            bg=self.HEADER_BG,
            fg="#6b7280",
        ).pack(side=tk.LEFT, pady=8)

        viewer_controls_shell = tk.Frame(viewer_frame, bg=self.APP_BG, bd=1, relief=tk.FLAT)
        viewer_controls_shell.pack(fill="x")

        viewer_controls_top = tk.Frame(viewer_controls_shell, bg=self.APP_BG)
        viewer_controls_top.pack(fill="x", padx=12, pady=(8, 2))
        tk.Label(viewer_controls_top, text="Projection", bg=self.APP_BG).pack(side=tk.LEFT, padx=(0, 4))
        projection_combo = ttk.Combobox(
            viewer_controls_top,
            textvariable=self._camera_scene_projection_mode,
            values=("Orthographic", "Perspective"),
            state="readonly",
            width=14,
        )
        projection_combo.pack(side=tk.LEFT, padx=(0, 8))
        projection_combo.bind(
            "<<ComboboxSelected>>",
            lambda _event: self._redraw_camera_scene_canvas(),
        )
        self._camera_scene_projection_combo = projection_combo
        tk.Label(viewer_controls_top, text="Display Up", bg=self.APP_BG).pack(
            side=tk.LEFT, padx=(8, 4)
        )
        camera_display_up_combo = ttk.Combobox(
            viewer_controls_top,
            textvariable=self._camera_scene_display_up_axis_var,
            values=("Z-up", "Y-down"),
            state="readonly",
            width=8,
        )
        camera_display_up_combo.pack(side=tk.LEFT, padx=(0, 8))
        camera_display_up_combo.bind(
            "<<ComboboxSelected>>",
            lambda _event: self._redraw_camera_scene_canvas(),
        )
        tk.Label(viewer_controls_top, text="Interactive Points", bg=self.APP_BG).pack(
            side=tk.LEFT, padx=(8, 4)
        )
        ttk.Combobox(
            viewer_controls_top,
            textvariable=self._camera_scene_interactive_point_cap_var,
            values=("10000", "100000", "1000000"),
            state="normal",
            width=9,
        ).pack(side=tk.LEFT)
        tk.Label(viewer_controls_top, text="Final Points", bg=self.APP_BG).pack(
            side=tk.LEFT, padx=(8, 4)
        )
        ttk.Combobox(
            viewer_controls_top,
            textvariable=self._camera_scene_point_cap_var,
            values=("50000", "500000", "5000000"),
            state="normal",
            width=9,
        ).pack(side=tk.LEFT)

        viewer_controls_middle = tk.Frame(viewer_controls_shell, bg=self.APP_BG)
        viewer_controls_middle.pack(fill="x", padx=12, pady=(0, 2))
        tk.Label(viewer_controls_middle, text="Point size", bg=self.APP_BG).pack(
            side=tk.LEFT, padx=(8, 4)
        )
        ttk.Combobox(
            viewer_controls_middle,
            textvariable=self._camera_scene_point_size_var,
            values=("1", "2", "3", "5"),
            state="normal",
            width=5,
        ).pack(side=tk.LEFT)
        tk.Label(viewer_controls_middle, text="Camera scale", bg=self.APP_BG).pack(
            side=tk.LEFT, padx=(8, 4)
        )
        ttk.Combobox(
            viewer_controls_middle,
            textvariable=self._camera_scene_camera_scale_var,
            values=("0.01", "0.1", "1", "10", "100"),
            state="normal",
            width=8,
        ).pack(side=tk.LEFT)
        tk.Label(viewer_controls_middle, text="Camera stride", bg=self.APP_BG).pack(
            side=tk.LEFT, padx=(8, 4)
        )
        ttk.Combobox(
            viewer_controls_middle,
            textvariable=self._camera_scene_camera_stride_var,
            values=("1", "10", "50"),
            state="normal",
            width=6,
        ).pack(side=tk.LEFT)
        tk.Label(viewer_controls_middle, text="Grid step", bg=self.APP_BG).pack(
            side=tk.LEFT, padx=(10, 4)
        )
        ttk.Combobox(
            viewer_controls_middle,
            textvariable=self._camera_scene_grid_step_var,
            values=("0.01", "1", "10", "100"),
            state="normal",
            width=7,
        ).pack(side=tk.LEFT)
        tk.Label(viewer_controls_middle, text="Grid span", bg=self.APP_BG).pack(
            side=tk.LEFT, padx=(10, 4)
        )
        ttk.Combobox(
            viewer_controls_middle,
            textvariable=self._camera_scene_grid_span_var,
            values=("auto", "5", "10", "20", "50", "100", "200", "500"),
            state="normal",
            width=7,
        ).pack(side=tk.LEFT)
        tk.Checkbutton(
            viewer_controls_middle,
            text="Ground Grid",
            variable=self._camera_scene_show_grid_var,
            command=self._redraw_camera_scene_canvas,
            bg=self.APP_BG,
        ).pack(side=tk.LEFT, padx=(12, 0))
        tk.Checkbutton(
            viewer_controls_middle,
            text="World XYZ Axes",
            variable=self._camera_scene_show_world_axes_var,
            command=self._redraw_camera_scene_canvas,
            bg=self.APP_BG,
        ).pack(side=tk.LEFT, padx=(12, 0))
        viewer_controls_bottom = tk.Frame(viewer_controls_shell, bg=self.APP_BG)
        viewer_controls_bottom.pack(fill="x", padx=12, pady=(0, 2))
        tk.Checkbutton(
            viewer_controls_bottom,
            text="Draw PointCloud",
            variable=self._camera_scene_draw_points_var,
            command=self._redraw_camera_scene_canvas,
            bg=self.APP_BG,
        ).pack(side=tk.LEFT, padx=(12, 0))
        tk.Checkbutton(
            viewer_controls_bottom,
            text="Depth View",
            variable=self._camera_scene_monochrome_points_var,
            command=self._on_camera_scene_depth_view_toggle,
            bg=self.APP_BG,
        ).pack(side=tk.LEFT, padx=(12, 0))
        camera_front_occlusion_cb = tk.Checkbutton(
            viewer_controls_bottom,
            text="Depth Occlusion",
            variable=self._camera_scene_front_occlusion_var,
            command=self._on_camera_scene_depth_occlusion_toggle,
            bg=self.APP_BG,
        )
        camera_front_occlusion_cb.pack(side=tk.LEFT, padx=(12, 0))
        self._camera_scene_front_occlusion_checkbutton = camera_front_occlusion_cb
        self._enforce_camera_scene_depth_view_constraints()
        tk.Checkbutton(
            viewer_controls_bottom,
            text="Draw cameras",
            variable=self._camera_scene_draw_cameras_var,
            command=self._redraw_camera_scene_canvas,
            bg=self.APP_BG,
        ).pack(side=tk.LEFT, padx=(12, 0))
        viewer_controls_actions = tk.Frame(viewer_controls_shell, bg=self.APP_BG)
        viewer_controls_actions.pack(fill="x", padx=12, pady=(0, 4))
        tk.Button(
            viewer_controls_actions,
            text="Reset View",
            command=self._on_reset_camera_scene_camera_view,
        ).pack(side=tk.LEFT, padx=(8, 0))

        status_band = tk.Frame(viewer_frame, bg=self.APP_BG)
        status_band.pack(fill="x")
        status_label = tk.Label(
            status_band,
            textvariable=self._camera_scene_info_var,
            anchor="w",
            bg=self.APP_BG,
            fg="#0f172a",
            font=("TkDefaultFont", 10, "bold"),
        )
        status_label.pack(fill="x", padx=12, pady=(0, 4))

        canvas_wrap = tk.Frame(viewer_frame, bg="#0f172a")
        canvas_wrap.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        canvas = tk.Canvas(
            canvas_wrap,
            width=max(PLY_VIEW_CANVAS_WIDTH, 1180),
            height=520,
            bg="#101010",
            highlightthickness=0,
        )
        canvas.pack(fill="both", expand=True)
        self._camera_scene_canvas_image_id = canvas.create_image(
            0, 0, anchor="nw"
        )
        self._camera_scene_canvas = canvas
        canvas.bind("<Configure>", self._on_camera_scene_canvas_configure)
        canvas.bind("<ButtonPress-1>", self._on_camera_scene_drag_start)
        canvas.bind("<Double-Button-1>", self._on_camera_scene_double_click)
        canvas.bind("<B1-Motion>", self._on_camera_scene_drag_move)
        canvas.bind("<ButtonRelease-1>", self._on_camera_scene_drag_end)
        canvas.bind("<ButtonPress-3>", self._on_camera_scene_pan_start)
        canvas.bind("<B3-Motion>", self._on_camera_scene_pan_move)
        canvas.bind("<ButtonRelease-3>", self._on_camera_scene_pan_end)
        canvas.bind("<Leave>", self._on_camera_scene_drag_end)
        canvas.bind("<MouseWheel>", self._on_camera_scene_zoom)
        canvas.bind("<Button-4>", self._on_camera_scene_zoom)
        canvas.bind("<Button-5>", self._on_camera_scene_zoom)

        def _init_camera_format_workspace() -> None:
            try:
                total_w = max(workspace.winfo_width(), 1280)
                sash_x = max(540, min(total_w - 760, 620))
                workspace.sash_place(0, sash_x, 1)
            except Exception:
                pass

        self.root.after_idle(_init_camera_format_workspace)

        self._update_camera_scene_input_state()
        self._redraw_camera_scene_canvas()

    def _build_preview_tab(self, parent: tk.Widget) -> None:
        main = self._create_tab_shell(
            parent,
            "360PerspCut",
            "Preview and adjust panorama cut layouts before generating perspective camera outputs.",
        )

        work_frame = tk.Frame(main)
        work_frame.pack(fill="both", expand=True, padx=8, pady=(8, 4))

        left_right_wrapper = tk.Frame(work_frame)
        left_right_wrapper.pack(fill="both", expand=True)

        self.preview_frame = tk.LabelFrame(left_right_wrapper, text="Preview")
        self.preview_frame.pack(side=tk.LEFT, padx=(0, 4), pady=0, fill="y", expand=False, anchor="n")
        self.preview_frame.pack_propagate(False)
        self.preview_frame.configure(
            width=self.display_width + 10,
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

        self.controls_frame = tk.LabelFrame(left_right_wrapper, text="Controls", width=500)
        self.controls_frame.pack(side=tk.LEFT, fill="y", expand=False, padx=(4, 0), pady=0)
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
        buttons_frame.pack(side=tk.BOTTOM, fill="x", pady=(8, 0), ipady=2)

        buttons_inner = tk.Frame(buttons_frame)
        buttons_inner.pack(fill="both", expand=True, padx=10, pady=(8, 8))
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

        self._update_jpeg_quality_state()
        self._update_preview_inspect_state()
        self._sync_panel_heights()

    def _build_config_tab(self, parent: tk.Widget) -> None:
        container = self._create_tab_shell(
            parent,
            "Config",
            "Configure shared application paths and defaults used across the tool suite.",
        )

        appearance = tk.LabelFrame(container, text="Appearance")
        appearance.pack(fill="x", padx=8, pady=(8, 0))
        appearance.columnconfigure(1, weight=1)

        tk.Label(appearance, text="UI style").grid(
            row=0, column=0, padx=4, pady=4, sticky="e"
        )
        theme_combo = ttk.Combobox(
            appearance,
            textvariable=self._ui_theme_name_var,
            values=tuple(self.UI_THEMES.keys()),
            state="readonly",
            width=20,
        )
        theme_combo.grid(row=0, column=1, padx=4, pady=4, sticky="w")
        theme_combo.bind("<<ComboboxSelected>>", self._on_ui_theme_selected)
        tk.Label(
            appearance,
            text="Theme changes apply immediately. Use Save Config to keep the current style and ffmpeg path.",
            fg=self.MUTED_FG,
        ).grid(row=0, column=2, padx=(8, 4), pady=4, sticky="w")

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

        actions = tk.Frame(container, bg=self.APP_BG)
        actions.pack(fill="x", padx=8, pady=(0, 8))
        save_btn = tk.Button(actions, text="Save Config", command=self._save_config_settings, width=16)
        save_btn.pack(side=tk.LEFT)

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
        if key == "human":
            self._cleanup_human_manual_mask_temp_dir()
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
                    f"[time] PointCloudOptimizer elapsed: {elapsed:.2f}s",
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
        settings = self._collect_human_mask_settings()
        if settings is None:
            return
        manual_override_dir = self._prepare_human_manual_mask_override_dir()
        self._clear_human_preview_cache(clear_display=True)

        cmd: List[str] = [
            sys.executable,
            str(self.cli_tools_dir / "gs360_SegmentationMaskTool.py"),
            "-i",
            str(settings["input_path"]),
        ]

        output_dir = settings["output_dir"]
        if output_dir:
            cmd.extend(["-o", output_dir])

        cmd.extend(["--mode", settings["mode"]])
        for target_name in settings["targets"]:
            if target_name in HUMAN_TARGET_CHOICES:
                cmd.extend(["--target", target_name])
            else:
                cmd.extend(["--target-name", target_name])

        if settings["cpu"]:
            cmd.append("--cpu")
        cmd.extend(["--cpu-workers", str(settings["cpu_workers"])])
        if settings["include_shadow"]:
            cmd.append("--include_shadow")
        if manual_override_dir is not None:
            cmd.extend(["--manual-mask-dir", str(manual_override_dir)])

        cmd.extend(["--mask-expand-mode", settings["expand_mode"]])
        if settings["expand_mode"] == "pixels":
            cmd.extend(["--mask-expand-pixels", str(settings["expand_pixels"])])
        else:
            cmd.extend(["--mask-expand-percent", str(settings["expand_percent"])])
        cmd.extend(["--edge-fuse-pixels", str(settings["edge_fuse_pixels"])])

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
            if (
                format_value == "all"
                and bool(self.msxml_vars["pc_rotate_x"].get())
            ):
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

    def _run_camera_format_converter(self) -> None:
        if not self.camera_scene_vars or not self.camera_converter_vars:
            return
        transform_values = self._collect_camera_scene_transform_values(
            show_error=True,
        )
        if transform_values is None:
            return

        source_type = self.camera_scene_vars["source_type"].get().strip()
        output_dir = self.camera_converter_vars["output_dir"].get().strip()
        if not output_dir:
            messagebox.showerror(
                "CameraFormatConverter",
                "Output root is required.",
            )
            return

        cmd: List[str] = [
            sys.executable,
            str(self.cli_tools_dir / "gs360_CameraFormatConverter.py"),
        ]
        try:
            width_value, height_value = self._get_camera_scene_optional_width_height()
        except Exception as exc:
            messagebox.showerror(
                "CameraFormatConverter",
                str(exc),
            )
            return

        if source_type == CAMERA_SCENE_SOURCE_CHOICES[0]:
            colmap_dir = self.camera_scene_vars["colmap_dir"].get().strip()
            if not colmap_dir:
                messagebox.showerror(
                    "CameraFormatConverter",
                    "COLMAP folder is required.",
                )
                return
            if not Path(colmap_dir).expanduser().exists():
                messagebox.showerror(
                    "CameraFormatConverter",
                    f"COLMAP folder not found:\n{colmap_dir}",
                )
                return
            cmd.extend(["colmap", colmap_dir, "-o", output_dir])
        elif source_type == CAMERA_SCENE_SOURCE_CHOICES[1]:
            transforms_json = self.camera_scene_vars["transforms_json"].get().strip()
            transforms_ply = self.camera_scene_vars["transforms_ply"].get().strip()
            if not transforms_json:
                messagebox.showerror(
                    "CameraFormatConverter",
                    "transforms.json is required.",
                )
                return
            if not Path(transforms_json).expanduser().exists():
                messagebox.showerror(
                    "CameraFormatConverter",
                    f"transforms.json not found:\n{transforms_json}",
                )
                return
            cmd.extend(
                [
                    "transforms-json",
                    "-o",
                    output_dir,
                    "--transforms-json",
                    transforms_json,
                ]
            )
            if transforms_ply:
                if not Path(transforms_ply).expanduser().exists():
                    messagebox.showerror(
                        "CameraFormatConverter",
                        f"transforms PLY not found:\n{transforms_ply}",
                    )
                    return
                cmd.extend(["--transforms-ply", transforms_ply])
        elif source_type == CAMERA_SCENE_SOURCE_CHOICES[2]:
            csv_path = self.camera_scene_vars["csv_path"].get().strip()
            csv_ply = self.camera_scene_vars["csv_ply"].get().strip()
            if not csv_path:
                messagebox.showerror(
                    "CameraFormatConverter",
                    "RealityScan CSV is required.",
                )
                return
            if not Path(csv_path).expanduser().exists():
                messagebox.showerror(
                    "CameraFormatConverter",
                    f"RealityScan CSV not found:\n{csv_path}",
                )
                return
            if width_value is None or height_value is None:
                messagebox.showerror(
                    "CameraFormatConverter",
                    "Width and Height must be positive integers for RealityScan CSV input.",
                )
                return
            cmd.extend(
                [
                    "realityscan-csv",
                    "-o",
                    output_dir,
                    "--csv",
                    csv_path,
                    "--width",
                    str(width_value),
                    "--height",
                    str(height_value),
                ]
            )
            if csv_ply:
                if not Path(csv_ply).expanduser().exists():
                    messagebox.showerror(
                        "CameraFormatConverter",
                        f"RealityScan PLY not found:\n{csv_ply}",
                    )
                    return
                cmd.extend(["--realityscan-ply", csv_ply])
        elif source_type == CAMERA_SCENE_SOURCE_CHOICES[3]:
            xmp_dir = self.camera_scene_vars["xmp_dir"].get().strip()
            xmp_ply = self.camera_scene_vars["xmp_ply"].get().strip()
            if not xmp_dir:
                messagebox.showerror(
                    "CameraFormatConverter",
                    "RealityScan XMP directory is required.",
                )
                return
            if not Path(xmp_dir).expanduser().exists():
                messagebox.showerror(
                    "CameraFormatConverter",
                    f"RealityScan XMP directory not found:\n{xmp_dir}",
                )
                return
            if width_value is None or height_value is None:
                messagebox.showerror(
                    "CameraFormatConverter",
                    "Width and Height are required for RealityScan XMP input.",
                )
                return
            cmd.extend(
                [
                    "realityscan-xmp",
                    "-o",
                    output_dir,
                    "--realityscan-xmp-dir",
                    xmp_dir,
                    "--width",
                    str(width_value),
                    "--height",
                    str(height_value),
                ]
            )
            if xmp_ply:
                if not Path(xmp_ply).expanduser().exists():
                    messagebox.showerror(
                        "CameraFormatConverter",
                        f"RealityScan PLY not found:\n{xmp_ply}",
                    )
                    return
                cmd.extend(["--realityscan-ply", xmp_ply])
        else:
            metashape_xml = self.camera_scene_vars["metashape_xml"].get().strip()
            metashape_ply = self.camera_scene_vars["metashape_ply"].get().strip()
            if not metashape_xml:
                messagebox.showerror(
                    "CameraFormatConverter",
                    "Metashape XML is required.",
                )
                return
            if not Path(metashape_xml).expanduser().exists():
                messagebox.showerror(
                    "CameraFormatConverter",
                    f"Metashape XML not found:\n{metashape_xml}",
                )
                return
            cmd.extend(
                [
                    "metashape-xml",
                    "-o",
                    output_dir,
                    "--metashape-xml",
                    metashape_xml,
                ]
            )
            if width_value is not None and height_value is not None:
                cmd.extend(["--width", str(width_value), "--height", str(height_value)])
            if metashape_ply:
                if not Path(metashape_ply).expanduser().exists():
                    messagebox.showerror(
                        "CameraFormatConverter",
                        f"RealityScan PLY not found:\n{metashape_ply}",
                    )
                    return
                cmd.extend(["--realityscan-ply", metashape_ply])

        export_flag_map = (
            ("export_colmap", "--export-colmap"),
            ("export_csv", "--export-realityscan-csv"),
            ("export_ply", "--export-realityscan-ply"),
            ("export_transforms", "--export-transforms-json"),
            ("export_transforms_ply", "--export-transforms-ply"),
            ("export_xmp", "--export-realityscan-xmp"),
            ("export_metashape_xml", "--export-metashape-xml"),
        )
        for var_name, flag in export_flag_map:
            if bool(self.camera_converter_vars[var_name].get()):
                cmd.append(flag)

        rotation_flag_map = (
            ("camera_rot_x_deg", "--camera-rot-x-deg"),
            ("camera_rot_y_deg", "--camera-rot-y-deg"),
            ("camera_rot_z_deg", "--camera-rot-z-deg"),
            ("pointcloud_rot_x_deg", "--pointcloud-rot-x-deg"),
            ("pointcloud_rot_y_deg", "--pointcloud-rot-y-deg"),
            ("pointcloud_rot_z_deg", "--pointcloud-rot-z-deg"),
        )
        for var_name, flag in rotation_flag_map:
            value = float(transform_values[var_name])
            if abs(value) > 1e-9:
                cmd.extend([flag, str(value)])
        scale_flag_map = (
            ("camera_scale", "--camera-scale"),
            ("pointcloud_scale", "--pointcloud-scale"),
        )
        for var_name, flag in scale_flag_map:
            value = float(transform_values[var_name])
            if abs(value - 1.0) > 1e-9:
                cmd.extend([flag, str(value)])

        self._run_cli_command(
            cmd,
            self.camera_converter_log,
            self.camera_converter_run_button,
            process_key="camera_converter",
            stop_button=self.camera_converter_stop_button,
            cwd=self.cli_tools_dir,
        )

    def _run_video_tool(self) -> None:
        if not self.video_vars:
            return
        video_path = self.video_vars["video"].get().strip()
        if not video_path:
            messagebox.showerror("gs360_Video2Frames", "Input video is required.")
            return
        video_file = Path(video_path).expanduser()
        if not video_file.exists():
            messagebox.showerror(
                "gs360_Video2Frames",
                f"Input video not found:\n{video_path}",
            )
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
        else:
            prefix_text = ""
        if not prefix_text:
            prefix_text = re.sub(r"\s+", "_", video_file.stem) or "dualfisheye"
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

        ffmpeg_path = self.ffmpeg_path_var.get().strip()
        if ffmpeg_path:
            base_cmd.extend(["--ffmpeg", ffmpeg_path])

        if bool(self.video_vars["experimental_dualfisheye"].get()):
            extract_y_cmd = list(base_cmd)
            extract_y_cmd.extend(["--map-stream", "0:v:0", "--name-suffix", "_Y"])
            extract_x_cmd = list(base_cmd)
            extract_x_cmd.extend(["--map-stream", "0:v:1", "--name-suffix", "_X"])
            self._set_text_widget(self.video_log, "")
            self._append_text_widget(
                self.video_log,
                "[INFO] Experimental DualFisheye extraction started: raw 360 video -> fisheye pair folder",
            )
            self._append_text_widget(
                self.video_log,
                "[INFO] Queue order: lens Y (0:v:0) then lens X (0:v:1)",
            )
            self._queued_cli_commands.pop("video", None)
            self._queued_cli_commands["video"] = [
                (
                    extract_x_cmd,
                    self.cli_tools_dir,
                    False,
                    "[INFO] Starting lens X extraction...",
                )
            ]
            self._run_cli_command(
                extract_y_cmd,
                self.video_log,
                self.video_run_button,
                process_key="video",
                stop_button=self.video_stop_button,
                cwd=self.cli_tools_dir,
                clear_log=False,
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

    def _run_dualfisheye_extract_tool(self) -> None:
        if not self.dualfisheye_vars:
            return

        video_path = self.dualfisheye_vars["video"].get().strip()
        if not video_path:
            messagebox.showerror(
                "DualFisheyePipeline", "Input raw video is required."
            )
            return
        video_file = Path(video_path).expanduser()
        if not video_file.exists():
            messagebox.showerror(
                "DualFisheyePipeline",
                f"Input raw video not found:\n{video_path}",
            )
            return

        pairs_output = self.dualfisheye_vars["pairs_output"].get().strip()
        if not pairs_output:
            messagebox.showerror(
                "DualFisheyePipeline",
                "Extracted pair folder is required.",
            )
            return

        fps_value = self.dualfisheye_vars["fps"].get().strip()
        try:
            fps_float = float(fps_value)
            if fps_float <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror(
                "DualFisheyePipeline", "FPS must be a positive number."
            )
            return
        fps_formatted = self._format_fps_for_output(fps_value) or f"{fps_float}"

        ext_value = self.dualfisheye_vars["ext"].get().strip() or "jpg"
        prefix_text = self.dualfisheye_vars["prefix"].get().strip()
        if not prefix_text:
            prefix_text = re.sub(r"\s+", "_", video_file.stem) or "dualfisheye"

        base_cmd: List[str] = [
            sys.executable,
            str(self.cli_tools_dir / "gs360_Video2Frames.py"),
            "-i",
            video_path,
            "-f",
            fps_formatted,
            "-o",
            pairs_output,
            "--ext",
            ext_value,
            "--prefix",
            prefix_text,
        ]

        start_value = self.dualfisheye_vars["start"].get().strip()
        if start_value:
            try:
                float(start_value)
            except ValueError:
                messagebox.showerror(
                    "DualFisheyePipeline",
                    "Start time must be numeric.",
                )
                return
            base_cmd.extend(["--start", start_value])

        end_value = self.dualfisheye_vars["end"].get().strip()
        if end_value:
            try:
                float(end_value)
            except ValueError:
                messagebox.showerror(
                    "DualFisheyePipeline",
                    "End time must be numeric.",
                )
                return
            base_cmd.extend(["--end", end_value])

        if not bool(self.dualfisheye_vars["keep_rec709"].get()):
            base_cmd.append("--keep-rec709")
        if bool(self.dualfisheye_vars["overwrite"].get()):
            base_cmd.append("--overwrite")

        ffmpeg_path = self.ffmpeg_path_var.get().strip()
        if ffmpeg_path:
            base_cmd.extend(["--ffmpeg", ffmpeg_path])

        extract_y_cmd = list(base_cmd)
        extract_y_cmd.extend(["--map-stream", "0:v:0", "--name-suffix", "_Y"])
        extract_x_cmd = list(base_cmd)
        extract_x_cmd.extend(["--map-stream", "0:v:1", "--name-suffix", "_X"])

        self._set_text_widget(self.dualfisheye_log, "")
        self._append_text_widget(
            self.dualfisheye_log,
            "[INFO] Stage 1 extraction started: raw video -> fisheye pair folder",
        )
        self._append_text_widget(
            self.dualfisheye_log,
            "[INFO] Queue order: lens Y (0:v:0) then lens X (0:v:1)",
        )
        self._queued_cli_commands.pop("dualfisheye_extract", None)
        self._queued_cli_commands["dualfisheye_extract"] = [
            (
                extract_x_cmd,
                self.cli_tools_dir,
                False,
                "[next] Extract raw lens X frames (map 0:v:1)",
            ),
        ]
        self._run_cli_command(
            extract_y_cmd,
            self.dualfisheye_log,
            self.dualfisheye_extract_run_button,
            process_key="dualfisheye_extract",
            stop_button=self.dualfisheye_extract_stop_button,
            cwd=self.cli_tools_dir,
            clear_log=False,
        )

    def _run_dualfisheye_calibration_tool(self) -> None:
        if not self.dualfisheye_vars:
            return

        dry_run = bool(self.dualfisheye_vars["dry_run"].get())
        metadata_only = bool(self.dualfisheye_vars["metadata_only"].get())

        pairs_output = self.dualfisheye_vars["pair_input"].get().strip()
        if not pairs_output and not metadata_only:
            messagebox.showerror(
                "DualFisheyePipeline",
                "Pair folder is required.",
            )
            return
        pairs_dir: Optional[Path] = None
        if pairs_output:
            pairs_dir = Path(pairs_output).expanduser()
        if pairs_dir is not None and not pairs_dir.exists():
            messagebox.showerror(
                "DualFisheyePipeline",
                f"Pair folder not found:\n{pairs_output}",
            )
            return

        no_perspective = bool(self.dualfisheye_vars["no_perspective"].get())
        save_fisheye = bool(self.dualfisheye_vars["save_fisheye_output"].get())
        save_color = bool(
            self.dualfisheye_vars["save_color_corrected_output"].get()
        )
        if (
            (not metadata_only)
            and no_perspective
            and not save_fisheye
            and not save_color
        ):
            messagebox.showerror(
                "DualFisheyePipeline",
                (
                    "Enable at least one output type when perspective "
                    "output is disabled."
                ),
            )
            return

        camera_extrinsics_xml = self.dualfisheye_vars[
            "camera_extrinsics_xml"
        ].get().strip()
        camera_extrinsics_xml_path: Optional[Path] = None
        if camera_extrinsics_xml:
            camera_extrinsics_xml_path = Path(camera_extrinsics_xml).expanduser()
            if not camera_extrinsics_xml_path.is_absolute():
                camera_extrinsics_xml_path = (
                    self.cli_tools_dir.parent / camera_extrinsics_xml
                ).resolve()
            if not camera_extrinsics_xml_path.exists():
                messagebox.showerror(
                    "DualFisheyePipeline",
                    (
                        "Perspective camera extrinsics XML not found:\n"
                        f"{camera_extrinsics_xml}"
                    ),
                )
                return
            if no_perspective and not metadata_only:
                messagebox.showerror(
                    "DualFisheyePipeline",
                    (
                        "Perspective camera extrinsics XML requires "
                        "perspective output to be enabled."
                    ),
                )
                return

        if metadata_only and camera_extrinsics_xml_path is None:
            messagebox.showerror(
                "DualFisheyePipeline",
                "COLMAP + XML only requires Perspective Camera Extrinsics XML.",
            )
            return

        camera_xml = self.dualfisheye_vars["camera_xml"].get().strip()
        camera_xml_path: Optional[Path] = None
        if camera_extrinsics_xml_path is None and camera_xml:
            camera_xml_path = Path(camera_xml).expanduser()
            if not camera_xml_path.is_absolute():
                camera_xml_path = (
                    self.cli_tools_dir.parent / camera_xml
                ).resolve()
            if not camera_xml_path.exists():
                messagebox.showerror(
                    "DualFisheyePipeline",
                    f"Calibration XML not found:\n{camera_xml}",
                )
                return

        if (
            (not metadata_only)
            and camera_extrinsics_xml_path is None
            and camera_xml_path is None
        ):
            messagebox.showerror(
                "DualFisheyePipeline",
                (
                    "Extrinsics XML or Fisheye Distortion XML is required."
                ),
            )
            return

        pointcloud_ply = self.dualfisheye_vars["pointcloud_ply"].get().strip()
        pointcloud_ply_path: Optional[Path] = None
        if pointcloud_ply:
            pointcloud_ply_path = Path(pointcloud_ply).expanduser()
            if not pointcloud_ply_path.is_absolute():
                pointcloud_ply_path = (
                    self.cli_tools_dir.parent / pointcloud_ply
                ).resolve()
            if not pointcloud_ply_path.exists():
                messagebox.showerror(
                    "DualFisheyePipeline",
                    f"Metashape point cloud PLY not found:\n{pointcloud_ply}",
                )
                return
        if metadata_only and pointcloud_ply_path is None:
            messagebox.showerror(
                "DualFisheyePipeline",
                "COLMAP + XML only requires Metashape PointCloud PLY.",
            )
            return

        mask_input = self.dualfisheye_vars["mask_input"].get().strip()
        mask_input_path: Optional[Path] = None
        if mask_input and not metadata_only:
            mask_input_path = Path(mask_input).expanduser()
            if not mask_input_path.is_absolute():
                mask_input_path = (
                    self.cli_tools_dir.parent / mask_input
                ).resolve()
            if not mask_input_path.exists():
                messagebox.showerror(
                    "DualFisheyePipeline",
                    f"Mask folder not found:\n{mask_input}",
                )
                return
            if no_perspective:
                messagebox.showerror(
                    "DualFisheyePipeline",
                    "Mask folder requires perspective output to be enabled.",
                )
                return

        calibration_cmd: List[str] = [
            sys.executable,
            str(
                self.cli_tools_dir
                / "gs360_DualFisheyeDistortionCalibration.py"
            ),
        ]
        if pairs_dir is not None:
            calibration_cmd.extend(["-i", pairs_output])
        if metadata_only:
            calibration_cmd.append("--metadata-only")
        if camera_xml_path is not None and not metadata_only:
            calibration_cmd.extend(["-x", str(camera_xml_path)])

        fisheye_output = self.dualfisheye_vars["fisheye_output"].get().strip()
        if save_fisheye and fisheye_output and not metadata_only:
            calibration_cmd.extend(["-o", fisheye_output])

        use_input_lut = bool(self.dualfisheye_vars["use_input_lut"].get())
        input_lut = self.dualfisheye_vars["input_lut"].get().strip()
        if use_input_lut and input_lut and not metadata_only:
            input_lut_path = Path(input_lut).expanduser()
            if not input_lut_path.is_absolute():
                input_lut_path = (self.cli_tools_dir.parent / input_lut).resolve()
            if not input_lut_path.exists():
                messagebox.showerror(
                    "DualFisheyePipeline",
                    f"Input LUT not found:\n{input_lut}",
                )
                return
            calibration_cmd.extend(["--input-lut", str(input_lut_path)])

        lut_output_color_space = (
            self.dualfisheye_vars["lut_output_color_space"].get().strip()
            or "sRGB"
        )
        calibration_cmd.extend(
            ["--lut-output-color-space", lut_output_color_space.lower()]
        )

        perspective_enabled = metadata_only or (not no_perspective)
        if no_perspective and not metadata_only:
            calibration_cmd.append("--no-perspective")
        if perspective_enabled:
            perspective_output = self.dualfisheye_vars[
                "perspective_output"
            ].get().strip()
            if perspective_output:
                calibration_cmd.extend(
                    ["--perspective-output-dir", perspective_output]
                )
            perspective_ext = (
                self.dualfisheye_vars["perspective_ext"].get().strip() or "jpg"
            )
            calibration_cmd.extend(["--perspective-ext", perspective_ext])
            perspective_mask_ext = (
                self.dualfisheye_vars["perspective_mask_ext"].get().strip()
                or "png"
            )
            calibration_cmd.extend(
                ["--perspective-mask-ext", perspective_mask_ext]
            )

            perspective_size = self.dualfisheye_vars[
                "perspective_size"
            ].get().strip()
            if perspective_size:
                try:
                    int(perspective_size)
                except ValueError:
                    messagebox.showerror(
                        "DualFisheyePipeline",
                        "Perspective size must be an integer.",
                    )
                    return
                calibration_cmd.extend(
                    ["--perspective-size", perspective_size]
                )

            perspective_focal_mm = self.dualfisheye_vars[
                "perspective_focal_mm"
            ].get().strip()
            if perspective_focal_mm:
                try:
                    float(perspective_focal_mm)
                except ValueError:
                    messagebox.showerror(
                        "DualFisheyePipeline",
                        "Perspective focal mm must be numeric.",
                    )
                    return
                calibration_cmd.extend(
                    ["--perspective-focal-mm", perspective_focal_mm]
                )

            if camera_extrinsics_xml:
                calibration_cmd.extend(
                    [
                        "--camera-extrinsics-xml",
                        str(camera_extrinsics_xml_path),
                    ]
                )
            if pointcloud_ply:
                calibration_cmd.extend(
                    ["--pointcloud-ply", str(pointcloud_ply_path)]
                )
            if mask_input_path is not None and not metadata_only:
                calibration_cmd.extend(
                    ["--mask-input-dir", str(mask_input_path)]
                )

        workers_value = self.dualfisheye_vars["workers"].get().strip()
        if not workers_value:
            messagebox.showerror(
                "DualFisheyePipeline",
                "Workers must be an integer >= 1.",
            )
            return
        try:
            workers_int = int(workers_value)
        except ValueError:
            messagebox.showerror(
                "DualFisheyePipeline",
                "Workers must be an integer >= 1.",
            )
            return
        if workers_int < 1:
            messagebox.showerror(
                "DualFisheyePipeline",
                "Workers must be an integer >= 1.",
            )
            return
        calibration_cmd.extend(["--workers", str(workers_int)])

        memory_throttle_value = self.dualfisheye_vars[
            "memory_throttle_percent"
        ].get().strip()
        if not memory_throttle_value:
            messagebox.showerror(
                "DualFisheyePipeline",
                "Memory throttle % must be > 0 and <= 100.",
            )
            return
        try:
            memory_throttle_float = float(memory_throttle_value)
        except ValueError:
            messagebox.showerror(
                "DualFisheyePipeline",
                "Memory throttle % must be > 0 and <= 100.",
            )
            return
        if memory_throttle_float <= 0.0 or memory_throttle_float > 100.0:
            messagebox.showerror(
                "DualFisheyePipeline",
                "Memory throttle % must be > 0 and <= 100.",
            )
            return
        calibration_cmd.extend(
            [
                "--memory-throttle-percent",
                str(memory_throttle_float),
            ]
        )

        if save_fisheye and not metadata_only:
            calibration_cmd.append("--save-fisheye-output")

        if save_color and not metadata_only:
            calibration_cmd.append("--save-color-corrected-output")
            color_output = self.dualfisheye_vars["color_output"].get().strip()
            if color_output:
                calibration_cmd.extend(
                    ["--color-corrected-output-dir", color_output]
                )

        if dry_run:
            calibration_cmd.append("--dry-run")

        self._set_text_widget(self.dualfisheye_log, "")
        self._append_text_widget(
            self.dualfisheye_log,
            (
                "[INFO] Stage 3 calibration started: XML/PLY -> metadata only"
                if metadata_only else
                "[INFO] Stage 3 calibration started: pair folder -> outputs"
            ),
        )
        if camera_extrinsics_xml_path is not None:
            self._append_text_widget(
                self.dualfisheye_log,
                (
                    "[INFO] Distortion source: Extrinsics XML "
                    "(adjusted calibration preferred)"
                ),
            )
            if camera_xml:
                self._append_text_widget(
                    self.dualfisheye_log,
                    "[INFO] Fisheye Distortion XML ignored because Extrinsics XML is set.",
                )
        elif camera_xml_path is not None:
            self._append_text_widget(
                self.dualfisheye_log,
                "[INFO] Distortion source: Fisheye Distortion XML",
            )
        self._append_text_widget(
            self.dualfisheye_log,
            "[INFO] Pair-worker mode: {} workers, memory throttle {}%".format(
                workers_int,
                memory_throttle_float,
            ),
        )
        if perspective_enabled:
            self._append_text_widget(
                self.dualfisheye_log,
                "[INFO] Perspective / COLMAP root: {}".format(
                    self._compute_dualfisheye_perspective_root_path()
                    or self.dualfisheye_vars["perspective_output"].get().strip()
                ),
            )
            self._append_text_widget(
                self.dualfisheye_log,
                "[INFO] Perspective XML: {}".format(
                    self.dualfisheye_xml_output_var.get().strip()
                ),
            )
            self._append_text_widget(
                self.dualfisheye_log,
                "[INFO] COLMAP Images: {}".format(
                    self.dualfisheye_colmap_images_var.get().strip()
                ),
            )
            self._append_text_widget(
                self.dualfisheye_log,
                "[INFO] COLMAP Masks: {}".format(
                    self.dualfisheye_colmap_masks_var.get().strip()
                ),
            )
            self._append_text_widget(
                self.dualfisheye_log,
                "[INFO] COLMAP Sparse\\0: {}".format(
                    self.dualfisheye_colmap_sparse_var.get().strip()
                ),
            )
        if camera_extrinsics_xml:
            self._append_text_widget(
                self.dualfisheye_log,
                (
                    "[INFO] Perspective metadata export enabled: XML + "
                    "COLMAP from current dual-fisheye alignment"
                ),
            )
        if perspective_enabled:
            self._append_text_widget(
                self.dualfisheye_log,
                "[INFO] Perspective image ext: {}".format(
                    self.dualfisheye_vars["perspective_ext"].get().strip() or "jpg"
                ),
            )
            self._append_text_widget(
                self.dualfisheye_log,
                "[INFO] Perspective mask ext: {}".format(
                    self.dualfisheye_vars["perspective_mask_ext"].get().strip() or "png"
                ),
            )
        if pointcloud_ply:
            self._append_text_widget(
                self.dualfisheye_log,
                "[INFO] Metashape point cloud PLY: {}".format(pointcloud_ply),
            )
        if mask_input_path is not None:
            self._append_text_widget(
                self.dualfisheye_log,
                "[INFO] Pair mask folder: {}".format(str(mask_input_path)),
            )
        self._queued_cli_commands.pop("dualfisheye_calibration", None)
        self._run_cli_command(
            calibration_cmd,
            self.dualfisheye_log,
            self.dualfisheye_calibration_run_button,
            process_key="dualfisheye_calibration",
            stop_button=self.dualfisheye_calibration_stop_button,
            cwd=self.cli_tools_dir,
            clear_log=False,
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
        compute_optical_flow_enabled = bool(
            self.selector_vars["compute_optical_flow"].get()
        )
        compute_optical_flow_cli_enabled = compute_optical_flow_enabled
        augment_motion_enabled = bool(self.selector_vars["augment_motion"].get())
        if compute_optical_flow_enabled:
            if self._selector_optical_flow_threshold_value(show_error=True) is None:
                return

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

        input_mode_label = self.selector_vars["input_mode"].get().strip()
        input_mode = SELECTOR_INPUT_MODE_LABEL_TO_CLI.get(
            input_mode_label,
            input_mode_label.lower(),
        )
        if input_mode:
            cmd.extend(["--input_mode", input_mode])

        score_backend = self.selector_vars["score_backend"].get().strip()
        if score_backend:
            cmd.extend(["--score_backend", score_backend])

        csv_mode = self.selector_vars["csv_mode"].get().strip()
        csv_path = self.selector_vars["csv_path"].get().strip()
        if (
            compute_optical_flow_enabled
            and csv_mode == "reselect"
            and csv_path
        ):
            flow_exists = self._csv_has_numeric_flow_motion(
                csv_path,
                base_dir=in_dir,
            )
            if flow_exists:
                compute_optical_flow_cli_enabled = False
                self._append_text_widget(
                    self.selector_log,
                    "[info] reselect CSV already has numeric flow_motion values; reusing them without recomputation.",
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

        gap_mode_label = self.selector_vars["augment_gap_mode"].get().strip()
        gap_mode = SELECTOR_GAP_MODE_LABEL_TO_CLI.get(
            gap_mode_label,
            gap_mode_label.lower(),
        )
        if gap_mode and gap_mode != "off":
            cmd.append("--augment_gaps")
            cmd.extend(["--augment_gap_mode", gap_mode])
        else:
            cmd.append("--no_augment_gaps")
        if bool(self.selector_vars["augment_lowlight"].get()):
            cmd.append("--augment_lowlight")
        if compute_optical_flow_cli_enabled:
            cmd.append("--compute_optical_flow")
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

    def _update_selector_optical_flow_state(self, *_args) -> None:
        """Enable the low-motion threshold entry only when optical flow is requested."""
        entry = self.selector_optical_flow_threshold_entry
        if entry is None:
            return
        enabled = False
        if self.selector_vars:
            flow_var = self.selector_vars.get("compute_optical_flow")
            enabled = bool(flow_var.get()) if flow_var is not None else False
        try:
            entry.configure(state="normal" if enabled else "disabled")
        except tk.TclError:
            pass

    def _selector_optical_flow_threshold_value(
        self,
        show_error: bool = False,
    ) -> Optional[float]:
        """Parse the low-motion threshold used for overview highlighting."""
        threshold_var = self.selector_vars.get("optical_flow_low_threshold")
        raw_value = threshold_var.get().strip() if threshold_var is not None else ""
        if not raw_value:
            raw_value = "0.10"
        try:
            threshold = float(raw_value)
        except ValueError:
            if show_error:
                messagebox.showerror(
                    "gs360_FrameSelector",
                    "Optical flow low-motion threshold must be numeric.",
                )
            return None
        if threshold < 0.0:
            if show_error:
                messagebox.showerror(
                    "gs360_FrameSelector",
                    "Optical flow low-motion threshold must be zero or greater.",
                )
            return None
        return threshold

    def _selector_collect_low_motion_spans(
        self,
        parsed_entries: Sequence[Dict[str, Any]],
        flow_threshold: float,
    ) -> List[Dict[str, Any]]:
        """Collect spans where motion stays low between consecutive selected frames."""
        selected_positions = [
            pos
            for pos, entry in enumerate(parsed_entries)
            if bool(entry.get("selected_current", False))
        ]
        if len(selected_positions) < 2:
            return []

        valid_pair_spans: List[Dict[str, Any]] = []
        for left_pos, right_pos in zip(selected_positions, selected_positions[1:]):
            interval_flow_values: List[float] = []
            valid_span = True
            for entry in parsed_entries[left_pos:right_pos + 1]:
                flow_val = entry.get("flow_motion")
                if flow_val is None:
                    valid_span = False
                    break
                try:
                    flow_value = float(flow_val)
                except (TypeError, ValueError):
                    valid_span = False
                    break
                if not math.isfinite(flow_value) or flow_value > flow_threshold:
                    valid_span = False
                    break
                interval_flow_values.append(flow_value)
            if not valid_span or not interval_flow_values:
                continue
            valid_pair_spans.append(
                {
                    "start_pos": left_pos,
                    "end_pos": right_pos,
                    "selected_positions": [left_pos, right_pos],
                    "max_flow": max(interval_flow_values),
                }
            )

        if not valid_pair_spans:
            return []

        merged_spans: List[Dict[str, Any]] = []
        current_span: Optional[Dict[str, Any]] = None
        for span in valid_pair_spans:
            if current_span is None:
                current_span = dict(span)
                current_span["selected_positions"] = list(span["selected_positions"])
                continue
            current_selected_positions = current_span.get("selected_positions", [])
            current_end_selected = (
                current_selected_positions[-1]
                if current_selected_positions else current_span["end_pos"]
            )
            if span["start_pos"] == current_end_selected:
                current_span["end_pos"] = span["end_pos"]
                current_span["max_flow"] = max(
                    float(current_span["max_flow"]),
                    float(span["max_flow"]),
                )
                current_span["selected_positions"].append(span["end_pos"])
                continue
            merged_spans.append(current_span)
            current_span = dict(span)
            current_span["selected_positions"] = list(span["selected_positions"])
        if current_span is not None:
            merged_spans.append(current_span)

        result_spans: List[Dict[str, Any]] = []
        for span in merged_spans:
            selected_pos_list = list(span.get("selected_positions", []))
            start_pos = int(span["start_pos"])
            end_pos = int(span["end_pos"])
            result_spans.append(
                {
                    "start_pos": start_pos,
                    "end_pos": end_pos,
                    "selected_count": len(selected_pos_list),
                    "frame_count": max(1, end_pos - start_pos + 1),
                    "max_flow": float(span["max_flow"]),
                }
            )
        return result_spans

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

    def _csv_has_numeric_flow_motion(
        self,
        csv_path: str,
        base_dir: Optional[str] = None,
    ) -> Optional[bool]:
        """Return True when flow_motion contains at least one numeric value."""
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
                saw_value = False
                for row in reader:
                    raw = row.get(flow_key)
                    if raw is None:
                        continue
                    text = str(raw).strip()
                    if not text:
                        continue
                    try:
                        float(text)
                    except ValueError:
                        continue
                    saw_value = True
                    break
                return saw_value
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

    @staticmethod
    def _is_colmap_text_model_dir(path: Path) -> bool:
        return (
            path.is_dir()
            and (path / "cameras.txt").is_file()
            and (path / "images.txt").is_file()
            and (path / "points3D.txt").is_file()
        )

    def _ply_default_output_for_input(self, input_text: str) -> Optional[str]:
        text = input_text.strip()
        if not text:
            return None
        try:
            path_obj = Path(text).expanduser()
        except Exception:
            return None
        if self._is_colmap_text_model_dir(path_obj):
            try:
                return str(path_obj.with_name(f"{path_obj.name}_output"))
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

    def _clear_ply_append_files(self) -> None:
        if not self.ply_vars:
            return
        append_var = self.ply_vars.get("append_ply")
        if append_var is not None:
            append_var.set("")

    def _clear_ply_remove_snapshot(self) -> None:
        self._ply_pre_remove_points = None
        self._ply_pre_remove_colors = None
        self._ply_pre_remove_point_ids = None
        self._ply_pre_remove_total_points = 0
        self._ply_pre_remove_sample_step = 1
        self._ply_pre_remove_sky_points = None
        self._ply_pre_remove_sky_colors = None

    def _clear_appended_ply_from_viewer(self) -> None:
        if (
            self._ply_pre_append_points is None
            or self._ply_pre_append_colors is None
        ):
            return
        self._ply_view_points = self._ply_pre_append_points.astype(
            np.float32,
            copy=True,
        )
        self._ply_view_colors = self._ply_pre_append_colors.astype(
            np.uint8,
            copy=True,
        )
        self._ply_view_point_ids = (
            self._ply_pre_append_point_ids.astype(np.int64, copy=True)
            if self._ply_pre_append_point_ids is not None
            else None
        )
        self._ply_view_points_centered = self._ply_view_points.astype(
            np.float32,
            copy=False,
        )
        self._ply_view_total_points = int(
            max(self._ply_pre_append_total_points, self._ply_view_points.shape[0])
        )
        self._ply_view_sample_step = max(1, int(self._ply_pre_append_sample_step))
        self._ply_pre_append_points = None
        self._ply_pre_append_colors = None
        self._ply_pre_append_point_ids = None
        self._ply_pre_append_total_points = 0
        self._ply_pre_append_sample_step = 1
        self._clear_ply_remove_snapshot()
        self._refresh_ply_view_rgb_mean()
        self._update_ply_view_info_from_current()
        self._redraw_ply_canvas()

    def _append_ply_files_to_viewer(self) -> None:
        """Append additional PLY files into the current viewer buffers."""
        if self._ply_view_points is None or self._ply_view_colors is None:
            messagebox.showerror(
                "Point Cloud Viewer",
                "Load a point cloud before appending files to the viewer.",
            )
            return
        if not self.ply_vars:
            return
        append_var = self.ply_vars.get("append_ply")
        raw_text = append_var.get().strip() if append_var is not None else ""
        items = self._parse_ply_append_items(raw_text)
        if not items:
            messagebox.showerror(
                "Point Cloud Viewer",
                "Specify at least one append PLY file.",
            )
            return

        max_points = self._get_ply_view_high_max_points(show_error=True)
        if max_points is None:
            return

        append_points: List[np.ndarray] = []
        append_colors: List[np.ndarray] = []
        added_loaded = 0
        added_original = 0
        failed: List[str] = []
        for raw_path in items:
            try:
                path_obj = Path(raw_path).expanduser()
            except Exception:
                failed.append(f"{raw_path} (invalid path)")
                continue
            if not path_obj.is_absolute():
                path_obj = (self.base_dir / path_obj).resolve()
            if not path_obj.exists():
                failed.append(f"{path_obj} (not found)")
                continue
            try:
                pts, cols, orig_count, _sample = self._load_binary_ply_points(
                    path_obj,
                    max_points=max_points,
                )
            except Exception as exc:
                failed.append(f"{path_obj} ({exc})")
                continue
            if pts is None or cols is None or pts.size == 0 or cols.size == 0:
                failed.append(f"{path_obj} (no points)")
                continue
            append_points.append(pts.astype(np.float32, copy=False))
            append_colors.append(cols.astype(np.uint8, copy=False))
            added_loaded += int(pts.shape[0])
            added_original += int(max(orig_count, pts.shape[0]))

        if not append_points:
            messagebox.showerror(
                "Point Cloud Viewer",
                "Failed to append PLY files to viewer.",
            )
            if failed:
                self._append_text_widget(
                    self.ply_log,
                    "[viewer-append] failed: {}".format("; ".join(failed[:5])),
                )
            return

        base_points = self._ply_view_points.astype(np.float32, copy=False)
        base_colors = self._ply_view_colors.astype(np.uint8, copy=False)
        base_loaded = int(base_points.shape[0])
        base_original = int(max(self._ply_view_total_points, base_loaded))
        if self._ply_pre_append_points is None or self._ply_pre_append_colors is None:
            self._ply_pre_append_points = self._ply_view_points.astype(
                np.float32,
                copy=True,
            )
            self._ply_pre_append_colors = self._ply_view_colors.astype(
                np.uint8,
                copy=True,
            )
            self._ply_pre_append_point_ids = (
                self._ply_view_point_ids.astype(np.int64, copy=True)
                if self._ply_view_point_ids is not None
                else None
            )
            self._ply_pre_append_total_points = int(self._ply_view_total_points)
            self._ply_pre_append_sample_step = int(self._ply_view_sample_step)
        self._clear_ply_remove_snapshot()

        merged_points = [base_points] + append_points
        merged_colors = [base_colors] + append_colors
        self._ply_view_points = np.concatenate(merged_points, axis=0).astype(
            np.float32, copy=False
        )
        self._ply_view_colors = np.concatenate(merged_colors, axis=0).astype(
            np.uint8, copy=False
        )
        if self._ply_view_point_ids is not None:
            merged_ids = [self._ply_view_point_ids.astype(np.int64, copy=False)]
            for pts in append_points:
                merged_ids.append(
                    np.full(pts.shape[0], -1, dtype=np.int64)
                )
            self._ply_view_point_ids = np.concatenate(
                merged_ids, axis=0
            ).astype(np.int64, copy=False)
        self._ply_view_points_centered = self._ply_view_points.astype(
            np.float32, copy=False
        )
        self._ply_view_total_points = int(base_original + added_original)
        self._ply_view_sample_step = 1

        spans = np.maximum(
            self._ply_view_points.max(axis=0) - self._ply_view_points.min(axis=0),
            1e-6,
        )
        max_extent = float(np.max(spans))
        self._ply_view_max_extent = max_extent
        self._ply_view_depth_offset = max_extent * 2.5
        self._refresh_ply_view_rgb_mean()
        self._update_ply_view_info_from_current()
        self._redraw_ply_canvas()

        self._append_text_widget(
            self.ply_log,
            (
                "[viewer-append] appended {} file(s): +{:,.0f} loaded pts "
                "(+{:,.0f} source pts).".format(
                    len(append_points),
                    float(added_loaded),
                    float(added_original),
                )
            ),
        )
        if failed:
            self._append_text_widget(
                self.ply_log,
                "[viewer-append] failed {} file(s): {}".format(
                    len(failed),
                    "; ".join(failed[:5]),
                ),
            )

    def _run_ply_optimizer(self) -> None:
        if not self.ply_vars:
            return
        input_path = self.ply_vars["input"].get().strip()
        if not input_path:
            messagebox.showerror(
                "PointCloudOptimizer",
                "Input point cloud is required.",
            )
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
                messagebox.showerror(
                    "PointCloudOptimizer",
                    "Target points must be an integer.",
                )
                return
            cmd.extend(["-t", target_value])
        elif mode_key == "percent" and target_value:
            try:
                float(target_value)
            except ValueError:
                messagebox.showerror(
                    "PointCloudOptimizer",
                    "Target percent must be numeric.",
                )
                return
            cmd.extend(["-r", target_value])
        elif mode_key == "voxel" and target_value:
            try:
                float(target_value)
            except ValueError:
                messagebox.showerror(
                    "PointCloudOptimizer",
                    "Voxel size must be numeric.",
                )
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
                    messagebox.showerror(
                        "PointCloudOptimizer",
                        "Adaptive weight must be numeric.",
                    )
                    return
                cmd.extend(["--adaptive-weight", weight_text])

        keep_strategy = self.ply_vars["keep_strategy"].get().strip()
        if keep_strategy:
            cmd.extend(["-k", keep_strategy])

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
        path = self._get_ply_var_path("output")
        if path is None:
            messagebox.showerror(
                "Show Point Cloud",
                "Output point cloud path is not set.",
            )
            return
        self._show_ply_from_path(path, "output")

    def _on_show_input_ply(self) -> None:
        path = self._get_ply_var_path("input")
        if path is None:
            messagebox.showerror(
                "Show Point Cloud",
                "Input point cloud path is not set.",
            )
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
        self._ply_view_info_var.set("Point cloud viewer is idle")
        self._ply_view_points = None
        self._ply_view_points_centered = None
        self._ply_view_colors = None
        self._ply_view_point_ids = None
        self._ply_view_center = np.zeros(3, dtype=np.float32)
        self._ply_sky_points = None
        self._ply_sky_colors = None
        self._ply_exp_points = None
        self._ply_exp_colors = None
        self._reset_ply_exp_bbox_defaults()
        self._ply_loaded_points = None
        self._ply_loaded_colors = None
        self._ply_loaded_point_ids = None
        self._ply_loaded_total_points = 0
        self._ply_loaded_sample_step = 1
        self._ply_pre_append_points = None
        self._ply_pre_append_colors = None
        self._ply_pre_append_point_ids = None
        self._ply_pre_append_total_points = 0
        self._ply_pre_append_sample_step = 1
        self._clear_ply_remove_snapshot()
        self._ply_view_total_points = 0
        self._ply_view_sample_step = 1
        self._ply_view_source_label = "PLY"
        self._ply_current_file_path = None
        self._ply_source_kind = "ply"
        self._ply_colmap_model = None
        self._ply_sky_save_path_var.set("")
        self._ply_view_rgb_mean = None
        self._ply_initial_view_state = None
        self._ply_rendered_grid_span = 0.0
        self._ply_render_sample_step = 1
        self._ply_rendered_point_count = 0
        remove_color = self._parse_color_to_rgb(self._ply_sky_color_var.get())
        if remove_color is not None:
            sky_hex = self._ply_sky_color_var.get()
            self._ply_remove_color_var.set(sky_hex)
            self._update_remove_color_display(remove_color, sky_hex)
        self._reset_ply_view_transform()
        self._redraw_ply_canvas()

    def _show_ply_from_path(self, path: Path, label: str) -> None:
        if not path.exists():
            messagebox.showerror("Show Point Cloud", f"{path} was not found.")
            return
        self._update_ply_high_max_default_from_path(path)
        self._ply_view_source_label = label
        canvas = self._ensure_ply_viewer_window()
        if canvas is None:
            messagebox.showerror(
                "Show Point Cloud",
                "Point cloud viewer canvas is not available.",
            )
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
                    "Point Cloud Viewer",
                    "Interactive Render Points must be numeric.",
                )
            return None
        if value <= 0:
            if show_error:
                messagebox.showerror(
                    "Point Cloud Viewer",
                    "Interactive Render Points must be greater than zero.",
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

    def _read_colmap_point_count(self, path: Path) -> Optional[int]:
        points_path = path / "points3D.txt"
        if not points_path.exists():
            return None
        count = 0
        try:
            with points_path.open("r", encoding="utf-8") as handle:
                for raw in handle:
                    line = raw.strip()
                    if line and not line.startswith("#"):
                        count += 1
        except Exception:
            return None
        return count

    def _update_ply_high_max_default_from_path(self, path: Path) -> None:
        if self._is_colmap_text_model_dir(path):
            vertex_count = self._read_colmap_point_count(path)
        else:
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
                    "Point Cloud Viewer",
                    "Final Render Points must be numeric.",
                )
            return None
        if value <= 0:
            if show_error:
                messagebox.showerror(
                    "Point Cloud Viewer",
                    "Final Render Points must be greater than zero.",
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

    def _load_point_cloud_view_data(
        self,
        path: Path,
        max_points: Optional[int],
    ) -> Tuple[np.ndarray, np.ndarray, int, int, Dict[str, Any]]:
        if self._is_colmap_text_model_dir(path):
            loaded = pointcloud_optimizer.load_point_cloud_input(str(path))
            total_count = int(loaded.xyz.shape[0])
            if max_points is None or max_points <= 0:
                sample_step = 1
            else:
                sample_step = self._compute_sample_step(total_count, max_points)
            points = loaded.xyz[::sample_step].astype(np.float32, copy=False)
            colors = loaded.rgb[::sample_step].astype(np.uint8, copy=False)
            point_ids = None
            if loaded.point_ids is not None:
                point_ids = loaded.point_ids[::sample_step].astype(
                    np.int64,
                    copy=False,
                )
            meta = {
                "input_kind": "colmap",
                "point_ids": point_ids,
                "colmap_model": loaded.colmap_model,
            }
            return points, colors, total_count, max(1, sample_step), meta

        points, colors, total_count, sample_step = self._load_binary_ply_points(
            path,
            max_points=max_points,
        )
        meta = {
            "input_kind": "ply",
            "point_ids": None,
            "colmap_model": None,
        }
        return points, colors, total_count, sample_step, meta

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
            points, colors, original_count, sample_step, meta = self._load_point_cloud_view_data(
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
                meta,
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
        meta: Dict[str, Any],
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
            messagebox.showinfo(
                "Show Point Cloud",
                "No points were available for display.",
            )
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
        point_ids = meta.get("point_ids")
        self._ply_view_point_ids = (
            point_ids.astype(np.int64, copy=False)
            if isinstance(point_ids, np.ndarray)
            else None
        )
        self._ply_view_max_extent = max_extent
        self._ply_view_depth_offset = max_extent * 2.5
        self._ply_view_center = center.astype(np.float32, copy=False)
        self._ply_current_file_path = path
        self._ply_source_kind = str(meta.get("input_kind") or "ply")
        self._ply_colmap_model = meta.get("colmap_model")
        self._ply_view_total_points = original_count
        self._ply_view_sample_step = max(1, sample_step)
        self._ply_loaded_points = points.astype(np.float32, copy=True)
        self._ply_loaded_colors = colors.astype(np.uint8, copy=True)
        self._ply_loaded_point_ids = (
            self._ply_view_point_ids.astype(np.int64, copy=True)
            if self._ply_view_point_ids is not None
            else None
        )
        self._ply_loaded_total_points = original_count
        self._ply_loaded_sample_step = max(1, sample_step)
        self._ply_pre_append_points = None
        self._ply_pre_append_colors = None
        self._ply_pre_append_point_ids = None
        self._ply_pre_append_total_points = 0
        self._ply_pre_append_sample_step = 1
        self._clear_ply_remove_snapshot()
        self._reset_ply_view_defaults()
        self._reset_ply_view_transform()
        self._capture_ply_initial_view_state()
        label = self._ply_view_source_label or "PLY"
        self._ply_view_info_var.set(
            self._build_ply_info_text(
                point_count,
                original_count,
                self._ply_view_sample_step,
            )
        )
        self._update_sky_save_default(path)
        default_sky_count = max(1, int(round(original_count * 0.05)))
        self._ply_sky_count_var.set(str(default_sky_count))
        self._ply_exp_points = None
        self._ply_exp_colors = None
        self._reset_ply_exp_bbox_defaults()
        remove_color = self._parse_color_to_rgb(self._ply_sky_color_var.get())
        if remove_color is not None:
            sky_hex = self._ply_sky_color_var.get()
            self._ply_remove_color_var.set(sky_hex)
            self._update_remove_color_display(remove_color, sky_hex)
        self._redraw_ply_canvas()

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
        self._ply_view_info_var.set("Point cloud viewer is idle")
        self._redraw_ply_canvas()
        messagebox.showerror(
            "Show Point Cloud",
            f"Failed to load the point cloud.\n{exc}",
        )

    def _compute_fit_zoom_pan(
        self,
        points: np.ndarray,
        quat: np.ndarray,
        max_extent: float,
        width: int,
        height: int,
        fill_ratio: float = 0.82,
    ) -> Tuple[float, Tuple[float, float]]:
        if points.size == 0:
            return 1.0, (0.0, 0.0)
        safe_extent = max(float(max_extent), 1e-6)
        safe_width = max(1, int(width))
        safe_height = max(1, int(height))
        usable_w = max(32.0, float(safe_width) * float(fill_ratio))
        usable_h = max(32.0, float(safe_height) * float(fill_ratio))
        rotation = self._quat_to_matrix(quat)
        rotated = points @ rotation.T
        x_coords = rotated[:, 0]
        y_coords = rotated[:, 1]
        abs_x = np.abs(x_coords)
        abs_y = np.abs(y_coords)
        half_span_x = max(float(np.quantile(abs_x, 0.90)), 1e-6)
        half_span_y = max(float(np.quantile(abs_y, 0.90)), 1e-6)
        base_scale = max(min(safe_width, safe_height) * 0.45 / safe_extent, 1e-6)
        zoom_x = usable_w / ((half_span_x * 2.0) * base_scale)
        zoom_y = usable_h / ((half_span_y * 2.0) * base_scale)
        zoom = max(0.1, min(PLY_VIEW_MAX_ZOOM, min(zoom_x, zoom_y)))
        return zoom, (0.0, 0.0)

    def _get_ply_viewport_size(self) -> Tuple[int, int]:
        canvas = self._ply_view_canvas
        width = 0
        height = 0
        if canvas is not None and canvas.winfo_exists():
            try:
                canvas.update_idletasks()
            except Exception:
                pass
            width = int(canvas.winfo_width())
            height = int(canvas.winfo_height())
        if width < 10 or height < 10:
            width = PLY_VIEW_CANVAS_WIDTH
            height = max(460, int(PLY_VIEW_CANVAS_HEIGHT * 0.55))
        return width, height

    def _compute_ply_initial_view(self) -> Tuple[float, Tuple[float, float]]:
        if self._ply_view_points_centered is None or self._ply_view_points_centered.size == 0:
            return 1.0, (0.0, 0.0)
        width, height = self._get_ply_viewport_size()
        return self._compute_fit_zoom_pan(
            self._ply_view_points_centered,
            self._ply_view_quat,
            self._ply_view_max_extent,
            width,
            height,
            fill_ratio=0.84,
        )

    def _reset_ply_view_transform(self) -> None:
        yaw, pitch = self._get_default_display_view_angles(
            self._ply_display_up_axis_var.get()
        )
        q_yaw = self._quat_from_axis_angle((0.0, 1.0, 0.0), yaw)
        q_pitch = self._quat_from_axis_angle((1.0, 0.0, 0.0), pitch)
        self._ply_view_quat = self._quat_normalize(
            self._quat_multiply(q_pitch, q_yaw)
        )
        self._ply_view_zoom, self._ply_view_pan = self._compute_ply_initial_view()
        self._ply_drag_last = None
        self._ply_pan_last = None

    def _capture_ply_initial_view_state(self) -> None:
        self._ply_initial_view_state = {
            "quat": np.asarray(self._ply_view_quat, dtype=np.float32).copy(),
            "zoom": float(self._ply_view_zoom),
            "pan": (
                float(self._ply_view_pan[0]),
                float(self._ply_view_pan[1]),
            ),
            "projection_mode": self._ply_projection_mode.get(),
            "display_up_axis": self._ply_display_up_axis_var.get(),
            "interactive_point_cap": self._ply_view_interactive_max_points_var.get(),
            "point_cap": self._ply_view_high_max_points_var.get(),
            "point_size": self._ply_point_size_var.get(),
            "grid_step": self._ply_grid_step_var.get(),
            "grid_span": self._ply_grid_span_var.get(),
            "draw_points": bool(self._ply_draw_points_var.get()),
            "monochrome": bool(self._ply_monochrome_var.get()),
            "front_occlusion": bool(self._ply_front_occlusion_var.get()),
            "show_world_axes": bool(self._ply_show_world_axes_var.get()),
            "show_grid": bool(self._ply_show_grid_var.get()),
        }

    def _reset_ply_view_defaults(self) -> None:
        self._ply_projection_mode.set("Orthographic")
        self._ply_display_up_axis_var.set("Z-up")
        self._ply_view_interactive_max_points_var.set(
            str(PLY_VIEW_INTERACTIVE_MAX_POINTS)
        )
        self._ply_view_high_max_points_var.set(str(PLY_VIEW_MAX_POINTS))
        self._ply_point_size_var.set("2")
        self._ply_grid_step_var.set("1.0")
        self._ply_grid_span_var.set("auto")
        self._ply_draw_points_var.set(True)
        self._ply_monochrome_var.set(False)
        self._ply_front_occlusion_var.set(True)
        self._ply_show_world_axes_var.set(True)
        self._ply_show_grid_var.set(True)
        self._enforce_ply_depth_view_constraints()

    def _reset_ply_view(self) -> None:
        state = self._ply_initial_view_state
        if state is None:
            self._reset_ply_view_transform()
            self._redraw_ply_canvas()
            return
        self._ply_view_quat = np.asarray(state["quat"], dtype=np.float32).copy()
        self._ply_view_zoom = float(state["zoom"])
        self._ply_view_pan = (
            float(state["pan"][0]),
            float(state["pan"][1]),
        )
        self._ply_drag_last = None
        self._ply_pan_last = None
        self._ply_projection_mode.set(str(state["projection_mode"]))
        self._ply_display_up_axis_var.set(
            self._normalize_display_up_axis(
                str(state.get("display_up_axis", "Z-up"))
            )
        )
        self._ply_view_interactive_max_points_var.set(
            str(state["interactive_point_cap"])
        )
        self._ply_view_high_max_points_var.set(str(state["point_cap"]))
        self._ply_point_size_var.set(str(state["point_size"]))
        self._ply_grid_step_var.set(str(state["grid_step"]))
        self._ply_grid_span_var.set(str(state["grid_span"]))
        self._ply_draw_points_var.set(bool(state["draw_points"]))
        self._ply_monochrome_var.set(bool(state["monochrome"]))
        self._ply_front_occlusion_var.set(
            bool(state.get("front_occlusion", True))
        )
        self._ply_show_world_axes_var.set(bool(state["show_world_axes"]))
        self._ply_show_grid_var.set(bool(state["show_grid"]))
        self._enforce_ply_depth_view_constraints()
        self._redraw_ply_canvas(force=True)

    def _on_reset_ply_camera_view(self) -> None:
        self._reset_ply_view_transform()
        self._redraw_ply_canvas(force=True)

    def _enforce_ply_depth_view_constraints(self) -> None:
        if bool(self._ply_monochrome_var.get()):
            self._ply_front_occlusion_var.set(True)
        state = "disabled" if bool(self._ply_monochrome_var.get()) else "normal"
        if self._ply_front_occlusion_checkbutton is not None:
            try:
                self._ply_front_occlusion_checkbutton.configure(state=state)
            except Exception:
                pass

    def _on_ply_depth_view_toggle(self) -> None:
        self._enforce_ply_depth_view_constraints()
        self._redraw_ply_canvas()

    def _on_ply_depth_occlusion_toggle(self) -> None:
        self._enforce_ply_depth_view_constraints()
        self._redraw_ply_canvas()

    def _get_ply_grid_step(self) -> float:
        text = self._ply_grid_step_var.get().strip()
        if not text:
            return 1.0
        try:
            value = float(text)
        except Exception:
            return 1.0
        if value <= 0.0:
            return 1.0
        return value

    def _get_ply_grid_span(self) -> Optional[float]:
        text = self._ply_grid_span_var.get().strip()
        if not text or text.lower() == "auto":
            return None
        try:
            value = float(text)
        except Exception:
            return None
        if value <= 0.0:
            return None
        return value

    def _get_ply_point_size(self) -> int:
        text = self._ply_point_size_var.get().strip()
        if not text:
            return 1
        try:
            value = int(round(float(text)))
        except Exception:
            return 1
        if value < 1:
            return 1
        return min(value, 9)

    def _get_ply_axis_length(self) -> float:
        return max(self._ply_view_max_extent * 0.2, 1e-3)

    def _redraw_ply_canvas(self, force: bool = False) -> None:
        canvas = self._ply_view_canvas
        if canvas is None or not canvas.winfo_exists():
            return
        if not force and not self._is_ply_tab_active():
            self._ply_redraw_pending = True
            return
        self._ply_redraw_pending = False
        self._render_ply_points(canvas)

    def _on_ply_canvas_configure(self, _event=None) -> None:
        self._redraw_ply_canvas()

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
        try:
            event.widget.focus_set()
        except Exception:
            pass
        if self._try_begin_ply_exp_drag(event):
            return
        self._begin_ply_interaction()
        self._ply_drag_last = (event.x, event.y)

    def _on_ply_drag_move(self, event: tk.Event) -> None:
        if self._ply_view_points_centered is None:
            return
        if self._ply_exp_drag_kind and self._ply_exp_drag_last is not None:
            last_x, last_y = self._ply_exp_drag_last
            dx = event.x - last_x
            dy = event.y - last_y
            self._ply_exp_drag_last = (event.x, event.y)
            if self._ply_exp_drag_kind in ("move", "move_axis"):
                self._move_ply_exp_bbox_by_screen_delta(dx, dy)
            elif self._ply_exp_drag_kind == "scale":
                self._scale_ply_exp_bbox_by_screen_delta(dx, dy)
            self._sync_ply_exp_bbox_vars_from_state()
            self._redraw_ply_canvas()
            return
        if self._ply_drag_last is None:
            return
        last_x, last_y = self._ply_drag_last
        dx = event.x - last_x
        dy = event.y - last_y
        self._ply_drag_last = (event.x, event.y)
        angle_y = math.radians(-dx * 0.35)
        angle_x = math.radians(-dy * 0.35)
        q_y = self._quat_from_axis_angle((0.0, 1.0, 0.0), angle_y)
        q_x = self._quat_from_axis_angle((1.0, 0.0, 0.0), angle_x)
        q_inc = self._quat_multiply(q_x, q_y)
        self._ply_view_quat = self._quat_normalize(
            self._quat_multiply(q_inc, self._ply_view_quat)
        )
        self._redraw_ply_canvas()

    def _on_ply_drag_end(self, _event=None) -> None:
        self._ply_drag_last = None
        self._ply_exp_drag_kind = None
        self._ply_exp_drag_axis = None
        self._ply_exp_drag_last = None
        self._ply_exp_drag_axis_world = np.zeros(3, dtype=np.float32)
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
        sky_percent: float,
        color_rgb: Tuple[int, int, int],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        direction = self._axis_direction(axis_label)
        if direction is None:
            return None, None
        count = max(1000, min(20000, int(count)))
        indices = np.arange(count, dtype=np.float32)
        phi = math.pi * (3.0 - math.sqrt(5.0))
        coverage = float(np.clip(sky_percent, 0.0, 100.0)) / 100.0
        z_min = 1.0 - 2.0 * coverage
        z = 1.0 - (indices / count) * (1.0 - z_min)
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

    def _on_reset_ply_remove_controls(self) -> None:
        default_hex = self._ply_sky_color_var.get().strip() or "#87cefa"
        self._ply_remove_color_var.set(default_hex)
        self._ply_remove_color_tol_var.set("125")
        rgb = self._hex_to_rgb(default_hex)
        if rgb is not None:
            self._update_remove_color_display(rgb, default_hex)
        if self._ply_pre_remove_points is not None and self._ply_pre_remove_colors is not None:
            self._ply_view_points = self._ply_pre_remove_points.astype(
                np.float32,
                copy=True,
            )
            self._ply_view_colors = self._ply_pre_remove_colors.astype(
                np.uint8,
                copy=True,
            )
            self._ply_view_point_ids = (
                self._ply_pre_remove_point_ids.astype(np.int64, copy=True)
                if self._ply_pre_remove_point_ids is not None
                else None
            )
            self._ply_view_points_centered = self._ply_view_points.astype(
                np.float32,
                copy=False,
            )
            self._ply_view_total_points = int(
                max(
                    self._ply_pre_remove_total_points,
                    self._ply_view_points.shape[0],
                )
            )
            self._ply_view_sample_step = max(
                1,
                int(self._ply_pre_remove_sample_step),
            )
            self._ply_sky_points = (
                self._ply_pre_remove_sky_points.astype(np.float32, copy=True)
                if self._ply_pre_remove_sky_points is not None
                else None
            )
            self._ply_sky_colors = (
                self._ply_pre_remove_sky_colors.astype(np.uint8, copy=True)
                if self._ply_pre_remove_sky_colors is not None
                else None
            )
            self._clear_ply_remove_snapshot()
            self._refresh_ply_view_rgb_mean()
            self._update_ply_view_info_from_current()
            self._redraw_ply_canvas()

    def _sample_auto_sky_color(self) -> Optional[Tuple[int, int, int]]:
        if self._ply_view_points is None or self._ply_view_colors is None:
            return None
        points = self._ply_view_points
        colors = self._ply_view_colors
        if points.size == 0 or colors.size == 0:
            return None
        count = min(int(points.shape[0]), int(colors.shape[0]))
        if count <= 0:
            return None
        points_arr = points[:count].astype(np.float32, copy=False)
        colors_arr = colors[:count].astype(np.uint8, copy=False)
        center = points_arr.mean(axis=0, dtype=np.float64)
        diff = points_arr - center
        dist2 = np.einsum("ij,ij->i", diff, diff, optimize=True)
        if dist2.size <= 0:
            return None
        farthest_idx = int(np.argmax(dist2))
        rgb_arr = colors_arr[farthest_idx]
        rgb = (int(rgb_arr[0]), int(rgb_arr[1]), int(rgb_arr[2]))
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
                    "Point Cloud Viewer",
                    "Remove color tolerance must be numeric.",
                )
            return None
        if value < 0:
            if show_error:
                messagebox.showerror(
                    "Point Cloud Viewer",
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

    def _update_ply_view_info_from_current(self) -> None:
        """Refresh the viewer info text from current base/sky point buffers."""
        if self._ply_view_points is None:
            return
        point_count = int(self._ply_view_points.shape[0])
        original_count = int(
            max(self._ply_view_total_points, point_count)
        )
        info = self._build_ply_info_text(
            point_count,
            original_count,
            self._ply_view_sample_step,
        )
        self._ply_view_info_var.set(info)

    def _build_ply_info_text(
        self, point_count: int, original_count: int, sample_step: int
    ) -> str:
        label = self._ply_view_source_label or "PLY"
        base_count = max(0, int(point_count))
        src_count = max(0, int(original_count))
        sky_count = 0
        exp_count = 0
        if self._ply_sky_points is not None:
            sky_count = int(self._ply_sky_points.shape[0])
        if self._ply_exp_points is not None:
            exp_count = int(self._ply_exp_points.shape[0])
        total_count = base_count + sky_count + exp_count
        if src_count > 0 and (sample_step > 1 or src_count != base_count):
            suffix = (
                f"{base_count:,} / {src_count:,} pts"
                if sample_step <= 1
                else f"{base_count:,} / {src_count:,} pts, step {sample_step}"
            )
        else:
            suffix = f"{base_count:,} pts"
        if sky_count > 0:
            suffix = f"{suffix} + sky {sky_count:,}"
        if exp_count > 0:
            suffix = f"{suffix} + exp {exp_count:,}"
        if sky_count > 0 or exp_count > 0:
            suffix = f"{suffix} = {total_count:,}"
        return f"{label} ({suffix})"

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
            messagebox.showerror(
                "Sky PointCloud",
                "Load a point cloud before adding sky points.",
            )
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
        percent_text = self._ply_sky_percent_var.get().strip() or "50"
        try:
            percent_value = float(percent_text)
        except ValueError:
            messagebox.showerror(
                "Sky PointCloud",
                "Sky sphere % must be numeric.",
            )
            return
        if percent_value <= 0 or percent_value > 100:
            messagebox.showerror(
                "Sky PointCloud",
                "Sky sphere % must be > 0 and <= 100.",
            )
            return
        color_text = self._ply_sky_color_var.get().strip() or "#87cefa"
        color_rgb = self._parse_color_to_rgb(color_text)
        if color_rgb is None:
            messagebox.showerror("Sky PointCloud", "Sky color must be a valid color (e.g., #87cefa).")
            return
        self._update_sky_color_display(color_rgb, color_text)
        points, colors = self._generate_sky_points(
            axis_label,
            scale_value,
            count_value,
            percent_value,
            color_rgb,
        )
        if points is None or colors is None:
            messagebox.showerror("Sky PointCloud", "Failed to generate sky points. Check axis selection.")
            return
        self._clear_ply_remove_snapshot()
        self._ply_sky_points = points
        self._ply_sky_colors = colors
        self._update_ply_view_info_from_current()
        self._redraw_ply_canvas()

    def _on_clear_sky_points(self) -> None:
        if self._ply_sky_points is None and self._ply_sky_colors is None:
            return
        self._clear_ply_remove_snapshot()
        self._ply_sky_points = None
        self._ply_sky_colors = None
        self._update_ply_view_info_from_current()
        self._redraw_ply_canvas()

    def _format_scalar_text(self, value: float) -> str:
        return f"{float(value):.6g}"

    def _set_ply_exp_axis_vars(
        self,
        axis_vars: Tuple[tk.StringVar, tk.StringVar, tk.StringVar],
        values: np.ndarray,
    ) -> None:
        arr = np.asarray(values, dtype=np.float32).reshape(3)
        for var, value in zip(axis_vars, arr.tolist()):
            var.set(self._format_scalar_text(float(value)))

    def _parse_ply_exp_axis_vars(
        self,
        axis_vars: Tuple[tk.StringVar, tk.StringVar, tk.StringVar],
        *,
        label: str,
        positive: bool = False,
    ) -> np.ndarray:
        values: List[float] = []
        for axis_label, axis_var in zip(("X", "Y", "Z"), axis_vars):
            text = axis_var.get().strip()
            if not text:
                raise ValueError(f"{label} {axis_label} must not be empty.")
            try:
                value = float(text)
            except Exception:
                raise ValueError(f"{label} {axis_label} must be numeric.")
            if positive and value <= 0.0:
                raise ValueError(f"{label} {axis_label} must be greater than zero.")
            values.append(value)
        return np.array(values, dtype=np.float32)

    def _reset_ply_exp_bbox_defaults(self) -> None:
        if self._ply_view_points is None or self._ply_view_points.size == 0:
            center = np.zeros(3, dtype=np.float32)
            size = np.ones(3, dtype=np.float32)
        else:
            xyz_min = self._ply_view_points.min(axis=0).astype(np.float32)
            xyz_max = self._ply_view_points.max(axis=0).astype(np.float32)
            center = ((xyz_min + xyz_max) * 0.5).astype(np.float32, copy=False)
            size = np.maximum(xyz_max - xyz_min, 1e-3).astype(
                np.float32,
                copy=False,
            )
        self._ply_exp_bbox_center = center
        self._ply_exp_bbox_size = size
        self._ply_exp_bbox_rotation = np.eye(3, dtype=np.float32)
        self._sync_ply_exp_bbox_vars_from_state()

    def _apply_ply_exp_bbox_from_vars(self, show_error: bool = True) -> bool:
        try:
            center = self._parse_ply_exp_axis_vars(
                (
                    self._ply_exp_bbox_center_x_var,
                    self._ply_exp_bbox_center_y_var,
                    self._ply_exp_bbox_center_z_var,
                ),
                label="BBox Center",
            )
            size = self._parse_ply_exp_axis_vars(
                (
                    self._ply_exp_bbox_size_x_var,
                    self._ply_exp_bbox_size_y_var,
                    self._ply_exp_bbox_size_z_var,
                ),
                label="BBox Size",
                positive=True,
            )
        except Exception as exc:
            if show_error:
                messagebox.showerror("BBox Scatter", str(exc))
            return False
        self._ply_exp_bbox_center = center.astype(np.float32, copy=False)
        self._ply_exp_bbox_size = np.maximum(size, 1e-3).astype(
            np.float32,
            copy=False,
        )
        self._ply_exp_bbox_rotation = np.eye(3, dtype=np.float32)
        return True

    def _get_ply_exp_bbox_corners(self) -> Optional[np.ndarray]:
        if self._ply_view_points is None or self._ply_view_points.size == 0:
            return None
        half = np.maximum(self._ply_exp_bbox_size * 0.5, 1e-6)
        signs = np.array(
            [
                [-1.0, -1.0, -1.0],
                [1.0, -1.0, -1.0],
                [1.0, 1.0, -1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0],
                [1.0, -1.0, 1.0],
                [1.0, 1.0, 1.0],
                [-1.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )
        local = signs * half.reshape(1, 3)
        return (
            local @ self._ply_exp_bbox_rotation.T + self._ply_exp_bbox_center
        ).astype(np.float32, copy=False)

    def _sync_ply_exp_bbox_vars_from_state(self) -> None:
        self._set_ply_exp_axis_vars(
            (
                self._ply_exp_bbox_center_x_var,
                self._ply_exp_bbox_center_y_var,
                self._ply_exp_bbox_center_z_var,
            ),
            self._ply_exp_bbox_center,
        )
        self._set_ply_exp_axis_vars(
            (
                self._ply_exp_bbox_size_x_var,
                self._ply_exp_bbox_size_y_var,
                self._ply_exp_bbox_size_z_var,
            ),
            self._ply_exp_bbox_size,
        )

    def _set_ply_exp_edit_mode(self, mode: str) -> None:
        normalized = "Scale" if (mode or "").strip().lower().startswith("s") else "Move"
        self._ply_exp_edit_mode_var.set(normalized)
        self._redraw_ply_canvas()

    def _on_ply_exp_edit_mode_selected(self, _event=None) -> Optional[str]:
        self._set_ply_exp_edit_mode(self._ply_exp_edit_mode_var.get())
        return "break"

    def _on_ply_exp_mode_move(self, _event=None) -> Optional[str]:
        self._set_ply_exp_edit_mode("Move")
        return "break"

    def _on_ply_exp_mode_scale(self, _event=None) -> Optional[str]:
        self._set_ply_exp_edit_mode("Scale")
        return "break"

    def _get_ply_view_render_context(
        self,
    ) -> Tuple[int, int, np.ndarray, float, float, bool, float, float]:
        canvas = self._ply_view_canvas
        width = PLY_VIEW_CANVAS_WIDTH
        height = PLY_VIEW_CANVAS_HEIGHT
        if canvas is not None and canvas.winfo_exists():
            try:
                canvas.update_idletasks()
            except Exception:
                pass
            width = max(1, int(canvas.winfo_width()))
            height = max(1, int(canvas.winfo_height()))
        if width < 10 or height < 10:
            width = PLY_VIEW_CANVAS_WIDTH
            height = PLY_VIEW_CANVAS_HEIGHT
        rotation = self._quat_to_matrix(self._ply_view_quat)
        proj_scale = min(width, height) * 0.9 * self._ply_view_zoom
        ortho_scale = self._compute_ortho_scale(width, height)
        projection_mode = (self._ply_projection_mode.get() or "").lower()
        is_orthographic = projection_mode.startswith("ortho")
        pan_x, pan_y = self._ply_view_pan
        return (
            width,
            height,
            rotation,
            proj_scale,
            ortho_scale,
            is_orthographic,
            float(pan_x),
            float(pan_y),
        )

    def _get_ply_exp_handle_positions(
        self,
    ) -> Optional[Dict[str, object]]:
        corners = self._get_ply_exp_bbox_corners()
        if corners is None:
            return None
        display_matrix = self._get_ply_display_up_axis_matrix()
        (
            width,
            height,
            rotation,
            proj_scale,
            ortho_scale,
            is_orthographic,
            pan_x,
            pan_y,
        ) = self._get_ply_view_render_context()
        center_pt = self._project_point_to_screen(
            tuple(float(v) for v in self._ply_exp_bbox_center),
            width,
            height,
            rotation,
            self._ply_view_depth_offset,
            proj_scale,
            ortho_scale,
            is_orthographic,
            pan_x,
            pan_y,
            display_matrix=display_matrix,
        )
        axis_handles: List[Optional[Tuple[float, float]]] = []
        axis_world: List[np.ndarray] = []
        half = np.maximum(self._ply_exp_bbox_size * 0.5, 1e-6)
        for axis_idx in range(3):
            axis_vec = self._ply_exp_bbox_rotation.T[:, axis_idx].astype(
                np.float32,
                copy=False,
            )
            axis_world.append(axis_vec)
            endpoint = self._ply_exp_bbox_center + axis_vec * half[axis_idx]
            axis_handles.append(
                self._project_point_to_screen(
                    tuple(float(v) for v in endpoint),
                    width,
                    height,
                    rotation,
                    self._ply_view_depth_offset,
                    proj_scale,
                    ortho_scale,
                    is_orthographic,
                    pan_x,
                    pan_y,
                    display_matrix=display_matrix,
                )
            )
        return {
            "center": center_pt,
            "axes": axis_handles,
            "axis_world": axis_world,
            "context": (
                width,
                height,
                rotation,
                proj_scale,
                ortho_scale,
                is_orthographic,
                pan_x,
                pan_y,
            ),
        }

    def _try_begin_ply_exp_drag(self, event: tk.Event) -> bool:
        if not bool(self._ply_exp_bbox_active_var.get()):
            return False
        handles = self._get_ply_exp_handle_positions()
        if handles is None:
            return False
        center_pt = handles["center"]
        if center_pt is None:
            return False
        event.widget.focus_set()
        mode = (self._ply_exp_edit_mode_var.get() or "Move").strip().lower()
        hit_radius_sq = 12.0 * 12.0
        dx_center = float(center_pt[0] - event.x)
        dy_center = float(center_pt[1] - event.y)
        if mode.startswith("m") and (dx_center * dx_center + dy_center * dy_center) <= hit_radius_sq:
            self._ply_exp_drag_kind = "move"
            self._ply_exp_drag_axis = None
            self._ply_exp_drag_last = (event.x, event.y)
            self._ply_exp_drag_center_start = self._ply_exp_bbox_center.astype(
                np.float32,
                copy=True,
            )
            self._begin_ply_interaction()
            return True
        axis_handles = handles["axes"]
        axis_world = handles["axis_world"]
        for axis_idx, axis_pt in enumerate(axis_handles):
            if axis_pt is None:
                continue
            dx = float(axis_pt[0] - event.x)
            dy = float(axis_pt[1] - event.y)
            if (dx * dx + dy * dy) > hit_radius_sq:
                continue
            screen_vec = np.array(
                [axis_pt[0] - center_pt[0], axis_pt[1] - center_pt[1]],
                dtype=np.float32,
            )
            length = float(np.linalg.norm(screen_vec))
            if length <= 1e-6:
                continue
            half_extent = max(float(self._ply_exp_bbox_size[axis_idx] * 0.5), 1e-6)
            self._ply_exp_drag_kind = "scale" if mode.startswith("s") else "move_axis"
            self._ply_exp_drag_axis = axis_idx
            self._ply_exp_drag_last = (event.x, event.y)
            self._ply_exp_drag_size_start = self._ply_exp_bbox_size.astype(
                np.float32,
                copy=True,
            )
            self._ply_exp_drag_axis_world = np.asarray(
                axis_world[axis_idx],
                dtype=np.float32,
            )
            self._ply_exp_drag_screen_dir = (
                screen_vec / length
            ).astype(np.float32, copy=False)
            self._ply_exp_drag_pixels_per_world = length / half_extent
            self._begin_ply_interaction()
            return True
        return False

    def _move_ply_exp_bbox_by_screen_delta(self, dx: int, dy: int) -> None:
        display_matrix = self._get_ply_display_up_axis_matrix()
        if self._ply_exp_drag_kind == "move_axis" and self._ply_exp_drag_axis is not None:
            handles = self._get_ply_exp_handle_positions()
            if handles is not None:
                center_pt = handles["center"]
                axis_pt = handles["axes"][self._ply_exp_drag_axis]
                if center_pt is not None and axis_pt is not None:
                    screen_vec = np.array(
                        [axis_pt[0] - center_pt[0], axis_pt[1] - center_pt[1]],
                        dtype=np.float32,
                    )
                    length = float(np.linalg.norm(screen_vec))
                    if length > 1e-6:
                        half_extent = max(
                            float(self._ply_exp_bbox_size[self._ply_exp_drag_axis] * 0.5),
                            1e-6,
                        )
                        self._ply_exp_drag_screen_dir = (
                            screen_vec / length
                        ).astype(np.float32, copy=False)
                        self._ply_exp_drag_pixels_per_world = length / half_extent
            screen_dir = self._ply_exp_drag_screen_dir.astype(np.float32, copy=False)
            drag_pixels = float(dx) * float(screen_dir[0]) + float(dy) * float(screen_dir[1])
            pixels_per_world = max(self._ply_exp_drag_pixels_per_world, 1e-6)
            axis_delta = drag_pixels / pixels_per_world
            self._ply_exp_bbox_center = (
                self._ply_exp_bbox_center
                + self._ply_exp_drag_axis_world.astype(np.float32, copy=False) * axis_delta
            ).astype(np.float32, copy=False)
            return
        (
            _width,
            _height,
            rotation,
            proj_scale,
            ortho_scale,
            is_orthographic,
            _pan_x,
            _pan_y,
        ) = self._get_ply_view_render_context()
        center_display = display_matrix @ self._ply_exp_bbox_center.astype(np.float32)
        center_view = rotation @ center_display
        depth = float(center_view[2] + max(self._ply_view_depth_offset, 1e-6))
        if is_orthographic:
            screen_scale = max(ortho_scale, 1e-6)
        else:
            screen_scale = max(proj_scale / max(depth, 1e-6), 1e-6)
        delta_view = np.array(
            [float(dx) / screen_scale, -float(dy) / screen_scale, 0.0],
            dtype=np.float32,
        )
        delta_world = display_matrix.T @ (rotation.T @ delta_view)
        self._ply_exp_bbox_center = (
            self._ply_exp_bbox_center + delta_world
        ).astype(np.float32, copy=False)

    def _scale_ply_exp_bbox_by_screen_delta(self, dx: int, dy: int) -> None:
        axis_idx = self._ply_exp_drag_axis
        if axis_idx is None:
            return
        handles = self._get_ply_exp_handle_positions()
        if handles is not None:
            center_pt = handles["center"]
            axis_pt = handles["axes"][axis_idx]
            if center_pt is not None and axis_pt is not None:
                screen_vec = np.array(
                    [axis_pt[0] - center_pt[0], axis_pt[1] - center_pt[1]],
                    dtype=np.float32,
                )
                length = float(np.linalg.norm(screen_vec))
                if length > 1e-6:
                    half_extent = max(float(self._ply_exp_bbox_size[axis_idx] * 0.5), 1e-6)
                    self._ply_exp_drag_screen_dir = (
                        screen_vec / length
                    ).astype(np.float32, copy=False)
                    self._ply_exp_drag_pixels_per_world = length / half_extent
        screen_dir = self._ply_exp_drag_screen_dir.astype(np.float32, copy=False)
        drag_pixels = float(dx) * float(screen_dir[0]) + float(dy) * float(screen_dir[1])
        pixels_per_world = max(self._ply_exp_drag_pixels_per_world, 1e-6)
        half_delta = drag_pixels / pixels_per_world
        new_size = self._ply_exp_bbox_size.astype(np.float32, copy=True)
        new_half = max(1e-3, float(new_size[axis_idx] * 0.5) + half_delta)
        new_size[axis_idx] = new_half * 2.0
        self._ply_exp_bbox_size = new_size.astype(np.float32, copy=False)

    def _get_ply_exp_color_count(self) -> int:
        text = self._ply_exp_color_count_var.get().strip()
        if not text:
            raise ValueError("Color count must not be empty.")
        try:
            count = int(float(text))
        except Exception:
            raise ValueError("Color count must be numeric.")
        if count <= 0:
            raise ValueError("Color count must be greater than zero.")
        return count

    def _get_ply_exp_source_points_and_colors(
        self,
    ) -> Tuple[np.ndarray, np.ndarray]:
        points = self._ply_view_points
        colors = self._ply_view_colors
        if (
            points is None
            or colors is None
            or points.size == 0
            or colors.size == 0
        ):
            raise ValueError("Load a point cloud before sampling scatter colors.")
        return (
            points.astype(np.float32, copy=False),
            colors.astype(np.uint8, copy=False),
        )

    def _sample_ply_exp_edge_palette(self, palette_size: int) -> np.ndarray:
        points, colors = self._get_ply_exp_source_points_and_colors()
        half = np.maximum(self._ply_exp_bbox_size * 0.5, 1e-6).reshape(1, 3)
        local = (
            (points - self._ply_exp_bbox_center.reshape(1, 3))
            @ self._ply_exp_bbox_rotation
        ).astype(np.float32, copy=False)
        abs_local = np.abs(local)
        outside_mask = np.any(abs_local > half, axis=1)
        if not np.any(outside_mask):
            raise ValueError("No source points were found outside the BBox.")
        excess = np.maximum(abs_local - half, 0.0)
        edge_dist = np.linalg.norm(excess, axis=1)
        outside_idx = np.flatnonzero(outside_mask)
        ordered_idx = outside_idx[np.argsort(edge_dist[outside_mask], kind="mergesort")]
        pool_size = min(ordered_idx.shape[0], max(palette_size * 64, 256))
        pool = ordered_idx[:pool_size]
        if pool.size == 0:
            raise ValueError("Failed to collect edge-near source colors.")
        rng = np.random.default_rng()
        take = min(palette_size, pool.size)
        chosen = rng.choice(pool, size=take, replace=False)
        palette = colors[chosen].astype(np.uint8, copy=False)
        if palette.shape[0] < palette_size:
            extra = colors[
                rng.choice(pool, size=palette_size - palette.shape[0], replace=True)
            ].astype(np.uint8, copy=False)
            palette = np.concatenate((palette, extra), axis=0).astype(
                np.uint8,
                copy=False,
            )
        return palette

    def _sample_ply_exp_main_palette(self, palette_size: int) -> np.ndarray:
        points, colors = self._get_ply_exp_source_points_and_colors()
        half = np.maximum(self._ply_exp_bbox_size * 0.5, 1e-6).reshape(1, 3)
        local = (
            (points - self._ply_exp_bbox_center.reshape(1, 3))
            @ self._ply_exp_bbox_rotation
        ).astype(np.float32, copy=False)
        inside_mask = np.all(np.abs(local) <= (half + 1e-6), axis=1)
        if not np.any(inside_mask):
            raise ValueError("No source points were found inside the BBox.")
        inside_colors = colors[inside_mask].astype(np.uint8, copy=False)
        if inside_colors.shape[0] == 0:
            raise ValueError("No source colors were found inside the BBox.")
        bins = np.clip(inside_colors.astype(np.int32) // 32, 0, 7)
        codes = bins[:, 0] + bins[:, 1] * 8 + bins[:, 2] * 64
        unique_codes, inverse_idx, counts = np.unique(
            codes,
            return_inverse=True,
            return_counts=True,
        )
        order = np.argsort(-counts, kind="mergesort")
        palette_list: List[np.ndarray] = []
        for code_idx in order[:palette_size]:
            mask = inverse_idx == code_idx
            if not np.any(mask):
                continue
            mean_color = np.mean(
                inside_colors[mask].astype(np.float32, copy=False),
                axis=0,
            )
            palette_list.append(
                np.clip(np.rint(mean_color), 0, 255).astype(np.uint8)
            )
        if not palette_list:
            raise ValueError("Failed to derive main colors from points inside the BBox.")
        palette = np.stack(palette_list, axis=0).astype(np.uint8, copy=False)
        if palette.shape[0] < palette_size:
            rng = np.random.default_rng()
            extra = inside_colors[
                rng.choice(
                    inside_colors.shape[0],
                    size=palette_size - palette.shape[0],
                    replace=True,
                )
            ].astype(np.uint8, copy=False)
            palette = np.concatenate((palette, extra), axis=0).astype(
                np.uint8,
                copy=False,
            )
        return palette

    def _generate_ply_exp_colors(self, count: int) -> np.ndarray:
        rng = np.random.default_rng()
        mode = (self._ply_exp_color_mode_var.get() or "").strip().lower()
        if mode.startswith("edge"):
            palette = self._sample_ply_exp_edge_palette(self._get_ply_exp_color_count())
            palette_idx = rng.integers(0, palette.shape[0], size=count)
            return palette[palette_idx].astype(np.uint8, copy=False)
        if mode.startswith("main"):
            palette = self._sample_ply_exp_main_palette(self._get_ply_exp_color_count())
            palette_idx = rng.integers(0, palette.shape[0], size=count)
            return palette[palette_idx].astype(np.uint8, copy=False)
        return rng.integers(0, 256, size=(count, 3), dtype=np.uint8)

    def _generate_ply_exp_points(self, count: int) -> np.ndarray:
        half = np.maximum(self._ply_exp_bbox_size * 0.5, 1e-6).astype(
            np.float32,
            copy=False,
        )
        mode = (self._ply_exp_mode_var.get() or "Inside").strip().lower()
        rng = np.random.default_rng()
        if mode.startswith("inside"):
            local = rng.uniform(-half, half, size=(count, 3)).astype(np.float32)
        else:
            outer_mult = float(self._ply_exp_outer_mult_var.get().strip() or "2.0")
            if outer_mult <= 1.0:
                raise ValueError(
                    "Outer distance multiplier must be greater than 1 for Outside mode."
                )
            outer_half = half * outer_mult
            batches: List[np.ndarray] = []
            remaining = count
            attempts = 0
            while remaining > 0 and attempts < 32:
                batch_size = max(remaining * 2, 2048)
                candidates = rng.uniform(
                    -outer_half,
                    outer_half,
                    size=(batch_size, 3),
                ).astype(np.float32)
                keep = np.any(np.abs(candidates) > half.reshape(1, 3), axis=1)
                accepted = candidates[keep]
                if accepted.size > 0:
                    batches.append(accepted[:remaining])
                    remaining -= int(min(remaining, accepted.shape[0]))
                attempts += 1
            if remaining > 0:
                raise ValueError(
                    "Failed to generate enough Outside points. Increase the outer distance multiplier."
                )
            local = np.concatenate(batches, axis=0).astype(np.float32, copy=False)
        return (
            local @ self._ply_exp_bbox_rotation.T + self._ply_exp_bbox_center
        ).astype(np.float32, copy=False)

    def _on_apply_ply_exp_bbox(self, _event=None) -> Optional[str]:
        if not self._apply_ply_exp_bbox_from_vars(show_error=True):
            return "break"
        self._sync_ply_exp_bbox_vars_from_state()
        self._redraw_ply_canvas()
        return "break"

    def _on_ply_exp_bbox_focus_out(self, _event=None) -> Optional[str]:
        if self._apply_ply_exp_bbox_from_vars(show_error=False):
            self._sync_ply_exp_bbox_vars_from_state()
            self._redraw_ply_canvas()
        return None

    def _on_add_ply_exp_points(self) -> None:
        if self._ply_view_points is None or self._ply_view_points.size == 0:
            messagebox.showerror(
                "BBox Scatter",
                "Load a point cloud before adding scatter points.",
            )
            return
        if not self._apply_ply_exp_bbox_from_vars(show_error=True):
            return
        try:
            count = int(float(self._ply_exp_point_count_var.get().strip() or "5000"))
        except Exception:
            messagebox.showerror("BBox Scatter", "Point count must be numeric.")
            return
        if count <= 0:
            messagebox.showerror("BBox Scatter", "Point count must be greater than zero.")
            return
        try:
            points = self._generate_ply_exp_points(count)
            colors = self._generate_ply_exp_colors(count)
        except Exception as exc:
            messagebox.showerror("BBox Scatter", str(exc))
            return
        self._clear_ply_remove_snapshot()
        if self._ply_exp_points is None or self._ply_exp_colors is None:
            self._ply_exp_points = points
            self._ply_exp_colors = colors
        else:
            self._ply_exp_points = np.concatenate(
                (self._ply_exp_points, points),
                axis=0,
            ).astype(np.float32, copy=False)
            self._ply_exp_colors = np.concatenate(
                (self._ply_exp_colors, colors),
                axis=0,
            ).astype(np.uint8, copy=False)
        self._update_ply_view_info_from_current()
        self._redraw_ply_canvas()

    def _on_reset_ply_exp_points(self) -> None:
        self._ply_exp_points = None
        self._ply_exp_colors = None
        self._update_ply_view_info_from_current()
        self._redraw_ply_canvas()

    def _on_reset_ply_exp_bbox(self) -> None:
        self._reset_ply_exp_bbox_defaults()
        self._set_ply_exp_edit_mode("Move")
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
            messagebox.showerror(
                "Sky PointCloud",
                "Load a point cloud before auto picking a color.",
            )
            return
        rgb = self._sample_auto_sky_color()
        if rgb is None:
            messagebox.showerror(
                "Sky PointCloud",
                "Failed to auto pick sky color from current points.",
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
            messagebox.showerror(
                "Point Cloud Viewer",
                "Load a point cloud before removing colors.",
            )
            return
        target_text = self._ply_remove_color_var.get().strip()
        target_rgb = self._parse_color_to_rgb(target_text)
        if target_rgb is None:
            messagebox.showerror(
                "Point Cloud Viewer",
                "Remove color must be a valid color (e.g., #87cefa).",
            )
            return
        tol = self._get_remove_color_tolerance(show_error=True)
        if tol is None:
            return
        self._update_remove_color_display(target_rgb, target_text)
        target = np.array(target_rgb, dtype=np.int32).reshape(1, 3)
        tol2 = float(tol) * float(tol)
        self._ply_pre_remove_points = self._ply_view_points.astype(
            np.float32,
            copy=True,
        )
        self._ply_pre_remove_colors = self._ply_view_colors.astype(
            np.uint8,
            copy=True,
        )
        self._ply_pre_remove_point_ids = (
            self._ply_view_point_ids.astype(np.int64, copy=True)
            if self._ply_view_point_ids is not None
            else None
        )
        self._ply_pre_remove_total_points = int(self._ply_view_total_points)
        self._ply_pre_remove_sample_step = int(self._ply_view_sample_step)
        self._ply_pre_remove_sky_points = (
            self._ply_sky_points.astype(np.float32, copy=True)
            if self._ply_sky_points is not None
            else None
        )
        self._ply_pre_remove_sky_colors = (
            self._ply_sky_colors.astype(np.uint8, copy=True)
            if self._ply_sky_colors is not None
            else None
        )
        base_colors = self._ply_view_colors.astype(np.int32, copy=False)
        diff = base_colors - target
        diff64 = diff.astype(np.int64, copy=False)
        dist2 = (diff64 * diff64).sum(axis=1)
        keep_mask = dist2 > tol2
        removed_base = int(np.count_nonzero(~keep_mask))
        if removed_base <= 0:
            self._clear_ply_remove_snapshot()
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
        if self._ply_view_point_ids is not None:
            self._ply_view_point_ids = self._ply_view_point_ids[
                keep_mask
            ].astype(np.int64, copy=False)
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
        remaining_base = int(self._ply_view_points.shape[0])
        remaining_sky = 0
        if self._ply_sky_points is not None:
            remaining_sky = int(self._ply_sky_points.shape[0])
        remaining_total = remaining_base + remaining_sky
        total_removed = removed_base + removed_sky
        self._ply_view_info_var.set(
            f"Removed {total_removed:,} pts by color "
            f"({target_text}, tol={tol:.1f}) -> {remaining_total:,} pts "
            f"(base {remaining_base:,}, sky {remaining_sky:,})"
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
        self._ply_exp_points = None
        self._ply_exp_colors = None
        self._ply_view_points = self._ply_loaded_points.astype(
            np.float32, copy=True
        )
        self._ply_view_colors = self._ply_loaded_colors.astype(
            np.uint8, copy=True
        )
        self._ply_view_point_ids = (
            self._ply_loaded_point_ids.astype(np.int64, copy=True)
            if self._ply_loaded_point_ids is not None
            else None
        )
        self._ply_view_points_centered = self._ply_view_points.astype(
            np.float32, copy=False
        )
        self._ply_view_total_points = int(
            max(self._ply_loaded_total_points, self._ply_view_points.shape[0])
        )
        self._ply_view_sample_step = max(1, int(self._ply_loaded_sample_step))
        self._ply_pre_append_points = None
        self._ply_pre_append_colors = None
        self._ply_pre_append_point_ids = None
        self._ply_pre_append_total_points = 0
        self._ply_pre_append_sample_step = 1
        self._clear_ply_remove_snapshot()
        self._refresh_ply_view_rgb_mean()
        self._reset_ply_exp_bbox_defaults()
        point_count = int(self._ply_view_points.shape[0])
        info = self._build_ply_info_text(
            point_count,
            self._ply_view_total_points,
            self._ply_view_sample_step,
        )
        self._ply_view_info_var.set(info)
        self._redraw_ply_canvas()

    def _update_sky_save_default(self, source_path: Path) -> None:
        if self._ply_source_kind == "colmap" or self._is_colmap_text_model_dir(
            source_path
        ):
            candidate = source_path.with_name(f"{source_path.name}_viewed")
        else:
            suffix = source_path.suffix or ".ply"
            candidate = source_path.with_name(
                f"{source_path.stem}_viewed{suffix}"
            )
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
        if self._ply_source_kind == "colmap":
            path = filedialog.askdirectory(
                title="Save Viewed COLMAP Folder",
                initialdir=str(initial_dir),
            )
        else:
            path = filedialog.asksaveasfilename(
                title="Save Viewed PLY",
                initialdir=str(initial_dir),
                defaultextension=".ply",
                filetypes=[("PLY files", "*.ply"), ("All files", "*.*")],
            )
        if path:
            self._ply_sky_save_path_var.set(path)

    def _on_save_sky_points(self) -> None:
        if self._ply_view_points is None or self._ply_view_colors is None:
            messagebox.showerror(
                "Point Cloud Viewer",
                "Load a point cloud before saving.",
            )
            return
        dest_text = self._ply_sky_save_path_var.get().strip()
        if not dest_text:
            messagebox.showerror(
                "Point Cloud Viewer",
                "Specify a save path first.",
            )
            return
        dest_path = Path(dest_text).expanduser()
        try:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            messagebox.showerror(
                "Point Cloud Viewer",
                f"Failed to prepare destination.\n{exc}",
            )
            return
        try:
            if self._ply_source_kind == "colmap":
                self._write_colmap_from_view(dest_path)
            else:
                self._write_binary_ply_from_view(dest_path)
        except Exception as exc:
            messagebox.showerror(
                "Point Cloud Viewer",
                f"Failed to save point cloud.\n{exc}",
            )
            return
        messagebox.showinfo(
            "Point Cloud Viewer",
            f"Saved viewed point cloud:\n{dest_path}",
        )

    def _write_binary_ply_from_view(self, path: Path) -> None:
        base_points = self._ply_view_points
        base_colors = self._ply_view_colors
        sky_points = self._ply_sky_points
        sky_colors = self._ply_sky_colors
        exp_points = self._ply_exp_points
        exp_colors = self._ply_exp_colors
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
        if exp_points is not None and exp_colors is not None:
            points_list.append(exp_points.astype(np.float32, copy=False))
            colors_list.append(exp_colors.astype(np.uint8, copy=False))
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

    def _write_colmap_from_view(self, path: Path) -> None:
        if self._ply_colmap_model is None:
            raise ValueError("COLMAP model metadata is not available.")
        base_points = self._ply_view_points
        base_colors = self._ply_view_colors
        if base_points is None or base_colors is None:
            raise ValueError("Missing loaded point cloud data.")
        point_ids = self._ply_view_point_ids
        if point_ids is None:
            point_ids = np.full(base_points.shape[0], -1, dtype=np.int64)
        sky_points = self._ply_sky_points
        sky_colors = self._ply_sky_colors
        exp_points = self._ply_exp_points
        exp_colors = self._ply_exp_colors
        points_list: List[np.ndarray] = [base_points.astype(np.float32, copy=False)]
        colors_list: List[np.ndarray] = [base_colors.astype(np.uint8, copy=False)]
        ids_list: List[np.ndarray] = [point_ids.astype(np.int64, copy=False)]
        if sky_points is not None and sky_colors is not None:
            center = self._ply_view_center
            if not isinstance(center, np.ndarray):
                center = np.zeros(3, dtype=np.float32)
            sky_world = (sky_points + center).astype(np.float32, copy=False)
            points_list.append(sky_world)
            colors_list.append(sky_colors.astype(np.uint8, copy=False))
            ids_list.append(np.full(sky_world.shape[0], -1, dtype=np.int64))
        if exp_points is not None and exp_colors is not None:
            points_list.append(exp_points.astype(np.float32, copy=False))
            colors_list.append(exp_colors.astype(np.uint8, copy=False))
            ids_list.append(np.full(exp_points.shape[0], -1, dtype=np.int64))
        xyz = np.concatenate(points_list, axis=0).astype(np.float32, copy=False)
        rgb = np.concatenate(colors_list, axis=0).astype(np.uint8, copy=False)
        merged_ids = np.concatenate(ids_list, axis=0).astype(
            np.int64,
            copy=False,
        )
        pointcloud_optimizer.save_colmap_text_model(
            path,
            self._ply_colmap_model,
            xyz,
            rgb,
            merged_ids,
        )


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
        colors = None
        color_candidates = (
            ("red", "green", "blue"),
            ("r", "g", "b"),
            ("diffuse_red", "diffuse_green", "diffuse_blue"),
        )
        for r_name, g_name, b_name in color_candidates:
            if all(
                channel in data.dtype.names
                for channel in (r_name, g_name, b_name)
            ):
                colors = np.stack(
                    (data[r_name], data[g_name], data[b_name]),
                    axis=1,
                )
                if np.issubdtype(colors.dtype, np.floating):
                    colors = self._float_colors_to_u8(colors)
                else:
                    colors = colors.astype(np.uint8, copy=False)
                break
        if colors is None and all(
            channel in data.dtype.names
            for channel in ("f_dc_0", "f_dc_1", "f_dc_2")
        ):
            colors = pointcloud_optimizer.convert_3dgs_dc_to_rgb8(
                np.stack(
                    (data["f_dc_0"], data["f_dc_1"], data["f_dc_2"]),
                    axis=1,
                )
            )
        if colors is None:
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

    @staticmethod
    def _compute_depth_norm(depth_values: np.ndarray) -> np.ndarray:
        depth32 = depth_values.astype(np.float32, copy=False)
        finite = depth32[np.isfinite(depth32)]
        if finite.size == 0:
            return np.zeros(depth32.shape, dtype=np.float32)
        depth_min = float(finite.min())
        depth_max = float(finite.max())
        if depth_max <= depth_min + 1e-6:
            return np.zeros(depth32.shape, dtype=np.float32)
        return np.clip(
            (depth32 - depth_min) / (depth_max - depth_min),
            0.0,
            1.0,
        ).astype(np.float32, copy=False)

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
        if self._ply_exp_points is not None and self._ply_exp_colors is not None:
            points = np.concatenate((points, self._ply_exp_points), axis=0)
            colors = np.concatenate((colors, self._ply_exp_colors), axis=0)
        depth_view = bool(self._ply_monochrome_var.get())
        front_occlusion = bool(self._ply_front_occlusion_var.get()) or depth_view
        show_world_axes = bool(self._ply_show_world_axes_var.get()) and (
            not bool(self._ply_exp_bbox_active_var.get())
        )
        display_matrix = self._get_ply_display_up_axis_matrix()
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
        self._ply_render_sample_step = max(1, sample_step)
        self._ply_rendered_point_count = 0
        if sample_step > 1:
            points = points[::sample_step]
            colors = colors[::sample_step]
        depth_offset = max(self._ply_view_depth_offset, 1e-6)
        rotation = self._quat_to_matrix(self._ply_view_quat)
        display_points = points @ display_matrix.T
        rotated = display_points @ rotation.T
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
        bg_rgb = (0x11, 0x11, 0x11)
        buf = np.full((height, width, 3), bg_rgb[0], dtype=np.uint8)
        if bool(self._ply_show_grid_var.get()):
            image_bg = Image.fromarray(buf, mode="RGB")
            draw_bg = ImageDraw.Draw(image_bg)
            self._draw_ply_ground_grid(
                draw_bg,
                width,
                height,
                rotation,
                proj_scale,
                ortho_scale,
                is_orthographic,
                pan_x,
                pan_y,
                display_matrix,
            )
            buf = np.asarray(image_bg, dtype=np.uint8).copy()
        else:
            self._ply_rendered_grid_span = 0.0
        depth_buf = np.full((height, width), np.inf, dtype=np.float32)
        ix = np.rint(sx).astype(np.int32)
        iy = np.rint(sy).astype(np.int32)
        valid &= (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
        if bool(self._ply_draw_points_var.get()) and np.any(valid):
            ix_valid = ix[valid]
            iy_valid = iy[valid]
            depth_valid = depth[valid]
            colors_valid = colors[valid]
            point_size = self._get_ply_point_size()
            self._draw_projected_point_layer(
                buf,
                depth_buf,
                width,
                height,
                ix_valid,
                iy_valid,
                depth_valid,
                colors_valid,
                monochrome=depth_view,
                point_size=point_size,
                front_occlusion=front_occlusion,
            )
            self._ply_rendered_point_count = int(np.count_nonzero(valid))
        image = Image.fromarray(buf, mode="RGB")
        draw = ImageDraw.Draw(image)
        if show_world_axes:
            self._draw_ply_world_axes(
                draw,
                width,
                height,
                rotation,
                proj_scale,
                ortho_scale,
                is_orthographic,
                pan_x,
                pan_y,
                display_matrix,
            )
        self._draw_ply_exp_bbox(
            draw,
            width,
            height,
            rotation,
            proj_scale,
            ortho_scale,
            is_orthographic,
            pan_x,
            pan_y,
            display_matrix,
        )
        self._draw_ply_info_overlay(draw, image)
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
        self._ply_rendered_grid_span = 0.0
        self._ply_render_sample_step = 1
        self._ply_rendered_point_count = 0
        draw = ImageDraw.Draw(image)
        info = self._ply_view_info_var.get().strip() or "Point cloud viewer is idle"
        bbox = draw.textbbox((0, 0), info)
        text_w = max(0, int(bbox[2] - bbox[0]))
        text_h = max(0, int(bbox[3] - bbox[1]))
        x = max(12, (width - text_w) // 2)
        y = max(12, (height - text_h) // 2)
        draw.text((x, y), info, fill=(220, 220, 220))
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

    @staticmethod
    def _normalize_display_up_axis(mode: str) -> str:
        if (mode or "").strip().lower().startswith("z"):
            return "Z-up"
        return "Y-down"

    @staticmethod
    def _get_default_display_view_angles(mode: str) -> Tuple[float, float]:
        _ = PreviewApp._normalize_display_up_axis(mode)
        return math.radians(145.0), math.radians(-25.0)

    @staticmethod
    def _get_display_up_axis_matrix(mode: str) -> np.ndarray:
        mode_name = PreviewApp._normalize_display_up_axis(mode)
        if mode_name == "Z-up":
            return np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=np.float32,
            )
        return np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    def _get_ply_display_up_axis_matrix(self) -> np.ndarray:
        return self._get_display_up_axis_matrix(self._ply_display_up_axis_var.get())

    def _get_camera_scene_display_up_axis_matrix(self) -> np.ndarray:
        return self._get_display_up_axis_matrix(
            self._camera_scene_display_up_axis_var.get()
        )

    def _draw_ply_ground_grid(
        self,
        draw: ImageDraw.ImageDraw,
        width: int,
        height: int,
        rotation: np.ndarray,
        proj_scale: float,
        ortho_scale: float,
        is_orthographic: bool,
        pan_x: float,
        pan_y: float,
        display_matrix: np.ndarray,
    ) -> None:
        step = self._get_ply_grid_step()
        span = self._get_ply_grid_span()
        if span is None:
            half_span = max(self._ply_view_max_extent * 0.75, step * 2.0, 1.0)
        else:
            half_span = max(float(span) * 0.5, step * 2.0)
        line_count = max(2, int(math.ceil(half_span / step)))
        line_count = min(line_count, CAMERA_SCENE_GRID_MAX_HALF_LINES)
        grid_limit = line_count * step
        self._ply_rendered_grid_span = float(grid_limit * 2.0)
        minor_color = (48, 56, 68)
        major_color = (74, 86, 102)
        axis_color = (105, 123, 146)
        for index in range(-line_count, line_count + 1):
            coord = index * step
            line_color = major_color if index % 5 == 0 else minor_color
            if index == 0:
                line_color = axis_color
            if (self._ply_display_up_axis_var.get() or "").strip().lower().startswith("z"):
                grid_lines = (
                    ((-grid_limit, coord, 0.0), (grid_limit, coord, 0.0)),
                    ((coord, -grid_limit, 0.0), (coord, grid_limit, 0.0)),
                )
            else:
                grid_lines = (
                    ((-grid_limit, 0.0, coord), (grid_limit, 0.0, coord)),
                    ((coord, 0.0, -grid_limit), (coord, 0.0, grid_limit)),
                )
            for start, end in grid_lines:
                start_pt = self._project_point_to_screen(
                    start,
                    width,
                    height,
                    rotation,
                    self._ply_view_depth_offset,
                    proj_scale,
                    ortho_scale,
                    is_orthographic,
                    pan_x,
                    pan_y,
                    display_matrix=display_matrix,
                )
                end_pt = self._project_point_to_screen(
                    end,
                    width,
                    height,
                    rotation,
                    self._ply_view_depth_offset,
                    proj_scale,
                    ortho_scale,
                    is_orthographic,
                    pan_x,
                    pan_y,
                    display_matrix=display_matrix,
                )
                if start_pt is None or end_pt is None:
                    continue
                draw.line([start_pt, end_pt], fill=line_color, width=1)

    def _draw_ply_world_axes(
        self,
        draw: ImageDraw.ImageDraw,
        width: int,
        height: int,
        rotation: np.ndarray,
        proj_scale: float,
        ortho_scale: float,
        is_orthographic: bool,
        pan_x: float,
        pan_y: float,
        display_matrix: np.ndarray,
    ) -> None:
        axis_len = self._get_ply_axis_length()
        origin = (0.0, 0.0, 0.0)
        display_mode = self._normalize_display_up_axis(
            self._ply_display_up_axis_var.get()
        )
        origin_pt = self._project_point_to_screen(
            origin,
            width,
            height,
            rotation,
            self._ply_view_depth_offset,
            proj_scale,
            ortho_scale,
            is_orthographic,
            pan_x,
            pan_y,
            display_matrix=display_matrix,
        )
        if origin_pt is None:
            return
        y_axis_value = -axis_len if display_mode == "Y-down" else axis_len
        for label, endpoint, color in (
            ("X", (axis_len, 0.0, 0.0), (255, 80, 80)),
            ("Y", (0.0, y_axis_value, 0.0), (80, 255, 120)),
            ("Z", (0.0, 0.0, axis_len), (80, 160, 255)),
        ):
            screen_pt = self._project_point_to_screen(
                endpoint,
                width,
                height,
                rotation,
                self._ply_view_depth_offset,
                proj_scale,
                ortho_scale,
                is_orthographic,
                pan_x,
                pan_y,
                display_matrix=display_matrix,
            )
            if screen_pt is None:
                continue
            draw.line([origin_pt, screen_pt], fill=color, width=2)
            self._draw_axis_arrowhead(draw, origin_pt, screen_pt, color)
            draw.text((screen_pt[0] + 4, screen_pt[1] - 12), label, fill=color)
        radius = 4
        draw.ellipse(
            [
                (origin_pt[0] - radius, origin_pt[1] - radius),
                (origin_pt[0] + radius, origin_pt[1] + radius),
            ],
            fill=(255, 255, 255),
        )

    @staticmethod
    def _draw_axis_arrowhead(
        draw: ImageDraw.ImageDraw,
        start_pt: Tuple[float, float],
        end_pt: Tuple[float, float],
        color: Tuple[int, int, int],
    ) -> None:
        vec = np.array(
            [float(end_pt[0] - start_pt[0]), float(end_pt[1] - start_pt[1])],
            dtype=np.float32,
        )
        length = float(np.linalg.norm(vec))
        if length <= 1e-6:
            return
        unit = vec / length
        perp = np.array([-unit[1], unit[0]], dtype=np.float32)
        arrow_len = 10.0
        arrow_wing = 4.0
        tip = np.array([float(end_pt[0]), float(end_pt[1])], dtype=np.float32)
        base = tip - unit * arrow_len
        wing_a = base + perp * arrow_wing
        wing_b = base - perp * arrow_wing
        draw.polygon(
            [
                (float(tip[0]), float(tip[1])),
                (float(wing_a[0]), float(wing_a[1])),
                (float(wing_b[0]), float(wing_b[1])),
            ],
            fill=color,
            outline=(255, 255, 255),
        )

    def _draw_ply_exp_bbox(
        self,
        draw: ImageDraw.ImageDraw,
        width: int,
        height: int,
        rotation: np.ndarray,
        proj_scale: float,
        ortho_scale: float,
        is_orthographic: bool,
        pan_x: float,
        pan_y: float,
        display_matrix: np.ndarray,
    ) -> None:
        if not bool(self._ply_exp_bbox_active_var.get()):
            return
        corners = self._get_ply_exp_bbox_corners()
        if corners is None or corners.shape[0] != 8:
            return
        edge_pairs = (
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        )
        projected: List[Optional[Tuple[float, float]]] = []
        for corner in corners:
            projected.append(
                self._project_point_to_screen(
                    tuple(float(v) for v in corner),
                    width,
                    height,
                    rotation,
                    self._ply_view_depth_offset,
                    proj_scale,
                    ortho_scale,
                    is_orthographic,
                    pan_x,
                    pan_y,
                    display_matrix=display_matrix,
                )
            )
        edge_color = (255, 196, 64)
        for a_idx, b_idx in edge_pairs:
            a_pt = projected[a_idx]
            b_pt = projected[b_idx]
            if a_pt is None or b_pt is None:
                continue
            draw.line([a_pt, b_pt], fill=edge_color, width=2)
        handles = self._get_ply_exp_handle_positions()
        center_pt = None if handles is None else handles.get("center")
        move_mode = (self._ply_exp_edit_mode_var.get() or "").strip().lower().startswith(
            "m"
        )
        scale_mode = (self._ply_exp_edit_mode_var.get() or "").strip().lower().startswith(
            "s"
        )
        if center_pt is not None:
            handle_radius = 6
            if move_mode:
                draw.ellipse(
                    [
                        (center_pt[0] - handle_radius, center_pt[1] - handle_radius),
                        (center_pt[0] + handle_radius, center_pt[1] + handle_radius),
                    ],
                    fill=(255, 255, 255),
                    outline=edge_color,
                )
            else:
                draw.rectangle(
                    [
                        (center_pt[0] - handle_radius, center_pt[1] - handle_radius),
                        (center_pt[0] + handle_radius, center_pt[1] + handle_radius),
                    ],
                    fill=(255, 255, 255),
                    outline=edge_color,
                )
        if handles is None or center_pt is None:
            return
        axis_colors = (
            (255, 96, 96),
            (96, 224, 96),
            (96, 160, 255),
        )
        for axis_idx, axis_pt in enumerate(handles["axes"]):
            if axis_pt is None:
                continue
            color = axis_colors[axis_idx]
            draw.line([center_pt, axis_pt], fill=color, width=2)
            if move_mode:
                vec = np.array(
                    [axis_pt[0] - center_pt[0], axis_pt[1] - center_pt[1]],
                    dtype=np.float32,
                )
                length = float(np.linalg.norm(vec))
                if length > 1e-6:
                    unit = vec / length
                    perp = np.array([-unit[1], unit[0]], dtype=np.float32)
                    arrow_len = 12.0
                    arrow_wing = 4.5
                    base = np.array(axis_pt, dtype=np.float32) - unit * arrow_len
                    wing_a = base + perp * arrow_wing
                    wing_b = base - perp * arrow_wing
                    draw.polygon(
                        [
                            tuple(axis_pt),
                            (float(wing_a[0]), float(wing_a[1])),
                            (float(wing_b[0]), float(wing_b[1])),
                        ],
                        fill=color,
                        outline=(255, 255, 255),
                    )
            elif scale_mode:
                radius = 6
                draw.rectangle(
                    [
                        (axis_pt[0] - radius, axis_pt[1] - radius),
                        (axis_pt[0] + radius, axis_pt[1] + radius),
                    ],
                    fill=color,
                    outline=(255, 255, 255),
                )

    def _draw_ply_info_overlay(
        self,
        draw: ImageDraw.ImageDraw,
        image: Image.Image,
    ) -> None:
        info_text = self._ply_view_info_var.get().strip() or "Point cloud viewer"
        render_count = (
            self._ply_rendered_point_count
            if bool(self._ply_draw_points_var.get())
            else 0
        )
        lines = [
            info_text,
            "render: {} pts (step {})".format(
                render_count,
                self._ply_render_sample_step,
            ),
        ]
        self._draw_overlay_lines(draw, image, lines)

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
        display_matrix: Optional[np.ndarray] = None,
    ) -> Optional[Tuple[float, float]]:
        px, py, pz = point
        p = np.array([px, py, pz], dtype=np.float32)
        if display_matrix is not None:
            p = display_matrix @ p
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

    def _update_camera_scene_input_state(self) -> None:
        source_var = self.camera_scene_vars.get("source_type")
        mode = source_var.get().strip() if source_var is not None else ""
        visible_group = {
            CAMERA_SCENE_SOURCE_CHOICES[0]: "colmap",
            CAMERA_SCENE_SOURCE_CHOICES[1]: "transforms",
            CAMERA_SCENE_SOURCE_CHOICES[2]: "realityscan",
            CAMERA_SCENE_SOURCE_CHOICES[3]: "xmp",
            CAMERA_SCENE_SOURCE_CHOICES[4]: "metashape",
        }.get(mode, "colmap")
        for group_name, frame in self._camera_scene_input_groups.items():
            if frame is None:
                continue
            if group_name == visible_group:
                frame.pack(fill="x", pady=0)
            else:
                frame.pack_forget()
        title_map = {
            "colmap": "Load COLMAP View",
            "transforms": "Load transforms View",
            "realityscan": "Load RealityScan View",
            "xmp": "Load RealityScan XMP View",
            "metashape": "Load Metashape XML View",
        }
        if self.camera_scene_load_button is not None:
            self.camera_scene_load_button.configure(
                text=title_map.get(visible_group, "Load Camera View")
            )

    def _get_camera_scene_optional_width_height(
        self,
    ) -> Tuple[Optional[int], Optional[int]]:
        width_text = self.camera_converter_vars.get("width", tk.StringVar()).get().strip()
        height_text = self.camera_converter_vars.get("height", tk.StringVar()).get().strip()
        width_value: Optional[int]
        height_value: Optional[int]
        if width_text:
            width_value = int(width_text)
            if width_value <= 0:
                raise ValueError("Width must be a positive integer.")
        else:
            width_value = None
        if height_text:
            height_value = int(height_text)
            if height_value <= 0:
                raise ValueError("Height must be a positive integer.")
        else:
            height_value = None
        return width_value, height_value

    def _load_camera_scene(self) -> None:
        if not self.camera_scene_vars:
            return
        source_type = self.camera_scene_vars["source_type"].get().strip()
        try:
            width_value, height_value = self._get_camera_scene_optional_width_height()
            if source_type == CAMERA_SCENE_SOURCE_CHOICES[0]:
                source_dir = self.camera_scene_vars["colmap_dir"].get().strip()
                if not source_dir:
                    raise ValueError("COLMAP folder is required.")
                scene = camera_pose_scene.load_scene_from_colmap_dir(Path(source_dir))
            elif source_type == CAMERA_SCENE_SOURCE_CHOICES[1]:
                transforms_json = self.camera_scene_vars["transforms_json"].get().strip()
                transforms_ply = self.camera_scene_vars["transforms_ply"].get().strip()
                if not transforms_json or not transforms_ply:
                    raise ValueError("transforms.json and transforms PLY are required.")
                scene = camera_pose_scene.load_scene_from_transforms_set(
                    Path(transforms_json),
                    Path(transforms_ply),
                )
            elif source_type == CAMERA_SCENE_SOURCE_CHOICES[2]:
                csv_path = self.camera_scene_vars["csv_path"].get().strip()
                csv_ply = self.camera_scene_vars["csv_ply"].get().strip()
                if not csv_path or not csv_ply:
                    raise ValueError("RealityScan CSV and RealityScan PLY are required.")
                scene = camera_pose_scene.load_scene_from_realityscan_csv_set(
                    Path(csv_path),
                    Path(csv_ply),
                )
            elif source_type == CAMERA_SCENE_SOURCE_CHOICES[3]:
                xmp_dir = self.camera_scene_vars["xmp_dir"].get().strip()
                xmp_ply = self.camera_scene_vars["xmp_ply"].get().strip()
                if not xmp_dir:
                    raise ValueError("RealityScan XMP directory is required.")
                if width_value is None or height_value is None:
                    raise ValueError(
                        "Width and Height are required for RealityScan XMP preview."
                    )
                scene = camera_pose_scene.load_scene_from_realityscan_xmp_set(
                    Path(xmp_dir),
                    Path(xmp_ply) if xmp_ply else None,
                    width=width_value,
                    height=height_value,
                )
            else:
                metashape_xml = self.camera_scene_vars["metashape_xml"].get().strip()
                metashape_ply = self.camera_scene_vars["metashape_ply"].get().strip()
                if not metashape_xml:
                    raise ValueError("Metashape XML is required.")
                scene = camera_pose_scene.load_scene_from_metashape_xml_set(
                    Path(metashape_xml),
                    Path(metashape_ply) if metashape_ply else None,
                    width=width_value,
                    height=height_value,
                )
        except Exception as exc:
            messagebox.showerror(
                "CameraFormatConverter",
                f"Failed to load camera scene.\n{exc}",
            )
            return
        self._set_camera_scene(scene)

    def _clear_camera_scene(self) -> None:
        self._camera_scene_initial_view_state = None
        self._camera_scene_base_points = np.zeros((0, 3), dtype=np.float32)
        self._camera_scene_base_colors = np.zeros((0, 3), dtype=np.uint8)
        self._camera_scene_base_camera_items = []
        self._camera_scene_base_info_text = "Camera format viewer is idle"
        self._camera_scene_base_source_label = "CameraPoseScene"
        self._camera_scene_points = None
        self._camera_scene_points_centered = None
        self._camera_scene_colors = None
        self._camera_scene_camera_items = []
        self._camera_scene_center = np.zeros(3, dtype=np.float32)
        self._camera_scene_view_center = np.zeros(3, dtype=np.float32)
        self._camera_scene_origin_centered = np.zeros(3, dtype=np.float32)
        self._camera_scene_ground_y = 0.0
        self._camera_scene_total_points = 0
        self._camera_scene_total_cameras = 0
        self._camera_scene_sample_step = 1
        self._camera_scene_camera_sample_step = 1
        self._camera_scene_rendered_camera_count = 0
        self._camera_scene_rendered_grid_span = 0.0
        self._camera_scene_source_label = "CameraPoseScene"
        self._camera_scene_info_var.set("Camera format viewer is idle")
        self._reset_camera_scene_transform()
        self._redraw_camera_scene_canvas()

    def _log_camera_scene_normalization(self, scene: Any) -> None:
        lines = list(getattr(scene, "normalization_log", []) or [])
        if not lines:
            return
        self._append_text_widget(self.camera_converter_log, "[preview] --------")
        for line in lines:
            self._append_text_widget(self.camera_converter_log, str(line))
        self._append_text_widget(
            self.camera_converter_log,
            "[preview] Additional preview transform defaults: "
            "camera rot=(0,0,0), pointcloud rot=(0,0,0), "
            "camera scale=1.0, pointcloud scale=1.0, link=ON",
        )

    def _clone_camera_scene_camera_items(
        self,
        camera_items: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        cloned: List[Dict[str, Any]] = []
        for item in camera_items:
            cloned.append(
                {
                    "name": str(item["name"]),
                    "center": np.asarray(item["center"], dtype=np.float32).copy(),
                    "rotation_cw": np.asarray(
                        item["rotation_cw"],
                        dtype=np.float32,
                    ).copy(),
                    "frustum_half_w": float(item["frustum_half_w"]),
                    "frustum_half_h": float(item["frustum_half_h"]),
                }
            )
        return cloned

    def _set_camera_scene(self, scene: Any) -> None:
        self._camera_scene_base_points = np.asarray(
            scene.points_xyz,
            dtype=np.float32,
        ).copy()
        self._camera_scene_base_colors = np.asarray(
            scene.points_rgb,
            dtype=np.uint8,
        ).copy()
        self._camera_scene_base_camera_items = [
            {
                "name": str(cam.name),
                "center": np.asarray(cam.center, dtype=np.float32).copy(),
                "rotation_cw": np.asarray(
                    cam.rotation_cw,
                    dtype=np.float32,
                ).copy(),
                "frustum_half_w": float(cam.frustum_half_w),
                "frustum_half_h": float(cam.frustum_half_h),
            }
            for cam in scene.cameras
        ]
        self._camera_scene_base_source_label = str(scene.source_kind)
        self._camera_scene_base_info_text = str(scene.info_text)
        self._log_camera_scene_normalization(scene)
        self._reset_camera_scene_load_defaults()
        self._apply_camera_scene_preview_transform()

    def _set_camera_scene_view_data(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        camera_items: Sequence[Dict[str, Any]],
        source_label: str,
        info_text: str,
    ) -> None:
        points = np.asarray(points, dtype=np.float32)
        colors = np.asarray(colors, dtype=np.uint8)
        camera_items_local = self._clone_camera_scene_camera_items(camera_items)
        camera_centers = [
            np.asarray(item["center"], dtype=np.float32)
            for item in camera_items_local
        ]
        bounds_parts: List[np.ndarray] = []
        if points.size:
            bounds_parts.append(points)
        if camera_centers:
            bounds_parts.append(np.vstack(camera_centers).astype(np.float32))
        if bounds_parts:
            all_xyz = np.vstack(bounds_parts)
            xyz_min = all_xyz.min(axis=0)
            xyz_max = all_xyz.max(axis=0)
            extent = np.maximum(xyz_max - xyz_min, 1e-6)
            max_extent = float(np.max(extent))
        else:
            max_extent = 1.0
        self._camera_scene_points = points
        self._camera_scene_points_centered = (
            points.astype(np.float32, copy=False)
            if points.size
            else np.zeros((0, 3), dtype=np.float32)
        )
        self._camera_scene_colors = colors
        self._camera_scene_camera_items = []
        for item in camera_items_local:
            center_abs = np.asarray(item["center"], dtype=np.float32)
            self._camera_scene_camera_items.append(
                {
                    "name": str(item["name"]),
                    "center": center_abs,
                    "center_centered": center_abs.astype(np.float32, copy=False),
                    "rotation_cw": np.asarray(
                        item["rotation_cw"],
                        dtype=np.float32,
                    ),
                    "frustum_half_w": float(item["frustum_half_w"]),
                    "frustum_half_h": float(item["frustum_half_h"]),
                }
            )
        self._camera_scene_center = np.zeros(3, dtype=np.float32)
        self._camera_scene_view_center = np.zeros(3, dtype=np.float32)
        self._camera_scene_origin_centered = np.zeros(3, dtype=np.float32)
        self._camera_scene_ground_y = 0.0
        self._camera_scene_total_points = int(points.shape[0]) if points.ndim == 2 else 0
        self._camera_scene_total_cameras = len(camera_items_local)
        self._camera_scene_camera_sample_step = 1
        self._camera_scene_rendered_camera_count = self._camera_scene_total_cameras
        self._camera_scene_source_label = str(source_label)
        self._camera_scene_max_extent = max(max_extent, 1e-3)
        self._camera_scene_depth_offset = self._camera_scene_max_extent * 2.5
        self._camera_scene_info_var.set(str(info_text))
        self._set_camera_scene_view_center(
            np.zeros(3, dtype=np.float32),
            reset_view_transform=True,
            capture_initial=True,
            redraw=True,
        )

    def _sync_camera_scene_linked_transform_vars(self, *_args) -> None:
        if not self.camera_converter_vars:
            return
        link_var = self.camera_converter_vars.get("link_transform")
        if link_var is None or not bool(link_var.get()):
            return
        for src_key, dst_key in (
            ("camera_rot_x_deg", "pointcloud_rot_x_deg"),
            ("camera_rot_y_deg", "pointcloud_rot_y_deg"),
            ("camera_rot_z_deg", "pointcloud_rot_z_deg"),
            ("camera_scale", "pointcloud_scale"),
        ):
            src_var = self.camera_converter_vars.get(src_key)
            dst_var = self.camera_converter_vars.get(dst_key)
            if src_var is None or dst_var is None:
                continue
            src_value = src_var.get()
            if dst_var.get() != src_value:
                dst_var.set(src_value)

    def _sync_camera_scene_transform_link_state(self) -> None:
        link_var = self.camera_converter_vars.get("link_transform")
        linked = bool(link_var.get()) if link_var is not None else True
        if linked:
            self._sync_camera_scene_linked_transform_vars()
        state = "disabled" if linked else "normal"
        for widget in self._camera_scene_point_transform_widgets:
            try:
                widget.configure(state=state)
            except Exception:
                pass

    def _on_camera_scene_link_transform_changed(self, *_args) -> None:
        self._sync_camera_scene_transform_link_state()

    def _collect_camera_scene_transform_values(
        self,
        show_error: bool = True,
    ) -> Optional[Dict[str, float]]:
        if not self.camera_converter_vars:
            return None
        self._sync_camera_scene_linked_transform_vars()
        value_map: Dict[str, float] = {}
        defaults = {
            "camera_rot_x_deg": 0.0,
            "camera_rot_y_deg": 0.0,
            "camera_rot_z_deg": 0.0,
            "pointcloud_rot_x_deg": 0.0,
            "pointcloud_rot_y_deg": 0.0,
            "pointcloud_rot_z_deg": 0.0,
            "camera_scale": 1.0,
            "pointcloud_scale": 1.0,
        }
        for key, default_value in defaults.items():
            var = self.camera_converter_vars.get(key)
            text = var.get().strip() if var is not None else ""
            if not text:
                value = default_value
            else:
                try:
                    value = float(text)
                except Exception:
                    if show_error:
                        messagebox.showerror(
                            "CameraFormatConverter",
                            f"{key} must be numeric.",
                        )
                    return None
            if key.endswith("_scale") and value <= 0.0:
                if show_error:
                    messagebox.showerror(
                        "CameraFormatConverter",
                        f"{key} must be greater than 0.",
                    )
                return None
            value_map[key] = value
        return value_map

    def _apply_camera_scene_preview_transform(self) -> None:
        transform_values = self._collect_camera_scene_transform_values(
            show_error=True,
        )
        if transform_values is None:
            return
        if (
            self._camera_scene_base_points.size == 0
            and not self._camera_scene_base_camera_items
        ):
            self._redraw_camera_scene_canvas()
            return
        camera_rot = np.asarray(
            camera_pose_scene.camera_converter.build_world_rotation_xyz_deg(
                transform_values["camera_rot_x_deg"],
                transform_values["camera_rot_y_deg"],
                transform_values["camera_rot_z_deg"],
            ),
            dtype=np.float32,
        )
        point_rot = np.asarray(
            camera_pose_scene.camera_converter.build_world_rotation_xyz_deg(
                transform_values["pointcloud_rot_x_deg"],
                transform_values["pointcloud_rot_y_deg"],
                transform_values["pointcloud_rot_z_deg"],
            ),
            dtype=np.float32,
        )
        points = np.asarray(
            self._camera_scene_base_points,
            dtype=np.float32,
        ).copy()
        if points.size:
            points = (points @ point_rot.T).astype(np.float32, copy=False)
            points *= float(transform_values["pointcloud_scale"])
        colors = np.asarray(
            self._camera_scene_base_colors,
            dtype=np.uint8,
        ).copy()
        camera_items = self._clone_camera_scene_camera_items(
            self._camera_scene_base_camera_items
        )
        for item in camera_items:
            center = np.asarray(item["center"], dtype=np.float32)
            center = (center @ camera_rot.T).astype(np.float32, copy=False)
            center *= float(transform_values["camera_scale"])
            item["center"] = center
            item["rotation_cw"] = (
                camera_rot @ np.asarray(item["rotation_cw"], dtype=np.float32)
            ).astype(np.float32, copy=False)
        info_text = str(self._camera_scene_base_info_text)
        if any(
            abs(transform_values[key] - default) > 1e-9
            for key, default in (
                ("camera_rot_x_deg", 0.0),
                ("camera_rot_y_deg", 0.0),
                ("camera_rot_z_deg", 0.0),
                ("pointcloud_rot_x_deg", 0.0),
                ("pointcloud_rot_y_deg", 0.0),
                ("pointcloud_rot_z_deg", 0.0),
                ("camera_scale", 1.0),
                ("pointcloud_scale", 1.0),
            )
        ):
            info_text = (
                f"{info_text}  |  preview transform applied"
            )
        self._set_camera_scene_view_data(
            points,
            colors,
            camera_items,
            self._camera_scene_base_source_label,
            info_text,
        )

    def _reset_camera_scene_transform(self) -> None:
        yaw, pitch = self._get_default_display_view_angles(
            self._camera_scene_display_up_axis_var.get()
        )
        q_yaw = self._quat_from_axis_angle((0.0, 1.0, 0.0), yaw)
        q_pitch = self._quat_from_axis_angle((1.0, 0.0, 0.0), pitch)
        self._camera_scene_quat = self._quat_normalize(
            self._quat_multiply(q_pitch, q_yaw)
        )
        self._camera_scene_zoom, self._camera_scene_pan = self._compute_camera_scene_initial_view()
        self._camera_scene_drag_last = None
        self._camera_scene_pan_last = None

    def _get_camera_scene_viewport_size(self) -> Tuple[int, int]:
        canvas = self._camera_scene_canvas
        width = 0
        height = 0
        if canvas is not None and canvas.winfo_exists():
            try:
                canvas.update_idletasks()
            except Exception:
                pass
            width = int(canvas.winfo_width())
            height = int(canvas.winfo_height())
        if width < 10 or height < 10:
            width = PLY_VIEW_CANVAS_WIDTH
            height = 520
        return width, height

    def _compute_camera_scene_initial_view(self) -> Tuple[float, Tuple[float, float]]:
        points_parts: List[np.ndarray] = []
        if (
            self._camera_scene_points_centered is not None
            and self._camera_scene_points_centered.size > 0
        ):
            points_parts.append(self._camera_scene_points_centered)
        if self._camera_scene_camera_items:
            camera_points = np.vstack(
                [
                    np.asarray(item["center_centered"], dtype=np.float32)
                    for item in self._camera_scene_camera_items
                ]
            ).astype(np.float32, copy=False)
            if camera_points.size > 0:
                points_parts.append(camera_points)
        if not points_parts:
            return 2.0, (0.0, 0.0)
        fit_points = np.vstack(points_parts).astype(np.float32, copy=False)
        width, height = self._get_camera_scene_viewport_size()
        return self._compute_fit_zoom_pan(
            fit_points,
            self._camera_scene_quat,
            self._camera_scene_max_extent,
            width,
            height,
            fill_ratio=0.82,
        )

    def _capture_camera_scene_initial_view_state(self) -> None:
        self._camera_scene_initial_view_state = {
            "quat": np.asarray(self._camera_scene_quat, dtype=np.float32).copy(),
            "zoom": float(self._camera_scene_zoom),
            "pan": (
                float(self._camera_scene_pan[0]),
                float(self._camera_scene_pan[1]),
            ),
            "view_center": np.asarray(
                self._camera_scene_view_center,
                dtype=np.float32,
            ).copy(),
            "projection_mode": self._camera_scene_projection_mode.get(),
            "display_up_axis": self._camera_scene_display_up_axis_var.get(),
            "point_cap": self._camera_scene_point_cap_var.get(),
            "interactive_point_cap": self._camera_scene_interactive_point_cap_var.get(),
            "point_size": self._camera_scene_point_size_var.get(),
            "camera_scale": self._camera_scene_camera_scale_var.get(),
            "camera_stride": self._camera_scene_camera_stride_var.get(),
            "grid_step": self._camera_scene_grid_step_var.get(),
            "grid_span": self._camera_scene_grid_span_var.get(),
            "draw_points": bool(self._camera_scene_draw_points_var.get()),
            "draw_cameras": bool(self._camera_scene_draw_cameras_var.get()),
            "monochrome_points": bool(
                self._camera_scene_monochrome_points_var.get()
            ),
            "front_occlusion": bool(
                self._camera_scene_front_occlusion_var.get()
            ),
            "show_world_axes": bool(self._camera_scene_show_world_axes_var.get()),
            "show_camera_axes": bool(self._camera_scene_show_camera_axes_var.get()),
            "show_labels": bool(self._camera_scene_show_labels_var.get()),
            "show_grid": bool(self._camera_scene_show_grid_var.get()),
        }

    def _reset_camera_scene_view(self) -> None:
        if self._camera_scene_initial_view_state is None:
            self._reset_camera_scene_transform()
            self._redraw_camera_scene_canvas()
            return
        state = self._camera_scene_initial_view_state
        self._camera_scene_quat = np.asarray(
            state["quat"],
            dtype=np.float32,
        ).copy()
        self._camera_scene_zoom = float(state["zoom"])
        self._camera_scene_pan = (
            float(state["pan"][0]),
            float(state["pan"][1]),
        )
        self._set_camera_scene_view_center(
            np.asarray(state["view_center"], dtype=np.float32),
            reset_view_transform=False,
            capture_initial=False,
            redraw=False,
        )
        self._camera_scene_drag_last = None
        self._camera_scene_pan_last = None
        self._camera_scene_projection_mode.set(str(state["projection_mode"]))
        self._camera_scene_display_up_axis_var.set(
            self._normalize_display_up_axis(
                str(state.get("display_up_axis", "Z-up"))
            )
        )
        self._camera_scene_point_cap_var.set(str(state["point_cap"]))
        self._camera_scene_interactive_point_cap_var.set(
            str(state["interactive_point_cap"])
        )
        self._camera_scene_point_size_var.set(str(state["point_size"]))
        self._camera_scene_camera_scale_var.set(str(state["camera_scale"]))
        self._camera_scene_camera_stride_var.set(str(state["camera_stride"]))
        self._camera_scene_grid_step_var.set(str(state["grid_step"]))
        self._camera_scene_grid_span_var.set(str(state["grid_span"]))
        self._camera_scene_draw_points_var.set(bool(state["draw_points"]))
        self._camera_scene_draw_cameras_var.set(bool(state["draw_cameras"]))
        self._camera_scene_monochrome_points_var.set(
            bool(state["monochrome_points"])
        )
        self._camera_scene_front_occlusion_var.set(
            bool(state.get("front_occlusion", True))
        )
        self._camera_scene_show_world_axes_var.set(bool(state["show_world_axes"]))
        self._camera_scene_show_camera_axes_var.set(bool(state["show_camera_axes"]))
        self._camera_scene_show_labels_var.set(bool(state["show_labels"]))
        self._camera_scene_show_grid_var.set(bool(state["show_grid"]))
        self._enforce_camera_scene_depth_view_constraints()
        self._redraw_camera_scene_canvas(force=True)

    def _on_reset_camera_scene_camera_view(self) -> None:
        self._reset_camera_scene_transform()
        self._redraw_camera_scene_canvas(force=True)

    def _reset_camera_scene_load_defaults(self) -> None:
        if self.camera_converter_vars:
            self.camera_converter_vars["link_transform"].set(True)
            self.camera_converter_vars["camera_rot_x_deg"].set("0")
            self.camera_converter_vars["camera_rot_y_deg"].set("0")
            self.camera_converter_vars["camera_rot_z_deg"].set("0")
            self.camera_converter_vars["camera_scale"].set("1.0")
            self.camera_converter_vars["pointcloud_rot_x_deg"].set("0")
            self.camera_converter_vars["pointcloud_rot_y_deg"].set("0")
            self.camera_converter_vars["pointcloud_rot_z_deg"].set("0")
            self.camera_converter_vars["pointcloud_scale"].set("1.0")
            self._sync_camera_scene_transform_link_state()
        self._camera_scene_projection_mode.set("Orthographic")
        self._camera_scene_display_up_axis_var.set("Z-up")
        self._camera_scene_point_cap_var.set(str(PLY_VIEW_MAX_POINTS))
        self._camera_scene_interactive_point_cap_var.set(
            str(PLY_VIEW_INTERACTIVE_MAX_POINTS)
        )
        self._camera_scene_point_size_var.set("2")
        self._camera_scene_camera_scale_var.set("1")
        self._camera_scene_camera_stride_var.set("1")
        self._camera_scene_grid_step_var.set("1.0")
        self._camera_scene_grid_span_var.set("auto")
        self._camera_scene_draw_points_var.set(True)
        self._camera_scene_draw_cameras_var.set(True)
        self._camera_scene_monochrome_points_var.set(False)
        self._camera_scene_front_occlusion_var.set(True)
        self._camera_scene_show_world_axes_var.set(True)
        self._camera_scene_show_camera_axes_var.set(False)
        self._camera_scene_show_labels_var.set(False)
        self._camera_scene_show_grid_var.set(True)
        self._enforce_camera_scene_depth_view_constraints()

    def _enforce_camera_scene_depth_view_constraints(self) -> None:
        if bool(self._camera_scene_monochrome_points_var.get()):
            self._camera_scene_front_occlusion_var.set(True)
        state = (
            "disabled"
            if bool(self._camera_scene_monochrome_points_var.get())
            else "normal"
        )
        if self._camera_scene_front_occlusion_checkbutton is not None:
            try:
                self._camera_scene_front_occlusion_checkbutton.configure(state=state)
            except Exception:
                pass

    def _on_camera_scene_depth_view_toggle(self) -> None:
        self._enforce_camera_scene_depth_view_constraints()
        self._redraw_camera_scene_canvas()

    def _on_camera_scene_depth_occlusion_toggle(self) -> None:
        self._enforce_camera_scene_depth_view_constraints()
        self._redraw_camera_scene_canvas()

    def _get_camera_scene_axis_length(self) -> float:
        return max(self._camera_scene_max_extent * 0.2, 1e-3)

    def _set_camera_scene_view_center(
        self,
        center_xyz: np.ndarray,
        reset_view_transform: bool = False,
        capture_initial: bool = False,
        redraw: bool = True,
    ) -> None:
        center = np.asarray(center_xyz, dtype=np.float32).reshape(3)
        self._camera_scene_view_center = center.copy()
        self._camera_scene_center = center.copy()
        if self._camera_scene_points is not None and self._camera_scene_points.size:
            self._camera_scene_points_centered = (
                self._camera_scene_points - center
            ).astype(np.float32, copy=False)
        else:
            self._camera_scene_points_centered = np.zeros((0, 3), dtype=np.float32)
        for item in self._camera_scene_camera_items:
            center_abs = np.asarray(item["center"], dtype=np.float32)
            item["center_centered"] = (
                center_abs - center
            ).astype(np.float32, copy=False)
        self._camera_scene_origin_centered = (-center).astype(
            np.float32,
            copy=False,
        )
        self._camera_scene_ground_y = float(self._camera_scene_origin_centered[1])
        if reset_view_transform:
            self._reset_camera_scene_transform()
        if capture_initial:
            self._capture_camera_scene_initial_view_state()
        if redraw:
            self._redraw_camera_scene_canvas(force=True)

    def _get_camera_scene_grid_step(self) -> float:
        text = self._camera_scene_grid_step_var.get().strip()
        if not text:
            return 1.0
        try:
            value = float(text)
        except Exception:
            return 1.0
        if value <= 0.0:
            return 1.0
        return value

    def _get_camera_scene_point_size(self) -> int:
        text = self._camera_scene_point_size_var.get().strip()
        if not text:
            return 1
        try:
            value = int(round(float(text)))
        except Exception:
            return 1
        if value < 1:
            return 1
        return min(value, 9)

    def _get_camera_scene_grid_span(self) -> Optional[float]:
        text = self._camera_scene_grid_span_var.get().strip()
        if not text or text.lower() == "auto":
            return None
        try:
            value = float(text)
        except Exception:
            return None
        if value <= 0.0:
            return None
        return value

    def _redraw_camera_scene_canvas(self, force: bool = False) -> None:
        canvas = self._camera_scene_canvas
        if canvas is None or not canvas.winfo_exists():
            return
        if not force and not self._is_camera_scene_tab_active():
            self._camera_scene_redraw_pending = True
            return
        self._camera_scene_redraw_pending = False
        self._render_camera_scene(canvas)

    def _on_camera_scene_canvas_configure(self, _event=None) -> None:
        self._redraw_camera_scene_canvas()

    def _begin_camera_scene_interaction(self) -> None:
        self._camera_scene_is_interacting = True
        if self._camera_scene_interaction_after_id is not None:
            try:
                self.root.after_cancel(self._camera_scene_interaction_after_id)
            except Exception:
                pass
            self._camera_scene_interaction_after_id = None

    def _schedule_end_camera_scene_interaction(
        self, delay_ms: int = PLY_INTERACTION_SETTLE_DELAY_MS
    ) -> None:
        if self._camera_scene_interaction_after_id is not None:
            try:
                self.root.after_cancel(self._camera_scene_interaction_after_id)
            except Exception:
                pass
            self._camera_scene_interaction_after_id = None
        self._camera_scene_interaction_after_id = self.root.after(
            delay_ms,
            self._finish_camera_scene_interaction,
        )

    def _finish_camera_scene_interaction(self) -> None:
        if self._camera_scene_interaction_after_id is not None:
            try:
                self.root.after_cancel(self._camera_scene_interaction_after_id)
            except Exception:
                pass
            self._camera_scene_interaction_after_id = None
        was_interacting = self._camera_scene_is_interacting
        self._camera_scene_is_interacting = False
        if was_interacting:
            self._redraw_camera_scene_canvas()

    def _on_camera_scene_drag_start(self, event: tk.Event) -> None:
        if not self._camera_scene_camera_items and self._camera_scene_points_centered is None:
            return
        self._begin_camera_scene_interaction()
        self._camera_scene_drag_last = (event.x, event.y)

    def _on_camera_scene_drag_move(self, event: tk.Event) -> None:
        if self._camera_scene_drag_last is None:
            return
        last_x, last_y = self._camera_scene_drag_last
        dx = event.x - last_x
        dy = event.y - last_y
        self._camera_scene_drag_last = (event.x, event.y)
        angle_y = math.radians(-dx * 0.35)
        angle_x = math.radians(-dy * 0.35)
        q_y = self._quat_from_axis_angle((0.0, 1.0, 0.0), angle_y)
        q_x = self._quat_from_axis_angle((1.0, 0.0, 0.0), angle_x)
        q_inc = self._quat_multiply(q_x, q_y)
        self._camera_scene_quat = self._quat_normalize(
            self._quat_multiply(q_inc, self._camera_scene_quat)
        )
        self._redraw_camera_scene_canvas()

    def _on_camera_scene_drag_end(self, _event=None) -> None:
        self._camera_scene_drag_last = None
        self._camera_scene_pan_last = None
        self._schedule_end_camera_scene_interaction()

    def _on_camera_scene_double_click(self, event: tk.Event) -> Optional[str]:
        if not self._camera_scene_camera_items or self._camera_scene_canvas is None:
            return None
        canvas = self._camera_scene_canvas
        width = max(1, canvas.winfo_width())
        height = max(1, canvas.winfo_height())
        rotation = self._quat_to_matrix(self._camera_scene_quat)
        proj_scale = min(width, height) * 0.9 * self._camera_scene_zoom
        ortho_scale = self._compute_camera_scene_ortho_scale(width, height)
        projection_mode = (self._camera_scene_projection_mode.get() or "").lower()
        is_orthographic = projection_mode.startswith("ortho")
        pan_x, pan_y = self._camera_scene_pan
        best_item: Optional[Dict[str, Any]] = None
        best_distance_sq: Optional[float] = None
        for item in self._camera_scene_camera_items:
            center_pt = self._project_point_to_screen(
                tuple(float(v) for v in item["center_centered"]),
                width,
                height,
                rotation,
                self._camera_scene_depth_offset,
                proj_scale,
                ortho_scale,
                is_orthographic,
                pan_x,
                pan_y,
            )
            if center_pt is None:
                continue
            dx = float(center_pt[0] - event.x)
            dy = float(center_pt[1] - event.y)
            distance_sq = dx * dx + dy * dy
            if best_distance_sq is None or distance_sq < best_distance_sq:
                best_distance_sq = distance_sq
                best_item = item
        if best_item is None or best_distance_sq is None:
            return "break"
        if best_distance_sq > (36.0 * 36.0):
            return "break"
        self._camera_scene_pan = (0.0, 0.0)
        self._camera_scene_drag_last = None
        self._camera_scene_pan_last = None
        self._set_camera_scene_view_center(
            np.asarray(best_item["center"], dtype=np.float32),
            reset_view_transform=False,
            capture_initial=False,
            redraw=True,
        )
        return "break"

    def _on_camera_scene_pan_start(self, event: tk.Event) -> None:
        if not self._camera_scene_camera_items and self._camera_scene_points_centered is None:
            return
        self._begin_camera_scene_interaction()
        self._camera_scene_pan_last = (event.x, event.y)

    def _on_camera_scene_pan_move(self, event: tk.Event) -> None:
        if self._camera_scene_pan_last is None:
            return
        last_x, last_y = self._camera_scene_pan_last
        dx = event.x - last_x
        dy = event.y - last_y
        pan_x, pan_y = self._camera_scene_pan
        self._camera_scene_pan = (pan_x + dx, pan_y + dy)
        self._camera_scene_pan_last = (event.x, event.y)
        self._redraw_camera_scene_canvas()

    def _on_camera_scene_pan_end(self, _event=None) -> None:
        self._camera_scene_pan_last = None
        self._schedule_end_camera_scene_interaction()

    def _on_camera_scene_zoom(self, event: tk.Event) -> Optional[str]:
        if not self._camera_scene_camera_items and self._camera_scene_points_centered is None:
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
        new_zoom = self._camera_scene_zoom * factor
        self._camera_scene_zoom = max(0.1, min(PLY_VIEW_MAX_ZOOM, new_zoom))
        self._begin_camera_scene_interaction()
        self._redraw_camera_scene_canvas()
        self._schedule_end_camera_scene_interaction()
        return "break"

    def _get_camera_scene_point_cap(
        self,
        interactive: bool,
        show_error: bool = False,
    ) -> int:
        var = (
            self._camera_scene_interactive_point_cap_var
            if interactive
            else self._camera_scene_point_cap_var
        )
        default = (
            PLY_VIEW_INTERACTIVE_MAX_POINTS
            if interactive
            else PLY_VIEW_MAX_POINTS
        )
        text = var.get().strip()
        try:
            value = int(text)
        except Exception:
            value = default
        if value <= 0:
            if show_error:
                messagebox.showerror(
                    "CameraFormatConverter",
                    "Point cap must be a positive integer.",
                )
            return default
        return value

    def _get_camera_scene_camera_stride(self) -> int:
        text = self._camera_scene_camera_stride_var.get().strip()
        try:
            value = int(text)
        except Exception:
            value = 1
        return max(1, value)

    def _get_camera_scene_effective_camera_stride(self) -> int:
        stride = self._get_camera_scene_camera_stride()
        if not self._camera_scene_is_interacting:
            return stride
        interactive_stride = self._compute_sample_step(
            self._camera_scene_total_cameras,
            CAMERA_SCENE_INTERACTIVE_MAX_CAMERAS,
        )
        return max(stride, interactive_stride)

    def _get_camera_scene_camera_scale(self) -> float:
        text = self._camera_scene_camera_scale_var.get().strip()
        if not text:
            return 1.0
        try:
            value = float(text)
        except Exception:
            return 1.0
        if value <= 0:
            return 1.0
        return value

    def _compute_camera_scene_ortho_scale(self, width: int, height: int) -> float:
        max_extent = max(self._camera_scene_max_extent, 1e-6)
        return max(
            1e-6,
            self._camera_scene_zoom * (min(width, height) * 0.45 / max_extent),
        )

    def _project_camera_scene_xyz(
        self,
        xyz: np.ndarray,
        width: int,
        height: int,
        rotation: np.ndarray,
        proj_scale: float,
        ortho_scale: float,
        is_orthographic: bool,
        pan_x: float,
        pan_y: float,
        display_matrix: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        arr = np.asarray(xyz, dtype=np.float32)
        if arr.size == 0:
            empty = np.zeros((0,), dtype=np.float32)
            return empty, empty, empty, np.zeros((0,), dtype=bool)
        arr = arr @ display_matrix.T
        rotated = arr @ rotation.T
        x1 = rotated[:, 0]
        y1 = rotated[:, 1]
        z2 = rotated[:, 2]
        depth = z2 + self._camera_scene_depth_offset
        if is_orthographic:
            sx = width / 2.0 + x1 * ortho_scale + pan_x
            sy = height / 2.0 - y1 * ortho_scale + pan_y
            valid = np.ones_like(depth, dtype=bool)
        else:
            valid = depth > 1e-4
            scale = np.zeros_like(depth)
            scale[valid] = proj_scale / depth[valid]
            sx = width / 2.0 + x1 * scale + pan_x
            sy = height / 2.0 - y1 * scale + pan_y
        return sx, sy, depth, valid

    def _draw_camera_scene_point_layer(
        self,
        buf: np.ndarray,
        depth_buf: np.ndarray,
        width: int,
        height: int,
        rotation: np.ndarray,
        proj_scale: float,
        ortho_scale: float,
        is_orthographic: bool,
        pan_x: float,
        pan_y: float,
        display_matrix: np.ndarray,
    ) -> None:
        if (
            self._camera_scene_points_centered is None
            or self._camera_scene_points_centered.size == 0
            or self._camera_scene_colors is None
            or not bool(self._camera_scene_draw_points_var.get())
        ):
            self._camera_scene_sample_step = 1
            return
        points = self._camera_scene_points_centered
        colors = self._camera_scene_colors
        render_cap = self._get_camera_scene_point_cap(
            interactive=self._camera_scene_is_interacting,
            show_error=False,
        )
        sample_step = self._compute_sample_step(points.shape[0], render_cap)
        self._camera_scene_sample_step = max(1, sample_step)
        if sample_step > 1:
            points = points[::sample_step]
            colors = colors[::sample_step]
        sx, sy, depth, valid = self._project_camera_scene_xyz(
            points,
            width,
            height,
            rotation,
            proj_scale,
            ortho_scale,
            is_orthographic,
            pan_x,
            pan_y,
            display_matrix,
        )
        ix = np.rint(sx).astype(np.int32)
        iy = np.rint(sy).astype(np.int32)
        valid &= (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
        if not np.any(valid):
            return
        self._draw_projected_point_layer(
            buf,
            depth_buf,
            width,
            height,
            ix[valid],
            iy[valid],
            depth[valid],
            colors[valid],
            monochrome=bool(self._camera_scene_monochrome_points_var.get()),
            point_size=self._get_camera_scene_point_size(),
            front_occlusion=bool(self._camera_scene_front_occlusion_var.get())
            or bool(self._camera_scene_monochrome_points_var.get()),
        )

    def _render_camera_scene(self, canvas: tk.Canvas) -> None:
        has_points = (
            self._camera_scene_points_centered is not None
            and self._camera_scene_points_centered.size > 0
        )
        has_cameras = bool(self._camera_scene_camera_items)
        if not has_points and not has_cameras:
            self._camera_scene_rendered_grid_span = 0.0
            self._render_camera_scene_idle_canvas(canvas)
            return
        canvas.update_idletasks()
        width = max(1, canvas.winfo_width())
        height = max(1, canvas.winfo_height())
        if width < 10 or height < 10:
            width = PLY_VIEW_CANVAS_WIDTH
            height = PLY_VIEW_CANVAS_HEIGHT
        rotation = self._quat_to_matrix(self._camera_scene_quat)
        display_matrix = self._get_camera_scene_display_up_axis_matrix()
        proj_scale = min(width, height) * 0.9 * self._camera_scene_zoom
        ortho_scale = self._compute_camera_scene_ortho_scale(width, height)
        projection_mode = (self._camera_scene_projection_mode.get() or "").lower()
        is_orthographic = projection_mode.startswith("ortho")
        pan_x, pan_y = self._camera_scene_pan

        buf = np.full((height, width, 3), 0x11, dtype=np.uint8)
        if bool(self._camera_scene_show_grid_var.get()):
            image_bg = Image.fromarray(buf, mode="RGB")
            draw_bg = ImageDraw.Draw(image_bg)
            self._draw_camera_scene_ground_grid(
                draw_bg,
                width,
                height,
                rotation,
                proj_scale,
                ortho_scale,
                is_orthographic,
                pan_x,
                pan_y,
                display_matrix,
            )
            buf = np.asarray(image_bg, dtype=np.uint8).copy()
        depth_buf = np.full((height, width), np.inf, dtype=np.float32)
        if has_points:
            self._draw_camera_scene_point_layer(
                buf,
                depth_buf,
                width,
                height,
                rotation,
                proj_scale,
                ortho_scale,
                is_orthographic,
                pan_x,
                pan_y,
                display_matrix,
            )
        image = Image.fromarray(buf, mode="RGB")
        draw = ImageDraw.Draw(image)
        if bool(self._camera_scene_show_world_axes_var.get()):
            self._draw_camera_scene_world_axes(
                draw,
                width,
                height,
                rotation,
                proj_scale,
                ortho_scale,
                is_orthographic,
                pan_x,
                pan_y,
                display_matrix,
            )
        self._camera_scene_camera_sample_step = self._get_camera_scene_effective_camera_stride()
        self._camera_scene_rendered_camera_count = 0
        if bool(self._camera_scene_draw_cameras_var.get()) and has_cameras:
            self._draw_camera_scene_camera_overlay(
                draw,
                width,
                height,
                rotation,
                proj_scale,
                ortho_scale,
                is_orthographic,
                pan_x,
                pan_y,
                display_matrix,
            )
        self._draw_camera_scene_info_overlay(draw, image)
        photo = ImageTk.PhotoImage(image=image)
        self._camera_scene_canvas_photo = photo
        if self._camera_scene_canvas_image_id is None:
            self._camera_scene_canvas_image_id = canvas.create_image(
                0, 0, anchor="nw", image=photo
            )
        else:
            canvas.itemconfigure(self._camera_scene_canvas_image_id, image=photo)
        canvas.configure(scrollregion=(0, 0, width, height))

    def _render_camera_scene_idle_canvas(self, canvas: tk.Canvas) -> None:
        canvas.update_idletasks()
        width = max(1, canvas.winfo_width())
        height = max(1, canvas.winfo_height())
        if width < 10 or height < 10:
            width = PLY_VIEW_CANVAS_WIDTH
            height = PLY_VIEW_CANVAS_HEIGHT
        image = Image.new("RGB", (width, height), (0x11, 0x11, 0x11))
        draw = ImageDraw.Draw(image)
        info = self._camera_scene_info_var.get().strip() or "Camera format viewer is idle"
        bbox = draw.textbbox((0, 0), info)
        text_w = max(0, int(bbox[2] - bbox[0]))
        text_h = max(0, int(bbox[3] - bbox[1]))
        x = max(12, (width - text_w) // 2)
        y = max(12, (height - text_h) // 2)
        draw.text((x, y), info, fill=(220, 220, 220))
        photo = ImageTk.PhotoImage(image=image)
        self._camera_scene_canvas_photo = photo
        if self._camera_scene_canvas_image_id is None:
            self._camera_scene_canvas_image_id = canvas.create_image(
                0, 0, anchor="nw", image=photo
            )
        else:
            canvas.itemconfigure(self._camera_scene_canvas_image_id, image=photo)
        canvas.configure(scrollregion=(0, 0, width, height))

    def _draw_camera_scene_ground_grid(
        self,
        draw: ImageDraw.ImageDraw,
        width: int,
        height: int,
        rotation: np.ndarray,
        proj_scale: float,
        ortho_scale: float,
        is_orthographic: bool,
        pan_x: float,
        pan_y: float,
        display_matrix: np.ndarray,
    ) -> None:
        step = self._get_camera_scene_grid_step()
        span = self._get_camera_scene_grid_span()
        if span is None:
            half_span = max(self._camera_scene_max_extent * 0.75, step * 2.0, 1.0)
        else:
            half_span = max(float(span) * 0.5, step * 2.0)
        line_count = max(2, int(math.ceil(half_span / step)))
        line_count = min(line_count, CAMERA_SCENE_GRID_MAX_HALF_LINES)
        grid_limit = line_count * step
        self._camera_scene_rendered_grid_span = float(grid_limit * 2.0)
        y_value = float(self._camera_scene_origin_centered[1])
        origin_x = float(self._camera_scene_origin_centered[0])
        origin_z = float(self._camera_scene_origin_centered[2])
        z_value = float(self._camera_scene_origin_centered[2])
        minor_color = (48, 56, 68)
        major_color = (74, 86, 102)
        axis_color = (105, 123, 146)
        for index in range(-line_count, line_count + 1):
            coord = index * step
            line_color = major_color if index % 5 == 0 else minor_color
            if index == 0:
                line_color = axis_color
            if (self._camera_scene_display_up_axis_var.get() or "").strip().lower().startswith(
                "z"
            ):
                axis_a_line = (
                    (origin_x - grid_limit, y_value + coord, z_value),
                    (origin_x + grid_limit, y_value + coord, z_value),
                )
                axis_b_line = (
                    (origin_x + coord, y_value - grid_limit, z_value),
                    (origin_x + coord, y_value + grid_limit, z_value),
                )
            else:
                axis_a_line = (
                    (origin_x - grid_limit, y_value, origin_z + coord),
                    (origin_x + grid_limit, y_value, origin_z + coord),
                )
                axis_b_line = (
                    (origin_x + coord, y_value, origin_z - grid_limit),
                    (origin_x + coord, y_value, origin_z + grid_limit),
                )
            for start, end in (axis_a_line, axis_b_line):
                start_pt = self._project_point_to_screen(
                    start,
                    width,
                    height,
                    rotation,
                    self._camera_scene_depth_offset,
                    proj_scale,
                    ortho_scale,
                    is_orthographic,
                    pan_x,
                    pan_y,
                    display_matrix=display_matrix,
                )
                end_pt = self._project_point_to_screen(
                    end,
                    width,
                    height,
                    rotation,
                    self._camera_scene_depth_offset,
                    proj_scale,
                    ortho_scale,
                    is_orthographic,
                    pan_x,
                    pan_y,
                    display_matrix=display_matrix,
                )
                if start_pt is None or end_pt is None:
                    continue
                draw.line([start_pt, end_pt], fill=line_color, width=1)

    def _draw_camera_scene_world_axes(
        self,
        draw: ImageDraw.ImageDraw,
        width: int,
        height: int,
        rotation: np.ndarray,
        proj_scale: float,
        ortho_scale: float,
        is_orthographic: bool,
        pan_x: float,
        pan_y: float,
        display_matrix: np.ndarray,
    ) -> None:
        axis_len = self._get_camera_scene_axis_length()
        origin = tuple(float(v) for v in self._camera_scene_origin_centered)
        display_mode = self._normalize_display_up_axis(
            self._camera_scene_display_up_axis_var.get()
        )
        colors = {
            "X": (255, 80, 80),
            "Y": (80, 255, 120),
            "Z": (80, 160, 255),
        }
        origin_pt = self._project_point_to_screen(
            origin,
            width,
            height,
            rotation,
            self._camera_scene_depth_offset,
            proj_scale,
            ortho_scale,
            is_orthographic,
            pan_x,
            pan_y,
            display_matrix=display_matrix,
        )
        if origin_pt is None:
            return
        y_axis_value = -axis_len if display_mode == "Y-down" else axis_len
        for label, endpoint in (
            ("X", (origin[0] + axis_len, origin[1], origin[2])),
            ("Y", (origin[0], origin[1] + y_axis_value, origin[2])),
            ("Z", (origin[0], origin[1], origin[2] + axis_len)),
        ):
            screen_pt = self._project_point_to_screen(
                endpoint,
                width,
                height,
                rotation,
                self._camera_scene_depth_offset,
                proj_scale,
                ortho_scale,
                is_orthographic,
                pan_x,
                pan_y,
                display_matrix=display_matrix,
            )
            if screen_pt is None:
                continue
            color = colors[label]
            draw.line([origin_pt, screen_pt], fill=color, width=2)
            self._draw_axis_arrowhead(draw, origin_pt, screen_pt, color)
            draw.text((screen_pt[0] + 4, screen_pt[1] - 12), label, fill=color)
        radius = 4
        draw.ellipse(
            [
                (origin_pt[0] - radius, origin_pt[1] - radius),
                (origin_pt[0] + radius, origin_pt[1] + radius),
            ],
            fill=(255, 255, 255),
        )
    def _draw_camera_scene_camera_overlay(
        self,
        draw: ImageDraw.ImageDraw,
        width: int,
        height: int,
        rotation: np.ndarray,
        proj_scale: float,
        ortho_scale: float,
        is_orthographic: bool,
        pan_x: float,
        pan_y: float,
        display_matrix: np.ndarray,
    ) -> None:
        camera_scale = self._get_camera_scene_camera_scale()
        camera_stride = self._get_camera_scene_effective_camera_stride()
        self._camera_scene_camera_sample_step = camera_stride
        rendered_camera_count = 0
        show_labels = False
        show_axes = False
        frustum_color = (255, 196, 92)
        for index, item in enumerate(self._camera_scene_camera_items):
            if (index % camera_stride) != 0:
                continue
            rendered_camera_count += 1
            center = item["center_centered"]
            center_tuple = tuple(float(v) for v in center)
            center_pt = self._project_point_to_screen(
                center_tuple,
                width,
                height,
                rotation,
                self._camera_scene_depth_offset,
                proj_scale,
                ortho_scale,
                is_orthographic,
                pan_x,
                pan_y,
                display_matrix=display_matrix,
            )
            if center_pt is None:
                continue
            rot_cw = np.asarray(item["rotation_cw"], dtype=np.float32)
            half_w = float(item["frustum_half_w"]) * camera_scale
            half_h = float(item["frustum_half_h"]) * camera_scale
            depth = camera_scale
            corners_local = (
                (-half_w, -half_h, depth),
                (half_w, -half_h, depth),
                (half_w, half_h, depth),
                (-half_w, half_h, depth),
            )
            corners_world: List[Tuple[float, float, float]] = []
            corner_points: List[Tuple[float, float]] = []
            for local in corners_local:
                world = center + (rot_cw @ np.asarray(local, dtype=np.float32))
                world_tuple = tuple(float(v) for v in world)
                corners_world.append(world_tuple)
                screen_pt = self._project_point_to_screen(
                    world_tuple,
                    width,
                    height,
                    rotation,
                    self._camera_scene_depth_offset,
                    proj_scale,
                    ortho_scale,
                    is_orthographic,
                    pan_x,
                    pan_y,
                    display_matrix=display_matrix,
                )
                if screen_pt is not None:
                    corner_points.append(screen_pt)
            if len(corner_points) == 4:
                for corner_pt in corner_points:
                    draw.line([center_pt, corner_pt], fill=frustum_color, width=1)
                for idx_line in range(4):
                    draw.line(
                        [
                            corner_points[idx_line],
                            corner_points[(idx_line + 1) % 4],
                        ],
                        fill=frustum_color,
                        width=1,
                    )
            draw.ellipse(
                [
                    (center_pt[0] - 2, center_pt[1] - 2),
                    (center_pt[0] + 2, center_pt[1] + 2),
                ],
                fill=frustum_color,
            )
            if show_axes:
                axis_defs = (
                    ("X", (camera_scale * 0.45, 0.0, 0.0), (255, 80, 80)),
                    ("Y", (0.0, camera_scale * 0.45, 0.0), (80, 255, 120)),
                    ("Z", (0.0, 0.0, camera_scale * 0.65), (80, 160, 255)),
                )
                for _label, local_axis, color in axis_defs:
                    axis_world = center + (
                        rot_cw @ np.asarray(local_axis, dtype=np.float32)
                    )
                    axis_pt = self._project_point_to_screen(
                        tuple(float(v) for v in axis_world),
                        width,
                        height,
                        rotation,
                        self._camera_scene_depth_offset,
                        proj_scale,
                        ortho_scale,
                        is_orthographic,
                        pan_x,
                        pan_y,
                        display_matrix=display_matrix,
                    )
                    if axis_pt is not None:
                        draw.line([center_pt, axis_pt], fill=color, width=2)
            if show_labels:
                draw.text(
                    (center_pt[0] + 4, center_pt[1] - 10),
                    item["name"],
                    fill=(255, 255, 255),
                )
        self._camera_scene_rendered_camera_count = rendered_camera_count

    def _draw_camera_scene_info_overlay(
        self,
        draw: ImageDraw.ImageDraw,
        image: Image.Image,
    ) -> None:
        info_text = self._camera_scene_info_var.get().strip()
        stride = self._camera_scene_camera_sample_step
        cap = self._get_camera_scene_point_cap(
            interactive=self._camera_scene_is_interacting,
            show_error=False,
        )
        point_render_count = (
            min(self._camera_scene_total_points, cap)
            if bool(self._camera_scene_draw_points_var.get())
            else 0
        )
        camera_render_count = (
            self._camera_scene_rendered_camera_count
            if bool(self._camera_scene_draw_cameras_var.get())
            else 0
        )
        axis_len = self._get_camera_scene_axis_length()
        grid_step = self._get_camera_scene_grid_step()
        grid_span = self._get_camera_scene_grid_span()
        if grid_span is None:
            grid_span_text = "{:.3f} units (auto)".format(
                float(self._camera_scene_rendered_grid_span)
            )
        else:
            grid_span_text = "{:.3f} units".format(grid_span)
        lines = [
            info_text or "Camera format viewer",
            "render: {} pts (step {}) / {} cams (stride {})".format(
                point_render_count,
                self._camera_scene_sample_step,
                camera_render_count,
                stride,
            ),
            "XYZ axis length: {:.3f}".format(axis_len),
            "Grid step: {:.3f} units".format(grid_step),
            "Grid span: {}".format(grid_span_text),
        ]
        self._draw_overlay_lines(draw, image, lines)

    def _draw_overlay_lines(
        self,
        draw: ImageDraw.ImageDraw,
        image: Image.Image,
        lines: Sequence[str],
        x0: int = 8,
        y0: int = 8,
    ) -> None:
        y = y0
        for line in lines:
            if not line:
                continue
            bbox = draw.textbbox((0, 0), line)
            text_w = max(0, int(bbox[2] - bbox[0]))
            text_h = max(0, int(bbox[3] - bbox[1]))
            x1 = min(image.width - 8, x0 + text_w + 8)
            y1 = y + text_h + 8
            draw.rectangle([(x0, y), (x1, y1)], fill=(0, 0, 0))
            draw.text((x0 + 4, y + 4), line, fill=(255, 255, 255))
            y = y1 + 4

    def _draw_projected_point_layer(
        self,
        buf: np.ndarray,
        depth_buf: np.ndarray,
        width: int,
        height: int,
        ix: np.ndarray,
        iy: np.ndarray,
        depth: np.ndarray,
        colors: np.ndarray,
        *,
        monochrome: bool,
        point_size: int,
        front_occlusion: bool,
    ) -> None:
        if ix.size == 0 or iy.size == 0 or depth.size == 0 or colors.size == 0:
            return
        pixel_index = (iy * width + ix).astype(np.int64, copy=False)
        depth_valid = depth.astype(np.float32, copy=False)
        colors_valid = colors.astype(np.uint8, copy=False)
        if monochrome:
            depth_norm = self._compute_depth_norm(depth_valid)
            gray = np.clip(
                np.rint((1.0 - depth_norm) * 255.0),
                0,
                255,
            ).astype(np.uint8)
            colors_valid = np.stack((gray, gray, gray), axis=1)
        if not front_occlusion:
            flat_buf = buf.reshape(-1, 3)
            if point_size <= 1:
                flat_buf[pixel_index, :] = colors_valid
                return
            radius = max(0, point_size // 2)
            for dy in range(-radius, radius + 1):
                y_pix = iy + dy
                y_ok = (y_pix >= 0) & (y_pix < height)
                if not np.any(y_ok):
                    continue
                for dx in range(-radius, radius + 1):
                    x_pix = ix + dx
                    valid_pix = y_ok & (x_pix >= 0) & (x_pix < width)
                    if not np.any(valid_pix):
                        continue
                    pixel_block = (
                        y_pix[valid_pix] * width + x_pix[valid_pix]
                    ).astype(np.int64, copy=False)
                    flat_buf[pixel_block, :] = colors_valid[valid_pix]
            return
        order = np.lexsort((depth_valid, pixel_index))
        pixel_sorted = pixel_index[order]
        depth_sorted = depth_valid[order]
        colors_sorted = colors_valid[order]
        unique_mask = np.ones(pixel_sorted.shape[0], dtype=bool)
        if pixel_sorted.shape[0] > 1:
            unique_mask[1:] = pixel_sorted[1:] != pixel_sorted[:-1]
        selected_pixels = pixel_sorted[unique_mask]
        selected_depth = depth_sorted[unique_mask]
        selected_colors = colors_sorted[unique_mask]
        flat_depth = depth_buf.reshape(-1)
        flat_buf = buf.reshape(-1, 3)
        nearer_mask = selected_depth < flat_depth[selected_pixels]
        if not np.any(nearer_mask):
            return
        target_pixels = selected_pixels[nearer_mask]
        target_depth = selected_depth[nearer_mask]
        target_colors = selected_colors[nearer_mask]
        if point_size <= 1:
            flat_depth[target_pixels] = target_depth
            flat_buf[target_pixels, :] = target_colors
            return
        target_x = (target_pixels % width).astype(np.int32, copy=False)
        target_y = (target_pixels // width).astype(np.int32, copy=False)
        radius = max(0, point_size // 2)
        for dy in range(-radius, radius + 1):
            y_pix = target_y + dy
            y_ok = (y_pix >= 0) & (y_pix < height)
            if not np.any(y_ok):
                continue
            for dx in range(-radius, radius + 1):
                x_pix = target_x + dx
                valid_pix = y_ok & (x_pix >= 0) & (x_pix < width)
                if not np.any(valid_pix):
                    continue
                pixel_block = (
                    y_pix[valid_pix] * width + x_pix[valid_pix]
                ).astype(np.int64, copy=False)
                block_depth = target_depth[valid_pix]
                block_colors = target_colors[valid_pix]
                nearer_block = block_depth < flat_depth[pixel_block]
                if not np.any(nearer_block):
                    continue
                write_pixels = pixel_block[nearer_block]
                flat_depth[write_pixels] = block_depth[nearer_block]
                flat_buf[write_pixels, :] = block_colors[nearer_block]

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
        self.selector_motion_suspect_positions = set()
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
        is_score_suspect = idx in self.selector_score_suspect_positions
        is_motion_suspect = idx in self.selector_motion_suspect_positions
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
        if norm > 0.0:
            # Log display scaling makes lower-score bars easier to see while preserving order.
            log_base = math.log1p(SELECTOR_OVERVIEW_LOG_SCALE)
            if log_base > 1e-12:
                norm = math.log1p(SELECTOR_OVERVIEW_LOG_SCALE * norm) / log_base
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
        if is_score_suspect:
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
        if is_motion_suspect:
            canvas.create_rectangle(
                x0 + 2.5,
                y0 + 2.5,
                x1 - 2.5,
                bar_area_height - 2.5,
                fill="",
                outline="#f4d35e",
                width=2,
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
        pair_base = str(entry.get("pair_base", "") or "").strip()
        if pair_base:
            try:
                return Path(pair_base).stem
            except Exception:
                return pair_base
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

    def _selector_input_base_dir(self) -> Optional[Path]:
        """Return the current FrameSelector input directory as a Path."""
        if not self.selector_vars:
            return None
        in_var = self.selector_vars.get("in_dir")
        if in_var is None:
            return None
        raw_base = str(in_var.get()).strip()
        if not raw_base:
            return None
        try:
            return Path(raw_base).expanduser()
        except Exception:
            return None

    def _selector_resolve_image_name(self, raw_name: str) -> Optional[Path]:
        """Resolve one CSV image name using the selector input directory."""
        raw_name = str(raw_name or "").strip()
        if not raw_name:
            return None
        try:
            candidate = Path(raw_name).expanduser()
        except Exception:
            return None
        if candidate.is_absolute():
            return candidate if candidate.exists() else None
        base_dir = self._selector_input_base_dir()
        if base_dir is None:
            return None
        path_obj = base_dir / candidate
        return path_obj if path_obj.exists() else None

    def _selector_image_paths_for_entry(self, entry: Dict[str, Any]) -> List[Path]:
        """Resolve one or two preview image paths for a CSV entry."""
        raw_names: List[str] = []
        for key in ("x_filename", "y_filename"):
            raw_value = str(entry.get(key, "") or "").strip()
            if raw_value:
                raw_names.append(raw_value)
        if not raw_names:
            raw_value = str(entry.get("filename", "") or "").strip()
            if raw_value:
                raw_names.append(raw_value)

        resolved: List[Path] = []
        seen: Set[str] = set()
        for raw_name in raw_names:
            path_obj = self._selector_resolve_image_name(raw_name)
            if path_obj is None:
                return []
            key = str(path_obj).lower()
            if key in seen:
                continue
            seen.add(key)
            resolved.append(path_obj)
        return resolved

    @staticmethod
    def _compose_selector_preview_sheet(
        image_paths: Sequence[Path],
        images: Sequence[Image.Image],
    ) -> Tuple[Image.Image, str]:
        """Create a side-by-side preview sheet for one or more images."""
        if not images:
            raise ValueError("No preview images to compose.")
        if len(images) == 1:
            return images[0].copy(), image_paths[0].name

        margin = 20
        gap = 20
        label_height = 24
        widths = [img.width for img in images]
        heights = [img.height for img in images]
        sheet_w = margin * 2 + sum(widths) + gap * (len(images) - 1)
        sheet_h = margin * 2 + label_height + max(heights)
        sheet = Image.new("RGB", (sheet_w, sheet_h), "#202020")
        draw = ImageDraw.Draw(sheet)
        x_pos = margin
        for path_obj, image in zip(image_paths, images):
            sheet.paste(image, (x_pos, margin + label_height))
            draw.text((x_pos, margin), path_obj.name, fill="#f0f0f0")
            x_pos += image.width + gap
        label = " | ".join(path_obj.name for path_obj in image_paths)
        return sheet, label

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
        self.selector_preview_panel_select_toggle_button = None
        self.selector_preview_panel_jump_current_button = None
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
        self.selector_preview_panel_select_toggle_button = None
        self.selector_preview_panel_jump_current_button = None

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
        select_toggle_button = tk.Button(
            slider_frame,
            text="Select On/Off",
            width=12,
            command=self._selector_preview_toggle_active_selection,
        )
        select_toggle_button.pack(side=tk.LEFT, padx=(8, 0))
        jump_current_button = tk.Button(
            slider_frame,
            text="Jump to Current Image",
            width=16,
            command=self._selector_preview_jump_to_active_overview,
        )
        jump_current_button.pack(side=tk.LEFT, padx=(4, 0))
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
        self.selector_preview_panel_select_toggle_button = select_toggle_button
        self.selector_preview_panel_jump_current_button = jump_current_button
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

    def _selector_preview_toggle_active_selection(self) -> None:
        """Toggle selected ON/OFF for the currently shown preview image."""
        idx = self.selector_preview_panel_active_idx
        if idx is None:
            self._append_text_widget(
                self.selector_log,
                "[preview] No active preview image to toggle selection.",
            )
            return
        self._toggle_selector_selected_by_index(int(idx))

    def _selector_preview_jump_to_active_overview(self) -> None:
        """Jump the overview to the currently shown preview image."""
        idx = self.selector_preview_panel_active_idx
        if idx is None:
            self._append_text_widget(
                self.selector_log,
                "[preview] No active preview image to jump.",
            )
            return
        if idx < 0 or idx >= len(self.selector_score_entries):
            return
        self._selector_overview_set_zoom(
            self._selector_overview_zoom_for_visible_bars(
                SELECTOR_OVERVIEW_PRESET_VISIBLE_BARS_MAX
            ),
            focus_idx=int(idx),
        )
        self._append_text_widget(
            self.selector_log,
            "[overview] Jumped to current preview image: {}".format(
                self._selector_entry_label_text(self.selector_score_entries[int(idx)])
            ),
        )

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
            label = str(item.get("label", "") or "").strip()
            if label:
                name = label
            else:
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
        self._toggle_selector_selected_by_index(idx)

    def _on_selector_score_canvas_right_click(self, event: tk.Event) -> str:
        """Toggle preview window for the clicked overview bar."""
        idx = self._selector_bar_index_from_event(event)
        if idx is None:
            return "break"
        self._toggle_selector_preview_window(idx)
        return "break"

    def _toggle_selector_selected_by_index(self, idx: int) -> bool:
        """Toggle selected state for one overview entry and refresh UI."""
        if idx < 0 or idx >= len(self.selector_score_entries):
            return False
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
        return True

    def _add_selector_preview_item_by_index(
        self,
        idx: int,
        show_errors: bool = True,
    ) -> bool:
        """Load one single image or X/Y pair and add it to the preview set."""
        if idx < 0 or idx >= len(self.selector_score_entries):
            return False
        entry = self.selector_score_entries[idx]
        image_paths = self._selector_image_paths_for_entry(entry)
        if not image_paths:
            if show_errors:
                messagebox.showerror(
                    "FrameSelector",
                    "Could not resolve preview image path(s) for {}.".format(
                        self._selector_entry_label_text(entry)
                    ),
                )
            return False
        preview_images: List[Image.Image] = []
        try:
            for image_path in image_paths:
                with Image.open(image_path) as img:
                    preview_images.append(img.convert("RGB").copy())
        except Exception as exc:
            if show_errors:
                messagebox.showerror(
                    "FrameSelector",
                    "Failed to open preview image(s):\n{}\n\n{}".format(
                        "\n".join(str(path_obj) for path_obj in image_paths),
                        exc,
                    ),
                )
            return False
        try:
            image, label = self._compose_selector_preview_sheet(
                image_paths,
                preview_images,
            )
        except Exception as exc:
            if show_errors:
                messagebox.showerror(
                    "FrameSelector",
                    "Failed to compose preview image(s):\n{}\n\n{}".format(
                        "\n".join(str(path_obj) for path_obj in image_paths),
                        exc,
                    ),
                )
            return False
        self.selector_preview_items[idx] = {
            "image": image,
            "path": image_paths[0],
            "paths": list(image_paths),
            "label": label,
        }
        return True

    def _toggle_selector_preview_window(self, idx: int) -> None:
        """Open/close one image inside the consolidated preview panel."""
        if self._remove_selector_preview_item(idx):
            return
        had_preview_items = bool(self.selector_preview_items)
        old_active = self.selector_preview_panel_active_idx
        if not self._add_selector_preview_item_by_index(idx, show_errors=True):
            return
        self.selector_preview_panel_active_idx = idx
        preserve_zoom = old_active is not None
        self._sync_selector_preview_panel_controls(preserve_zoom=preserve_zoom)
        if not had_preview_items and self.selector_preview_panel_active_idx is not None:
            self.selector_preview_panel_zoom_ratio = max(
                0.05,
                min(16.0, float(SELECTOR_PREVIEW_DEFAULT_OPEN_ZOOM_RATIO)),
            )
            self._render_selector_preview_panel_image()
        self._center_selector_preview_view_on_image_center()
        if not had_preview_items:
            try:
                self.root.after_idle(self._center_selector_preview_view_on_image_center)
            except Exception:
                pass
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
        if not had_preview_items and self.selector_preview_panel_active_idx is not None:
            self.selector_preview_panel_zoom_ratio = max(
                0.05,
                min(16.0, float(SELECTOR_PREVIEW_DEFAULT_OPEN_ZOOM_RATIO)),
            )
            self._render_selector_preview_panel_image()
        self._center_selector_preview_view_on_image_center()
        if not had_preview_items:
            try:
                self.root.after_idle(self._center_selector_preview_view_on_image_center)
            except Exception:
                pass
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

    def _selector_overview_current_center_index(self) -> Optional[int]:
        """Return the bar index near the current overview viewport center."""
        canvas = self.selector_score_canvas
        total = len(self.selector_score_entries)
        if canvas is None or total <= 0 or self.selector_score_bar_width <= 0.0:
            return None
        try:
            canvas.update_idletasks()
        except Exception:
            pass
        view_w = max(1.0, float(canvas.winfo_width() or 1.0))
        x_center = float(canvas.canvasx(0)) + (view_w * 0.5)
        idx = int(x_center // float(self.selector_score_bar_width))
        if idx < 0:
            return 0
        if idx >= total:
            return total - 1
        return idx

    def _selector_overview_scroll_to_index(self, idx: int) -> None:
        """Scroll the overview horizontally so the target bar is centered."""
        canvas = self.selector_score_canvas
        total = len(self.selector_score_entries)
        if canvas is None or total <= 0:
            return
        if idx < 0 or idx >= total:
            return
        if self.selector_score_bar_width <= 0.0:
            return
        try:
            canvas.update_idletasks()
        except Exception:
            pass
        total_w = max(1.0, float(self.selector_score_total_width or 1.0))
        view_w = max(1.0, float(canvas.winfo_width() or 1.0))
        target_x = (float(idx) * float(self.selector_score_bar_width)) + (
            float(self.selector_score_bar_width) * 0.5
        )
        left = target_x - (view_w * 0.5)
        max_left = max(0.0, total_w - view_w)
        left = max(0.0, min(max_left, left))
        try:
            canvas.xview_moveto(left / total_w)
        except Exception:
            pass

    def _selector_overview_set_zoom(
        self,
        zoom_value: float,
        focus_idx: Optional[int] = None,
    ) -> None:
        """Set overview X zoom and optionally center a target bar."""
        if not self.selector_score_entries:
            return
        new_zoom = max(
            SELECTOR_OVERVIEW_X_ZOOM_MIN,
            min(SELECTOR_OVERVIEW_X_ZOOM_MAX, float(zoom_value)),
        )
        if abs(float(self.selector_score_zoom) - new_zoom) > 1e-9:
            self.selector_score_zoom = new_zoom
            self._refresh_selector_score_view(reset_view=False)
        if focus_idx is not None:
            self._selector_overview_scroll_to_index(int(focus_idx))

    def _selector_overview_zoom_for_visible_bars(self, visible_bars: int) -> float:
        """Return zoom value that aims to show approximately N bars in the viewport."""
        total = len(self.selector_score_entries)
        try:
            target = int(visible_bars)
        except Exception:
            target = 1
        target = max(1, target)
        if total <= 0:
            return SELECTOR_OVERVIEW_X_ZOOM_MIN
        zoom = float(total) / float(target)
        return max(
            SELECTOR_OVERVIEW_X_ZOOM_MIN,
            min(SELECTOR_OVERVIEW_X_ZOOM_MAX, zoom),
        )

    def _selector_overview_zoom_max(self) -> None:
        """Set overview X zoom to the preset that shows about 50 bars."""
        self._selector_overview_set_zoom(
            self._selector_overview_zoom_for_visible_bars(
                SELECTOR_OVERVIEW_PRESET_VISIBLE_BARS_MAX
            )
        )

    def _selector_overview_zoom_half(self) -> None:
        """Set overview X zoom to the preset that shows about 500 bars."""
        self._selector_overview_set_zoom(
            self._selector_overview_zoom_for_visible_bars(
                SELECTOR_OVERVIEW_PRESET_VISIBLE_BARS_HALF
            )
        )

    def _selector_overview_zoom_min(self) -> None:
        """Set overview X zoom to the minimum (fit whole view behavior)."""
        total = len(self.selector_score_entries)
        if total <= 0:
            return
        self._selector_overview_set_zoom(
            self._selector_overview_zoom_for_visible_bars(total)
        )

    def _jump_to_next_selector_suspect(self) -> None:
        """Jump the overview viewport to the next suspect bar and max zoom."""
        if not self.selector_score_entries:
            self._append_text_widget(
                self.selector_log,
                "[overview] No score overview data. Run Check Selection first.",
            )
            return
        suspect_indices = sorted(
            idx
            for idx in self.selector_score_suspect_positions
            if 0 <= idx < len(self.selector_score_entries)
        )
        if not suspect_indices:
            self._append_text_widget(
                self.selector_log,
                "[overview] No suspects are currently marked.",
            )
            return

        current_idx: Optional[int] = None
        active_idx = self.selector_preview_panel_active_idx
        if active_idx in self.selector_score_suspect_positions:
            current_idx = int(active_idx)
        elif (
            self.selector_last_suspect_jump_idx is not None
            and self.selector_last_suspect_jump_idx in self.selector_score_suspect_positions
        ):
            current_idx = int(self.selector_last_suspect_jump_idx)
        else:
            current_idx = self._selector_overview_current_center_index()

        target_idx = suspect_indices[0]
        if current_idx is not None:
            for idx in suspect_indices:
                if idx > current_idx:
                    target_idx = idx
                    break

        self.selector_last_suspect_jump_idx = int(target_idx)
        self._selector_overview_set_zoom(
            self._selector_overview_zoom_for_visible_bars(
                SELECTOR_OVERVIEW_PRESET_VISIBLE_BARS_MAX
            ),
            focus_idx=int(target_idx),
        )

        if 0 <= target_idx < len(self.selector_score_entries):
            label = self._selector_entry_label_text(self.selector_score_entries[target_idx])
            self._append_text_widget(
                self.selector_log,
                "[overview] Jumped to next suspect: {}".format(label),
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
        new_zoom = max(
            SELECTOR_OVERVIEW_X_ZOOM_MIN,
            min(SELECTOR_OVERVIEW_X_ZOOM_MAX, old_zoom * zoom_step),
        )
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
                    brightness_key = field_map.get("brightness_mean")
                    flow_key = field_map.get("flow_motion")
                    filename_key = field_map.get("filename")
                    index_key = field_map.get("index")
                    input_mode_key = field_map.get("input_mode")
                    pair_base_key = field_map.get("pair_base")
                    x_filename_key = field_map.get("x_filename")
                    y_filename_key = field_map.get("y_filename")
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
                        brightness_val: Optional[float] = None
                        if brightness_key:
                            brightness_raw = row.get(brightness_key)
                            try:
                                brightness_val = float(brightness_raw)
                            except (TypeError, ValueError):
                                brightness_val = None
                            if brightness_val is not None and not math.isfinite(brightness_val):
                                brightness_val = None
                        flow_val: Optional[float] = None
                        if flow_key:
                            flow_raw = row.get(flow_key)
                            try:
                                flow_val = float(flow_raw)
                            except (TypeError, ValueError):
                                flow_val = None
                            if flow_val is not None and not math.isfinite(flow_val):
                                flow_val = None
                        if index_key:
                            idx_raw = row.get(index_key, "")
                        else:
                            idx_raw = ""
                        try:
                            frame_idx = int(idx_raw)
                        except (TypeError, ValueError):
                            frame_idx = row_counter
                        fname = row.get(filename_key, "") if filename_key else ""
                        input_mode = row.get(input_mode_key, "") if input_mode_key else ""
                        pair_base = row.get(pair_base_key, "") if pair_base_key else ""
                        x_filename = row.get(x_filename_key, "") if x_filename_key else ""
                        y_filename = row.get(y_filename_key, "") if y_filename_key else ""
                        idx_text = row.get(index_key, "") if index_key else ""
                        parsed_entries.append(
                            {
                                "row_counter": row_counter,
                                "frame_idx": frame_idx,
                                "selected_original": selected_flag,
                                "selected_current": selected_flag,
                                "score": score_val,
                                "brightness_mean": brightness_val,
                                "flow_motion": flow_val,
                                "filename": fname,
                                "input_mode": input_mode,
                                "pair_base": pair_base,
                                "x_filename": x_filename,
                                "y_filename": y_filename,
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
            selected_entries: List[Dict[str, Any]] = []
            for pos, entry in enumerate(parsed_entries):
                if not bool(entry.get("selected_current", False)):
                    continue
                score_val = entry.get("score")
                if score_val is None:
                    continue
                brightness_val = entry.get("brightness_mean")
                if brightness_val is not None:
                    try:
                        brightness_val = float(brightness_val)
                    except (TypeError, ValueError):
                        brightness_val = None
                selected_entries.append(
                    {
                        "score": float(score_val),
                        "filename": str(
                            entry.get("pair_base", "")
                            or entry.get("filename", "")
                            or ""
                        ),
                        "index_text": str(entry.get("index_text", "")),
                        "pos": pos,
                        "brightness_mean": brightness_val,
                    }
                )

            selected_entries.sort(key=lambda item: float(item["score"]))
            selected_flag_total = sum(1 for entry in parsed_entries if bool(entry.get("selected_current", False)))
            limit_percent = self._selector_suspect_percent()
            if selected_entries:
                max_lines = max(1, min(200, math.ceil((limit_percent / 100.0) * len(selected_entries))))
                suspects = []
                if max_lines > 0:
                    valid_brightness_entries = [
                        item for item in selected_entries
                        if item.get("brightness_mean") is not None
                    ]
                    if valid_brightness_entries:
                        b_vals = [float(item["brightness_mean"]) for item in valid_brightness_entries]
                        b_min = min(b_vals)
                        b_max = max(b_vals)
                    else:
                        b_min = 0.0
                        b_max = 0.0

                    use_banded = (
                        len(valid_brightness_entries) >= 2
                        and (b_max - b_min) > 1e-9
                        and max_lines >= 2
                    )
                    if use_banded:
                        bin_count = max(
                            2,
                            min(
                                SELECTOR_SUSPECT_BRIGHTNESS_BINS,
                                max_lines,
                                len(valid_brightness_entries),
                            ),
                        )
                        buckets: List[List[Dict[str, Any]]] = [[] for _ in range(bin_count)]
                        for item in valid_brightness_entries:
                            b_val = float(item["brightness_mean"])
                            norm_b = (b_val - b_min) / (b_max - b_min)
                            bin_idx = int(norm_b * bin_count)
                            if bin_idx >= bin_count:
                                bin_idx = bin_count - 1
                            elif bin_idx < 0:
                                bin_idx = 0
                            buckets[bin_idx].append(item)
                        for bucket in buckets:
                            bucket.sort(key=lambda item: float(item["score"]))

                        chosen_pos: Set[int] = set()
                        while len(suspects) < max_lines:
                            progressed = False
                            for bucket in buckets:
                                while bucket:
                                    candidate = bucket.pop(0)
                                    pos_val = int(candidate["pos"])
                                    if pos_val in chosen_pos:
                                        continue
                                    suspects.append(candidate)
                                    chosen_pos.add(pos_val)
                                    progressed = True
                                    break
                                if len(suspects) >= max_lines:
                                    break
                            if not progressed:
                                break
                        if len(suspects) < max_lines:
                            for item in selected_entries:
                                pos_val = int(item["pos"])
                                if pos_val in chosen_pos:
                                    continue
                                suspects.append(item)
                                chosen_pos.add(pos_val)
                                if len(suspects) >= max_lines:
                                    break
                    else:
                        suspects = list(selected_entries[:max_lines])
            else:
                max_lines = 0
                suspects = []
            score_suspect_positions = {int(item["pos"]) for item in suspects}
            motion_suspect_positions: Set[int] = set()
            low_motion_spans: List[Dict[str, Any]] = []
            compute_flow_var = self.selector_vars.get("compute_optical_flow")
            compute_flow_highlight = (
                bool(compute_flow_var.get()) if compute_flow_var is not None else False
            )
            if compute_flow_highlight:
                flow_threshold = self._selector_optical_flow_threshold_value(
                    show_error=True
                )
                if flow_threshold is None:
                    return
                low_motion_spans = self._selector_collect_low_motion_spans(
                    parsed_entries,
                    flow_threshold,
                )
                for span in low_motion_spans:
                    motion_suspect_positions.update(
                        range(int(span["start_pos"]), int(span["end_pos"]) + 1)
                    )

            self.selector_score_entries = parsed_entries
            self.selector_score_suspect_positions = score_suspect_positions
            self.selector_motion_suspect_positions = motion_suspect_positions
            self.selector_score_csv_path_loaded = csv_path
            self.selector_score_csv_fieldnames = list(headers)
            self.selector_score_selected_key = selected_key
            self._refresh_selector_score_view(reset_view=True)

            self._append_text_widget(
                self.selector_log,
                (
                    f"[score] {csv_path.name}: selected={selected_flag_total} "
                    f"showing {len(suspects)} score suspects ({limit_percent:.1f}% of selected, limit={max_lines}, brightness-banded)"
                ),
            )
            if low_motion_spans:
                flow_threshold = self._selector_optical_flow_threshold_value()
                self._append_text_widget(
                    self.selector_log,
                    (
                        f"[flow] low-motion spans={len(low_motion_spans)} "
                        f"(threshold <= {float(flow_threshold):.4f}); highlighted in gold and excluded from suspect count."
                    ),
                )
            if not suspects:
                self._append_text_widget(
                    self.selector_log,
                    "[score] No low-score selections detected.",
                )
                if not low_motion_spans:
                    return

            for item in suspects:
                score = float(item["score"])
                fname = str(item.get("filename", ""))
                idx_text = str(item.get("index_text", ""))
                brightness_val = item.get("brightness_mean")
                label = fname or (f"index {idx_text}" if idx_text else "(unknown)")
                if brightness_val is None:
                    extra = ""
                else:
                    extra = f", brightness={float(brightness_val):.4f}"
                self._append_text_widget(
                    self.selector_log,
                    f"  - {label} (score={score:.4f}{extra})",
                )
            for span in low_motion_spans:
                start_pos = int(span["start_pos"])
                end_pos = int(span["end_pos"])
                start_entry = parsed_entries[start_pos]
                end_entry = parsed_entries[end_pos]
                start_label = self._selector_entry_label_text(start_entry)
                end_label = self._selector_entry_label_text(end_entry)
                flow_value = float(span.get("max_flow", 0.0))
                self._append_text_widget(
                    self.selector_log,
                    (
                        f"  - {start_label} -> {end_label} "
                        f"(selected={int(span.get('selected_count', 0))}, "
                        f"frames={int(span.get('frame_count', 0))}, "
                        f"max_flow={flow_value:.4f})"
                    ),
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
        zoom = max(
            SELECTOR_OVERVIEW_X_ZOOM_MIN,
            min(SELECTOR_OVERVIEW_X_ZOOM_MAX, float(self.selector_score_zoom)),
        )
        bar_width = max(
            SELECTOR_OVERVIEW_BAR_WIDTH_MIN,
            min(SELECTOR_OVERVIEW_BAR_WIDTH_MAX, approx_width * zoom),
        )
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
        screen_w = max(1, int(self.root.winfo_screenwidth()))
        screen_h = max(1, int(self.root.winfo_screenheight()))
        if self.controls_frame is not None:
            controls_width = max(
                self.controls_frame.winfo_width(),
                self.controls_frame.winfo_reqwidth(),
                420,
            )
        else:
            controls_width = 420
        total_width = int(self.display_width + controls_width + 48)
        total_height = int(self.display_height + 120)
        target_default_w, target_default_h = self._physical_px_to_logical(
            self.DEFAULT_WINDOW_WIDTH_PX,
            self.DEFAULT_WINDOW_HEIGHT_PX,
        )
        target_min_w, target_min_h = self._physical_px_to_logical(
            self.DEFAULT_WINDOW_MIN_WIDTH_PX,
            self.DEFAULT_WINDOW_MIN_HEIGHT_PX,
        )
        max_w = max(target_min_w, screen_w - 90)
        max_h = max(target_min_h, screen_h - 90)
        default_w = min(target_default_w, max_w)
        default_h = min(target_default_h, max_h)
        min_w = min(max(target_min_w, min(total_width, default_w)), max_w)
        min_h = min(max(target_min_h, min(total_height, default_h)), max_h)
        current_w = self.root.winfo_width()
        current_h = self.root.winfo_height()
        current_x = max(0, int(self.root.winfo_x()))
        current_y = max(0, int(self.root.winfo_y()))
        desired_w = default_w
        desired_h = default_h
        self.root.minsize(min_w, min_h)
        if (
            desired_w != current_w
            or desired_h != current_h
            or min_h > current_h
            or current_h > max_h
        ):
            max_x = max(24, screen_w - desired_w - 24)
            max_y = max(24, screen_h - desired_h - 48)
            new_x = min(max(24, current_x), max_x)
            new_y = min(max(24, current_y), max_y)
            self.root.geometry(f"{desired_w}x{desired_h}+{new_x}+{new_y}")

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
            self.preview_frame.configure(width=self.display_width + 10)
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

 
