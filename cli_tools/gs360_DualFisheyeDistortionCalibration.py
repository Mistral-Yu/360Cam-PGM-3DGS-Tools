#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Dual-fisheye distortion correction tool using Metashape XML calibration.

Overview:
    - Reads camera intrinsics/distortion from Metashape XML.
    - Applies equisolid_fisheye projection + Brown distortion model.
    - Undistorts fisheye frame pairs (e.g. *_X.jpg / *_Y.jpg).
    - Optionally exports perspective views from corrected fisheye frames.

Reference formulas:
    Agisoft Metashape Professional User Manual 2.3
    Appendix D. Camera models, "Fisheye cameras" section.
"""

import argparse
import ctypes
import math
import os
import pathlib
import sys
import xml.etree.ElementTree as ET

import gs360_CameraFormatConverter as camera_converter
import gs360_MS360xmlToPersCams as msxml_converter

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

try:
    import cv2  # type: ignore
except ImportError as exc:  # pragma: no cover
    print(
        "[ERR] OpenCV is required: pip install opencv-python",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    print("[ERR] NumPy is required: pip install numpy", file=sys.stderr)
    raise SystemExit(1) from exc


SUPPORTED_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
SUPPORTED_MODELS = {"equisolid_fisheye"}
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_CAMERA_XML = (
    SCRIPT_DIR / "templates" / "Osmo360-Fisheye-Distortion.xml"
)
DEFAULT_DLOGM_LUT = SCRIPT_DIR / "templates" / \
    "DJI Osmo 360 D-Log M to Rec.709 V1.cube"
DEFAULT_WORKERS = max(1, os.cpu_count() or 1)
DEFAULT_PERSPECTIVE_METASHAPE_XML_NAME = "perspective_cams.xml"

INTERPOLATION_MAP = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "lanczos4": cv2.INTER_LANCZOS4,
}


@dataclass
class SensorCalibration:
    """Metashape sensor calibration for fisheye undistortion."""

    sensor_id: str
    model_type: str
    width: int
    height: int
    f: float
    cx: float
    cy: float
    k1: float
    k2: float
    k3: float
    k4: float
    p1: float
    p2: float
    b1: float
    b2: float


@dataclass
class RemapCache:
    """Precomputed remap data for one calibration profile."""

    map_x: np.ndarray
    map_y: np.ndarray
    valid_mask: np.ndarray
    undistort_zoom: float


@dataclass
class CubeLUT:
    """3D LUT and domain parameters loaded from a .cube file."""

    size: int
    table: np.ndarray
    domain_min: np.ndarray
    domain_max: np.ndarray


class MemoryStatusEx(ctypes.Structure):
    """Windows MEMORYSTATUSEX structure."""

    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def parse_arguments() -> argparse.Namespace:
    """Build and parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Dual-fisheye calibration tool: export perspective views by "
            "default, "
            "optionally save validation-oriented fisheye outputs, and "
            "optionally export color-corrected-only images. "
            "Use gs360_Video2Frames.py beforehand for video frame extraction."
        )
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        required=False,
        help=(
            "Input directory containing fisheye frame pairs "
            "(e.g. *_X.jpg, *_Y.jpg). "
            "Video files are not accepted in this tool. "
            "Optional when --metadata-only is used."
        ),
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help=(
            "Export COLMAP text + perspective Metashape XML only. "
            "Skips fisheye/perspective image rendering and allows running "
            "from --camera-extrinsics-xml + --pointcloud-ply without "
            "--input-dir."
        ),
    )
    parser.add_argument(
        "-x",
        "--camera-xml",
        default=str(DEFAULT_CAMERA_XML),
        help=(
            "Metashape camera XML path (contains sensor calibration). "
            "Optional when --camera-extrinsics-xml provides adjusted "
            "calibration. Default: "
            "cli_tools/templates/Osmo360-Fisheye-Distortion.xml"
        ),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help=(
            "Undistorted fisheye output directory "
            "(default: <fisheye_dir>_undistorted)."),
    )
    parser.add_argument(
        "--suffixes",
        default="_X,_Y",
        help="Comma-separated stem suffix filter (default: _X,_Y).",
    )
    parser.add_argument(
        "--ext",
        default="jpg,jpeg,png,tif,tiff",
        help=(
            "Comma-separated extensions to process "
            "(default: jpg,jpeg,png,tif,tiff)."
        ),
    )
    parser.add_argument(
        "--input-lut",
        default=None,
        help=(
            "Optional input 3D LUT (.cube) applied before undistortion. "
            "Use this for camera/log transforms."
        ),
    )
    parser.add_argument(
        "--lut-output-color-space",
        metavar="{passthrough,srgb}",
        default="srgb",
        help=(
            "Color space to save after LUT application. Use 'srgb' to "
            "convert LUT output from Rec.709 to sRGB before saving, or "
            "'passthrough' to keep the LUT output as-is. Without --input-lut, "
            "'passthrough' means the original image color space is preserved."
        ),
    )
    parser.add_argument(
        "--input-color-profile",
        choices=("native", "osmo360-dlogm"),
        default="native",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--dlogm-lut",
        default=str(DEFAULT_DLOGM_LUT),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--sensor-id-x",
        default=None,
        help="Optional sensor_id override for *_X frames.",
    )
    parser.add_argument(
        "--sensor-id-y",
        default=None,
        help="Optional sensor_id override for *_Y frames.",
    )
    parser.add_argument(
        "--interpolation",
        choices=tuple(INTERPOLATION_MAP.keys()),
        default="cubic",
        help="Resampling interpolation (default: cubic).",
    )
    parser.add_argument(
        "--undistort-zoom",
        default="auto",
        help=(
            "Undistort zoom factor. Use a positive float (e.g. 1.1) or "
            "'auto' to choose the minimum zoom that avoids "
            "source out-of-bounds."
        ),
    )
    parser.add_argument(
        "--mask-outside-model",
        dest="mask_outside_model",
        action="store_true",
        help="Mask pixels outside the ideal model radius as constant color.",
    )
    parser.add_argument(
        "--no-mask-outside-model",
        dest="mask_outside_model",
        action="store_false",
        help=(
            "Disable model/FOV based masking for undistorted fisheye outputs."
        ),
    )
    parser.set_defaults(mask_outside_model=True)
    parser.add_argument(
        "--mask-value",
        type=int,
        default=0,
        help="Mask fill value in [0,255] when model/FOV masking is enabled.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=(
            "Worker threads for pair processing "
            "(default: CPU core count = {})."
        ).format(DEFAULT_WORKERS),
    )
    parser.add_argument(
        "--memory-throttle-percent",
        type=float,
        default=80.0,
        help=(
            "Reduce active worker submission when system memory usage "
            "exceeds this percent (default: 80)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List targets and calibration mapping without writing files.",
    )
    parser.add_argument(
        "--report-json",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-perspective",
        action="store_true",
        help="Disable perspective conversion stage.",
    )
    parser.add_argument(
        "--save-fisheye-output",
        action="store_true",
        help=(
            "Save undistorted fisheye images for validation "
            "(default: disabled)."
        ),
    )
    parser.add_argument(
        "--save-color-corrected-output",
        action="store_true",
        help=(
            "Save input images after input color-profile conversion only, "
            "before "
            "undistortion (default: disabled)."),
    )
    parser.add_argument(
        "--color-corrected-output-dir",
        default=None,
        help=(
            "Color-corrected-only output dir "
            "(default: <fisheye_dir>_colorcorrected)."
        ),
    )
    parser.add_argument(
        "--fisheye-output-dir",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-fisheye-output",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--perspective-output-dir",
        default=None,
        help=(
            "Perspective / COLMAP root dir "
            "(default: <fisheye_dir>_perspective_colmap)."
        ),
    )
    parser.add_argument(
        "--perspective-ext",
        default="jpg",
        help="Perspective output extension (default: jpg).",
    )
    parser.add_argument(
        "--perspective-mask-ext",
        default="png",
        help=(
            "Perspective mask output extension (default: png). "
            "Used only when --mask-input-dir is provided."
        ),
    )
    parser.add_argument(
        "--perspective-size",
        type=int,
        default=1750,
        help="Perspective output size (default: 1750).",
    )
    parser.add_argument(
        "--perspective-focal-mm",
        type=float,
        default=14.0,
        help="Perspective focal length in mm (default: 14).",
    )
    parser.add_argument(
        "--perspective-sensor-mm",
        default="36 36",
        help="Perspective sensor size string (default: '36 36').",
    )
    parser.add_argument(
        "--perspective-yaw-delta-deg",
        type=float,
        default=40.0,
        help="Yaw delta in degrees for SFM10 layout (default: 40).",
    )
    parser.add_argument(
        "--perspective-pitch-delta-deg",
        type=float,
        default=40.0,
        help="Pitch delta in degrees for SFM10 layout (default: 40).",
    )
    parser.add_argument(
        "--perspective-jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality for perspective outputs.",
    )
    parser.add_argument(
        "--lens-fov-deg",
        type=float,
        default=190.0,
        help=(
            "Usable fisheye FOV per lens in degrees "
            "(default: 190 for >180 lenses)."
        ),
    )
    parser.add_argument(
        "--lens-x-yaw-deg",
        type=float,
        default=0.0,
        help="Rig yaw offset for X lens.",
    )
    parser.add_argument(
        "--lens-y-yaw-deg",
        type=float,
        default=180.0,
        help="Rig yaw offset for Y lens.",
    )
    parser.add_argument(
        "--camera-extrinsics-xml",
        default=None,
        help=(
            "Optional Metashape alignment XML for the input dual-fisheye "
            "pairs. When provided, the tool also exports perspective "
            "Metashape XML and COLMAP text using the current perspective "
            "layout."
        ),
    )
    parser.add_argument(
        "--pointcloud-ply",
        default=None,
        help=(
            "Optional Metashape point cloud PLY used when exporting "
            "perspective COLMAP text."
        ),
    )
    parser.add_argument(
        "--mask-input-dir",
        default=None,
        help=(
            "Optional mask folder matching the pair images by file name. "
            "When provided, perspective masks are cut with the same remap "
            "and written under the COLMAP Masks folder."
        ),
    )
    parser.add_argument(
        "--perspective-metashape-xml-name",
        default=DEFAULT_PERSPECTIVE_METASHAPE_XML_NAME,
        help=(
            "Perspective Metashape XML file name written under the "
            "perspective output directory."
        ),
    )
    return parser.parse_args()


def _parse_float(node: Optional[ET.Element],
                 key: str, default: float = 0.0) -> float:
    """Parse float value from child tag and return default on absence."""

    if node is None:
        return default
    target = node.find(key)
    if target is None or target.text is None:
        return default
    return float(target.text)


def parse_undistort_zoom_arg(value: str) -> Optional[float]:
    """Parse undistort zoom argument.

    Returns:
        None when auto mode is requested, otherwise a positive float.
    """

    text = (value or "").strip().lower()
    if not text or text == "auto":
        return None
    zoom = float(text)
    if zoom <= 0.0:
        raise ValueError("undistort zoom must be > 0")
    return zoom


def normalize_lut_output_color_space(value: str) -> str:
    """Normalize LUT output color space and accept legacy aliases."""

    text = str(value or "passthrough").strip().lower()
    if text == "native":
        return "passthrough"
    if text in {"passthrough", "srgb"}:
        return text
    raise ValueError(
        "Unsupported --lut-output-color-space: {}".format(value)
    )


def load_cube_lut(lut_path: pathlib.Path) -> CubeLUT:
    """Load a 3D LUT from .cube format."""

    if not lut_path.exists() or not lut_path.is_file():
        raise FileNotFoundError("LUT file not found: {}".format(lut_path))

    size: Optional[int] = None
    domain_min = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    domain_max = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    rows: List[Tuple[float, float, float]] = []

    with lut_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            upper = line.upper()
            if upper.startswith("TITLE"):
                continue
            if upper.startswith("LUT_3D_SIZE"):
                parts = line.split()
                if len(parts) < 2:
                    raise ValueError(
                        "Invalid LUT_3D_SIZE line: {}".format(line))
                size = int(parts[1])
                continue
            if upper.startswith("DOMAIN_MIN"):
                parts = line.split()
                if len(parts) != 4:
                    raise ValueError(
                        "Invalid DOMAIN_MIN line: {}".format(line))
                domain_min = np.array([float(parts[1]), float(
                    parts[2]), float(parts[3])], dtype=np.float32)
                continue
            if upper.startswith("DOMAIN_MAX"):
                parts = line.split()
                if len(parts) != 4:
                    raise ValueError(
                        "Invalid DOMAIN_MAX line: {}".format(line))
                domain_max = np.array([float(parts[1]), float(
                    parts[2]), float(parts[3])], dtype=np.float32)
                continue
            parts = line.split()
            if len(parts) == 3:
                rows.append(
                    (float(parts[0]), float(parts[1]), float(parts[2])))

    if size is None:
        raise ValueError("LUT_3D_SIZE is missing in {}".format(lut_path))
    if size <= 1:
        raise ValueError("LUT_3D_SIZE must be > 1 in {}".format(lut_path))

    expected = int(size ** 3)
    if len(rows) != expected:
        raise ValueError(
            "LUT row count mismatch in {}: got {}, expected {}".format(
                lut_path,
                len(rows),
                expected,
            )
        )

    span = domain_max - domain_min
    if np.any(span <= 0.0):
        raise ValueError("Invalid LUT domain range in {}".format(lut_path))

    table = np.asarray(rows, dtype=np.float32).reshape((size, size, size, 3))
    return CubeLUT(
        size=size,
        table=table,
        domain_min=domain_min,
        domain_max=domain_max)


def rec709_to_linear(values: np.ndarray) -> np.ndarray:
    """Convert Rec.709 encoded values [0,1] to linear light."""

    v = np.clip(values.astype(np.float32), 0.0, 1.0)
    out = np.empty_like(v, dtype=np.float32)
    low = v < 0.081
    out[low] = v[low] / 4.5
    out[~low] = np.power((v[~low] + 0.099) / 1.099,
                         1.0 / 0.45).astype(np.float32)
    return out


def linear_to_srgb(values: np.ndarray) -> np.ndarray:
    """Convert linear light values [0,1] to sRGB encoded values."""

    v = np.clip(values.astype(np.float32), 0.0, 1.0)
    out = np.empty_like(v, dtype=np.float32)
    low = v <= 0.0031308
    out[low] = 12.92 * v[low]
    out[~low] = (1.055 * np.power(v[~low], 1.0 / 2.4) -
                 0.055).astype(np.float32)
    return np.clip(out, 0.0, 1.0)


def rec709_to_srgb(values: np.ndarray) -> np.ndarray:
    """Convert Rec.709 encoded values [0,1] to sRGB encoded values [0,1]."""

    linear = rec709_to_linear(values)
    return linear_to_srgb(linear)


def image_to_float01(image: np.ndarray) -> np.ndarray:
    """Convert uint8/uint16 image data to float32 [0,1]."""

    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    if image.dtype == np.uint16:
        return image.astype(np.float32) / 65535.0
    if np.issubdtype(image.dtype, np.floating):
        return np.clip(image.astype(np.float32), 0.0, 1.0)
    raise TypeError(
        "Unsupported image dtype for conversion: {}".format(image.dtype))


def float01_to_image(values: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Convert float32 [0,1] to target dtype."""

    v = np.clip(values.astype(np.float32), 0.0, 1.0)
    if dtype == np.uint8:
        return np.rint(v * 255.0).astype(np.uint8)
    if dtype == np.uint16:
        return np.rint(v * 65535.0).astype(np.uint16)
    if np.issubdtype(dtype, np.floating):
        return v.astype(dtype)
    raise TypeError("Unsupported output dtype: {}".format(dtype))


def apply_cube_lut_trilinear(
    rgb_values: np.ndarray,
    lut: CubeLUT,
    chunk_pixels: int = 250000,
) -> np.ndarray:
    """Apply a 3D LUT with trilinear interpolation in RGB space."""

    if rgb_values.ndim < 3 or rgb_values.shape[-1] != 3:
        raise ValueError("apply_cube_lut_trilinear expects (..., 3) RGB array")

    flat = rgb_values.reshape((-1, 3)).astype(np.float32)
    out = np.empty_like(flat, dtype=np.float32)

    domain_min = lut.domain_min.reshape((1, 3))
    domain_max = lut.domain_max.reshape((1, 3))
    span = domain_max - domain_min
    size = int(lut.size)
    max_index = size - 1
    table = lut.table

    total = flat.shape[0]
    step = max(1, int(chunk_pixels))
    for start in range(0, total, step):
        end = min(total, start + step)
        chunk = flat[start:end]
        coord = np.clip((chunk - domain_min) / span, 0.0, 1.0)
        pos = coord * float(max_index)

        idx0 = np.floor(pos).astype(np.int32)
        idx1 = np.minimum(idx0 + 1, max_index)
        frac = pos - idx0.astype(np.float32)

        r0, g0, b0 = idx0[:, 0], idx0[:, 1], idx0[:, 2]
        r1, g1, b1 = idx1[:, 0], idx1[:, 1], idx1[:, 2]

        c000 = table[b0, g0, r0]
        c100 = table[b0, g0, r1]
        c010 = table[b0, g1, r0]
        c110 = table[b0, g1, r1]
        c001 = table[b1, g0, r0]
        c101 = table[b1, g0, r1]
        c011 = table[b1, g1, r0]
        c111 = table[b1, g1, r1]

        fr = frac[:, 0:1]
        fg = frac[:, 1:2]
        fb = frac[:, 2:3]

        c00 = c000 + (c100 - c000) * fr
        c10 = c010 + (c110 - c010) * fr
        c01 = c001 + (c101 - c001) * fr
        c11 = c011 + (c111 - c011) * fr
        c0 = c00 + (c10 - c00) * fg
        c1 = c01 + (c11 - c01) * fg
        out[start:end] = c0 + (c1 - c0) * fb

    return out.reshape(rgb_values.shape)


def apply_input_color_pipeline(
    image: np.ndarray,
    input_lut: Optional[CubeLUT],
    lut_output_color_space: str,
) -> np.ndarray:
    """Apply optional LUT-based input conversion before undistortion."""

    if input_lut is None:
        return image

    if image.ndim < 3 or image.shape[2] < 3:
        raise ValueError(
            "LUT-based input conversion requires at least 3-channel "
            "RGB image input"
        )

    original_dtype = image.dtype
    bgr = image[..., :3]
    rgb = bgr[..., ::-1]

    rgb_float = image_to_float01(rgb)
    lut_float = apply_cube_lut_trilinear(rgb_float, input_lut)

    output_space = normalize_lut_output_color_space(
        lut_output_color_space
    )
    if output_space == "srgb":
        encoded_float = rec709_to_srgb(lut_float)
    elif output_space == "passthrough":
        encoded_float = np.clip(lut_float, 0.0, 1.0)
    else:
        raise ValueError("Unexpected LUT output color space")

    out_rgb = float01_to_image(encoded_float, original_dtype)
    out_bgr = out_rgb[..., ::-1]

    if image.shape[2] == 3:
        return out_bgr

    out = image.copy()
    out[..., :3] = out_bgr
    return out


def load_prepared_input_image(
    image_path: pathlib.Path,
    input_lut: Optional[CubeLUT],
    lut_output_color_space: str,
) -> np.ndarray:
    """Load one image and apply the configured color pipeline."""

    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError("Failed to read image: {}".format(image_path))
    return apply_input_color_pipeline(
        image=image,
        input_lut=input_lut,
        lut_output_color_space=lut_output_color_space,
    )


def write_standard_image(path: pathlib.Path, image: np.ndarray) -> None:
    """Write a non-perspective image using OpenCV defaults."""

    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise RuntimeError("Failed to write image: {}".format(path))


def _choose_calibration_node(sensor: ET.Element) -> Optional[ET.Element]:
    """Pick adjusted calibration node first, then initial, then fallback."""

    calibrations = sensor.findall("calibration")
    if not calibrations:
        return None
    for class_name in ("adjusted", "initial"):
        for calib in calibrations:
            if calib.attrib.get("class", "").strip().lower() == class_name:
                return calib
    return calibrations[0]


def load_metashape_calibration(
    xml_path: pathlib.Path,
) -> Tuple[Dict[str, SensorCalibration], Dict[str, str]]:
    """Load sensor calibrations and camera-label to sensor mapping from XML."""

    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    sensor_map: Dict[str, SensorCalibration] = {}
    camera_to_sensor: Dict[str, str] = {}

    for sensor in root.findall(".//sensors/sensor"):
        sensor_id = sensor.attrib.get("id", "").strip()
        if not sensor_id:
            continue
        calib_node = _choose_calibration_node(sensor)
        if calib_node is None:
            continue

        model_type = (
            calib_node.attrib.get("type")
            or sensor.attrib.get("type")
            or ""
        ).strip().lower()

        res_node = calib_node.find("resolution") or sensor.find("resolution")
        if res_node is None:
            continue
        width = int(res_node.attrib.get("width", "0"))
        height = int(res_node.attrib.get("height", "0"))
        if width <= 0 or height <= 0:
            continue

        calibration = SensorCalibration(
            sensor_id=sensor_id,
            model_type=model_type,
            width=width,
            height=height,
            f=_parse_float(calib_node, "f", 0.0),
            cx=_parse_float(calib_node, "cx", 0.0),
            cy=_parse_float(calib_node, "cy", 0.0),
            k1=_parse_float(calib_node, "k1", 0.0),
            k2=_parse_float(calib_node, "k2", 0.0),
            k3=_parse_float(calib_node, "k3", 0.0),
            k4=_parse_float(calib_node, "k4", 0.0),
            p1=_parse_float(calib_node, "p1", 0.0),
            p2=_parse_float(calib_node, "p2", 0.0),
            b1=_parse_float(calib_node, "b1", 0.0),
            b2=_parse_float(calib_node, "b2", 0.0),
        )

        if calibration.f <= 0.0:
            continue
        sensor_map[sensor_id] = calibration

    for camera in root.findall(".//cameras/camera"):
        label = camera.attrib.get("label", "").strip()
        sensor_id = camera.attrib.get("sensor_id", "").strip()
        if label and sensor_id:
            camera_to_sensor[label] = sensor_id

    return sensor_map, camera_to_sensor


def gather_input_images(
    input_dir: pathlib.Path,
    ext_filter: Sequence[str],
    suffix_filter: Sequence[str],
) -> List[pathlib.Path]:
    """Collect image files in input directory with extension/suffix filters."""

    images: List[pathlib.Path] = []
    for path in sorted(input_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower().lstrip(".") not in ext_filter:
            continue
        if suffix_filter and not any(
                path.stem.endswith(suffix) for suffix in suffix_filter):
            continue
        images.append(path)
    return images


def resolve_sensor_id_for_file(
    image_path: pathlib.Path,
    camera_to_sensor: Dict[str, str],
    sensor_map: Dict[str, SensorCalibration],
    sensor_id_x: Optional[str],
    sensor_id_y: Optional[str],
    x_suffix: str = "_X",
    y_suffix: str = "_Y",
) -> Optional[str]:
    """Resolve sensor ID using camera label mapping and suffix fallback."""

    stem = image_path.stem
    if stem in camera_to_sensor:
        sensor_id = camera_to_sensor[stem]
        if sensor_id in sensor_map:
            return sensor_id

    if sensor_id_x and stem.endswith(x_suffix) and sensor_id_x in sensor_map:
        return sensor_id_x
    if sensor_id_y and stem.endswith(y_suffix) and sensor_id_y in sensor_map:
        return sensor_id_y

    if len(sensor_map) == 1:
        return next(iter(sensor_map.keys()))

    return None


def split_stem_suffix(stem: str, x_suffix: str,
                      y_suffix: str) -> Tuple[str, str]:
    """Split stem into base + lens token."""

    if stem.endswith(x_suffix):
        return stem[: -len(x_suffix)], "X"
    if stem.endswith(y_suffix):
        return stem[: -len(y_suffix)], "Y"
    return stem, ""


def build_pair_records(
    image_paths: Sequence[pathlib.Path],
    x_suffix: str,
    y_suffix: str,
) -> List[Tuple[str, pathlib.Path, pathlib.Path]]:
    """Build X/Y pair records from filename suffixes."""

    table: Dict[str, Dict[str, pathlib.Path]] = {}
    for path in image_paths:
        base, key = split_stem_suffix(
            path.stem, x_suffix=x_suffix, y_suffix=y_suffix)
        if key not in {"X", "Y"}:
            continue
        entry = table.setdefault(base, {})
        entry[key] = path

    pairs: List[Tuple[str, pathlib.Path, pathlib.Path]] = []
    for base in sorted(table.keys()):
        item = table[base]
        x_path = item.get("X")
        y_path = item.get("Y")
        if x_path is None or y_path is None:
            continue
        pairs.append((base, x_path, y_path))
    return pairs


def build_metadata_only_resolved_pairs(
    camera_to_sensor: Dict[str, str],
    sensor_map: Dict[str, SensorCalibration],
    x_suffix: str,
    y_suffix: str,
    available_labels: Optional[Set[str]] = None,
) -> List[Tuple[int, str, pathlib.Path, pathlib.Path, str, str]]:
    """Build synthetic X/Y pair records from XML camera labels only."""

    table: Dict[str, Dict[str, Tuple[str, str]]] = {}
    for label, sensor_id in sorted(camera_to_sensor.items()):
        if sensor_id not in sensor_map:
            continue
        if available_labels is not None and label not in available_labels:
            continue
        base, key = split_stem_suffix(
            label,
            x_suffix=x_suffix,
            y_suffix=y_suffix,
        )
        if key not in {"X", "Y"}:
            continue
        entry = table.setdefault(base, {})
        entry[key] = (label, sensor_id)

    resolved_pairs: List[
        Tuple[int, str, pathlib.Path, pathlib.Path, str, str]
    ] = []
    for pair_idx, base in enumerate(sorted(table.keys()), start=1):
        item = table[base]
        x_item = item.get("X")
        y_item = item.get("Y")
        if x_item is None or y_item is None:
            continue
        x_label, sensor_id_x = x_item
        y_label, sensor_id_y = y_item
        resolved_pairs.append(
            (
                pair_idx,
                base,
                pathlib.Path(x_label + ".jpg"),
                pathlib.Path(y_label + ".jpg"),
                sensor_id_x,
                sensor_id_y,
            )
        )
    return resolved_pairs


def build_camera_transform_map(
    xml_path: pathlib.Path,
) -> Dict[str, List[List[float]]]:
    """Load Metashape camera transforms keyed by label."""

    records = msxml_converter.load_metashape_cameras(xml_path)
    return {str(label): mat for _cam_id, label, mat in records}


def _apply_brown_distortion(
    x: np.ndarray,
    y: np.ndarray,
    calib: SensorCalibration,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply Brown distortion in normalized image coordinates."""

    r2 = (x * x) + (y * y)
    r4 = r2 * r2
    r6 = r4 * r2
    r8 = r4 * r4

    radial = (
        1.0
        + (calib.k1 * r2)
        + (calib.k2 * r4)
        + (calib.k3 * r6)
        + (calib.k4 * r8)
    )

    xy = x * y
    x_dist = x * radial
    y_dist = y * radial

    if calib.p1 != 0.0 or calib.p2 != 0.0:
        x_dist = x_dist + (calib.p1 * (r2 + (2.0 * x * x))
                           ) + (2.0 * calib.p2 * xy)
        y_dist = y_dist + (calib.p2 * (r2 + (2.0 * y * y))
                           ) + (2.0 * calib.p1 * xy)

    return x_dist, y_dist, r2


def _remap_for_zoom(
    calib: SensorCalibration,
    dst_x: np.ndarray,
    dst_y: np.ndarray,
    center_x: float,
    center_y: float,
    zoom: float,
    lens_fov_deg: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute source coordinates and validity mask for a given output zoom."""

    denom_y = calib.f
    denom_x = calib.f + calib.b1
    if abs(denom_y) < 1e-12 or abs(denom_x) < 1e-12:
        raise ValueError(
            "Invalid focal/b1 configuration caused division by zero.")

    y0 = (dst_y - center_y) / denom_y
    x0 = (dst_x - center_x - (y0 * calib.b2)) / denom_x

    x = x0 / zoom
    y = y0 / zoom

    x_dist, y_dist, r2 = _apply_brown_distortion(x=x, y=y, calib=calib)

    src_x = center_x + (x_dist * calib.f) + (x_dist *
                                             calib.b1) + (y_dist * calib.b2)
    src_y = center_y + (y_dist * calib.f)

    # Metashape fisheye model uses theta from optical axis.
    # Restrict valid rays to the physical lens FOV to avoid fold-over
    # artifacts.
    r = np.sqrt(np.maximum(r2, 0.0))
    theta = 2.0 * np.arcsin(np.clip(r * 0.5, 0.0, 1.0))
    theta_max = math.radians(max(1.0, min(360.0, float(lens_fov_deg))) * 0.5)
    valid_model = theta <= theta_max
    valid_bounds = (
        (src_x >= 0.0)
        & (src_x <= (calib.width - 1))
        & (src_y >= 0.0)
        & (src_y <= (calib.height - 1))
    )
    valid = valid_model & valid_bounds
    return src_x, src_y, valid, valid_model


def estimate_auto_undistort_zoom(
    calib: SensorCalibration,
    sample_count: int = 192,
    lens_fov_deg: float = 190.0,
) -> float:
    """Estimate minimum zoom avoiding out-of-bounds samples in valid model."""

    width = int(calib.width)
    height = int(calib.height)
    center_x = (width * 0.5) + calib.cx
    center_y = (height * 0.5) + calib.cy

    steps = max(32, int(sample_count))
    grid_x = np.linspace(0.0, float(width - 1), steps, dtype=np.float32)
    grid_y = np.linspace(0.0, float(height - 1), steps, dtype=np.float32)
    dst_x, dst_y = np.meshgrid(grid_x, grid_y)

    def overflow(zoom: float) -> float:
        src_x, src_y, _valid, valid_model = _remap_for_zoom(
            calib=calib,
            dst_x=dst_x,
            dst_y=dst_y,
            center_x=center_x,
            center_y=center_y,
            zoom=zoom,
            lens_fov_deg=lens_fov_deg,
        )
        if not np.any(valid_model):
            return 0.0
        sx = src_x[valid_model]
        sy = src_y[valid_model]
        over_left = np.maximum(0.0, -sx)
        over_right = np.maximum(0.0, sx - (width - 1))
        over_top = np.maximum(0.0, -sy)
        over_bottom = np.maximum(0.0, sy - (height - 1))
        return float(
            max(
                float(np.max(over_left)),
                float(np.max(over_right)),
                float(np.max(over_top)),
                float(np.max(over_bottom)),
            )
        )

    base_overflow = overflow(1.0)
    if base_overflow <= 0.0:
        return 1.0

    low = 1.0
    high = 1.0
    for _ in range(20):
        high *= 1.2
        if overflow(high) <= 0.0:
            break
    if overflow(high) > 0.0:
        return high

    for _ in range(20):
        mid = (low + high) * 0.5
        if overflow(mid) <= 0.0:
            high = mid
        else:
            low = mid
    return high


def build_remap_cache(
    calib: SensorCalibration,
    undistort_zoom: Optional[float],
    lens_fov_deg: float,
) -> RemapCache:
    """Build undistortion remap for equisolid fisheye calibration."""

    if calib.model_type not in SUPPORTED_MODELS:
        raise ValueError(
            "Unsupported sensor model '{}' (supported: {}).".format(
                calib.model_type,
                ", ".join(sorted(SUPPORTED_MODELS)),
            )
        )

    width = int(calib.width)
    height = int(calib.height)

    grid_x = np.arange(width, dtype=np.float32)
    grid_y = np.arange(height, dtype=np.float32)
    dst_x, dst_y = np.meshgrid(grid_x, grid_y)

    center_x = (width * 0.5) + calib.cx
    center_y = (height * 0.5) + calib.cy

    zoom = (
        float(undistort_zoom)
        if undistort_zoom is not None
        else estimate_auto_undistort_zoom(
            calib,
            lens_fov_deg=float(lens_fov_deg),
        )
    )
    zoom = max(1e-6, zoom)

    src_x, src_y, valid, _valid_model = _remap_for_zoom(
        calib=calib,
        dst_x=dst_x,
        dst_y=dst_y,
        center_x=center_x,
        center_y=center_y,
        zoom=zoom,
        lens_fov_deg=float(lens_fov_deg),
    )

    return RemapCache(
        map_x=src_x.astype(np.float32),
        map_y=src_y.astype(np.float32),
        valid_mask=valid,
        undistort_zoom=zoom,
    )


def undistort_prepared_image(
    image: np.ndarray,
    image_name: str,
    out_path: pathlib.Path,
    remap: RemapCache,
    interpolation: int,
    expected_size: Tuple[int, int],
    mask_outside_model: bool,
    mask_value: int,
) -> None:
    """Apply precomputed remap to one prepared image and write output."""

    in_height, in_width = image.shape[:2]
    exp_width, exp_height = expected_size
    if in_width != exp_width or in_height != exp_height:
        raise RuntimeError(
            "Resolution mismatch for {}: got {}x{}, expected {}x{}".format(
                image_name,
                in_width,
                in_height,
                exp_width,
                exp_height,
            )
        )

    corrected = cv2.remap(
        image,
        remap.map_x,
        remap.map_y,
        interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float(mask_value),
    )

    if mask_outside_model:
        mask = remap.valid_mask
        if corrected.ndim == 2:
            corrected[~mask] = mask_value
        else:
            corrected[~mask, :] = mask_value

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_path), corrected)
    if not ok:
        raise RuntimeError("Failed to write image: {}".format(out_path))


def parse_sensor_dimensions(sensor_mm: str) -> Tuple[float, float]:
    """Parse sensor size string and return width/height in mm."""

    text = str(sensor_mm or "").strip().replace(
        "x", " ").replace("X", " ").replace(",", " ")
    tokens = [tok for tok in text.split() if tok]
    values: List[float] = []
    for token in tokens:
        try:
            values.append(float(token))
        except ValueError:
            continue
    if not values:
        raise ValueError(
            "Invalid --perspective-sensor-mm: '{}'".format(sensor_mm))
    width_mm = float(values[0])
    height_mm = float(values[1] if len(values) > 1 else values[0])
    if width_mm <= 0.0 or height_mm <= 0.0:
        raise ValueError(
            "Sensor dimensions must be positive: '{}'".format(sensor_mm))
    return width_mm, height_mm


def compute_view_fov_deg(
        focal_mm: float, sensor_mm: str) -> Tuple[float, float]:
    """Compute HFOV/VFOV from focal length and sensor dimensions."""

    f = float(focal_mm)
    if f <= 0.0:
        raise ValueError("--perspective-focal-mm must be > 0")
    sensor_w_mm, sensor_h_mm = parse_sensor_dimensions(sensor_mm)
    hfov_deg = math.degrees(2.0 * math.atan(sensor_w_mm / (2.0 * f)))
    vfov_deg = math.degrees(2.0 * math.atan(sensor_h_mm / (2.0 * f)))
    hfov_deg = max(1.0, min(179.9, hfov_deg))
    vfov_deg = max(1.0, min(179.9, vfov_deg))
    return hfov_deg, vfov_deg


def build_sfm10_specs(
    output_size: int,
    focal_mm: float,
    sensor_mm: str,
    yaw_delta_deg: float,
    pitch_delta_deg: float,
) -> List[Dict[str, float]]:
    """Build 10-view SFM layout specs around dual-fisheye front/back axes."""

    size = int(output_size)
    if size <= 0:
        raise ValueError("--perspective-size must be > 0")

    yaw_delta = float(yaw_delta_deg)
    pitch_delta = float(pitch_delta_deg)
    if yaw_delta <= 0.0 or yaw_delta >= 180.0:
        raise ValueError("--perspective-yaw-delta-deg must be in (0, 180)")
    if pitch_delta <= 0.0 or pitch_delta >= 89.9:
        raise ValueError("--perspective-pitch-delta-deg must be in (0, 89.9)")

    hfov_deg, vfov_deg = compute_view_fov_deg(
        focal_mm=focal_mm, sensor_mm=sensor_mm)

    layout = [
        ("A", 0.0, 0.0),
        ("A_U", 0.0, +pitch_delta),
        ("A_D", 0.0, -pitch_delta),
        ("B", +yaw_delta, 0.0),
        ("E", 180.0 - yaw_delta, 0.0),
        ("F", 180.0, 0.0),
        ("F_U", 180.0, +pitch_delta),
        ("F_D", 180.0, -pitch_delta),
        ("G", 180.0 + yaw_delta, 0.0),
        ("J", 360.0 - yaw_delta, 0.0),
    ]

    specs: List[Dict[str, float]] = []
    for view_id, yaw_deg, pitch_deg in layout:
        specs.append(
            {
                "view_id": str(view_id),
                "yaw_deg": float(yaw_deg),
                "pitch_deg": float(pitch_deg),
                "hfov_deg": float(hfov_deg),
                "vfov_deg": float(vfov_deg),
                "width": size,
                "height": size,
            }
        )
    return specs


def rotate_view_vectors(
        vectors: np.ndarray,
        yaw_deg: float,
        pitch_deg: float) -> np.ndarray:
    """Rotate vectors by pitch then yaw (same convention as GUI preview)."""

    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)

    cos_p = math.cos(pitch)
    sin_p = math.sin(pitch)
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)

    x = vectors[..., 0]
    y = vectors[..., 1]
    z = vectors[..., 2]

    y1 = (cos_p * y) + (sin_p * z)
    z1 = (-sin_p * y) + (cos_p * z)
    x1 = x

    x2 = (cos_y * x1) + (sin_y * z1)
    z2 = (-sin_y * x1) + (cos_y * z1)

    out = np.empty_like(vectors)
    out[..., 0] = x2
    out[..., 1] = y1
    out[..., 2] = z2
    return out


def wrap_angle_deg(angle_deg: float) -> float:
    """Wrap angle to [-180, 180) degrees."""

    return ((float(angle_deg) + 180.0) % 360.0) - 180.0


def build_perspective_pose_frames(
    extrinsics_xml_path: pathlib.Path,
    resolved_pairs: Sequence[
        Tuple[int, str, pathlib.Path, pathlib.Path, str, str]
    ],
    successful_bases: Optional[Set[str]],
    perspective_specs: Sequence[Dict[str, object]],
    perspective_map_cache: Dict[Tuple[str, str], Dict[str, Dict[str, object]]],
    perspective_out_ext: str,
    lens_x_yaw_deg: float,
    lens_y_yaw_deg: float,
) -> List[Dict[str, object]]:
    """Build perspective frame poses from aligned dual-fisheye cameras."""

    camera_transform_map = build_camera_transform_map(extrinsics_xml_path)
    frames: List[Dict[str, object]] = []
    missing_labels: List[str] = []

    for _pair_idx, base_stem, x_path, y_path, sensor_id_x, sensor_id_y in resolved_pairs:
        if successful_bases is not None and base_stem not in successful_bases:
            continue
        x_label = x_path.stem
        y_label = y_path.stem
        x_c2w_cv = camera_transform_map.get(x_label)
        y_c2w_cv = camera_transform_map.get(y_label)
        if x_c2w_cv is None:
            missing_labels.append(x_label)
            continue
        if y_c2w_cv is None:
            missing_labels.append(y_label)
            continue

        spec_maps = perspective_map_cache.get((sensor_id_x, sensor_id_y))
        if not spec_maps:
            raise ValueError(
                "Perspective remap cache missing for sensor pair {} / {}".format(
                    sensor_id_x,
                    sensor_id_y,
                )
            )

        for spec in perspective_specs:
            view_id = str(spec["view_id"])
            mapping = spec_maps.get(view_id)
            if mapping is None:
                raise ValueError(
                    "Perspective view '{}' missing from remap cache.".format(
                        view_id
                    )
                )
            lens_key = str(mapping["lens_key"]).upper()
            if lens_key == "X":
                base_c2w_cv = x_c2w_cv
                source_label = x_label
                lens_yaw_deg = float(lens_x_yaw_deg)
            elif lens_key == "Y":
                base_c2w_cv = y_c2w_cv
                source_label = y_label
                lens_yaw_deg = float(lens_y_yaw_deg)
            else:
                raise ValueError(
                    "Unsupported lens key '{}' for view '{}'.".format(
                        lens_key,
                        view_id,
                    )
                )

            yaw_rel = wrap_angle_deg(float(spec["yaw_deg"]) - lens_yaw_deg)
            pitch_deg = float(spec["pitch_deg"])
            base_gl = msxml_converter.mat4_mul(
                base_c2w_cv,
                msxml_converter.CV_TO_GL,
            )
            rel_rot = msxml_converter.yaw_pitch_to_rot_gl(
                yaw_rel,
                pitch_deg,
            )
            c2w_gl = msxml_converter.mat4_mul(
                base_gl,
                msxml_converter.mat3_to_mat4(rel_rot),
            )
            c2w_cv = msxml_converter.mat4_mul(
                c2w_gl,
                msxml_converter.CV_TO_GL,
            )
            frames.append(
                {
                    "file_path": "{}_{}{}".format(
                        base_stem,
                        view_id,
                        perspective_out_ext,
                    ),
                    "c2w_gl": c2w_gl,
                    "c2w_cv": c2w_cv,
                    "source_name": base_stem,
                    "source_label": source_label,
                    "view_id": view_id,
                    "lens_key": lens_key,
                    "yaw_rel_deg": yaw_rel,
                    "pitch_deg": pitch_deg,
                }
            )

    if missing_labels:
        missing_sorted = sorted(set(missing_labels))
        preview = ", ".join(missing_sorted[:8])
        if len(missing_sorted) > 8:
            preview += ", ..."
        raise ValueError(
            "Missing camera transforms in extrinsics XML: {}".format(preview)
        )
    if not frames:
        raise ValueError("No perspective pose frames could be generated.")
    return frames


def build_colmap_model_from_pose_frames(
    frames: Sequence[Dict[str, object]],
    perspective_size: int,
    perspective_focal_mm: float,
    perspective_sensor_mm: str,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Build a single-camera COLMAP model from perspective pose frames."""

    width = int(perspective_size)
    height = int(perspective_size)
    sensor_w_mm, sensor_h_mm = parse_sensor_dimensions(perspective_sensor_mm)
    fx, fy = camera_converter.focal_mm_to_pixels(
        perspective_focal_mm,
        width,
        height,
        sensor_w_mm,
        sensor_h_mm,
    )
    cx = width * 0.5
    cy = height * 0.5
    cameras = [
        {
            "camera_id": 1,
            "model": "PINHOLE",
            "width": width,
            "height": height,
            "params": [fx, fy, cx, cy],
        }
    ]
    images: List[Dict[str, object]] = []
    for image_id, frame in enumerate(frames, start=1):
        r_wc, t_wc = msxml_converter.compute_colmap_pose(frame, 0.0)
        qw, qx, qy, qz = camera_converter.rotmat_to_quat_wxyz(r_wc)
        images.append(
            {
                "image_id": image_id,
                "qw": qw,
                "qx": qx,
                "qy": qy,
                "qz": qz,
                "tx": t_wc[0],
                "ty": t_wc[1],
                "tz": t_wc[2],
                "camera_id": 1,
                "name": str(frame["file_path"]),
                "points2d_line": "",
            }
        )
    return cameras, images


def build_colmap_points_from_metashape_ply(
    ply_path: pathlib.Path,
) -> List[Dict[str, object]]:
    """Convert Metashape PLY points into the COLMAP-like export space."""

    identity_world = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    return msxml_converter.build_points_outputs(
        pathlib.Path(ply_path),
        pathlib.Path(ply_path).parent,
        identity_world,
        msxml_converter.POINTCLOUD_PLY_X_DEG,
        1.0,
        write_transforms_ply=False,
    )


def get_perspective_images_dir(root_dir: pathlib.Path) -> pathlib.Path:
    """Return perspective image directory under the COLMAP-style root."""

    return pathlib.Path(root_dir) / "Images"


def get_perspective_masks_dir(root_dir: pathlib.Path) -> pathlib.Path:
    """Return perspective mask directory under the COLMAP-style root."""

    return pathlib.Path(root_dir) / "Masks"


def get_perspective_sparse_dir(root_dir: pathlib.Path) -> pathlib.Path:
    """Return COLMAP sparse text directory under the COLMAP-style root."""

    return pathlib.Path(root_dir) / "Sparse" / "0"


def load_mask_image(mask_path: pathlib.Path) -> np.ndarray:
    """Load one mask image without color transforms."""

    image = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError("Failed to read mask image: {}".format(mask_path))
    return image


def collect_mask_pair_paths(
    mask_dir: pathlib.Path,
    resolved_pairs: Sequence[
        Tuple[int, str, pathlib.Path, pathlib.Path, str, str]
    ],
) -> Dict[str, Tuple[pathlib.Path, pathlib.Path]]:
    """Match X/Y mask files to each resolved pair by exact file name."""

    name_map: Dict[str, pathlib.Path] = {}
    for path in sorted(mask_dir.iterdir()):
        if path.is_file():
            name_map[path.name] = path

    matched: Dict[str, Tuple[pathlib.Path, pathlib.Path]] = {}
    missing: List[str] = []
    for _pair_idx, base_stem, x_path, y_path, _sx, _sy in resolved_pairs:
        x_mask = name_map.get(x_path.name)
        y_mask = name_map.get(y_path.name)
        if x_mask is None:
            missing.append(x_path.name)
        if y_mask is None:
            missing.append(y_path.name)
        if x_mask is None or y_mask is None:
            continue
        matched[base_stem] = (x_mask, y_mask)

    if missing:
        preview = ", ".join(sorted(set(missing))[:8])
        if len(set(missing)) > 8:
            preview += ", ..."
        raise ValueError(
            "Missing mask images in {}: {}".format(mask_dir, preview)
        )
    return matched


def export_perspective_camera_metadata(
    args: argparse.Namespace,
    resolved_pairs: Sequence[
        Tuple[int, str, pathlib.Path, pathlib.Path, str, str]
    ],
    successful_bases: Set[str],
    perspective_specs: Sequence[Dict[str, object]],
    perspective_map_cache: Dict[Tuple[str, str], Dict[str, Dict[str, object]]],
    perspective_out_dir: pathlib.Path,
    perspective_out_ext: str,
    dry_run: bool,
) -> None:
    """Export perspective camera XML / COLMAP from dual-fisheye alignment."""

    extrinsics_xml_value = str(
        getattr(args, "camera_extrinsics_xml", "") or ""
    ).strip()
    if not extrinsics_xml_value:
        return
    if bool(args.no_perspective) and not bool(
        getattr(args, "metadata_only", False)
    ):
        raise ValueError(
            "--camera-extrinsics-xml requires perspective output to be enabled."
        )
    extrinsics_xml_path = pathlib.Path(extrinsics_xml_value).expanduser().resolve()
    if not extrinsics_xml_path.exists() or not extrinsics_xml_path.is_file():
        raise ValueError(
            "Camera extrinsics XML not found: {}".format(extrinsics_xml_path)
        )

    frames = build_perspective_pose_frames(
        extrinsics_xml_path=extrinsics_xml_path,
        resolved_pairs=resolved_pairs,
        successful_bases=successful_bases,
        perspective_specs=perspective_specs,
        perspective_map_cache=perspective_map_cache,
        perspective_out_ext=perspective_out_ext,
        lens_x_yaw_deg=float(args.lens_x_yaw_deg),
        lens_y_yaw_deg=float(args.lens_y_yaw_deg),
    )
    cameras, images = build_colmap_model_from_pose_frames(
        frames=frames,
        perspective_size=int(args.perspective_size),
        perspective_focal_mm=float(args.perspective_focal_mm),
        perspective_sensor_mm=str(args.perspective_sensor_mm),
    )

    pointcloud_ply_value = str(getattr(args, "pointcloud_ply", "") or "").strip()
    points: List[Dict[str, object]] = []
    if pointcloud_ply_value:
        pointcloud_ply_path = pathlib.Path(pointcloud_ply_value).expanduser().resolve()
        if not pointcloud_ply_path.exists() or not pointcloud_ply_path.is_file():
            raise ValueError(
                "Point cloud PLY not found: {}".format(pointcloud_ply_path)
            )
        points = build_colmap_points_from_metashape_ply(pointcloud_ply_path)

    out_images = get_perspective_images_dir(perspective_out_dir)
    out_masks = get_perspective_masks_dir(perspective_out_dir)
    out_xml = perspective_out_dir / str(args.perspective_metashape_xml_name)
    out_colmap = get_perspective_sparse_dir(perspective_out_dir)

    if dry_run:
        print(
            "[DRY][META] frames={} images={} xml={} colmap={} masks={} points={}".format(
                len(frames),
                out_images,
                out_xml,
                out_colmap,
                out_masks,
                len(points),
            )
        )
        return

    camera_converter.export_metashape_perspective_xml(out_xml, cameras, images)
    camera_converter.write_colmap_text_model(out_colmap, cameras, images, points)
    print("[OK] Perspective images root: {}".format(out_images))
    print("[OK] Perspective Metashape XML: {}".format(out_xml))
    print(
        "[OK] Perspective COLMAP text: {} (images={}, points={})".format(
            out_colmap,
            len(images),
            len(points),
        )
    )
    print("[OK] Perspective masks root: {}".format(out_masks))


def build_perspective_map_for_lens(
    calib: SensorCalibration,
    yaw_deg: float,
    pitch_deg: float,
    hfov_deg: float,
    vfov_deg: float,
    out_w: int,
    out_h: int,
    lens_fov_deg: float,
    undistort_zoom: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build remap from one equisolid fisheye lens to one perspective view."""

    xs = (
        ((np.arange(out_w, dtype=np.float32) + 0.5) / float(out_w)) * 2.0
        - 1.0
    )
    ys = (
        ((np.arange(out_h, dtype=np.float32) + 0.5) / float(out_h)) * 2.0
        - 1.0
    )
    uu, vv = np.meshgrid(xs, ys)

    hfov_rad = math.radians(max(1e-3, min(179.9, hfov_deg)))
    vfov_rad = math.radians(max(1e-3, min(179.9, vfov_deg)))

    rays = np.empty((out_h, out_w, 3), dtype=np.float32)
    rays[..., 0] = np.tan(hfov_rad * 0.5) * uu
    rays[..., 1] = np.tan(vfov_rad * 0.5) * (-vv)
    rays[..., 2] = 1.0

    norms = np.linalg.norm(rays, axis=2, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    rays = rays / norms
    rays = rotate_view_vectors(rays, yaw_deg=yaw_deg, pitch_deg=pitch_deg)

    rx = rays[..., 0]
    ry = rays[..., 1]
    rz = rays[..., 2]

    rz_clamped = np.clip(rz, -1.0, 1.0)
    theta = np.arccos(rz_clamped)
    theta_max = math.radians(max(1.0, min(360.0, lens_fov_deg)) * 0.5)

    rho = np.sqrt((rx * rx) + (ry * ry))
    scale = np.zeros_like(rho, dtype=np.float32)
    nz = rho > 1e-12
    scale[nz] = (2.0 * np.sin(theta[nz] * 0.5) / rho[nz]).astype(np.float32)
    x_n = rx * scale
    # Image Y grows downward; use -ry to keep perspective outputs upright.
    y_n = -ry * scale

    center_x = (calib.width * 0.5) + calib.cx
    center_y = (calib.height * 0.5) + calib.cy

    zoom = max(1e-6, float(undistort_zoom))
    focal_y = calib.f * zoom
    focal_x = (calib.f + calib.b1) * zoom
    skew = calib.b2 * zoom

    map_x = center_x + (x_n * focal_x) + (y_n * skew)
    map_y = center_y + (y_n * focal_y)

    valid = theta <= theta_max
    valid = valid & (map_x >= 0.0) & (map_x <= (calib.width - 1))
    valid = valid & (map_y >= 0.0) & (map_y <= (calib.height - 1))

    return map_x.astype(np.float32), map_y.astype(np.float32), valid


def build_direct_perspective_map_for_lens(
    calib: SensorCalibration,
    yaw_deg: float,
    pitch_deg: float,
    hfov_deg: float,
    vfov_deg: float,
    out_w: int,
    out_h: int,
    lens_fov_deg: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build remap from one equisolid fisheye lens directly to perspective."""

    xs = (
        ((np.arange(out_w, dtype=np.float32) + 0.5) / float(out_w)) * 2.0
        - 1.0
    )
    ys = (
        ((np.arange(out_h, dtype=np.float32) + 0.5) / float(out_h)) * 2.0
        - 1.0
    )
    uu, vv = np.meshgrid(xs, ys)

    hfov_rad = math.radians(max(1e-3, min(179.9, hfov_deg)))
    vfov_rad = math.radians(max(1e-3, min(179.9, vfov_deg)))

    rays = np.empty((out_h, out_w, 3), dtype=np.float32)
    rays[..., 0] = np.tan(hfov_rad * 0.5) * uu
    rays[..., 1] = np.tan(vfov_rad * 0.5) * (-vv)
    rays[..., 2] = 1.0

    norms = np.linalg.norm(rays, axis=2, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    rays = rays / norms
    rays = rotate_view_vectors(rays, yaw_deg=yaw_deg, pitch_deg=pitch_deg)

    rx = rays[..., 0]
    ry = rays[..., 1]
    rz = rays[..., 2]

    rz_clamped = np.clip(rz, -1.0, 1.0)
    theta = np.arccos(rz_clamped)
    theta_max = math.radians(max(1.0, min(360.0, lens_fov_deg)) * 0.5)

    rho = np.sqrt((rx * rx) + (ry * ry))
    scale = np.zeros_like(rho, dtype=np.float32)
    nz = rho > 1e-12
    scale[nz] = (2.0 * np.sin(theta[nz] * 0.5) / rho[nz]).astype(np.float32)
    x_n = rx * scale
    # Image Y grows downward; use -ry to keep perspective outputs upright.
    y_n = -ry * scale

    x_dist, y_dist, _r2 = _apply_brown_distortion(x=x_n, y=y_n, calib=calib)

    center_x = (calib.width * 0.5) + calib.cx
    center_y = (calib.height * 0.5) + calib.cy
    map_x = center_x + (x_dist * calib.f) + (x_dist * calib.b1) + (
        y_dist * calib.b2
    )
    map_y = center_y + (y_dist * calib.f)

    valid = theta <= theta_max
    valid = valid & (map_x >= 0.0) & (map_x <= (calib.width - 1))
    valid = valid & (map_y >= 0.0) & (map_y <= (calib.height - 1))

    return map_x.astype(np.float32), map_y.astype(np.float32), valid


def write_perspective_image(
        path: pathlib.Path,
        image: np.ndarray,
        jpeg_quality: int) -> None:
    """Write perspective output with optional JPEG quality."""

    ext = path.suffix.lower()
    path.parent.mkdir(parents=True, exist_ok=True)
    params: List[int] = []
    if ext in {".jpg", ".jpeg"}:
        q = int(max(1, min(100, jpeg_quality)))
        params = [int(cv2.IMWRITE_JPEG_QUALITY), q]
    ok = cv2.imwrite(str(path), image, params)
    if not ok:
        raise RuntimeError("Failed to write image: {}".format(path))


def get_system_memory_usage_ratio() -> Optional[float]:
    """Return system memory usage ratio in [0, 1], if available."""

    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(MemoryStatusEx)
    try:
        ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    except Exception:
        return None
    if not ok:
        return None
    return max(0.0, min(1.0, float(status.dwMemoryLoad) / 100.0))


def build_perspective_spec_maps(
    sensor_map: Dict[str, SensorCalibration],
    sensor_id_x: str,
    sensor_id_y: str,
    specs: Sequence[Dict[str, object]],
    lens_x_yaw_deg: float,
    lens_y_yaw_deg: float,
    lens_fov_deg: float,
) -> Dict[str, Dict[str, object]]:
    """Build direct perspective remap tables for one sensor-id pair."""

    spec_maps: Dict[str, Dict[str, object]] = {}
    for spec in specs:
        view_id = str(spec["view_id"])
        yaw_world = float(spec["yaw_deg"])
        pitch_world = float(spec["pitch_deg"])
        hfov_deg = float(spec["hfov_deg"])
        vfov_deg = float(spec["vfov_deg"])
        out_w = int(spec["width"])
        out_h = int(spec["height"])
        candidates: List[Tuple[float, float, str,
                               np.ndarray, np.ndarray, np.ndarray]] = []
        for lens_key, lens_yaw_deg, sensor_id in (
            ("X", lens_x_yaw_deg, sensor_id_x),
            ("Y", lens_y_yaw_deg, sensor_id_y),
        ):
            yaw_rel = wrap_angle_deg(yaw_world - lens_yaw_deg)
            calib = sensor_map[sensor_id]
            map_x, map_y, valid = build_direct_perspective_map_for_lens(
                calib=calib,
                yaw_deg=yaw_rel,
                pitch_deg=pitch_world,
                hfov_deg=hfov_deg,
                vfov_deg=vfov_deg,
                out_w=out_w,
                out_h=out_h,
                lens_fov_deg=lens_fov_deg,
            )
            valid_ratio = float(np.mean(valid))
            candidates.append(
                (valid_ratio, -abs(yaw_rel), lens_key, map_x, map_y, valid)
            )

        best = max(candidates, key=lambda item: (item[0], item[1]))
        spec_maps[view_id] = {
            "lens_key": best[2],
            "map_x": best[3],
            "map_y": best[4],
            "valid": best[5],
        }
    return spec_maps


def process_pair_task(
    base_stem: str,
    x_path: pathlib.Path,
    y_path: pathlib.Path,
    sensor_id_x: str,
    sensor_id_y: str,
    x_mask_path: Optional[pathlib.Path],
    y_mask_path: Optional[pathlib.Path],
    input_lut: Optional[CubeLUT],
    lut_output_color_space: str,
    save_color_corrected_output: bool,
    color_corrected_out_dir: pathlib.Path,
    write_fisheye_output: bool,
    output_dir: pathlib.Path,
    remap_cache: Dict[str, RemapCache],
    sensor_map: Dict[str, SensorCalibration],
    write_perspective_output: bool,
    perspective_out_dir: pathlib.Path,
    perspective_maps: Optional[Dict[str, Dict[str, object]]],
    perspective_specs: Sequence[Dict[str, object]],
    perspective_out_ext: str,
    perspective_mask_ext: str,
    perspective_jpeg_quality: int,
    write_perspective_masks: bool,
    interpolation: int,
    mask_outside_model: bool,
    mask_value: int,
) -> Dict[str, object]:
    """Process all requested outputs for one X/Y fisheye pair."""

    image_x = load_prepared_input_image(
        image_path=x_path,
        input_lut=input_lut,
        lut_output_color_space=lut_output_color_space,
    )
    image_y = load_prepared_input_image(
        image_path=y_path,
        input_lut=input_lut,
        lut_output_color_space=lut_output_color_space,
    )
    mask_x = load_mask_image(x_mask_path) if x_mask_path is not None else None
    mask_y = load_mask_image(y_mask_path) if y_mask_path is not None else None

    color_outputs: List[str] = []
    if save_color_corrected_output:
        color_x = color_corrected_out_dir / x_path.name
        color_y = color_corrected_out_dir / y_path.name
        write_standard_image(color_x, image_x)
        write_standard_image(color_y, image_y)
        color_outputs.extend([color_x.name, color_y.name])

    fisheye_outputs: List[str] = []
    if write_fisheye_output:
        calib_x = sensor_map[sensor_id_x]
        calib_y = sensor_map[sensor_id_y]
        out_x = output_dir / x_path.name
        out_y = output_dir / y_path.name
        undistort_prepared_image(
            image=image_x,
            image_name=x_path.name,
            out_path=out_x,
            remap=remap_cache[sensor_id_x],
            interpolation=interpolation,
            expected_size=(calib_x.width, calib_x.height),
            mask_outside_model=mask_outside_model,
            mask_value=mask_value,
        )
        undistort_prepared_image(
            image=image_y,
            image_name=y_path.name,
            out_path=out_y,
            remap=remap_cache[sensor_id_y],
            interpolation=interpolation,
            expected_size=(calib_y.width, calib_y.height),
            mask_outside_model=mask_outside_model,
            mask_value=mask_value,
        )
        fisheye_outputs.extend([out_x.name, out_y.name])

    perspective_outputs: List[str] = []
    mask_outputs: List[str] = []
    if write_perspective_output:
        if perspective_maps is None:
            raise RuntimeError("Missing perspective remap cache.")
        perspective_images_dir = get_perspective_images_dir(perspective_out_dir)
        perspective_masks_dir = get_perspective_masks_dir(perspective_out_dir)
        for spec in perspective_specs:
            view_id = str(spec["view_id"])
            m = perspective_maps[view_id]
            use_x_lens = str(m["lens_key"]) == "X"
            src = image_x if use_x_lens else image_y
            rendered = cv2.remap(
                src,
                m["map_x"],  # type: ignore[arg-type]
                m["map_y"],  # type: ignore[arg-type]
                interpolation,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=float(mask_value),
            )
            valid_mask = m["valid"]  # type: ignore[assignment]
            if mask_outside_model:
                if rendered.ndim == 2:
                    rendered[~valid_mask] = mask_value
                else:
                    rendered[~valid_mask, :] = mask_value

            out_name = "{}_{}{}".format(base_stem, view_id, perspective_out_ext)
            out_path = perspective_images_dir / out_name
            write_perspective_image(
                path=out_path,
                image=rendered,
                jpeg_quality=perspective_jpeg_quality,
            )
            perspective_outputs.append(out_name)

            if write_perspective_masks:
                src_mask = mask_x if use_x_lens else mask_y
                if src_mask is None:
                    raise RuntimeError(
                        "Mask source missing for pair '{}'.".format(base_stem)
                    )
                rendered_mask = cv2.remap(
                    src_mask,
                    m["map_x"],  # type: ignore[arg-type]
                    m["map_y"],  # type: ignore[arg-type]
                    cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                if mask_outside_model:
                    if rendered_mask.ndim == 2:
                        rendered_mask[~valid_mask] = 0
                    else:
                        rendered_mask[~valid_mask, :] = 0
                mask_name = "{}_{}{}".format(
                    base_stem,
                    view_id,
                    perspective_mask_ext,
                )
                mask_path = perspective_masks_dir / mask_name
                write_perspective_image(
                    path=mask_path,
                    image=rendered_mask,
                    jpeg_quality=perspective_jpeg_quality,
                )
                mask_outputs.append(mask_name)

    return {
        "base_stem": base_stem,
        "files_processed": 2,
        "color_outputs": color_outputs,
        "fisheye_outputs": fisheye_outputs,
        "perspective_outputs": perspective_outputs,
        "mask_outputs": mask_outputs,
    }


def main() -> None:
    """CLI entry point."""

    args = parse_arguments()
    try:
        undistort_zoom_override = parse_undistort_zoom_arg(args.undistort_zoom)
    except Exception as exc:
        print("[ERR] --undistort-zoom: {}".format(exc), file=sys.stderr)
        sys.exit(1)

    metadata_only = bool(getattr(args, "metadata_only", False))
    input_dir_value = str(getattr(args, "input_dir", "") or "").strip()
    input_path: Optional[pathlib.Path] = None
    if input_dir_value:
        input_path = pathlib.Path(input_dir_value).expanduser().resolve()
    elif not metadata_only:
        print(
            "[ERR] --input-dir is required unless --metadata-only is used.",
            file=sys.stderr,
        )
        sys.exit(1)

    legacy_input_color_profile = str(
        args.input_color_profile
    ).strip().lower()
    lut_output_color_space = str(
        args.lut_output_color_space
    ).strip().lower()
    input_lut_path: Optional[pathlib.Path] = None
    input_lut: Optional[CubeLUT] = None

    if args.input_lut:
        input_lut_path = pathlib.Path(args.input_lut).expanduser().resolve()
    elif legacy_input_color_profile == "osmo360-dlogm":
        input_lut_path = pathlib.Path(args.dlogm_lut).expanduser().resolve()
    elif legacy_input_color_profile != "native":
        print(
            "[ERR] Unsupported --input-color-profile: {}".format(
                legacy_input_color_profile
            ),
            file=sys.stderr,
        )
        sys.exit(1)

    if input_lut_path is not None:
        try:
            input_lut = load_cube_lut(input_lut_path)
        except Exception as exc:
            print(
                "[ERR] Failed to load input LUT: {}".format(exc),
                file=sys.stderr,
            )
            sys.exit(1)

    try:
        lut_output_color_space = normalize_lut_output_color_space(
            lut_output_color_space
        )
    except Exception as exc:
        print("[ERR] {}".format(exc), file=sys.stderr)
        sys.exit(1)

    camera_xml_path: Optional[pathlib.Path] = None
    camera_xml_value = str(getattr(args, "camera_xml", "") or "").strip()
    if camera_xml_value:
        camera_xml_path = pathlib.Path(camera_xml_value).expanduser().resolve()

    suffix_filter = [token.strip()
                     for token in args.suffixes.split(",") if token.strip()]
    if len(suffix_filter) < 2:
        print(
            "[ERR] --suffixes must include at least two values like '_X,_Y'.",
            file=sys.stderr)
        sys.exit(1)
    x_suffix = suffix_filter[0]
    y_suffix = suffix_filter[1]

    fisheye_dir: Optional[pathlib.Path] = None
    if input_path is not None:
        if input_path.is_file():
            print(
                "[ERR] Input must be a directory of fisheye frames, "
                "not a video file.\n"
                "Use gs360_Video2Frames.py to extract *_X/*_Y images first.",
                file=sys.stderr,
            )
            sys.exit(1)
        if not input_path.is_dir():
            print(
                "[ERR] Input path not found: {}".format(input_path),
                file=sys.stderr,
            )
            sys.exit(1)
        fisheye_dir = input_path

    write_fisheye_output = bool(args.save_fisheye_output) and not metadata_only
    save_color_corrected_output = (
        bool(args.save_color_corrected_output) and not metadata_only
    )
    write_perspective_output = (
        (not bool(args.no_perspective)) and not metadata_only
    )
    need_undistorted_stage = write_fisheye_output
    if (
        (not metadata_only)
        and
        (not write_fisheye_output)
        and (not write_perspective_output)
        and (not save_color_corrected_output)
    ):
        print(
            "[ERR] All outputs are disabled. Enable perspective, "
            "--save-fisheye-output, or --save-color-corrected-output.",
            file=sys.stderr,
        )
        sys.exit(1)

    extrinsics_xml_path: Optional[pathlib.Path] = None
    extrinsics_xml_value = str(
        getattr(args, "camera_extrinsics_xml", "") or ""
    ).strip()
    if extrinsics_xml_value:
        extrinsics_xml_path = pathlib.Path(
            extrinsics_xml_value
        ).expanduser().resolve()
        if not extrinsics_xml_path.exists() or not extrinsics_xml_path.is_file():
            print(
                "[ERR] Camera extrinsics XML not found: {}".format(
                    extrinsics_xml_path
                ),
                file=sys.stderr,
            )
            sys.exit(1)
        if not write_perspective_output and not metadata_only:
            print(
                "[ERR] --camera-extrinsics-xml requires perspective output.",
                file=sys.stderr,
            )
            sys.exit(1)

    fisheye_output_arg = args.output_dir or args.fisheye_output_dir
    if fisheye_output_arg:
        output_dir = pathlib.Path(fisheye_output_arg).expanduser().resolve()
    elif fisheye_dir is not None:
        output_dir = fisheye_dir.with_name(fisheye_dir.name + "_undistorted")
    else:
        output_dir = pathlib.Path.cwd() / "_unused_dualfisheye_undistorted"

    if args.perspective_output_dir:
        perspective_out_dir = pathlib.Path(
            args.perspective_output_dir
        ).expanduser().resolve()
    elif fisheye_dir is not None:
        perspective_out_dir = fisheye_dir.with_name(
            fisheye_dir.name + "_perspective_colmap"
        )
    elif extrinsics_xml_path is not None:
        perspective_out_dir = extrinsics_xml_path.with_name(
            extrinsics_xml_path.stem + "_perspective_colmap"
        )
    else:
        perspective_out_dir = pathlib.Path.cwd() / "perspective_colmap"

    if args.color_corrected_output_dir:
        color_corrected_out_dir = pathlib.Path(
            args.color_corrected_output_dir
        ).expanduser().resolve()
    elif fisheye_dir is not None:
        color_corrected_out_dir = fisheye_dir.with_name(
            fisheye_dir.name + "_colorcorrected"
        )
    else:
        color_corrected_out_dir = pathlib.Path.cwd() / "_unused_colorcorrected"

    pointcloud_ply_path: Optional[pathlib.Path] = None
    pointcloud_ply_value = str(getattr(args, "pointcloud_ply", "") or "").strip()
    if pointcloud_ply_value:
        pointcloud_ply_path = pathlib.Path(
            pointcloud_ply_value
        ).expanduser().resolve()
        if not pointcloud_ply_path.exists() or not pointcloud_ply_path.is_file():
            print(
                "[ERR] Point cloud PLY not found: {}".format(
                    pointcloud_ply_path
                ),
                file=sys.stderr,
            )
            sys.exit(1)
    if metadata_only:
        if extrinsics_xml_path is None:
            print(
                "[ERR] --metadata-only requires --camera-extrinsics-xml.",
                file=sys.stderr,
            )
            sys.exit(1)
        if pointcloud_ply_path is None:
            print(
                "[ERR] --metadata-only requires --pointcloud-ply.",
                file=sys.stderr,
            )
            sys.exit(1)

    calibration_xml_path: Optional[pathlib.Path] = None
    if extrinsics_xml_path is not None:
        calibration_xml_path = extrinsics_xml_path
    elif camera_xml_path is not None:
        calibration_xml_path = camera_xml_path
    if calibration_xml_path is None:
        print(
            "[ERR] Specify --camera-extrinsics-xml or --camera-xml.",
            file=sys.stderr,
        )
        sys.exit(1)
    if not calibration_xml_path.exists() or not calibration_xml_path.is_file():
        print(
            "[ERR] Calibration XML not found: {}".format(calibration_xml_path),
            file=sys.stderr,
        )
        sys.exit(1)

    mask_input_dir_path: Optional[pathlib.Path] = None
    mask_input_dir_value = str(getattr(args, "mask_input_dir", "") or "").strip()
    if mask_input_dir_value:
        mask_input_dir_path = pathlib.Path(mask_input_dir_value).expanduser().resolve()
        if not mask_input_dir_path.exists() or not mask_input_dir_path.is_dir():
            print(
                "[ERR] Mask input directory not found: {}".format(
                    mask_input_dir_path
                ),
                file=sys.stderr,
            )
            sys.exit(1)
        if not write_perspective_output and not metadata_only:
            print(
                "[ERR] --mask-input-dir requires perspective output.",
                file=sys.stderr,
            )
            sys.exit(1)

    ext_filter = [token.strip().lower().lstrip(".")
                  for token in args.ext.split(",") if token.strip()]
    if not ext_filter:
        ext_filter = [ext.lstrip(".") for ext in SUPPORTED_EXTS]

    sensor_map, camera_to_sensor = load_metashape_calibration(
        calibration_xml_path
    )
    if not sensor_map:
        print("[ERR] No usable calibration found in XML.", file=sys.stderr)
        sys.exit(1)

    unsupported = [s.sensor_id for s in sensor_map.values(
    ) if s.model_type not in SUPPORTED_MODELS]
    if unsupported:
        print(
            "[ERR] Unsupported model types in sensors: {}".format(
                ", ".join(sorted(unsupported))
            ),
            file=sys.stderr,
        )
        sys.exit(1)

    images: List[pathlib.Path] = []
    pairs: List[Tuple[str, pathlib.Path, pathlib.Path]] = []
    if fisheye_dir is not None:
        images = gather_input_images(fisheye_dir, ext_filter, suffix_filter)
        if not images:
            print(
                "[ERR] No target images found in {}".format(fisheye_dir),
                file=sys.stderr,
            )
            sys.exit(1)

        pairs = build_pair_records(
            images,
            x_suffix=x_suffix,
            y_suffix=y_suffix,
        )
        if not pairs:
            print(
                "[ERR] No valid X/Y fisheye pairs found in {}".format(
                    fisheye_dir
                ),
                file=sys.stderr,
            )
            sys.exit(1)
    if args.limit:
        print(
            "[WARN] --limit is deprecated and ignored. Processing all pairs."
        )
    if args.report_json:
        print("[WARN] --report-json is deprecated and ignored.")
    pair_images: List[pathlib.Path] = []
    for _base, x_path, y_path in pairs:
        pair_images.extend([x_path, y_path])

    interpolation = INTERPOLATION_MAP[args.interpolation]
    mask_value = int(max(0, min(255, args.mask_value)))
    workers = int(args.workers)
    if workers < 1:
        print("[ERR] --workers must be >= 1.", file=sys.stderr)
        sys.exit(1)
    memory_reduce_threshold = float(args.memory_throttle_percent) / 100.0
    if memory_reduce_threshold <= 0.0 or memory_reduce_threshold > 1.0:
        print(
            "[ERR] --memory-throttle-percent must be > 0 and <= 100.",
            file=sys.stderr,
        )
        sys.exit(1)

    remap_cache: Dict[str, RemapCache] = {}
    processed = 0
    skipped = 0
    errors: List[str] = []
    color_output_count = 0
    perspective_output_count = 0
    perspective_mask_count = 0
    successful_pair_bases: Set[str] = set()

    if fisheye_dir is not None:
        print("[INFO] input:  {}".format(fisheye_dir))
    else:
        print("[INFO] input:  disabled (--metadata-only)")
    if write_fisheye_output:
        print("[INFO] fisheye output: {}".format(output_dir))
    else:
        print("[INFO] fisheye output: disabled")
    if write_perspective_output or metadata_only:
        print("[INFO] perspective output: {}".format(perspective_out_dir))
        print(
            "[INFO] perspective xml: {}".format(
                perspective_out_dir / args.perspective_metashape_xml_name
            )
        )
        print(
            "[INFO] perspective images dir: {}".format(
                get_perspective_images_dir(perspective_out_dir)
            )
        )
        print(
            "[INFO] perspective sparse dir: {}".format(
                get_perspective_sparse_dir(perspective_out_dir)
            )
        )
        print(
            "[INFO] perspective masks dir: {}".format(
                get_perspective_masks_dir(perspective_out_dir)
            )
        )
    else:
        print("[INFO] perspective output: disabled")
    if save_color_corrected_output:
        print(
            "[INFO] color-corrected output: {}".format(
                color_corrected_out_dir
            )
        )
    else:
        print("[INFO] color-corrected output: disabled")
    print("[INFO] calibration xml: {}".format(calibration_xml_path))
    print("[INFO] pairs:  {}".format(len(pairs)))
    print("[INFO] files:  {}".format(len(pair_images)))
    if extrinsics_xml_path is not None:
        print("[INFO] camera extrinsics xml: {}".format(extrinsics_xml_path))
    else:
        print("[INFO] camera extrinsics xml: disabled")
    if pointcloud_ply_path is not None:
        print("[INFO] pointcloud ply: {}".format(pointcloud_ply_path))
    else:
        print("[INFO] pointcloud ply: disabled")
    if mask_input_dir_path is not None and not metadata_only:
        print("[INFO] mask input dir: {}".format(mask_input_dir_path))
    elif mask_input_dir_path is not None:
        print("[INFO] mask input dir: ignored (--metadata-only)")
    else:
        print("[INFO] mask input dir: disabled")
    print(
        "[INFO] workers: {} (memory auto-throttle > {}%)".format(
            workers,
            "{:.1f}".format(memory_reduce_threshold * 100.0),
        )
    )
    if metadata_only:
        print("[INFO] pair worker mode: disabled (--metadata-only)")
    else:
        print("[INFO] pair worker mode: enabled")
    if input_lut_path is not None:
        print("[INFO] input LUT: {}".format(input_lut_path))
        print("[INFO] LUT output color space: {}".format(
            lut_output_color_space
        ))
    else:
        print("[INFO] input LUT: disabled")
    if write_fisheye_output:
        if undistort_zoom_override is None:
            print("[INFO] undistort zoom: auto")
        else:
            print(
                "[INFO] undistort zoom: {:.6f}".format(
                    undistort_zoom_override
                )
            )
    else:
        print("[INFO] undistort zoom: unused (direct perspective path)")

    resolved_pairs: List[
        Tuple[int, str, pathlib.Path, pathlib.Path, str, str]
    ] = []
    used_sensor_ids: Set[str] = set()
    used_sensor_pairs: Set[Tuple[str, str]] = set()
    if metadata_only:
        available_labels = None
        if extrinsics_xml_path is not None:
            available_labels = set(
                build_camera_transform_map(extrinsics_xml_path).keys()
            )
        resolved_pairs = build_metadata_only_resolved_pairs(
            camera_to_sensor=camera_to_sensor,
            sensor_map=sensor_map,
            x_suffix=x_suffix,
            y_suffix=y_suffix,
            available_labels=available_labels,
        )
        if not resolved_pairs:
            print(
                "[ERR] No valid X/Y camera label pairs found in extrinsics XML.",
                file=sys.stderr,
            )
            sys.exit(1)
        for _pair_idx, _base_stem, _x_path, _y_path, sensor_id_x, sensor_id_y in resolved_pairs:
            used_sensor_ids.update([sensor_id_x, sensor_id_y])
            used_sensor_pairs.add((sensor_id_x, sensor_id_y))
    else:
        for pair_idx, (base_stem, x_path, y_path) in enumerate(pairs, start=1):
            sensor_id_x = resolve_sensor_id_for_file(
                image_path=x_path,
                camera_to_sensor=camera_to_sensor,
                sensor_map=sensor_map,
                sensor_id_x=args.sensor_id_x,
                sensor_id_y=args.sensor_id_y,
                x_suffix=x_suffix,
                y_suffix=y_suffix,
            )
            sensor_id_y = resolve_sensor_id_for_file(
                image_path=y_path,
                camera_to_sensor=camera_to_sensor,
                sensor_map=sensor_map,
                sensor_id_x=args.sensor_id_x,
                sensor_id_y=args.sensor_id_y,
                x_suffix=x_suffix,
                y_suffix=y_suffix,
            )
            if sensor_id_x is None or sensor_id_y is None:
                skipped += 2
                message = "[SKIP] {}: sensor_id unresolved".format(base_stem)
                print(message)
                continue
            resolved_pairs.append(
                (pair_idx, base_stem, x_path, y_path, sensor_id_x, sensor_id_y)
            )
            used_sensor_ids.update([sensor_id_x, sensor_id_y])
            used_sensor_pairs.add((sensor_id_x, sensor_id_y))

    pair_mask_paths: Dict[str, Tuple[pathlib.Path, pathlib.Path]] = {}
    if mask_input_dir_path is not None and not metadata_only:
        try:
            pair_mask_paths = collect_mask_pair_paths(
                mask_input_dir_path,
                resolved_pairs,
            )
        except Exception as exc:
            print("[ERR] {}".format(exc), file=sys.stderr)
            sys.exit(1)

    if need_undistorted_stage:
        for sensor_id in sorted(used_sensor_ids):
            try:
                calib = sensor_map[sensor_id]
                remap_cache[sensor_id] = build_remap_cache(
                    calib,
                    undistort_zoom=undistort_zoom_override,
                    lens_fov_deg=float(args.lens_fov_deg),
                )
                print(
                    "[INFO] sensor {} undistort_zoom={:.6f}".format(
                        sensor_id,
                        remap_cache[sensor_id].undistort_zoom,
                    )
                )
            except Exception as exc:  # pragma: no cover
                err = "[ERR] sensor {}: remap build failed ({})".format(
                    sensor_id,
                    exc,
                )
                print(err)
                errors.append(err)
        if errors:
            sys.exit(2)

    perspective_specs: List[Dict[str, object]] = []
    perspective_map_cache: Dict[Tuple[str, str], Dict[str, Dict[str, object]]] = {}
    if write_perspective_output or metadata_only:
        perspective_specs = build_sfm10_specs(
            output_size=int(args.perspective_size),
            focal_mm=float(args.perspective_focal_mm),
            sensor_mm=str(args.perspective_sensor_mm),
            yaw_delta_deg=float(args.perspective_yaw_delta_deg),
            pitch_delta_deg=float(args.perspective_pitch_delta_deg),
        )
        if not perspective_specs:
            print(
                "[ERR] perspective view specs could not be generated.",
                file=sys.stderr,
            )
            sys.exit(1)
        for sensor_pair in sorted(used_sensor_pairs):
            try:
                perspective_map_cache[sensor_pair] = build_perspective_spec_maps(
                    sensor_map=sensor_map,
                    sensor_id_x=sensor_pair[0],
                    sensor_id_y=sensor_pair[1],
                    specs=perspective_specs,
                    lens_x_yaw_deg=float(args.lens_x_yaw_deg),
                    lens_y_yaw_deg=float(args.lens_y_yaw_deg),
                    lens_fov_deg=float(args.lens_fov_deg),
                )
            except Exception as exc:
                err = (
                    "[ERR] perspective remap build failed for sensor pair "
                    "{} / {} ({})".format(sensor_pair[0], sensor_pair[1], exc)
                )
                print(err)
                errors.append(err)
        if errors:
            sys.exit(2)

    perspective_out_ext = "." + args.perspective_ext.strip().lstrip(".").lower()
    perspective_mask_ext = (
        "." + args.perspective_mask_ext.strip().lstrip(".").lower()
    )
    perspective_jpeg_quality = int(args.perspective_jpeg_quality)

    if args.dry_run:
        dry_pair_total = max(1, len(resolved_pairs))
        for pair_idx, base_stem, x_path, y_path, sensor_id_x, sensor_id_y in resolved_pairs:
            if save_color_corrected_output:
                for image_path in (x_path, y_path):
                    color_corrected_path = color_corrected_out_dir / image_path.name
                    print(
                        "[DRY][COLOR] {:4d}/{:4d} {} -> {}".format(
                            pair_idx,
                            dry_pair_total,
                            image_path.name,
                            color_corrected_path.name,
                        )
                    )
                color_output_count += 2
            if need_undistorted_stage:
                print(
                    "[DRY] {:4d}/{:4d} {} -> {} (sensor_id={})".format(
                        pair_idx,
                        dry_pair_total,
                        x_path.name,
                        x_path.name,
                        sensor_id_x,
                    )
                )
                print(
                    "[DRY] {:4d}/{:4d} {} -> {} (sensor_id={})".format(
                        pair_idx,
                        dry_pair_total,
                        y_path.name,
                        y_path.name,
                        sensor_id_y,
                    )
                )
            if write_perspective_output:
                for spec in perspective_specs:
                    view_id = str(spec["view_id"])
                    out_name = "{}_{}{}".format(
                        base_stem, view_id, perspective_out_ext
                    )
                    print(
                        "[DRY][PERSP] {:4d}/{:4d} {}".format(
                            pair_idx,
                            dry_pair_total,
                            out_name,
                        )
                    )
                    if mask_input_dir_path is not None:
                        mask_name = "{}_{}{}".format(
                            base_stem,
                            view_id,
                            perspective_mask_ext,
                        )
                        print(
                            "[DRY][MASK ] {:4d}/{:4d} {}".format(
                                pair_idx,
                                dry_pair_total,
                                mask_name,
                            )
                        )
                perspective_output_count += len(perspective_specs)
                if mask_input_dir_path is not None:
                    perspective_mask_count += len(perspective_specs)
            if not metadata_only:
                processed += 2
            successful_pair_bases.add(base_stem)
    elif not metadata_only:
        adaptive_limit = workers
        pending = set()
        future_meta: Dict[object, Tuple[int, str]] = {}

        def consume_completed(
            done_futures: Sequence[object],
            active_limit: int,
        ) -> int:
            nonlocal processed, skipped
            nonlocal color_output_count, perspective_output_count
            nonlocal perspective_mask_count

            next_limit = active_limit
            mem_ratio = get_system_memory_usage_ratio()
            if mem_ratio is not None and mem_ratio > memory_reduce_threshold:
                reduced_limit = max(1, active_limit - 1)
                if reduced_limit != active_limit:
                    next_limit = reduced_limit
                    print(
                        "[INFO] memory usage {:.1f}% > {}%; "
                        "reducing active workers to {}".format(
                            mem_ratio * 100.0,
                            int(memory_reduce_threshold * 100.0),
                            next_limit,
                        )
                    )

            for future in done_futures:
                pair_idx, base_stem = future_meta.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    skipped += 2
                    err = "[ERR] {}: {}".format(base_stem, exc)
                    print(err)
                    errors.append(err)
                    continue

                color_names = list(result["color_outputs"])
                fisheye_names = list(result["fisheye_outputs"])
                perspective_names = list(result["perspective_outputs"])
                mask_names = list(result["mask_outputs"])
                for name in color_names:
                    print(
                        "[OK ][COLOR] {:4d}/{:4d} {} -> {}".format(
                            pair_idx,
                            len(pairs),
                            name,
                            name,
                        )
                    )
                for name in fisheye_names:
                    print(
                        "[OK ][FISH] {:4d}/{:4d} {} -> {}".format(
                            pair_idx,
                            len(pairs),
                            name,
                            name,
                        )
                    )
                if perspective_names:
                    print(
                        "[OK ][PERSP] {:4d}/{:4d} {} -> {} views".format(
                            pair_idx,
                            len(pairs),
                            base_stem,
                            len(perspective_names),
                        )
                    )
                if mask_names:
                    print(
                        "[OK ][MASK ] {:4d}/{:4d} {} -> {} masks".format(
                            pair_idx,
                            len(pairs),
                            base_stem,
                            len(mask_names),
                        )
                    )
                processed += int(result["files_processed"])
                color_output_count += len(color_names)
                perspective_output_count += len(perspective_names)
                perspective_mask_count += len(mask_names)
                successful_pair_bases.add(base_stem)
            return next_limit

        with ThreadPoolExecutor(max_workers=workers) as executor:
            for pair_idx, base_stem, x_path, y_path, sensor_id_x, sensor_id_y in resolved_pairs:
                x_mask_path, y_mask_path = pair_mask_paths.get(
                    base_stem, (None, None)
                )
                while pending and len(pending) >= adaptive_limit:
                    done, pending = wait(
                        pending,
                        return_when=FIRST_COMPLETED,
                    )
                    adaptive_limit = consume_completed(
                        list(done),
                        adaptive_limit,
                    )

                future = executor.submit(
                    process_pair_task,
                    base_stem,
                    x_path,
                    y_path,
                    sensor_id_x,
                    sensor_id_y,
                    x_mask_path,
                    y_mask_path,
                    input_lut,
                    lut_output_color_space,
                    save_color_corrected_output,
                    color_corrected_out_dir,
                    write_fisheye_output,
                    output_dir,
                    remap_cache,
                    sensor_map,
                    write_perspective_output,
                    perspective_out_dir,
                    perspective_map_cache.get((sensor_id_x, sensor_id_y)),
                    perspective_specs,
                    perspective_out_ext,
                    perspective_mask_ext,
                    perspective_jpeg_quality,
                    bool(mask_input_dir_path is not None),
                    interpolation,
                    bool(args.mask_outside_model),
                    mask_value,
                )
                pending.add(future)
                future_meta[future] = (pair_idx, base_stem)

            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                adaptive_limit = consume_completed(list(done), adaptive_limit)

    if metadata_only:
        successful_pair_bases = {
            base_stem for _pair_idx, base_stem, _x, _y, _sx, _sy in resolved_pairs
        }

    try:
        export_perspective_camera_metadata(
            args=args,
            resolved_pairs=resolved_pairs,
            successful_bases=successful_pair_bases,
            perspective_specs=perspective_specs,
            perspective_map_cache=perspective_map_cache,
            perspective_out_dir=perspective_out_dir,
            perspective_out_ext=perspective_out_ext,
            dry_run=bool(args.dry_run),
        )
    except Exception as exc:
        err = "[ERR] perspective camera metadata export failed ({})".format(
            exc
        )
        print(err, file=sys.stderr)
        errors.append(err)

    print(
        "[DONE] processed={} skipped={} total={} persp_outputs={} "
        "mask_outputs={} color_outputs={} errors={}".format(
            processed,
            skipped,
            len(pair_images),
            perspective_output_count,
            perspective_mask_count,
            color_output_count,
            len(errors),
        )
    )

    if errors:
        sys.exit(2)


if __name__ == "__main__":
    main()
