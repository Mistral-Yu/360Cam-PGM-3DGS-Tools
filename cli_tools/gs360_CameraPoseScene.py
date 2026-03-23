#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Normalize camera-pose scene inputs for GUI visualization."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Sequence, Tuple

import numpy as np

import gs360_CameraFormatConverter as camera_converter
import gs360_MS360xmlToPersCams as msxml_converter
import gs360_PlyOptimizer as pointcloud_optimizer


@dataclass
class CameraPose:
    """Normalized camera pose for scene rendering."""

    name: str
    center: np.ndarray
    rotation_cw: np.ndarray
    frustum_half_w: float
    frustum_half_h: float


@dataclass
class CameraPoseScene:
    """Combined point-cloud and camera-pose scene."""

    source_kind: str
    source_path: Path
    points_xyz: np.ndarray
    points_rgb: np.ndarray
    cameras: List[CameraPose]
    info_text: str
    normalization_log: List[str] = field(default_factory=list)


def _as_float32_xyz(xyz: np.ndarray) -> np.ndarray:
    arr = np.asarray(xyz, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("xyz must have shape (N, 3)")
    return arr


def _as_uint8_rgb(rgb: np.ndarray, count: int) -> np.ndarray:
    arr = np.asarray(rgb, dtype=np.uint8)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("rgb must have shape (N, 3)")
    if arr.shape[0] != count:
        raise ValueError("rgb row count must match xyz row count")
    return arr


def _rows_to_point_arrays(points: Sequence[dict]) -> Tuple[np.ndarray, np.ndarray]:
    if not points:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.uint8),
        )
    xyz = np.asarray(
        [[float(row["x"]), float(row["y"]), float(row["z"])] for row in points],
        dtype=np.float32,
    )
    rgb = np.asarray(
        [[int(row["r"]), int(row["g"]), int(row["b"])] for row in points],
        dtype=np.uint8,
    )
    return xyz, rgb


def _extract_colmap_intrinsics(camera_row: dict) -> Tuple[float, float, int, int]:
    model = str(camera_row.get("model", "")).upper()
    params = [float(value) for value in camera_row.get("params", [])]
    width = int(camera_row.get("width", 1))
    height = int(camera_row.get("height", 1))
    if model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"} and len(params) >= 1:
        fx = fy = float(params[0])
    elif model in {"PINHOLE", "OPENCV", "FULL_OPENCV", "OPENCV_FISHEYE"} and len(params) >= 2:
        fx = float(params[0])
        fy = float(params[1])
    else:
        fallback = float(params[0]) if params else max(width, height) * 0.5
        fx = fy = fallback
    return fx, fy, width, height


def _frustum_half_extents_from_intrinsics(
    fx: float,
    fy: float,
    width: int,
    height: int,
) -> Tuple[float, float]:
    width_px = max(int(width), 1)
    height_px = max(int(height), 1)
    fx_safe = max(abs(float(fx)), 1e-6)
    fy_safe = max(abs(float(fy)), 1e-6)
    return (
        0.5 * float(width_px) / fx_safe,
        0.5 * float(height_px) / fy_safe,
    )


def _rotation_cw_from_colmap_pose(r_wc: Sequence[Sequence[float]]) -> np.ndarray:
    return np.asarray(camera_converter.mat3_transpose(r_wc), dtype=np.float32)


def _camera_pose_from_colmap(
    name: str,
    r_wc: Sequence[Sequence[float]],
    t_wc: Sequence[float],
    frustum_half_w: float,
    frustum_half_h: float,
) -> CameraPose:
    center = np.asarray(
        camera_converter.camera_center_from_colmap_pose(r_wc, t_wc),
        dtype=np.float32,
    )
    return CameraPose(
        name=str(name),
        center=center,
        rotation_cw=_rotation_cw_from_colmap_pose(r_wc),
        frustum_half_w=float(frustum_half_w),
        frustum_half_h=float(frustum_half_h),
    )


def _camera_poses_from_colmap_model(
    cameras: Sequence[dict],
    images: Sequence[dict],
) -> List[CameraPose]:
    camera_map = {
        int(camera_row["camera_id"]): camera_row
        for camera_row in cameras
    }
    poses: List[CameraPose] = []
    for image_row in images:
        camera_id = int(image_row["camera_id"])
        camera_row = camera_map.get(camera_id)
        if camera_row is None:
            continue
        fx, fy, width, height = _extract_colmap_intrinsics(camera_row)
        half_w, half_h = _frustum_half_extents_from_intrinsics(fx, fy, width, height)
        r_wc = camera_converter.quat_wxyz_to_rotmat(
            float(image_row["qw"]),
            float(image_row["qx"]),
            float(image_row["qy"]),
            float(image_row["qz"]),
        )
        t_wc = [
            float(image_row["tx"]),
            float(image_row["ty"]),
            float(image_row["tz"]),
        ]
        poses.append(
            _camera_pose_from_colmap(
                image_row["name"],
                r_wc,
                t_wc,
                half_w,
                half_h,
            )
        )
    return poses


def _camera_pose_from_transforms_frame(
    frame: dict,
    fx: float,
    fy: float,
    width: int,
    height: int,
) -> CameraPose:
    c2w_gl = frame.get("transform_matrix")
    if c2w_gl is None:
        raise ValueError("transforms.json frame is missing transform_matrix")
    c2w_fixed = msxml_converter.apply_x_fix_gl(
        c2w_gl,
        -float(getattr(msxml_converter, "TRANSFORMS_X_FIX_DEG", 270.0)),
    )
    frame_for_pose = {"c2w_gl": c2w_fixed}
    r_wc, t_wc = msxml_converter.compute_colmap_pose(frame_for_pose, 0.0)
    half_w, half_h = _frustum_half_extents_from_intrinsics(fx, fy, width, height)
    return _camera_pose_from_colmap(
        frame.get("file_path", ""),
        r_wc,
        t_wc,
        half_w,
        half_h,
    )


def _camera_pose_from_realityscan_row(row: dict) -> CameraPose:
    r_xmp = camera_converter.hpr_to_rs_rotation(
        row["heading"],
        row["pitch"],
        row["roll"],
    )
    r_wc = camera_converter.rs_rot_to_colmap_pose_rot(r_xmp)
    center = camera_converter.rs_world_to_colmap_world(
        [row["x"], row["y"], row["alt"]]
    )
    focal_mm = max(abs(float(row.get("f", 17.0))), 1e-6)
    sensor_w = float(getattr(camera_converter, "DEFAULT_SENSOR_W_MM", 36.0))
    sensor_h = float(getattr(camera_converter, "DEFAULT_SENSOR_H_MM", 36.0))
    half_w = 0.5 * sensor_w / focal_mm
    half_h = 0.5 * sensor_h / focal_mm
    return CameraPose(
        name=str(row.get("name", "")),
        center=np.asarray(center, dtype=np.float32),
        rotation_cw=_rotation_cw_from_colmap_pose(r_wc),
        frustum_half_w=float(half_w),
        frustum_half_h=float(half_h),
    )


def _realityscan_points_to_colmap(xyz: np.ndarray) -> np.ndarray:
    if xyz.size == 0:
        return _as_float32_xyz(xyz)
    out = np.empty_like(xyz, dtype=np.float32)
    out[:, 0] = xyz[:, 0]
    out[:, 1] = -xyz[:, 2]
    out[:, 2] = xyz[:, 1]
    return out


def _transforms_points_to_colmap(xyz: np.ndarray) -> np.ndarray:
    arr = _as_float32_xyz(xyz)
    if arr.size == 0:
        return arr
    out = arr.copy()
    angle_deg = float(getattr(msxml_converter, "POINTCLOUD_PLY_X_DEG", 180.0))
    if abs(angle_deg) < 1e-6:
        return out
    rot_inv = np.asarray(msxml_converter.rot_x_deg(-angle_deg), dtype=np.float32)
    return (out @ rot_inv.T).astype(np.float32, copy=False)


def convert_xyz_for_scene_source(source_kind: str, xyz: np.ndarray) -> np.ndarray:
    """Convert scene vertices into the common COLMAP-like display space."""

    kind = str(source_kind or "").strip().lower()
    arr = _as_float32_xyz(xyz)
    if arr.size == 0:
        return arr
    if kind == "transforms":
        return _transforms_points_to_colmap(arr)
    if kind.startswith("realityscan"):
        return _realityscan_points_to_colmap(arr)
    return arr.astype(np.float32, copy=False)


def _build_import_args(
    width: Optional[int] = None,
    height: Optional[int] = None,
    image_dir: Optional[str] = None,
    xmp_dir: Optional[str] = None,
    metashape_xml: Optional[str] = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        width=width,
        height=height,
        image_dir=image_dir,
        xmp_dir=xmp_dir,
        xmp_image_ext="jpg",
        metashape_xml=metashape_xml,
        metashape_xml_image_ext="jpg",
        sensor_width_mm=float(
            getattr(camera_converter, "DEFAULT_SENSOR_W_MM", 36.0)
        ),
        sensor_height_mm=float(
            getattr(camera_converter, "DEFAULT_SENSOR_H_MM", 36.0)
        ),
        single_camera=False,
        transforms_x_fix_deg=float(
            getattr(msxml_converter, "TRANSFORMS_X_FIX_DEG", 270.0)
        ),
        point_id_start=1,
    )


def load_scene_from_colmap_dir(source_dir: Path) -> CameraPoseScene:
    source = Path(source_dir).expanduser().resolve()
    cameras_path = source / "cameras.txt"
    images_path = source / "images.txt"
    points_path = source / "points3D.txt"
    if not cameras_path.is_file() or not images_path.is_file() or not points_path.is_file():
        raise ValueError(
            "COLMAP text model requires cameras.txt, images.txt, and points3D.txt"
        )
    cameras = camera_converter.parse_colmap_cameras_txt(cameras_path)
    images = camera_converter.parse_colmap_images_txt(images_path)
    points = camera_converter.parse_colmap_points3d_txt(points_path)
    xyz, rgb = _rows_to_point_arrays(points)
    poses: List[CameraPose] = []
    for image_row in images:
        camera_id = int(image_row["camera_id"])
        camera_row = cameras.get(camera_id)
        if camera_row is None:
            continue
        fx, fy, width, height = _extract_colmap_intrinsics(camera_row)
        half_w, half_h = _frustum_half_extents_from_intrinsics(fx, fy, width, height)
        r_wc = camera_converter.quat_wxyz_to_rotmat(
            float(image_row["qw"]),
            float(image_row["qx"]),
            float(image_row["qy"]),
            float(image_row["qz"]),
        )
        t_wc = [float(image_row["tx"]), float(image_row["ty"]), float(image_row["tz"])]
        poses.append(
            _camera_pose_from_colmap(
                image_row["name"],
                r_wc,
                t_wc,
                half_w,
                half_h,
            )
        )
    return CameraPoseScene(
        source_kind="colmap",
        source_path=source,
        points_xyz=_as_float32_xyz(xyz),
        points_rgb=_as_uint8_rgb(rgb, xyz.shape[0]),
        cameras=poses,
        info_text=(
            f"COLMAP  ({xyz.shape[0]:,} pts, {len(poses):,} cams)"
        ),
        normalization_log=[
            "[preview] Input normalization",
            "[preview] Source: COLMAP text model",
            "[preview] Camera space: no axis conversion applied",
            "[preview] PointCloud space: no axis conversion applied",
            "[preview] Display space: COLMAP-like world as loaded",
        ],
    )


def load_scene_from_transforms_set(
    transforms_json_path: Path,
    pointcloud_ply_path: Path,
) -> CameraPoseScene:
    json_path = Path(transforms_json_path).expanduser().resolve()
    ply_path = Path(pointcloud_ply_path).expanduser().resolve()
    if not json_path.is_file():
        raise ValueError(f"transforms.json not found: {json_path}")
    if not ply_path.is_file():
        raise ValueError(f"Companion PLY not found: {ply_path}")
    frames, intr = camera_converter.read_transforms_json(json_path)
    fx, fy, _cx, _cy, width, height = intr
    poses = [
        _camera_pose_from_transforms_frame(frame, fx, fy, width, height)
        for frame in frames
    ]
    xyz, rgb = pointcloud_optimizer.load_ply_xyz_rgb(str(ply_path))
    xyz = _transforms_points_to_colmap(xyz)
    return CameraPoseScene(
        source_kind="transforms",
        source_path=json_path,
        points_xyz=_as_float32_xyz(xyz),
        points_rgb=_as_uint8_rgb(rgb, xyz.shape[0]),
        cameras=poses,
        info_text=(
            f"transforms.json + PLY  ({xyz.shape[0]:,} pts, {len(poses):,} cams)"
        ),
        normalization_log=[
            "[preview] Input normalization",
            "[preview] Source: transforms.json + PLY",
            "[preview] Camera: apply_x_fix_gl(-TRANSFORMS_X_FIX_DEG) then compute COLMAP pose",
            "[preview] PointCloud: rotate X by -POINTCLOUD_PLY_X_DEG into COLMAP-like display space",
            "[preview] Display space: COLMAP-like world for preview",
        ],
    )


def load_scene_from_realityscan_csv_set(
    csv_path: Path,
    pointcloud_ply_path: Path,
) -> CameraPoseScene:
    csv_file = Path(csv_path).expanduser().resolve()
    ply_path = Path(pointcloud_ply_path).expanduser().resolve()
    if not csv_file.is_file():
        raise ValueError(f"Camera CSV not found: {csv_file}")
    if not ply_path.is_file():
        raise ValueError(f"Companion PLY not found: {ply_path}")
    rows = camera_converter.read_realityscan_csv(csv_file)
    poses = [_camera_pose_from_realityscan_row(row) for row in rows]
    xyz, rgb = pointcloud_optimizer.load_ply_xyz_rgb(str(ply_path))
    xyz = _realityscan_points_to_colmap(xyz)
    return CameraPoseScene(
        source_kind="realityscan-csv",
        source_path=csv_file,
        points_xyz=_as_float32_xyz(xyz),
        points_rgb=_as_uint8_rgb(rgb, xyz.shape[0]),
        cameras=poses,
        info_text=(
            f"RealityScan CSV + PLY  ({xyz.shape[0]:,} pts, {len(poses):,} cams)"
        ),
        normalization_log=[
            "[preview] Input normalization",
            "[preview] Source: RealityScan CSV + PLY",
            "[preview] Camera: rs_world_to_colmap_world + rs_rot_to_colmap_pose_rot",
            "[preview] PointCloud: RealityScan axis (x, y, z) -> COLMAP-like (x, -z, y)",
            "[preview] Display space: COLMAP-like world for preview",
        ],
    )


def load_scene_from_realityscan_xmp_set(
    xmp_dir_path: Path,
    pointcloud_ply_path: Optional[Path] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> CameraPoseScene:
    xmp_dir = Path(xmp_dir_path).expanduser().resolve()
    if not xmp_dir.is_dir():
        raise ValueError(f"RealityScan XMP directory not found: {xmp_dir}")
    args = _build_import_args(
        width=width,
        height=height,
        xmp_dir=str(xmp_dir),
    )
    cameras, images = camera_converter.build_colmap_from_rs_xmp(args)
    poses = _camera_poses_from_colmap_model(cameras, images)
    if pointcloud_ply_path is not None:
        ply_path = Path(pointcloud_ply_path).expanduser().resolve()
        if not ply_path.is_file():
            raise ValueError(f"RealityScan PLY not found: {ply_path}")
        xyz, rgb = pointcloud_optimizer.load_ply_xyz_rgb(str(ply_path))
        xyz = _realityscan_points_to_colmap(xyz)
        info_text = (
            f"RealityScan XMP + PLY  ({xyz.shape[0]:,} pts, {len(poses):,} cams)"
        )
        point_log = (
            "[preview] PointCloud: RealityScan axis (x, y, z) -> COLMAP-like (x, -z, y)"
        )
    else:
        xyz = np.zeros((0, 3), dtype=np.float32)
        rgb = np.zeros((0, 3), dtype=np.uint8)
        info_text = f"RealityScan XMP  ({len(poses):,} cams)"
        point_log = "[preview] PointCloud: none"
    return CameraPoseScene(
        source_kind="realityscan-xmp",
        source_path=xmp_dir,
        points_xyz=_as_float32_xyz(xyz),
        points_rgb=_as_uint8_rgb(rgb, xyz.shape[0]),
        cameras=poses,
        info_text=info_text,
        normalization_log=[
            "[preview] Input normalization",
            "[preview] Source: RealityScan XMP",
            "[preview] Camera: XMP RS pose -> COLMAP-like world via build_colmap_from_rs_xmp",
            point_log,
            "[preview] Display space: COLMAP-like world for preview",
        ],
    )


def load_scene_from_metashape_xml_set(
    metashape_xml_path: Path,
    pointcloud_ply_path: Optional[Path] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> CameraPoseScene:
    xml_path = Path(metashape_xml_path).expanduser().resolve()
    if not xml_path.is_file():
        raise ValueError(f"Metashape XML not found: {xml_path}")
    args = _build_import_args(
        width=width,
        height=height,
        metashape_xml=str(xml_path),
    )
    cameras, images = camera_converter.build_colmap_from_metashape_xml(args)
    poses = _camera_poses_from_colmap_model(cameras, images)
    if pointcloud_ply_path is not None:
        ply_path = Path(pointcloud_ply_path).expanduser().resolve()
        if not ply_path.is_file():
            raise ValueError(f"RealityScan PLY not found: {ply_path}")
        xyz, rgb = pointcloud_optimizer.load_ply_xyz_rgb(str(ply_path))
        xyz = _realityscan_points_to_colmap(xyz)
        info_text = (
            f"Metashape XML + PLY  ({xyz.shape[0]:,} pts, {len(poses):,} cams)"
        )
        point_log = (
            "[preview] PointCloud: optional PLY interpreted as RealityScan axis and converted to COLMAP-like (x, -z, y)"
        )
    else:
        xyz = np.zeros((0, 3), dtype=np.float32)
        rgb = np.zeros((0, 3), dtype=np.uint8)
        info_text = f"Metashape XML  ({len(poses):,} cams)"
        point_log = "[preview] PointCloud: none"
    return CameraPoseScene(
        source_kind="metashape-xml",
        source_path=xml_path,
        points_xyz=_as_float32_xyz(xyz),
        points_rgb=_as_uint8_rgb(rgb, xyz.shape[0]),
        cameras=poses,
        info_text=info_text,
        normalization_log=[
            "[preview] Input normalization",
            "[preview] Source: Metashape perspective XML",
            "[preview] Camera: Metashape XML -> RS CSV-like rows -> COLMAP-like world",
            point_log,
            "[preview] Display space: COLMAP-like world for preview",
        ],
    )
