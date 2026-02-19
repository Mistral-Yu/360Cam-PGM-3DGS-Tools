#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utilities for optimizing PGM PLY point clouds for 3DGS workflows.

The module loads XYZ/RGB data, can downsample voxels, append extra clouds,
and writes the result as a binary little-endian PLY file.
"""


from __future__ import annotations
import argparse
import heapq
import math
import os
from dataclasses import dataclass
from itertools import count
from typing import Dict, List, Optional, Tuple

import numpy as np
from plyfile import PlyData, PlyElement

SKY_AXIS_CHOICES = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")

# ------------------------------ Utilities ------------------------------


def _fmt3(a) -> str:
    """Format a coordinate triple for logging.

    Args:
        a: Iterable with three numeric entries.

    Returns:
        Compact string representation of the first three values.
    """
    return f"({float(a[0]):.6g}, {float(a[1]):.6g}, {float(a[2]):.6g})"


@dataclass
class PointCloudStats:
    """Statistics derived from a point cloud bounding volume.

    Attributes:
        count: Number of points in the cloud.
        xyz_min: Minimum coordinates along each axis.
        xyz_max: Maximum coordinates along each axis.
        extent: Length of the bounding box along each axis.
        volume: Volume of the bounding box in cubic units.
    """

    count: int
    xyz_min: np.ndarray
    xyz_max: np.ndarray
    extent: np.ndarray
    volume: float


def compute_point_cloud_stats(xyz: np.ndarray) -> PointCloudStats:
    """Compute bounding box statistics for a point cloud.

    Args:
        xyz: Array of shape (N, 3) containing point coordinates.

    Returns:
        Aggregated statistics for the provided points.
    """
    n = int(xyz.shape[0])
    if n == 0:
        zeros = np.zeros(3, dtype=np.float32)
        return PointCloudStats(0, zeros, zeros, zeros, 0.0)

    xyz_min = np.asarray(xyz.min(axis=0), dtype=np.float32)
    xyz_max = np.asarray(xyz.max(axis=0), dtype=np.float32)
    extent = np.maximum(xyz_max - xyz_min, 1e-9)
    volume = float(extent[0] * extent[1] * extent[2])
    return PointCloudStats(n, xyz_min, xyz_max, extent, volume)


def print_point_cloud_stats(
    stats: PointCloudStats,
    include_voxel_reference: bool = True,
) -> None:
    """Display point cloud statistics.

    Args:
        stats: Computed statistics for the point cloud.
        include_voxel_reference: Whether to show heuristic voxel sizing hints.
    """
    print(f"input_points={stats.count:,}")
    print(
        f"[aabb] min={_fmt3(stats.xyz_min)}  max={_fmt3(stats.xyz_max)}  "
        f"extent={_fmt3(stats.extent)}  volume~{stats.volume:.6g}"
    )
    if include_voxel_reference:
        if stats.volume > 0.0 and stats.count > 0:
            v0 = (stats.volume / float(stats.count)) ** (1.0 / 3.0)
        else:
            v0 = 1e-3
        lo = max(v0 / 64.0, 1e-9)
        hi = max(v0 * 64.0, lo * 2.0)
        print(f"[init] v0~{v0:.6g}  lo={lo:.6g}  hi={hi:.6g}")


# ------------------------------ Load / Save ------------------------------


def _find_vertex_data(ply: PlyData):
    """Return vertex data containing XYZ fields from a PLY structure.

    Args:
        ply: Parsed PLY data.

    Returns:
        Structured array with x, y, z columns, or None if unavailable.
    """
    try:
        return ply["vertex"].data
    except Exception:
        pass
    for el in getattr(ply, "elements", []):
        data = getattr(el, "data", None)
        names = (
            getattr(data.dtype, "names", None) if data is not None else None
        )
        if names and all(k in names for k in ("x", "y", "z")):
            return data
    return None


def _axis_direction(label: str) -> Optional[np.ndarray]:
    mapping = {
        "+X": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "-X": np.array([-1.0, 0.0, 0.0], dtype=np.float32),
        "+Y": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "-Y": np.array([0.0, -1.0, 0.0], dtype=np.float32),
        "+Z": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        "-Z": np.array([0.0, 0.0, -1.0], dtype=np.float32),
    }
    vec = mapping.get((label or "").upper())
    if vec is None:
        return None
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else None


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


def _rotation_matrix_from_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    src = source / max(np.linalg.norm(source), 1e-6)
    tgt = target / max(np.linalg.norm(target), 1e-6)
    c = float(np.dot(src, tgt))
    if c > 0.9999:
        return np.eye(3, dtype=np.float32)
    if c < -0.9999:
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(src[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        axis -= axis.dot(src) * src
        axis /= max(np.linalg.norm(axis), 1e-6)
        return _rotation_matrix_from_axis(axis, math.pi)
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


def _sample_hemisphere_points(count: int) -> np.ndarray:
    idx = np.arange(count, dtype=np.float32)
    phi = math.pi * (3.0 - math.sqrt(5.0))
    z = 1.0 - idx / count
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    x = np.cos(phi * idx) * radius
    y = np.sin(phi * idx) * radius
    return np.stack((x, y, z), axis=1)


def _generate_sky_points(
    center: np.ndarray,
    axis_vec: np.ndarray,
    scale: float,
    count: int,
    color: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    samples = _sample_hemisphere_points(count) * float(scale)
    rot = _rotation_matrix_from_vectors(
        np.array([0.0, 0.0, 1.0], dtype=np.float32), axis_vec
    )
    oriented = samples @ rot.T
    world = oriented + center
    colors = np.tile(color.astype(np.uint8, copy=False), (world.shape[0], 1))
    return world.astype(np.float32, copy=False), colors


def _parse_sky_color(text: Optional[str]) -> np.ndarray:
    default = np.array([135, 206, 250], dtype=np.uint8)
    if not text:
        return default
    value = text.strip()
    if not value:
        return default
    if "," in value:
        parts = [p.strip() for p in value.split(",")]
        if len(parts) != 3:
            raise ValueError("expected R,G,B components")
        comps = [int(float(p)) for p in parts]
    elif value.startswith("#"):
        hexval = value[1:]
        if len(hexval) == 3:
            hexval = "".join(ch * 2 for ch in hexval)
        if len(hexval) != 6:
            raise ValueError("hex color must be #RGB or #RRGGBB")
        comps = [int(hexval[i : i + 2], 16) for i in (0, 2, 4)]
    else:
        raise ValueError("use #RRGGBB or R,G,B format")
    clamped = [max(0, min(255, int(c))) for c in comps]
    return np.array(clamped, dtype=np.uint8)


def _extract_xyz_rgb_from_structured(v) -> Tuple[np.ndarray, np.ndarray]:
    """Extract XYZ and RGB arrays from a structured NumPy array.

    Args:
        v: Structured array produced by plyfile.

    Returns:
        Tuple containing XYZ float32 positions and uint8 RGB colors.

    Raises:
        ValueError: If the array lacks x, y, or z fields.
    """
    names = v.dtype.names
    if not all(k in names for k in ("x", "y", "z")):
        raise ValueError("Could not find x, y, z fields")

    xyz = np.stack(
        [
            v["x"].astype(np.float32),
            v["y"].astype(np.float32),
            v["z"].astype(np.float32),
        ],
        axis=1,
    )

    # Candidate color field triplets
    cand = [
        ("red", "green", "blue"),
        ("r", "g", "b"),
        ("diffuse_red", "diffuse_green", "diffuse_blue"),
    ]
    rgb = None
    for r, g, b in cand:
        if r in names and g in names and b in names:
            rr = v[r]
            gg = v[g]
            bb = v[b]
            if (
                rr.dtype.kind == "f"
                or gg.dtype.kind == "f"
                or bb.dtype.kind == "f"
            ):
                rr = np.clip(rr, 0.0, 1.0) * 255.0
                gg = np.clip(gg, 0.0, 1.0) * 255.0
                bb = np.clip(bb, 0.0, 1.0) * 255.0
                rgb = np.stack([rr, gg, bb], axis=1).round().astype(np.uint8)
            else:
                rgb = np.stack([rr, gg, bb], axis=1).astype(
                    np.uint8, copy=False
                )
            break
    if rgb is None:
        rgb = np.full((xyz.shape[0], 3), 255, dtype=np.uint8)

    return xyz, rgb


def load_ply_xyz_rgb(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load XYZ and RGB data from a PLY file.

    Args:
        path: Path to the PLY file.

    Returns:
        Tuple of XYZ positions and RGB colors.
    """
    ply = PlyData.read(path)
    vdata = _find_vertex_data(ply)
    if vdata is None:
        raise ValueError(
            f"Could not find a vertex element with x, y, z: {path}"
        )
    xyz, rgb = _extract_xyz_rgb_from_structured(vdata)
    return xyz, rgb


def save_ply_binary_little(
    path: str, xyz: np.ndarray, rgb: np.ndarray
) -> None:
    """Write XYZ and RGB arrays to a binary little-endian PLY file.

    Args:
        path: Destination path for the PLY file.
        xyz: Array of shape (N, 3) with point coordinates.
        rgb: Array of shape (N, 3) with 8-bit color values.

    Raises:
        ValueError: If xyz and rgb have mismatched lengths.
    """
    if xyz.shape[0] != rgb.shape[0]:
        raise ValueError("xyz and rgb must have the same number of rows")
    n = xyz.shape[0]
    arr = np.empty(
        n,
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    arr["x"] = xyz[:, 0].astype(np.float32, copy=False)
    arr["y"] = xyz[:, 1].astype(np.float32, copy=False)
    arr["z"] = xyz[:, 2].astype(np.float32, copy=False)
    arr["red"] = rgb[:, 0].astype(np.uint8, copy=False)
    arr["green"] = rgb[:, 1].astype(np.uint8, copy=False)
    arr["blue"] = rgb[:, 2].astype(np.uint8, copy=False)

    el = PlyElement.describe(arr, "vertex")
    PlyData([el], text=False, byte_order="<").write(path)


# ------------------------------ Downsampling ------------------------------


def _grid_keys(
    xyz: np.ndarray, voxel: float, xyz_min: Optional[np.ndarray] = None
) -> np.ndarray:
    """Compute integer voxel keys for each point.

    Args:
        xyz: Array of shape (N, 3) with point coordinates.
        voxel: Voxel edge length.
        xyz_min: Optional minimum XYZ values used as the grid origin.

    Returns:
        Integer keys identifying the voxel that contains each point.

    Raises:
        ValueError: If voxel is not strictly positive.
    """
    if voxel <= 0:
        raise ValueError("voxel must be > 0")
    if xyz_min is None:
        xyz_min = xyz.min(axis=0, keepdims=True)
    keys = np.floor((xyz - xyz_min) / voxel).astype(np.int64, copy=False)
    return keys


def voxel_downsample_by_size(
    xyz: np.ndarray,
    rgb: np.ndarray,
    voxel: float,
    *,
    representative: str = "centroid",
) -> Tuple[np.ndarray, np.ndarray]:
    """Downsample points using a fixed voxel size.

    Args:
        xyz: Array of shape (N, 3) with point coordinates.
        rgb: Array of shape (N, 3) with 8-bit color values.
        voxel: Voxel edge length.
        representative: Strategy used to pick the voxel representative
            ('centroid', 'center', 'first', or 'random').

    Returns:
        Downsampled XYZ and RGB arrays.

    Raises:
        ValueError: If representative is not recognized.
    """
    if xyz.shape[0] == 0:
        return xyz.astype(np.float32, copy=False), rgb.astype(
            np.uint8, copy=False
        )

    xyz_min = xyz.min(axis=0, keepdims=True)
    keys = _grid_keys(xyz, voxel, xyz_min)  # (N,3) int64
    uniq, inv, counts = np.unique(
        keys, axis=0, return_inverse=True, return_counts=True
    )
    k = uniq.shape[0]

    xyz32 = xyz.astype(np.float32, copy=False)

    rng = np.random.default_rng()

    if representative == "first":
        pick_idx = np.full(k, -1, dtype=np.int64)
        for idx, gid in enumerate(inv):
            if pick_idx[gid] == -1:
                pick_idx[gid] = idx
    elif representative == "random":
        order = np.argsort(inv, kind="mergesort")
        inv_sorted = inv[order]
        starts = np.flatnonzero(np.r_[True, inv_sorted[1:] != inv_sorted[:-1]])
        ends = np.r_[starts[1:], inv_sorted.size]
        pick_idx = np.empty(k, dtype=np.int64)
        for a, b in zip(starts, ends):
            gid = int(inv_sorted[a])
            rand_offset = int(rng.integers(max(1, b - a)))
            pick_idx[gid] = int(order[a + rand_offset])
    else:
        if representative == "center":
            targets = (
                xyz_min + (uniq.astype(np.float32) + 0.5) * voxel
            ).astype(np.float32)
        elif representative == "centroid":
            sums = np.zeros((k, 3), dtype=np.float64)
            np.add.at(sums, inv, xyz.astype(np.float64, copy=False))
            centroids = sums / counts[:, None]
            targets = centroids.astype(np.float32, copy=False)
        else:
            raise ValueError(
                f"Unknown representative strategy: {representative}"
            )

        target_per_point = targets[inv]  # (N,3)
        diff = xyz32 - target_per_point
        dist2 = (diff * diff).sum(axis=1)  # (N,)

        # Sort by voxel id to pick the minimum dist2 per group
        order = np.argsort(inv, kind="mergesort")  # Stable sort
        inv_sorted = inv[order]
        dist2_sorted = dist2[order]

        # Detect the start/end of each group
        starts = np.flatnonzero(np.r_[True, inv_sorted[1:] != inv_sorted[:-1]])
        ends = np.r_[starts[1:], inv_sorted.size]

        pick_idx = np.empty(k, dtype=np.int64)
        for a, b in zip(starts, ends):
            gid = int(inv_sorted[a])
            # Argmin within the group and map back to the original index
            rel = a + int(np.argmin(dist2_sorted[a:b]))
            orig = int(order[rel])
            pick_idx[gid] = orig

    out_xyz = xyz[pick_idx].astype(np.float32, copy=False)
    out_rgb = rgb[pick_idx].astype(np.uint8, copy=False)
    return out_xyz, out_rgb


def _unique_voxel_count(
    xyz: np.ndarray, voxel: float, xyz_min: Optional[np.ndarray] = None
) -> int:
    """Count unique voxels occupied by the point cloud.

    Args:
        xyz: Array of shape (N, 3) with point coordinates.
        voxel: Voxel edge length.
        xyz_min: Optional grid origin.

    Returns:
        Number of occupied voxels.
    """
    if xyz.shape[0] == 0:
        return 0
    keys = _grid_keys(xyz, voxel, xyz_min)
    return int(np.unique(keys, axis=0).shape[0])


def voxel_downsample_to_target(
    xyz: np.ndarray,
    rgb: np.ndarray,
    target_points: int,
    tol_ratio: float = 0.02,
    max_iter: int = 32,
    stats: Optional[PointCloudStats] = None,
    log_bounds: bool = True,
    representative: str = "centroid",
) -> Tuple[np.ndarray, np.ndarray]:
    """Downsample points to approach a target count.

    Args:
        xyz: Array of shape (N, 3) with point coordinates.
        rgb: Array of shape (N, 3) with 8-bit color values.
        target_points: Desired number of output points.
        tol_ratio: Acceptable relative deviation from the target count.
        max_iter: Maximum number of binary search iterations.
        stats: Optional precomputed statistics for the point cloud.
        log_bounds: Whether to log the search bounds while iterating.
        representative: Strategy for choosing voxel representatives.

    Returns:
        Downsampled XYZ and RGB arrays.

    Raises:
        ValueError: If target_points is not positive.

    Notes:
        The representative argument accepts 'centroid', 'center', or 'first'.
    """

    n = xyz.shape[0]
    if n == 0:
        return xyz.astype(np.float32, copy=False), rgb.astype(
            np.uint8, copy=False
        )

    if stats is None or stats.count != n:
        stats = compute_point_cloud_stats(xyz)

    print(
        f"[target] input_points={n:,}  target={target_points:,}  "
        f"tol=+/-{tol_ratio * 100:.1f}%  max_iter={max_iter}"
    )
    if target_points <= 0 or target_points >= n:
        print(
            "[target] skip: "
            f"target={target_points} is out of range "
            "(output = input)"
        )
        return xyz.astype(np.float32, copy=False), rgb.astype(
            np.uint8, copy=False
        )

    xyz_min = stats.xyz_min
    vol = stats.volume
    extent = stats.extent
    v0 = (vol / float(target_points)) ** (1.0 / 3.0) if vol > 0 else 1e-3

    # Cache unique-voxel counts by voxel size during bound search.
    count_cache: Dict[float, int] = {}

    def _count_unique_cached(voxel_size: float) -> int:
        key = round(float(voxel_size), 12)
        cached = count_cache.get(key)
        if cached is not None:
            return cached
        cnt = _unique_voxel_count(xyz, voxel_size, xyz_min)
        count_cache[key] = cnt
        return cnt

    min_voxel = 1e-9
    lo = max(v0 / 64.0, min_voxel)
    hi = max(v0 * 64.0, lo * 2.0)

    cnt_lo = _count_unique_cached(lo)
    if cnt_lo < target_points:
        print(
            f"[shrink] initial lo={lo:.6g} -> unique={cnt_lo:,} "
            "(below target); shrinking lower bound"
        )
        shrink_iter = 0
        while cnt_lo < target_points and lo > min_voxel:
            prev_lo = lo
            lo = max(lo * 0.5, min_voxel)
            if lo == prev_lo:
                break
            cnt_lo = _count_unique_cached(lo)
            shrink_iter += 1
            print(
                f"[shrink {shrink_iter:02d}] lo={lo:.6g} -> unique={cnt_lo:,}"
            )
            if shrink_iter >= 32:
                break
        if cnt_lo < target_points:
            print(
                "[shrink] warning: minimum voxel size still below target; "
                "will use best available count."
            )
    hi = max(hi, lo * 2.0)

    if log_bounds:
        print(
            f"[aabb] min={_fmt3(xyz_min)}  max={_fmt3(stats.xyz_max)}  "
            f"extent={_fmt3(extent)}  volume~{vol:.6g}"
        )
    print(f"[init] v0~{v0:.6g}  lo={lo:.6g}  hi={hi:.6g}")

    # Expand hi until the unique voxel count is at most the target.
    for _ in range(10):
        cnt_hi = _count_unique_cached(hi)
        print(f"[expand] try hi={hi:.6g} -> unique={cnt_hi:,}")
        if cnt_hi <= target_points:
            break
        hi *= 2.0
    else:
        print("[expand] warning: could not find a sufficient upper bound")

    best_voxel = v0
    best_diff = 10**18
    best_cnt = n

    for it in range(1, max_iter + 1):
        mid = 0.5 * (lo + hi)
        cnt = _count_unique_cached(mid)
        diff = abs(cnt - target_points)
        ratio = diff / float(target_points)
        if diff < best_diff:
            best_diff = diff
            best_voxel = mid
            best_cnt = cnt
        print(f"[iter {it:02d}] voxel={mid:.2f}  unique={cnt:,}")
        if ratio <= tol_ratio:
            print(f"[stop] within tolerance: voxel={mid:.6g}  unique={cnt:,}")
            best_voxel = mid
            best_cnt = cnt
            break
        if cnt > target_points:
            lo = mid
        else:
            hi = mid

    print(
        f"[best] voxel~{best_voxel:.6g}  unique~{best_cnt:,}  "
        f"(best_diff={best_diff:,})"
    )

    out_xyz, out_rgb = voxel_downsample_by_size(
        xyz, rgb, best_voxel, representative=representative
    )
    print(f"[final] voxel={best_voxel:.6g}  out_points={out_xyz.shape[0]:,}")
    return out_xyz, out_rgb


def spatial_hash_downsample_one_pass(
    xyz: np.ndarray,
    rgb: np.ndarray,
    target_points: Optional[int] = None,
    voxel_size: Optional[float] = None,
    stats: Optional[PointCloudStats] = None,
    representative: str = "centroid",
) -> Tuple[np.ndarray, np.ndarray]:
    """Approximate downsampling with a single spatial-hash voxel pass.

    This mode avoids iterative voxel-size search. When voxel_size is given,
    it is used directly. Otherwise, a heuristic voxel size is estimated from
    the target point count and the cloud bounding-box volume.

    Args:
        xyz: Array of shape (N, 3) with point coordinates.
        rgb: Array of shape (N, 3) with 8-bit color values.
        target_points: Approximate desired number of output points.
        voxel_size: Optional fixed voxel size. Takes precedence over target.
        stats: Optional precomputed statistics for the input cloud.
        representative: Strategy for choosing voxel representatives.

    Returns:
        Downsampled XYZ and RGB arrays.
    """
    n = int(xyz.shape[0])
    if n == 0:
        return xyz.astype(np.float32, copy=False), rgb.astype(
            np.uint8, copy=False
        )

    voxel: Optional[float] = None
    if voxel_size is not None and voxel_size > 0:
        voxel = float(voxel_size)
        print(f"[spatial-hash] fixed voxel-size={voxel:.6g}")
    elif target_points is not None and target_points > 0:
        target = int(max(1, min(n, target_points)))
        if target >= n:
            print(
                "[spatial-hash] skip: "
                f"target={target:,} is not smaller than input={n:,}"
            )
            return xyz.astype(np.float32, copy=False), rgb.astype(
                np.uint8, copy=False
            )
        if stats is None or stats.count != n:
            stats = compute_point_cloud_stats(xyz)
        vol = stats.volume
        voxel_init = (vol / float(target)) ** (1.0 / 3.0) if vol > 0 else 1e-3
        voxel_init = max(float(voxel_init), 1e-9)
        voxel = voxel_init
        xyz_min = stats.xyz_min
        probe_logs: List[Tuple[int, float, int]] = []
        prev_voxel: Optional[float] = None
        prev_count: Optional[int] = None
        max_probes = 3
        for probe_idx in range(1, max_probes + 1):
            cnt_probe = _unique_voxel_count(xyz, voxel, xyz_min)
            probe_logs.append((probe_idx, voxel, cnt_probe))
            if cnt_probe <= 0:
                break
            ratio = float(cnt_probe) / float(target)
            if abs(ratio - 1.0) <= 0.06:
                break
            if probe_idx >= max_probes:
                break
            if (
                prev_voxel is not None
                and prev_count is not None
                and prev_count > 0
                and cnt_probe != prev_count
                and abs(voxel - prev_voxel) > 1e-12
            ):
                try:
                    effective_dim = math.log(
                        float(cnt_probe) / float(prev_count)
                    ) / math.log(float(prev_voxel) / float(voxel))
                except (ValueError, ZeroDivisionError):
                    effective_dim = 2.0
                if not np.isfinite(effective_dim):
                    effective_dim = 2.0
                effective_dim = max(1.2, min(3.0, abs(float(effective_dim))))
            else:
                if ratio < 0.2:
                    effective_dim = 1.45
                elif ratio < 0.5:
                    effective_dim = 1.7
                elif ratio > 2.0:
                    effective_dim = 2.6
                else:
                    effective_dim = 2.1
            scale = ratio ** (1.0 / effective_dim)
            scale = min(2.8, max(0.12, scale))
            new_voxel = max(voxel * scale, 1e-9)
            prev_voxel = voxel
            prev_count = cnt_probe
            if abs(new_voxel - voxel) <= max(1e-9, voxel * 1e-4):
                break
            voxel = new_voxel
        if not probe_logs:
            probe_logs.append((1, voxel, 0))
        for probe_idx, probe_voxel, probe_count in probe_logs:
            print(
                f"[spatial-hash probe {probe_idx}] "
                f"voxel={probe_voxel:.6g} unique={probe_count:,}"
            )
        print(
            f"[spatial-hash] target~{target:,} "
            f"-> voxel={voxel:.6g} probes={len(probe_logs)}"
        )
    else:
        print("[spatial-hash] skip (no voxel-size/target-points)")
        return xyz.astype(np.float32, copy=False), rgb.astype(
            np.uint8, copy=False
        )

    out_xyz, out_rgb = voxel_downsample_by_size(
        xyz, rgb, voxel, representative=representative
    )
    print(
        f"[spatial-hash] voxel={voxel:.6g}  "
        f"out_points={out_xyz.shape[0]:,} (approx)"
    )
    return out_xyz, out_rgb


def adaptive_voxel_downsample(
    xyz: np.ndarray,
    rgb: np.ndarray,
    target_points: Optional[int],
    weight_power: float = 1.0,
    stats: Optional[PointCloudStats] = None,
    min_voxel_size: Optional[float] = None,
    representative: str = "centroid",
    max_depth: int = 12,
) -> Tuple[np.ndarray, np.ndarray]:
    """Adaptive octree-based voxel sampling that prefers dense regions.

    Args:
        xyz: Array of shape (N, 3) with point coordinates.
        rgb: Array of shape (N, 3) with 8-bit color values.
        target_points: Desired number of output points (clamped to [1, N]).
        weight_power: Exponent applied to voxel population when prioritising
            splits (>1 emphasises dense voxels, 0 makes all voxels equal).
        stats: Optional precomputed statistics for the point cloud.
        min_voxel_size: Optional minimum voxel edge length; prevents recursive
            subdivision beyond this size when supplied.
        representative: Strategy for picking a point from a leaf voxel
            ('centroid', 'center', 'first', or 'random').
        max_depth: Maximum octree depth.

    Returns:
        Downsampled XYZ and RGB arrays selected by adaptive sampling.
    """

    n = int(xyz.shape[0])
    if n == 0:
        return xyz.astype(np.float32, copy=False), rgb.astype(np.uint8, copy=False)

    if target_points is None or target_points <= 0:
        target = n
    else:
        target = int(max(1, min(n, target_points)))

    if target >= n:
        return xyz.astype(np.float32, copy=False), rgb.astype(np.uint8, copy=False)

    xyz32 = xyz.astype(np.float32, copy=False)
    rgb8 = rgb.astype(np.uint8, copy=False)
    rng = np.random.default_rng()

    if stats is None or stats.count != n:
        stats = compute_point_cloud_stats(xyz32)

    extent = np.asarray(stats.extent, dtype=np.float32)
    cube_size = float(np.max(extent))
    if cube_size <= 0.0:
        keep = np.arange(0, target, dtype=np.int64)
        return xyz32[keep], rgb8[keep]

    pad = np.maximum((cube_size - extent) * 0.5, 0.0)
    cube_min = np.asarray(stats.xyz_min - pad, dtype=np.float32)

    min_voxel = float(min_voxel_size) if min_voxel_size else None
    weight_power = float(weight_power) if weight_power is not None else 1.0
    if weight_power < 0.0:
        weight_power = 0.0

    def _weight(count_value: int) -> float:
        if count_value <= 0:
            return 0.0
        if weight_power == 0.0:
            return 1.0
        return float(count_value) ** weight_power

    @dataclass
    class _Node:
        indices: np.ndarray
        min_corner: np.ndarray
        size: float
        depth: int
        count: int
        weight: float

    root_indices = np.arange(n, dtype=np.int64)
    root = _Node(
        indices=root_indices,
        min_corner=cube_min,
        size=cube_size,
        depth=0,
        count=n,
        weight=_weight(n),
    )

    heap: List[Tuple[float, int, _Node]] = []
    seq = count()
    heapq.heappush(heap, (-root.weight, next(seq), root))
    leaves: List[_Node] = []

    eps = 1e-9
    desired = target

    def _can_split(node: _Node) -> bool:
        if node.count <= 1:
            return False
        if node.depth >= max_depth:
            return False
        if min_voxel is not None and node.size <= (min_voxel + eps):
            return False
        if node.size * 0.5 <= eps:
            return False
        return True

    while heap and (len(leaves) + len(heap)) < desired:
        _, _, node = heapq.heappop(heap)
        if not _can_split(node):
            leaves.append(node)
            continue

        half = node.size * 0.5
        pts = xyz32[node.indices]
        centre = node.min_corner + half
        codes = (
            ((pts[:, 0] >= centre[0]).astype(np.int8) << 2)
            | ((pts[:, 1] >= centre[1]).astype(np.int8) << 1)
            | (pts[:, 2] >= centre[2]).astype(np.int8)
        )

        child_nodes: List[_Node] = []
        for child_code in range(8):
            mask = codes == child_code
            if not np.any(mask):
                continue
            child_idx = node.indices[mask]
            child_count = int(child_idx.size)
            child_min = node.min_corner + np.array(
                [
                    half if (child_code & 4) else 0.0,
                    half if (child_code & 2) else 0.0,
                    half if (child_code & 1) else 0.0,
                ],
                dtype=np.float32,
            )
            child_nodes.append(
                _Node(
                    indices=child_idx,
                    min_corner=child_min,
                    size=half,
                    depth=node.depth + 1,
                    count=child_count,
                    weight=_weight(child_count),
                )
            )

        if not child_nodes:
            leaves.append(node)
            continue

        for child in child_nodes:
            if child.count <= 1:
                leaves.append(child)
            else:
                heapq.heappush(heap, (-child.weight, next(seq), child))

        if len(leaves) + len(heap) >= desired:
            break

    leaves.extend(item[2] for item in heap)
    leaves = [leaf for leaf in leaves if leaf.count > 0]
    if not leaves:
        idx = np.arange(0, min(n, desired), dtype=np.int64)
        return xyz32[idx], rgb8[idx]

    leaves.sort(
        key=lambda node: (node.weight, node.count, -int(node.indices[0])),
        reverse=True,
    )
    keep_count = min(len(leaves), desired)
    selected = leaves[:keep_count]

    def _pick_index(node: _Node) -> int:
        idx = node.indices
        if idx.size == 0:
            return -1
        if representative == "first" or idx.size == 1:
            return int(idx[0])
        pts = xyz32[idx]
        if representative == "center":
            target_point = node.min_corner + node.size * 0.5
        elif representative == "centroid":
            target_point = pts.mean(axis=0)
        elif representative == "random":
            return int(idx[int(rng.integers(idx.size))])
        else:
            raise ValueError(
                f"Unknown representative strategy: {representative}"
            )
        diff = pts - target_point
        dist2 = (diff * diff).sum(axis=1)
        return int(idx[int(np.argmin(dist2))])

    chosen: List[int] = []
    seen = set()
    for node in selected:
        pick = _pick_index(node)
        if pick >= 0 and pick not in seen:
            chosen.append(pick)
            seen.add(pick)

    if not chosen:
        idx = np.arange(0, min(n, desired), dtype=np.int64)
        return xyz32[idx], rgb8[idx]

    chosen_idx = np.asarray(chosen, dtype=np.int64)
    return xyz32[chosen_idx], rgb8[chosen_idx]


# ------------------------------ CLI ------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for the gs360_PlyOptimizer CLI.

    Args:
        argv: Optional argument vector to parse instead of sys.argv.

    Returns:
        Process exit code.
    """
    ap = argparse.ArgumentParser(
        prog="gs360_PlyOptimizer",
        description=(
            "PGM to 3DGS PLY optimizer (XYZ+RGB load/save, "
            "voxel downsampling, append)"
        ),
    )
    ap.add_argument(
        "-i",
        "--in",
        dest="input",
        required=True,
        help="Input PLY file path",
    )
    ap.add_argument(
        "-o",
        "--out",
        dest="output",
        default=None,
        help=(
            "Output PLY (binary little-endian). "
            "Omit to only display point cloud statistics."
        ),
    )
    ap.add_argument(
        "-t",
        "--target-points",
        type=int,
        default=None,
        help=(
            "Target number of points after downsampling (approximate). "
            "Overrides --voxel-size when provided."
        ),
    )
    ap.add_argument(
        "-r",
        "--target-percent",
        type=float,
        default=None,
        help=(
            "Target percentage of the input point count. "
            "The percentage is converted to --target-points after loading."
        ),
    )
    ap.add_argument(
        "-v",
        "--voxel-size",
        type=float,
        default=None,
        help="Fixed voxel size in meters.",
    )
    ap.add_argument(
        "--downsample-method",
        choices=("voxel", "spatial-hash", "adaptive"),
        default="voxel",
        help=(
            "Downsampling method: "
            "voxel=existing voxel mode (fixed size or target search), "
            "spatial-hash=single-pass approximate voxel hashing, "
            "adaptive=octree-based adaptive sampling."
        ),
    )
    ap.add_argument(
        "--adaptive",
        action="store_true",
        help=(
            "Enable adaptive voxel sampling (octree). "
            "Requires target points/percent; keeps more points in dense areas. "
            "When --voxel-size is set it becomes the minimum leaf size."
        ),
    )
    ap.add_argument(
        "--adaptive-weight",
        type=float,
        default=1.0,
        metavar="POWER",
        help=(
            "Weight exponent for adaptive sampling. Values >1 favour splitting "
            "dense voxels more aggressively, 0 treats all voxels equally."
        ),
    )
    ap.add_argument(
        "-a",
        "--append-ply",
        action="append",
        default=[],
        help=(
            "Additional PLY files to append after downsampling."
            " Paths are resolved relative to the input file when not absolute."
            " May be specified multiple times."
        ),
    )
    ap.add_argument(
        "-k",
        "--keep-strategy",
        choices=("centroid", "center", "first", "random"),
        default="centroid",
        help=(
            "Representative selection per voxel: "
            "centroid=closest to centroid (default), "
            "center=closest to voxel center, "
            "first=first point encountered, "
            "random=random point per voxel."
        ),
    )
    ap.add_argument(
        "--sky-axis",
        choices=SKY_AXIS_CHOICES,
        default=None,
        help=(
            "Add a synthetic hemispherical sky point cloud from this axis "
            "direction (e.g., +Z). Requires --out."
        ),
    )
    ap.add_argument(
        "--sky-scale",
        type=float,
        default=100.0,
        help="Radius of the generated sky hemisphere (same units as the PLY).",
    )
    ap.add_argument(
        "--sky-count",
        type=int,
        default=4000,
        help="Number of points to generate for the sky hemisphere.",
    )
    ap.add_argument(
        "--sky-color",
        type=str,
        default="#87cefa",
        help="Sky color as #RRGGBB or R,G,B (0-255).",
    )
    args = ap.parse_args(argv)

    if args.target_points is not None and args.target_points <= 0:
        ap.error("--target-points must be greater than 0")
    sky_color = None
    if args.sky_axis:
        if args.sky_scale is None or args.sky_scale <= 0:
            ap.error("--sky-scale must be > 0 when --sky-axis is set")
        if args.sky_count is None or args.sky_count <= 0:
            ap.error("--sky-count must be > 0 when --sky-axis is set")
        try:
            sky_color = _parse_sky_color(args.sky_color)
        except ValueError as exc:
            ap.error(f"--sky-color {exc}")

    base_path = os.path.expanduser(args.input)
    if not os.path.isabs(base_path):
        base_path = os.path.abspath(base_path)
    base_dir = os.path.dirname(base_path) or "."

    # 1) Load
    xyz, rgb = load_ply_xyz_rgb(base_path)
    print(f"[load] base: {base_path}  points={xyz.shape[0]:,}")

    stats = compute_point_cloud_stats(xyz)

    target_points = (
        args.target_points
        if args.target_points and args.target_points > 0
        else None
    )
    if args.target_percent is not None:
        pct = args.target_percent
        if pct <= 0 or stats.count == 0:
            computed_target = 0
        else:
            computed_target = int(round(stats.count * (pct / 100.0)))
            computed_target = max(1, min(stats.count, computed_target))
        print(
            "[target-percent]",
            f"{pct:.6g}% of {stats.count:,} -> "
            f"target_points={computed_target:,}",
        )
        if computed_target > 0:
            if target_points is not None and target_points != computed_target:
                print(
                    "[target-percent] override",
                    f"replacing explicit target_points={target_points:,}",
                )
            target_points = computed_target

    include_voxel_reference = not (
        target_points is not None and target_points > 0
    )
    print_point_cloud_stats(
        stats, include_voxel_reference=include_voxel_reference
    )

    if args.output is None:
        if (
            (args.voxel_size is not None and args.voxel_size > 0)
            or (target_points is not None and target_points > 0)
            or args.adaptive
            or args.downsample_method != "voxel"
            or args.append_ply
        ):
            print("[warn] --out missing; skipping downsample/append options.")
        else:
            print("[info] --out not provided; statistics only.")
        return 0

    # 2) Downsample
    downsample_method = args.downsample_method
    if args.adaptive and downsample_method != "adaptive":
        print(
            "[warn] --adaptive is deprecated by --downsample-method; "
            "forcing method=adaptive."
        )
        downsample_method = "adaptive"

    adaptive_min_voxel = (
        args.voxel_size if args.voxel_size is not None and args.voxel_size > 0 else None
    )
    if downsample_method == "adaptive":
        adaptive_target = (
            target_points if target_points is not None and target_points > 0 else stats.count
        )
        if adaptive_target == stats.count:
            print(
                "[adaptive] target not specified; defaulting to input count "
                "(no reduction)."
            )
        print(
            "[adaptive] weight={:.3g} min_voxel={}".format(
                args.adaptive_weight,
                f"{adaptive_min_voxel:.6g}" if adaptive_min_voxel else "auto",
            )
        )
        xyz, rgb = adaptive_voxel_downsample(
            xyz,
            rgb,
            adaptive_target,
            weight_power=args.adaptive_weight,
            stats=stats,
            min_voxel_size=adaptive_min_voxel,
            representative=args.keep_strategy,
        )
        print(
            f"[adaptive] target~{adaptive_target:,} -> {xyz.shape[0]:,} points"
        )
    elif downsample_method == "spatial-hash":
        xyz, rgb = spatial_hash_downsample_one_pass(
            xyz,
            rgb,
            target_points=target_points,
            voxel_size=args.voxel_size if args.voxel_size and args.voxel_size > 0 else None,
            stats=stats,
            representative=args.keep_strategy,
        )
    elif args.voxel_size is not None and args.voxel_size > 0:
        print(f"[downsample] fixed voxel-size={args.voxel_size:.6g}")
        xyz, rgb = voxel_downsample_by_size(
            xyz, rgb, args.voxel_size, representative=args.keep_strategy
        )
        print(f"[downsample] -> {xyz.shape[0]:,} points")
    elif target_points is not None and target_points > 0:
        xyz, rgb = voxel_downsample_to_target(
            xyz,
            rgb,
            target_points,
            stats=stats,
            log_bounds=False,
            representative=args.keep_strategy,
        )
        print(
            f"[downsample] target_points={target_points:,} -> "
            f"{xyz.shape[0]:,} points"
        )
    else:
        print("[downsample] skip (no voxel-size/target-points)")

    # 3) Append additional clouds (after downsampling)
    total_added = 0
    for apath in args.append_ply:
        full_path = os.path.expanduser(apath)
        if not os.path.isabs(full_path):
            full_path = os.path.join(base_dir, full_path)
        full_path = os.path.abspath(full_path)
        ax, ac = load_ply_xyz_rgb(full_path)
        xyz = np.concatenate([xyz, ax], axis=0)
        rgb = np.concatenate([rgb, ac], axis=0)
        total_added += ax.shape[0]
        print(
            f"[append] {full_path} +{ax.shape[0]:,} -> total {xyz.shape[0]:,}"
        )
    if total_added > 0:
        print(f"[append] total added: {total_added:,}")

    if args.sky_axis:
        axis_vec = _axis_direction(args.sky_axis)
        if axis_vec is None:
            ap.error(f"Invalid --sky-axis value: {args.sky_axis}")
        stats_after = compute_point_cloud_stats(xyz)
        center = (stats_after.xyz_min + stats_after.xyz_max) * 0.5
        sky_points, sky_colors = _generate_sky_points(
            center.astype(np.float32, copy=False),
            axis_vec,
            float(args.sky_scale),
            int(args.sky_count),
            sky_color if sky_color is not None else np.array([135, 206, 250], dtype=np.uint8),
        )
        xyz = np.concatenate([xyz, sky_points], axis=0)
        rgb = np.concatenate([rgb, sky_colors], axis=0)
        print(
            f"[sky] axis={args.sky_axis} scale={args.sky_scale:.6g} "
            f"count={sky_points.shape[0]:,} -> total {xyz.shape[0]:,}"
        )

    # 4) Save
    out_path = os.path.expanduser(args.output)
    if not os.path.isabs(out_path):
        out_path = os.path.abspath(out_path)
    save_ply_binary_little(out_path, xyz, rgb)
    print(
        f"[save] {out_path}  points={xyz.shape[0]:,}  (binary little-endian)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
