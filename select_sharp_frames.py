#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
select_sharp_frames.py (listdir-based sharp frame selector)

Workflow:
  1) Gather images directly under the input directory (tif/png/jpg by default)
  2) Sort them according to the chosen rule
  3) Score sharpness in parallel (grayscale read + optional resize/crop)
  4) Split the list into --group batches, keep the sharpest frame(s) per batch,
     and move the rest into in_dir/blur/

Defaults:
  - ext: all (tif/png/jpg)
  - metric: hybrid (0.6 * laplacian variance + 0.4 * fft energy)
  - low group: off unless explicitly enabled (ratio controls extra keeps)
  - crop_ratio: 0.8 (evaluate the central 80%)
  - sort: lastnum (prefer trailing numbers, fallback to name)
  - use_exposure: off (enable with --use_exposure)
  - csv: off (enable with --csv)
  - max_long: 1280 px (0 disables downscale)
  - workers: min(8, os.cpu_count() or 4)
  - opencv_threads: 0 (leave OpenCV threading untouched)

Python 3.7+ / OpenCV 4.x
"""

import os
import sys
import csv
import argparse
import shutil
import re
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np


# ---------- Collection and sorting (dedupe) ----------

EXTS = {
    "tif": {".tif", ".tiff"},
    "jpg": {".jpg", ".jpeg"},
    "png": {".png"},
}
ALL_EXTS = set().union(*EXTS.values())

_num_pat = re.compile(r'(\d+)')

def _extract_number_groups(stem):
    return _num_pat.findall(stem)

def sort_key_lastnum(path):
    base = os.path.basename(path)
    stem, _ = os.path.splitext(base)
    gs = _extract_number_groups(stem)
    if gs:
        return (0, int(gs[-1]), base.lower())
    return (1, base.lower())

def sort_key_firstnum(path):
    base = os.path.basename(path)
    stem, _ = os.path.splitext(base)
    gs = _extract_number_groups(stem)
    if gs:
        return (0, int(gs[0]), base.lower())
    return (1, base.lower())

def sort_key_name(path):
    return os.path.basename(path).lower()

def sort_key_mtime(path):
    try:
        return os.path.getmtime(path)
    except Exception:
        return 0.0

SORTERS = {
    "lastnum": sort_key_lastnum,
    "firstnum": sort_key_firstnum,
    "name": sort_key_name,
    "mtime": sort_key_mtime,
}

def positive_int(value):
    try:
        ivalue = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("--group must be a positive integer")
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("--group must be a positive integer")
    return ivalue

def crop_ratio_arg(value):
    try:
        fvalue = float(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("--crop_ratio must be a number")
    if not (0.0 < fvalue <= 1.0):
        raise argparse.ArgumentTypeError("--crop_ratio must be in (0, 1]")
    return fvalue

def percentile_arg(value):
    try:
        fvalue = float(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("percentile must be in [0, 100]")
    if not (0.0 <= fvalue <= 100.0):
        raise argparse.ArgumentTypeError("percentile must be in [0, 100]")
    return fvalue

def fraction_arg(value):
    try:
        fvalue = float(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("fraction must be in [0, 1]")
    if not (0.0 <= fvalue <= 1.0):
        raise argparse.ArgumentTypeError("fraction must be in [0, 1]")
    return fvalue


# Validators
def non_negative_int(value):
    try:
        ivalue = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("value must be >= 0")
    if ivalue < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return ivalue

HYBRID_LAPVAR_WEIGHT = 0.5
HYBRID_TENENGRAD_WEIGHT = 0.3
HYBRID_FFT_WEIGHT = 0.2
HYBRID_MOTION_REFERENCE = 5000.0
HYBRID_MOTION_PENALTY_WEIGHT = 0.4
MOTION_ANISO_WEIGHT = 0.5
FLOW_DOWNSCALE = 320
FLOW_MOTION_WEIGHT = 0.6
FLOW_HIGH_MOTION_THRESHOLD = 0.5
FLOW_HIGH_MOTION_RATIO = 0.4
GROUP_BRIGHTNESS_POWER = 1.5
HYBRID_DARK_THRESHOLD = 0.35
HYBRID_DARK_PENALTY_WEIGHT = 0.5
PROGRESS_INTERVAL = 5

def update_progress(label, completed, total, last_pct):
    if total <= 0:
        return last_pct
    pct = int((completed * 100) / total)
    if last_pct < 0 or pct >= 100 or pct - last_pct >= PROGRESS_INTERVAL:
        sys.stdout.write(f"{label}... {pct:3d}% ({completed}/{total})\r")
        sys.stdout.flush()
        return pct
    return last_pct


def round_half_up(value):
    """Return the value rounded to the nearest integer (half up)."""
    return int(math.floor(value + 0.5))


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def unique_path(dst_path):
    if not os.path.exists(dst_path):
        return dst_path
    base, ext = os.path.splitext(dst_path)
    k = 1
    while True:
        cand = "{}_{}{}".format(base, k, ext)
        if not os.path.exists(cand):
            return cand
        k += 1

def safe_move(src, dst):
    """Move a file safely, falling back to copy+delete on failure."""
    if not os.path.isfile(src):
        return None
    dst_final = unique_path(dst)
    ensure_dir(os.path.dirname(dst_final))
    try:
        shutil.move(src, dst_final)
        return dst_final
    except Exception:
        try:
            shutil.copy2(src, dst_final)
            try:
                os.remove(src)
            except Exception:
                pass
            return dst_final
        except Exception:
            return None

def gather_files(in_dir, ext_mode="all"):
    """
    Collect image files in the given directory without descending into subfolders.
    
    Args:
        in_dir (str): Directory to scan for image files.
        ext_mode (str): Extension key or "all" for every supported extension.
    
    Returns:
        list[str]: Absolute file paths that pass the extension filter.
    """
    target_exts = ALL_EXTS if ext_mode == "all" else EXTS[ext_mode]
    raw = []
    for name in os.listdir(in_dir):
        fp = os.path.join(in_dir, name)
        if not os.path.isfile(fp):
            continue
        _, ext = os.path.splitext(name)
        if ext.lower() in target_exts:
            raw.append(fp)

    # Deduplicate using normalized absolute paths (case-insensitive on Windows).
    seen = set()
    files = []
    for f in raw:
        key = os.path.normcase(os.path.abspath(f))
        if key in seen:
            continue
        seen.add(key)
        files.append(f)
    return files


# ---------- High-speed scoring helpers ----------

def downscale_gray(gray, max_long):
    """
    Optionally resize the grayscale frame so that the long side stays under max_long.
    
    Args:
        gray (np.ndarray): Input grayscale image.
        max_long (int): Maximum allowed long side length (0 disables scaling).
    
    Returns:
        np.ndarray: Resized (or original) grayscale image.
    """
    if not max_long or max_long <= 0:
        return gray
    h, w = gray.shape[:2]
    long_side = max(h, w)
    if long_side <= max_long:
        return gray
    scale = float(max_long) / float(long_side)
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    return cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)

def crop_by_ratio_gray(gray, crop_ratio):

    """
    Center-crop the grayscale image by the given ratio.
    
    Args:
        gray (np.ndarray): Input grayscale image.
        crop_ratio (float | None): Portion to keep (0 < ratio <= 1).
    
    Returns:
        np.ndarray: Cropped grayscale image.
    """
    if crop_ratio is None:
        return gray
    if not (0.0 < crop_ratio <= 1.0):
        raise ValueError("crop_ratio must be in (0, 1]")
    if abs(crop_ratio - 1.0) < 1e-6:
        return gray
    h, w = gray.shape[:2]
    nh = max(1, int(h * crop_ratio))
    nw = max(1, int(w * crop_ratio))
    y0 = (h - nh) // 2
    x0 = (w - nw) // 2
    return gray[y0:y0+nh, x0:x0+nw]

def lapvar32(gray):
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    # var = (std)^2
    _, std = cv2.meanStdDev(lap)
    return float(std[0,0] * std[0,0])

def tenengrad32(gray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag2 = cv2.multiply(gx, gx) + cv2.multiply(gy, gy)
    m = cv2.mean(mag2)[0]
    return float(m)

def fft_energy_fast(gray):
    """
    Estimate sharpness from the mean magnitude of high-frequency FFT components.
    
    Args:
        gray (np.ndarray): Input grayscale image.
    
    Returns:
        float: Mean magnitude of the clipped FFT spectrum.
    """
    g = downscale_gray(gray, 512)
    f = np.fft.fft2(g.astype(np.float32))
    fshift = np.fft.fftshift(f)
    h, w = g.shape
    cy, cx = h//2, w//2
    r = max(1, min(h, w) // 8)  # Reject low frequencies around the center
    # Use a donut-shaped mask to keep only high-frequency energy
    yy, xx = np.ogrid[:h, :w]
    dist2 = (yy - cy)**2 + (xx - cx)**2
    mask = (dist2 >= r*r).astype(np.float32)
    hf = fshift * mask
    return float(np.mean(np.abs(hf)))

    # Assume 8-bit input (IMREAD_GRAYSCALE).
    p0 = float(np.mean(gray <= 2))
    p255 = float(np.mean(gray >= 253))
    return p0, p255

def score_one_file(
        fp,
        metric,
        crop_ratio,
        use_exposure,
        clip_penalty,
        clip_thresh,
        max_long,
        enhance_motion,
):
    """Compute the sharpness score and ancillary metrics for one frame."""
    try:
        gray = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            return None, 0.0, 0.0, 0.0, 1.0, None, None, None, 1.0, 1.0
        gray = downscale_gray(gray, max_long)
        gray = crop_by_ratio_gray(gray, crop_ratio)

        brightness_mean = float(np.mean(gray) / 255.0)
        brightness_weight = 1.0
        lap_feature = None
        ten_feature = None
        fft_feature = None
        motion_factor = 1.0
        exposure_factor = 1.0

        if metric == "lapvar":
            lap_score = lapvar32(gray)
            sharp = lap_score
            lap_feature = lap_score * lap_score
        elif metric == "tenengrad":
            ten_score = tenengrad32(gray)
            sharp = ten_score
            ten_feature = ten_score
        elif metric == "fft":
            fft_score = fft_energy_fast(gray)
            sharp = fft_score
            fft_feature = fft_score
        elif metric == "hybrid":
            lap_score = lapvar32(gray)
            ten_score = tenengrad32(gray)
            fft_score = fft_energy_fast(gray)
            lap_energy = lap_score * lap_score
            lap_feature = lap_energy
            ten_feature = ten_score
            fft_feature = fft_score

            hybrid_raw = (
                HYBRID_LAPVAR_WEIGHT * lap_energy
                + HYBRID_TENENGRAD_WEIGHT * ten_score
                + HYBRID_FFT_WEIGHT * fft_score
            )

            motion_ratio = ten_score / (ten_score + HYBRID_MOTION_REFERENCE)
            motion_ratio = max(0.0, min(1.0, motion_ratio))
            motion_factor = 1.0 - HYBRID_MOTION_PENALTY_WEIGHT * (1.0 - motion_ratio)
            motion_factor = max(0.0, motion_factor)

            if enhance_motion:
                # Motion-sensitive enhancement is now handled during post-selection augmentation.
                pass

            if brightness_mean < HYBRID_DARK_THRESHOLD:
                dark_ratio = brightness_mean / HYBRID_DARK_THRESHOLD
            else:
                dark_ratio = 1.0
            dark_ratio = max(0.0, min(1.0, dark_ratio))
            brightness_weight = 1.0 - HYBRID_DARK_PENALTY_WEIGHT * (1.0 - dark_ratio)
            brightness_weight = max(0.0, brightness_weight)

            sharp = hybrid_raw * motion_factor
        else:
            return None, 0.0, 0.0, brightness_mean, brightness_weight, None, None, None, 1.0, 1.0

        if use_exposure:
            p0, p255 = exposure_clip_stats(gray)
            clip = p0 + p255
            if clip > clip_thresh:
                exposure_factor = clip_penalty
            sharp *= exposure_factor
        else:
            p0, p255 = 0.0, 0.0

        return (
            sharp,
            p0,
            p255,
            brightness_mean,
            brightness_weight,
            lap_feature,
            ten_feature,
            fft_feature,
            motion_factor,
            exposure_factor,
        )

    except ValueError:
        raise
    except Exception:
        return None, 0.0, 0.0, 0.0, 1.0, None, None, None, 1.0, 1.0





def _score_or_negative_infinity(scores, index):
    """Return the score for an index, falling back to negative infinity.

    Args:
        scores (list[float | None]): Sharpness scores.
        index (int): Frame index.

    Returns:
        float: Score value or negative infinity when the score is missing.
    """
    value = scores[index]
    return float(value) if value is not None else float("-inf")

def _pick_even_candidate(
    existing_indices, initial_selected, scores, used, target_pos, selected, min_diff
):
    """Pick the best candidate near the target position for even spacing.

    Args:
        existing_indices (list[int]): Indices of frames that still exist on disk.
        initial_selected (set[int]): Indices selected during grouping.
        scores (list[float | None]): Sharpness scores for all frames.
        used (set[int]): Indices already chosen for the final selection.
        target_pos (int): Desired position in the existing_indices list.

    Returns:
        int | None: The chosen index or None when nothing is left.
    """
    length = len(existing_indices)
    if length == 0:
        return None

    best = None
    best_key = None
    radius = 0
    while radius < length:
        start_pos = max(0, target_pos - radius)
        end_pos = min(length, target_pos + radius + 1)
        for pos in range(start_pos, end_pos):
            idx = existing_indices[pos]
            if idx in used:
                continue
            if min_diff > 1 and selected:
                if any(abs(idx - sel) < min_diff for sel in selected):
                    continue
            key = (
                1 if idx in initial_selected else 0,
                _score_or_negative_infinity(scores, idx),
                -abs(pos - target_pos),
                -idx,
            )
            if best_key is None or key > best_key:
                best_key = key
                best = idx
        if best_key is not None and best_key[0] == 1:
            return best
        radius += 1
    return best


def _pick_best_between(existing_indices, scores, used, start_pos, end_pos, target_pos, initial_selected, selected, min_diff):
    """Pick the best index between two positions prioritizing sharpness and proximity."""
    best_idx = None
    best_key = None
    for pos in range(start_pos + 1, end_pos):
        idx = existing_indices[pos]
        if idx in used:
            continue
        score = scores[idx]
        if score is None:
            continue
        if min_diff > 1 and selected:
            if any(abs(idx - sel) < min_diff for sel in selected):
                continue
        key = (
            1 if idx in initial_selected else 0,
            score,
            -abs(pos - target_pos),
            -idx,
        )
        if best_key is None or key > best_key:
            best_key = key
            best_idx = idx
    return best_idx



def augment_spacing(final_selected, existing_indices, scores, initial_selected, max_spacing, min_diff):
    """Augment the selection by inserting frames when spacing exceeds the limit."""
    if max_spacing is None or max_spacing <= 0:
        return set(final_selected)
    position_map = {idx: pos for pos, idx in enumerate(existing_indices)}
    augmented = set(final_selected)
    used = set(final_selected)
    changed = True
    while changed:
        changed = False
        selected_sorted = sorted(augmented)
        for left_idx, right_idx in zip(selected_sorted, selected_sorted[1:]):
            pos_left = position_map.get(left_idx)
            pos_right = position_map.get(right_idx)
            if pos_left is None or pos_right is None:
                continue
            gap = pos_right - pos_left
            if gap <= max_spacing:
                continue
            target_pos = int(round((pos_left + pos_right) / 2.0))
            candidate = _pick_best_between(
                existing_indices,
                scores,
                used,
                pos_left,
                pos_right,
                target_pos,
                initial_selected,
                selected_sorted,
                min_diff,
            )
            if candidate is None:
                continue
            augmented.add(candidate)
            used.add(candidate)
            changed = True
            break
    return augmented

def _load_flow_gray(path):
    """Load and optionally downscale a grayscale frame for flow."""
    flow_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if flow_gray is None:
        return None
    h, w = flow_gray.shape[:2]
    if FLOW_DOWNSCALE and max(h, w) > FLOW_DOWNSCALE:
        scale = FLOW_DOWNSCALE / float(max(h, w))
        flow_gray = cv2.resize(
            flow_gray,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return flow_gray



def _compute_pair_flow_magnitude(prev_path, curr_path):
    """Compute mean optical flow magnitude between two frames."""
    prev_gray = _load_flow_gray(prev_path)
    if prev_gray is None:
        return None
    curr_gray = _load_flow_gray(curr_path)
    if curr_gray is None or prev_gray.shape != curr_gray.shape:
        return None
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 1, 15, 3, 5, 1.1, 0)
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return float(np.mean(mag))



def augment_motion_segments(
    final_selected,
    group_infos,
    existing_indices,
    scores,
    flow_mag_arr,
    min_diff,
):
    """Add extra frames to high-motion groups after gap augmentation."""
    motion_values = [v for v in flow_mag_arr if v > 0.0 and np.isfinite(v)]
    if not motion_values:
        return set(final_selected)

    percentile_threshold = float(np.percentile(motion_values, 75.0))
    threshold = max(FLOW_HIGH_MOTION_THRESHOLD, percentile_threshold)
    augmented = set(final_selected)
    existing_set = set(existing_indices)
    ratio_limit = max(0.0, min(1.0, FLOW_HIGH_MOTION_RATIO))
    spacing = max(1, min_diff)

    for info in group_infos:
        start = info["start"]
        end = info["end"]
        segment_indices = [
            idx
            for idx in range(start, end)
            if idx in existing_set and scores[idx] is not None and np.isfinite(flow_mag_arr[idx])
        ]
        if not segment_indices:
            continue

        segment_motion = max(flow_mag_arr[idx] for idx in segment_indices)
        if not np.isfinite(segment_motion) or segment_motion < threshold:
            continue

        current_in_segment = [idx for idx in augmented if start <= idx < end]
        segment_span = max(1, end - start)
        spacing_limit = math.ceil(segment_span / spacing)
        available_budget = max(0, spacing_limit - len(current_in_segment))
        if available_budget <= 0:
            continue

        if ratio_limit > 0.0:
            ratio_cap = max(1, round_half_up(segment_span * ratio_limit))
            available_budget = min(available_budget, ratio_cap)
            if available_budget <= 0:
                continue

        candidates = [idx for idx in segment_indices if idx not in augmented]
        if not candidates:
            continue

        candidates.sort(
            key=lambda idx: (
                flow_mag_arr[idx],
                _score_or_negative_infinity(scores, idx),
                -idx,
            ),
            reverse=True,
        )

        added = 0
        for idx in candidates:
            if added >= available_budget:
                break
            if min_diff > 1 and any(abs(idx - sel) < min_diff for sel in augmented):
                continue
            augmented.add(idx)
            added += 1

    return augmented

def evenly_distribute_indices(existing_indices, initial_selected, scores, min_diff):
    """Return an evenly spaced selection that still prefers sharp frames.

    Args:
        existing_indices (list[int]): Indices of frames that still exist on disk.
        initial_selected (set[int]): Indices selected during grouping.
        scores (list[float | None]): Sharpness scores for all frames.

    Returns:
        set[int]: Final indices to keep after spacing adjustments.
    """
    desired_count = len(initial_selected)
    if desired_count <= 0:
        return set()
    if desired_count >= len(existing_indices):
        return set(existing_indices)

    used = set()
    selected = []
    max_pos = len(existing_indices) - 1
    step = max_pos / max(desired_count - 1, 1)

    for order in range(desired_count):
        if desired_count == 1:
            target_pos = max_pos // 2
        else:
            target_pos = int(round(order * step))
        candidate = _pick_even_candidate(
            existing_indices,
            initial_selected,
            scores,
            used,
            target_pos,
            selected,
            min_diff,
            )
        if candidate is None:
            break
        used.add(candidate)
        selected.append(candidate)

    if len(selected) < desired_count:
        remaining = [idx for idx in existing_indices if idx not in used]
        remaining.sort(
            key=lambda idx: (
                1 if idx in initial_selected else 0,
                _score_or_negative_infinity(scores, idx),
                -idx,
            ),
            reverse=True,
        )
        for idx in remaining:
            if len(selected) >= desired_count:
                break
            if min_diff > 1 and selected:
                if any(abs(idx - sel) < min_diff for sel in selected):
                    continue
            selected.append(idx)

    return set(selected)
# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser(
        description=(
            "Score frames, keep the sharp ones, and move the rest into in_dir/blur."
        )
    )
    ap.add_argument(
        "-i",
        "--in_dir",
        required=True,
        help="Input directory containing frames (non-recursive).",
    )
    ap.add_argument(
        "-g",
        "--group",
        type=positive_int,
        required=True,
        help="Number of frames per group before selection.",
    )
    ap.add_argument(
        "-e",
        "--ext",
        choices=["all", "tif", "jpg", "png"],
        default="all",
        help="File extension filter (default: all).",
    )
    ap.add_argument(
        "-s",
        "--sort",
        choices=["lastnum", "firstnum", "name", "mtime"],
        default="lastnum",
        help="Sorting rule applied before scoring.",
    )
    ap.add_argument(
        "-m",
        "--metric",
        choices=["hybrid", "lapvar", "tenengrad", "fft"],
        default="hybrid",
        help="Sharpness metric (default: hybrid 0.6*lapvar + 0.3*tenengrad + 0.2*fft).",
    )
    ap.add_argument(
        "-P",
        "--low_group_percentile",
        type=percentile_arg,
        default=30.0,
        help=(
            "Percentile threshold for detecting low-quality groups (default: 30)."
        ),
    )
    ap.add_argument(
        "-r",
        "--low_group_keep_ratio",
        type=fraction_arg,
        default=0.4,
        help=(
            "Fraction of frames to keep in low-quality groups (0 disables)."
        ),
    )
    ap.add_argument(
        "-k",
        "--low_group_min_keep",
        type=positive_int,
        default=2,
        help="Minimum number of frames to keep when a group is flagged low.",
    )
    ap.add_argument(
        "-c",
        "--crop_ratio",
        type=crop_ratio_arg,
        default=0.8,
        help="Center crop ratio used during scoring (0.8 keeps 80%%).",
    )
    ap.add_argument(
        "-S",
        "--max_spacing",
        type=non_negative_int,
        default=0,
        help="Maximum allowed gap between kept frames before backfilling (0 uses 80%% of group size).",
    )
    ap.add_argument(
        "-E",
        "--use_exposure",
        action="store_true",
        help="Apply exposure penalty when black/white clipping is detected.",
    )
    ap.add_argument(
        "-p",
        "--clip_penalty",
        type=float,
        default=0.5,
        help="Multiplier applied to scores when clipping exceeds the threshold.",
    )
    ap.add_argument(
        "-t",
        "--clip_thresh",
        type=float,
        default=0.25,
        help="Exposure clipping threshold for applying the penalty.",
    )
    ap.add_argument(
        "-M",
        "--max_long",
        type=int,
        default=0,
        help="Maximum long edge for scoring (0 keeps the original resolution).",
    )
    ap.add_argument(
        "-w",
        "--workers",
        type=int,
        help="Override the worker pool size (default: min(8, cpu or 4)).",
    )
    ap.add_argument(
        "-o",
        "--opencv_threads",
        type=int,
        default=0,
        help="Set OpenCV thread count (0 leaves the default).",
    )
    ap.add_argument(
        "-n",
        "--dry_run",
        action="store_true",
        help="Perform scoring and selection without moving files.",
    )
    ap.add_argument(
        "--enhance_motion",
        action="store_true",
        help="Apply additional motion blur penalty during hybrid scoring.",
    )
    ap.add_argument(
        "--allow_initial_adjacent",
        action="store_true",
        help="Allow adjacent frames during initial per-group selection.",
    )
    ap.add_argument(
        "-C",
        "--csv",
        help="Optional CSV output path relative to the input directory.",
    )

    args = ap.parse_args()


    # Keep OpenCV from competing with the Python thread pool.
    try:
        if args.opencv_threads and args.opencv_threads > 0:
            cv2.setNumThreads(args.opencv_threads)
    except Exception:
        pass

    files = gather_files(args.in_dir, args.ext)
    if not files:
        print(f"No input images found: {args.in_dir}")
        sys.exit(1)

    if args.max_spacing <= 0:
        args.max_spacing = round_half_up(args.group * 0.8)

    min_spacing_raw = round_half_up(args.group * 0.2)
    if min_spacing_raw <= 1:
        min_diff = 2
    else:
        min_diff = min_spacing_raw + 1

    sorter = SORTERS[args.sort]
    files = sorted(files, key=sorter)

    blur_dir = os.path.join(args.in_dir, "blur")
    ensure_dir(blur_dir)

    # Score every file in parallel
    n = len(files)
    scores = [None] * n
    p0_arr = [0.0] * n
    p255_arr = [0.0] * n
    brightness_arr = [1.0] * n
    brightness_mean_arr = [0.0] * n
    lap_arr = [None] * n
    ten_arr = [None] * n
    fft_arr = [None] * n
    motion_arr = [1.0] * n
    exposure_arr = [1.0] * n
    group_score_arr = [0.0] * n
    flow_mag_arr = [0.0] * n

    workers = (
        args.workers
        if (args.workers and args.workers > 0)
        else min(8, (os.cpu_count() or 4))
    )
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(
                score_one_file, files[i],
                args.metric, args.crop_ratio,
                args.use_exposure, args.clip_penalty, args.clip_thresh,
                args.max_long, args.enhance_motion
            ): i for i in range(n)
        }
        completed = 0
        last_pct = -1
        for fut in as_completed(futs):
            i = futs[fut]
            (
                s,
                p0,
                p255,
                brightness_mean,
                brightness_weight,
                lap_feature,
                ten_feature,
                fft_feature,
                motion_factor,
                exposure_factor,
            ) = fut.result()
            scores[i] = s
            p0_arr[i] = p0
            p255_arr[i] = p255
            brightness_mean_arr[i] = brightness_mean
            brightness_arr[i] = brightness_weight
            lap_arr[i] = lap_feature
            ten_arr[i] = ten_feature
            fft_arr[i] = fft_feature
            motion_arr[i] = motion_factor
            exposure_arr[i] = exposure_factor
            completed += 1
            last_pct = update_progress("Scoring", completed, n, last_pct)



    if args.enhance_motion and n > 1:
        flow_pairs = []
        prev_idx = None
        for idx in range(n):
            fp = files[idx]
            if not os.path.isfile(fp):
                prev_idx = None
                continue
            if prev_idx is not None:
                flow_pairs.append((prev_idx, idx))
            prev_idx = idx
        if flow_pairs:
            with ThreadPoolExecutor(max_workers=workers) as flow_executor:
                flow_futs = {
                    flow_executor.submit(
                        _compute_pair_flow_magnitude,
                        files[left_idx],
                        files[right_idx],
                    ): (left_idx, right_idx)
                    for left_idx, right_idx in flow_pairs
                }
                for fut in as_completed(flow_futs):
                    left_idx, right_idx = flow_futs[fut]
                    try:
                        mean_mag = fut.result()
                    except Exception:
                        mean_mag = None
                    if mean_mag is None:
                        continue
                    flow_mag_arr[right_idx] = max(flow_mag_arr[right_idx], mean_mag)
                    flow_mag_arr[left_idx] = max(flow_mag_arr[left_idx], mean_mag)


    if args.metric == "hybrid":
        lap_values = np.array([v for v in lap_arr if v is not None], dtype=np.float64)
        ten_values = np.array([v for v in ten_arr if v is not None], dtype=np.float64)
        fft_values = np.array([v for v in fft_arr if v is not None], dtype=np.float64)

        lap_values = [v for v in lap_arr if v is not None]
        ten_values = [v for v in ten_arr if v is not None]
        fft_values = [v for v in fft_arr if v is not None]

        def _normalize_feature(values, value):
            if not values or value is None:
                return 0.0
            vmin = min(values)
            vmax = max(values)
            if math.isclose(vmax, vmin):
                return 0.0
            return (value - vmin) / (vmax - vmin)

        for idx in range(n):
            if lap_arr[idx] is None:
                continue
            lap_norm = _normalize_feature(lap_values, lap_arr[idx])
            ten_norm = _normalize_feature(ten_values, ten_arr[idx])
            fft_norm = _normalize_feature(fft_values, fft_arr[idx])

            combined = (
                HYBRID_LAPVAR_WEIGHT * lap_norm
                + HYBRID_TENENGRAD_WEIGHT * ten_norm
                + HYBRID_FFT_WEIGHT * fft_norm
            )

            combined *= motion_arr[idx]
            combined *= exposure_arr[idx]
            scores[idx] = combined
    if n:
        print(f"Scoring... 100% ({n}/{n})")
    # Prepare optional CSV output
    csv_writer = None
    fcsv = None
    if args.csv:
        csv_path = (
            args.csv
            if os.path.isabs(args.csv)
            else os.path.join(args.in_dir, args.csv)
        )
        fcsv = open(csv_path, "w", newline="")
        csv_writer = csv.writer(fcsv)
        csv_writer.writerow(
            [
                "index",
                "filename",
                "score",
                "brightness_mean",
                "group_score",
                "flow_motion",
                "selected(1=keep)",
            ]
        )

    # Grouping and relocation
    total = n
    group_infos = []
    for grp_start in range(0, total, args.group):
        grp_end = min(total, grp_start + args.group)
        valid_idx = []
        group_sum = 0.0
        for i in range(grp_start, grp_end):
            s = scores[i]
            if s is None:
                continue
            valid_idx.append(i)
            if s > 0.0:
                brightness_factor = brightness_arr[i] * (max(brightness_mean_arr[i], 1e-6) ** GROUP_BRIGHTNESS_POWER)
                group_sum += s * brightness_factor
        for idx_in_group in range(grp_start, grp_end):
            group_score_arr[idx_in_group] = group_sum
        group_infos.append(
            {
                "start": grp_start,
                "end": grp_end,
                "valid_idx": valid_idx,
                "group_sum": group_sum,
            }
        )

    group_sums = [info["group_sum"] for info in group_infos if info["valid_idx"]]
    low_threshold = None
    if args.low_group_keep_ratio > 0.0:
        if group_sums:
            low_threshold = float(
                np.percentile(group_sums, args.low_group_percentile)
            )
        else:
            low_threshold = 0.0

    initial_selected = set()
    for info in group_infos:
        grp_start = info["start"]
        grp_end = info["end"]
        group_range = range(grp_start, grp_end)
        existing_indices = [
            i for i in group_range if os.path.isfile(files[i])
        ]
        valid_indices = [
            i for i in existing_indices if scores[i] is not None
        ]

        is_low_group = False
        if args.low_group_keep_ratio > 0.0:
            if not valid_indices:
                is_low_group = True
            elif (
                low_threshold is None
                or info["group_sum"] <= low_threshold
            ):
                is_low_group = True

        if not valid_indices:
            selected_indices = set(existing_indices)
        else:
            sorted_valid = sorted(
                valid_indices,
                key=lambda idx: (scores[idx], idx),
                reverse=True,
            )
            keep_count = 1
            if is_low_group:
                raw_keep = round_half_up(
                    len(sorted_valid) * args.low_group_keep_ratio
                )
                desired = max(args.low_group_min_keep, raw_keep)
                keep_count = min(len(sorted_valid), max(1, desired))

            chosen = []
            if args.allow_initial_adjacent:
                chosen = sorted_valid[:keep_count]
            else:
                for idx in sorted_valid:
                    if not chosen:
                        chosen.append(idx)
                    elif min_diff <= 1 or all(abs(idx - sel) >= min_diff for sel in chosen):
                        chosen.append(idx)
                    if len(chosen) >= keep_count:
                        break
                if len(chosen) < keep_count:
                    for idx in sorted_valid:
                        if idx not in chosen:
                            chosen.append(idx)
                        if len(chosen) >= keep_count:
                            break
            selected_indices = set(chosen)

        initial_selected.update(selected_indices)

    existing_indices = [
        i for i in range(total) if os.path.isfile(files[i])
    ]
    initial_selected &= set(existing_indices)
    final_selected = evenly_distribute_indices(
        existing_indices,
        initial_selected,
        scores,
        min_diff,
    )
    if not final_selected and initial_selected:
        final_selected = set(initial_selected)

    final_selected = augment_spacing(
        final_selected,
        existing_indices,
        scores,
        initial_selected,
        args.max_spacing,
        min_diff,
    )

    if args.enhance_motion:
        final_selected = augment_motion_segments(
            final_selected,
            group_infos,
            existing_indices,
            scores,
            flow_mag_arr,
            min_diff,
        )

    kept = 0
    moved = 0
    skipped = 0
    processed = 0
    last_group_pct = -1

    for i in range(total):
        s = scores[i]
        processed += 1
        file_exists = os.path.isfile(files[i])
        if not file_exists or s is None:
            skipped += 1
            if csv_writer:
                csv_writer.writerow([
                    i,
                    os.path.basename(files[i]),
                    -1.0,
                    0.0,
                    group_score_arr[i],
                    flow_mag_arr[i],
                    0,
                ])
            last_group_pct = update_progress("Grouping", processed, total, last_group_pct)
            continue

        score_val = s if s is not None else 0.0
        mean_val = brightness_mean_arr[i]

        if i in final_selected:
            kept += 1
            if csv_writer:
                csv_writer.writerow([
                    i,
                    os.path.basename(files[i]),
                    score_val,
                    mean_val,
                    group_score_arr[i],
                    flow_mag_arr[i],
                    1,
                ])
        else:
            if args.dry_run:
                moved += 1
            else:
                dst = os.path.join(blur_dir, os.path.basename(files[i]))
                if safe_move(files[i], dst) is None:
                    skipped += 1
                else:
                    moved += 1
            if csv_writer:
                csv_writer.writerow([
                    i,
                    os.path.basename(files[i]),
                    score_val,
                    mean_val,
                    group_score_arr[i],
                    flow_mag_arr[i],
                    0,
                ])
        last_group_pct = update_progress("Grouping", processed, total, last_group_pct)
    if total:
        print(f"Grouping... 100% ({total}/{total})")
    if fcsv:
        fcsv.close()

    print(f"Done: input {total} / kept {kept} / moved {moved} / skipped {skipped}")
    if args.dry_run:
        print("Blur directory (dry run, no files moved):", blur_dir)
    else:
        print("Blur directory:", blur_dir)
    print(
        f"workers={workers}, opencv_threads={args.opencv_threads}, "
        f"max_long={args.max_long}"
    )


if __name__ == "__main__":
    main()
