#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gs360_SegmentationMaskTool.py
  - Generate target masks with Python 3.7 / torchvision Mask R-CNN.
  - Inputs: image folder only (non-recursive, selected via -i/--in).
  - Output modes:
      mask        : Binary mask PNG (person=black, background=white) -> <source>.png
      alpha       : RGBA cutout PNG (person=opaque, background=transparent) -> <source>.png
      cutout      : RGBA cutout PNG (person only, filename <source>_cutout.png)
      keep_person : Color image preserving selected targets and filling the background with black.
      remove_person: Paint selected targets black to remove them.
      inpaint     : Replace selected target regions with classical inpainting (OpenCV Telea).
  - Default output location: automatically create _mask/ next to the input folder.
  - Performance settings (score, morphology, resize) and shadow parameters are fixed constants.
"""

import argparse
import sys
import re
import warnings
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import torch
from torchvision import transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn

try:
    # Available in torch 1.10+ and only useful with GPU inference.
    from torch.cuda.amp import autocast
except Exception:
    autocast = None

# Silence torch.meshgrid future warning triggered by torchvision internals.
warnings.filterwarnings(
    "ignore",
    message="torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument.",
    category=UserWarning,
)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

MIN_SIZE = 640
MAX_SIZE = 1024
RPN_PRE_NMS = 400
RPN_POST_NMS = 100
DETECTIONS_PER_IMG = 15
USE_FP16 = False
WARMUP_ITERS = 1
SCORE_THRESH = 0.7
MASK_THRESH = 0.5
CLOSE_KERNEL = 5
DEFAULT_MASK_EXPAND_MODE = "pixels"
DEFAULT_MASK_EXPAND_PIXELS = 15
DEFAULT_MASK_EXPAND_PERCENT = 1.0
DEFAULT_EDGE_FUSE_PIXELS = 25
MASK_EXPAND_MODE_CHOICES = ["pixels", "percent"]
SHADOW_T = 0.7
SHADOW_SIGMA = 15
SHADOW_NEAR = 25
SHADOW_MIN_AREA = 500
PROGRESS_INTERVAL = 5
INPAINT_RADIUS = 5
TARGET_TO_COCO_LABELS = {
    "person": [1],
    "bicycle": [2],
    "car": [3],
    "animal": [16, 17, 18],  # bird, cat, dog
}
TARGET_CHOICES = ["person", "bicycle", "car", "animal"]


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTS


def update_progress(label: str, completed: int, total: int, last_pct: int) -> int:
    if total <= 0:
        return last_pct
    pct = int((completed * 100) / total)
    if last_pct < 0 or pct >= 100 or (pct - last_pct) >= PROGRESS_INTERVAL:
        sys.stdout.write(f"{label}... {pct:3d}% ({completed}/{total})\r")
        sys.stdout.flush()
        return pct
    return last_pct


def load_model(device, conf_thres: float = SCORE_THRESH):
    model = maskrcnn_resnet50_fpn(pretrained=True).to(device)
    model.eval()

    # Input resolution.
    try:
        model.transform.min_size = (int(MIN_SIZE),)
        model.transform.max_size = int(MAX_SIZE)
    except Exception:
        pass

    # Score threshold for early filtering.
    try:
        model.roi_heads.score_thresh = float(conf_thres)
    except Exception:
        pass

    # Configure RPN top-N values safely (compatibility fix).
    _set_rpn_topn_compat(model.rpn, RPN_PRE_NMS, RPN_POST_NMS)

    # Final detections per image.
    try:
        model.roi_heads.detections_per_img = int(DETECTIONS_PER_IMG)
    except Exception:
        pass

    return model


def _set_rpn_topn_compat(rpn, pre: int, post: int) -> bool:
    """
    Safely set the RPN pre/post NMS top-N values across torchvision versions.
    Returns:
        True if the assignment succeeded.
    """
    pre = int(pre); post = int(post)

    # case A: Preferred path with internal _pre_nms_top_n / _post_nms_top_n dictionaries.
    try:
        if hasattr(rpn, "_pre_nms_top_n") and isinstance(rpn._pre_nms_top_n, dict):
            rpn._pre_nms_top_n = {"training": pre, "testing": pre}
        if hasattr(rpn, "_post_nms_top_n") and isinstance(rpn._post_nms_top_n, dict):
            rpn._post_nms_top_n = {"training": post, "testing": post}
        return True
    except Exception:
        pass

    # case B: Fallback for dict attributes (do not override the pre_nms_top_n() method).
    try:
        p = getattr(rpn, "pre_nms_top_n", None)
        q = getattr(rpn, "post_nms_top_n", None)
        if isinstance(p, dict):
            p["training"] = pre; p["testing"] = pre
        if isinstance(q, dict):
            q["training"] = post; q["testing"] = post
        if isinstance(p, dict) or isinstance(q, dict):
            return True
    except Exception:
        pass

    # case C: Legacy attributes *_train / *_test.
    try:
        rpn.pre_nms_top_n_train  = pre
        rpn.pre_nms_top_n_test   = pre
        rpn.post_nms_top_n_train = post
        rpn.post_nms_top_n_test  = post
        return True
    except Exception:
        pass

    return False

def target_mask_from_prediction(pred, targets,
                                score_thres=SCORE_THRESH, mask_thres=MASK_THRESH):
    """
    pred: Dictionary produced by model(x)[0].
    return: HxW (uint8) mask with selected target pixels = 255.
    """
    labels = pred["labels"].detach().cpu().numpy()
    scores = pred["scores"].detach().cpu().numpy()
    masks = pred["masks"].detach().cpu().numpy()         # (N,1,H,W) float [0..1]

    target_ids = set()
    for name in targets:
        target_ids.update(TARGET_TO_COCO_LABELS.get(name, []))
    if not target_ids:
        return None

    keep = np.isin(labels, list(target_ids)) & (scores >= score_thres)
    if keep.sum() == 0:
        return None

    ms = masks[keep, 0, ...] > mask_thres               # (K,H,W) bool
    combined = np.any(ms, axis=0).astype(np.uint8) * 255
    return combined


def resolve_mask_expand_pixels(expand_mode=DEFAULT_MASK_EXPAND_MODE,
                               expand_pixels=DEFAULT_MASK_EXPAND_PIXELS,
                               expand_percent=DEFAULT_MASK_EXPAND_PERCENT,
                               image_shape=None):
    """
    Resolve the requested mask expansion amount to pixels.

    Args:
        expand_mode: Expansion mode, either "pixels" or "percent".
        expand_pixels: Expansion amount in pixels.
        expand_percent: Expansion amount as a percentage of the longer image edge.
        image_shape: Optional (height, width) tuple used for percentage mode.
    """
    mode = str(expand_mode or DEFAULT_MASK_EXPAND_MODE).strip().lower()
    if mode == "pixels":
        return max(0, int(round(float(expand_pixels))))
    if mode == "percent":
        if image_shape is None or len(image_shape) < 2:
            return 0
        base_len = max(int(image_shape[0]), int(image_shape[1]))
        px = int(round(base_len * (float(expand_percent) / 100.0)))
        return max(0, px)
    raise ValueError("Unsupported mask expand mode: {}".format(expand_mode))


def refine_mask(mask, close=CLOSE_KERNEL,
                expand_mode=DEFAULT_MASK_EXPAND_MODE,
                expand_pixels=DEFAULT_MASK_EXPAND_PIXELS,
                expand_percent=DEFAULT_MASK_EXPAND_PERCENT,
                image_shape=None):
    """
    Morphological post-processing (hole filling / edge reinforcement).

    Args:
        mask: Binary mask array.
        close: Kernel size in pixels for the closing operation.
        expand_mode: Expansion mode, "pixels" or "percent".
        expand_pixels: Expansion amount in pixels.
        expand_percent: Expansion amount as a percentage of the longer image edge.
        image_shape: Optional (height, width) tuple for percentage sizing.
    """
    if mask is None:
        return None
    k = max(1, int(close))
    if k > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return expand_mask(mask,
                       expand_mode=expand_mode,
                       expand_pixels=expand_pixels,
                       expand_percent=expand_percent,
                       image_shape=image_shape)


def expand_mask(mask,
                expand_mode=DEFAULT_MASK_EXPAND_MODE,
                expand_pixels=DEFAULT_MASK_EXPAND_PIXELS,
                expand_percent=DEFAULT_MASK_EXPAND_PERCENT,
                image_shape=None):
    """Expand a binary mask using the configured pixels or percentage mode."""
    if mask is None:
        return None
    if image_shape is None or len(image_shape) < 2:
        image_shape = mask.shape
    expand_px = resolve_mask_expand_pixels(
        expand_mode=expand_mode,
        expand_pixels=expand_pixels,
        expand_percent=expand_percent,
        image_shape=image_shape,
    )
    if expand_px > 0:
        kernel_size = max(1, (expand_px * 2) + 1)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def fuse_mask_to_edges(mask,
                       edge_fuse_pixels=DEFAULT_EDGE_FUSE_PIXELS):
    """Fuse only mask pixels already near an edge, then extend them to that edge."""
    if mask is None:
        return None
    fuse_px = max(0, int(edge_fuse_pixels))
    if fuse_px <= 0:
        return mask
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    if not np.any(binary):
        return mask
    height, width = binary.shape[:2]
    result = binary.copy()
    spread_px = max(1, int(round(float(fuse_px) * 0.35)))

    top_seed = binary[:fuse_px, :].copy()
    bottom_seed = binary[height - fuse_px:, :].copy()
    left_seed = binary[:, :fuse_px].copy()
    right_seed = binary[:, width - fuse_px:].copy()

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        ((spread_px * 2) + 1, 1),
    )
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (1, (spread_px * 2) + 1),
    )

    top_seed = cv2.dilate(top_seed, horizontal_kernel, iterations=1)
    bottom_seed = cv2.dilate(bottom_seed, horizontal_kernel, iterations=1)
    left_seed = cv2.dilate(left_seed, vertical_kernel, iterations=1)
    right_seed = cv2.dilate(right_seed, vertical_kernel, iterations=1)

    for x_pos in np.where(np.any(top_seed > 0, axis=0))[0]:
        y_values = np.where(top_seed[:, int(x_pos)] > 0)[0]
        if y_values.size > 0:
            fill_end = int(y_values.min())
            result[:fill_end + 1, int(x_pos)] = 255

    for x_pos in np.where(np.any(bottom_seed > 0, axis=0))[0]:
        y_values = np.where(bottom_seed[:, int(x_pos)] > 0)[0]
        if y_values.size > 0:
            fill_start = int((height - fuse_px) + y_values.max())
            result[fill_start:, int(x_pos)] = 255

    for y_pos in np.where(np.any(left_seed > 0, axis=1))[0]:
        x_values = np.where(left_seed[int(y_pos), :] > 0)[0]
        if x_values.size > 0:
            fill_end = int(x_values.min())
            result[int(y_pos), :fill_end + 1] = 255

    for y_pos in np.where(np.any(right_seed > 0, axis=1))[0]:
        x_values = np.where(right_seed[int(y_pos), :] > 0)[0]
        if x_values.size > 0:
            fill_start = int((width - fuse_px) + x_values.max())
            result[int(y_pos), fill_start:] = 255
    return result


def estimate_shadow_mask(img_rgb, person_mask,
                         t=0.7, sigma=15, near_px=25, min_area=500):
    """
    Extract shadow candidates near the person mask (0/255 mask).
    - ratio = gray / (blurred gray) < t to detect darker regions.
    - Keep only pixels with low saturation.
    - Limit to areas near a dilated person mask.
    """
    if person_mask is None:
        return None

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    illum = cv2.GaussianBlur(gray, (0, 0), sigma)
    ratio = gray / (illum + 1e-6)
    shadow = (ratio < t).astype(np.uint8) * 255

    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    low_sat = (hsv[:, :, 1] < 80).astype(np.uint8) * 255
    shadow = cv2.bitwise_and(shadow, low_sat)

    k = max(3, int(near_px) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    near = cv2.dilate(person_mask, kernel, iterations=1)
    shadow = cv2.bitwise_and(shadow, near)

    cnts, _ = cv2.findContours(shadow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean = np.zeros_like(shadow)
    th = max(1, int(min_area))
    for c in cnts:
        if cv2.contourArea(c) >= th:
            cv2.drawContours(clean, [c], -1, 255, -1)
    return clean


def save_mask_png(mask, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(str(out_path))


def extract_multicam_view_id(stem: str):
    """Extract trailing view ID token (e.g. A, A_U, A_D20) from a file stem."""
    pattern = r"_((?:[A-Z]|\d{2,})(?:_(?:U|D|U\d+|D\d+))?)$"
    match = re.search(pattern, stem.upper())
    if not match:
        return None
    return match.group(1)


def manual_mask_key_for_path(in_path: Path):
    """Build the shared manual-mask key for a given image path."""
    view_id = extract_multicam_view_id(in_path.stem)
    if view_id:
        return "view__{}".format(view_id)
    return "file__{}".format(in_path.stem)


def load_manual_mask_layer(mask_path: Path, image_size=None):
    """Load a manual add/erase layer PNG saved as 8-bit binary."""
    if not mask_path.exists():
        return None
    mask_img = Image.open(str(mask_path)).convert("L")
    if image_size is not None and mask_img.size != image_size:
        mask_img = mask_img.resize(image_size, Image.NEAREST)
    mask = np.array(mask_img, dtype=np.uint8)
    return np.where(mask > 127, 255, 0).astype(np.uint8)


def load_manual_mask_layers(in_path: Path,
                            manual_mask_dir: Path = None,
                            image_size=None):
    """Load a shared manual add layer for the matching view/file key."""
    if manual_mask_dir is None:
        return None
    manual_key = manual_mask_key_for_path(in_path)
    add_mask = load_manual_mask_layer(
        manual_mask_dir / "{}__add.png".format(manual_key),
        image_size=image_size,
    )
    return add_mask


def apply_manual_mask_layers(mask,
                             add_mask=None,
                             image_shape=None):
    """Apply a shared manual add layer after auto mask estimation."""
    if mask is None:
        if add_mask is None:
            return None
        if image_shape is None:
            raise ValueError("image_shape is required when base mask is None")
        mask = np.zeros(image_shape, dtype=np.uint8)
    else:
        mask = np.where(mask > 0, 255, 0).astype(np.uint8)
    if add_mask is not None:
        mask[add_mask > 0] = 255
    if not np.any(mask):
        return None
    return mask


def save_cutout_rgba(img_rgb, mask, out_path: Path):
    """
    Cut out the person with transparency (RGBA, alpha=mask).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = mask.shape
    if img_rgb.shape[:2] != (h, w):
        img_rgb = cv2.resize(img_rgb, (w, h), interpolation=cv2.INTER_CUBIC)
    rgba = np.dstack([img_rgb, mask])
    Image.fromarray(rgba).save(str(out_path))


def apply_mode(img_rgb, mask, mode: str):
    """
    mode:
      - mask: mask only (H,W) uint8
      - cutout: transparent PNG containing only the person
      - keep_person: keep the person in color with a black background
      - remove_person: paint the person black
      - inpaint: fill the person region with inpainted background
    """
    if mode == "mask":
        return mask

    h, w = mask.shape
    if img_rgb.shape[:2] != (h, w):
        img_rgb = cv2.resize(img_rgb, (w, h), interpolation=cv2.INTER_CUBIC)

    m_bool = (mask > 0)
    if mode == "keep_person":
        out = np.zeros_like(img_rgb)
        out[m_bool] = img_rgb[m_bool]
        return out
    elif mode == "remove_person":
        out = img_rgb.copy()
        out[m_bool] = 0
        return out
    elif mode == "inpaint":
        if not np.any(m_bool):
            return img_rgb
        inpaint_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        inpaint_mask = (m_bool.astype(np.uint8)) * 255
        filled = cv2.inpaint(inpaint_bgr, inpaint_mask, INPAINT_RADIUS, cv2.INPAINT_TELEA)
        return cv2.cvtColor(filled, cv2.COLOR_BGR2RGB)
    else:
        return img_rgb


def process_image(model, device, in_path: Path, out_dir: Path,
                  score_thres: float = SCORE_THRESH, mask_thres: float = MASK_THRESH,
                  close: int = CLOSE_KERNEL, mode: str = "mask",
                  include_shadow: bool = False,
                  mask_expand_mode: str = DEFAULT_MASK_EXPAND_MODE,
                  mask_expand_pixels: int = DEFAULT_MASK_EXPAND_PIXELS,
                  mask_expand_percent: float = DEFAULT_MASK_EXPAND_PERCENT,
                  edge_fuse_pixels: int = DEFAULT_EDGE_FUSE_PIXELS,
                  manual_mask_dir: Path = None,
                  targets=None):
    if targets is None:
        targets = ["person"]

    img = Image.open(str(in_path)).convert("RGB")
    t = transforms.ToTensor()(img).to(device)

    # Inference.
    with torch.no_grad():
        if device.type == "cuda" and USE_FP16 and autocast is not None:
            with autocast():
                pred = model([t])[0]
        else:
            pred = model([t])[0]

    # Selected target mask (target=255, background=0).
    mask = target_mask_from_prediction(pred, targets, score_thres, mask_thres)
    mask = refine_mask(mask,
                       close=close,
                       expand_mode="pixels",
                       expand_pixels=0,
                       expand_percent=0.0,
                       image_shape=(img.size[1], img.size[0]))

    # Merge shadow mask (optional).
    if include_shadow:
        shadow = estimate_shadow_mask(np.array(img),
                                      mask,
                                      t=SHADOW_T,
                                      sigma=SHADOW_SIGMA,
                                      near_px=SHADOW_NEAR,
                                      min_area=SHADOW_MIN_AREA)
        if shadow is not None:
            base = np.zeros_like(shadow) if mask is None else mask
            mask = np.maximum(base, shadow)
    mask = expand_mask(mask,
                       expand_mode=mask_expand_mode,
                       expand_pixels=mask_expand_pixels,
                       expand_percent=mask_expand_percent,
                       image_shape=(img.size[1], img.size[0]))
    mask = fuse_mask_to_edges(
        mask,
        edge_fuse_pixels=edge_fuse_pixels,
    )
    add_mask = load_manual_mask_layers(
        in_path,
        manual_mask_dir=manual_mask_dir,
        image_size=img.size,
    )
    mask = apply_manual_mask_layers(
        mask,
        add_mask=add_mask,
        image_shape=(img.size[1], img.size[0]),
    )

    stem = in_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    if mode == "alpha":
        # Save RGBA as <source>.png.
        if mask is None:
            w, h = img.size
            alpha = np.zeros((h, w), np.uint8)
            out_path = out_dir / f"{stem}.png"
            rgba = np.dstack([np.array(img), alpha])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgba).save(str(out_path))
        else:
            mask = 255 - mask
            save_cutout_rgba(np.array(img), mask, out_dir / f"{stem}.png")

    elif mode == "cutout":
        if mask is None:
            w, h = img.size
            alpha = np.zeros((h, w), np.uint8)
            rgba = np.dstack([np.array(img), alpha])
            Image.fromarray(rgba).save(str(out_dir / f"{stem}_cutout.png"))
        else:
            save_cutout_rgba(np.array(img), mask, out_dir / f"{stem}_cutout.png")

    elif mode == "mask":
        # Invert: person=black (0), background=white (255) -> <source>.png.
        if mask is None:
            mask = np.zeros((img.size[1], img.size[0]), np.uint8)
        # Ensure dimensions match.
        h, w = img.size[1], img.size[0]
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        mask = 255 - mask
        save_mask_png(mask, out_dir / f"{stem}.png")

    else:
        # keep_person / remove_person
        if mask is None:
            out = np.array(img)
        else:
            out = apply_mode(np.array(img), mask, mode)
        Image.fromarray(out).save(str(out_dir / f"{stem}_{mode}.png"))


def collect_images(in_path: Path):
    if in_path.is_file():
        return [in_path]
    files = []
    for p in sorted(in_path.iterdir()):
        if p.is_file() and is_image(p):
            files.append(p)
    return files


def main():
    ap = argparse.ArgumentParser(
        description="Tool to generate segmentation masks with Mask R-CNN (COCO)."
    )
    ap.add_argument("-i", "--in", dest="input_dir", type=str, required=True,
                    help="Input directory containing images (non-recursive)")
    ap.add_argument(
        "-o",
        "--out",
        type=str,
        default=None,
        help="Output directory (default: create _mask/ next to input folder)",
    )
    ap.add_argument("--mode", type=str, default="mask",
                    choices=["mask", "alpha", "cutout", "keep_person", "remove_person", "inpaint"],
                    help="Output mode")

    ap.add_argument("--cpu", action="store_true", help="Force CPU")
    ap.add_argument(
        "--target",
        action="append",
        dest="targets",
        choices=TARGET_CHOICES,
        default=None,
        help="Target category to mask (repeatable). default: person",
    )
    ap.add_argument("--include_shadow", action="store_true",
                    help="Include adjacent shadows in the mask")
    ap.add_argument(
        "--mask-expand-mode",
        type=str,
        choices=MASK_EXPAND_MODE_CHOICES,
        default=DEFAULT_MASK_EXPAND_MODE,
        help="Mask expansion mode: pixels or percent of the longer image edge",
    )
    ap.add_argument(
        "--mask-expand-pixels",
        type=int,
        default=DEFAULT_MASK_EXPAND_PIXELS,
        help="Mask expansion amount in pixels when --mask-expand-mode=pixels",
    )
    ap.add_argument(
        "--mask-expand-percent",
        type=float,
        default=DEFAULT_MASK_EXPAND_PERCENT,
        help="Mask expansion amount in percent when --mask-expand-mode=percent",
    )
    ap.add_argument(
        "--edge-fuse-pixels",
        type=int,
        default=DEFAULT_EDGE_FUSE_PIXELS,
        help=(
            "If a mask component is within this many pixels of an image edge, "
            "extend it to the frame edge after mask expand and before manual paint"
        ),
    )
    ap.add_argument(
        "--manual-mask-dir",
        type=str,
        default=None,
        help=(
            "Optional directory containing shared manual add PNG layers "
            "named <view_or_file_key>__add.png"
        ),
    )
    args = ap.parse_args()

    if args.mask_expand_pixels < 0:
        ap.error("--mask-expand-pixels must be 0 or greater")
    if args.mask_expand_percent < 0:
        ap.error("--mask-expand-percent must be 0 or greater")
    if args.edge_fuse_pixels < 0:
        ap.error("--edge-fuse-pixels must be 0 or greater")

    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    if args.out is None:
        parent_dir = input_dir.parent
        out_dir = (input_dir / "_mask") if parent_dir == input_dir else (
            parent_dir / "_mask"
        )
    else:
        out_dir = Path(args.out)
    manual_mask_dir = None
    if args.manual_mask_dir:
        manual_mask_dir = Path(args.manual_mask_dir)
        if not manual_mask_dir.exists() or not manual_mask_dir.is_dir():
            print(
                "Manual mask directory not found: {}".format(manual_mask_dir),
                file=sys.stderr,
            )
            sys.exit(1)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Device: {device}")
    targets = args.targets if args.targets else ["person"]
    deduped_targets = []
    for target_name in targets:
        if target_name not in deduped_targets:
            deduped_targets.append(target_name)
    targets = deduped_targets
    print(f"Targets: {', '.join(targets)}")
    if args.mask_expand_mode == "pixels":
        print(f"Mask expand: {args.mask_expand_pixels} px")
    else:
        print(f"Mask expand: {args.mask_expand_percent}%")
    print(f"Edge fuse: {args.edge_fuse_pixels} px")
    if manual_mask_dir is not None:
        print("Manual mask dir: {}".format(manual_mask_dir))

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    model = load_model(device, conf_thres=SCORE_THRESH)

    # Optional warm-up.
    if device.type == "cuda" and WARMUP_ITERS > 0:
        dummy = torch.zeros(3, MIN_SIZE, MIN_SIZE, dtype=torch.float32)
        with torch.no_grad():
            for _ in range(WARMUP_ITERS):
                if USE_FP16 and autocast is not None:
                    with autocast():
                        _ = model([dummy.to(device)])
                else:
                    _ = model([dummy.to(device)])

    images = collect_images(input_dir)

    if len(images) == 0:
        print("No images found.", file=sys.stderr)
        sys.exit(1)

    last_pct = -1
    total = len(images)
    for idx, p in enumerate(images, 1):
        last_pct = update_progress("Processing", idx, total, last_pct)
        process_image(model, device, p, out_dir,
                      score_thres=SCORE_THRESH, mask_thres=MASK_THRESH,
                      close=CLOSE_KERNEL,
                      mode=args.mode,
                      include_shadow=args.include_shadow,
                      mask_expand_mode=args.mask_expand_mode,
                      mask_expand_pixels=args.mask_expand_pixels,
                      mask_expand_percent=args.mask_expand_percent,
                      edge_fuse_pixels=args.edge_fuse_pixels,
                      manual_mask_dir=manual_mask_dir,
                      targets=targets)
    if total > 0:
        update_progress("Processing", total, total, last_pct)
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
