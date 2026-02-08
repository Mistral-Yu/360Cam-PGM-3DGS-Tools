#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gs360_HumanMaskTool.py
  - Generate target masks with Python 3.7 / torchvision Mask R-CNN.
  - Inputs: image folder only (non-recursive, selected via -i/--in).
  - Output modes:
      mask        : Binary mask PNG (person=black, background=white) -> <source>.png
      alpha       : RGBA cutout PNG (person=opaque, background=transparent) -> <source>.png
      cutout      : RGBA cutout PNG (person only, filename <source>_cutout.png)
      keep_person : Color image preserving selected targets and filling the background with black.
      remove_person: Paint selected targets black to remove them.
      inpaint     : Replace selected target regions with classical inpainting (OpenCV Telea).
  - Default output location: automatically create masks/ directly under the input.
  - Performance settings (score, morphology, resize) and shadow parameters are fixed constants.
"""

import argparse
import sys
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

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

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


def refine_mask(mask, close=CLOSE_KERNEL, dilate_ratio=0.01, image_shape=None):
    """
    Morphological post-processing (hole filling / edge reinforcement).

    Args:
        mask: Binary mask array.
        close: Kernel size in pixels for the closing operation.
        dilate_ratio: Fraction of the image size (0-1) used to derive the dilation kernel size.
        image_shape: Optional (height, width) tuple for dilation kernel sizing.
    """
    if mask is None:
        return None
    k = max(1, int(close))
    if k > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    if dilate_ratio and dilate_ratio > 0:
        if image_shape is None or len(image_shape) < 2:
            image_shape = mask.shape
        base_len = max(int(image_shape[0]), int(image_shape[1]))
        kd = int(round(base_len * float(dilate_ratio)))
        if kd <= 1:
            return mask
        if kd % 2 == 0:
            kd += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kd, kd))
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


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
                       dilate_ratio=0.0,
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
        description="Tool to generate target masks with Mask R-CNN (COCO)."
    )
    ap.add_argument("-i", "--in", dest="input_dir", type=str, required=True,
                    help="Input directory containing images (non-recursive)")
    ap.add_argument("-o", "--out", type=str, default=None,
                    help="Output directory (default: create masks/ under the input root)")
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
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    if args.out is None:
        out_dir = input_dir / "_mask"
    else:
        out_dir = Path(args.out)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Device: {device}")
    targets = args.targets if args.targets else ["person"]
    deduped_targets = []
    for target_name in targets:
        if target_name not in deduped_targets:
            deduped_targets.append(target_name)
    targets = deduped_targets
    print(f"Targets: {', '.join(targets)}")

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
                      targets=targets)
    if total > 0:
        update_progress("Processing", total, total, last_pct)
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
