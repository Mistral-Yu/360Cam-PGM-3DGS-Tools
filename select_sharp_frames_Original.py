#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
select_sharp_frames_extwise_move.py  (dup-safe, listdir-based, fast)

手順:
  1) 指定フォルダ直下の画像を収集（既定: all = tif/png/jpg、listdirで重複なし）
  2) 指定ルールでソート
  3) 全画像の「シャープネス」を並列で計算（GRAYSCALE直読み + 縮小 + 中心クロップ + 32F）
  4) --group ごとに分割し、各グループで最もシャープな1枚だけ残す
     → それ以外は in_dir/blur/ に移動

既定:
  - ext: all（tif/png/jpg すべて）
  - metric: lapvar（Laplacian variance, 32F）
  - crop_ratio: 0.8（中心80%で評価）
  - sort: lastnum（ファイル名末尾の数字優先、無ければ名前順）
  - use_exposure: OFF（--use_exposure でON）
  - CSV: OFF（--csv で出力）
  - max_long: 1280（長辺ピクセル。0で縮小なし）
  - workers: min(8, os.cpu_count() or 4)
  - opencv_threads: 0（OpenCVの内部スレッド制御。0=デフォルトのまま）

Python 3.7 / OpenCV 4.x
"""

import os
import sys
import csv
import argparse
import shutil
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np


# ---------- 収集とソート（重複なし） ----------

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
    """存在確認しつつ安全移動。失敗時はコピー→削除でフォールバック"""
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
    サブフォルダは見ない。listdirで列挙→拡張子lowerでフィルタ→正規化パスで重複排除。
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

    # 正規化パスで重複排除（Windows: 大文字小文字/区切り差異を吸収）
    seen = set()
    files = []
    for f in raw:
        key = os.path.normcase(os.path.abspath(f))
        if key in seen:
            continue
        seen.add(key)
        files.append(f)
    return files


# ---------- 高速スコアリング ----------

def downscale_gray(gray, max_long):
    """長辺が max_long を超える場合のみ縮小（INTER_AREA）。0またはNoneで無効。"""
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
    高速化のため、さらに長辺512に制限してから低周波カット平均。
    """
    g = downscale_gray(gray, 512)
    f = np.fft.fft2(g.astype(np.float32))
    fshift = np.fft.fftshift(f)
    h, w = g.shape
    cy, cx = h//2, w//2
    r = max(1, min(h, w) // 8)  # 低周波を落とす半径
    # ドーナツ型マスク（低周波カット）
    yy, xx = np.ogrid[:h, :w]
    dist2 = (yy - cy)**2 + (xx - cx)**2
    mask = (dist2 >= r*r).astype(np.float32)
    hf = fshift * mask
    return float(np.mean(np.abs(hf)))

def exposure_clip_stats(gray):
    # 8bit前提（IMREAD_GRAYSCALE）
    p0 = float(np.mean(gray <= 2))
    p255 = float(np.mean(gray >= 253))
    return p0, p255

def score_one_file(fp, metric, crop_ratio, use_exposure, clip_penalty, clip_thresh, max_long):
    """
    1ファイルを読み→縮小→クロップ→スコア計算して返す（Noneなら失敗）
    """
    try:
        gray = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)  # 1ch/8bit直読み
        if gray is None:
            return None, 0.0, 0.0
        gray = downscale_gray(gray, max_long)
        gray = crop_by_ratio_gray(gray, crop_ratio)

        if metric == "lapvar":
            sharp = lapvar32(gray)
        elif metric == "tenengrad":
            sharp = tenengrad32(gray)
        elif metric == "fft":
            sharp = fft_energy_fast(gray)
        else:
            return None, 0.0, 0.0

        if use_exposure:
            p0, p255 = exposure_clip_stats(gray)
            clip = p0 + p255
            if clip > clip_thresh:
                sharp *= clip_penalty
            return sharp, p0, p255
        else:
            return sharp, 0.0, 0.0

    except ValueError:
        raise
    except Exception:
        return None, 0.0, 0.0


# ---------- メイン ----------

def main():
    ap = argparse.ArgumentParser(description="全画像をソート→グループ分け→各グループで最シャープ以外を blur/ へ移動（高速化版）")
    ap.add_argument("--in_dir", required=True, help="入力フォルダ（サブフォルダは見ない）")
    ap.add_argument("--group", type=positive_int, required=True, help="ブロック枚数（例: 12fps→3fpsなら 4）")
    ap.add_argument("--ext", choices=["all","tif","jpg","png"], default="all",
                    help="対象拡張子（既定: all = tif/png/jpg）")
    ap.add_argument("--sort", choices=["lastnum","firstnum","name","mtime"], default="lastnum",
                    help="並び順（既定: lastnum）")
    ap.add_argument("--metric", choices=["lapvar","tenengrad","fft"], default="lapvar",
                    help="シャープネス指標（既定: lapvar）")
    ap.add_argument("--crop_ratio", type=crop_ratio_arg, default=0.8,
                    help="中心クロップ比（既定0.8、1.0で無効）")

    # 露出ペナルティ（既定OFF）
    ap.add_argument("--use_exposure", action="store_true",
                    help="露出クリップを考慮（既定OFF）")
    ap.add_argument("--clip_penalty", type=float, default=0.5)
    ap.add_argument("--clip_thresh", type=float, default=0.25)

    # 高速化オプション
    ap.add_argument("--max_long", type=int, default=0,
                    help="評価用の長辺上限ピクセル（0で無効, 既定0）")
    ap.add_argument("--workers", type=int,
                    help="並列ワーカー数（未指定なら min(8, os.cpu_count() or 4)）")
    ap.add_argument("--opencv_threads", type=int, default=0,
                    help="OpenCVの内部スレッド数（0=デフォルトのまま）")

    # CSVは既定OFF
    ap.add_argument("--csv", help="CSVを書き出すパス（指定時のみ）")

    args = ap.parse_args()

    # OpenCV内部スレッド制御（Python側のThreadPoolと二重化しすぎないように）
    try:
        if args.opencv_threads and args.opencv_threads > 0:
            cv2.setNumThreads(args.opencv_threads)
    except Exception:
        pass

    files = gather_files(args.in_dir, args.ext)
    if not files:
        print("入力が見つかりません: {}".format(args.in_dir))
        sys.exit(1)

    sorter = SORTERS[args.sort]
    files = sorted(files, key=sorter)

    blur_dir = os.path.join(args.in_dir, "blur")
    ensure_dir(blur_dir)

    # 並列で全ファイルのスコアを計算
    n = len(files)
    scores = [None] * n
    p0_arr = [0.0] * n
    p255_arr = [0.0] * n

    workers = args.workers if (args.workers and args.workers > 0) else min(8, (os.cpu_count() or 4))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(
                score_one_file, files[i],
                args.metric, args.crop_ratio,
                args.use_exposure, args.clip_penalty, args.clip_thresh,
                args.max_long
            ): i for i in range(n)
        }
        completed = 0
        last_pct = -1
        for fut in as_completed(futs):
            i = futs[fut]
            s, p0, p255 = fut.result()
            scores[i] = s
            p0_arr[i] = p0
            p255_arr[i] = p255
            completed += 1
            last_pct = update_progress("Scoring", completed, n, last_pct)

    if n:
        print(f"Scoring... 100% ({n}/{n})")
    # CSV準備
    csv_writer = None
    fcsv = None
    if args.csv:
        csv_path = args.csv if os.path.isabs(args.csv) else os.path.join(args.in_dir, args.csv)
        fcsv = open(csv_path, "w", newline="")
        csv_writer = csv.writer(fcsv)
        csv_writer.writerow(["index","filename","score","p0_black","p255_white","selected(1=keep)"])

    # グルーピング＆移動
    total = n
    kept = 0
    moved = 0
    skipped = 0
    processed = 0
    last_group_pct = -1

    for start in range(0, total, args.group):
        end = min(total, start + args.group)
        best_idx = -1
        best_score = -1.0

        # ベスト決定（このループは軽い）
        for i in range(start, end):
            s = scores[i]
            if s is None:
                continue
            if s > best_score:
                best_score = s
                best_idx = i

        # 移動/残す
        for i in range(start, end):
            s = scores[i]
            processed += 1
            if s is None or not os.path.isfile(files[i]):
                skipped += 1
                if csv_writer:
                    csv_writer.writerow([i, os.path.basename(files[i]), -1.0, 0.0, 0.0, 0])
                last_group_pct = update_progress("Grouping", processed, total, last_group_pct)
                continue

            if i == best_idx:
                kept += 1
                if csv_writer:
                    csv_writer.writerow([i, os.path.basename(files[i]), s, p0_arr[i], p255_arr[i], 1])
            else:
                dst = os.path.join(blur_dir, os.path.basename(files[i]))
                if safe_move(files[i], dst) is None:
                    skipped += 1
                else:
                    moved += 1
                if csv_writer:
                    csv_writer.writerow([i, os.path.basename(files[i]), s, p0_arr[i], p255_arr[i], 0])
            last_group_pct = update_progress("Grouping", processed, total, last_group_pct)

    if total:
        print(f"Grouping... 100% ({total}/{total})")
    if fcsv:
        fcsv.close()

    print("完了: 入力 {} 枚 → 残した {} 枚 / blur に移動 {} 枚 / スキップ {} 件".format(total, kept, moved, skipped))
    print("blur フォルダ:", blur_dir)
    print("workers={}, opencv_threads={}, max_long={}".format(workers, args.opencv_threads, args.max_long))


if __name__ == "__main__":
    main()
