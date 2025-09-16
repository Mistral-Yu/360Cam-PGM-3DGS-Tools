
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ply_xyz_rgb_tool.py

- .ply から (x,y,z) と (R,G,B) のみをロード
- グリッド(ボクセル)でダウンサンプル
  - --target で目標点数に近づくようボクセルサイズを自動調整（詳細ログを標準出力へ）
  - --voxel-size で固定サイズ
- ダウンサンプル後に --add で別点群を連結（addPoints）
- 出力は binary little-endian PLY（vertex: x,y,z,red,green,blue）

※ 固定ボクセル時の代表点は「各ボクセル中心に最も近い**元の点**」を選びます（重心座標ではなく）

依存: numpy, plyfile
    pip install numpy plyfile
"""

from __future__ import annotations
import argparse
from typing import List, Optional, Tuple
import numpy as np
from plyfile import PlyData, PlyElement


# ------------------------------ Utilities ------------------------------

def _fmt3(a) -> str:
    return f"({float(a[0]):.6g}, {float(a[1]):.6g}, {float(a[2]):.6g})"


# ------------------------------ Load / Save ------------------------------

def _find_vertex_data(ply: PlyData):
    """vertex相当（x,y,z を持つ要素）を返す。"""
    try:
        return ply['vertex'].data
    except Exception:
        pass
    for el in getattr(ply, 'elements', []):
        data = getattr(el, 'data', None)
        names = getattr(data.dtype, 'names', None) if data is not None else None
        if names and all(k in names for k in ('x', 'y', 'z')):
            return data
    return None


def _extract_xyz_rgb_from_structured(v) -> Tuple[np.ndarray, np.ndarray]:
    names = v.dtype.names
    if not all(k in names for k in ('x', 'y', 'z')):
        raise ValueError("x,y,z が見つかりません")

    xyz = np.stack([v['x'].astype(np.float32),
                    v['y'].astype(np.float32),
                    v['z'].astype(np.float32)], axis=1)

    # 色フィールドの候補
    cand = [
        ('red', 'green', 'blue'),
        ('r', 'g', 'b'),
        ('diffuse_red', 'diffuse_green', 'diffuse_blue'),
    ]
    rgb = None
    for r, g, b in cand:
        if r in names and g in names and b in names:
            rr = v[r]; gg = v[g]; bb = v[b]
            if rr.dtype.kind == 'f' or gg.dtype.kind == 'f' or bb.dtype.kind == 'f':
                rr = np.clip(rr, 0.0, 1.0) * 255.0
                gg = np.clip(gg, 0.0, 1.0) * 255.0
                bb = np.clip(bb, 0.0, 1.0) * 255.0
                rgb = np.stack([rr, gg, bb], axis=1).round().astype(np.uint8)
            else:
                rgb = np.stack([rr, gg, bb], axis=1).astype(np.uint8, copy=False)
            break
    if rgb is None:
        rgb = np.full((xyz.shape[0], 3), 255, dtype=np.uint8)

    return xyz, rgb


def load_ply_xyz_rgb(path: str) -> Tuple[np.ndarray, np.ndarray]:
    ply = PlyData.read(path)
    vdata = _find_vertex_data(ply)
    if vdata is None:
        raise ValueError(f"x,y,z を持つ頂点要素が見つかりません: {path}")
    xyz, rgb = _extract_xyz_rgb_from_structured(vdata)
    return xyz, rgb


def save_ply_binary_little(path: str, xyz: np.ndarray, rgb: np.ndarray) -> None:
    if xyz.shape[0] != rgb.shape[0]:
        raise ValueError("xyz と rgb の行数が一致しません")
    n = xyz.shape[0]
    arr = np.empty(n, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    arr['x'] = xyz[:, 0].astype(np.float32, copy=False)
    arr['y'] = xyz[:, 1].astype(np.float32, copy=False)
    arr['z'] = xyz[:, 2].astype(np.float32, copy=False)
    arr['red'] = rgb[:, 0].astype(np.uint8, copy=False)
    arr['green'] = rgb[:, 1].astype(np.uint8, copy=False)
    arr['blue'] = rgb[:, 2].astype(np.uint8, copy=False)

    el = PlyElement.describe(arr, 'vertex')
    PlyData([el], text=False, byte_order='<').write(path)


# ------------------------------ Downsampling ------------------------------

def _grid_keys(xyz: np.ndarray, voxel: float, xyz_min: Optional[np.ndarray] = None) -> np.ndarray:
    if voxel <= 0:
        raise ValueError("voxel > 0 が必要です")
    if xyz_min is None:
        xyz_min = xyz.min(axis=0, keepdims=True)
    keys = np.floor((xyz - xyz_min) / voxel).astype(np.int64, copy=False)
    return keys


def voxel_downsample_by_size(
    xyz: np.ndarray,
    rgb: np.ndarray,
    voxel: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    ボクセルサイズ固定のダウンサンプリング。
    代表点は「各ボクセル中心に最も近い元の1点」を選択（xyz/rgbともにその点の値）。
    """
    xyz_min = xyz.min(axis=0, keepdims=True)
    keys = _grid_keys(xyz, voxel, xyz_min)                                # (N,3) int64
    uniq, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    k = uniq.shape[0]

    # 各ボクセルの中心座標 (K,3)
    centers = (xyz_min + (uniq.astype(np.float32) + 0.5) * voxel).astype(np.float32)

    # 各点ごとに対応ボクセル中心との差を取り、距離二乗を算出
    center_per_point = centers[inv]                                        # (N,3)
    diff = xyz.astype(np.float32, copy=False) - center_per_point
    dist2 = (diff * diff).sum(axis=1)                                      # (N,)

    # inv でソートして各グループごとに最小の dist2 のインデックスを取る
    order = np.argsort(inv, kind='mergesort')                              # 安全な安定ソート
    inv_sorted = inv[order]
    dist2_sorted = dist2[order]

    # グループの開始位置を検出
    starts = np.flatnonzero(np.r_[True, inv_sorted[1:] != inv_sorted[:-1]])
    ends = np.r_[starts[1:], inv_sorted.size]

    pick_idx = np.empty(k, dtype=np.int64)
    for a, b in zip(starts, ends):
        gid = int(inv_sorted[a])
        # グループ内の相対 argmin を求め、元のインデックスに戻す
        rel = a + int(np.argmin(dist2_sorted[a:b]))
        orig = int(order[rel])
        pick_idx[gid] = orig

    out_xyz = xyz[pick_idx].astype(np.float32, copy=False)
    out_rgb = rgb[pick_idx].astype(np.uint8, copy=False)
    return out_xyz, out_rgb


def _unique_voxel_count(xyz: np.ndarray, voxel: float, xyz_min: Optional[np.ndarray] = None) -> int:
    keys = _grid_keys(xyz, voxel, xyz_min)
    return int(np.unique(keys, axis=0).shape[0])


def voxel_downsample_to_target(
    xyz: np.ndarray,
    rgb: np.ndarray,
    target_n: int,
    tol_ratio: float = 0.02,
    max_iter: int = 25,
) -> Tuple[np.ndarray, np.ndarray]:
    """目標点数に近づくようにボクセルサイズを自動調整（詳細ログ出力）。"""
    n = xyz.shape[0]
    print(f"[target] n={n:,}  target={target_n:,}  tol=±{tol_ratio*100:.1f}%  max_iter={max_iter}")
    if target_n <= 0 or target_n >= n:
        print(f"[target] skip: target={target_n} は範囲外（出力=入力のまま）")
        return xyz.astype(np.float32, copy=False), rgb.astype(np.uint8, copy=False)

    xyz_min = xyz.min(axis=0)
    xyz_max = xyz.max(axis=0)
    extent = np.maximum(xyz_max - xyz_min, 1e-9)
    vol = float(extent[0] * extent[1] * extent[2])
    v0 = (vol / float(target_n)) ** (1.0 / 3.0) if vol > 0 else 1e-3

    lo = max(v0 / 64.0, 1e-9)
    hi = max(v0 * 64.0, lo * 2.0)
    print(f"[aabb] min={_fmt3(xyz_min)}  max={_fmt3(xyz_max)}  extent={_fmt3(extent)}  vol≈{vol:.6g}")
    print(f"[init] v0≈{v0:.6g}  lo={lo:.6g}  hi={hi:.6g}")

    # hi を拡張して "ユニークボクセル数 <= target" を満たす上限を確保
    for k in range(10):
        cnt_hi = _unique_voxel_count(xyz, hi, xyz_min)
        print(f"[expand] try hi={hi:.6g} -> unique={cnt_hi:,}")
        if cnt_hi <= target_n:
            break
        hi *= 2.0
    else:
        print("[expand] warn: 十分な上限が見つからない可能性")

    best_voxel = v0
    best_diff = 10**18
    best_cnt = n

    for it in range(1, max_iter + 1):
        mid = 0.5 * (lo + hi)
        cnt = _unique_voxel_count(xyz, mid, xyz_min)
        diff = abs(cnt - target_n)
        ratio = diff / float(target_n)
        if diff < best_diff:
            best_diff = diff
            best_voxel = mid
            best_cnt = cnt
        decision = "lo=mid (cnt>target → voxel↑)" if cnt > target_n else "hi=mid (cnt<target → voxel↓)"
        print(f"[iter {it:02d}] voxel={mid:.6g}  unique={cnt:,}  diff={diff:,} ({ratio:.2%})  -> {decision}")
        if ratio <= tol_ratio:
            print(f"[stop] 許容内に到達: voxel={mid:.6g}  unique={cnt:,}")
            best_voxel = mid
            best_cnt = cnt
            break
        if cnt > target_n:
            lo = mid
        else:
            hi = mid

    print(f"[best] voxel≈{best_voxel:.6g}  unique≈{best_cnt:,}  (best_diff={best_diff:,})")

    out_xyz, out_rgb = voxel_downsample_by_size(xyz, rgb, best_voxel)
    print(f"[final] voxel={best_voxel:.6g}  out_points={out_xyz.shape[0]:,}")
    return out_xyz, out_rgb


# ------------------------------ CLI ------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="PLY(XYZ+RGB) ロード/保存 + グリッドダウンサンプル + addPoints 連結")
    ap.add_argument("input", help="入力 PLY（ベース点群）")
    ap.add_argument("-o", "--output", required=True, help="出力 PLY（binary little-endian）")
    ap.add_argument("--target", type=int, default=None, help="目標点数（近似）。指定時は --voxel-size より優先")
    ap.add_argument("--voxel-size", type=float, default=None, help="固定ボクセルサイズ（m）")
    ap.add_argument("--add", action="append", default=[], help="ダウンサンプル後に追加する PLY（複数可）")
    ap.add_argument("--no-downsample", action="store_true", help="ダウンサンプリングを行わない")
    args = ap.parse_args(argv)

    # 1) 読み込み
    xyz, rgb = load_ply_xyz_rgb(args.input)
    print(f"[load] base: {args.input}  points={xyz.shape[0]:,}")

    # 2) ダウンサンプル
    if not args.no_downsample:
        if args.voxel_size is not None and args.voxel_size > 0:
            print(f"[downsample] fixed voxel-size={args.voxel_size:.6g}")
            xyz, rgb = voxel_downsample_by_size(xyz, rgb, args.voxel_size)
            print(f"[downsample] -> {xyz.shape[0]:,} points")
        elif args.target is not None and args.target > 0:
            xyz, rgb = voxel_downsample_to_target(xyz, rgb, args.target)
            print(f"[downsample] target={args.target:,} -> {xyz.shape[0]:,} points")
        else:
            print("[downsample] skip (no voxel-size/target)")
    else:
        print("[downsample] skipped by --no-downsample")

    # 3) 追加連結（ダウンサンプルの後）
    total_added = 0
    for apath in args.add:
        ax, ac = load_ply_xyz_rgb(apath)
        xyz = np.concatenate([xyz, ax], axis=0)
        rgb = np.concatenate([rgb, ac], axis=0)
        total_added += ax.shape[0]
        print(f"[addPoints] {apath} +{ax.shape[0]:,} -> total {xyz.shape[0]:,}")
    if total_added > 0:
        print(f"[addPoints] total added: {total_added:,}")

    # 4) 保存
    save_ply_binary_little(args.output, xyz, rgb)
    print(f"[save] {args.output}  points={xyz.shape[0]:,}  (binary little-endian)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
