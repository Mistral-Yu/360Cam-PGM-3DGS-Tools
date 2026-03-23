#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Camera format converter (COLMAP / RealityScan / transforms.json)."""

from __future__ import print_function

import argparse
import csv
import json
import math
import pathlib
import re
import sys
import xml.etree.ElementTree as ET


DEFAULT_SENSOR_W_MM = 36.0
DEFAULT_SENSOR_H_MM = 36.0
DEFAULT_TRANSFORMS_X_FIX_DEG = 270.0
CV_TO_GL = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]
REALITYSCAN_AXIS = [
    [1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
    [0.0, 1.0, 0.0],
]


def mat3_mul(a, b):
    out = [[0.0] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            out[i][j] = sum(a[i][k] * b[k][j] for k in range(3))
    return out


def mat4_mul(a, b):
    out = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            out[i][j] = sum(a[i][k] * b[k][j] for k in range(4))
    return out


def mat3_transpose(a):
    return [
        [a[0][0], a[1][0], a[2][0]],
        [a[0][1], a[1][1], a[2][1]],
        [a[0][2], a[1][2], a[2][2]],
    ]


def mat3_vec_mul(a, v):
    return [
        a[0][0] * v[0] + a[0][1] * v[1] + a[0][2] * v[2],
        a[1][0] * v[0] + a[1][1] * v[1] + a[1][2] * v[2],
        a[2][0] * v[0] + a[2][1] * v[1] + a[2][2] * v[2],
    ]


def mat3_to_mat4_with_translation(r, tvec):
    return [
        [r[0][0], r[0][1], r[0][2], tvec[0]],
        [r[1][0], r[1][1], r[1][2], tvec[1]],
        [r[2][0], r[2][1], r[2][2], tvec[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def rot_x_deg(deg):
    rad = math.radians(float(deg))
    c = math.cos(rad)
    s = math.sin(rad)
    return [
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s, c],
    ]


def rot_y_deg(deg):
    rad = math.radians(float(deg))
    c = math.cos(rad)
    s = math.sin(rad)
    return [
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c],
    ]


def rot_z_deg(deg):
    rad = math.radians(float(deg))
    c = math.cos(rad)
    s = math.sin(rad)
    return [
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ]


def apply_x_fix_gl(c2w_gl, deg):
    if deg is None or abs(float(deg)) < 1e-9:
        return c2w_gl
    fix_rot = mat3_to_mat4_with_translation(
        rot_x_deg(float(deg)),
        (0.0, 0.0, 0.0),
    )
    return mat4_mul(fix_rot, c2w_gl)


def normalize_angle_deg(angle):
    out = ((float(angle) + 180.0) % 360.0) - 180.0
    if abs(out + 180.0) < 1e-9:
        return 180.0
    return out


def build_world_rotation_xyz_deg(x_deg, y_deg, z_deg):
    """Compose world rotation in X -> Y -> Z order."""

    rx = rot_x_deg(float(x_deg))
    ry = rot_y_deg(float(y_deg))
    rz = rot_z_deg(float(z_deg))
    return mat3_mul(rz, mat3_mul(ry, rx))


def dot3(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross3(a, b):
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def norm3(v):
    return math.sqrt(dot3(v, v))


def normalize3(v):
    n = norm3(v)
    if n <= 0.0:
        return [0.0, 0.0, 0.0]
    return [v[0] / n, v[1] / n, v[2] / n]


def rotate_vec_axis_angle(v, axis, deg):
    axis = normalize3(axis)
    if norm3(axis) <= 0.0 or abs(float(deg)) < 1e-12:
        return [v[0], v[1], v[2]]
    rad = math.radians(float(deg))
    c = math.cos(rad)
    s = math.sin(rad)
    kv = cross3(axis, v)
    kk = dot3(axis, v)
    return [
        v[0] * c + kv[0] * s + axis[0] * kk * (1.0 - c),
        v[1] * c + kv[1] * s + axis[1] * kk * (1.0 - c),
        v[2] * c + kv[2] * s + axis[2] * kk * (1.0 - c),
    ]


def quat_wxyz_to_rotmat(qw, qx, qy, qz):
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if n <= 0.0:
        return [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    qw /= n
    qx /= n
    qy /= n
    qz /= n
    return [
        [
            1.0 - 2.0 * (qy * qy + qz * qz),
            2.0 * (qx * qy - qz * qw),
            2.0 * (qx * qz + qy * qw),
        ],
        [
            2.0 * (qx * qy + qz * qw),
            1.0 - 2.0 * (qx * qx + qz * qz),
            2.0 * (qy * qz - qx * qw),
        ],
        [
            2.0 * (qx * qz - qy * qw),
            2.0 * (qy * qz + qx * qw),
            1.0 - 2.0 * (qx * qx + qy * qy),
        ],
    ]


def rotmat_to_quat_wxyz(r):
    trace = r[0][0] + r[1][1] + r[2][2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (r[2][1] - r[1][2]) / s
        qy = (r[0][2] - r[2][0]) / s
        qz = (r[1][0] - r[0][1]) / s
    elif (r[0][0] > r[1][1]) and (r[0][0] > r[2][2]):
        s = math.sqrt(1.0 + r[0][0] - r[1][1] - r[2][2]) * 2.0
        qw = (r[2][1] - r[1][2]) / s
        qx = 0.25 * s
        qy = (r[0][1] + r[1][0]) / s
        qz = (r[0][2] + r[2][0]) / s
    elif r[1][1] > r[2][2]:
        s = math.sqrt(1.0 + r[1][1] - r[0][0] - r[2][2]) * 2.0
        qw = (r[0][2] - r[2][0]) / s
        qx = (r[0][1] + r[1][0]) / s
        qy = 0.25 * s
        qz = (r[1][2] + r[2][1]) / s
    else:
        s = math.sqrt(1.0 + r[2][2] - r[0][0] - r[1][1]) * 2.0
        qw = (r[1][0] - r[0][1]) / s
        qx = (r[0][2] + r[2][0]) / s
        qy = (r[1][2] + r[2][1]) / s
        qz = 0.25 * s
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if n <= 0.0:
        return (1.0, 0.0, 0.0, 0.0)
    return (qw / n, qx / n, qy / n, qz / n)


def parse_ply_header(fp):
    header = []
    while True:
        line = fp.readline()
        if not line:
            raise ValueError("unexpected EOF while reading PLY header")
        header.append(line)
        if line.strip() == b"end_header":
            break

    fmt = None
    vertex_count = 0
    props = []
    in_vertex = False
    for raw in header:
        line = raw.decode("ascii", "ignore").strip()
        if line.startswith("format "):
            fmt = line.split()[1]
        elif line.startswith("element "):
            parts = line.split()
            if len(parts) >= 3 and parts[1] == "vertex":
                in_vertex = True
                vertex_count = int(parts[2])
            else:
                in_vertex = False
        elif line.startswith("property ") and in_vertex:
            parts = line.split()
            if parts[1] == "list":
                raise ValueError("PLY list properties are not supported")
            if len(parts) >= 3:
                props.append((parts[1], parts[2]))
    if fmt is None:
        raise ValueError("PLY format not found")
    return fmt, vertex_count, props


def ply_type_info(type_name):
    mapping = {
        "float": ("f", 4),
        "float32": ("f", 4),
        "double": ("d", 8),
        "float64": ("d", 8),
        "uchar": ("B", 1),
        "uint8": ("B", 1),
        "char": ("b", 1),
        "int8": ("b", 1),
        "short": ("h", 2),
        "int16": ("h", 2),
        "ushort": ("H", 2),
        "uint16": ("H", 2),
        "int": ("i", 4),
        "int32": ("i", 4),
        "uint": ("I", 4),
        "uint32": ("I", 4),
    }
    if type_name not in mapping:
        raise ValueError("unsupported PLY type: {}".format(type_name))
    return mapping[type_name]


def read_ply_vertices(ply_path):
    import struct

    with pathlib.Path(ply_path).open("rb") as fp:
        fmt, vertex_count, props = parse_ply_header(fp)
        if fmt not in ("binary_little_endian", "ascii"):
            raise ValueError("unsupported PLY format: {}".format(fmt))

        prop_types = []
        prop_names = []
        for typ, name in props:
            fmt_char, _ = ply_type_info(typ)
            prop_types.append(fmt_char)
            prop_names.append(name)

        vertices = []
        if fmt == "ascii":
            for _ in range(vertex_count):
                line = fp.readline()
                if not line:
                    raise ValueError("unexpected EOF in PLY vertices")
                parts = line.decode("ascii", "ignore").strip().split()
                if len(parts) < len(prop_names):
                    raise ValueError("invalid PLY vertex row")
                row = {}
                for (typ, name), token in zip(props, parts):
                    if typ in ("float", "float32", "double", "float64"):
                        row[name] = float(token)
                    else:
                        row[name] = int(float(token))
                vertices.append(row)
        else:
            st = struct.Struct("<" + "".join(prop_types))
            for _ in range(vertex_count):
                data = fp.read(st.size)
                if len(data) != st.size:
                    raise ValueError("unexpected EOF in PLY vertices")
                values = st.unpack(data)
                row = {}
                for name, value in zip(prop_names, values):
                    row[name] = value
                vertices.append(row)
    return vertices, prop_names


def write_ply_vertices(out_path, vertices, prop_names):
    import struct

    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as fp:
        fp.write(b"ply\n")
        fp.write(b"format binary_little_endian 1.0\n")
        fp.write(
            "element vertex {}\n".format(len(vertices)).encode("ascii")
        )
        for name in prop_names:
            if name in ("x", "y", "z"):
                fp.write(b"property float %b\n" % name.encode("ascii"))
            else:
                fp.write(b"property uchar %b\n" % name.encode("ascii"))
        fp.write(b"end_header\n")

        fmt_list = []
        for name in prop_names:
            if name in ("x", "y", "z"):
                fmt_list.append("f")
            else:
                fmt_list.append("B")
        st = struct.Struct("<" + "".join(fmt_list))

        for row in vertices:
            data = []
            for name in prop_names:
                data.append(row[name])
            fp.write(st.pack(*data))


def colmap_camera_to_pinhole_intrinsics(cam):
    model = cam["model"].upper()
    p = cam["params"]
    if model == "SIMPLE_PINHOLE":
        fx = fy = p[0]
        cx = p[1]
        cy = p[2]
    elif model == "PINHOLE":
        fx, fy, cx, cy = p[:4]
    elif model in ("SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE"):
        fx = fy = p[0]
        cx = p[1]
        cy = p[2]
    elif model in ("RADIAL", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"):
        fx, fy, cx, cy = p[:4]
    else:
        raise ValueError("unsupported COLMAP camera model: {}".format(model))
    return (
        float(fx), float(fy), float(cx), float(cy),
        int(cam["width"]), int(cam["height"])
    )


def parse_colmap_cameras_txt(path):
    cameras = {}
    with pathlib.Path(path).open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            cameras[int(parts[0])] = {
                "camera_id": int(parts[0]),
                "model": parts[1],
                "width": int(parts[2]),
                "height": int(parts[3]),
                "params": [float(x) for x in parts[4:]],
            }
    return cameras


def parse_colmap_images_txt(path):
    images = []
    lines = pathlib.Path(path).read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        images.append({
            "image_id": int(parts[0]),
            "qw": float(parts[1]),
            "qx": float(parts[2]),
            "qy": float(parts[3]),
            "qz": float(parts[4]),
            "tx": float(parts[5]),
            "ty": float(parts[6]),
            "tz": float(parts[7]),
            "camera_id": int(parts[8]),
            "name": " ".join(parts[9:]),
            "points2d_line": lines[i] if i < len(lines) else "",
        })
        i += 1
    return images


def parse_colmap_points3d_txt(path):
    points = []
    p = pathlib.Path(path)
    if not p.exists():
        return points
    with p.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            points.append({
                "id": int(parts[0]),
                "x": float(parts[1]),
                "y": float(parts[2]),
                "z": float(parts[3]),
                "r": int(parts[4]),
                "g": int(parts[5]),
                "b": int(parts[6]),
                "err": float(parts[7]),
            })
    return points


def write_colmap_text_model(out_dir, cameras, images, points):
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cam_path = out_dir / "cameras.txt"
    img_path = out_dir / "images.txt"
    pts_path = out_dir / "points3D.txt"

    with cam_path.open("w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: {}\n".format(len(cameras)))
        for cam in sorted(cameras, key=lambda x: x["camera_id"]):
            f.write(
                "{camera_id} {model} {width} {height} {params}\n".format(
                    camera_id=cam["camera_id"],
                    model=cam["model"],
                    width=cam["width"],
                    height=cam["height"],
                    params=" ".join(
                        "{:.12g}".format(v) for v in cam["params"]
                    ),
                )
            )

    with img_path.open("w", encoding="utf-8") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(
            "# Number of images: {}, mean observations per image: 0\n".format(
                len(images)
            )
        )
        for img in sorted(images, key=lambda x: x["image_id"]):
            f.write(
                (
                    "{image_id} {qw:.12g} {qx:.12g} {qy:.12g} {qz:.12g} "
                    "{tx:.12g} {ty:.12g} {tz:.12g} {camera_id} {name}\n"
                ).format(**img)
            )
            f.write((img.get("points2d_line", "") or "") + "\n")

    with pts_path.open("w", encoding="utf-8") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write(
            "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as "
            "(IMAGE_ID, POINT2D_IDX)\n"
        )
        f.write(
            "# Number of points: {}, mean track length: 0\n".format(
                len(points)
            )
        )
        for pt in points:
            f.write(
                "{id} {x:.12g} {y:.12g} {z:.12g} {r} {g} {b} {err:.6g}\n"
                .format(**pt)
            )


def get_export_colmap_dir(out_root):
    """Return dedicated COLMAP export directory under the output root."""

    return pathlib.Path(out_root).expanduser().resolve() / "COLMAP_text_export"


def read_realityscan_csv(path):
    rows = []
    with pathlib.Path(path).open("r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for raw in rd:
            name_key = "#name" if "#name" in raw else "name"
            if not raw.get(name_key):
                continue
            rows.append({
                "name": raw[name_key],
                "x": float(raw["x"]),
                "y": float(raw["y"]),
                "alt": float(raw["alt"]),
                "heading": float(raw["heading"]),
                "pitch": float(raw["pitch"]),
                "roll": float(raw["roll"]),
                "f": float(raw["f"]),
            })
    return rows


def write_realityscan_csv(path, rows):
    header = [
        "#name", "x", "y", "alt", "heading", "pitch", "roll", "f",
        "px", "py", "k1", "k2", "k3", "k4", "t1", "t2",
    ]
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(header)
        for row in rows:
            wr.writerow([
                row["name"],
                "{:.15g}".format(row["x"]),
                "{:.15g}".format(row["y"]),
                "{:.15g}".format(row["alt"]),
                "{:.15g}".format(row["heading"]),
                "{:.15g}".format(row["pitch"]),
                "{:.15g}".format(row["roll"]),
                "{:.15g}".format(row["f"]),
                "0", "0", "0", "0", "0", "0", "0", "0",
            ])


def read_transforms_json(path):
    data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    intr = (
        float(data["fl_x"]), float(data["fl_y"]),
        float(data["cx"]), float(data["cy"]),
        int(data["w"]), int(data["h"])
    )
    frames = []
    for fr in data.get("frames", []):
        frames.append({
            "file_path": fr.get("file_path", ""),
            "transform_matrix": fr["transform_matrix"],
        })
    return frames, intr


def _xmp_extract_tag_text(text, tag_name):
    pattern = r"<xcr:{0}>([^<]+)</xcr:{0}>".format(re.escape(tag_name))
    m = re.search(pattern, text)
    if not m:
        raise ValueError("xmp missing xcr:{} tag".format(tag_name))
    return m.group(1).strip()


def _xmp_extract_attr(text, attr_name, default=None):
    pattern = r'{0}="([^"]+)"'.format(re.escape(attr_name))
    m = re.search(pattern, text)
    if not m:
        return default
    return m.group(1)


def _xmp_extract_tag_or_attr_text(text, key_name):
    try:
        return _xmp_extract_tag_text(text, key_name)
    except ValueError:
        pass
    attr_val = _xmp_extract_attr(text, "xcr:{}".format(key_name))
    if attr_val is None:
        raise ValueError("xmp missing xcr:{} tag/attr".format(key_name))
    return attr_val.strip()


def read_realityscan_xmp_dir(xmp_dir, image_ext="jpg"):
    """Read RealityScan XMP camera files.

    Returns list of dicts:
    - name
    - r_xmp (3x3, world->camera in RS convention)
    - pos_rs ([x, y, alt])
    - focal_mm
    """
    xmp_dir = pathlib.Path(xmp_dir)
    if not xmp_dir.exists():
        raise ValueError("xmp dir not found: {}".format(xmp_dir))
    xmp_files = sorted(xmp_dir.glob("*.xmp"))
    if not xmp_files:
        raise ValueError("no .xmp files found in {}".format(xmp_dir))
    ext = str(image_ext or "").lstrip(".")
    rows = []
    for xmp_path in xmp_files:
        text = xmp_path.read_text(encoding="utf-8")
        rot_vals = [
            float(x)
            for x in _xmp_extract_tag_or_attr_text(text, "Rotation").split()
        ]
        pos_vals = [
            float(x)
            for x in _xmp_extract_tag_or_attr_text(text, "Position").split()
        ]
        if len(rot_vals) != 9:
            raise ValueError(
                "invalid xcr:Rotation value count in {}".format(xmp_path)
            )
        if len(pos_vals) != 3:
            raise ValueError(
                "invalid xcr:Position value count in {}".format(xmp_path)
            )
        focal_attr = _xmp_extract_attr(text, "xcr:FocalLength35mm")
        if focal_attr is None:
            raise ValueError(
                "missing xcr:FocalLength35mm in {}".format(xmp_path)
            )
        r_xmp = [
            rot_vals[0:3],
            rot_vals[3:6],
            rot_vals[6:9],
        ]
        file_name = xmp_path.stem
        if ext:
            file_name = "{}.{}".format(file_name, ext)
        rows.append({
            "name": file_name,
            "r_xmp": r_xmp,
            "pos_rs": pos_vals,
            "focal_mm": float(focal_attr),
        })
    return rows


def _list_image_files(image_dir):
    image_dir = pathlib.Path(image_dir)
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".exr"}
    files = [
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    ]
    files.sort()
    return files


def infer_image_size_from_dir(image_dir):
    """Infer image width/height from the first readable image in a folder."""
    files = _list_image_files(image_dir)
    if not files:
        raise ValueError("no image files found in {}".format(image_dir))
    try:
        import cv2
    except Exception as exc:
        raise ValueError(
            "cv2 is required to infer image size from --image-dir: {}".format(
                exc
            )
        )
    for path in files:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        h, w = img.shape[:2]
        if int(w) > 0 and int(h) > 0:
            return int(w), int(h)
    raise ValueError(
        "failed to read any image for size inference in {}".format(image_dir)
    )


def map_stem_to_image_name(image_dir):
    """Map file stem -> actual image file name (for XMP basename matching)."""
    out = {}
    if not image_dir:
        return out
    for p in _list_image_files(image_dir):
        out[p.stem] = p.name
    return out


def write_transforms_json(path, frames, intrinsics):
    fx, fy, cx, cy, w, h = intrinsics
    payload = {
        "camera_model": "OPENCV",
        "fl_x": fx,
        "fl_y": fy,
        "cx": cx,
        "cy": cy,
        "w": int(w),
        "h": int(h),
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "frames": frames,
    }
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _xml_indent(elem, level=0):
    newline = "\n" + ("  " * level)
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = newline + "  "
        for child in elem:
            _xml_indent(child, level + 1)
        if not elem[-1].tail or not elem[-1].tail.strip():
            elem[-1].tail = newline
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = newline


def _parse_metashape_transform_text(text, xml_path):
    vals = [float(x) for x in str(text or "").split()]
    if len(vals) != 16:
        raise ValueError(
            "invalid <transform> value count in {} (expected 16)".format(
                xml_path
            )
        )
    return [
        vals[0:4],
        vals[4:8],
        vals[8:12],
        vals[12:16],
    ]


def _find_metashape_sensor_resolution(sensor_elem):
    res = sensor_elem.find("resolution")
    if res is None:
        res = sensor_elem.find("./calibration/resolution")
    if res is None:
        return None, None
    w = res.attrib.get("width")
    h = res.attrib.get("height")
    if w is None or h is None:
        return None, None
    return int(float(w)), int(float(h))


def _find_metashape_sensor_focal_px(sensor_elem):
    f_node = sensor_elem.find("./calibration/f")
    if f_node is None or f_node.text is None:
        return None
    return float(f_node.text.strip())


def read_metashape_perspective_xml(path, args, image_name_map=None):
    """Read Metashape perspective XML and normalize to RS CSV-like rows."""
    xml_path = pathlib.Path(path)
    root = ET.parse(str(xml_path)).getroot()
    chunk = root.find("chunk")
    if chunk is None:
        raise ValueError(
            "invalid Metashape XML (missing <chunk>): {}".format(path)
        )

    sensors_root = chunk.find("sensors")
    cams_root = chunk.find("cameras")
    if sensors_root is None or cams_root is None:
        raise ValueError(
            "invalid Metashape XML (missing <sensors>/<cameras>): {}".format(
                path
            )
        )

    sensors = {}
    for sensor in sensors_root.findall("sensor"):
        if sensor.attrib.get("master_id") is not None:
            raise ValueError(
                "Multi-Camera-System XML is not supported yet: {}".format(path)
            )
        if (
            sensor.find("rotation") is not None or
            sensor.find("location") is not None
        ):
            raise ValueError(
                "Multi-Camera-System XML is not supported yet: {}".format(path)
            )
        sid = int(sensor.attrib["id"])
        w, h = _find_metashape_sensor_resolution(sensor)
        f_px = _find_metashape_sensor_focal_px(sensor)
        sensors[sid] = {"w": w, "h": h, "f_px": f_px}

    image_name_map = image_name_map or {}
    rows = []
    width = None
    height = None
    for cam in cams_root.findall("camera"):
        tr_node = cam.find("transform")
        if tr_node is None or not (tr_node.text or "").strip():
            continue
        label = cam.attrib.get("label")
        if not label:
            continue
        sensor_id = cam.attrib.get("sensor_id")
        if sensor_id is None:
            if len(sensors) != 1:
                raise ValueError(
                    "camera missing sensor_id in multi-sensor XML: {}".format(
                        path
                    )
                )
            sensor_info = list(sensors.values())[0]
        else:
            sensor_info = sensors.get(int(sensor_id))
            if sensor_info is None:
                raise ValueError(
                    "unknown sensor_id {} in {}".format(sensor_id, path)
                )

        w = sensor_info.get("w")
        h = sensor_info.get("h")
        f_px = sensor_info.get("f_px")
        if (
            (w is None or h is None) and
            args.width is not None and
            args.height is not None
        ):
            w = int(args.width)
            h = int(args.height)
        if w is None or h is None:
            raise ValueError("Metashape XML sensor resolution missing")
        if f_px is None:
            raise ValueError("Metashape XML sensor focal <f> missing")
        if width is None:
            width = int(w)
            height = int(h)
        elif int(w) != width or int(h) != height:
            raise ValueError(
                "mixed image resolutions in Metashape XML are not "
                "supported yet"
            )

        c2w_cv = _parse_metashape_transform_text(tr_node.text, xml_path)
        r_cw = [row[:3] for row in c2w_cv[:3]]
        center = [c2w_cv[0][3], c2w_cv[1][3], c2w_cv[2][3]]
        r_wc = mat3_transpose(r_cw)
        center_rs = colmap_world_to_rs_world(center)
        r_xmp = colmap_pose_rot_to_rs_rot(r_wc)
        heading, pitch, roll = rs_rotation_to_hpr(r_xmp)
        focal_mm = focal_pixels_to_mm(
            f_px, f_px, w, h, args.sensor_width_mm, args.sensor_height_mm
        )

        if "." not in label:
            img_name = "{}.{}".format(label, args.metashape_xml_image_ext)
        else:
            img_name = label
        img_name = image_name_map.get(pathlib.Path(img_name).stem, img_name)
        rows.append(
            make_rs_csv_row(
                img_name,
                center_rs[0],
                center_rs[1],
                center_rs[2],
                heading,
                pitch,
                roll,
                focal_mm,
            )
        )

    if not rows:
        raise ValueError(
            "no cameras with <transform> found in {}".format(path)
        )
    return rows, width, height


def export_metashape_perspective_xml(path, cameras, images):
    """Export Metashape perspective camera XML (transform = c2w_cv)."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cam_list = list(cameras)
    img_list = list(images)
    cam_by_id = {int(c["camera_id"]): c for c in cam_list}

    sensor_defs = {}
    sensor_id_by_cam_id = {}
    next_sensor_id = 0
    for img in img_list:
        cam = cam_by_id[int(img["camera_id"])]
        fx, fy, _cx, _cy, w, h = colmap_camera_to_pinhole_intrinsics(cam)
        key = (int(w), int(h), round(float(fx), 9), round(float(fy), 9))
        if key not in sensor_defs:
            sensor_defs[key] = {
                "id": next_sensor_id,
                "w": int(w),
                "h": int(h),
                "f": 0.5 * (float(fx) + float(fy)),
            }
            next_sensor_id += 1
        sensor_id_by_cam_id[int(cam["camera_id"])] = sensor_defs[key]["id"]

    doc = ET.Element("document", {"version": "1.2.0"})
    chunk = ET.SubElement(
        doc, "chunk", {"label": "unknown", "enabled": "true"}
    )
    sensors_node = ET.SubElement(
        chunk, "sensors", {"next_id": str(next_sensor_id)}
    )
    for s in sorted(sensor_defs.values(), key=lambda x: x["id"]):
        sensor = ET.SubElement(
            sensors_node,
            "sensor",
            {
                "id": str(s["id"]),
                "label": "virtual_fisheyelike",
                "type": "frame",
            },
        )
        ET.SubElement(
            sensor, "resolution",
            {"width": str(s["w"]), "height": str(s["h"])},
        )
        ET.SubElement(
            sensor, "property", {"name": "layer_index", "value": "0"}
        )
        ET.SubElement(sensor, "data_type").text = "uint8"
        calib = ET.SubElement(
            sensor, "calibration", {"type": "frame", "class": "initial"}
        )
        ET.SubElement(
            calib, "resolution",
            {"width": str(s["w"]), "height": str(s["h"])},
        )
        ET.SubElement(calib, "f").text = "{:.15g}".format(s["f"])
        ET.SubElement(sensor, "black_level").text = "0 0 0"
        ET.SubElement(sensor, "sensitivity").text = "1 1 1"

    comps = ET.SubElement(
        chunk, "components", {"next_id": "1", "active_id": "0"}
    )
    comp = ET.SubElement(
        comps, "component", {"id": "0", "label": "Component 1"}
    )
    ET.SubElement(comp, "partition")

    cams_node = ET.SubElement(
        chunk,
        "cameras",
        {"next_id": str(len(img_list)), "next_group_id": "0"},
    )
    for idx, img in enumerate(img_list):
        r_wc = quat_wxyz_to_rotmat(
            img["qw"], img["qx"], img["qy"], img["qz"]
        )
        t_wc = [img["tx"], img["ty"], img["tz"]]
        center = camera_center_from_colmap_pose(r_wc, t_wc)
        r_cw = mat3_transpose(r_wc)
        c2w_cv = mat3_to_mat4_with_translation(r_cw, center)
        cam_node = ET.SubElement(
            cams_node,
            "camera",
            {
                "id": str(idx),
                "sensor_id": str(
                    sensor_id_by_cam_id[int(img["camera_id"])]
                ),
                "component_id": "0",
                "label": pathlib.Path(img["name"]).stem,
            },
        )
        flat = []
        for row in c2w_cv:
            for v in row:
                flat.append("{:.15g}".format(float(v)))
        ET.SubElement(cam_node, "transform").text = " ".join(flat)

    _xml_indent(doc)
    with path.open("wb") as f:
        f.write(b"<?xml version='1.0' encoding='UTF-8'?>\n")
        f.write(ET.tostring(doc, encoding="utf-8"))
        f.write(b"\n")


def export_realityscan_xmp(out_dir, records):
    """Write RealityScan XMP camera files.

    records item keys:
    - name
    - r_xmp (world->camera 3x3, RS axis convention)
    - pos_rs ([x, y, alt])
    - focal_mm
    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for rec in records:
        stem = pathlib.Path(rec["name"]).stem
        xmp_path = out_dir / (stem + ".xmp")
        rotation_text = " ".join(
            "{:.15g}".format(v)
            for row in rec["r_xmp"]
            for v in row
        )
        pos = rec["pos_rs"]
        position_text = "{:.15g} {:.15g} {:.15g}".format(
            pos[0], pos[1], pos[2]
        )
        focal_text = "{:g}".format(float(rec["focal_mm"]))
        lines = [
            '<x:xmpmeta xmlns:x="adobe:ns:meta/">',
            '  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/'
            '02/22-rdf-syntax-ns#">',
            '    <rdf:Description xcr:Version="3" xcr:PosePrior="initial" '
            'xcr:Coordinates="absolute"',
            '       xcr:DistortionModel="perspective" '
            'xcr:DistortionCoeficients="0 0 0 0 0 0"',
            '       xcr:FocalLength35mm="{}" xcr:Skew="0" xcr:AspectRatio="1" '
            'xcr:PrincipalPointU="0"'.format(focal_text),
            '       xcr:PrincipalPointV="0" xcr:CalibrationPrior="initial" '
            'xcr:CalibrationGroup="0"',
            '       xcr:DistortionGroup="0" xcr:InTexturing="1" '
            'xcr:InMeshing="1" '
            'xmlns:xcr="http://www.capturingreality.com/ns/xcr/1.1#">',
            "      <xcr:Rotation>{}</xcr:Rotation>".format(rotation_text),
            "      <xcr:Position>{}</xcr:Position>".format(position_text),
            "    </rdf:Description>",
            "  </rdf:RDF>",
            "</x:xmpmeta>",
        ]
        with xmp_path.open("w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


def camera_center_from_colmap_pose(r_wc, t_wc):
    r_cw = mat3_transpose(r_wc)
    return mat3_vec_mul(r_cw, [-t_wc[0], -t_wc[1], -t_wc[2]])


def colmap_pose_from_camera_center(r_wc, center):
    return mat3_vec_mul(r_wc, [-center[0], -center[1], -center[2]])


def colmap_pose_to_c2w_gl(r_wc, t_wc):
    center = camera_center_from_colmap_pose(r_wc, t_wc)
    r_cw = mat3_transpose(r_wc)
    c2w_cv = mat3_to_mat4_with_translation(r_cw, center)
    return mat4_mul(c2w_cv, CV_TO_GL)


def c2w_gl_to_colmap_pose(c2w_gl):
    c2w_cv = mat4_mul(c2w_gl, CV_TO_GL)
    r_cw = [row[:3] for row in c2w_cv[:3]]
    r_wc = mat3_transpose(r_cw)
    center = [c2w_cv[0][3], c2w_cv[1][3], c2w_cv[2][3]]
    t_wc = colmap_pose_from_camera_center(r_wc, center)
    return r_wc, t_wc


def colmap_world_to_rs_world(v):
    axis_t = mat3_transpose(REALITYSCAN_AXIS)
    return mat3_vec_mul(axis_t, v)  # [x, z, -y]


def rs_world_to_colmap_world(v):
    return mat3_vec_mul(REALITYSCAN_AXIS, v)


def colmap_pose_rot_to_rs_rot(r_wc):
    return mat3_mul(r_wc, REALITYSCAN_AXIS)


def rs_rot_to_colmap_pose_rot(r_xmp):
    return mat3_mul(r_xmp, mat3_transpose(REALITYSCAN_AXIS))


def rs_rotation_to_hpr(r_xmp):
    """Match sample Align_RS_PerspCams.csv heading/pitch/roll convention."""
    r_cw = mat3_transpose(r_xmp)
    fwd = normalize3([r_cw[0][1], r_cw[1][1], r_cw[2][1]])
    up = normalize3([r_cw[0][2], r_cw[1][2], r_cw[2][2]])
    heading = normalize_angle_deg(
        math.degrees(math.atan2(fwd[0], fwd[1])) - 180.0
    )
    pitch = -math.degrees(
        math.atan2(fwd[2], math.sqrt(fwd[0] * fwd[0] + fwd[1] * fwd[1]))
    )
    world_up = [0.0, 0.0, 1.0]
    right0 = cross3(world_up, fwd)
    if norm3(right0) < 1e-9:
        right0 = [1.0, 0.0, 0.0]
    right0 = normalize3(right0)
    up0 = normalize3(cross3(fwd, right0))
    s = dot3(fwd, cross3(up0, up))
    c = dot3(up0, up)
    roll_signed = math.degrees(math.atan2(s, c))
    roll = normalize_angle_deg(180.0 - roll_signed)
    return heading, pitch, roll


def hpr_to_rs_rotation(heading, pitch, roll):
    az = math.radians(normalize_angle_deg(float(heading) + 180.0))
    elev = math.radians(-float(pitch))
    cos_e = math.cos(elev)
    fwd = normalize3([
        math.sin(az) * cos_e,
        math.cos(az) * cos_e,
        math.sin(elev),
    ])
    world_up = [0.0, 0.0, 1.0]
    right0 = cross3(world_up, fwd)
    if norm3(right0) < 1e-9:
        right0 = [1.0, 0.0, 0.0]
    right0 = normalize3(right0)
    up0 = normalize3(cross3(fwd, right0))
    roll_signed = normalize_angle_deg(180.0 - float(roll))
    up = rotate_vec_axis_angle(up0, fwd, roll_signed)
    # Enforce right-handed camera basis: x(right), y(forward), z(up)
    right = normalize3(cross3(fwd, up))
    up = normalize3(cross3(right, fwd))
    r_cw = [
        [right[0], fwd[0], up[0]],
        [right[1], fwd[1], up[1]],
        [right[2], fwd[2], up[2]],
    ]
    return mat3_transpose(r_cw)


def focal_pixels_to_mm(fx, fy, w, h, sensor_w_mm, sensor_h_mm):
    mm_x = float(fx) * (float(sensor_w_mm) / float(w))
    mm_y = float(fy) * (float(sensor_h_mm) / float(h))
    return 0.5 * (mm_x + mm_y)


def focal_mm_to_pixels(f_mm, w, h, sensor_w_mm, sensor_h_mm):
    fx = float(f_mm) / (float(sensor_w_mm) / float(w))
    fy = float(f_mm) / (float(sensor_h_mm) / float(h))
    return fx, fy


def make_rs_csv_row(name, x, y, alt, heading, pitch, roll, focal_mm):
    """Canonical internal camera row (RealityScan CSV-compatible fields)."""
    return {
        "name": str(name),
        "x": float(x),
        "y": float(y),
        "alt": float(alt),
        "heading": float(heading),
        "pitch": float(pitch),
        "roll": float(roll),
        "f": float(focal_mm),
    }


def build_colmap_from_rs_rows(rows, args, w, h, image_name_map=None):
    """Build COLMAP camera/images from canonical RS CSV-like rows."""
    w = int(w)
    h = int(h)
    image_name_map = image_name_map or {}

    cameras = []
    cam_map = {}
    images = []
    next_cam_id = 1
    for idx, row in enumerate(rows, start=1):
        r_xmp = hpr_to_rs_rotation(row["heading"], row["pitch"], row["roll"])
        r_wc = rs_rot_to_colmap_pose_rot(r_xmp)
        center = rs_world_to_colmap_world([row["x"], row["y"], row["alt"]])
        t_wc = colmap_pose_from_camera_center(r_wc, center)
        qw, qx, qy, qz = rotmat_to_quat_wxyz(r_wc)
        fx, fy = focal_mm_to_pixels(
            row["f"], w, h, args.sensor_width_mm, args.sensor_height_mm
        )
        cx = w * 0.5
        cy = h * 0.5
        if args.single_camera:
            cam_id = 1
            if not cameras:
                cameras.append({
                    "camera_id": 1,
                    "model": "PINHOLE",
                    "width": w,
                    "height": h,
                    "params": [fx, fy, cx, cy],
                })
        else:
            key = (round(fx, 6), round(fy, 6), w, h)
            if key not in cam_map:
                cam_map[key] = next_cam_id
                cameras.append({
                    "camera_id": next_cam_id,
                    "model": "PINHOLE",
                    "width": w,
                    "height": h,
                    "params": [fx, fy, cx, cy],
                })
                next_cam_id += 1
            cam_id = cam_map[key]
        img_name = image_name_map.get(
            pathlib.Path(row["name"]).stem,
            row["name"],
        )
        images.append({
            "image_id": idx,
            "qw": qw,
            "qx": qx,
            "qy": qy,
            "qz": qz,
            "tx": t_wc[0],
            "ty": t_wc[1],
            "tz": t_wc[2],
            "camera_id": cam_id,
            "name": img_name,
            "points2d_line": "",
        })
    return cameras, images


def points_to_rs_vertices(points):
    """Canonical internal point format = RealityScan PLY axis/order."""
    vertices = []
    for pt in points:
        x, y, z = colmap_world_to_rs_world([pt["x"], pt["y"], pt["z"]])
        vertices.append({
            "x": x, "y": y, "z": z,
            "red": int(pt["r"]), "green": int(pt["g"]), "blue": int(pt["b"]),
        })
    return vertices, ["x", "y", "z", "red", "green", "blue"]


def points_to_transforms_ply_vertices(points):
    """Companion PLY axis for transforms.json set.

    Matches `pointcloud_for_transforms.ply`.
    """
    vertices = []
    for pt in points:
        x = float(pt["x"])
        y = -float(pt["y"])
        z = -float(pt["z"])
        vertices.append({
            "x": x, "y": y, "z": z,
            "red": int(pt["r"]), "green": int(pt["g"]), "blue": int(pt["b"]),
        })
    return vertices, ["x", "y", "z", "red", "green", "blue"]


def rs_vertices_to_points(vertices, point_id_start):
    points = []
    pid = int(point_id_start)
    for v in vertices:
        x, y, z = rs_world_to_colmap_world(
            [
                float(v.get("x", 0.0)),
                float(v.get("y", 0.0)),
                float(v.get("z", 0.0)),
            ]
        )
        points.append({
            "id": pid,
            "x": x,
            "y": y,
            "z": z,
            "r": int(v.get("red", 128)),
            "g": int(v.get("green", 128)),
            "b": int(v.get("blue", 128)),
            "err": 0.0,
        })
        pid += 1
    return points


def transforms_ply_vertices_to_points(vertices, point_id_start):
    """Inverse of points_to_transforms_ply_vertices."""
    points = []
    pid = int(point_id_start)
    for v in vertices:
        points.append({
            "id": pid,
            "x": float(v.get("x", 0.0)),
            "y": -float(v.get("y", 0.0)),
            "z": -float(v.get("z", 0.0)),
            "r": int(v.get("red", 128)),
            "g": int(v.get("green", 128)),
            "b": int(v.get("blue", 128)),
            "err": 0.0,
        })
        pid += 1
    return points


def rotate_colmap_images(images, rot_world):
    """Rotate COLMAP camera centers/orientations around world origin."""

    rot_world_t = mat3_transpose(rot_world)
    rotated = []
    for img in images:
        new_img = dict(img)
        r_wc = quat_wxyz_to_rotmat(
            img["qw"], img["qx"], img["qy"], img["qz"]
        )
        t_wc = [img["tx"], img["ty"], img["tz"]]
        center = camera_center_from_colmap_pose(r_wc, t_wc)
        center_rot = mat3_vec_mul(rot_world, center)
        r_wc_rot = mat3_mul(r_wc, rot_world_t)
        qw, qx, qy, qz = rotmat_to_quat_wxyz(r_wc_rot)
        t_wc_rot = colmap_pose_from_camera_center(r_wc_rot, center_rot)
        new_img.update(
            {
                "qw": qw,
                "qx": qx,
                "qy": qy,
                "qz": qz,
                "tx": t_wc_rot[0],
                "ty": t_wc_rot[1],
                "tz": t_wc_rot[2],
            }
        )
        rotated.append(new_img)
    return rotated


def rotate_colmap_points(points, rot_world):
    """Rotate COLMAP pointcloud around world origin."""

    rotated = []
    for pt in points:
        new_pt = dict(pt)
        xyz_rot = mat3_vec_mul(
            rot_world,
            [float(pt["x"]), float(pt["y"]), float(pt["z"])],
        )
        new_pt.update({"x": xyz_rot[0], "y": xyz_rot[1], "z": xyz_rot[2]})
        rotated.append(new_pt)
    return rotated


def scale_colmap_images(images, scale_world):
    """Scale COLMAP camera centers around the world origin."""

    scale_value = float(scale_world)
    if abs(scale_value - 1.0) <= 1e-12:
        return images
    scaled = []
    for img in images:
        new_img = dict(img)
        r_wc = quat_wxyz_to_rotmat(
            img["qw"], img["qx"], img["qy"], img["qz"]
        )
        t_wc = [img["tx"], img["ty"], img["tz"]]
        center = camera_center_from_colmap_pose(r_wc, t_wc)
        center_scaled = [
            scale_value * float(center[0]),
            scale_value * float(center[1]),
            scale_value * float(center[2]),
        ]
        t_wc_scaled = colmap_pose_from_camera_center(r_wc, center_scaled)
        new_img.update(
            {
                "tx": t_wc_scaled[0],
                "ty": t_wc_scaled[1],
                "tz": t_wc_scaled[2],
            }
        )
        scaled.append(new_img)
    return scaled


def scale_colmap_points(points, scale_world):
    """Scale COLMAP pointcloud around the world origin."""

    scale_value = float(scale_world)
    if abs(scale_value - 1.0) <= 1e-12:
        return points
    scaled = []
    for pt in points:
        new_pt = dict(pt)
        new_pt.update(
            {
                "x": scale_value * float(pt["x"]),
                "y": scale_value * float(pt["y"]),
                "z": scale_value * float(pt["z"]),
            }
        )
        scaled.append(new_pt)
    return scaled


def apply_optional_scene_rotations(args, images, points):
    """Apply user-requested camera / pointcloud world transforms."""

    camera_rot = build_world_rotation_xyz_deg(
        getattr(args, "camera_rot_x_deg", 0.0),
        getattr(args, "camera_rot_y_deg", 0.0),
        getattr(args, "camera_rot_z_deg", 0.0),
    )
    point_rot = build_world_rotation_xyz_deg(
        getattr(args, "pointcloud_rot_x_deg", 0.0),
        getattr(args, "pointcloud_rot_y_deg", 0.0),
        getattr(args, "pointcloud_rot_z_deg", 0.0),
    )
    camera_scale = float(getattr(args, "camera_scale", 1.0))
    point_scale = float(getattr(args, "pointcloud_scale", 1.0))
    if any(
        abs(float(v)) > 1e-9
        for v in (
            getattr(args, "camera_rot_x_deg", 0.0),
            getattr(args, "camera_rot_y_deg", 0.0),
            getattr(args, "camera_rot_z_deg", 0.0),
        )
    ):
        images = rotate_colmap_images(images, camera_rot)
    if abs(camera_scale - 1.0) > 1e-9:
        images = scale_colmap_images(images, camera_scale)
    if any(
        abs(float(v)) > 1e-9
        for v in (
            getattr(args, "pointcloud_rot_x_deg", 0.0),
            getattr(args, "pointcloud_rot_y_deg", 0.0),
            getattr(args, "pointcloud_rot_z_deg", 0.0),
        )
    ):
        points = rotate_colmap_points(points, point_rot)
    if abs(point_scale - 1.0) > 1e-9:
        points = scale_colmap_points(points, point_scale)
    return images, points


def export_from_colmap_model(args, cameras, images, points, out_dir):
    """Export selected formats from an in-memory COLMAP-style model."""
    if isinstance(cameras, dict):
        cam_map = cameras
        cam_list = list(cameras.values())
    else:
        cam_list = list(cameras)
        cam_map = {int(c["camera_id"]): c for c in cam_list}
    csv_rows = []
    tf_frames = []
    xmp_records = []
    intr_ref = None

    for img in images:
        cam = cam_map[img["camera_id"]]
        intr = colmap_camera_to_pinhole_intrinsics(cam)
        fx, fy, cx, cy, w, h = intr
        if intr_ref is None:
            intr_ref = intr
        elif getattr(args, "export_transforms", False):
            if any(
                abs(float(a) - float(b)) > 1e-6
                for a, b in zip(intr_ref, intr)
            ):
                raise ValueError(
                    "transforms.json export requires uniform intrinsics"
                )
        focal_mm = focal_pixels_to_mm(
            fx, fy, w, h, args.sensor_width_mm, args.sensor_height_mm
        )

        r_wc = quat_wxyz_to_rotmat(img["qw"], img["qx"], img["qy"], img["qz"])
        t_wc = [img["tx"], img["ty"], img["tz"]]
        center_colmap = camera_center_from_colmap_pose(r_wc, t_wc)
        center_rs = colmap_world_to_rs_world(center_colmap)
        r_xmp = colmap_pose_rot_to_rs_rot(r_wc)
        heading, pitch, roll = rs_rotation_to_hpr(r_xmp)

        csv_rows.append(
            make_rs_csv_row(
                img["name"],
                center_rs[0],
                center_rs[1],
                center_rs[2],
                heading,
                pitch,
                roll,
                focal_mm,
            )
        )
        if getattr(args, "export_xmp", False):
            xmp_records.append({
                "name": img["name"],
                "r_xmp": r_xmp,
                "pos_rs": center_rs,
                "focal_mm": focal_mm,
            })

        if getattr(args, "export_transforms", False):
            c2w_gl = colmap_pose_to_c2w_gl(r_wc, t_wc)
            c2w_gl = apply_x_fix_gl(c2w_gl, args.transforms_x_fix_deg)
            tf_frames.append({
                "file_path": img["name"],
                "transform_matrix": c2w_gl,
            })

    if getattr(args, "export_csv", False):
        out_csv = out_dir / args.csv_name
        write_realityscan_csv(out_csv, csv_rows)
        print("[OK] RealityScan CSV:", out_csv)

    if getattr(args, "export_ply", False):
        if points:
            out_ply = out_dir / args.ply_name
            vertices, prop_names = points_to_rs_vertices(points)
            write_ply_vertices(out_ply, vertices, prop_names)
            print("[OK] RealityScan PLY:", out_ply)
        else:
            print(
                "[WARN] points3D.txt not found or empty; PLY skipped",
                file=sys.stderr,
            )
    if getattr(args, "export_transforms_ply", False):
        if points:
            out_ply = out_dir / args.transforms_ply_name
            vertices, prop_names = points_to_transforms_ply_vertices(points)
            write_ply_vertices(out_ply, vertices, prop_names)
            print("[OK] transforms PLY:", out_ply)
        else:
            print(
                "[WARN] points3D.txt not found or empty; "
                "transforms PLY skipped",
                file=sys.stderr,
            )

    if getattr(args, "export_transforms", False):
        out_tf = out_dir / args.transforms_name
        write_transforms_json(out_tf, tf_frames, intr_ref)
        print("[OK] transforms.json:", out_tf)

    if getattr(args, "export_xmp", False):
        xmp_dir = out_dir / args.xmp_dir_name
        export_realityscan_xmp(xmp_dir, xmp_records)
        print("[OK] RealityScan XMP:", xmp_dir)

    if getattr(args, "export_metashape_xml", False):
        out_xml = out_dir / args.metashape_xml_name
        export_metashape_perspective_xml(out_xml, cam_list, images)
        print("[OK] Metashape XML:", out_xml)


def colmap_to_rs(args):
    colmap_dir = pathlib.Path(args.colmap_dir)
    cameras = parse_colmap_cameras_txt(colmap_dir / "cameras.txt")
    images = parse_colmap_images_txt(colmap_dir / "images.txt")
    points = parse_colmap_points3d_txt(colmap_dir / "points3D.txt")
    if not cameras or not images:
        raise ValueError("missing COLMAP text files in {}".format(colmap_dir))
    images, points = apply_optional_scene_rotations(args, images, points)

    out_dir = pathlib.Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if getattr(args, "export_colmap", False):
        out_colmap = get_export_colmap_dir(out_dir)
        write_colmap_text_model(out_colmap, list(cameras.values()), images, points)
        print("[OK] COLMAP text:", out_colmap)
    export_from_colmap_model(args, cameras, images, points, out_dir)


def build_colmap_from_transforms_json(args):
    frames, intr = read_transforms_json(args.transforms_json)
    fx, fy, cx, cy, w, h = intr
    focal_mm = focal_pixels_to_mm(
        fx, fy, w, h, args.sensor_width_mm, args.sensor_height_mm
    )
    rs_rows = []
    for fr in frames:
        c2w_gl = fr["transform_matrix"]
        c2w_gl = apply_x_fix_gl(c2w_gl, -args.transforms_x_fix_deg)
        r_wc, t_wc = c2w_gl_to_colmap_pose(c2w_gl)
        center_colmap = camera_center_from_colmap_pose(r_wc, t_wc)
        center_rs = colmap_world_to_rs_world(center_colmap)
        r_xmp = colmap_pose_rot_to_rs_rot(r_wc)
        heading, pitch, roll = rs_rotation_to_hpr(r_xmp)
        rs_rows.append(
            make_rs_csv_row(
                fr["file_path"],
                center_rs[0],
                center_rs[1],
                center_rs[2],
                heading,
                pitch,
                roll,
                focal_mm,
            )
        )
    return build_colmap_from_rs_rows(rs_rows, args, w, h)


def build_colmap_from_rs_csv(args):
    rows = read_realityscan_csv(args.csv)
    if args.width is None or args.height is None:
        raise ValueError(
            "--width and --height are required for --pose-source csv"
        )
    return build_colmap_from_rs_rows(
        rows,
        args,
        int(args.width),
        int(args.height),
    )


def build_colmap_from_rs_xmp(args):
    if args.width is None or args.height is None:
        if not args.image_dir:
            raise ValueError(
                "--width and --height are required for --pose-source xmp "
                "(or specify --image-dir)"
            )
        w, h = infer_image_size_from_dir(args.image_dir)
    else:
        w = int(args.width)
        h = int(args.height)
    xmp_rows = read_realityscan_xmp_dir(
        args.xmp_dir,
        image_ext=args.xmp_image_ext,
    )
    image_name_map = map_stem_to_image_name(args.image_dir)
    rs_rows = []
    for row in xmp_rows:
        heading, pitch, roll = rs_rotation_to_hpr(row["r_xmp"])
        pos = row["pos_rs"]
        rs_rows.append(
            make_rs_csv_row(
                row["name"],
                pos[0],
                pos[1],
                pos[2],
                heading,
                pitch,
                roll,
                row["focal_mm"],
            )
        )
    return build_colmap_from_rs_rows(
        rs_rows,
        args,
        w,
        h,
        image_name_map=image_name_map,
    )


def build_colmap_from_metashape_xml(args):
    image_name_map = map_stem_to_image_name(args.image_dir)
    rs_rows, w, h = read_metashape_perspective_xml(
        args.metashape_xml,
        args,
        image_name_map=image_name_map,
    )
    return build_colmap_from_rs_rows(
        rs_rows,
        args,
        w,
        h,
        image_name_map=image_name_map,
    )


def rs_to_colmap(args):
    if args.pose_source == "auto":
        if args.transforms_json:
            pose_source = "transforms"
        elif args.csv:
            pose_source = "csv"
        elif args.xmp_dir:
            pose_source = "xmp"
        elif args.metashape_xml:
            pose_source = "metashape-xml"
        else:
            raise ValueError(
                "specify --transforms-json or --csv or --xmp-dir or "
                "--metashape-xml"
            )
    else:
        pose_source = args.pose_source

    if pose_source == "transforms":
        if not args.transforms_json:
            raise ValueError("--transforms-json is required")
        cameras, images = build_colmap_from_transforms_json(args)
    elif pose_source == "csv":
        if not args.csv:
            raise ValueError("--csv is required")
        cameras, images = build_colmap_from_rs_csv(args)
    elif pose_source == "xmp":
        if not args.xmp_dir:
            raise ValueError("--xmp-dir is required")
        cameras, images = build_colmap_from_rs_xmp(args)
    else:
        if not args.metashape_xml:
            raise ValueError("--metashape-xml is required")
        cameras, images = build_colmap_from_metashape_xml(args)

    points = []
    if getattr(args, "transforms_ply", None):
        vertices, _ = read_ply_vertices(pathlib.Path(args.transforms_ply))
        points = transforms_ply_vertices_to_points(
            vertices, args.point_id_start
        )
    elif args.ply:
        vertices, _ = read_ply_vertices(pathlib.Path(args.ply))
        points = rs_vertices_to_points(vertices, args.point_id_start)
    images, points = apply_optional_scene_rotations(args, images, points)

    if getattr(args, "export_colmap", True):
        out_colmap = get_export_colmap_dir(args.out)
        write_colmap_text_model(out_colmap, cameras, images, points)
        print("[OK] COLMAP text:", out_colmap)

    if any([
        bool(getattr(args, "export_colmap", False)),
        bool(getattr(args, "export_csv", False)),
        bool(getattr(args, "export_ply", False)),
        bool(getattr(args, "export_transforms", False)),
        bool(getattr(args, "export_transforms_ply", False)),
        bool(getattr(args, "export_xmp", False)),
        bool(getattr(args, "export_metashape_xml", False)),
    ]):
        out_dir = pathlib.Path(args.out).expanduser().resolve()
        export_from_colmap_model(args, cameras, images, points, out_dir)


def _add_r2c_common_output_args(parser, allow_ply_input=True):
    parser.add_argument(
        "-o", "--out", required=True,
        help=(
            "Output root directory. COLMAP text is written to "
            "`--out/COLMAP_text_export` when `--export-colmap` is enabled."
        ),
    )
    parser.add_argument(
        "--image-dir",
        default=None,
        help=(
            "Image folder for XMP/Metashape XML import: infer width/height "
            "from images and prefer actual file names by stem match"
        ),
    )
    if allow_ply_input:
        parser.add_argument(
            "--realityscan-ply", "--ply",
            dest="ply",
            default=None,
            help=(
                "RealityScan pointcloud PLY input (RealityScan PLY axis). "
                "Optional companion pointcloud for the input camera poses."
            ),
        )
    parser.add_argument(
        "--transforms-x-fix-deg",
        type=float,
        default=DEFAULT_TRANSFORMS_X_FIX_DEG,
    )
    parser.add_argument(
        "--sensor-width-mm", type=float, default=DEFAULT_SENSOR_W_MM
    )
    parser.add_argument(
        "--sensor-height-mm", type=float, default=DEFAULT_SENSOR_H_MM
    )
    parser.add_argument("--single-camera", action="store_true")
    parser.add_argument("--point-id-start", type=int, default=0)
    parser.add_argument(
        "--camera-rot-x-deg",
        type=float,
        default=0.0,
        help="Rotate camera world around X before export (degrees)",
    )
    parser.add_argument(
        "--camera-rot-y-deg",
        type=float,
        default=0.0,
        help="Rotate camera world around Y before export (degrees)",
    )
    parser.add_argument(
        "--camera-rot-z-deg",
        type=float,
        default=0.0,
        help="Rotate camera world around Z before export (degrees)",
    )
    parser.add_argument(
        "--pointcloud-rot-x-deg",
        type=float,
        default=0.0,
        help="Rotate pointcloud around X before export (degrees)",
    )
    parser.add_argument(
        "--pointcloud-rot-y-deg",
        type=float,
        default=0.0,
        help="Rotate pointcloud around Y before export (degrees)",
    )
    parser.add_argument(
        "--pointcloud-rot-z-deg",
        type=float,
        default=0.0,
        help="Rotate pointcloud around Z before export (degrees)",
    )
    parser.add_argument(
        "--camera-scale",
        type=float,
        default=1.0,
        help="Scale camera world around origin before export",
    )
    parser.add_argument(
        "--pointcloud-scale",
        type=float,
        default=1.0,
        help="Scale pointcloud around origin before export",
    )
    parser.set_defaults(
        csv=None,
        xmp_dir=None,
        transforms_json=None,
        metashape_xml=None,
        ply=None,
        transforms_ply=None,
        export_colmap=False,
        xmp_image_ext="jpg",
        metashape_xml_image_ext="jpg",
    )


def _add_optional_export_args(parser):
    parser.add_argument(
        "--export-colmap",
        dest="export_colmap",
        action="store_true",
        default=False,
        help=(
            "Export COLMAP text to `--out/COLMAP_text_export`. "
            "If any `--export-*` is "
            "specified, only selected outputs are written."
        ),
    )
    parser.add_argument(
        "--realityscan-csv-file",
        dest="csv_name",
        default="Align_RS_PerspCams.csv",
        help="RealityScan camera CSV output file name (relative to --out)",
    )
    parser.add_argument(
        "--realityscan-ply-file",
        dest="ply_name",
        default="Align_RS_PerspCams.ply",
        help="RealityScan pointcloud PLY output file name (relative to --out)",
    )
    parser.add_argument(
        "--transforms-json-file",
        dest="transforms_name",
        default="transforms.json",
        help="transforms.json output file name (relative to --out)",
    )
    parser.add_argument(
        "--transforms-ply-file",
        dest="transforms_ply_name",
        default="pointcloud_for_transforms.ply",
        help=(
            "transforms companion PLY output file name (relative to --out, "
            "`pointcloud_for_transforms.ply` axis)"
        ),
    )
    parser.add_argument(
        "--realityscan-xmp-output-dir",
        "--realityscan-xmp-dir-name",
        dest="xmp_dir_name",
        default="cameras_RealityScan",
        help="RealityScan XMP output directory name",
    )
    parser.add_argument(
        "--metashape-xml-file",
        dest="metashape_xml_name",
        default="perspective_cams.xml",
        help="Metashape perspective camera XML output file name",
    )
    parser.add_argument(
        "--export-realityscan-csv",
        dest="export_csv",
        action="store_true",
        default=False,
        help=(
            "Export RealityScan camera CSV. If any `--export-*` is "
            "specified, only selected outputs are written."
        ),
    )
    parser.add_argument(
        "--export-realityscan-ply",
        dest="export_ply",
        action="store_true",
        default=False,
        help=(
            "Export RealityScan PLY (RealityScan PLY axis). Requires input "
            "pointcloud. If any `--export-*` is specified, only selected "
            "outputs are written."
        ),
    )
    parser.add_argument(
        "--export-transforms-json",
        dest="export_transforms",
        action="store_true",
        default=False,
        help=(
            "Export transforms.json. If any `--export-*` is specified, only "
            "selected outputs are written."
        ),
    )
    parser.add_argument(
        "--export-transforms-ply",
        dest="export_transforms_ply",
        action="store_true",
        default=False,
        help=(
            "Export companion PLY for transforms.json "
            "(`pointcloud_for_transforms.ply` axis). "
            "Requires input pointcloud."
        ),
    )
    parser.add_argument(
        "--export-realityscan-xmp",
        dest="export_xmp",
        action="store_true",
        default=False,
        help=(
            "Export RealityScan XMP camera files. If any `--export-*` is "
            "specified, only selected outputs are written."
        ),
    )
    parser.add_argument(
        "--export-metashape-xml",
        dest="export_metashape_xml",
        action="store_true",
        default=False,
        help=(
            "Export Metashape perspective camera XML. If any `--export-*` is "
            "specified, only selected outputs are written."
        ),
    )


def build_arg_parser():
    ap = argparse.ArgumentParser(
        description=(
            "Camera format converter between COLMAP, RealityScan "
            "(CSV/PLY/XMP), transforms.json, and Metashape perspective XML.\n"
            "Input is selected by subcommand name. For non-`colmap` inputs, "
            "if no `--export-*` is specified, all camera formats are "
            "exported by default (plus PLY variants when pointcloud input is "
            "available)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd")
    sub.required = True

    c2r = sub.add_parser(
        "colmap",
        aliases=["colmap-to-rs"],
        help=(
            "Input: COLMAP text. Output: selectable camera/pointcloud "
            "formats."
        ),
        description=(
            "Input COLMAP text model and export selected formats "
            "(RealityScan CSV/PLY/XMP, transforms.json(+PLY), "
            "Metashape perspective XML)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    c2r.add_argument("colmap_dir")
    c2r.add_argument("-o", "--out", required=True)
    c2r.add_argument(
        "--sensor-width-mm", type=float, default=DEFAULT_SENSOR_W_MM
    )
    c2r.add_argument(
        "--sensor-height-mm", type=float, default=DEFAULT_SENSOR_H_MM
    )
    c2r.add_argument(
        "--camera-rot-x-deg",
        type=float,
        default=0.0,
        help="Rotate camera world around X before export (degrees)",
    )
    c2r.add_argument(
        "--camera-rot-y-deg",
        type=float,
        default=0.0,
        help="Rotate camera world around Y before export (degrees)",
    )
    c2r.add_argument(
        "--camera-rot-z-deg",
        type=float,
        default=0.0,
        help="Rotate camera world around Z before export (degrees)",
    )
    c2r.add_argument(
        "--pointcloud-rot-x-deg",
        type=float,
        default=0.0,
        help="Rotate pointcloud around X before export (degrees)",
    )
    c2r.add_argument(
        "--pointcloud-rot-y-deg",
        type=float,
        default=0.0,
        help="Rotate pointcloud around Y before export (degrees)",
    )
    c2r.add_argument(
        "--pointcloud-rot-z-deg",
        type=float,
        default=0.0,
        help="Rotate pointcloud around Z before export (degrees)",
    )
    c2r.add_argument(
        "--camera-scale",
        type=float,
        default=1.0,
        help="Scale camera world around origin before export",
    )
    c2r.add_argument(
        "--pointcloud-scale",
        type=float,
        default=1.0,
        help="Scale pointcloud around origin before export",
    )
    c2r.add_argument(
        "--transforms-x-fix-deg",
        type=float,
        default=DEFAULT_TRANSFORMS_X_FIX_DEG,
    )
    c2r.add_argument(
        "--realityscan-csv-file", "--realityscan-csv-name", "--csv-name",
        dest="csv_name",
        default="Align_RS_PerspCams.csv",
        help="RealityScan camera CSV output file name (relative to --out)",
    )
    c2r.add_argument(
        "--realityscan-ply-file", "--realityscan-ply-name", "--ply-name",
        dest="ply_name",
        default="Align_RS_PerspCams.ply",
        help="RealityScan pointcloud PLY output file name (relative to --out)",
    )
    c2r.add_argument(
        "--transforms-json-file",
        "--transforms-json-name",
        "--transforms-name",
        dest="transforms_name",
        default="transforms.json",
        help="transforms.json output file name (relative to --out)",
    )
    c2r.add_argument(
        "--transforms-ply-file",
        dest="transforms_ply_name",
        default="pointcloud_for_transforms.ply",
        help="transforms companion PLY output file name (relative to --out)",
    )
    c2r.add_argument(
        "--realityscan-xmp-dir", "--xmp-dir-name",
        dest="xmp_dir_name",
        default="cameras_RealityScan",
        help="RealityScan XMP output directory name",
    )
    c2r.add_argument(
        "--metashape-xml-file",
        dest="metashape_xml_name",
        default="perspective_cams.xml",
        help="Metashape perspective camera XML output file name",
    )
    c2r.add_argument(
        "--export-colmap",
        dest="export_colmap",
        action="store_true",
        default=False,
        help=(
            "Export COLMAP text to `--out/COLMAP_text_export`. Useful when writing "
            "a rotated or scaled copy of the input COLMAP scene."
        ),
    )
    c2r.add_argument(
        "--export-realityscan-csv", "--export-csv",
        dest="export_csv",
        action="store_true",
        default=False,
        help=(
            "Export RealityScan camera CSV (camera set). If any --export-* "
            "is specified, only specified exports are written."
        ),
    )
    c2r.add_argument(
        "--export-realityscan-ply", "--export-ply",
        dest="export_ply",
        action="store_true",
        default=False,
        help=(
            "Export RealityScan pointcloud PLY (companion to RealityScan "
            "CSV or transforms.json set). If any --export-* is specified, "
            "only specified exports are written."
        ),
    )
    c2r.add_argument(
        "--export-transforms-json", "--export-transforms",
        dest="export_transforms",
        action="store_true",
        default=False,
        help=(
            "Export transforms.json. If any --export-* is specified, only "
            "specified exports are written."
        ),
    )
    c2r.add_argument(
        "--export-transforms-ply",
        dest="export_transforms_ply",
        action="store_true",
        default=False,
        help=(
            "Export companion PLY for transforms.json (legacy transforms "
            "axis). If any --export-* is specified, only specified exports "
            "are written."
        ),
    )
    c2r.add_argument(
        "--export-realityscan-xmp", "--export-xmp",
        dest="export_xmp",
        action="store_true",
        default=False,
        help=(
            "Export RealityScan XMP camera files (independent camera info). "
            "If any --export-* is specified, only specified exports are "
            "written."
        ),
    )
    c2r.add_argument(
        "--export-metashape-xml",
        dest="export_metashape_xml",
        action="store_true",
        default=False,
        help=(
            "Export Metashape perspective camera XML (independent camera "
            "info). If any --export-* is specified, only specified exports "
            "are written."
        ),
    )
    c2r.set_defaults(func=colmap_to_rs)

    rs_csv = sub.add_parser(
        "realityscan-csv",
        help="Input: RealityScan CSV (+ optional RealityScan PLY).",
        description=(
            "Input RealityScan CSV camera poses (and optional RealityScan "
            "PLY pointcloud), build an internal COLMAP model, and export "
            "selected outputs. If no `--export-*` is specified, exports "
            "COLMAP + all camera formats by default."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_r2c_common_output_args(rs_csv)
    _add_optional_export_args(rs_csv)
    rs_csv.add_argument(
        "--realityscan-csv", "--csv",
        dest="csv",
        required=True,
        help="RealityScan camera CSV input file",
    )
    rs_csv.add_argument("--width", type=int, required=True)
    rs_csv.add_argument("--height", type=int, required=True)
    rs_csv.set_defaults(pose_source="csv", func=rs_to_colmap)

    rs_xmp = sub.add_parser(
        "realityscan-xmp",
        help="Input: RealityScan XMP dir (+ optional RealityScan PLY).",
        description=(
            "Input RealityScan XMP camera files (and optional RealityScan "
            "PLY pointcloud), build an internal COLMAP model, and export "
            "selected outputs. Use `--image-dir` or `--width/--height` when "
            "XMP does not provide image size."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_r2c_common_output_args(rs_xmp)
    _add_optional_export_args(rs_xmp)
    rs_xmp.add_argument(
        "--realityscan-xmp-dir", "--xmp-dir",
        dest="xmp_dir",
        required=True,
        help="RealityScan XMP input directory",
    )
    rs_xmp.add_argument(
        "--realityscan-xmp-image-ext", "--xmp-image-ext",
        dest="xmp_image_ext",
        default="jpg",
        help="Image extension used when deriving names from XMP files",
    )
    rs_xmp.add_argument("--width", type=int, default=None)
    rs_xmp.add_argument("--height", type=int, default=None)
    rs_xmp.set_defaults(pose_source="xmp", func=rs_to_colmap)

    tfj = sub.add_parser(
        "transforms-json",
        help=(
            "Input: transforms.json (+ optional PLY in RS or transforms "
            "axis)."
        ),
        description=(
            "Input transforms.json camera poses with optional companion "
            "pointcloud. `--transforms-ply` expects "
            "`pointcloud_for_transforms.ply` axis. "
            "`--realityscan-ply` expects "
            "RealityScan PLY axis."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_r2c_common_output_args(tfj)
    _add_optional_export_args(tfj)
    tfj.add_argument("--transforms-json", required=True)
    tfj.add_argument(
        "--transforms-ply",
        default=None,
        help="Companion PLY for transforms.json (legacy transforms axis)",
    )
    tfj.add_argument("--width", type=int, default=None)
    tfj.add_argument("--height", type=int, default=None)
    tfj.set_defaults(pose_source="transforms", func=rs_to_colmap)

    msx = sub.add_parser(
        "metashape-xml",
        help="Input: Metashape perspective XML (+ optional RealityScan PLY).",
        description=(
            "Input Metashape perspective camera XML with optional RealityScan "
            "PLY pointcloud, build an internal COLMAP model, and "
            "export selected outputs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_r2c_common_output_args(msx, allow_ply_input=True)
    _add_optional_export_args(msx)
    msx.add_argument(
        "--metashape-xml",
        required=True,
        help="Metashape perspective camera XML input file",
    )
    msx.add_argument(
        "--metashape-xml-image-ext",
        default="jpg",
        help="Extension for Metashape camera labels without suffix",
    )
    msx.add_argument("--width", type=int, default=None)
    msx.add_argument("--height", type=int, default=None)
    msx.set_defaults(pose_source="metashape-xml", func=rs_to_colmap)

    return ap


def main():
    ap = build_arg_parser()
    args = ap.parse_args()
    if args.cmd in ("colmap", "colmap-to-rs"):
        any_export_selected = any([
            bool(getattr(args, "export_colmap", False)),
            bool(args.export_csv),
            bool(args.export_ply),
            bool(args.export_transforms),
            bool(args.export_transforms_ply),
            bool(args.export_xmp),
            bool(args.export_metashape_xml),
        ])
        if not any_export_selected:
            # Default set: RealityScan CSV + optional PLY
            args.export_csv = True
            args.export_ply = True
            args.export_transforms = False
            args.export_transforms_ply = False
            args.export_xmp = False
            args.export_metashape_xml = False
    elif hasattr(args, "export_colmap"):
        any_export_selected = any([
            bool(args.export_colmap),
            bool(args.export_csv),
            bool(args.export_ply),
            bool(args.export_transforms),
            bool(args.export_transforms_ply),
            bool(args.export_xmp),
            bool(args.export_metashape_xml),
        ])
        if not any_export_selected:
            # Default for non-COLMAP inputs: emit all camera formats.
            # PLY variants are enabled only when a pointcloud input exists.
            has_any_pointcloud_input = bool(
                getattr(args, "ply", None) or
                getattr(args, "transforms_ply", None)
            )
            args.export_colmap = True
            args.export_csv = True
            args.export_transforms = True
            args.export_xmp = True
            args.export_metashape_xml = True
            args.export_ply = has_any_pointcloud_input
            args.export_transforms_ply = has_any_pointcloud_input
    try:
        args.func(args)
    except Exception as exc:
        print("[ERR] {}".format(exc), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
