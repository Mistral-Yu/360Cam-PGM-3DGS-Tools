#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert Metashape camera XML (spherical) into virtual perspective cameras.

This tool reads Metashape's cameras_XML.xml, builds virtual camera views based
on the 360PerspCut presets (default / fisheyelike / full360coverage), and
exports either transforms.json (LichtFeld Studio / PostShot style) or
COLMAP text model files. Image cutting is NOT performed here.

Assumptions:
- Metashape camera transform is camera-to-world in OpenCV camera coords
  (x right, y down, z forward).
- transforms.json uses OpenGL camera coords (x right, y up, z back).
- World units default to meters (scale=1). Use --scale 100 for cm.

Adjust the WORLD_FROM_METASHAPE matrix if a world-axis conversion is needed.
"""

import argparse
import json
import math
import pathlib
import subprocess
import sys
import xml.etree.ElementTree as ET


PRESET_DEFAULT = "default"
PRESET_FISHEYELIKE = "fisheyelike"
PRESET_FULL360 = "full360coverage"
PRESET_2VIEWS = "2views"
PRESET_EVEN_MINUS = "evenMinus30"
PRESET_EVEN_PLUS = "evenPlus30"
PRESET_CUBE = "cube90"
PRESET_CHOICES = [
    PRESET_DEFAULT,
    PRESET_FISHEYELIKE,
    PRESET_FULL360,
    PRESET_2VIEWS,
    PRESET_EVEN_MINUS,
    PRESET_EVEN_PLUS,
    PRESET_CUBE,
]

# Intrinsics derived from gs360_360PerspCut defaults.
SENSOR_W_MM = 36.0
SENSOR_H_MM = 36.0
DEFAULT_SIZE = 1600
DEFAULT_FOCAL_MM = 12.0
FISHEYELIKE_FOCAL_MM = 17.0
FULL360_FOCAL_MM = 14.0
ADD_CAM_DEG = 30.0
CUBE_FOV_DEG = 90.0
TRANSFORMS_X_FIX_DEG = 270.0
COLMAP_X_BASE_DEG = 0.0
COLMAP_POINTS_X_DEG = 90.0
POINTCLOUD_PLY_X_DEG = 180.0

# OpenCV camera coords -> OpenGL camera coords.
CV_TO_GL = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]


def letter_tag(idx):
    base = ord("A")
    return chr(base + idx) if idx < 26 else f"{idx + 1:02d}"


def normalize_angle_deg(angle):
    angle = ((angle + 180.0) % 360.0) - 180.0
    return 180.0 if abs(angle + 180.0) < 1e-6 else angle


def extra_suffix(delta_pitch, default_deg=30.0):
    sign = "_U" if delta_pitch > 0 else "_D"
    mag = abs(delta_pitch)
    if abs(mag - default_deg) < 1e-6:
        return sign
    if float(mag).is_integer():
        return f"{sign}{int(round(mag))}"
    return f"{sign}{mag:g}"


def mat4_mul(a, b):
    out = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            out[i][j] = sum(a[i][k] * b[k][j] for k in range(4))
    return out


def mat3_mul(a, b):
    out = [[0.0] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            out[i][j] = sum(a[i][k] * b[k][j] for k in range(3))
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


def rot_x_deg(deg):
    rad = math.radians(deg)
    c = math.cos(rad)
    s = math.sin(rad)
    return [
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s, c],
    ]


def rot_y_deg(deg):
    rad = math.radians(deg)
    c = math.cos(rad)
    s = math.sin(rad)
    return [
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c],
    ]


def axis_angle_to_mat3(axis, deg):
    x, y, z = axis
    norm = math.sqrt(x * x + y * y + z * z)
    if norm <= 0.0 or abs(deg) < 1e-12:
        return [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    x /= norm
    y /= norm
    z /= norm
    rad = math.radians(deg)
    c = math.cos(rad)
    s = math.sin(rad)
    t = 1.0 - c
    return [
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ]


def mat3_to_mat4_with_translation(r, tvec=None):
    if tvec is None:
        tvec = (0.0, 0.0, 0.0)
    return [
        [r[0][0], r[0][1], r[0][2], tvec[0]],
        [r[1][0], r[1][1], r[1][2], tvec[1]],
        [r[2][0], r[2][1], r[2][2], tvec[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def yaw_pitch_to_rot_gl(yaw_deg, pitch_deg):
    yaw = -float(yaw_deg)   # positive yaw to the right
    pitch = float(pitch_deg)
    r_yaw = rot_y_deg(yaw)
    r_pitch = rot_x_deg(pitch)
    return mat3_mul(r_yaw, r_pitch)


def mat3_to_mat4(r):
    return [
        [r[0][0], r[0][1], r[0][2], 0.0],
        [r[1][0], r[1][1], r[1][2], 0.0],
        [r[2][0], r[2][1], r[2][2], 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def apply_x_fix_gl(c2w_gl, deg):
    if deg is None or abs(deg) < 1e-6:
        return c2w_gl
    fix_rot = mat3_to_mat4(rot_x_deg(deg))
    return mat4_mul(fix_rot, c2w_gl)


def compute_colmap_images(frames, x_fix_deg):
    images = []
    image_id = 1
    for frame in frames:
        c2w_gl = apply_x_fix_gl(frame["c2w_gl"], x_fix_deg)
        c2w_cv = mat4_mul(c2w_gl, CV_TO_GL)
        r_wc = mat3_transpose([row[:3] for row in c2w_cv[:3]])
        t_wc = c2w_cv[0][3], c2w_cv[1][3], c2w_cv[2][3]
        t = mat3_vec_mul(r_wc, [-t_wc[0], -t_wc[1], -t_wc[2]])
        qw, qx, qy, qz = rotmat_to_quat_wxyz(r_wc)
        images.append({
            "image_id": image_id,
            "qw": qw,
            "qx": qx,
            "qy": qy,
            "qz": qz,
            "tx": t[0],
            "ty": t[1],
            "tz": t[2],
            "name": frame["file_path"],
        })
        image_id += 1
    return images


def apply_unit_scale(mat, scale):
    out = [row[:] for row in mat]
    for i in range(3):
        out[i][3] *= scale
    return out


def parse_camera_transform(text):
    values = [float(x) for x in text.strip().split()]
    if len(values) != 16:
        raise ValueError("transform must have 16 floats")
    return [
        values[0:4],
        values[4:8],
        values[8:12],
        values[12:16],
    ]


def load_metashape_cameras(xml_path):
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    chunk = root.find("chunk")
    if chunk is None:
        raise ValueError("missing <chunk> in XML")
    cameras_node = chunk.find("cameras")
    if cameras_node is None:
        raise ValueError("missing <cameras> in XML")

    cameras = []
    for cam in cameras_node.findall("camera"):
        enabled = cam.get("enabled")
        if enabled is not None and enabled.lower() == "false":
            continue
        transform_node = cam.find("transform")
        if transform_node is None or not (transform_node.text or "").strip():
            continue
        label = cam.get("label") or f"camera_{cam.get('id', '0')}"
        cam_id = int(cam.get("id", "0"))
        mat = parse_camera_transform(transform_node.text)
        cameras.append((cam_id, label, mat))
    cameras.sort(key=lambda x: x[0])
    return cameras


def focal_from_hfov_deg(hfov_deg, sensor_w_mm):
    return sensor_w_mm / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))


def preset_config(preset_name):
    if preset_name == PRESET_DEFAULT:
        return {
            "count": 8,
            "focal_mm": DEFAULT_FOCAL_MM,
            "size": DEFAULT_SIZE,
            "del_letters": [],
            "add_letters": [],
            "even_pitch": None,
            "hfov_deg": None,
            "explicit_views": None,
        }
    if preset_name == PRESET_FISHEYELIKE:
        return {
            "count": 10,
            "focal_mm": FISHEYELIKE_FOCAL_MM,
            "size": DEFAULT_SIZE,
            "del_letters": ["C", "D", "H", "I"],
            "add_letters": ["A", "F"],
            "even_pitch": None,
            "hfov_deg": None,
            "explicit_views": None,
        }
    if preset_name == PRESET_FULL360:
        return {
            "count": 8,
            "focal_mm": FULL360_FOCAL_MM,
            "size": DEFAULT_SIZE,
            "del_letters": ["B", "D", "F", "H"],
            "add_letters": ["B", "D", "F", "H"],
            "even_pitch": None,
            "hfov_deg": None,
            "explicit_views": None,
        }
    if preset_name == PRESET_2VIEWS:
        return {
            "count": 8,
            "focal_mm": 6.0,
            "size": 3600,
            "del_letters": ["B", "C", "D", "F", "G", "H"],
            "add_letters": [],
            "even_pitch": None,
            "hfov_deg": None,
            "explicit_views": None,
        }
    if preset_name == PRESET_EVEN_MINUS:
        return {
            "count": 8,
            "focal_mm": DEFAULT_FOCAL_MM,
            "size": DEFAULT_SIZE,
            "del_letters": [],
            "add_letters": [],
            "even_pitch": -30.0,
            "hfov_deg": None,
            "explicit_views": None,
        }
    if preset_name == PRESET_EVEN_PLUS:
        return {
            "count": 8,
            "focal_mm": DEFAULT_FOCAL_MM,
            "size": DEFAULT_SIZE,
            "del_letters": [],
            "add_letters": [],
            "even_pitch": 30.0,
            "hfov_deg": None,
            "explicit_views": None,
        }
    if preset_name == PRESET_CUBE:
        return {
            "count": 6,
            "focal_mm": None,
            "size": DEFAULT_SIZE,
            "del_letters": [],
            "add_letters": [],
            "even_pitch": None,
            "hfov_deg": CUBE_FOV_DEG,
            "explicit_views": [
                ("A", 0.0, 0.0),
                ("B", 90.0, 0.0),
                ("C", 180.0, 0.0),
                ("D", -90.0, 0.0),
                ("E", 0.0, 90.0),
                ("F", 0.0, -90.0),
            ],
        }
    raise ValueError("unknown preset: " + preset_name)


def build_views(preset_name):
    cfg = preset_config(preset_name)
    if cfg.get("explicit_views"):
        return list(cfg["explicit_views"])
    count = cfg["count"]
    even_pitch = cfg["even_pitch"]
    del_set = {letter.upper() for letter in cfg["del_letters"]}
    add_set = {letter.upper() for letter in cfg["add_letters"]}
    yaw_step = 360.0 / float(count)
    views = []

    for idx in range(count):
        tag = letter_tag(idx)
        yaw = normalize_angle_deg(idx * yaw_step)
        pitch = 0.0
        if even_pitch is not None and ((idx + 1) % 2) == 0:
            pitch = float(even_pitch)

        if tag not in del_set:
            views.append((tag, yaw, pitch))

        if tag in add_set:
            for delta in (ADD_CAM_DEG, -ADD_CAM_DEG):
                p2 = max(-90.0, min(90.0, pitch + delta))
                suf = extra_suffix(delta, ADD_CAM_DEG)
                views.append((f"{tag}{suf}", yaw, p2))

    return views


def compute_intrinsics(focal_mm, width, height):
    pixel_size_w = SENSOR_W_MM / float(width)
    pixel_size_h = SENSOR_H_MM / float(height)
    fl_x = focal_mm / pixel_size_w
    fl_y = focal_mm / pixel_size_h
    cx = width * 0.5
    cy = height * 0.5
    hfov = math.degrees(2.0 * math.atan(SENSOR_W_MM / (2.0 * focal_mm)))
    vfov = math.degrees(2.0 * math.atan(SENSOR_H_MM / (2.0 * focal_mm)))
    return fl_x, fl_y, cx, cy, hfov, vfov


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

    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm == 0.0:
        return 1.0, 0.0, 0.0, 0.0
    return qw / norm, qx / norm, qy / norm, qz / norm


def safe_name(name):
    name = name.replace("\\", "_").replace("/", "_")
    return name.strip()


def extract_rot3(mat4):
    return [
        [mat4[0][0], mat4[0][1], mat4[0][2]],
        [mat4[1][0], mat4[1][1], mat4[1][2]],
        [mat4[2][0], mat4[2][1], mat4[2][2]],
    ]


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

    with ply_path.open("rb") as fp:
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
                for (name, typ), token in zip(props, parts):
                    if typ in ("float", "float32", "double", "float64"):
                        row[name] = float(token)
                    else:
                        row[name] = int(float(token))
                vertices.append(row)
        else:
            struct_fmt = "<" + "".join(prop_types)
            st = struct.Struct(struct_fmt)
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


def build_points_outputs(
    ply_path,
    out_dir,
    world_from_metashape,
    rotate_x_plus180,
    scale_cm,
):
    vertices, prop_names = read_ply_vertices(ply_path)
    has_color = (
        "red" in prop_names and "green" in prop_names and "blue" in prop_names
    )
    rot_world = extract_rot3(world_from_metashape)
    rot_ply = None
    if rotate_x_plus180:
        rot_ply = rot_x_deg(POINTCLOUD_PLY_X_DEG)

    points = []
    out_vertices = []
    for idx, vtx in enumerate(vertices, start=1):
        x = float(vtx.get("x", 0.0))
        y = float(vtx.get("y", 0.0))
        z = float(vtx.get("z", 0.0))
        rx, ry, rz = mat3_vec_mul(rot_world, [x, y, z])
        px, py, pz = rx, ry, rz
        if rot_ply is not None:
            px, py, pz = mat3_vec_mul(rot_ply, [px, py, pz])
        rx *= scale_cm
        ry *= scale_cm
        rz *= scale_cm
        px *= scale_cm
        py *= scale_cm
        pz *= scale_cm
        if has_color:
            r = int(vtx.get("red", 128))
            g = int(vtx.get("green", 128))
            b = int(vtx.get("blue", 128))
        else:
            r = g = b = 128
        points.append({
            "id": idx,
            "x": rx,
            "y": ry,
            "z": rz,
            "r": r,
            "g": g,
            "b": b,
            "err": 0.0,
        })
        out_row = {"x": px, "y": py, "z": pz}
        if has_color:
            out_row.update({"red": r, "green": g, "blue": b})
        out_vertices.append(out_row)

    if has_color:
        out_prop_names = ["x", "y", "z", "red", "green", "blue"]
    else:
        out_prop_names = ["x", "y", "z"]
    out_ply = out_dir / "pointcloud_rotated.ply"
    write_ply_vertices(out_ply, out_vertices, out_prop_names)
    print("[OK] Rotated pointcloud:", out_ply)
    return points


def export_transforms_json(out_path, frames, intrinsics, x_fix_deg=0.0):
    fl_x, fl_y, cx, cy, width, height = intrinsics
    frame_payload = []
    for frame in frames:
        c2w_gl = apply_x_fix_gl(frame["c2w_gl"], x_fix_deg)
        frame_payload.append({
            "file_path": frame["file_path"],
            "transform_matrix": c2w_gl,
        })
    payload = {
        "camera_model": "OPENCV",
        "fl_x": fl_x,
        "fl_y": fl_y,
        "cx": cx,
        "cy": cy,
        "w": width,
        "h": height,
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "frames": frame_payload,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def export_colmap(out_dir, images, intrinsics, points):
    out_dir.mkdir(parents=True, exist_ok=True)
    cam_path = out_dir / "cameras.txt"
    img_path = out_dir / "images.txt"
    pts_path = out_dir / "points3D.txt"

    fl_x, fl_y, cx, cy, width, height = intrinsics
    with cam_path.open("w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: 1\n")
        f.write(
            "1 PINHOLE {} {} {:.12g} {:.12g} {:.12g} {:.12g}\n".format(
                width, height, fl_x, fl_y, cx, cy
            )
        )

    with img_path.open("w", encoding="utf-8") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write(
            "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
        )
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(
            "# Number of images: {}, mean observations per image: 0\n".format(
                len(images)
            )
        )
        for img in images:
            line = (
                "{image_id} {qw:.12g} {qx:.12g} {qy:.12g} {qz:.12g} "
                "{tx:.12g} {ty:.12g} {tz:.12g} 1 {name}\n"
            ).format(
                image_id=img["image_id"],
                qw=img["qw"],
                qx=img["qx"],
                qy=img["qy"],
                qz=img["qz"],
                tx=img["tx"],
                ty=img["ty"],
                tz=img["tz"],
                name=img["name"],
            )
            f.write(line)
            f.write("\n")

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
                "{pid} {x:.12g} {y:.12g} {z:.12g} {r} {g} {b} "
                "{err:.6g}\n".format(
                    pid=pt["id"],
                    x=pt["x"],
                    y=pt["y"],
                    z=pt["z"],
                    r=pt["r"],
                    g=pt["g"],
                    b=pt["b"],
                    err=pt["err"],
                )
            )


def export_metashape_xml(
    xml_in_path,
    out_path,
    frames,
    intrinsics,
    preset_name,
):
    fl_x, fl_y, _, _, width, height = intrinsics
    tree = ET.parse(str(xml_in_path))
    root = tree.getroot()
    chunk = root.find("chunk")
    if chunk is None:
        raise ValueError("missing <chunk> in XML")

    data_type = "uint8"
    black_level = "0 0 0"
    sensitivity = "1 1 1"
    sensors_node = chunk.find("sensors")
    if sensors_node is not None:
        first_sensor = sensors_node.find("sensor")
        if first_sensor is not None:
            data_type = (
                first_sensor.findtext("data_type") or data_type
            ).strip()
            black_level = (
                first_sensor.findtext("black_level") or black_level
            ).strip()
            sensitivity = (
                first_sensor.findtext("sensitivity") or sensitivity
            ).strip()

    if sensors_node is None:
        sensors_node = ET.SubElement(chunk, "sensors")
    sensors_node.clear()
    sensors_node.set("next_id", "1")

    sensor = ET.SubElement(
        sensors_node,
        "sensor",
        id="0",
        label="virtual_" + preset_name,
        type="frame",
    )
    ET.SubElement(
        sensor,
        "resolution",
        width=str(int(width)),
        height=str(int(height)),
    )
    ET.SubElement(sensor, "property", name="layer_index", value="0")
    calib = ET.SubElement(sensor, "calibration", type="frame")
    ET.SubElement(
        calib,
        "resolution",
        width=str(int(width)),
        height=str(int(height)),
    )
    ET.SubElement(calib, "f").text = f"{fl_x:.6f}"
    ET.SubElement(calib, "cx").text = "0"
    ET.SubElement(calib, "cy").text = "0"
    ET.SubElement(calib, "k1").text = "0"
    ET.SubElement(calib, "k2").text = "0"
    ET.SubElement(calib, "p1").text = "0"
    ET.SubElement(calib, "p2").text = "0"
    ET.SubElement(sensor, "data_type").text = data_type
    ET.SubElement(sensor, "black_level").text = black_level
    ET.SubElement(sensor, "sensitivity").text = sensitivity

    component_id = "0"
    src_cameras_node = chunk.find("cameras")
    if src_cameras_node is not None:
        first_camera = src_cameras_node.find("camera")
        if first_camera is not None:
            component_id = first_camera.get("component_id", component_id)
    cameras_node = src_cameras_node
    if cameras_node is None:
        cameras_node = ET.SubElement(chunk, "cameras")
    cameras_node.clear()
    cameras_node.set("next_id", str(len(frames)))
    cameras_node.set("next_group_id", "0")

    for idx, frame in enumerate(frames):
        label = pathlib.Path(frame["file_path"]).stem
        cam = ET.SubElement(
            cameras_node,
            "camera",
            id=str(idx),
            sensor_id="0",
            component_id=component_id,
            label=label,
        )
        mat = frame["c2w_cv"]
        flat = [mat[r][c] for r in range(4) for c in range(4)]
        cam_transform = " ".join("{:.15g}".format(v) for v in flat)
        ET.SubElement(cam, "transform").text = cam_transform

    tree.write(str(out_path), encoding="UTF-8", xml_declaration=True)


def build_outputs(
    cameras,
    preset_name,
    ext,
    scale_cm,
    world_from_metashape,
    world_axis,
    world_deg,
):
    views = build_views(preset_name)
    preset_cfg = preset_config(preset_name)
    width = height = int(preset_cfg["size"])
    focal_mm = preset_cfg["focal_mm"]
    if preset_cfg.get("hfov_deg") is not None:
        focal_mm = focal_from_hfov_deg(
            float(preset_cfg["hfov_deg"]),
            SENSOR_W_MM,
        )
    fl_x, fl_y, cx, cy, hfov, vfov = compute_intrinsics(
        focal_mm, width, height
    )
    intrinsics = (fl_x, fl_y, cx, cy, width, height)

    print(
        "[INFO] preset={} views={} focal_mm={}".format(
            preset_name, len(views), focal_mm
        )
    )
    print(
        "[INFO] intrinsics: size={}x{} hfov={:.2f} vfov={:.2f}".format(
            width, height, hfov, vfov
        )
    )
    print(f"[INFO] unit scale: 1.0m -> {scale_cm:.1f}cm")
    print(
        "[INFO] WORLD_FROM_METASHAPE axis=({:.6f} {:.6f} {:.6f}) deg={:.3f}"
        .format(world_axis[0], world_axis[1], world_axis[2], world_deg)
    )
    print("[INFO] WORLD_FROM_METASHAPE matrix:")
    for row in world_from_metashape:
        print("       " + " ".join(f"{v: .6f}" for v in row))
    print(
        "[INFO] transforms X fix: +{:.1f} deg".format(TRANSFORMS_X_FIX_DEG)
    )
    print(
        "[INFO] colmap X base: +{:.1f} deg".format(COLMAP_X_BASE_DEG)
    )
    print(
        "[INFO] colmap points X: +{:.1f} deg".format(COLMAP_POINTS_X_DEG)
    )
    print(
        "[INFO] pointcloud ply X: +{:.1f} deg".format(POINTCLOUD_PLY_X_DEG)
    )

    frames = []

    for _, label, mat in cameras:
        base_name = safe_name(label)
        mat_scaled = apply_unit_scale(mat, scale_cm)
        mat_world = mat4_mul(world_from_metashape, mat_scaled)
        base_gl = mat4_mul(mat_world, CV_TO_GL)

        for view_id, yaw, pitch in views:
            r_rel = yaw_pitch_to_rot_gl(yaw, pitch)
            r_rel4 = mat3_to_mat4(r_rel)
            c2w_gl = mat4_mul(base_gl, r_rel4)

            file_name = f"{base_name}_{view_id}.{ext}"
            c2w_cv = mat4_mul(c2w_gl, CV_TO_GL)
            frames.append({
                "file_path": file_name,
                "c2w_gl": c2w_gl,
                "c2w_cv": c2w_cv,
            })

    return frames, intrinsics


def build_arg_parser():
    ap = argparse.ArgumentParser(
        description="Convert Metashape 360 XML to virtual camera transforms.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "xml",
        help="Metashape cameras_XML.xml path",
    )
    ap.add_argument(
        "--preset",
        choices=PRESET_CHOICES,
        default=PRESET_FULL360,
        help="Virtual camera preset (matches gs360_360PerspCut)",
    )
    ap.add_argument(
        "-o", "--out",
        default=None,
        help=(
            "Output directory "
            "(default: <xml_dir>/perspective_cams)"
        ),
    )
    ap.add_argument(
        "--format",
        choices=["transforms", "colmap", "metashape", "all"],
        default="metashape",
        help="Output format (all=transforms+colmap+metashape)",
    )
    ap.add_argument(
        "--ext",
        default="jpg",
        help="Image extension for file paths (without dot)",
    )
    ap.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="World unit scale applied to translations (meters to cm = 100)",
    )
    ap.add_argument(
        "--world-rot-axis",
        default="0 1 0",
        help="World rotation axis (x y z) for Metashape->PostShot",
    )
    ap.add_argument(
        "--world-rot-deg",
        type=float,
        default=0.0,
        help="World rotation angle in degrees for Metashape->PostShot",
    )
    ap.add_argument(
        "--persp-cut",
        dest="cut",
        action="store_true",
        help="Run gs360_360PerspCut.py to cut perspective images",
    )
    ap.add_argument(
        "--cut",
        dest="cut",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--cut-input",
        default=None,
        help=(
            "Input folder for equirectangular images "
            "(default: <xml_dir>/360imgs)"
        ),
    )
    ap.add_argument(
        "--cut-out",
        default=None,
        help="Output folder for cut images (default: tool's own default)",
    )
    ap.add_argument(
        "--points-ply",
        default=None,
        help="Input pointcloud PLY (required when --format includes colmap)",
    )
    ap.add_argument(
        "--pc-rotate-x-plus180",
        dest="pc_rotate_x_plus180",
        action="store_true",
        help="Rotate pointcloud PLY (fixed X +180); points3D uses +90",
    )
    ap.add_argument(
        "--pc-rotate-x-plus90",
        dest="pc_rotate_x_plus180",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--pc-rotate-x-minus90",
        dest="pc_rotate_x_plus180",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return ap


def parse_axis(text):
    text = text.replace(",", " ").strip()
    parts = [p for p in text.split() if p]
    if len(parts) != 3:
        raise ValueError("axis must have 3 values (x y z)")
    return [float(parts[0]), float(parts[1]), float(parts[2])]


def resolve_cut_paths(args, xml_dir):
    if args.cut_input:
        cut_in = pathlib.Path(args.cut_input).expanduser().resolve()
    else:
        cut_in = xml_dir / "360imgs"
    if not cut_in.exists():
        raise ValueError("cut input not found: {}".format(cut_in))
    if args.cut_out:
        cut_out = pathlib.Path(args.cut_out).expanduser().resolve()
    else:
        cut_out = None
    return cut_in, cut_out


def run_cut(preset_name, cut_in, cut_out):
    tool_path = (
        pathlib.Path(__file__).resolve().parent / "gs360_360PerspCut.py"
    )
    if not tool_path.exists():
        raise ValueError(
            "gs360_360PerspCut.py not found: {}".format(tool_path)
        )
    cmd = [
        sys.executable,
        str(tool_path),
        "-i",
        str(cut_in),
        "--preset",
        preset_name,
    ]
    if preset_name == PRESET_CUBE:
        cmd = [
            sys.executable,
            str(tool_path),
            "-i",
            str(cut_in),
            "--count",
            "4",
            "--hfov",
            str(CUBE_FOV_DEG),
            "--add-top",
            "--add-bottom",
        ]
    if cut_out is not None:
        cmd += ["-o", str(cut_out)]
    print("[INFO] Running cut: " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    ap = build_arg_parser()
    args = ap.parse_args()

    xml_path = pathlib.Path(args.xml).expanduser().resolve()
    if not xml_path.exists():
        print("[ERR] XML not found:", xml_path, file=sys.stderr)
        sys.exit(1)

    if args.out:
        out_dir = pathlib.Path(args.out).expanduser().resolve()
    else:
        out_dir = xml_path.parent / "perspective_cams"
    out_dir.mkdir(parents=True, exist_ok=True)

    ext = args.ext.lstrip(".")
    axis = parse_axis(args.world_rot_axis)
    rot3 = axis_angle_to_mat3(axis, args.world_rot_deg)
    world_from_metashape = mat3_to_mat4_with_translation(rot3)
    cameras = load_metashape_cameras(xml_path)
    if not cameras:
        print("[WARN] No camera transforms found", file=sys.stderr)
        sys.exit(1)

    frames, intrinsics = build_outputs(
        cameras,
        args.preset,
        ext,
        args.scale,
        world_from_metashape,
        axis,
        args.world_rot_deg,
    )

    if args.format in ("transforms", "all"):
        out_json = out_dir / "transforms.json"
        export_transforms_json(
            out_json,
            frames,
            intrinsics,
            x_fix_deg=TRANSFORMS_X_FIX_DEG,
        )
        print("[OK] transforms.json:", out_json)

    points = []
    if args.format in ("colmap", "all"):
        if not args.points_ply:
            print(
                "[ERR] --points-ply is required when --format includes colmap",
                file=sys.stderr,
            )
            sys.exit(1)
        points_ply_path = pathlib.Path(args.points_ply).expanduser().resolve()
        if not points_ply_path.exists():
            print(
                "[ERR] points PLY not found: {}".format(points_ply_path),
                file=sys.stderr,
            )
            sys.exit(1)
        points = build_points_outputs(
            points_ply_path,
            out_dir,
            world_from_metashape,
            args.pc_rotate_x_plus180,
            args.scale,
        )
        colmap_images = compute_colmap_images(frames, COLMAP_X_BASE_DEG)
        colmap_dir = out_dir / "sparse" / "0"
        export_colmap(colmap_dir, colmap_images, intrinsics, points)
        print("[OK] COLMAP text:", colmap_dir)

    if args.format in ("metashape", "all"):
        out_xml = out_dir / "perspective_cams.xml"
        export_metashape_xml(
            xml_path,
            out_xml,
            frames,
            intrinsics,
            args.preset,
        )
        print("[OK] Metashape cameras XML:", out_xml)

    if args.cut:
        cut_in, cut_out = resolve_cut_paths(args, xml_path.parent)
        run_cut(args.preset, cut_in, cut_out)

    print(
        "[INFO] If you still need to cut images, run "
        "gs360_360PerspCut.py separately."
    )


if __name__ == "__main__":
    main()
