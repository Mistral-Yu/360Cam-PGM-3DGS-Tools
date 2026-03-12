#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Overview:
    CLI script that drives ffmpeg to extract frames from a video
    while tracking bit depth and progress.

Dependencies:
    - Python standard library: argparse, pathlib, re, shlex,
      shutil, signal, subprocess, sys, time, typing
    - External tool: ffmpeg
"""

import argparse
import pathlib
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time

from typing import Optional, Tuple

from gs360_360PerspCut import fov_from_focal_mm, v_fov_from_hfov

PROGRESS_INTERVAL = 5
FISHEYE_SENSOR_WIDTH_MM = 36.0
FISHEYE_INPUT_FOV_DEG = 190.0


def infer_bit_depth_from_pix_fmt(pix_fmt: str) -> int:
    """Infer a practical bit depth bucket from an ffmpeg pixel format name."""

    pix_fmt = (pix_fmt or '').strip().lower()
    if any(
        token in pix_fmt
        for token in (
            'p10', 'p12', 'p14', 'p16',
            'yuv420p10', 'yuv422p10', 'yuv444p10',
            'yuv420p12', 'yuv422p12', 'yuv444p12',
            'p010', 'p012', 'p016',
            'gbrp10', 'gbrp12', 'gbrp14', 'gbrp16',
            'rgb48', 'rgba64',
        )
    ):
        return 10
    return 8


def parse_map_stream_selector(
    map_stream: Optional[str],
) -> Tuple[Optional[int], Optional[int], bool]:
    """Parse an ffmpeg ``-map`` selector for input/video stream matching.

    Args:
        map_stream: Optional selector such as ``0:v:1`` or ``0:1``.

    Returns:
        Tuple of ``(input_index, stream_index, uses_video_ordinal)``.
    """

    if not map_stream:
        return 0, 0, True

    text = map_stream.strip().lower()
    match_video = re.match(r'^(?:(\d+):)?v:(\d+)$', text)
    if match_video:
        input_index = int(match_video.group(1) or '0')
        video_index = int(match_video.group(2))
        return input_index, video_index, True

    match_stream = re.match(r'^(?:(\d+):)?(\d+)$', text)
    if match_stream:
        input_index = int(match_stream.group(1) or '0')
        stream_index = int(match_stream.group(2))
        return input_index, stream_index, False

    return 0, 0, True


def update_progress(
    label: str,
    completed: float,
    total: float,
    last_pct: int,
    previous_len: int,
    extra: str = '',
) -> Tuple[int, int]:
    """Write a progress line when enough progress has been made.

    Args:
        label: Static label prefix for each progress line.
        completed: Work units completed.
        total: Total work units expected.
        last_pct: Last emitted percentage value.
        previous_len: Length of the previously printed line.
        extra: Additional detail appended after the label.

    Returns:
        Tuple with the updated percentage value and rendered line length.
    """
    if total <= 0:
        return last_pct, previous_len

    pct = int((completed * 100) / total)
    if last_pct < 0 or pct >= 100 or (pct - last_pct) >= PROGRESS_INTERVAL:
        line = f"{label}{extra} {pct:3d}%"
        sys.stdout.write('\r' + line.ljust(previous_len))
        sys.stdout.flush()
        return pct, len(line)

    return last_pct, previous_len


def probe_media_info_with_ffmpeg(
    ffmpeg_exec: str,
    in_path: pathlib.Path,
    map_stream: Optional[str] = None,
) -> Tuple[Optional[float], int]:
    """Probe duration and effective bit depth by parsing ``ffmpeg -i`` output.

    Args:
        ffmpeg_exec: Path to the ffmpeg executable.
        in_path: Path to the input media file.
        map_stream: Optional stream selector passed via ``-map``.

    Returns:
        Tuple of ``(duration_seconds, bit_depth_bucket)``.
    """
    cmd = [
        ffmpeg_exec,
        '-hide_banner',
        '-i',
        str(in_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
        )
    except Exception:
        return None, 8

    duration = None
    (
        input_index,
        selector_value,
        uses_video_ordinal,
    ) = parse_map_stream_selector(map_stream)
    selected_pix_fmt = None
    fallback_pix_fmt = None
    video_ordinal = {}
    duration_pattern = re.compile(r'Duration:\s*([0-9:.]+)')
    stream_pattern = re.compile(r'Stream #(\d+):(\d+)')

    for raw_line in result.stderr.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if duration is None:
            match_duration = duration_pattern.search(line)
            if match_duration:
                duration = parse_ffmpeg_timecode(match_duration.group(1))

        if ': Video:' not in line:
            continue

        match_stream = stream_pattern.search(line)
        if not match_stream:
            continue

        stream_input_index = int(match_stream.group(1))
        stream_index = int(match_stream.group(2))
        if input_index is not None and stream_input_index != input_index:
            continue

        ordinal = video_ordinal.get(stream_input_index, 0)
        video_ordinal[stream_input_index] = ordinal + 1

        video_info = line.split('Video:', 1)[1]
        parts = [part.strip() for part in video_info.split(',')]
        pix_fmt = ''
        if len(parts) >= 2:
            pix_fmt = parts[1].split('(', 1)[0].strip()
        if not pix_fmt:
            continue

        if fallback_pix_fmt is None:
            fallback_pix_fmt = pix_fmt

        if uses_video_ordinal:
            if selector_value == ordinal:
                selected_pix_fmt = pix_fmt
                break
        elif selector_value == stream_index:
            selected_pix_fmt = pix_fmt
            break

    effective_pix_fmt = selected_pix_fmt or fallback_pix_fmt or ''
    return duration, infer_bit_depth_from_pix_fmt(effective_pix_fmt)


def parse_ffmpeg_timecode(value: str) -> Optional[float]:
    """Convert an ffmpeg timecode string to seconds.

    Args:
        value: Timecode string in the format HH:MM:SS.mmm.

    Returns:
        Parsed seconds, or None when parsing fails.
    """
    try:
        hours_str, minutes_str, seconds_str = value.split(':')
        hours = int(hours_str)
        minutes = int(minutes_str)
        seconds = float(seconds_str)
        return hours * 3600.0 + minutes * 60.0 + seconds
    except Exception:
        return None


def format_seconds(seconds: float) -> str:
    """Format a duration in seconds into H:MM:SS.ss text.

    Args:
        seconds: Duration in seconds.

    Returns:
        Human readable time string.
    """
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:05.2f}"
    return f"{minutes:02d}:{secs:05.2f}"


def main() -> None:
    """CLI entry point that orchestrates frame extraction."""
    ap = argparse.ArgumentParser(
        description='Extract frames from a video at N fps using ffmpeg.'
    )
    ap.add_argument(
        '-i',
        '-in',
        dest='video',
        required=True,
        help='Input video file path.',
    )
    ap.add_argument(
        '-o',
        '-out',
        dest='output',
        default=None,
        help='Output directory (defaults next to the input video).',
    )
    ap.add_argument(
        '-f',
        '--fps',
        type=float,
        required=True,
        help='Frame extraction rate in frames per second (e.g. 5, 2.5).',
    )
    ap.add_argument(
        '-e',
        '--ext',
        default='jpg',
        help='Output image extension (e.g. jpg, png). Defaults to jpg.',
    )
    ap.add_argument(
        '--prefix',
        default='out',
        help='Filename prefix for extracted frames (default: out).',
    )
    ap.add_argument(
        '--start',
        type=float,
        default=0.0,
        help='Optional start time in seconds.',
    )
    ap.add_argument(
        '--end',
        type=float,
        default=None,
        help='Optional end time in seconds.',
    )
    ap.add_argument(
        '--keep-rec709',
        action='store_true',
        help='Keep Rec.709 characteristics instead of converting to sRGB.',
    )
    ap.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite output if it already exists.',
    )
    ap.add_argument(
        '--ffmpeg',
        default='ffmpeg',
        help='Path to the ffmpeg executable (default: ffmpeg).',
    )
    ap.add_argument(
        '--map-stream',
        dest='map_stream',
        default=None,
        help=(
            'Optional ffmpeg -map argument (e.g. 0:v:0) to select a '
            'specific stream.'
        ),
    )
    ap.add_argument(
        '--name-suffix',
        dest='name_suffix',
        default='',
        help=(
            'Optional suffix inserted before the file extension '
            '(e.g. _X).'
        ),
    )
    ap.add_argument(
        '--fisheye-perspective',
        action='store_true',
        help=(
            'Apply an experimental dual fisheye to perspective transform '
            'using ffmpeg v360 (intended for two fisheye streams).'
        ),
    )
    ap.add_argument(
        '--fisheye-focal-mm',
        type=float,
        default=8.0,
        help=(
            'Focal length in millimetres for deriving perspective FOV '
            'when --fisheye-perspective is set (default: 8).'
        ),
    )
    ap.add_argument(
        '--fisheye-size',
        type=int,
        default=3840,
        help=(
            'Output square size in pixels for --fisheye-perspective '
            '(default: 3840).'
        ),
    )
    ap.add_argument(
        '--fisheye-projection',
        type=lambda value: value.lower(),
        choices=('equidistant', 'equisolid'),
        default='equisolid',
        help=(
            'Input fisheye projection model for --fisheye-perspective '
            '(equidistant or equisolid, default: equisolid).'
        ),
    )
    ap.add_argument(
        '--fisheye-input-fov',
        type=float,
        default=FISHEYE_INPUT_FOV_DEG,
        help=(
            'Input fisheye field-of-view in degrees for --fisheye-perspective '
            f'(default: {FISHEYE_INPUT_FOV_DEG:.0f}).'
        ),
    )
    args = ap.parse_args()

    ffmpeg_exec = args.ffmpeg or 'ffmpeg'
    if not shutil.which(ffmpeg_exec):
        print(
            f'ffmpeg executable not found: {ffmpeg_exec}',
            file=sys.stderr,
        )
        sys.exit(1)
    if args.fisheye_perspective:
        if args.fisheye_focal_mm <= 0.0:
            print(
                'Focal length must be greater than zero when using '
                '--fisheye-perspective.',
                file=sys.stderr,
            )
            sys.exit(1)
        if args.fisheye_size <= 0:
            print(
                'Output size must be greater than zero when using '
                '--fisheye-perspective.',
                file=sys.stderr,
            )
            sys.exit(1)
        if args.fisheye_input_fov <= 0.0:
            print(
                'Input fisheye FOV must be greater than zero when using '
                '--fisheye-perspective.',
                file=sys.stderr,
            )
            sys.exit(1)

    in_path = pathlib.Path(args.video).expanduser().resolve()
    if not in_path.exists():
        print(f'Input file not found: {in_path}', file=sys.stderr)
        sys.exit(1)

    stem = in_path.stem
    fps_value = f'{args.fps}'
    fps_str = (
        fps_value.rstrip('0').rstrip('.')
        if '.' in fps_value
        else fps_value
    )

    if args.output:
        out_dir = pathlib.Path(args.output).expanduser().resolve()
    else:
        out_dir = in_path.parent / f'{stem}_frames_{fps_str}fps'

    if out_dir.exists() and not out_dir.is_dir():
        print(
            f'Output path exists and is not a directory: {out_dir}',
            file=sys.stderr,
        )
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    ext = args.ext.lstrip('.').lower()
    frame_prefix = (args.prefix or 'out').strip()
    frame_prefix = re.sub(r'\s+', '_', frame_prefix) if frame_prefix else 'out'
    if not frame_prefix:
        frame_prefix = 'out'
    suffix_text = args.name_suffix.strip() if args.name_suffix else ''
    if suffix_text:
        suffix_text = re.sub(r'\s+', '_', suffix_text)
    pattern = out_dir / f'{frame_prefix}_%07d{suffix_text}.{ext}'
    if not args.overwrite:
        glob_pattern = f"{frame_prefix}_*{suffix_text}.{ext}"
        existing = next(out_dir.glob(glob_pattern), None)
        if existing is not None:
            print(
                'Output exists and overwrite is disabled. '
                f'First match: {existing.name}',
                file=sys.stderr,
            )
            print(
                "Enable --overwrite to replace existing frames.",
                file=sys.stderr,
            )
            sys.exit(1)

    media_duration, inferred_bits = probe_media_info_with_ffmpeg(
        ffmpeg_exec=ffmpeg_exec,
        in_path=in_path,
        map_stream=args.map_stream,
    )
    out_bit_depth = 8 if inferred_bits <= 8 else 16

    colorspace_filter = 'colorspace=iall=bt709:all=smpte170m'
    if not args.keep_rec709:
        colorspace_filter += ':trc=iec61966-2-1'
    if args.fisheye_perspective:
        focal_mm = max(args.fisheye_focal_mm, 1e-6)
        size_px = max(args.fisheye_size, 1)
        projection_map = {
            'equidistant': 'fisheye',
            'equisolid': 'equisolid',
        }
        input_projection = projection_map.get(
            args.fisheye_projection,
            'fisheye',
        )
        input_fov_deg = max(1.0, min(360.0, args.fisheye_input_fov))
        hfov_deg = fov_from_focal_mm(focal_mm, FISHEYE_SENSOR_WIDTH_MM)
        hfov_deg = max(1.0, min(179.0, hfov_deg))
        vfov_deg = v_fov_from_hfov(hfov_deg, size_px, size_px)
        vfov_deg = max(1.0, min(179.0, vfov_deg))
        v360_filter = (
            f"v360={input_projection}:rectilinear:"
            f"ih_fov={input_fov_deg:.6f}:iv_fov={input_fov_deg:.6f}:"
            f"h_fov={hfov_deg:.6f}:v_fov={vfov_deg:.6f}:interp=cubic"
        )
        vf_chain = [v360_filter, f'fps={args.fps}']
        if ext in {'jpg', 'jpeg'}:
            vf_chain.append(f'{colorspace_filter}:range=jpeg:format=yuv444p')
        else:
            vf_chain.append(f'{colorspace_filter}:format=yuv444p')
        vf_chain.append(f'scale={size_px}:{size_px}')
    else:
        vf_chain = [f'fps={args.fps}']
        if ext in {'jpg', 'jpeg'}:
            # Keep JPEG outputs in 4:4:4 and match the expected color space.
            vf_chain.append(f'{colorspace_filter}:range=jpeg:format=yuv444p')
        else:
            # Normalize other outputs to the expected BT.709 color space.
            vf_chain.append(f'{colorspace_filter}:format=yuv444p')

    cmd = [ffmpeg_exec, '-hide_banner', '-y' if args.overwrite else '-n']
    if args.start is not None:
        cmd += ['-ss', str(args.start)]

    cmd += ['-i', str(in_path)]

    if args.end is not None:
        cmd += ['-to', str(args.end)]

    if args.map_stream:
        cmd += ['-map', args.map_stream]

    cmd += ['-vf', ','.join(vf_chain)]

    if ext in {'jpg', 'jpeg'}:
        cmd += [
            '-c:v',
            'mjpeg',
            '-q:v',
            '1',
            '-qmin',
            '1',
            '-qmax',
            '1',
            '-pix_fmt',
            'yuvj444p',
            '-huffman',
            'optimal',
            '-colorspace',
            'smpte170m',
            '-color_primaries',
            'smpte170m',
            '-color_trc',
            'smpte170m',
        ]
    elif ext in {'png', 'tif', 'tiff'}:
        if ext in {'tif', 'tiff'}:
            # Use lossless deflate compression for TIFF.
            cmd += ['-c:v', 'tiff', '-compression_algo', 'deflate']
        if out_bit_depth <= 8:
            cmd += ['-pix_fmt', 'rgb24']
        else:
            cmd += ['-pix_fmt', 'rgb48le']

    cmd += ['-vsync', 'vfr', '-start_number', '0', str(pattern)]

    start_offset = max(float(args.start or 0.0), 0.0)
    progress_span = None
    if media_duration is not None:
        natural_end = (
            media_duration
            if args.end is None
            else min(args.end, media_duration)
        )
        progress_span = max(natural_end - start_offset, 0.0)
    elif args.end is not None:
        progress_span = max(float(args.end) - start_offset, 0.0)
    estimated_frames = None
    if progress_span is not None and progress_span > 0 and args.fps > 0:
        estimated_frames = progress_span * float(args.fps)
    print("\n[exec] " + ' '.join(shlex.quote(x) for x in cmd) + "\n")
    creationflags = 0
    if sys.platform.startswith('win'):
        creationflags = getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0)
    frame_pattern = re.compile(r'frame=\s*(\d+)')
    time_pattern = re.compile(r'time=\s*([0-9:.]+)')
    duration_pattern = re.compile(r'Duration:\s*([0-9:.]+)')
    duration_locked = media_duration is not None
    progress_last_pct = -1
    progress_line_len = 0
    frame_seen = None
    time_seen = None
    start_wall = time.monotonic()
    proc = subprocess.Popen(
        cmd,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8',
        errors='replace',
        creationflags=creationflags,
    )
    try:
        assert proc.stderr is not None
        for raw_line in proc.stderr:
            line = raw_line.strip()
            if not line:
                continue

            if not duration_locked:
                match_duration = duration_pattern.search(line)
                if match_duration:
                    parsed_duration = parse_ffmpeg_timecode(
                        match_duration.group(1)
                    )
                    if parsed_duration is not None:
                        media_duration = parsed_duration
                        natural_end = (
                            media_duration
                            if args.end is None
                            else min(args.end, media_duration)
                        )
                        progress_span = max(natural_end - start_offset, 0.0)
                        if (
                            progress_span
                            and progress_span > 0
                            and args.fps > 0
                        ):
                            estimated_frames = progress_span * float(args.fps)
                    duration_locked = True

            match_frame = frame_pattern.search(line)
            if match_frame:
                frame_seen = int(match_frame.group(1))
            match_time = time_pattern.search(line)
            if match_time:
                parsed_time = parse_ffmpeg_timecode(match_time.group(1))
                if parsed_time is not None:
                    time_seen = parsed_time
            percent = None
            if progress_span and time_seen is not None and progress_span > 0:
                percent = min(
                    100.0,
                    max(0.0, (time_seen / progress_span) * 100.0),
                )
            elif (
                estimated_frames
                and frame_seen is not None
                and estimated_frames > 0
            ):
                percent = min(
                    100.0,
                    max(0.0, (frame_seen / estimated_frames) * 100.0),
                )

            extra_parts = []
            if frame_seen is not None:
                extra_parts.append(f' frame {frame_seen}')
            if time_seen is not None:
                extra_parts.append(f' time {format_seconds(time_seen)}')
            if percent is not None and 0.0 < percent < 100.0:
                elapsed = time.monotonic() - start_wall
                if elapsed > 0.0:
                    remaining_ratio = (100.0 - percent) / percent
                    eta_seconds = elapsed * remaining_ratio
                    extra_parts.append(f' ETA {format_seconds(eta_seconds)}')
            extra_text = ''.join(extra_parts)

            if percent is not None:
                completed_units = 0.0
                total_units = 0.0
                if (
                    progress_span
                    and time_seen is not None
                    and progress_span > 0
                ):
                    completed_units = min(time_seen, progress_span)
                    total_units = progress_span
                elif (
                    estimated_frames
                    and frame_seen is not None
                    and estimated_frames > 0
                ):
                    completed_units = min(frame_seen, estimated_frames)
                    total_units = estimated_frames
                else:
                    completed_units = percent
                    total_units = 100.0
                progress_last_pct, progress_line_len = update_progress(
                    '[progress]',
                    completed_units,
                    total_units,
                    progress_last_pct,
                    progress_line_len,
                    extra_text,
                )
            elif extra_text:
                line_text = f'[progress]{extra_text}'
                sys.stdout.write('\r' + line_text.ljust(progress_line_len))
                sys.stdout.flush()
                progress_line_len = len(line_text)
        proc.stderr.close()
    except KeyboardInterrupt:
        sys.stdout.write('\n[INFO] Interrupt received, stopping ffmpeg...\n')
        sys.stdout.flush()
        try:
            if (
                sys.platform.startswith('win')
                and hasattr(signal, 'CTRL_BREAK_EVENT')
            ):
                proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                proc.send_signal(signal.SIGINT)
        except Exception:
            proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        sys.exit(130)
    retcode = proc.wait()
    if progress_line_len:
        sys.stdout.write('\n')
        sys.stdout.flush()
    if retcode != 0:
        print(
            f'ffmpeg execution failed (returncode={retcode})',
            file=sys.stderr,
        )
        sys.exit(retcode)
    print(f'Completed: {out_dir}')


if __name__ == '__main__':
    main()
