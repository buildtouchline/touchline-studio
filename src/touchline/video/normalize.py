"""Explicit CPU-only CFR analysis copies; original pixels' geometry and source remain intact."""
from fractions import Fraction
import json
import math
from pathlib import Path

from ..analysis_policy import preflight_analysis_video
from ..data.manifest import sha256_file
from ..diagnostics import diagnosed, diagnostic_progress
from .decode import decode_video, probe_video


@diagnosed("media.normalize")
def normalize_video(source, output, *, max_seconds=None, max_output_frames=None):
    """Resample to 24 Hz using the most recent source frame at each output timestamp.

    Write a video-only H.264 copy and exact zero-based source-frame correspondence.
    No speed change, resize, interpolation, source edits, audio edits or model calls.
    The final frame interval uses declared video duration when credible, otherwise
    the last source cadence. Output duration rounds up by less than one 24-Hz tick.
    Existing output directories are refused; failed partial copies are retained.
    """
    import av
    if max_output_frames is not None and (type(max_output_frames) is not int or max_output_frames <= 0):
        raise ValueError('Positive output-frame bound required')
    if max_seconds is not None and (type(max_seconds) not in (int,float) or not math.isfinite(max_seconds) or not 0<max_seconds<=120):
        raise ValueError('Excerpt duration must be positive and at most 120 seconds')
    limit=None if max_seconds is None else Fraction(str(max_seconds))
    source, output = Path(source), Path(output)
    info = probe_video(source)
    width, height = info['width'], info['height']
    if width <= 0 or height <= 0 or width < height or info['is_rotated']:
        raise ValueError('Normalize requires unrotated landscape video')
    output.mkdir(parents=True, exist_ok=False)
    source_hash = sha256_file(source)
    destination = output/'video24.mp4'
    mapping = output/'frame-map.jsonl'
    count = decoded = 0
    origin = previous_time = None
    last_interval = None
    reader = decode_video(source, 'normalization-source')
    try:
        with av.open(str(destination), mode='w') as encoded, mapping.open('x', encoding='utf-8') as log:
            stream = encoded.add_stream('libx264', rate=24)
            stream.width, stream.height = width, height
            stream.pix_fmt = 'yuv420p' if width % 2 == 0 and height % 2 == 0 else 'yuv444p'
            stream.options = {'crf': '18', 'preset': 'veryfast'}
            stream.codec_context.time_base = Fraction(1, 24)

            def emit_until(frame, end):
                nonlocal count
                source_pts = Fraction(frame.pts * frame.time_base_num, frame.time_base_den)
                # Build once per source interval; encoder receives explicit CFR PTS.
                pixels = None
                while Fraction(count, 24) < end:
                    if max_output_frames is not None and count >= max_output_frames:
                        raise ValueError('Source exceeds the bounded Panorama frame scope')
                    if pixels is None:
                        pixels = frame.image.to_ndarray(format='rgb24')
                    image = av.VideoFrame.from_ndarray(pixels, format='rgb24')
                    image.pts, image.time_base = count, Fraction(1, 24)
                    for packet in stream.encode(image):
                        encoded.mux(packet)
                    log.write(json.dumps({'output_frame_index': count, 'output_pts': count,
                        'output_time_base': [1, 24], 'source_frame_index': frame.index,
                        'source_pts': frame.pts,
                        'source_time_base': [frame.time_base_num, frame.time_base_den],
                        'source_relative_time': [*(source_pts-origin).as_integer_ratio()]})+'\n')
                    count += 1
                    if count % 240 == 0:
                        diagnostic_progress('media.normalize', selected_frames=count)

            current = None
            excerpt_ended=False
            for frame in reader:
                pts = Fraction(frame.pts * frame.time_base_num, frame.time_base_den)
                if origin is None:
                    origin = pts
                if current is not None:
                    last_interval = pts - previous_time
                    emit_until(current, min(pts-origin,limit) if limit is not None else pts-origin)
                    if limit is not None and pts-origin>=limit:
                        excerpt_ended=True
                        break
                current, previous_time = frame, pts
                decoded += 1
            if decoded < 2:
                raise ValueError('At least two timed source frames are required')
            final_end = limit if excerpt_ended else previous_time-origin+last_interval
            duration = info.get('duration_s')
            if duration is not None and not excerpt_ended:
                declared_end = Fraction(str(duration))
                if previous_time-origin < declared_end <= previous_time-origin+2*last_interval:
                    final_end = declared_end
            if limit is not None:final_end=min(final_end,limit)
            emit_until(current, final_end)
            for packet in stream.encode():
                encoded.mux(packet)
    finally:
        reader.close()
    check = preflight_analysis_video(destination)
    if sha256_file(source) != source_hash:
        raise ValueError('Source changed during normalization; copy is not accepted')
    manifest = {'schema': 'touchline.video-normalization.v1', 'status': 'complete',
        'source': str(source.resolve()), 'source_sha256': source_hash,
        'source_metadata': info, 'source_frames': decoded,
        'output': str(destination.resolve()), 'output_sha256': sha256_file(destination),
        'output_frames': count, 'output_duration_s': count/24,
        'source_duration_used_s': float(final_end),
        'source_processing_scope':'bounded_excerpt' if max_seconds is not None else 'full_source',
        'requested_max_seconds':max_seconds,
        'source_origin_pts': [*origin.as_integer_ratio()],
        'frame_map_sha256': sha256_file(mapping), 'source_indices_zero_based': True,
        'sampling': 'latest-source-frame-at-or-before-output-time',
        'geometry': [width, height], 'audio': 'not included; original untouched',
        'encoder': {'codec': 'libx264', 'crf': 18, 'preset': 'veryfast', 'pyav': av.__version__},
        'preflight': check, 'accuracy_target_verified': False}
    (output/'normalization.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    return destination
