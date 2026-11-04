"""Isolated stationary-panorama review: native R-CNN anchors and 24 Hz motion.

This operator worker does not train, promote weights, or claim player identity.
The legacy detector/coach pipeline is unchanged.
"""
from collections import defaultdict
from contextlib import redirect_stderr, redirect_stdout
from fractions import Fraction
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import uuid
import zipfile

WORK = Path('/work/panorama')
LOCK = threading.Lock()
DONE = threading.Event()
STATE = {'stage': 'input', 'frames': 0, 'selected_frames': 0, 'event_seq': 0}


def emit(stage, status, **values):
    with LOCK:
        if STATE.get('terminal'):
            return
        STATE.update(stage=stage, status=status, **values)
        STATE['event_seq'] += 1
        STATE['worker_epoch'] = time.time()
        item = {k: v for k, v in STATE.items() if k != 'terminal'}
        if status == 'failed' or stage == 'complete':
            STATE['terminal'] = True
        temporary = WORK / 'worker-state.tmp'
        temporary.write_text(json.dumps(item))
        temporary.replace(WORK / 'worker-state.json')
        sys.__stdout__.write(json.dumps(item) + '\n')
        sys.__stdout__.flush()


def validate_job(job):
    for key in ('video_sha256', 'weights_sha256', 'code_sha256', 'frame_map_sha256'):
        value = job.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Pinned input digests required')
    if job.get('expected_frames') != 5760 or job.get('geometry') != [4096, 1080]:
        raise ValueError('This reviewed trial is only for the four-minute 118577 panorama')
    if job.get('detector_stride') != 4 or job.get('session_id_cap') != 24 or job.get('stationary_camera') is not True:
        raise ValueError('Reviewed six-Hz/24-ID stationary scope required')
    return job


def confirmed_birth(box, score, previous, overlap):
    """A single-frame false positive must not consume one of the 24 session IDs."""
    return score >= .6 and any(s >= .6 and overlap(box, b) >= .25 for b, s in previous)


def main():
    WORK.mkdir(exist_ok=False)
    job = validate_job(json.loads(Path('/work/panorama-job.json').read_text()))
    emit('input', 'started')

    def pulse():
        while not DONE.wait(30):
            emit(STATE['stage'], 'heartbeat')

    threading.Thread(target=pulse, daemon=True).start()
    for filename, key in [('video24.mp4', 'video_sha256'), ('weights.pth', 'weights_sha256'),
                          ('code.zip', 'code_sha256'), ('frame-map.jsonl', 'frame_map_sha256')]:
        with (Path('/work') / filename).open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != job[key]:
                raise ValueError('Pinned panorama input changed')
    if 'worker_script_sha256' in job:
        with Path('/work/panorama-worker.py').open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != job['worker_script_sha256']:
                raise ValueError('Panorama worker script changed')
    project = WORK / 'project'
    project.mkdir()
    with zipfile.ZipFile('/work/code.zip') as archive:
        for entry in archive.infolist():
            target = (project / entry.filename).resolve()
            if not target.is_relative_to(project) or (entry.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('Unsafe source member')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(entry))
    os.chdir(project)
    sys.path.insert(0, str(project / 'src'))
    checks = {}
    for name, args in [('tests', []), ('artifacts', ['artifacts'])]:
        with (WORK / (name + '.log')).open('wb') as log:
            result = subprocess.run([sys.executable, '-B', 'tools/run_offline_checks.py', *args],
                                    stdout=log, stderr=subprocess.STDOUT, timeout=180)
        checks[name] = result.returncode
        if result.returncode:
            raise RuntimeError('Panorama source checks failed')
    emit('checks', 'passed')

    import av
    import cv2
    import numpy as np
    import torch
    from PIL import Image, ImageDraw, ImageFont
    from torchvision.transforms.functional import to_tensor
    from touchline.adapters.rcnn_comparison import RcnnComparisonDetector
    from touchline.motion_tracking import MotionTracker, appearance, estimate_motion, iou
    from touchline.pitch_scope import estimate_pitch_mask, footpoint_scope
    from touchline.video.decode import decode_video, probe_video

    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('Explicit GPU required')
    source = Path('/work/video24.mp4')
    info = probe_video(source)
    if info['average_fps'] != 24 or info['frame_count'] != 5760 or [info['width'], info['height']] != job['geometry']:
        raise ValueError('Exact panoramic analysis copy required')
    detector = RcnnComparisonDetector('/work/weights.pth', device='cuda', min_size=1080, max_size=4096)
    model = detector.model
    use_amp = True

    def predict(frame, amp=True):
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16, enabled=amp):
            tensor = to_tensor(frame.image.to_image().convert('RGB')).to('cuda')
            images, _ = model.transform([tensor])
            features = model.backbone(images.tensors)
            proposals, _ = model.rpn(images, features, None)
            predictions, _ = model.roi_heads(features, proposals, images.image_sizes, None)
            prediction = model.transform.postprocess(predictions, images.image_sizes, [(frame.height, frame.width)])[0]
            keep = torch.where((prediction['labels'] == 1) & (prediction['scores'] >= .1))[0]
            boxes, scores = prediction['boxes'][keep], prediction['scores'][keep]
            scaled = boxes.clone()
            scaled[:, [0, 2]] *= images.image_sizes[0][1] / frame.width
            scaled[:, [1, 3]] *= images.image_sizes[0][0] / frame.height
            pooled = model.roi_heads.box_roi_pool(features, [scaled], images.image_sizes)
            embeddings = model.roi_heads.box_head(pooled).float().clamp_min(0)
            embeddings /= embeddings.sum(1, keepdim=True).clamp_min(1e-9)
            return boxes.float().cpu().tolist(), scores.float().cpu().tolist(), embeddings.cpu().numpy()

    parity = []
    for frame in decode_video(source, job['video_sha256'], decoder_threads=4):
        if frame.index > 240:
            break
        if frame.index not in {0, 96, 240}:
            continue
        full_boxes, full_scores, _ = predict(frame, False)
        amp_boxes, amp_scores, _ = predict(frame, True)
        a = [b for b, s in zip(full_boxes, full_scores) if s >= .5]
        b = [b for b, s in zip(amp_boxes, amp_scores) if s >= .5]
        matched = sum(max((iou(x, y) for y in b), default=0) >= .85 for x in a)
        passed = bool(a) and abs(len(a) - len(b)) <= 2 and matched / len(a) >= .8
        parity.append({'frame_index': frame.index, 'fp32_people': len(a), 'fp16_people': len(b), 'matched_iou85': matched, 'passed': passed})
        use_amp &= passed
    (WORK / 'precision-parity.json').write_text(json.dumps(parity, indent=2))
    emit('precision_parity', 'passed')

    tracker = MotionTracker(4096, 1080, max_gap_s=.65, retain_s=30, max_session_ids=24)
    # Pixel flow is measured at 2048-wide on every decoded frame. Static-camera
    # identity transforms are an owner-declared assumption, not estimated pose.
    identity = np.array([[1., 0., 0.], [0., 1., 0.]])
    previous = None
    previous_candidates = []
    render_width, render_height = 4096, 1080
    small_width, small_height = 2048, 540
    scale = 2.
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 22)
    small_font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 17)
    run_id = str(uuid.uuid4())
    gallery = Image.new('RGB', (1440, 1440), '#151515')
    gallery_seen = set()
    contact = Image.new('RGB', (1920, 870), 'black')
    contact_frames = {round(i * 5759 / 5): i for i in range(6)}
    review_frames = {0, 960, 1920, 2880, 3840, 4800}
    (WORK / 'sam-inputs').mkdir()
    frames_metadata = []
    times = defaultdict(list)
    counters = defaultdict(int)
    start = time.monotonic()
    emit('analysis', 'started', gpu_memory_mib=round(torch.cuda.get_device_properties(0).total_memory / 1024**2))
    with (WORK / 'raw_proposals.jsonl').open('w') as raw, (WORK / 'observations.jsonl').open('w') as observations, (WORK / 'coverage.jsonl').open('w') as coverage, av.open(str(WORK / 'overlay24.mp4'), 'w') as encoded:
        stream = encoded.add_stream('libx264', rate=24)
        stream.width, stream.height = render_width, render_height
        stream.pix_fmt = 'yuv420p'
        stream.options = {'crf': '21', 'preset': 'veryfast'}
        stream.codec_context.thread_count = 2
        for frame in decode_video(source, job['video_sha256'], decoder_threads=4):
            if abs(frame.timestamp_s - frame.index / 24) > 1e-4:
                raise ValueError('Unexpected 24 Hz frame timing')
            rgb = frame.image.reformat(width=small_width, height=small_height, format='rgb24').to_ndarray()
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            flow = {}
            if previous is not None:
                active = {k: t.box for k, t in tracker.tracks.items() if frame.timestamp_s - t.last_anchor <= tracker.max_gap}
                _, _, _, flow = estimate_motion(previous, gray, active, 4096)
            previous = gray
            candidates = []
            detector_frame = frame.index % 4 == 0
            outside = unknown = 0
            if detector_frame:
                boxes, scores, features = predict(frame, use_amp)
                pitch = estimate_pitch_mask(rgb)
                current_candidates = []
                for box, score, feature in zip(boxes, scores, features):
                    box = [max(0., box[0]), max(0., box[1]), min(4096., box[2]), min(1080., box[3])]
                    if not np.isfinite(box).all() or not box[0] < box[2] or not box[1] < box[3]:
                        continue
                    scope = footpoint_scope(box, 4096, 1080, pitch)
                    proposal = {'frame_index': frame.index, 'timestamp_s': frame.timestamp_s, 'bbox_xyxy': box, 'score': score, 'pitch_scope': scope}
                    raw.write(json.dumps(proposal) + '\n')
                    counters['raw_proposals'] += 1
                    outside += scope == 'outside_estimated_pitch'
                    unknown += scope == 'unknown'
                    if scope == 'outside_estimated_pitch':
                        continue
                    color = appearance(rgb, np.asarray(box) / scale)
                    if color is None:
                        color = np.zeros(208)
                    descriptor = np.r_[.6 * feature, .4 * color]
                    birth = confirmed_birth(box, score, previous_candidates, iou)
                    candidates.append({'bbox_xyxy': box, 'score': score, 'feature': descriptor,
                                       'new_track_allowed': birth and scope != 'outside_estimated_pitch'})
                    current_candidates.append((box, score))
                previous_candidates = current_candidates
                counters['detector_frames'] += 1
            rows = tracker.update(frame.timestamp_s, candidates, camera=identity, camera_valid=False, foreground=flow)
            image = frame.image.to_image().convert('RGB')
            if frame.index in review_frames:
                name = f'{frame.index:06d}.jpg'
                image.save(WORK / 'sam-inputs' / name, quality=92)
                frames_metadata.append({'frame_index': frame.index, 'timestamp_s': frame.timestamp_s, 'width': 4096, 'height': 1080, 'image': name})
            draw = ImageDraw.Draw(image)
            for row in rows:
                tid = row['track_id']
                observed = row['evidence'] == 'detector_anchor'
                observations.write(json.dumps({'run_id': run_id, 'frame_index': frame.index, 'timestamp_s': frame.timestamp_s, **row}) + '\n')
                counters['detector_anchors' if observed else 'motion_predictions'] += 1
                times[tid].append(frame.timestamp_s)
                color = ((tid * 73) % 190 + 60, (tid * 131) % 190 + 60, (tid * 47) % 190 + 60)
                box = row['bbox_xyxy']
                if tid not in gallery_seen and observed:
                    crop = image.crop(tuple(map(int, box)))
                    crop.thumbnail((220, 270))
                    slot = tid - 1
                    gx, gy = slot % 6 * 240, slot // 6 * 360
                    gallery.paste(crop, (gx + (240 - crop.width) // 2, gy + 8))
                    gd = ImageDraw.Draw(gallery)
                    gd.text((gx + 10, gy + 290), f'T{tid} | {frame.timestamp_s:.2f}s', font=small_font, fill=color)
                    gallery_seen.add(tid)
                draw.rectangle(box, outline=color, width=3 if observed else 2)
                label = f'T{tid} {"OBS" if observed else "FLOW" if row["foreground_flow_valid"] else "HOLD"}'
                x, y = max(0, int(box[0])), max(65, int(box[1]) - 24)
                label_box = draw.textbbox((x, y), label, font=small_font)
                draw.rectangle((label_box[0] - 2, label_box[1] - 2, label_box[2] + 2, label_box[3] + 2), fill='black')
                draw.text((x, y), label, font=small_font, fill=color)
                trail = row['trail_xy']
                if len(trail) > 1:
                    draw.line([tuple(p) for p in trail], fill=color, width=2)
            if detector_frame:
                for row in tracker.unassigned:
                    draw.rectangle(row['bbox_xyxy'], outline='#ffb000', width=2)
                counters['unassigned_detections'] += len(tracker.unassigned)
            coverage.write(json.dumps({'frame_index': frame.index, 'timestamp_s': frame.timestamp_s, 'detector_sample': detector_frame,
                                       'visible_tracks': len(rows), 'outside_estimated_pitch': outside, 'unknown_scope': unknown,
                                       'unassigned': len(tracker.unassigned), 'flow_tracks': len(flow)}) + '\n')
            draw.rectangle((0, 0, render_width, 61), fill='black')
            draw.text((10, 0), f'{frame.timestamp_s:.2f}s | Native ResNet R-CNN | CUDA {torch.cuda.get_device_name(0)} | IDs {tracker.next_id - 1}/24', font=font, fill='white')
            draw.text((10, 31), '6 Hz detections + 24 Hz motion. OBS=detector; FLOW/HOLD=prediction; orange=unassigned. T IDs are temporary, not jersey numbers.', font=font, fill='white')
            if frame.index in contact_frames:
                slot = contact_frames[frame.index]
                thumb = image.copy()
                thumb.thumbnail((960, 270))
                contact.paste(thumb, (slot % 2 * 960, slot // 2 * 290))
            output = av.VideoFrame.from_image(image)
            output.pts, output.time_base = frame.index, Fraction(1, 24)
            for packet in stream.encode(output):
                encoded.mux(packet)
            STATE['frames'] = frame.index + 1
            STATE['selected_frames'] = counters['detector_frames']
            if frame.index % 120 == 0:
                raw.flush(); observations.flush(); coverage.flush()
                emit('analysis', 'progress', ids=tracker.next_id - 1)
        for packet in stream.encode():
            encoded.mux(packet)
    if STATE['frames'] != 5760 or counters['detector_frames'] != 1440:
        raise ValueError('Incomplete panoramic recording')
    gallery.save(WORK / 'players.jpg', quality=92)
    contact.save(WORK / 'contact-sheet.jpg', quality=92)
    (WORK / 'sam-inputs' / 'frames.json').write_text(json.dumps(frames_metadata, indent=2))
    with zipfile.ZipFile(WORK / 'sam-inputs.zip', 'w', zipfile.ZIP_STORED) as archive:
        for p in (WORK / 'sam-inputs').iterdir():
            archive.write(p, p.name)
    metadata = {'schema': 'touchline.panorama-review.v1', 'status': 'complete', 'run_id': run_id, 'frames': 5760,
                'duration_s': 240, 'source_fps': 24, 'original_fps': 25, 'source_geometry': [4096, 1080],
                'inference_resize': [1080, 4096], 'display_geometry': [render_width, render_height],
                'detector_fps': 6, 'motion_fps': 24, 'session_id_cap': 24, 'allocated_session_ids': tracker.next_id - 1,
                'visible_ids': len(times), 'counters': dict(counters), 'checkpoint_sha256': job['weights_sha256'],
                'source_sha256': job['video_sha256'], 'code_sha256': job['code_sha256'], 'frame_map_sha256': job['frame_map_sha256'],
                'architecture': detector.architecture, 'gpu_name': torch.cuda.get_device_name(0), 'device': detector.actual_device,
                'precision': 'fp16-autocast' if use_amp else 'fp32-parity-fallback', 'precision_parity': parity,
                'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'elapsed_s': round(time.monotonic() - start, 2),
                'training': 'not run', 'ap50': None, 'tracking_idf1': None, 'identity_confirmed': False,
                'camera_assumption': 'Owner declares stationary camera; motion uses identity transform, no physical calibration',
                'birth_policy': 'Two detector samples with IoU >= .25 and score >= .6, within estimated pitch or unknown scope',
                'limitations': 'Experimental generic ROI appearance, not trained person ReID. A 24-ID cap cannot establish identity accuracy. Missing/unassigned people remain explicit; optical flow is prediction. Coach schema adapter remains pending.',
                'tracks': [{'track_id': k, 'first_visible_s': min(v), 'last_visible_s': max(v), 'visible_frames': len(v)} for k, v in sorted(times.items())],
                'checks': checks}
    (WORK / 'summary.json').write_text(json.dumps(metadata, indent=2))
    (WORK / 'RESULTS.html').write_text('<!doctype html><meta charset="utf-8"><h1>Four-minute panoramic review</h1><video controls width="1400" src="overlay24.mp4"></video><p>Temporary IDs, not jersey numbers. Predictions are explicitly labelled. No measured AP50/IDF1 claim.</p><a href="players.jpg">Gallery</a> | <a href="summary.json">Run evidence</a>')
    with zipfile.ZipFile(WORK / 'evidence.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for p in WORK.iterdir():
            if p.is_file() and p.suffix in {'.json', '.jsonl', '.jpg', '.html', '.log'} and p.name != 'worker-state.json':
                archive.write(p, p.name)
    emit('complete', 'complete', run_id=run_id, ids=tracker.next_id - 1)


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        if WORK.is_dir():
            emit(STATE['stage'], 'failed', error_category=type(exc).__name__)
            (WORK / 'failure.json').write_text(json.dumps({'error_category': type(exc).__name__, 'message': str(exc)[:1000]}))
        raise SystemExit(1) from None
    finally:
        DONE.set()
