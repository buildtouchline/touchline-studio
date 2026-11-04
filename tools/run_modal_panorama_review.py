"""One owner-approved $2 panoramic comparison; no automatic retry or promotion."""
from contextlib import redirect_stderr, redirect_stdout
from decimal import Decimal
from pathlib import Path
import argparse
import base64
import hashlib
import json
import logging
import re
import shutil
import sqlite3
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
from run_modal_remote_checks import make_bundle
from run_modal_video_trial import Discard, modal_client, save
from touchline.data.manifest import sha256_file
from touchline.worker_status import parse_worker_status


def extract_selected(archive_path, destination, suffixes, limit):
    with zipfile.ZipFile(archive_path) as archive:
        entries = archive.infolist()
        if len(entries) > 100 or sum(e.file_size for e in entries) > limit:
            raise ValueError('Export expansion exceeds bound')
        targets = []
        for e in entries:
            target = (destination / e.filename).resolve()
            if not target.is_relative_to(destination.resolve()) or target.exists() or target.suffix not in suffixes or (e.external_attr >> 16) & 0o170000 == 0o120000 or target in targets:
                raise ValueError('Unsafe exported file')
            targets.append(target)
        for entry, target in zip(entries, targets):
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(entry) as src, target.open('xb') as dst:
                shutil.copyfileobj(src, dst, 1024**2)


def sam_review(work, settings):
    """Review six exact frames with SAM; no labels are admitted to training."""
    from touchline.modal_trial import parse_sam_boxes
    from touchline.spending import SpendingLedger
    rows = json.loads((work / 'sam-inputs/frames.json').read_text())
    if [r['frame_index'] for r in rows] != [0, 960, 1920, 2880, 3840, 4800]:
        raise ValueError('Unexpected SAM review frames')
    ledger = SpendingLedger(ROOT / 'runs/soccertrack-pxl-20261103-budget.sqlite3')
    claim = 'panorama-sam-review-20261104'
    digest = sha256_file(work / 'sam-inputs.zip')
    if not ledger.reserve('meta', claim, fingerprint=digest, worst_case_usd='1.00')['created']:
        raise ValueError('Consumed panorama SAM reservation')
    record = {'status': 'running', 'model': 'sam-3.1', 'reserved_usd': '1.00', 'billing': 'unreconciled',
              'training_eligible': False, 'purpose': 'Independent person-box proposal review, not identity or ground truth', 'requests': []}
    save(work / 'sam-receipt.json', record)
    proposals = []
    try:
        import httpx
        key = settings.credentials.get('META_API_KEY')
        if not key:
            raise ValueError('Meta runtime credential unavailable')
        with httpx.Client(trust_env=False, follow_redirects=False, timeout=60) as http:
            for row in rows:
                image = work / 'sam-inputs' / row['image']
                if image.resolve().parent != (work / 'sam-inputs').resolve() or image.stat().st_size > 8 * 1024**2:
                    raise ValueError('Unexpected review image')
                entry = {'frame_index': row['frame_index'], 'image_sha256': sha256_file(image), 'state': 'dispatched'}
                record['requests'].append(entry)
                save(work / 'sam-receipt.json', record)
                payload = {'model': 'sam-3.1', 'input': [{'type': 'message', 'role': 'user', 'content': [
                    {'type': 'input_text', 'text': 'person'}, {'type': 'input_image', 'image_url': 'data:image/jpeg;base64,' + base64.b64encode(image.read_bytes()).decode()}]}],
                    'stream': False, 'metadata': {'mask_encoding': 'one_bit', 'include_confidence': 'true'}}
                with http.stream('POST', 'https://api.meta.ai/v1/responses', headers={'Authorization': 'Bearer ' + key}, json=payload) as response:
                    entry['http_status'] = response.status_code
                    if response.status_code != 200:
                        raise RuntimeError('SAM rejected request; no retry')
                    body = bytearray()
                    for chunk in response.iter_bytes():
                        if len(body) + len(chunk) > 8 * 1024**2:
                            raise ValueError('Bounded SAM response required')
                        body.extend(chunk)
                    data = json.loads(body)
                if data.get('status') != 'completed':
                    raise ValueError('Incomplete SAM response')
                text = ''.join(c['text'] for msg in data.get('output', []) for c in msg.get('content', []) if c.get('type') == 'output_text')
                objects = parse_sam_boxes(text, row['width'], row['height'])
                entry.update(state='completed', objects=len(objects))
                if isinstance(data.get('id'), str) and re.fullmatch('[a-zA-Z0-9_-]{1,200}', data['id']):
                    entry['response_id'] = data['id']
                usage = data.get('usage', {})
                entry['usage'] = {k: v for k, v in usage.items() if k in {'input_tokens', 'output_tokens', 'total_tokens'} and type(v) is int and 0 <= v <= 10**8} if isinstance(usage, dict) else {}
                proposals.append({**row, 'sam_objects': objects})
                save(work / 'sam-proposals.json', proposals)
                save(work / 'sam-receipt.json', record)
                print(json.dumps({'stage': 'meta.sam', 'frame_index': row['frame_index'], 'http_status': 200, 'objects': len(objects)}), flush=True)
        record['status'] = 'complete'
    except Exception as exc:
        record.update(status='failed', failure_type=type(exc).__name__)
    finally:
        save(work / 'sam-receipt.json', record)
        ledger.reconcile(claim, finalized=False)
    return record


def run():
    work = ROOT / 'runs/modal-panorama-resnet-20261104'
    work.mkdir(exist_ok=False)
    base = ROOT / 'runs/modal-cpu-full-video-20261102/source'
    inputs = {'video24.mp4': base / 'video24.mp4', 'weights.pth': ROOT / 'artifacts/rcnn-comparison/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth',
              'frame-map.jsonl': base / 'frame-map.jsonl'}
    for name, expected in [('video24.mp4', 'cd15fab8a76e3c7c6ae5d3682ea96956dbedd83ebed3c4929cfb561819cc5baa'),
                           ('weights.pth', 'dd69338a24b8d7381807e247652bdc356325bcbaf1cd3e092e00e0a1a58706bf'),
                           ('frame-map.jsonl', 'ef0331c7d0691a06d5bc0687408a400dc4e9997a2260b560ddc034f5f9519fcb')]:
        if sha256_file(inputs[name]) != expected:
            raise ValueError('Preserved panoramic input lineage mismatch')
    digest = make_bundle(work / 'code.zip')
    inputs['code.zip'] = work / 'code.zip'
    bound = Decimal(2700) * (Decimal('.000306') + 4 * Decimal('.00003942') + 16 * Decimal('.00000667')) + Decimal('.08')
    if bound > Decimal('1.80'):
        raise ValueError('Reviewed resource bound exceeded')
    with sqlite3.connect(ROOT / 'runs/panorama-20261104-allowance.sqlite3') as db:
        db.execute('BEGIN IMMEDIATE')
        used = db.execute('SELECT COALESCE(SUM(reserved_micro_usd),0) FROM claims').fetchone()[0]
        if used + 1800000 > 2000000:
            raise ValueError('Shared panoramic allowance exhausted')
        db.execute('INSERT INTO claims VALUES (?,?,?)', ('native-resnet-panorama', 1800000, 2000000))
    job = {'video_sha256': sha256_file(inputs['video24.mp4']), 'weights_sha256': sha256_file(inputs['weights.pth']),
           'frame_map_sha256': sha256_file(inputs['frame-map.jsonl']), 'code_sha256': digest, 'expected_frames': 5760,
           'geometry': [4096, 1080], 'stationary_camera': True, 'session_id_cap': 24, 'detector_stride': 4}
    record = {'status': 'prepared', 'reserved_usd': '1.80', 'shared_panorama_cap_usd': '2.00', 'project_retained_ceiling_usd': '55.00',
              'gpu': 'A10', 'cpu': 4, 'memory_mib': 16384, 'timeout_s': 2700, 'resource_and_transfer_upper_bound_usd': str(bound),
              'pricing_source': 'https://modal.com/pricing', 'billing': 'unreconciled', 'worker_terminated': 'not_created',
              'code_sha256': digest, 'image_id': 'im-A4IdpUIDiLlRbBha7sqzYy', 'training': 'not run', 'job': job}
    save(work / 'receipt.json', record)
    sandbox = None
    try:
        from touchline.settings import load_operator_settings
        with redirect_stdout(Discard()), redirect_stderr(Discard()):
            settings = load_operator_settings(ROOT / '.env', environ={})
            modal, client = modal_client(settings, work)
            app = modal.App.lookup('touchline-panorama-analysis', client=client, environment_name='main', create_if_missing=True)
            sandbox = modal.Sandbox.create(app=app, client=client, name=work.name,
                image=modal.Image.from_id(record['image_id'], client=client), gpu='A10', cpu=(4, 4), memory=(16384, 16384), timeout=2700,
                runtime='gvisor', block_network=True, secrets=[], volumes={}, workdir='/work',
                env={'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'MPLCONFIGDIR': '/work/cache/matplotlib'})
        allocated = time.time()
        record.update(status='running', worker_id=sandbox.object_id, allocated_epoch=allocated, worker_terminated='pending')
        save(work / 'receipt.json', record)
        print(json.dumps({'stage': 'sandbox', 'status': 'created', 'worker_id': sandbox.object_id, 'gpu': 'A10'}), flush=True)
        with redirect_stdout(Discard()), redirect_stderr(Discard()):
            for remote_name, local in inputs.items():
                sandbox.filesystem.copy_from_local(local, '/work/' + remote_name)
            sandbox.filesystem.copy_from_local(ROOT / 'tools/modal_panorama_worker.py', '/work/panorama-worker.py')
            sandbox.filesystem.write_text(json.dumps(job), '/work/panorama-job.json')
            process = sandbox.exec('python', '-I', '/work/panorama-worker.py', timeout=2450, text=False)
        last = 0
        failures = 0
        complete = False
        while time.time() - allocated < 2500:
            try:
                with redirect_stdout(Discard()), redirect_stderr(Discard()):
                    state = parse_worker_status(sandbox.filesystem.read_text('/work/panorama/worker-state.json'))
                failures = 0
                if state['event_seq'] > last:
                    with (work / 'events.jsonl').open('a') as stream:
                        stream.write(json.dumps(state) + '\n')
                    print(json.dumps(state), flush=True)
                    last = state['event_seq']
                if state['status'] == 'failed':
                    raise RuntimeError('Panorama worker failed')
                if state['stage'] == 'complete' and state['status'] == 'complete':
                    complete = True
                    break
            except RuntimeError:
                raise
            except Exception as exc:
                failures += 1
                print(json.dumps({'stage': 'status_read', 'failure_type': type(exc).__name__, 'attempt': failures}), flush=True)
                with redirect_stdout(Discard()), redirect_stderr(Discard()):
                    exited = sandbox.poll()
                if exited is not None or failures > 6:
                    raise RuntimeError('Cannot verify live panorama status') from None
            time.sleep(10)
        if not complete:
            raise TimeoutError('Bounded panorama deadline reached')
        with redirect_stdout(Discard()), redirect_stderr(Discard()):
            summary = json.loads(sandbox.filesystem.read_text('/work/panorama/summary.json'))
            if summary['frames'] != 5760 or summary['checkpoint_sha256'] != job['weights_sha256'] or summary['source_sha256'] != job['video_sha256']:
                raise ValueError('Panorama result lineage mismatch')
            for name, limit in [('overlay24.mp4', 700 * 1024**2), ('evidence.zip', 128 * 1024**2), ('sam-inputs.zip', 32 * 1024**2)]:
                if not 0 < sandbox.filesystem.stat('/work/panorama/' + name).size <= limit:
                    raise ValueError('Panorama transfer exceeds reviewed bound')
                sandbox.filesystem.copy_to_local('/work/panorama/' + name, work / name)
        extract_selected(work / 'evidence.zip', work / 'evidence', {'.json', '.jsonl', '.jpg', '.html', '.log'}, 512 * 1024**2)
        extract_selected(work / 'sam-inputs.zip', work / 'sam-inputs', {'.jpg', '.json'}, 32 * 1024**2)
        record.update(status='complete', run_id=summary['run_id'], frames=summary['frames'],
                      result_sha256=sha256_file(work / 'overlay24.mp4'), summary=summary)
        # Preserve/terminate paid compute before independent hosted SAM review.
    except BaseException as exc:
        record.update(status='failed', failure_type=type(exc).__name__)
        if sandbox is not None:
            for name in ('worker-state.json', 'failure.json', 'tests.log', 'artifacts.log', 'summary.json', 'raw_proposals.jsonl', 'observations.jsonl', 'coverage.jsonl'):
                try:
                    with redirect_stdout(Discard()), redirect_stderr(Discard()):
                        if sandbox.filesystem.stat('/work/panorama/' + name).size < 256 * 1024**2:
                            sandbox.filesystem.copy_to_local('/work/panorama/' + name, work / ('partial-' + name))
                except Exception:
                    pass
    finally:
        if sandbox is not None:
            try:
                with redirect_stdout(Discard()), redirect_stderr(Discard()):
                    sandbox.terminate(wait=True)
                record['worker_terminated'] = True
            except Exception:
                record['worker_terminated'] = 'unknown'
        save(work / 'receipt.json', record)
    if record['status'] == 'complete':
        review = sam_review(work, settings)
        record['sam_review'] = {'status': review['status'], 'completed_requests': sum(r['state'] == 'completed' for r in review['requests'])}
        save(work / 'receipt.json', record)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute-reviewed-panorama', action='store_true')
    args = parser.parse_args()
    if not args.execute_reviewed_panorama:
        raise SystemExit('Explicit reviewed panoramic scope required')
    logging.disable(logging.CRITICAL)
    try:
        result = run()
        print(json.dumps({k: result.get(k) for k in ('status', 'worker_terminated', 'run_id', 'frames', 'sam_review')}), flush=True)
        if result['status'] != 'complete':
            raise SystemExit(1)
    except Exception as exc:
        print(json.dumps({'status': 'failed', 'failure_type': type(exc).__name__}), flush=True)
        raise SystemExit(1) from None
