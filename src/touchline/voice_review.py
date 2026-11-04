"""Pure coach-clip merging and explicit speech-provider selection; no API calls."""
import copy
import math


def select_speech_provider(requested, *, meta_key=None, sarvam_key=None):
    if requested not in {'meta', 'sarvam'}:
        raise ValueError('Explicit supported speech provider required')
    key = meta_key if requested == 'meta' else sarvam_key
    if not key:
        raise ValueError('Requested speech provider credential is unavailable')
    return requested, key


def append_transcript(existing, incoming):
    if not isinstance(existing, dict) or not isinstance(incoming, dict):
        raise ValueError('Transcript objects required')
    result = copy.deepcopy(incoming)
    clips = copy.deepcopy(existing.get('clips', []))
    if existing and not clips:
        clips.append({k: copy.deepcopy(existing.get(k)) for k in ('provider', 'model', 'alignment', 'audio_duration_s', 'created_at_utc')})
    clip_id = f'clip_{len(clips) + 1:06d}'
    clips.append({'clip_id': clip_id, **{k: copy.deepcopy(incoming.get(k)) for k in ('provider', 'model', 'alignment', 'audio_duration_s', 'created_at_utc')}})
    for field, id_key, prefix in [('segments', 'segment_id', 'seg'), ('manual_cues', 'cue_id', 'cue')]:
        old = copy.deepcopy(existing.get(field, []))
        new = copy.deepcopy(incoming.get(field, []))
        if not isinstance(old, list) or not isinstance(new, list):
            raise ValueError('Transcript segment/cue arrays required')
        used = {r.get(id_key) for r in old if isinstance(r, dict)}
        for row in new:
            if not isinstance(row, dict):
                raise ValueError('Transcript rows must be objects')
            start, end = row.get('video_start_s'), row.get('video_end_s')
            if (type(start) not in (int, float) or type(end) not in (int, float)
                    or not math.isfinite(start) or not math.isfinite(end) or not 0 <= start <= end):
                raise ValueError('Finite ordered video-relative speech interval required')
            index = len(old) + 1
            while f'{prefix}_{index:03d}' in used:
                index += 1
            row['source_' + id_key] = row.get(id_key)
            row[id_key] = f'{prefix}_{index:03d}'
            row['clip_id'] = clip_id
            used.add(row[id_key])
            old.append(row)
        result[field] = old
    result['clips'] = clips
    result['transcript_full'] = ' '.join(str(x.get('transcript_full', '')).strip() for x in (existing, incoming)).strip()
    return result
