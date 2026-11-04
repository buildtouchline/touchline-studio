"""Bounded read-only pages for the Studio's existing canonical observations."""
import json
from pathlib import Path


def observation_page(path, *, offset=0, limit=2000):
    if type(offset) is not int or not 0 <= offset <= 2000000 or type(limit) is not int or not 1 <= limit <= 5000:
        raise ValueError('Bounded observation offset/limit required')
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)) or path.name != 'observations.jsonl':
        raise ValueError('Canonical unlinked observation file required')
    rows = []
    count = 0
    has_more = False
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if not line.strip():continue
            if len(line) > 65536:raise ValueError('Observation record exceeds bound')
            if count < offset:
                count += 1
                continue
            if len(rows) == limit:
                has_more = True
                break
            row = json.loads(line)
            if not isinstance(row, dict):raise ValueError('Observation object required')
            rows.append(row)
            count += 1
    return {'observations': rows, 'offset': offset, 'next_offset': offset + len(rows) if has_more else None,
            'has_more': has_more}
