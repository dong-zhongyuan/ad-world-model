"""Read the user's new-network stop lists without changing upstream outputs."""
from pathlib import Path
import hashlib
import json

from paths import REPO_ROOT as PROJECT, DRUG_WORK
CURRENT_RESULTS = PROJECT / 'results/CURRENT_RESULTS.json'
HOLD_FILE = DRUG_WORK / 'execution/candidate_holds.json'


def authoritative_sources():
    raw = CURRENT_RESULTS.read_bytes()
    current = json.loads(raw)
    paths = []
    for key, name in [('network_results', 'network_summary.json'),
                      ('mr_results', 'union_summary.json')]:
        folder = Path(current[key])
        path = ((PROJECT / folder) if not folder.is_absolute() else folder) / name
        path = path.resolve()
        if PROJECT.resolve() not in path.parents:
            raise ValueError('Authority points outside the project: ' + str(path))
        paths.append(path)
    return paths, {'path': str(CURRENT_RESULTS),
                   'sha256': hashlib.sha256(raw).hexdigest(),
                   'network_results': current['network_results'],
                   'mr_results': current['mr_results']}


def status(target):
    sources, authority = authoritative_sources()
    records, stops = [], set()
    for path in sources:
        if not path.exists():
            records.append({'path': str(path), 'status': 'not_yet_present'})
            continue
        raw = path.read_bytes()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            records.append({'path': str(path), 'status': 'incomplete_json', 'active': False})
            continue
        state = data.get('status', '')
        required = 'network_complete' if path.name == 'network_summary.json' else 'MR_complete'
        active = state == required
        # A pending/preliminary file supplies no stop decision.
        listed = data.get('stop_docking_md_targets', []) if active else []
        if not isinstance(listed, list) or any(not isinstance(x, str) for x in listed):
            raise ValueError('Unexpected stop-list schema: ' + str(path))
        if active:
            stops.update(x.upper() for x in listed)
        records.append({'path': str(path), 'status': state, 'active': active,
                        'required_status': required,
                        'version': 'CURRENT_RESULTS_authority',
                        'sha256': hashlib.sha256(raw).hexdigest(), 'stop_targets': listed})
    aliases = {'GRN', 'PGRN'} if target.upper() in {'GRN', 'PGRN'} else {target.upper()}
    ready = len(records) == 2 and all(record.get('active', False) for record in records)
    candidate_allowed = ready and not bool(stops & aliases)
    held = []
    if HOLD_FILE.exists():
        held = json.loads(HOLD_FILE.read_text())['targets']
        if not isinstance(held, list) or any(not isinstance(x, str) for x in held):
            raise ValueError('Unexpected local hold schema: ' + str(HOLD_FILE))
        stops.update(x.upper() for x in held)
    return {'target': target, 'allowed': ready and not bool(stops & aliases),
            'candidate_allowed': candidate_allowed, 'authority': authority, 'sources': records,
            'held_targets': held}


def require_allowed(target):
    record = status(target)
    if not record['allowed']:
        raise RuntimeError('CANDIDATE_STOP: ' + target + ' is in the updated stop list')
    return record


if __name__ == '__main__':
    print(json.dumps([status(x) for x in ['BIN1', 'GRN']], indent=2))
