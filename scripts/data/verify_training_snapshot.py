"""Compare freshly generated training files against the published original hashes."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    manifest = json.loads((ROOT / 'configs/data_manifest.json').read_text())
    expected = manifest['data_files']
    differences = []
    for name, reference in expected.items():
        path = ROOT / 'data/processed' / name
        if not path.is_file():
            differences.append(f'{name}: missing')
            continue
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != reference:
            differences.append(f'{name}: {digest.hexdigest()} (expected {reference})')
    if differences:
        raise SystemExit('\n'.join(differences))
    print(f'All {len(expected)} training files match the frozen release by SHA-256.')


if __name__ == '__main__':
    main()
