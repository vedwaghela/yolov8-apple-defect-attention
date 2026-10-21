"""Restore portable experiment records and the frozen split to local paths."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def relocate(value):
    if isinstance(value, str):
        return value.replace('${PROJECT_ROOT}', ROOT.as_posix())
    if isinstance(value, list):
        return [relocate(x) for x in value]
    if isinstance(value, dict):
        return {k: relocate(v) for k, v in value.items()}
    return value

def write_new(path, data):
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f'Refusing to overwrite different local file: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--with-data', action='store_true', help='Verify prepared images and restore the original split')
    args = parser.parse_args()
    for source in (ROOT / 'results/records').rglob('*'):
        if not source.is_file():
            continue
        data = source.read_bytes()
        if source.suffix == '.json':
            data = json.dumps(relocate(json.loads(data)), indent=2).encode()
        target = ROOT / 'baseline/runs' / source.relative_to(ROOT / 'results/records')
        # Preserve any locally regenerated outputs when setup is repeated.
        if not target.exists():
            write_new(target, data)
    if args.with_data:
        import yaml
        manifest = relocate(json.loads((ROOT / 'data/splits/manifest.json').read_text()))
        for r in manifest['records']:
            image = Path(r['file'])
            label = image.parents[2] / 'labels' / image.parent.name / (image.stem + '.txt')
            for path, expected in [(image, r['sha256']), (label, r['label_sha256'])]:
                if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                    raise RuntimeError(f'Data hash mismatch: {path}')
        folder = ROOT / 'baseline/data/grouped_v1'
        raw = json.dumps(manifest, indent=2).encode()
        write_new(folder / 'manifest.json', raw)
        write_new(folder / 'manifest.sha256', hashlib.sha256(raw).hexdigest().encode())
        config = {'path': (ROOT / 'baseline/data/prepared').as_posix(), 'names': {0: 'scratch', 1: 'pest_damage'}}
        for split in ('train', 'val', 'test'):
            path = folder / f'{split}.txt'
            write_new(path, ('\n'.join(r['file'] for r in manifest['records'] if r['split'] == split) + '\n').encode())
            config[split] = path.as_posix()
        write_new(folder / 'ssda.yaml', yaml.safe_dump(config, sort_keys=False).encode())
        print('Verified all 710 image/label hashes and restored the recorded split; no re-splitting.')
    print('Local records restored. Original published records remain unchanged.')

if __name__ == '__main__':
    main()
