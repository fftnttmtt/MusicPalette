"""Download only necessary model files, or verify manually downloaded files."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent

def entries():
    for name, destination in [('music2emo_manifest.json', ROOT / 'vendor'),
                              ('mert_manifest.json', ROOT / 'models/MERT-v1-95M')]:
        manifest = json.loads((ROOT / name).read_text(encoding='utf-8'))
        for relative, info in manifest['files'].items():
            yield manifest, relative, destination / relative, info

def valid(path, info):
    if not path.is_file() or path.stat().st_size != info['size']:
        return False
    if info.get('sha256'):
        digest = hashlib.sha256()
        with path.open('rb') as source:
            for block in iter(lambda: source.read(1024 * 1024), b''):
                digest.update(block)
        return digest.hexdigest() == info['sha256']
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--links', action='store_true')
    args = parser.parse_args()
    failed = []
    for manifest, relative, target, info in entries():
        if args.links:
            print(f"{target.relative_to(ROOT)}\nhttps://huggingface.co/{manifest['repo']}/resolve/{manifest['revision']}/{relative}")
            continue
        if valid(target, info):
            print('OK:', target.relative_to(ROOT))
            continue
        if args.check_only:
            failed.append(str(target.relative_to(ROOT)))
            continue
        try:
            from huggingface_hub import hf_hub_download
            # Does not fetch the large fairseq file or unrelated checkpoints.
            destination = ROOT / 'vendor' if manifest['repo'].startswith('amaai-lab/') else ROOT / 'models/MERT-v1-95M'
            print('Downloading:', relative, f"({info['size'] / 1e6:.1f} MB)", flush=True)
            hf_hub_download(repo_id=manifest['repo'], filename=relative,
                            revision=manifest['revision'], local_dir=str(destination))
            if not valid(target, info):
                raise RuntimeError('File size or SHA-256 mismatch')
        except Exception as exc:
            failed.append(str(target.relative_to(ROOT)))
            print('FAILED:', relative, str(exc), file=sys.stderr)
    if failed:
        print('\nMissing or invalid files:\n' + '\n'.join(failed), file=sys.stderr)
        return 1
    if not args.links:
        print('All required files verified.')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
