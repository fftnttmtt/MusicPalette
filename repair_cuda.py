"""Install a matching torch/torchaudio pair, then test actual CUDA kernels."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from gpu_compat import isolated_probe, choose_profile, PROFILES

ROOT = Path(__file__).resolve().parent


def save_report(report):
    (ROOT / 'cuda_diagnostic.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=True, indent=2), flush=True)


def install_profile(profile):
    version, audio_version = PROFILES[profile]
    # Pip installs matching binaries together; no changes to downloaded model weights.
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', '--force-reinstall',
        '--no-cache-dir', f'torch=={version}', f'torchaudio=={audio_version}',
        '--index-url', f'https://download.pytorch.org/whl/{profile}'], check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--check', action='store_true', help='Report and test only; do not install')
    p.add_argument('--backend', choices=['auto', *PROFILES], default='auto')
    args = p.parse_args()
    before = isolated_probe()
    save_report(before)
    if args.check:
        return 0 if before.get('ok') else 1
    try:
        profile = choose_profile(before) if args.backend == 'auto' else args.backend
    except ValueError as exc:
        print(str(exc), flush=True)
        return 1
    print(f'Installing matching PyTorch / torchaudio profile: {profile}', flush=True)
    try:
        install_profile(profile)
        # Avoid pulling an unconstrained torch version when installing the app deps.
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', str(ROOT/'requirements.txt')], check=True)
        subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True)
    except subprocess.CalledProcessError as exc:
        print(f'Installation failed (exit {exc.returncode}). Read pip output above. Existing model weights were retained.', flush=True)
        return 1
    after = isolated_probe()
    after['installed_profile'] = profile
    save_report(after)
    if not after.get('ok'):
        print('CUDA self-test failed. Update your NVIDIA graphics driver, restart Windows, and rerun this script. '
              'If it still fails, send cuda_diagnostic.json. Do not run start.bat in the old process.', flush=True)
        return 1
    print('CUDA kernels and torchaudio self-test PASSED. Restart with start.bat and select cuda.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
