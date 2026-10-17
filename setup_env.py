import argparse
from pathlib import Path
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--torch', choices=['cpu', 'auto', 'cu121', 'cu128'], default='cpu')
    args = parser.parse_args()
    if sys.version_info[:2] not in [(3, 10), (3, 11)]:
        raise SystemExit('Please install Python 3.10 or 3.11. Recommended: Python 3.10 (64-bit).')
    env = ROOT / '.venv'
    venv.EnvBuilder(with_pip=True).create(env)
    python = env / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')
    def run(*cmd):
        subprocess.run([str(python), '-m', 'pip', *cmd], check=True)
    run('install', '--upgrade', 'pip')
    if args.torch == 'cpu':
        run('install', 'torch==2.5.1', 'torchaudio==2.5.1', '--index-url', 'https://download.pytorch.org/whl/cpu')
        run('install', '-r', str(ROOT / 'requirements.txt'))
        run('check')
    else:
        subprocess.run([str(python), str(ROOT / 'repair_cuda.py'), '--backend', args.torch], check=True)
    print('Environment ready. Download/place weights, then run start.bat.')

if __name__ == '__main__':
    main()
