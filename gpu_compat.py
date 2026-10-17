"""Isolated CUDA probing: availability alone does not establish kernel support."""
import functools
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PROFILES = {'cu128': ('2.7.1', '2.7.1'), 'cu121': ('2.5.1', '2.5.1')}


def nvidia_inventory():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,compute_cap,driver_version',
                                 '--format=csv,noheader'], capture_output=True, text=True,
                                encoding='utf-8', errors='replace', timeout=15)
        if result.returncode:
            return {'error': result.stderr.strip() or result.stdout.strip()}
        gpus = []
        for line in result.stdout.splitlines():
            parts = [p.strip() for p in line.rsplit(',', 2)]
            if len(parts) == 3:
                try:
                    major, minor = map(int, parts[1].split('.'))
                except (ValueError, TypeError):
                    continue
                gpus.append({'name': parts[0], 'capability': [major, minor], 'driver': parts[2]})
        return {'devices': gpus}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'error': str(exc)}


def probe_cuda():
    """Only run this function in a disposable child process after a failed CUDA call."""
    report = {'python': sys.version.split()[0], 'executable': sys.executable,
              'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'nvidia_smi': nvidia_inventory(), 'ok': False, 'devices': []}
    stage = 'import torch'
    try:
        import torch
        report.update(torch_version=torch.__version__, torch_cuda=torch.version.cuda,
                      available=torch.cuda.is_available())
        if not report['available']:
            raise RuntimeError('CUDA不可用：可能是CPU版PyTorch、驱动问题或GPU不可见。')
        report['compiled_arches'] = torch.cuda.get_arch_list()
        for i in range(torch.cuda.device_count()):
            report['devices'].append({'index': i, 'name': torch.cuda.get_device_name(i),
                                      'capability': list(torch.cuda.get_device_capability(i))})
        stage = 'CUDA elementwise + reflection pad + STFT + matrix multiplication'
        with torch.inference_mode():
            x = torch.randn(4096, device='cuda')
            x = torch.log1p(x.abs())
            padded = torch.nn.functional.pad(x.reshape(1, 1, -1), (32, 32), mode='reflect')
            spec = torch.stft(padded.flatten(), n_fft=256, hop_length=64,
                              window=torch.hann_window(256, device='cuda'), return_complex=True)
            matrix = torch.randn(32, 32, device='cuda')
            value = (matrix @ matrix.T).sum() + spec.abs().mean()
            stage = 'CUDA convolution + scaled dot product attention'
            conv = torch.nn.Conv1d(2, 4, 3).to('cuda')
            value = value + conv(torch.randn(1, 2, 32, device='cuda')).sum()
            q = torch.randn(1, 4, 16, 32, device='cuda')
            value = value + torch.nn.functional.scaled_dot_product_attention(q, q, q).sum()
            stage = 'torchaudio CUDA mel spectrogram'
            import torchaudio
            report['torchaudio_version'] = torchaudio.__version__
            mel = torchaudio.transforms.MelSpectrogram(sample_rate=22050, n_fft=1024,
                    hop_length=441, n_mels=128).to('cuda')(torch.randn(22050, device='cuda'))
            value = value + mel.mean()
            torch.cuda.synchronize()
            if not torch.isfinite(value).item():
                raise RuntimeError('CUDA自检返回非有限值。')
        report['ok'] = True
        report['stage'] = 'passed'
    except Exception as exc:
        report.update(stage=stage, error=f'{type(exc).__name__}: {exc}')
    return report


def isolated_probe():
    try:
        env = dict(os.environ, PYTHONIOENCODING='utf-8')
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--probe'],
                    capture_output=True, text=True, encoding='utf-8', errors='replace',
                    timeout=120, env=env)
        # Only the final line is protocol output; library warnings remain on stderr.
        if not result.stdout.strip():
            return {'ok': False, 'devices': [], 'error': result.stderr.strip() or f'CUDA probe exited with code {result.returncode}',
                    'nvidia_smi': nvidia_inventory()}
        report = json.loads(result.stdout.strip().splitlines()[-1])
        if result.stderr.strip():
            report['warnings'] = result.stderr.strip()
        return report
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired) as exc:
        return {'ok': False, 'error': str(exc), 'nvidia_smi': nvidia_inventory(), 'devices': []}


def choose_profile(report):
    devices = report.get('devices') or report.get('nvidia_smi', {}).get('devices', [])
    if not devices:
        raise ValueError('Cannot detect GPU compute capability. See cuda_diagnostic.json; update NVIDIA driver and retry.')
    major, minor = devices[0]['capability']
    if major >= 10:
        return 'cu128'
    if major >= 5:
        return 'cu121'
    raise ValueError(f'GPU compute capability {major}.{minor} is too old for these tested CUDA profiles. Use CPU and provide cuda_diagnostic.json for a specific build.')


@functools.lru_cache(maxsize=1)
def ensure_cuda_works():
    report = isolated_probe()
    if not report.get('ok'):
        (ROOT / 'cuda_diagnostic.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        devices = report.get('devices', [])
        name = devices[0]['name'] if devices else '未识别'
        raise RuntimeError(f'CUDA兼容性自检失败：显卡 {name}；PyTorch {report.get("torch_version", "未知")}。'
                           '请关闭程序和启动终端，运行 repair_cuda.bat，成功后重新启动 start.bat。'
                           '详细信息已保存到 cuda_diagnostic.json。\n' + report.get('error', ''))
    return report


if __name__ == '__main__':
    print(json.dumps(probe_cuda(), ensure_ascii=True))
