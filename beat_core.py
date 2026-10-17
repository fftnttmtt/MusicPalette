"""Sample-aligned bar segmentation, audio decoding and exports."""
from __future__ import annotations

import csv
import io
import json
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

MODEL_URL = "https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp"
MODELS = {"final0", "small0"}


def load_audio(path):
    """Decode once, preserving channels and sample rate for all subsequent work."""
    path = Path(path)
    try:
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
    except (sf.LibsndfileError, RuntimeError):
        import imageio_ffmpeg
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "decoded.wav"
            subprocess.run(
                [imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-y", "-v", "error",
                 "-i", str(path), "-vn", "-c:a", "pcm_f32le", str(target)],
                check=True, capture_output=True,
            )
            audio, sr = sf.read(target, dtype="float32", always_2d=True)
    if len(audio) < 1 or sr <= 0 or not np.isfinite(audio).all():
        raise ValueError("音频为空或包含无效采样点。")
    return audio, int(sr)


def get_tracker(model="final0", device="auto"):
    import torch
    from beat_this.inference import Audio2Beats
    if model not in MODELS:
        raise ValueError("只支持 final0 或 small0。")
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("无效的计算设备。")
    selected = "cuda" if device == "auto" and torch.cuda.is_available() else device
    if selected == "auto":
        selected = "cpu"
    if selected == "cuda":
        from gpu_compat import ensure_cuda_works
        ensure_cuda_works()
    if selected == "cpu":
        # Excessive worker threads make these small attention blocks slower.
        torch.set_num_threads(min(4, torch.get_num_threads()))
    # Keep downloads in the project; a manually downloaded model works offline.
    root = Path(__file__).resolve().parent
    torch.hub.set_dir(str(root / "model_cache"))
    local = root / "models" / f"{model}.ckpt"
    checkpoint = str(local) if local.is_file() else model
    tracker = Audio2Beats(checkpoint_path=checkpoint, device=selected, dbn=False)
    return tracker, selected


def clean_events(times, sr, frames):
    """Discard invalid events, deduplicate at sample precision, allow the EOF marker."""
    values = np.asarray(times, dtype=float).reshape(-1)
    duration = frames / sr
    values = values[np.isfinite(values) & (values >= 0) & (values <= duration)]
    return np.unique(np.clip(np.rint(values * sr).astype(np.int64), 0, frames))


def build_analysis(audio, sr, beats, downbeats, *, filename="audio", model="final0", device="cpu"):
    n = len(audio)
    if n < 1 or sr < 1:
        raise ValueError("音频为空。")
    beat_samples = clean_events(beats, sr, n)
    down_samples = clean_events(downbeats, sr, n)
    boundaries = np.unique(np.r_[0, down_samples, n]).astype(np.int64)
    segments = []
    bar_number = 0
    detected = set(map(int, down_samples))
    for i, (a, b) in enumerate(zip(boundaries[:-1], boundaries[1:]), start=1):
        a, b = int(a), int(b)
        if a in detected and b in detected:
            kind = "bar"
            bar_number += 1
            number = bar_number
            label = f"小节 {number}"
        elif len(down_samples) == 0:
            kind, number, label = "unmetered", None, "未检测到小节边界"
        elif a == 0 and a not in detected:
            kind, number, label = "leading", None, "开头剩余片段"
        else:
            kind, number, label = "trailing", None, "结尾剩余片段"
        count = int(np.searchsorted(beat_samples, b, side="left") -
                    np.searchsorted(beat_samples, a, side="left"))
        segments.append({
            "index": i, "bar_number": number, "kind": kind, "label": label,
            "start": a / sr, "end": b / sr, "duration": (b - a) / sr,
            "start_sample": a, "end_sample": b, "beat_count": count,
        })
    gaps = np.diff(beat_samples) / sr
    bpm = float(60 / np.median(gaps)) if len(gaps) and np.median(gaps) > 0 else None
    warnings = []
    if len(down_samples) < 2:
        warnings.append("小节第一拍少于两个，无法划分完整小节；已保留全部音频。")
    counts = [x["beat_count"] for x in segments if x["kind"] == "bar"]
    if len(set(counts)) > 1:
        warnings.append("各小节检测到的拍数不一致，可能是变拍，也可能有错拍或漏拍，请试听核对。")
    if any(x == 0 for x in counts):
        warnings.append("有小节内未检测到节拍，请检查模型结果。")
    if any(x == 1 or x > 12 for x in counts):
        warnings.append("有小节区间的检测拍数为 1 或超过 12，请重点检查小节第一拍是否误检或漏检。")
    return {
        "schema_version": 1, "filename": filename,
        "model": model, "device": device, "sample_rate": sr,
        "channels": 1 if audio.ndim == 1 else audio.shape[1],
        "total_samples": n, "duration": n / sr,
        "bpm_median": bpm, "bar_count": bar_number,
        "beats": (beat_samples / sr).tolist(),
        "downbeats": (down_samples / sr).tolist(),
        "beat_samples": beat_samples.tolist(), "downbeat_samples": down_samples.tolist(),
        "segments": segments, "warnings": warnings,
        "notes": ["bar 是相邻两个检测到的小节第一拍之间的区间，准确性取决于模型。",
                  "leading/trailing 保留首尾音频，不强行判断其是否为完整小节。",
                  "beat_count 是检测拍数，不是乐谱拍号；BPM 是相邻节拍间隔的中位数估计。"],
    }


def analyze_audio(audio, sr, tracker, *, merge=True, min_similarity=0.85,
                  sensitivity=0.8, context_bars=2.0, progress=None, **metadata):
    # Official Audio2Beats accepts [samples] or [samples, channels], not [channels, samples].
    beats, downbeats = tracker(np.asarray(audio, dtype=np.float32), sr)
    result = build_analysis(audio, sr, beats, downbeats, **metadata)
    if merge:
        from bar_merge import merge_similar_bars
        result = merge_similar_bars(audio, sr, result, min_similarity=min_similarity,
                                    sensitivity=sensitivity, context_bars=context_bars,
                                    progress=progress)
    else:
        result.update(segment_count=len(result["segments"]), merge_settings={"enabled": False})
    return result


def mix_clicks(audio, sr, analysis, gain=0.35):
    signal = np.asarray(audio, dtype=np.float32)
    mono = signal.ndim == 1
    if mono:
        signal = signal[:, None]
    mixed = signal.copy() * 0.72
    length = max(1, round(0.035 * sr))
    t = np.arange(length) / sr
    downbeats = set(analysis["downbeat_samples"])
    events = sorted(set(analysis["beat_samples"]) | downbeats)
    for sample in events:
        if sample >= len(signal):
            continue
        freq = 1800 if sample in downbeats else 900
        amplitude = gain if sample in downbeats else gain * 0.62
        click = amplitude * np.sin(2 * np.pi * freq * t) * np.exp(-t * 120)
        count = min(length, len(signal) - sample)
        mixed[sample:sample + count] += click[:count, None]
    # Soft gain reduction rather than clipping peaks in the verification track.
    peak = float(np.max(np.abs(mixed)))
    if peak > 0.98:
        mixed *= 0.98 / peak
    return mixed[:, 0] if mono else mixed


def waveform(audio, bins=1400):
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    edges = np.linspace(0, len(mono), min(bins, len(mono)) + 1).astype(int)
    return [[float(np.min(mono[a:b])), float(np.max(mono[a:b]))]
            for a, b in zip(edges[:-1], edges[1:])]


def json_bytes(result):
    return json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")


def csv_bytes(result, *, original=False):
    stream = io.StringIO(newline="")
    keys = ["index", "bar_number", "bar_start", "bar_end", "bar_count", "kind", "label",
            "start", "end", "duration", "start_sample", "end_sample", "beat_count", "boundary_reason"]
    writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
    writer.writeheader()
    for s in result.get("original_segments", result["segments"]) if original else result["segments"]:
        row = {k: s.get(k, "") for k in keys}
        if isinstance(row["boundary_reason"], list):
            row["boundary_reason"] = ";".join(row["boundary_reason"])
        writer.writerow(row)
    return stream.getvalue().encode("utf-8-sig")


def wav_bytes(audio, sr, subtype="FLOAT"):
    stream = io.BytesIO()
    sf.write(stream, audio, sr, format="WAV", subtype=subtype)
    return stream.getvalue()


def export_zip(path, audio, sr, result):
    """All intervals cover the decoded source exactly; no fade or resampling."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("analysis.json", json_bytes(result))
        z.writestr("bars.csv", csv_bytes(result))
        if "original_segments" in result:
            z.writestr("original_bars.csv", csv_bytes(result, original=True))
        for segment in result["segments"]:
            a, b = segment["start_sample"], segment["end_sample"]
            name = f'{segment["index"]:04d}_{segment["kind"]}.wav'
            if segment["kind"] == "bar_group":
                name = f'{segment["index"]:04d}_bars_{segment["bar_start"]:04d}-{segment["bar_end"]:04d}.wav'
            z.writestr(f"bars/{name}", wav_bytes(audio[a:b], sr))


def save_outputs(directory, audio, sr, result, include_zip=True):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "analysis.json").write_bytes(json_bytes(result))
    (directory / "bars.csv").write_bytes(csv_bytes(result))
    if "original_segments" in result:
        (directory / "original_bars.csv").write_bytes(csv_bytes(result, original=True))
    sf.write(directory / "original.wav", audio, sr, subtype="PCM_16")
    sf.write(directory / "clicks.wav", mix_clicks(audio, sr, result), sr, subtype="PCM_16")
    if include_zip:
        export_zip(directory / "bars.zip", audio, sr, result)
