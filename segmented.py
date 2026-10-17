"""Acoustic segmentation followed by one independent prediction per exact span."""
import csv
import io
import math
from pathlib import Path
import tempfile
import time
import traceback
import numpy as np
from segmentation import acoustic_segments

DEFAULTS = dict(min_duration=1.0, max_duration=10.0,
                sensitivity=0.8, snap_to_beat=True)


def select_audio(y, sr, start=0, seconds=0):
    y = np.asarray(y, dtype=np.float32)
    if not isinstance(sr, (int, np.integer)) or sr <= 0:
        raise ValueError('采样率必须为正整数。')
    if y.ndim != 1 or not len(y) or not np.isfinite(y).all():
        raise ValueError('音频必须是非空且数值有限的单声道音频。')
    if not all(math.isfinite(v) and v >= 0 for v in [start, seconds]):
        raise ValueError('开始位置和时长必须为有限的非负数。')
    first = int(start * sr)
    last = min(len(y), first + int(seconds * sr)) if seconds else len(y)
    if last <= first:
        raise ValueError('选中的音频为空，请调整开始位置和分析时长。')
    return y[first:last], first


def validate_spans(spans, n_samples):
    previous = 0
    normalized = []
    for pair in spans:
        if len(pair) != 2 or any(not isinstance(x, (int, np.integer)) for x in pair):
            raise ValueError('分段端点必须是整数采样位置。')
        left, right = map(int, pair)
        if left != previous or right <= left or right > n_samples:
            raise ValueError('分段必须连续、无重叠，且覆盖完整音频。')
        normalized.append((left, right))
        previous = right
    if not normalized or previous != n_samples:
        raise ValueError('分段未覆盖完整音频。')
    return normalized


def make_plan(y, sr, config=None, offset_sample=0):
    y = np.asarray(y, dtype=np.float32)
    if y.ndim != 1 or not len(y) or not np.isfinite(y).all() or sr <= 0:
        raise ValueError('分段需要非空、有限数值的单声道音频及正采样率。')
    params = {**DEFAULTS, **(config or {})}
    for key in ['min_duration', 'sensitivity']:
        if not math.isfinite(params[key]):
            raise ValueError('分段参数必须为有限数值。')
    if params['min_duration'] <= 0:
        raise ValueError('最短段长必须大于0。')
    if params['max_duration'] is not None and (not math.isfinite(params['max_duration']) or params['max_duration'] < 2 * params['min_duration']):
        raise ValueError('最长段长须关闭，或至少为最短段长的2倍。')
    spans, diagnostic = acoustic_segments(y, sr=sr, **params)
    spans = validate_spans(spans, len(y))
    reasons = {int(r['sample']): r for r in diagnostic['boundaries']}
    rows = []
    for index, (left, right) in enumerate(spans, 1):
        rows.append({'segment': index,
                     'start_sample': left + offset_sample,
                     'end_sample': right + offset_sample,
                     'start_seconds': (left + offset_sample) / sr,
                     'end_seconds': (right + offset_sample) / sr,
                     'duration_seconds': (right - left) / sr,
                     'boundary_reason': reasons[left]['reason'],
                     'novelty': float(reasons[left]['novelty'])})
    return {'sample_rate': int(sr), 'offset_sample': int(offset_sample),
            'num_samples': len(y), 'duration_seconds': len(y) / sr,
            'parameters': params, 'spans': spans, 'segment_rows': rows,
            'diagnostic': diagnostic}


def predict_plan(y, sr, plan, tester, threshold=0.5, progress=None):
    """Reuse one loaded tester; errors remain attached to their exact span."""
    import soundfile as sf
    y = np.asarray(y, dtype=np.float32)
    spans = validate_spans(plan['spans'], len(y))
    if plan['sample_rate'] != sr or plan['num_samples'] != len(y):
        raise ValueError('音频与分段计划不匹配。')
    if not np.isfinite(y).all() or y.ndim != 1:
        raise ValueError('音频必须是有限数值的单声道音频。')
    started = time.perf_counter()
    results = []
    with tempfile.TemporaryDirectory(prefix='music2emo_segments_') as directory:
        clip_path = Path(directory) / 'input.wav'
        for index, (left, right) in enumerate(spans):
            row = dict(plan['segment_rows'][index])
            clip = y[left:right]
            # Preserve exact source boundaries. Only a sub-second tail is padded
            # to 1s for feature extraction; no audio from neighbors is borrowed.
            padding = max(0, sr - len(clip))
            row['padding_seconds'] = padding / sr
            row['inference_duration_seconds'] = (len(clip) + padding) / sr
            try:
                if padding:
                    clip = np.pad(clip, (0, padding))
                sf.write(clip_path, clip, sr, subtype='FLOAT')
                prediction = tester.predict(clip_path, threshold=threshold)
                row.update(prediction)
                row['status'] = 'ok'
            except Exception as exc:
                row.update(status='error', error=f'{type(exc).__name__}: {exc}',
                           error_traceback=traceback.format_exc(),
                           valence=None, arousal=None, scores=[], predicted_tags=[])
            results.append(row)
            if progress is not None:
                progress(index + 1, len(spans), row)
    return {'mode': 'acoustic_segments', 'sample_rate': int(sr),
            'offset_sample': plan['offset_sample'], 'parameters': plan['parameters'],
            'duration_seconds': plan['duration_seconds'], 'threshold': threshold,
            'segment_count': len(results),
            'success_count': sum(r['status'] == 'ok' for r in results),
            'elapsed_seconds': round(time.perf_counter() - started, 3),
            'segments': results, 'diagnostic': plan['diagnostic'],
            'score_semantics': 'independent sigmoid scores; not emotion proportions'}


def wide_csv_bytes(result):
    from tag_scores import correct_result
    from backend import ZH
    tags = [t for t in ZH if t!='funny']
    base = ['segment', 'start_seconds', 'end_seconds', 'duration_seconds',
            'valence', 'arousal', 'status', 'padding_seconds', 'boundary_reason', 'error']
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=base + tags)
    writer.writeheader()
    for segment in correct_result(result)['segments']:
        row = {key: segment.get(key, '') for key in base}
        row.update({score['tag']: score['score'] for score in segment['scores']})
        writer.writerow(row)
    return stream.getvalue().encode('utf-8-sig')
