"""Bridge the existing inference pipeline to SPECTRA, without changing it."""
from copy import deepcopy
import math

TAGS = ('calm', 'romantic', 'dream', 'sad', 'epic', 'happy', 'love', 'nature', 'energetic')
DEFAULTS = {
    'device': 'cuda', 'beat_model': 'final0', 'prediction_unit': 'bar',
    'target_seconds': 6., 'minimum_seconds': 3., 'boundary_threshold': .7,
    'context_seconds': 2., 'compare_raw': True, 'display_threshold': .5,
    'saturation_k': 0., 'value_k': 0., 'hue_gamma': 2.,
    'saturation_gamma': 1., 'value_gamma': 1., 'use_song_mean_centers': True,
    'enhance_sv': True, 'enhancement_strength': .5, 'percentile_low': 10.,
    'percentile_high': 90., 'max_gain': 5., 'saturation_range': [.2, 1.],
    'value_range': [.25, 1.], 'transition_strength': 8.,
    'transition_mode': 'boundary_start', 'boundary_radius': .25,
    'max_transition_fraction': .15, 'boundary_statistic': 'mean',
    'disabled_tags': ['funny'], 'tag_score_cutoff': None, 'motif_score_gamma': 2.,
}


def make_spectra_bundle(result):
    """Use the exact default Python color mapper and transition implementation."""
    from hsv_timeline import map_colors
    from color_transition import build_transition, start_aligned_transition
    from tag_scores import correct_result
    result = correct_result(result)
    if not result.get('segments') or not isinstance(result.get('beat_analysis'), dict):
        raise ValueError('需要包含 beat_analysis 与 segments 的完整预测 JSON。')
    d = DEFAULTS
    mapped = map_colors(result, **{k: deepcopy(d[k]) for k in (
        'saturation_k', 'value_k', 'hue_gamma', 'saturation_gamma', 'value_gamma',
        'use_song_mean_centers', 'enhance_sv', 'enhancement_strength',
        'percentile_low', 'percentile_high', 'max_gain', 'saturation_range', 'value_range')})
    transition = build_transition(mapped, result, d['transition_strength'], True,
        mode='boundary_hold', boundary_radius=d['boundary_radius'],
        max_transition_fraction=d['max_transition_fraction'], boundary_statistic=d['boundary_statistic'])
    transition = start_aligned_transition(transition)
    raw = {r['segment']: r for r in result['segments']}
    rows = []
    for color in mapped['segments']:
        source = raw[color['segment']]
        scores = {i['tag']: float(i['score']) for i in source.get('scores', [])
                  if i.get('tag') in TAGS and isinstance(i.get('score'), (int, float))
                  and math.isfinite(i['score']) and 0 <= i['score'] <= 1}
        rows.append({**color, 'tag_scores': scores, 'prediction_status': source.get('status'),
                     'bar_number': source.get('bar_number'), 'kind': source.get('kind'),
                     'error': source.get('error') if source.get('status') == 'error' else None})
    analysis = result['beat_analysis']
    offset = result.get('offset_sample', 0) / result['sample_rate']
    bars = [{'id': r['index'], 'bar_number': r.get('bar_number'), 'kind': r['kind'],
             'label': r['label'], 'start_seconds': r['start'] + offset,
             'end_seconds': r['end'] + offset} for r in analysis['segments']]
    whole = result.get('whole_song_prediction') or {}
    duration = whole.get('end_seconds') or offset + result['num_samples'] / result['sample_rate']
    return {
        'schema': 'spectra-prediction-v1', 'prediction_unit': result.get('prediction_unit', 'bar'),
        'audio': {'name': result.get('audio_name', '音乐'), 'sha256': result.get('audio_sha256'),
                  'duration_seconds': duration},
        'defaults': deepcopy(DEFAULTS), 'prediction_parameters': result.get('parameters', {}),
        'beats': [float(t) + offset for t in analysis['beats']],
        'downbeats': [float(t) + offset for t in analysis['downbeats']],
        'bars': bars, 'segments': rows, 'transition': transition,
        'whole_song_color': mapped.get('whole_song_color'), 'mapping': mapped['mapping'],
        'warnings': analysis.get('warnings', []),
        'success_count': sum(r['status'] == 'ok' for r in rows),
    }


def predict_file(path, *, prediction_unit='bar', progress=None, resources=None, audio_name=None, sha256=None):
    """Called on the server worker thread. Models are reused between uploads."""
    from beat_core import load_audio, get_tracker
    from bar_context import make_bar_plan, predict_bar_plan
    from backend import Tester, missing_files
    if prediction_unit not in ('bar', 'beat'):
        raise ValueError('预测单位须为 bar 或 beat。')
    def report(message, fraction=0.):
        if progress:
            progress(message, fraction)
    missing = missing_files()
    if missing:
        raise RuntimeError('缺少现有 Music2Emo 模型文件，请放在原项目目录：' + ', '.join(missing))
    resources = resources if resources is not None else {}
    report('读取音频…', .01)
    audio, sr = load_audio(path)
    y = audio.mean(axis=1)
    report('加载 Beat This! final0 / CUDA…', .03)
    if 'tracker' not in resources:
        resources['tracker'] = get_tracker('final0', 'cuda')[0]
    report('检测节拍、小节与 Foote 变化…', .05)
    plan = make_bar_plan(y, sr, resources['tracker'], target_seconds=6.,
        boundary_threshold=.7, context_seconds=2., model='final0', device='cuda',
        prediction_unit=prediction_unit)
    report('加载 Music2Emo / CUDA…', .12)
    if 'tester' not in resources:
        resources['tester'] = Tester('cuda')
    def on_prediction(done, total, row):
        label = '整曲色彩' if row.get('kind') == 'whole_song' else '情绪片段'
        report(f'{label} · {done}/{total}', .15 + .8 * done / total)
    result = predict_bar_plan(y, sr, plan, resources['tester'], threshold=.5,
        progress=on_prediction, compare_raw=True, whole_audio=y)
    result.update(audio_name=audio_name or path.name, audio_sha256=sha256)
    report('映射颜色与过渡…', .97)
    bundle = make_spectra_bundle(result)
    if not bundle['success_count']:
        errors = [r.get('error') for r in result['segments'] if r.get('error')]
        raise RuntimeError('全部情绪片段预测失败：' + (errors[0] if errors else '没有有效结果'))
    return bundle
