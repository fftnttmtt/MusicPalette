"""Disable funny consistently in inference results, imports, plots and exports."""
import math

DISABLED_TAGS = frozenset({'funny'})


def valid_score(value):
    return isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 1


def corrected_score_items(items):
    rows = []
    for item in items:
        if item.get('tag') in DISABLED_TAGS:
            continue
        row = dict(item)
        rows.append(row)
    return sorted(rows, key=lambda r: r['score'] if valid_score(r.get('score')) else -1., reverse=True)


def correct_prediction(prediction):
    out = dict(prediction)
    if isinstance(prediction.get('scores'), list):
        out['scores'] = corrected_score_items(prediction['scores'])
        threshold = prediction.get('threshold', .5)
        threshold = threshold if valid_score(threshold) else .5
        out['predicted_tags'] = [r['tag'] for r in out['scores'] if valid_score(r.get('score')) and r['score'] > threshold]
        out['disabled_tags'] = sorted(set(prediction.get('disabled_tags', [])) | DISABLED_TAGS)
        out['score_semantics'] = 'independent sigmoid scores; funny disabled; not emotion proportions'
        out['score_adjustments'] = {k:v for k,v in prediction.get('score_adjustments', {}).items() if k not in DISABLED_TAGS}
    if isinstance(prediction.get('raw_prediction'), dict):
        out['raw_prediction'] = correct_prediction(prediction['raw_prediction'])
    return out


def correct_result(result):
    out={**result, 'segments': [correct_prediction(r) for r in result.get('segments', [])],
            'disabled_tags': ['funny'],
            'score_semantics': 'independent sigmoid scores; funny disabled; not emotion proportions'}
    if isinstance(result.get('whole_song_prediction'),dict):
        out['whole_song_prediction']=correct_prediction(result['whole_song_prediction'])
    return out
