"""Local, offline adapter for the official Music2Emo inference pipeline."""
import csv
import io
import json
import os
from pathlib import Path
import sys
import threading
import time
from tag_scores import correct_prediction, corrected_score_items

ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / 'vendor'
_LOCK = threading.RLock()
ZH = dict(zip(
    'action adventure advertising background ballad calm children christmas commercial cool corporate dark deep documentary drama dramatic dream emotional energetic epic fast film fun funny game groovy happy heavy holiday hopeful inspiring love meditative melancholic melodic motivational movie nature party positive powerful relaxing retro romantic sad sexy slow soft soundscape space sport summer trailer travel upbeat uplifting'.split(),
    '动作 冒险 广告 背景音乐 抒情曲 平静 儿童 圣诞 商业 酷 企业宣传 暗黑 深沉 纪录片 戏剧主题 戏剧性 梦幻 情感浓烈 活力充沛 史诗 快速 电影 欢乐 滑稽 游戏 律动 快乐 厚重 假日 希望 鼓舞 爱 冥想 忧郁 旋律性强 激励 电影 自然 派对 积极 强劲 放松 复古 浪漫 悲伤 性感 缓慢 柔和 声音景观 太空 运动 夏日 预告片 旅行 欢快 振奋'.split()
))
MERT_FILES = ['config.json', 'preprocessor_config.json', 'configuration_MERT.py',
              'modeling_MERT.py', 'pytorch_model.bin']
WEIGHTS = [VENDOR / 'saved_models/J_all.ckpt',
           VENDOR / 'inference/data/btc_model_large_voca.pt']


def missing_files():
    paths = WEIGHTS + [ROOT / 'models/MERT-v1-95M' / f for f in MERT_FILES]
    paths += [VENDOR / 'music2emo.py', VENDOR / 'inference/data/tag_list.npy']
    return [str(p.relative_to(ROOT)) for p in paths if not p.is_file() or p.stat().st_size == 0]


def score_result(labels, logits, valence, arousal, threshold=0.5):
    """Keep independent sigmoid scores; never normalize them to sum to one."""
    import numpy as np
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    if len(labels) != 56 or len(logits) != 56:
        raise ValueError('模型输出或标签数量不等于56，检查下载的版本。')
    if not np.isfinite(logits).all() or not np.isfinite([valence, arousal]).all():
        raise ValueError('模型返回非有限数值。')
    # Stable sigmoid without overflow.
    probs = np.exp(-np.logaddexp(0, -logits))
    rows = [{'tag': str(tag), '中文': ZH.get(str(tag), str(tag)),
             'score': float(p)} for tag, p in zip(labels, probs)]
    rows.sort(key=lambda r: r['score'], reverse=True)
    return correct_prediction({'valence': float(valence), 'arousal': float(arousal),
            'scores': rows, 'predicted_tags': [r['tag'] for r in rows if r['score'] > threshold],
            'threshold': float(threshold), 'score_semantics': 'independent sigmoid scores; funny disabled; not emotion proportions'})


def csv_bytes(rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=['tag', '中文', 'score'], extrasaction='ignore')
    writer.writeheader()
    writer.writerows(corrected_score_items(rows))
    return stream.getvalue().encode('utf-8-sig')


def patched_source(device):
    source = (VENDOR / 'music2emo.py').read_text(encoding='utf-8')
    changes = {
        'import gradio as gr': '# gradio is unused by inference',
        '\n        use_cuda = torch.cuda.is_available()\n': f'\n        use_cuda = {device == "cuda"!r}\n',
        'checkpoint = torch.load(model_file)': 'checkpoint = torch.load(model_file, map_location=self.device, weights_only=False)',
        'mode = key_signature.split()[-1]': 'mode = key_type',
        '            if len(chord_arr) == 1:\n                chordRootID = chordRootDic[chord_arr[0]]':
            '            if len(chord_arr) == 1:\n'
            '                if chord_arr[0] == "X":\n'
            '                    chord_arr[0] = "N"\n'
            '                chordRootID = chordRootDic[chord_arr[0]]',
        'final_embedding_mert = np.mean(segment_embeddings, axis=0)':
            "self.last_mert_chunk_sample_counts = [len(segment) for segment in segments]\n"
            "            self.last_mert_sample_rate = sample_rate\n"
            "            final_embedding_mert = duration_weighted_mert(segment_embeddings, [self.last_mert_chunk_sample_counts[int(name[8:-4])] for name in sorted(os.listdir(mert_dir)) if name.endswith('.npy')])",
    }
    for old, new in changes.items():
        if source.count(old) != 1:
            raise RuntimeError(f'官方代码版本不兼容：{old}')
        source = source.replace(old, new)
    return source


def duration_weighted_mert(embeddings, sample_counts):
    """Pool all MERT chunks by their real audio coverage, including the tail."""
    import numpy as np
    values=np.asarray(embeddings)
    weights=np.asarray(sample_counts,dtype=float)
    if (values.ndim!=2 or weights.ndim!=1 or len(values)!=len(weights) or not len(weights)
            or not np.isfinite(values).all() or not np.isfinite(weights).all() or np.any(weights<=0)):
        raise ValueError('MERT特征和实际分段长度不匹配或无效。')
    # Preserve the single-clip path exactly; short bar predictions are unchanged.
    if len(weights)==1:return values[0].copy()
    return np.average(values,axis=0,weights=weights).astype(values.dtype,copy=False)


def chord_features(audio_file, config):
    # Same CQT parameters and 10-second chunking as upstream; initialize
    # from a list so clips <=10s cannot reference an uninitialized variable.
    import librosa
    import numpy as np
    wav, sr = librosa.load(audio_file, sr=config.mp3['song_hz'], mono=True)
    chunk_size = int(config.mp3['song_hz'] * config.mp3['inst_len'])
    segments = split_segments(wav, chunk_size, 3 * sr)
    chunks = [librosa.cqt(segment, sr=sr,
               n_bins=config.feature['n_bins'],
               bins_per_octave=config.feature['bins_per_octave'],
               hop_length=config.feature['hop_length'])
              for segment in segments]
    feature = np.log(np.abs(np.concatenate(chunks, axis=1)) + 1e-6)
    return feature, config.mp3['inst_len'] / config.model['timestep'], len(wav) / sr


def split_segments(waveform, size, min_tail):
    segments = [waveform[i:i + size] for i in range(0, len(waveform), size)]
    if len(segments) > 1 and len(segments[-1]) < min_tail:
        start = (len(segments) - 2) * size
        segments[-2:] = [waveform[start:]]
    return segments


class Tester:
    def __init__(self, device='cpu'):
        import torch
        missing = missing_files()
        if missing:
            raise FileNotFoundError('缺少文件：\n' + '\n'.join(missing))
        if device == 'cuda':
            from gpu_compat import ensure_cuda_works
            ensure_cuda_works()
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        sys.path.insert(0, str(VENDOR))
        # Load bundled original code with narrow compatibility fixes in memory.
        import types
        source = patched_source(device)
        module = types.ModuleType('_music2emo_local')
        module.__file__ = str(VENDOR / 'music2emo.py')
        module.duration_weighted_mert = duration_weighted_mert
        exec(compile(source, module.__file__, 'exec'), module.__dict__)
        module.audio_file_to_features = chord_features
        # Include the final partial MERT segment instead of silently dropping it.
        module.split_audio = lambda waveform, sample_rate: split_segments(
            waveform, 30 * sample_rate, 3 * sample_rate)
        OriginalExtractor = module.FeatureExtractorMERT
        mert_path = str(ROOT / 'models/MERT-v1-95M')

        class LocalExtractor(OriginalExtractor):
            def __init__(self, model_name=None, **kwargs):
                super().__init__(model_name=mert_path, **kwargs)
                self.model.eval()
        module.FeatureExtractorMERT = LocalExtractor
        self.device = device
        self.torch = torch
        self._logits = None
        with _LOCK:
            previous = Path.cwd()
            try:
                os.chdir(VENDOR)
                self.model = module.Music2emo(model_weights=str(WEIGHTS[0]))
            finally:
                os.chdir(previous)
        import numpy as np
        tags = np.load(VENDOR / 'inference/data/tag_list.npy', allow_pickle=False)[127:]
        self.labels = [str(tag).replace('mood/theme---', '') for tag in tags]
        self._hook = self.model.music2emo_model.register_forward_hook(self._capture)

    def _capture(self, model, inputs, outputs):
        self._logits = outputs[0].detach().float().cpu().numpy().copy()

    def predict(self, wav_path, threshold=0.5):
        audio_path = str(Path(wav_path).resolve())
        with _LOCK:
            previous = Path.cwd()
            self._logits = None
            started = time.perf_counter()
            try:
                os.chdir(VENDOR)
                with self.torch.inference_mode():
                    output = self.model.predict(audio_path, threshold=threshold)
            finally:
                os.chdir(previous)
            if self._logits is None:
                raise RuntimeError('未捕获到模型输出。')
            result = score_result(self.labels, self._logits, output['valence'], output['arousal'], threshold)
            result['elapsed_seconds'] = round(time.perf_counter() - started, 3)
            result['device'] = self.device
            result['mert_pooling'] = 'audio_duration_weighted'
            result['inference_pipeline_version'] = 'duration_weighted_mert_v1'
            result['mert_chunk_durations_seconds'] = [n/self.model.last_mert_sample_rate
                for n in self.model.last_mert_chunk_sample_counts]
            return result


def main():
    import argparse
    import tempfile
    import soundfile as sf
    parser = argparse.ArgumentParser()
    parser.add_argument('audio')
    parser.add_argument('--start', type=float, default=0)
    parser.add_argument('--seconds', type=float, default=30, help='0 = to end')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--output', default='result.json')
    parser.add_argument('--segment', action='store_true', help='Acoustic segmentation then independent predictions')
    parser.add_argument('--min-duration', type=float, default=1.0)
    parser.add_argument('--max-duration', type=float, default=10.0, help='0 = no maximum')
    parser.add_argument('--sensitivity', type=float, default=0.8)
    parser.add_argument('--no-beat-snap', action='store_true')
    args = parser.parse_args()
    audio, sr = sf.read(args.audio, dtype='float32', always_2d=True)
    if args.segment:
        from segmented import select_audio, make_plan, predict_plan, wide_csv_bytes
        selected, offset = select_audio(audio.mean(axis=1), sr, args.start, args.seconds)
        config = dict(min_duration=args.min_duration, max_duration=args.max_duration or None,
                      sensitivity=args.sensitivity, snap_to_beat=not args.no_beat_snap)
        plan = make_plan(selected, sr, config, offset)
        tester = Tester(args.device)
        result = predict_plan(selected, sr, plan, tester)
        result['audio_name'] = Path(args.audio).name
        Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        Path(args.output).with_suffix('.csv').write_bytes(wide_csv_bytes(result))
        print(f"Segments: {result['segment_count']}, success: {result['success_count']}")
        return
    if args.start < 0 or args.seconds < 0:
        parser.error('start / seconds must be nonnegative')
    start = int(args.start * sr)
    end = min(len(audio), start + int(args.seconds * sr)) if args.seconds else len(audio)
    if end - start < 3 * sr:
        parser.error('audio selection must be at least 3 seconds')
    with tempfile.TemporaryDirectory(prefix='music2emo_') as tmp:
        wav = Path(tmp) / 'input.wav'
        sf.write(wav, audio[start:end].mean(axis=1), sr, subtype='PCM_16')
        result = Tester(args.device).predict(wav)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
