import csv
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import soundfile as sf
from backend import score_result, ZH, chord_features
from segmented import select_audio, make_plan, predict_plan, validate_spans, wide_csv_bytes
from types import SimpleNamespace

class FakeTester:
    def __init__(self, fail_on=None):
        self.inputs = []
        self.fail_on = fail_on

    def predict(self, path, threshold=0.5):
        audio, sr = sf.read(path, dtype='float32')
        self.inputs.append((audio, sr))
        if len(self.inputs) == self.fail_on:
            raise RuntimeError('Injected segment failure')
        result = score_result(list(ZH), np.linspace(-2, 2, 56), 4.1, 5.2, threshold)
        result.update(elapsed_seconds=0.1, device='fake')
        return result

class SegmentedChecks(unittest.TestCase):
    def test_uploaded_segmenter_and_full_coverage(self):
        sr = 22050
        t = np.arange(sr * 12) / sr
        # Alternating pitch/energy gives the supplied acoustic detector structure.
        y = (np.where(t < 4, .03, .2) * np.sin(2 * np.pi * np.where(t < 7, 220, 880) * t)).astype('float32')
        plan = make_plan(y, sr, dict(min_duration=1, max_duration=4, sensitivity=.8, snap_to_beat=False))
        spans = plan['spans']
        self.assertGreater(len(spans), 1)
        self.assertEqual(spans[0][0], 0)
        self.assertEqual(spans[-1][1], len(y))
        self.assertTrue(all(1 <= (right-left)/sr <= 4 for left,right in spans))
        np.testing.assert_array_equal(np.concatenate([y[left:right] for left,right in spans]), y)

    def test_exact_clip_inputs_offsets_and_failure_continuation(self):
        sr = 8000
        y = np.arange(sr * 3, dtype=np.float32) / (sr * 3)
        diagnostic = {'boundaries': [{'sample': p, 'reason': 'test', 'novelty': 0} for p in [0, sr, 2*sr, 3*sr]]}
        with patch('segmented.acoustic_segments', return_value=([(0,sr),(sr,2*sr),(2*sr,3*sr)], diagnostic)):
            plan = make_plan(y, sr, offset_sample=10*sr)
        tester = FakeTester(fail_on=2)
        progress = []
        result = predict_plan(y, sr, plan, tester, progress=lambda done,total,row: progress.append(done))
        self.assertEqual(result['success_count'], 2)
        self.assertEqual([r['status'] for r in result['segments']], ['ok','error','ok'])
        self.assertEqual(progress, [1,2,3])
        for index, (clip, actual_sr) in enumerate(tester.inputs):
            self.assertEqual(actual_sr, sr)
            np.testing.assert_array_equal(clip, y[index*sr:(index+1)*sr])
        self.assertEqual(result['segments'][0]['start_seconds'],10)
        self.assertEqual(result['segments'][-1]['end_seconds'],13)
        rows = list(csv.DictReader(io.StringIO(wide_csv_bytes(result).decode('utf-8-sig'))))
        self.assertEqual(len(rows),3)
        self.assertEqual(rows[1]['status'],'error')
        self.assertEqual(rows[1]['happy'],'')
        self.assertEqual(sum(tag in rows[0] for tag in ZH),56)

    def test_subsecond_tail_padding_does_not_shift_boundaries(self):
        sr = 8000
        y = np.full(sr//2, .125, dtype='float32')
        plan = make_plan(y, sr, offset_sample=5*sr)
        tester = FakeTester()
        row = predict_plan(y, sr, plan, tester)['segments'][0]
        self.assertEqual((row['start_seconds'],row['end_seconds']),(5,5.5))
        self.assertEqual(row['padding_seconds'],.5)
        np.testing.assert_array_equal(tester.inputs[0][0][:sr//2],y)
        np.testing.assert_array_equal(tester.inputs[0][0][sr//2:],np.zeros(sr//2))

    def test_boundary_validation_and_selection(self):
        y = np.zeros(8000,dtype='float32')
        clip, offset = select_audio(y,8000,.25,.5)
        self.assertEqual((len(clip),offset),(4000,2000))
        for spans in [[(0,3),(2,8)],[(1,8)],[(0,7)],[]]:
            with self.assertRaises(ValueError): validate_spans(spans,8)
        with self.assertRaises(ValueError): make_plan(y,8000,dict(min_duration=3,max_duration=4))
        with self.assertRaises(ValueError): select_audio(y,8000,2,0)

    def test_one_second_cqt_compatibility(self):
        sr=22050
        config=SimpleNamespace(mp3={'song_hz':sr,'inst_len':10.0},
             feature={'n_bins':144,'bins_per_octave':24,'hop_length':2048},model={'timestep':108})
        t=np.arange(sr)/sr
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'one_second.wav'
            sf.write(p,.1*np.sin(2*np.pi*440*t),sr)
            features,unit,duration=chord_features(p,config)
        self.assertEqual(features.shape[0],144)
        self.assertTrue(np.isfinite(features).all())
        self.assertEqual(duration,1)

if __name__ == '__main__':
    unittest.main()
