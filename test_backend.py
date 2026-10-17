import ast
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
import numpy as np
from backend import score_result, split_segments, patched_source, ZH, csv_bytes,duration_weighted_mert,VENDOR


def chord_block(source):
    tree=ast.parse(source)
    model=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Music2emo')
    predict=next(n for n in model.body if isinstance(n,ast.FunctionDef) and n.name=='predict')
    loop=next(n for n in predict.body if isinstance(n,ast.For) and isinstance(n.iter,ast.Name) and n.iter.id=='chords')
    return compile(ast.Module(body=[loop],type_ignores=[]),'<actual chord encoding>','exec')


def encode_chords(block,labels):
    data=VENDOR/'inference/data'
    scope={'chords':[(float(i),float(i+1),label) for i,label in enumerate(labels)],
           'chordRootDic':json.loads((data/'chord_root.json').read_text()),
           'chordAttrDic':json.loads((data/'chord_attr.json').read_text()),
           'chord_to_idx':json.loads((data/'chord.json').read_text()),
           'encoded':[],'encoded_root':[],'encoded_attr':[],'durations':[]}
    exec(block,scope)
    return {key:scope[key] for key in ['encoded','encoded_root','encoded_attr','durations']}

class BackendChecks(unittest.TestCase):
    def test_chord_encoding_with_modern_numpy(self):
        import mir_eval
        roots = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        qualities = ['min', 'maj', 'dim', 'aug', 'min6', 'maj6', 'min7',
                     'minmaj7', 'maj7', '7', 'dim7', 'hdim7', 'sus2', 'sus4']
        tags = [root + ':' + quality for root in roots for quality in qualities] + ['N', 'X']
        for tag in tags:
            root, bitmap, bass = mir_eval.chord.encode(tag)
            rotated = mir_eval.chord.rotate_bitmap_to_root(bitmap, root)
            self.assertEqual(rotated.shape, (12,))
        root, bitmap, bass = mir_eval.chord.encode('C')
        self.assertEqual(root, 0)
        np.testing.assert_array_equal(bitmap, [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])

    def test_independent_scores(self):
        result = score_result(list(ZH), np.zeros(56), 5, 5)
        self.assertAlmostEqual(sum(r['score'] for r in result['scores']), 27.5)
        self.assertTrue(csv_bytes(result['scores']).startswith(b'\xef\xbb\xbf'))

    def test_stable_extreme_logits(self):
        result = score_result(list(ZH), [1000, -1000] + [0]*54, 5, 5)
        self.assertEqual(result['scores'][0]['score'], 1)
        self.assertEqual(result['scores'][-1]['score'], 0)
        with self.assertRaises(ValueError):
            score_result(list(ZH), [float('nan')]*56, 5, 5)

    def test_complete_audio_coverage(self):
        for n in [300, 1000, 1001, 3001, 3500, 6000]:
            original = np.arange(n)
            segments = split_segments(original, 1000, 300)
            np.testing.assert_array_equal(np.concatenate(segments), original)
            self.assertTrue(all(len(s) >= 300 for s in segments))

    def test_source_adaptation(self):
        for device in ['cpu', 'cuda']:
            compile(patched_source(device), '<upstream>', 'exec')

    def test_unknown_chord_x_reproduces_old_error_and_uses_official_root_fallback(self):
        original=chord_block((VENDOR/'music2emo.py').read_text())
        fixed=chord_block(patched_source('cpu'))
        for labels in [['X'],['C','X','D:min','N']]:
            with self.assertRaises(KeyError) as error:encode_chords(original,labels)
            self.assertEqual(error.exception.args,('X',))
            out=encode_chords(fixed,labels)
            for index,label in enumerate(labels):
                if label=='X':
                    self.assertEqual(out['encoded'][index],157)
                    self.assertEqual(out['encoded_root'][index],0)
                    self.assertEqual(out['encoded_attr'][index],0)
            self.assertEqual(len(out['encoded_root']),len(labels))

    def test_supported_chord_encodings_stay_identical(self):
        labels=list(json.loads((VENDOR/'inference/data/chord.json').read_text()))
        labels.remove('X')
        original=chord_block((VENDOR/'music2emo.py').read_text())
        for device in ['cpu','cuda']:
            self.assertEqual(encode_chords(original,labels),encode_chords(chord_block(patched_source(device)),labels))

    def test_short_tail_is_weighted_by_actual_coverage(self):
        values=np.array([[0.,1.],[0.,1.],[1.,0.]],dtype=np.float32)
        pooled=duration_weighted_mert(values,[30,30,5])
        np.testing.assert_allclose(pooled,[5/65,60/65],rtol=1e-7)
        np.testing.assert_array_equal(duration_weighted_mert(values[:1],[5]),values[0])
        for bad in [[30,5],[30,30,0],[30,30,float('nan')]]:
            with self.assertRaises(ValueError):duration_weighted_mert(values,bad)

    def test_patched_mert_pipeline_covers_audio_with_numeric_chunk_weight_alignment(self):
        # Execute the real adapted feature/pooling block using a transparent
        # mean-valued extractor, without downloading or simulating model scores.
        tree=ast.parse(patched_source('cpu'))
        model=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Music2emo')
        predict=next(n for n in model.body if isinstance(n,ast.FunctionDef) and n.name=='predict')
        first=next(i for i,n in enumerate(predict.body) if isinstance(n,ast.Assign)
                   and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='torchaudio.load')
        last=next(i for i,n in enumerate(predict.body) if isinstance(n,ast.Assign)
                  and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='torch.from_numpy')
        block=ast.Module(body=predict.body[first:last],type_ignores=[])
        original=np.concatenate([np.full(30,i,dtype=np.float32) for i in range(11)]+[np.full(5,11,dtype=np.float32)])
        seen=[]
        class Extractor:
            def extract_features_from_segment(self,audio,sr,path):
                seen.append(len(audio))
                np.save(path,np.full((1,12,768),audio.mean(),dtype=np.float32))
        with tempfile.TemporaryDirectory() as tmp:
            scope={'self':types.SimpleNamespace(feature_extractor=Extractor()),'audio':'whole.wav',
                   'torchaudio':types.SimpleNamespace(load=lambda path:(original[None,:],1)),
                   'resample_waveform':lambda audio,sr,target:(audio,sr),'resample_rate':1,
                   'is_split':True,'split_audio':lambda audio,sr:split_segments(audio,30*sr,3*sr),
                   'mert_dir':Path(tmp),'os':os,'np':np,'duration_weighted_mert':duration_weighted_mert}
            exec(compile(block,'<actual MERT aggregation>','exec'),scope)
        self.assertEqual(seen,[30]*11+[5])
        np.testing.assert_allclose(scope['final_embedding_mert'],original.mean(),rtol=1e-7)

if __name__ == '__main__':
    unittest.main()
