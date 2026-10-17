import hashlib
import http.client
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch, Mock

from spectra_prediction import DEFAULTS, make_spectra_bundle, predict_file
from spectra_server import Jobs, make_server


def result_fixture():
    rows = [dict(segment=i+1, status='ok', start_seconds=i*4., end_seconds=(i+1)*4.,
        arousal=3.+i, valence=7.-i,
        scores=[{'tag': 'happy', 'score': .5}, {'tag': 'love', 'score': .25}, {'tag': 'funny', 'score': 1.}]) for i in range(2)]
    return dict(segments=rows, sample_rate=100, num_samples=800, offset_sample=0,
        audio_name='test.wav', audio_sha256='abc', prediction_unit='bar',
        beat_analysis=dict(beats=[i*.5 for i in range(16)], downbeats=[0.,4.],
            segments=[dict(index=i+1, start=i*4., end=(i+1)*4., kind='bar', label=f'小节 {i+1}',bar_number=i+1) for i in range(2)]),
        diagnostic=dict(time_s=[0.,4.,8.], novelty=[.1,.6,.9]))


class SpectraBridgeChecks(unittest.TestCase):
    def test_defaults_and_export_use_current_mapper_and_preserve_beat_anchors(self):
        from color_transition import transition_color
        r=result_fixture()
        r['whole_song_prediction']={**r['segments'][0], 'segment':0, 'start_seconds':0., 'end_seconds':8.}
        before=json.dumps(r,sort_keys=True)
        b=make_spectra_bundle(r)
        self.assertEqual(json.dumps(r,sort_keys=True),before)
        json.dumps(b,allow_nan=False)
        self.assertEqual(b['beats'],r['beat_analysis']['beats'])
        self.assertEqual(b['defaults']['device'],'cuda')
        self.assertEqual(b['defaults']['beat_model'],'final0')
        self.assertEqual(b['defaults']['prediction_unit'],'bar')
        self.assertEqual(b['defaults']['target_seconds'],6.)
        self.assertEqual(b['defaults']['motif_score_gamma'],2.)
        self.assertEqual(b['defaults']['boundary_threshold'],.7)
        self.assertEqual(b['mapping']['parameters']['hue_gamma'],2.)
        self.assertEqual(b['mapping']['parameters']['saturation_k'],0.)
        self.assertEqual(b['mapping']['parameters']['value_k'],0.)
        self.assertEqual(b['transition']['max_segment_fraction'],.15)
        self.assertEqual(b['transition']['mode'],'boundary_start')
        self.assertEqual(b['transition']['strength'],8.)
        self.assertIn('whole_song_color',b)
        for row in b['segments']:
            self.assertNotIn('funny',row['tag_scores'])
        for event in b['transition']['runs'][0]['events']:
            self.assertEqual(event['start_seconds'],event['boundary_seconds'])
            self.assertEqual(transition_color(b['transition'],event['start_seconds']-1e-6)['hex'],b['segments'][event['from_index']]['hex'])
            self.assertEqual(transition_color(b['transition'],event['end_seconds'])['hex'],b['segments'][event['to_index']]['hex'])

    def test_real_pipeline_call_uses_cuda_defaults_and_reuses_models(self):
        import numpy as np
        audio=np.ones((800,2),dtype=np.float32)
        resources={}
        tracker=object(); tester=object()
        with patch('beat_core.load_audio',return_value=(audio,100)), \
             patch('beat_core.get_tracker',return_value=(tracker,'cuda')) as get_tracker, \
             patch('backend.missing_files',return_value=[]), \
             patch('backend.Tester',return_value=tester) as make_tester, \
             patch('bar_context.make_bar_plan',return_value={'plan':True}) as make_plan, \
             patch('bar_context.predict_bar_plan',return_value=result_fixture()) as predict:
            for unit in ('bar','beat'):
                predict_file(Path('test.wav'),prediction_unit=unit,resources=resources)
                args=make_plan.call_args
                self.assertIs(args.args[2],tracker)
                self.assertEqual(args.kwargs,dict(target_seconds=6.,boundary_threshold=.7,context_seconds=2.,model='final0',device='cuda',prediction_unit=unit))
                self.assertIs(predict.call_args.args[3],tester)
                self.assertEqual(predict.call_args.kwargs['compare_raw'],True)
                self.assertIs(predict.call_args.kwargs['whole_audio'],predict.call_args.args[0])
            get_tracker.assert_called_once_with('final0','cuda')
            make_tester.assert_called_once_with('cuda')

    def test_offset_export_does_not_move_original_boundaries(self):
        r=result_fixture();r['offset_sample']=1000
        for row in r['segments']:
            row['start_seconds']+=10;row['end_seconds']+=10
        b=make_spectra_bundle(r)
        self.assertEqual(b['beats'][0],10.)
        self.assertEqual(b['bars'][0]['start_seconds'],10.)
        self.assertEqual(b['segments'][0]['start_seconds'],10.)

    def test_causal_fade_caps_to_receiving_interval_and_is_idempotent(self):
        from color_transition import build_transition, start_aligned_transition, transition_color
        mapped={'segments':[dict(segment=i+1,status='ok',start_seconds=a,end_seconds=b,
            rgb=rgb,hex=color) for i,(a,b,rgb,color) in enumerate([
                (0.,4.,[255,0,0],'#ff0000'),(4.,4.2,[0,0,255],'#0000ff'),
                (4.2,8.,[0,255,0],'#00ff00')])]}
        original=build_transition(mapped,{},mode='boundary_hold',max_transition_fraction=.15)
        curve=start_aligned_transition(original)
        e=curve['runs'][0]['events'][0]
        self.assertEqual(e['start_seconds'],4.)
        self.assertAlmostEqual(e['duration_seconds'],.03)
        self.assertLessEqual(e['end_seconds'],4.2)
        self.assertEqual(transition_color(curve,3.99)['hex'],'#ff0000')
        self.assertEqual(transition_color(curve,4.)['hex'],'#ff0000')
        self.assertNotIn(transition_color(curve,4.015)['hex'],['#ff0000','#0000ff'])
        self.assertEqual(transition_color(curve,4.03)['hex'],'#0000ff')
        self.assertEqual(start_aligned_transition(curve),curve)
        self.assertNotEqual(original['mode'],curve['mode'])

    def test_causal_strong_boundary_instant_change_and_failed_gap_never_anticipate(self):
        from color_transition import build_transition, start_aligned_transition, transition_color
        mapped={'segments':[dict(segment=i+1,status='ok',start_seconds=i*4.,end_seconds=(i+1)*4.,
            rgb=rgb,hex=color) for i,(rgb,color) in enumerate([([255,0,0],'#ff0000'),([0,0,255],'#0000ff')])]}
        curve=start_aligned_transition(build_transition(mapped,{'diagnostic':{'time_s':[0.,4.,8.],'novelty':[1.,1.,1.]}},mode='boundary_hold',max_transition_fraction=.15))
        self.assertEqual(transition_color(curve,4.-1e-6)['hex'],'#ff0000')
        self.assertEqual(transition_color(curve,4.)['hex'],'#0000ff')
        mapped['segments'][0]['end_seconds']=2.
        gap=start_aligned_transition(build_transition(mapped,{},mode='boundary_hold',max_transition_fraction=.15))
        self.assertIsNone(transition_color(gap,3.9))
        self.assertEqual(transition_color(gap,4.)['hex'],'#0000ff')

    def test_http_upload_progress_hash_unit_and_result_without_model_weights(self):
        payload=b'fake-audio-for-contract-test'
        observed={}
        entered=threading.Event();release=threading.Event()
        def predictor(path,**kwargs):
            observed.update(kwargs);observed['bytes']=path.read_bytes();observed['path']=path
            kwargs['progress']('working',.5);entered.set();release.wait(3)
            return {'schema':'test','audio':{'sha256':kwargs['sha256']}}
        jobs=Jobs(predictor)
        with tempfile.TemporaryDirectory() as root:
            Path(root,'index.html').write_text('SPECTRA')
            server=make_server(0,jobs=jobs,web_root=root)
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            conn=http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=4)
            try:
                conn.request('GET','/api/config');r=conn.getresponse()
                self.assertEqual(json.loads(r.read())['defaults']['hue_gamma'],2.)
                conn.request('POST','/api/analyze',payload,{'X-Spectra-Client':'1','Origin':'https://elsewhere.invalid'})
                r=conn.getresponse();r.read();self.assertEqual(r.status,403)
                conn.request('POST','/api/analyze?unit=beat',payload,{'X-Spectra-Client':'1','X-Audio-Name':'sample.wav'})
                r=conn.getresponse();key=json.loads(r.read())['job_id'];self.assertEqual(r.status,202)
                self.assertTrue(entered.wait(2))
                conn.request('POST','/api/analyze',payload,{'X-Spectra-Client':'1'})
                r=conn.getresponse();r.read();self.assertEqual(r.status,409)
                conn.request('GET','/api/jobs/'+key);status=json.loads(conn.getresponse().read())
                self.assertEqual(status['progress'],.5)
                release.set()
                deadline=time.monotonic()+3
                while time.monotonic()<deadline:
                    conn.request('GET','/api/jobs/'+key);status=json.loads(conn.getresponse().read())
                    if status['status']=='done':break
                    time.sleep(.01)
                self.assertEqual(status['status'],'done')
                self.assertEqual(status['result']['audio']['sha256'],hashlib.sha256(payload).hexdigest())
                self.assertEqual(observed['prediction_unit'],'beat')
                self.assertEqual(observed['bytes'],payload)
                jobs.executor.shutdown(wait=True)
                self.assertFalse(observed['path'].exists())
                conn.request('GET','/../backend.py');r=conn.getresponse();r.read();self.assertEqual(r.status,404)
            finally:
                release.set();conn.close();server.shutdown();server.server_close();jobs.executor.shutdown(wait=True);thread.join(2)


if __name__=='__main__':unittest.main()
