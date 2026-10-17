"""Opening-result inheritance and independent full-audio color labels."""
import copy
import json
import unittest
from unittest.mock import patch
import numpy as np
import soundfile as sf
from bar_context import make_bar_plan,predict_bar_plan,csv_bytes
from hsv_timeline import map_colors
from color_transition import build_transition,transition_color
from tag_charts import load_timeline_json


def waveform(seconds=24,sr=1000):
    return np.linspace(-.8,.8,round(seconds*sr),dtype=np.float32)


def make_plan(y,leading=.75,offset=0,downbeats=None,a=6.,prediction_unit='bar'):
    sr=1000
    class Tracker:
        def __call__(self,audio,rate):
            return np.arange(0,len(audio)/rate,.25), (np.arange(leading,len(audio)/rate+.001,2.)
                if downbeats is None else np.asarray(downbeats))
    times=np.arange(0,len(y)/sr+.001,.05)
    features={key:np.zeros((1,len(times))) for key in ['timbre','chroma','energy']}
    with patch('bar_context.extract_features',return_value=(features,times,.05)), \
         patch('bar_context.foote_novelty',return_value=np.ones(len(times))):
        return make_bar_plan(y,sr,Tracker(),target_seconds=a,offset_sample=offset,prediction_unit=prediction_unit)


class RecordingTester:
    def __init__(self,fail_whole=False,fail_bars=False):
        self.inputs=[];self.fail_whole=fail_whole;self.fail_bars=fail_bars
    def predict(self,path,threshold):
        audio,sr=sf.read(path,dtype='float32');self.inputs.append(audio.copy())
        if self.fail_whole and len(audio)>6000:raise RuntimeError('whole failed')
        if self.fail_bars and len(audio)<=6000:raise RuntimeError('bar failed')
        assert sr==1000 and len(audio)>=3000
        return {'valence':7. if len(audio)>6000 else 3.,'arousal':6.,
                'scores':[{'tag':'happy' if len(audio)>6000 else 'calm','score':.8,'中文':'测试'},
                          {'tag':'funny','score':1.,'中文':'滑稽'}],
                'threshold':threshold,'predicted_tags':[]}


class SongLabelChecks(unittest.TestCase):
    def test_short_opening_inherits_first_bar_input_and_result_without_moving_edges(self):
        y=waveform(8);plan=make_plan(y,offset=5000)
        before=copy.deepcopy(plan);first,donor=plan['segment_rows'][:2]
        self.assertEqual((first['start_sample'],first['end_sample']),(5000,5750))
        self.assertEqual(donor['start_sample'],5750)
        self.assertTrue(first['independent_prediction_skipped'])
        self.assertEqual(first['inherited_from_segment'],donor['segment'])
        for key in ['prediction_start_sample','prediction_end_sample','prediction_duration_seconds']:
            self.assertEqual(first[key],donor[key])
        tester=RecordingTester();result=predict_bar_plan(y,1000,plan,tester)
        first,donor=result['segments'][:2]
        for key in ['status','scores','valence','arousal']:
            self.assertEqual(first[key],donor[key])
        self.assertEqual(first['raw_prediction']['status'],'skipped_inherited')
        donor_audio=y[donor['prediction_start_sample']-5000:donor['prediction_end_sample']-5000]
        self.assertEqual(sum(np.array_equal(x,donor_audio) for x in tester.inputs),1)
        mapped=map_colors(result,saturation_k=0,value_k=0)
        self.assertEqual(mapped['segments'][0]['hex'],mapped['segments'][1]['hex'])
        self.assertEqual(plan,before)
        self.assertIn(b'inherited_from_segment',csv_bytes(result))

    def test_one_second_threshold_is_strict_in_samples(self):
        for duration,inherit in [(.001,True),(.999,True),(1.,False),(1.001,False)]:
            plan=make_plan(waveform(8),leading=duration)
            self.assertEqual(plan['segment_rows'][0]['independent_prediction_skipped'],inherit)

    def test_inherited_opening_and_fifteen_percent_fades_keep_all_start_anchors(self):
        y=waveform(8);plan=make_plan(y,offset=5000)
        result=predict_bar_plan(y,1000,plan,RecordingTester())
        # Use a distinct later prediction to exercise a visible transition.
        result['segments'][2]['scores']=[{'tag':'sad','score':.8}]
        mapped=map_colors(result,saturation_k=0,value_k=0)
        for mode in ['boundary_hold','foote_exponential']:
            curve=build_transition(mapped,result,mode=mode,max_transition_fraction=.15)
            self.assertEqual(transition_color(curve,5.)['hex'],mapped['segments'][1]['hex'])
            for row in mapped['segments']:
                self.assertEqual(transition_color(curve,row['start_seconds'])['hex'],row['hex'])
            if mode=='boundary_hold':
                for run in curve['runs']:
                    for event in run['events']:
                        self.assertLessEqual(event['duration_seconds'],.15*event['anchor_interval_seconds']+1e-9)
                        self.assertEqual(event['end_seconds'],event['boundary_seconds'])

    def test_no_complete_first_bar_means_no_invented_donor(self):
        for beats in [[.5],[]]:
            plan=make_plan(waveform(8),downbeats=beats)
            self.assertFalse(any(r['independent_prediction_skipped'] for r in plan['segment_rows']))
        plan=make_plan(waveform(8),leading=0.)
        self.assertEqual(plan['segment_rows'][0]['kind'],'bar')
        self.assertFalse(plan['segment_rows'][0]['independent_prediction_skipped'])

    def test_opening_uses_first_split_of_overlong_first_bar(self):
        plan=make_plan(waveform(8),downbeats=[.75,8.],a=3.)
        leading,first=plan['segment_rows'][:2]
        self.assertEqual(first['kind'],'bar_part');self.assertEqual(first['split_part'],1)
        self.assertEqual(first['bar_number'],1)
        self.assertEqual(leading['inherited_from_segment'],first['segment'])
        self.assertEqual(leading['prediction_start_sample'],first['prediction_start_sample'])

    def test_full_audio_input_is_independent_of_selected_range_and_a(self):
        full=waveform();selected=full[5000:13000];plan=make_plan(selected,offset=5000)
        tester=RecordingTester();progress=[]
        result=predict_bar_plan(selected,1000,plan,tester,threshold=.45,whole_audio=full,
            progress=lambda done,total,row:progress.append((done,total,row['kind'])))
        self.assertTrue(np.array_equal(tester.inputs[-1],full))
        self.assertTrue(all(3000<=len(x)<=6000 for x in tester.inputs[:-1]))
        song=result['whole_song_prediction']
        self.assertEqual((song['start_seconds'],song['end_seconds']),(0.,24.))
        self.assertEqual(song['analysis_scope'],'whole_track');self.assertEqual(song['threshold'],.45)
        self.assertEqual(song['valence'],7.);self.assertEqual(result['segments'][1]['valence'],3.)
        self.assertEqual(result['unique_inferences'],len(tester.inputs))
        self.assertEqual(result['whole_song_inferences'],1)
        self.assertEqual(progress[-1],(result['segment_count']+1,result['segment_count']+1,'whole_song'))
        json.dumps(result,allow_nan=False)

    def test_full_and_bar_inference_errors_remain_separate(self):
        full=waveform();selected=full[:8000];plan=make_plan(selected)
        for fail_whole,fail_bars in [(True,False),(False,True)]:
            result=predict_bar_plan(selected,1000,plan,RecordingTester(fail_whole,fail_bars),whole_audio=full)
            self.assertEqual(result['whole_song_prediction']['status'],'error' if fail_whole else 'ok')
            self.assertEqual(result['success_count'],0 if fail_bars else result['segment_count'])
            self.assertEqual(result['segments'][0]['status'],result['segments'][1]['status'])
            error=result['whole_song_prediction'] if fail_whole else result['segments'][0]
            self.assertEqual(error['error_type'],'RuntimeError')
            self.assertIn('Traceback (most recent call last)',error['error_traceback'])

    def test_identical_whole_waveform_reuses_existing_full_input(self):
        full=waveform(3);plan=make_plan(full,downbeats=[])
        tester=RecordingTester();result=predict_bar_plan(full,1000,plan,tester,whole_audio=full.copy())
        self.assertEqual(len(tester.inputs),1)
        self.assertEqual(result['whole_song_inferences'],0)
        self.assertEqual(result['whole_song_prediction']['end_seconds'],3.)

    def test_whole_color_uses_current_mapping_without_changing_segment_statistics(self):
        full=waveform();selected=full[5000:13000];plan=make_plan(selected,offset=5000)
        result=predict_bar_plan(selected,1000,plan,RecordingTester(),whole_audio=full)
        imported=load_timeline_json(json.dumps(result))
        self.assertNotIn('funny',[x['tag'] for x in imported['whole_song_prediction']['scores']])
        self.assertEqual(imported,load_timeline_json(json.dumps(imported)))
        # Deliberately keep a different global hue from all local predictions.
        mapped=map_colors(imported,saturation_k=0,value_k=0,enhance_sv=True,enhancement_strength=.5,
                          saturation_range=(.2,1.),value_range=(.25,1.),use_song_mean_centers=True)
        local=copy.deepcopy(imported);local.pop('whole_song_prediction')
        expected=map_colors(local,saturation_k=0,value_k=0,enhance_sv=True,enhancement_strength=.5,
                            saturation_range=(.2,1.),value_range=(.25,1.),use_song_mean_centers=True)
        self.assertEqual(mapped['segments'],expected['segments'])
        for field in ['sv_adaptation','song_means','parameters']:
            self.assertEqual(mapped['mapping'][field],expected['mapping'][field])
        song=mapped['whole_song_color']
        self.assertAlmostEqual(song['hue_degrees'],60.)
        self.assertAlmostEqual(mapped['segments'][0]['hue_degrees'],180.)
        self.assertAlmostEqual(song['value'],.25+.75*((7.-1.)/8.))
        changed=map_colors(imported,saturation_k=0,value_k=0,saturation_gamma=2.,value_gamma=2.,
                           saturation_range=(.2,1.),value_range=(.25,1.))['whole_song_color']
        self.assertAlmostEqual(changed['saturation'],.2+.8*((6.-1.)/8.)**2)
        self.assertAlmostEqual(changed['value'],.25+.75*((7.-1.)/8.)**2)
        json.dumps(mapped,allow_nan=False)

    def test_whole_color_is_independent_of_local_crop_statistics_and_failures(self):
        full=waveform();result=predict_bar_plan(full[:8000],1000,make_plan(full[:8000]),
                                               RecordingTester(),whole_audio=full)
        reference=None
        for values in [[2.,3.],[7.,8.],[]]:
            changed=copy.deepcopy(result)
            changed['segments']=[{**result['segments'][1],'segment':i+1,'valence':v,'arousal':v,
                                  'start_seconds':i*3.,'end_seconds':(i+1)*3.} for i,v in enumerate(values)]
            mapped=map_colors(changed,saturation_k=5.,value_k=7.,enhance_sv=True,
                              enhancement_strength=.5,use_song_mean_centers=True,
                              saturation_range=(.2,1.),value_range=(.25,1.))
            current=(mapped['whole_song_color'],mapped['whole_song_mapping'])
            if reference is None:reference=current
            else:self.assertEqual(current,reference)
            self.assertEqual(mapped['whole_song_mapping']['statistics_scope'],'whole_song_prediction_only')
            self.assertAlmostEqual(mapped['whole_song_mapping']['parameters']['saturation_center'],(6.-1.)/8.)
            self.assertAlmostEqual(mapped['whole_song_mapping']['parameters']['value_center'],(7.-1.)/8.)

    def test_whole_color_keeps_manual_centers_and_linear_global_va(self):
        full=waveform();result=predict_bar_plan(full[:8000],1000,make_plan(full[:8000]),
                                               RecordingTester(),whole_audio=full)
        mapped=map_colors(result,saturation_k=0,value_k=0,enhance_sv=True,enhancement_strength=1.,
                          saturation_center=.2,value_center=.8,use_song_mean_centers=False)
        song=mapped['whole_song_color'];params=mapped['whole_song_mapping']['parameters']
        self.assertEqual(params['saturation_center'],.2);self.assertEqual(params['value_center'],.8)
        self.assertEqual(song['saturation'],(6.-1.)/8.);self.assertEqual(song['value'],(7.-1.)/8.)


if __name__=='__main__':unittest.main()
