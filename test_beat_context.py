"""Beat targets share the existing bar context and color algorithms."""
import csv
import io
import json
import unittest
from unittest.mock import patch
import numpy as np
from bar_context import make_bar_plan,predict_bar_plan,csv_bytes
from hsv_timeline import map_colors,color_svg
from color_transition import build_transition,transition_color,transition_csv
from tag_charts import tag_timeline_rows,load_timeline_json
from test_song_labels import waveform,make_plan,RecordingTester


def plan_for(y,beats,downbeats,offset=0,a=6.,unit='beat'):
    times=np.arange(0,len(y)/1000+.001,.05)
    features={key:np.zeros((1,len(times))) for key in ['timbre','chroma','energy']}
    tracker=lambda audio,sr:(np.asarray(beats),np.asarray(downbeats))
    with patch('bar_context.extract_features',return_value=(features,times,.05)), \
         patch('bar_context.foote_novelty',return_value=np.ones(len(times))):
        return make_bar_plan(y,1000,tracker,offset_sample=offset,target_seconds=a,prediction_unit=unit)


class BeatContextChecks(unittest.TestCase):
    def test_default_bar_path_is_unchanged_and_beat_mode_has_finer_targets(self):
        y=waveform(8)
        self.assertEqual(make_plan(y),make_plan(y,prediction_unit='bar'))
        beats=np.arange(0,8.001,.5);downbeats=np.arange(0,8.001,2)
        bar=plan_for(y,beats,downbeats,unit='bar')
        beat=plan_for(y,beats,downbeats)
        self.assertEqual(len(bar['segment_rows']),4)
        self.assertEqual(len(beat['segment_rows']),16)
        self.assertEqual(beat['mode'],'beat_context')
        self.assertEqual(beat['parameters']['prediction_unit'],'beat')
        self.assertEqual(bar['beat_analysis'],beat['beat_analysis'])
        self.assertEqual([r['beat_in_bar'] for r in beat['segment_rows'][:8]],[1,2,3,4]*2)
        self.assertEqual([r['bar_number'] for r in beat['segment_rows'][:8]],[1]*4+[2]*4)

    def test_variable_beats_preserve_exact_samples_offsets_and_audio_edges(self):
        y=waveform(8)
        beats=[.125,.625,1.25,1.875,2.5,3.25,4.,4.75,5.5,6.25,7.,7.625]
        plan=plan_for(y,beats+[float('nan'),-.1,8.5,.625],[1.25,4.,7.],offset=5000)
        rows=plan['segment_rows']
        edges=[5000]+[5000+round(b*1000) for b in beats]+[13000]
        self.assertEqual([rows[0]['start_sample']]+[r['end_sample'] for r in rows],edges)
        self.assertEqual(rows[0]['kind'],'beat_leading')
        self.assertEqual(rows[-1]['kind'],'beat_tail')
        self.assertEqual([r['beat_number'] for r in rows],[None]+list(range(1,len(beats)+1)))
        self.assertEqual(sum(r['end_sample']-r['start_sample'] for r in rows),len(y))
        first_bar=next(r for r in rows if r['start_sample']==6250)
        self.assertEqual(first_bar['bar_number'],1);self.assertEqual(first_bar['beat_in_bar'],1)
        self.assertEqual((first_bar['bar_start_seconds'],first_bar['bar_end_seconds']),(6.25,9.))
        for row in rows:
            self.assertTrue(3<=row['prediction_duration_seconds']<=6)
            self.assertLessEqual(row['prediction_start_sample'],row['start_sample'])
            self.assertGreaterEqual(row['prediction_end_sample'],row['end_sample'])

    def test_beat_predictions_use_real_contexts_and_predict_whole_song_once(self):
        full=waveform();y=full[5000:13000]
        plan=plan_for(y,np.arange(0,8.001,.5),np.arange(0,8.001,2),offset=5000)
        tester=RecordingTester();result=predict_bar_plan(y,1000,plan,tester,whole_audio=full)
        self.assertEqual(result['prediction_unit'],'beat');self.assertEqual(result['segment_count'],16)
        self.assertTrue(all(3000<=len(audio)<=6000 for audio in tester.inputs[:-1]))
        np.testing.assert_array_equal(tester.inputs[-1],full)
        self.assertEqual(sum(len(audio)==len(full) for audio in tester.inputs),1)
        self.assertEqual(result['unique_inferences'],len(tester.inputs))
        self.assertTrue(all(r['raw_prediction']['status']=='skipped_minimum' for r in result['segments']))
        exported=list(csv.DictReader(io.StringIO(csv_bytes(result).decode('utf-8-sig'))))
        self.assertTrue(all(r['prediction_unit']=='beat' for r in exported))
        self.assertEqual(exported[0]['beat_number'],'1')
        self.assertEqual(load_timeline_json(json.dumps(result))['prediction_unit'],'beat')
        self.assertEqual(len(tag_timeline_rows(result)),9*16)

    def test_short_prebar_interval_inherits_first_bar_first_beat(self):
        y=waveform(8);plan=make_plan(y,offset=5000,prediction_unit='beat')
        rows=plan['segment_rows'];donor=next(r for r in rows if r['start_sample']==5750)
        intro=[r for r in rows if r['end_sample']<=5750]
        self.assertEqual(len(intro),3)
        self.assertTrue(all(r['inherited_from_segment']==donor['segment'] for r in intro))
        self.assertTrue(all(r['prediction_source']=='first_bar_first_beat' for r in intro))
        tester=RecordingTester();result=predict_bar_plan(y,1000,plan,tester)
        expected=next(r for r in result['segments'] if r['segment']==donor['segment'])
        self.assertTrue(all(r['scores']==expected['scores'] for r in result['segments'][:3]))
        self.assertTrue(all(r['raw_prediction']['status']=='skipped_inherited' for r in result['segments'][:3]))
        no_bar=make_plan(y,downbeats=[.75],prediction_unit='beat')
        self.assertFalse(any(r['independent_prediction_skipped'] for r in no_bar['segment_rows']))

    def test_no_beat_fallback_and_long_beat_gap_respect_a_without_losing_audio(self):
        y=waveform(13)
        for beats,downbeats in [([],[]),([0.,13.],[0.,13.])]:
            plan=plan_for(y,beats,downbeats);rows=plan['segment_rows']
            self.assertEqual(rows[0]['start_sample'],0);self.assertEqual(rows[-1]['end_sample'],13000)
            self.assertEqual(len(rows),3)
            self.assertTrue(all(r['duration_seconds']<=6 for r in rows))
            self.assertEqual(sum(r['end_sample']-r['start_sample'] for r in rows),13000)
            self.assertTrue(all(a['end_sample']==b['start_sample'] for a,b in zip(rows,rows[1:])))
            if not beats:self.assertTrue(any('未检测到可用节拍' in w for w in plan['beat_analysis']['warnings']))
            else:self.assertTrue(all(r['beat_number']==1 and r['forced_split'] for r in rows))

    def test_color_anchors_and_csv_stay_exactly_on_beats_with_both_transition_modes(self):
        y=waveform(8);beats=[0.,.333,.91,1.8,2.13,3.2,4.,4.77,5.125,6.3,7.,8.]
        result=predict_bar_plan(y,1000,plan_for(y,beats,[0.,4.,8.],offset=5000),RecordingTester())
        for i,row in enumerate(result['segments']):
            row['scores']=[{'tag':'energetic' if i%2 else 'sad','score':.8}]
        mapped=map_colors(result,saturation_k=0,value_k=0)
        self.assertEqual(mapped['prediction_unit'],'beat')
        for mode in ['boundary_hold','foote_exponential']:
            mapped['transition']=build_transition(mapped,result,mode=mode,max_transition_fraction=.15)
            for row in mapped['segments']:
                self.assertEqual(transition_color(mapped['transition'],row['start_seconds'])['hex'],row['hex'])
            exported=list(csv.DictReader(io.StringIO(transition_csv(mapped).decode('utf-8-sig'))))
            by_time={float(r['time_seconds']):r['hex'] for r in exported}
            for row in mapped['segments']:self.assertEqual(by_time[row['start_seconds']],row['hex'])
            self.assertIn('节拍起点',color_svg(mapped))

    def test_unknown_prediction_unit_is_rejected(self):
        with self.assertRaises(ValueError):
            make_bar_plan(waveform(8),1000,lambda y,sr:([],[]),prediction_unit='quarter')


if __name__=='__main__':unittest.main()
