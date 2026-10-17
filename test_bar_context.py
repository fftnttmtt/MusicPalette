import unittest
import numpy as np
import soundfile as sf
from bar_context import extend_interval,make_bar_plan,predict_bar_plan,csv_bytes,split_long_targets
from beat_core import build_analysis

class ContextTests(unittest.TestCase):
    sr=1000
    def extension(self,left,right,curve,a=6):
        times=np.arange(0,12.01,.05)
        return extend_interval(left,right,12000,self.sr,np.arange(0,12001,500),times,curve,a,.65)
    def test_low_change_and_edges(self):
        z=np.zeros(241)
        for left,right in [(0,1000),(11000,12000),(5000,7000)]:
            r=self.extension(left,right,z)
            self.assertEqual(r['prediction_duration_seconds'],6.)
            self.assertEqual(r['requested_target_seconds'],6.)
            self.assertEqual(r['target_duration_seconds'],6.)
            self.assertEqual(r['added_duration_seconds'],6.-(right-left)/1000)
            self.assertLessEqual(r['prediction_start_sample'],left)
            self.assertGreaterEqual(r['prediction_end_sample'],right)
            self.assertFalse(r['forced_expansion'])
    def test_one_boundary_expands_other_side(self):
        times=np.arange(0,12.01,.05);z=np.zeros(len(times));z[(times>=4.8)&(times<=5)]=1
        r=self.extension(5000,7000,z)
        self.assertEqual(r['prediction_start_sample'],5000)
        self.assertEqual(r['prediction_end_sample'],11000)
        self.assertFalse(r['forced_expansion'])
    def test_both_blocked_hard_minimum(self):
        r=self.extension(5000,6000,np.ones(241),a=8)
        self.assertEqual(r['prediction_duration_seconds'],3)
        self.assertTrue(r['forced_expansion'])
        self.assertEqual(r['stop_reason'],'strong_boundary')
    def test_three_seconds_respects_boundaries(self):
        r=self.extension(5000,8000,np.ones(241))
        self.assertEqual(r['prediction_duration_seconds'],3)
        self.assertFalse(r['forced_expansion'])
    def test_overlong_target_must_be_split_before_extension(self):
        with self.assertRaises(ValueError):self.extension(3000,10000,np.zeros(241))
    def test_short_audio_rejected(self):
        with self.assertRaises(ValueError):extend_interval(0,1000,2500,1000,[],[0,2.5],[0,0])
    def test_plan_prediction_offset_and_minimum(self):
        sr=22050;y=np.sin(np.arange(sr*8)*2*np.pi*440/sr).astype('float32')
        class Tracker:
            def __call__(self,y,sr):return np.arange(0,8,.5),np.arange(0,8.01,2)
        plan=make_bar_plan(y,sr,Tracker(),target_seconds=6,offset_sample=sr*10)
        class Tester:
            calls=0
            def predict(self,path,threshold):
                audio,rate=sf.read(path);assert len(audio)>=3*rate
                assert audio.max()>0
                self.calls+=1
                return {'valence':5.,'arousal':4.,'scores':[],'predicted_tags':[]}
        tester=Tester();r=predict_bar_plan(y,sr,plan,tester)
        self.assertEqual(r['success_count'],4)
        self.assertTrue(all(x['raw_prediction']['status']=='skipped_minimum' for x in r['segments']))
        self.assertEqual(r['segments'][0]['start_seconds'],10)
        self.assertEqual(r['segments'][-1]['end_seconds'],18)
        self.assertIn(b'prediction_duration_seconds',csv_bytes(r))
    def test_identical_context_reuses_inference(self):
        sr=1000;y=np.ones(3000,dtype='float32')
        row={'segment':1,'start_sample':0,'end_sample':1500,'prediction_start_sample':0,'prediction_end_sample':3000}
        plan={'num_samples':3000,'sample_rate':sr,'offset_sample':0,'segment_rows':[row,{**row,'segment':2,'start_sample':1500,'end_sample':3000}]}
        class Tester:
            calls=0
            def predict(self,*args,**kwargs):
                self.calls+=1
                return {'valence':1,'arousal':2,'scores':[],'predicted_tags':[]}
        tester=Tester();r=predict_bar_plan(y,sr,plan,tester)
        self.assertEqual(tester.calls,1)
        self.assertEqual(r['unique_inferences'],1)
    def test_target_met_but_three_second_minimum_wins(self):
        times=np.arange(0,12.01,.05)
        r=extend_interval(5000,5500,12000,1000,np.arange(0,12000,250),times,np.zeros(len(times)),3)
        self.assertGreaterEqual(r['prediction_duration_seconds'],3)
        self.assertGreater(r['prediction_beat_count'],2)
    def test_variable_tempo_does_not_change_target_seconds(self):
        times=np.arange(0,12.01,.05)
        beats=np.array([0,1000,2000,3000,4500,6500,9000,11000])
        r=extend_interval(0,3000,12000,1000,beats,times,np.zeros(len(times)),6.125)
        self.assertEqual(r['prediction_end_sample'],6125)
        self.assertEqual(r['prediction_beat_count'],5)
        self.assertEqual(r['prediction_duration_seconds'],6.125)
    def test_half_open_beat_count(self):
        r=self.extension(5000,8000,np.zeros(241),a=3)
        self.assertEqual(r['original_beat_count'],6)
        self.assertEqual(r['prediction_beat_count'],6)
        self.assertEqual(r['added_beat_count'],0)
    def test_invalid_target_and_missing_beats(self):
        times=np.arange(0,12.01,.05)
        for invalid in [0,-1,float('nan')]:
            with self.assertRaises(ValueError):extend_interval(0,3000,12000,1000,[],times,np.zeros(len(times)),invalid)
        r=extend_interval(0,3000,12000,1000,[],times,np.zeros(len(times)),6)
        self.assertEqual(r['beat_count_source'],'detected')
        self.assertEqual(r['detected_prediction_beat_count'],0)
        self.assertEqual(r['prediction_beat_count'],0)
        self.assertEqual(r['prediction_duration_seconds'],6)
    def test_four_beat_bar_extends_to_six_seconds_total(self):
        r=self.extension(5000,7000,np.zeros(241),a=6)
        self.assertEqual(r['original_beat_count'],4)
        self.assertEqual(r['prediction_duration_seconds'],6)
        self.assertEqual(r['prediction_beat_count'],12)
        self.assertEqual(r['added_beat_count'],8)
    def test_different_meters_have_same_duration_target(self):
        for x in [3,4,5]:
            r=self.extension(4000,4000+x*500,np.zeros(241),a=6)
            self.assertEqual(r['prediction_duration_seconds'],6.)
    def test_short_available_context_stops_at_audio_edges(self):
        r=extend_interval(1000,2000,4000,1000,[],np.arange(0,4.01,.05),np.zeros(81))
        self.assertEqual(r['prediction_duration_seconds'],4.)
        self.assertEqual(r['stop_reason'],'audio_edges')
        self.assertFalse(r['reached_target'])
    def test_proposal_does_not_check_boundary_beyond_actual_endpoint(self):
        times=np.array([0.,3.,6.1,12.]);curve=np.array([0.,0.,0.,1.])
        r=extend_interval(0,3000,12000,1000,[],times,curve,6.,.7)
        self.assertEqual(r['prediction_end_sample'],6000)
        self.assertTrue(r['reached_target'])

    def test_long_bar_splits_near_beats_with_balanced_parts_and_parent_identity(self):
        analysis=build_analysis(np.zeros(13000),1000,np.arange(0,14),[0,13])
        rows=split_long_targets(analysis['segments'],1000,6.,analysis['beat_samples'])
        self.assertEqual(len(rows),3)
        self.assertEqual(rows[0]['start_sample'],0)
        self.assertEqual(rows[-1]['end_sample'],13000)
        self.assertTrue(all(r['duration']<=6. for r in rows))
        self.assertTrue(all(r['bar_number']==1 and r['source_segment_index']==1 for r in rows))
        self.assertTrue(all(r['forced_split'] and r['kind']=='bar_part' for r in rows))
        self.assertEqual([r['split_part'] for r in rows],[1,2,3])
        self.assertTrue(all(r['split_count']==3 for r in rows))
        for left,right in zip(rows,rows[1:]):
            self.assertEqual(left['end_sample'],right['start_sample'])
            self.assertIn(right['start_sample'],analysis['beat_samples'])
            self.assertEqual(right['split_start_reason'],'detected_beat')
        self.assertEqual(analysis['segments'][0]['end_sample'],13000)
        self.assertEqual(len(analysis['segments']),1)

    def test_no_beat_split_covers_every_sample_with_no_tiny_remainder(self):
        analysis=build_analysis(np.zeros(13001),1000,[],[])
        rows=split_long_targets(analysis['segments'],1000,6.,[])
        self.assertEqual(len(rows),3)
        lengths=[r['end_sample']-r['start_sample'] for r in rows]
        self.assertEqual(sum(lengths),13001)
        self.assertLessEqual(max(lengths)-min(lengths),1)
        self.assertTrue(all(r['kind']=='unmetered_part' and r['bar_number'] is None for r in rows))
        self.assertTrue(all(r['split_start_reason']=='duration_limit_sample' for r in rows[1:]))

    def test_existing_short_and_exact_length_boundaries_are_not_moved(self):
        analysis=build_analysis(np.zeros(16000),1000,np.arange(17),[0,2,8,16])
        rows=split_long_targets(analysis['segments'],1000,6.,analysis['beat_samples'])
        self.assertEqual([(r['start_sample'],r['end_sample']) for r in rows[:2]],[(0,2000),(2000,8000)])
        self.assertFalse(rows[0]['forced_split']);self.assertFalse(rows[1]['forced_split'])
        edges={r['start_sample'] for r in rows}|{r['end_sample'] for r in rows}
        self.assertTrue(set(analysis['downbeat_samples']).issubset(edges))
        self.assertEqual([r['index'] for r in rows],list(range(1,len(rows)+1)))

    def test_fractional_second_limit_is_strict_at_sample_precision(self):
        sr=44100;limit=3.123456
        maximum=int(np.floor(limit*sr))
        analysis=build_analysis(np.zeros(maximum+1),sr,[],[])
        rows=split_long_targets(analysis['segments'],sr,limit,[])
        self.assertEqual(len(rows),2)
        self.assertTrue(all((r['end_sample']-r['start_sample'])/sr<=limit for r in rows))
        times=np.arange(0,10,.05)
        extended=extend_interval(0,1000,sr*10,sr,[],times,np.zeros(len(times)),limit)
        self.assertLessEqual(extended['prediction_duration_seconds'],limit)

    def test_split_plan_prediction_offsets_input_lengths_and_export_provenance(self):
        sr=1000;y=np.sin(np.arange(sr*13)*2*np.pi*44/sr).astype('float32')
        class Tracker:
            def __call__(self,y,sr):return np.arange(0,14),np.array([0,13])
        plan=make_bar_plan(y,sr,Tracker(),target_seconds=6.,offset_sample=5000)
        self.assertEqual(plan['beat_analysis']['bar_count'],1)
        self.assertEqual(plan['split_summary']['long_source_count'],1)
        self.assertEqual(plan['split_summary']['added_target_count'],2)
        self.assertEqual(plan['segment_rows'][0]['start_sample'],5000)
        self.assertEqual(plan['segment_rows'][-1]['end_sample'],18000)
        for row in plan['segment_rows']:
            self.assertLessEqual(row['duration_seconds'],6.)
            self.assertLessEqual(row['prediction_duration_seconds'],6.)
            self.assertGreaterEqual(row['prediction_duration_seconds'],3.)
            self.assertEqual(row['source_start_seconds'],5.)
            self.assertEqual(row['source_end_seconds'],18.)
        class Tester:
            def predict(self,path,threshold):
                audio,rate=sf.read(path)
                assert 3<=len(audio)/rate<=6
                return {'valence':5.,'arousal':5.,'scores':[],'predicted_tags':[]}
        predicted=predict_bar_plan(y,sr,plan,Tester())
        self.assertEqual(predicted['success_count'],3)
        self.assertTrue(all(row['raw_prediction']['status']=='ok' for row in predicted['segments']))
        self.assertIn(b'source_segment_index',csv_bytes(predicted))
        self.assertIn(b'forced_split',csv_bytes(predicted))

    def test_target_shorter_than_model_minimum_is_rejected(self):
        with self.assertRaises(ValueError):split_long_targets([],1000,2.,[])
        with self.assertRaises(ValueError):extend_interval(0,1000,4000,1000,[],[0.,4.],[0.,0.],2.)
if __name__=='__main__':unittest.main()
