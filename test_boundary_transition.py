import csv
import io
import json
from pathlib import Path
import shutil
import subprocess
import unittest
from xml.etree import ElementTree as ET
from color_transition import build_transition, transition_color, transition_csv
from hsv_timeline import color_svg
from test_color_transition import mapped_rows


def long_rows(offset=0):
    mapped=mapped_rows(offset)
    for i,row in enumerate(mapped['segments']):
        row.update(start_seconds=offset+i*4.,end_seconds=offset+(i+1)*4.)
    return mapped


def constant_curve(value):
    return {'diagnostic':{'time_s':[0.,4.,8.],'novelty':[value]*3}}


class BoundaryTransitionChecks(unittest.TestCase):
    def test_absolute_foote_level_changes_duration_without_normalization_cancellation(self):
        mapped=long_rows()
        low=build_transition(mapped,constant_curve(0),mode='boundary_hold')
        high=build_transition(mapped,constant_curve(1),mode='boundary_hold')
        a,b=low['runs'][0]['events'][0],high['runs'][0]['events'][0]
        self.assertEqual(a['duration_seconds'],4.)
        self.assertEqual(b['duration_seconds'],0.)
        self.assertEqual(transition_color(high,3.)['hex'],'#ff0000')
        self.assertNotEqual(transition_color(low,3.)['hex'],'#ff0000')
        # In the original normalized-speed scheme these uniform fields agree.
        old_low=build_transition(mapped,constant_curve(0))
        old_high=build_transition(mapped,constant_curve(1))
        self.assertEqual(transition_color(old_low,4.3),transition_color(old_high,4.3))

    def test_original_starts_are_exact_color_anchors_with_no_lag(self):
        curve=build_transition(long_rows(),constant_curve(.5),mode='boundary_hold')
        self.assertEqual(transition_color(curve,0.)['hex'],'#ff0000')
        event=curve['runs'][0]['events'][0]
        self.assertEqual(transition_color(curve,event['start_seconds'])['hex'],'#ff0000')
        self.assertEqual(event['end_seconds'],4.)
        end=event['end_seconds']
        for t in [end,8.]:self.assertEqual(transition_color(curve,t)['hex'],'#0000ff')
        self.assertNotIn(transition_color(curve,(event['start_seconds']+end)/2)['hex'],['#ff0000','#0000ff'])
        self.assertEqual(transition_color(curve,4.-1e-8),transition_color(curve,4.+1e-8))

    def test_short_targets_cap_duration_and_cannot_overlap(self):
        mapped=long_rows()
        mapped['segments'][0].update(start_seconds=0.,end_seconds=.1)
        mapped['segments'][1].update(start_seconds=.1,end_seconds=.2)
        mapped['segments'].append({**mapped['segments'][0],'segment':3,'start_seconds':.2,'end_seconds':5.})
        curve=build_transition(mapped,{},mode='boundary_hold')
        events=curve['runs'][0]['events']
        self.assertAlmostEqual(events[0]['duration_seconds'],.1)
        self.assertFalse(events[0]['duration_capped'])
        self.assertFalse(events[1]['duration_capped'])
        self.assertLessEqual(events[0]['end_seconds'],events[1]['start_seconds'])
        self.assertEqual(transition_color(curve,.2)['hex'],'#ff0000')
        for row in mapped['segments']:
            self.assertEqual(transition_color(curve,row['start_seconds'])['hex'],row['hex'])

    def test_missing_foote_and_failure_or_gap_do_not_create_false_fades(self):
        mapped=long_rows()
        curve=build_transition(mapped,{},mode='boundary_hold')
        event=curve['runs'][0]['events'][0]
        self.assertFalse(event['novelty_available'])
        self.assertIsNone(event['novelty'])
        self.assertEqual(event['duration_seconds'],4.)
        unrelated=build_transition(mapped,{'diagnostic':{'time_s':[100.,101.],'novelty':[1.,1.]}},mode='boundary_hold')
        self.assertFalse(unrelated['runs'][0]['events'][0]['novelty_available'])
        self.assertEqual(unrelated['runs'][0]['events'][0]['duration_seconds'],4.)
        mapped['segments'][1].update(hex=None,rgb=None,status='prediction_error')
        mapped['segments'].append({**mapped['segments'][0],'segment':3,'start_seconds':8.,'end_seconds':12.})
        failed=build_transition(mapped,{},mode='boundary_hold')
        self.assertFalse(any(run['events'] for run in failed['runs']))
        self.assertIsNone(transition_color(failed,6.))
        mapped['segments'].pop(1)
        gap=build_transition(mapped,{},mode='boundary_hold')
        self.assertFalse(any(run['events'] for run in gap['runs']))
        self.assertIsNone(transition_color(gap,6.))

    def test_partial_audio_offset_and_local_peak_measurement(self):
        result={'diagnostic':{'time_s':[0.,3.8,4.,4.2,8.],'novelty':[0.,0.,0.,1.,0.]}}
        a=build_transition(long_rows(),result,mode='boundary_hold')
        b=build_transition(long_rows(90),{**result,'offset_sample':9000,'sample_rate':100},mode='boundary_hold')
        self.assertEqual(a['runs'][0]['events'][0]['novelty_peak'],1.)
        self.assertLess(a['runs'][0]['events'][0]['novelty_mean'],1.)
        self.assertAlmostEqual(b['runs'][0]['events'][0]['end_seconds'],94.)
        for t in [3.9,4.,4.05,4.15,6.]:
            self.assertEqual(transition_color(a,t)['rgb'],transition_color(b,t+90)['rgb'])

    def test_parameters_and_sampling_independence_exports(self):
        mapped=long_rows()
        curve=build_transition(mapped,constant_curve(.5),strength=0.,mode='boundary_hold')
        self.assertAlmostEqual(curve['runs'][0]['events'][0]['duration_seconds'],2.)
        exact=transition_color(curve,4.31)
        for fps in [10,30,60]:
            mapped['transition']=curve
            data=list(csv.DictReader(io.StringIO(transition_csv(mapped,fps).decode('utf-8-sig'))))
            self.assertEqual(next(r for r in data if float(r['time_seconds'])==4.)['hex'],'#0000ff')
        self.assertEqual(exact,transition_color(curve,4.31))
        self.assertIn('边界持色过渡',color_svg(mapped))
        self.assertIn('duration_seconds',json.dumps(curve,allow_nan=False))
        for options in [dict(boundary_radius=-1),dict(mode='unknown')]:
            with self.assertRaises(ValueError):build_transition(mapped,{},**options)

    def test_moderate_foote_retains_long_transition(self):
        for novelty in [.1,.3,.5,.7]:
            curve=build_transition(long_rows(),constant_curve(novelty),mode='boundary_hold')
            event=curve['runs'][0]['events'][0]
            self.assertGreater(event['duration_seconds'],3.5)
        curve=build_transition(long_rows(),constant_curve(.5),mode='boundary_hold')
        self.assertAlmostEqual(curve['runs'][0]['events'][0]['duration_seconds'],3.928055160151634,places=6)

    def test_duration_is_limited_to_interval_between_original_start_anchors(self):
        mapped=mapped_rows()
        curve=build_transition(mapped,constant_curve(0),mode='boundary_hold')
        event=curve['runs'][0]['events'][0]
        self.assertEqual(event['duration_seconds'],2.)
        self.assertFalse(event['duration_capped'])
        mapped['segments'][0].update(end_seconds=.1)
        mapped['segments'][1].update(start_seconds=.1,end_seconds=2.1)
        curve=build_transition(mapped,{},mode='boundary_hold')
        self.assertAlmostEqual(curve['runs'][0]['events'][0]['duration_seconds'],.1)
        limited=build_transition(mapped,{},mode='boundary_hold',max_transition_fraction=.5)
        self.assertAlmostEqual(limited['runs'][0]['events'][0]['duration_seconds'],.05)

    def test_time_weighted_mean_does_not_treat_isolated_spike_as_sustained_strong_change(self):
        result={'diagnostic':{'time_s':[0.,3.99,4.,4.01,8.],'novelty':[0.,0.,1.,0.,0.]}}
        average=build_transition(long_rows(),result,mode='boundary_hold')
        peak=build_transition(long_rows(),result,mode='boundary_hold',boundary_statistic='peak')
        a,b=average['runs'][0]['events'][0],peak['runs'][0]['events'][0]
        self.assertAlmostEqual(a['novelty_mean'],.02)
        self.assertEqual(a['novelty_peak'],1.)
        self.assertGreater(a['duration_seconds'],3.9)
        self.assertEqual(b['duration_seconds'],0.)
        for options in [dict(max_transition_fraction=0),dict(max_transition_fraction=1.1),dict(boundary_statistic='unknown')]:
            with self.assertRaises(ValueError):build_transition(long_rows(),{},**options)

    def test_no_fixed_second_limits_for_long_or_short_targets(self):
        mapped=long_rows()
        mapped['segments'][0]['end_seconds']=20.
        mapped['segments'][1].update(start_seconds=20.,end_seconds=24.)
        curve=build_transition(mapped,constant_curve(0),mode='boundary_hold')
        self.assertEqual(curve['runs'][0]['events'][0]['duration_seconds'],20.)
        self.assertIsNone(curve['duration_limits_seconds'])
        mapped['segments'][0]['end_seconds']=.02
        mapped['segments'][1].update(start_seconds=.02,end_seconds=4.)
        curve=build_transition(mapped,constant_curve(.99),mode='boundary_hold')
        self.assertGreater(curve['runs'][0]['events'][0]['duration_seconds'],0.)
        self.assertLess(curve['runs'][0]['events'][0]['duration_seconds'],.01)

    def test_zero_duration_is_finite_in_python_browser_and_svg_without_early_fade(self):
        mapped=mapped_rows()
        curve=build_transition(mapped,constant_curve(1),mode='boundary_hold')
        mapped['transition']=curve
        self.assertEqual(transition_color(curve,1.9999)['hex'],'#ff0000')
        self.assertEqual(transition_color(curve,2.)['hex'],'#0000ff')
        json.dumps(curve,allow_nan=False)
        transition_csv(mapped)
        root=ET.fromstring(color_svg(mapped))
        gradients=root.findall('.//{http://www.w3.org/2000/svg}linearGradient')
        self.assertEqual({s.get('stop-color') for s in gradients[0]}, {'#ff0000'})
        self.assertEqual({s.get('stop-color') for s in gradients[1]}, {'#0000ff'})
        if shutil.which('node'):
            source=Path(__file__).with_name('color_player.js').read_text(encoding='utf-8')
            script=source+'\nconst d='+json.dumps(curve)+';process.stdout.write(JSON.stringify([1.9999,2.,2.0001].map(t=>transitionColor(d,t))));'
            result=json.loads(subprocess.check_output(['node','-e',script],text=True))
            self.assertEqual([r['hex'] for r in result],['#ff0000','#0000ff','#0000ff'])

    def test_all_original_start_anchors_match_with_unequal_lengths_and_partial_audio(self):
        mapped=long_rows(100.123)
        original_starts=[100.123,100.323,102.823,103.223]
        original_ends=original_starts[1:]+[108.123]
        palette=[([255,0,0],'#ff0000'),([0,0,255],'#0000ff'),([0,255,0],'#00ff00'),([255,255,0],'#ffff00')]
        template=mapped['segments'][0]
        mapped['segments']=[{**template,'segment':i+1,'start_seconds':start,'end_seconds':end,'rgb':rgb,'hex':color}
                            for i,(start,end,(rgb,color)) in enumerate(zip(original_starts,original_ends,palette))]
        result={'offset_sample':100123,'sample_rate':1000,
                'diagnostic':{'time_s':[0.,.2,2.7,3.1,8.],'novelty':[0.,.8,.2,1.,0.]}}
        before=json.dumps(mapped)
        for mode in ['boundary_hold','foote_exponential']:
            for strength in [0.,4.,8.,10.]:
                for fraction in [.25,1.]:
                    curve=build_transition(mapped,result,strength,mode=mode,max_transition_fraction=fraction)
                    self.assertEqual(curve['runs'][0]['times'],original_starts)
                    self.assertEqual(curve['anchor'],'original_segment_start')
                    for row in mapped['segments']:
                        self.assertEqual(transition_color(curve,row['start_seconds'])['hex'],row['hex'])
                    if mode=='boundary_hold':
                        for i,event in enumerate(curve['runs'][0]['events']):
                            self.assertGreaterEqual(event['start_seconds'],original_starts[i])
                            self.assertEqual(event['end_seconds'],original_starts[i+1])
                            self.assertEqual(event['boundary_seconds'],event['end_seconds'])
                            if i:self.assertGreaterEqual(event['start_seconds'],curve['runs'][0]['events'][i-1]['end_seconds'])
        self.assertEqual(json.dumps(mapped),before)
        for mode in ['boundary_hold','foote_exponential']:
            exported={**mapped,'transition':build_transition(mapped,result,mode=mode)}
            csv_rows=list(csv.DictReader(io.StringIO(transition_csv(exported).decode('utf-8-sig'))))
            by_time={float(row['time_seconds']):row for row in csv_rows}
            for row in mapped['segments']:
                self.assertIn(row['start_seconds'],by_time)
                self.assertEqual(by_time[row['start_seconds']]['hex'],row['hex'])

    def test_csv_at_original_bar_starts_matches_unsmoothed_colors_in_both_modes(self):
        mapped=long_rows(12.)
        result={'offset_sample':12000,'sample_rate':1000,**constant_curve(.3)}
        for mode in ['boundary_hold','foote_exponential']:
            mapped['transition']=build_transition(mapped,result,mode=mode)
            csv_rows=list(csv.DictReader(io.StringIO(transition_csv(mapped).decode('utf-8-sig'))))
            by_time={float(row['time_seconds']):row for row in csv_rows}
            for row in mapped['segments']:
                self.assertEqual(by_time[row['start_seconds']]['hex'],row['hex'])
        gap=long_rows()
        gap['segments'][1].update(start_seconds=8.,end_seconds=12.)
        gap['transition']=build_transition(gap,{},mode='boundary_hold')
        data=list(csv.DictReader(io.StringIO(transition_csv(gap).decode('utf-8-sig'))))
        by_time={float(row['time_seconds']):row for row in data}
        self.assertEqual(by_time[4.]['status'],'unmapped')
        self.assertEqual(by_time[8.]['hex'],'#0000ff')
        self.assertEqual(by_time[12.]['hex'],'#0000ff')

    @unittest.skipUnless(shutil.which('node'),'Node is optional for audio event simulation')
    def test_audio_seek_and_play_events_show_same_boundary_colors_with_smoothing_on_or_off(self):
        mapped=long_rows(12.)
        source=Path(__file__).with_name('color_player.js').read_text(encoding='utf-8')
        for mode in ['boundary_hold','foote_exponential']:
            curve=build_transition(mapped,{'offset_sample':12000,'sample_rate':1000,**constant_curve(.3)},mode=mode)
            script='const source='+json.dumps(source)+';const data='+json.dumps({'segments':mapped['segments'],'transition':curve})+r''';
const vm=require('vm'),assert=require('assert');
for(const enabled of [true,false]){
  data.transition.enabled=enabled;
  const elements={};
  for(const id of ['colorData','music','colorBlock','segmentLabel','timeLabel','details','seekbar','playhead','warning','colorTrack','jumpStart']){
    elements[id]={style:{},events:{},textContent:'',addEventListener(n,f){this.events[n]=f;},getBoundingClientRect(){return {left:0,width:100};}};
  }
  elements.colorData.textContent=JSON.stringify(data);
  Object.assign(elements.music,{currentTime:0,duration:100,paused:true,ended:false});
  Object.assign(elements.colorTrack,{width:1800,height:20,getContext:()=>({fillRect(){}})});
  const context={document:{getElementById:id=>elements[id]},requestAnimationFrame:()=>1,cancelAnimationFrame(){}};
  vm.createContext(context);vm.runInContext(source,context);
  elements.music.events.loadedmetadata();assert.strictEqual(elements.music.currentTime,12);
  for(const row of data.segments){
    elements.music.currentTime=row.start_seconds;
    for(const event of ['seeked','play','pause','timeupdate']){
      elements.music.events[event]();
      assert.strictEqual(elements.colorBlock.style.background,row.hex);
    }
  }
}'''
            subprocess.run(['node','-e',script],check=True,capture_output=True,text=True)

    @unittest.skipUnless(shutil.which('node'),'Node is optional for Python/browser parity')
    def test_browser_matches_boundary_curve_and_can_seek_in_arbitrary_order(self):
        curve=build_transition(long_rows(30),{**constant_curve(.8),'offset_sample':300,'sample_rate':10},mode='boundary_hold')
        times=[30.,33.9,34.,34.01,34.07,34.15,34.4,36.,38.,34.01,33.9]
        source=Path(__file__).with_name('color_player.js').read_text(encoding='utf-8')
        script=source+'\nconst d='+json.dumps({'curve':curve,'times':times})+';process.stdout.write(JSON.stringify(d.times.map(t=>transitionColor(d.curve,t))));'
        actual=json.loads(subprocess.check_output(['node','-e',script],text=True))
        self.assertEqual([r['rgb'] for r in actual],[transition_color(curve,t)['rgb'] for t in times])


if __name__=='__main__':unittest.main()
