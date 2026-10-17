import json
import math
import unittest
from xml.etree import ElementTree as ET
from hsv_timeline import TAG_HUES,segment_color,map_colors,color_svg,color_csv,normalized_sigmoid,gamma_transform,song_mean_centers


def row(scores=None,v=9,a=9):
    return {'segment':1,'start_seconds':0.,'end_seconds':2.,'status':'ok','valence':v,'arousal':a,
            'scores':[{'tag':t,'score':(scores or {}).get(t,0.)} for t in TAG_HUES]}

class HSVChecks(unittest.TestCase):
    def test_each_single_tag_matches_specified_hue(self):
        for tag,hue in TAG_HUES.items():
            out=segment_color(row({tag:1}))
            self.assertEqual(out['status'],'ok')
            self.assertAlmostEqual(out['hue_degrees'],hue)
    def test_circular_mean_wraps_at_red(self):
        out=segment_color(row({'love':1,'epic':1}))
        self.assertAlmostEqual(out['hue_degrees'],5.)
        out=segment_color(row({'love':1,'energetic':1}))
        self.assertAlmostEqual(out['hue_degrees'],350.)
    def test_other_tags_cannot_change_color(self):
        source=row({'happy':.3},v=5,a=5)
        before=segment_color(source)
        source['scores'] += [{'tag':'dark','score':1.},{'tag':'fast','score':1.}]
        self.assertEqual(segment_color(source),before)
        self.assertEqual(before['hex'],'#808040')
    def test_sigmoid_va_and_clipping(self):
        out=segment_color(row({'happy':1},v=5,a=3))
        self.assertAlmostEqual(out['value'],.5);self.assertAlmostEqual(out['saturation'],normalized_sigmoid(.25))
        clipped=segment_color(row({'happy':1},v=10,a=-1))
        self.assertEqual(clipped['value'],1.);self.assertEqual(clipped['saturation'],0.)
        self.assertTrue(clipped['va_clipped'])
    def test_achromatic_and_black_are_real_colors(self):
        self.assertEqual(segment_color(row({'happy':1},v=9,a=1))['hex'],'#ffffff')
        self.assertEqual(segment_color(row({'happy':1},v=1,a=9))['hex'],'#000000')
    def test_zero_or_cancelled_vectors_are_undefined(self):
        for source in [row({'sad':1,'happy':1})]:
            out=segment_color(source)
            self.assertEqual(out['status'],'undefined_hue');self.assertIsNone(out['hex'])
    def test_missing_tag_or_prediction_failure(self):
        source=row({'happy':1});source['scores']=[s for s in source['scores'] if s['tag']!='nature']
        self.assertEqual(segment_color(source)['status'],'ok')
        self.assertEqual(segment_color(source)['invalid_or_missing_tags'],['nature'])
        source['status']='error'
        self.assertEqual(segment_color(source)['status'],'prediction_error')
    def test_svg_geometry_and_exports(self):
        first=row({'happy':1});second={**row({'sad':1}),'segment':2,'start_seconds':2.,'end_seconds':6.}
        mapped=map_colors({'segments':[first,second]})
        svg=color_svg(mapped);root=ET.fromstring(svg)
        blocks=[r for r in root.findall('{http://www.w3.org/2000/svg}rect') if r.get('height')=='80']
        self.assertEqual(len(blocks),2)
        self.assertAlmostEqual(float(blocks[1].get('width'))/float(blocks[0].get('width')),2,places=5)
        self.assertIn('hue_degrees',color_csv(mapped).decode('utf-8-sig'))
        json.dumps(mapped,allow_nan=False)
    def test_invalid_va_remains_exportable(self):
        mapped=map_colors({'segments':[row({'happy':1},v=math.nan)]})
        self.assertEqual(mapped['segments'][0]['status'],'invalid_valence_arousal')
        json.dumps(mapped,allow_nan=False)
    def test_low_scores_participate_without_a_cutoff(self):
        out=segment_color(row({'happy':.3,'epic':.299999}))
        self.assertTrue(40<out['hue_degrees']<60)
        self.assertEqual(out['hue_weights'],{'epic':.299999**2,'happy':.3**2})
        self.assertEqual(out['hue_tag_count'],2)
        single=segment_color(row({'happy':.000001}))
        self.assertEqual(single['status'],'ok')
        self.assertIsNone(single['hue_fallback_tag'])
        self.assertAlmostEqual(single['hue_degrees'],60.)
        mixed=segment_color(row({'energetic':.29,'happy':.2}))
        self.assertIsNone(mixed['hue_fallback_tag'])
        self.assertTrue(10<mixed['hue_degrees']<60)
        self.assertEqual(segment_color(row())['hue_fallback_tag'],'calm')
    def test_sigmoid_endpoints_center_and_sensitivity(self):
        for k in [0,1e-9,1,10,40]:
            for center in [0,.2,.5,.8,1]:
                self.assertAlmostEqual(normalized_sigmoid(0,k,center),0)
                self.assertAlmostEqual(normalized_sigmoid(1,k,center),1)
        self.assertAlmostEqual(normalized_sigmoid(.5,10,.5),.5)
        self.assertGreater(normalized_sigmoid(.51,10)-normalized_sigmoid(.49,10),.02)
        self.assertGreater(normalized_sigmoid(.51,20)-normalized_sigmoid(.49,20),
                           normalized_sigmoid(.51,5)-normalized_sigmoid(.49,5))
    def test_independent_controls_and_json_parameters(self):
        source=row({'happy':1},v=4,a=4)
        a=segment_color(source,saturation_k=0,value_k=20)
        self.assertEqual(a['saturation'],.375)
        self.assertLess(a['value'],.375)
        mapped=map_colors({'segments':[source]},value_center=.7)
        self.assertEqual(mapped['mapping']['parameters']['value_center'],.7)
        self.assertFalse(mapped['mapping']['score_threshold_applied'])
        self.assertIsNone(mapped['mapping']['hue_threshold'])
    def test_linear_limit_and_invalid_parameters(self):
        self.assertEqual(normalized_sigmoid(.37,0,.5),.37)
        for k,center in [(-1,.5),(10,2),(math.nan,.5)]:
            with self.assertRaises(ValueError):normalized_sigmoid(.5,k,center)
    def test_gamma_identity_and_endpoints(self):
        for g in [.1,1,2,5]:
            self.assertEqual(gamma_transform(0,g),0)
            self.assertEqual(gamma_transform(1,g),1)
        self.assertEqual(gamma_transform(.3,1),.3)
        for g in [0,-1,math.nan]:
            with self.assertRaises(ValueError):gamma_transform(.3,g)
    def test_sv_gamma_after_sigmoid(self):
        source=row({'happy':1},v=4,a=3)
        base=segment_color(source)
        changed=segment_color(source,saturation_gamma=2,value_gamma=.5)
        self.assertAlmostEqual(changed['saturation'],base['saturation']**2)
        self.assertAlmostEqual(changed['value'],base['value']**.5)
        self.assertEqual(changed['hue_degrees'],base['hue_degrees'])
    def test_h_gamma_changes_weights_not_angles(self):
        source=row({'love':.4,'epic':.8})
        before=segment_color(source,hue_gamma=1.)
        after=segment_color(source,hue_gamma=2)
        self.assertAlmostEqual(after['hue_weights']['love'],.16)
        self.assertAlmostEqual(after['hue_weights']['epic'],.64)
        self.assertLess(abs(after['hue_degrees']-40),abs(before['hue_degrees']-40))
        single=segment_color(row({'calm':.6}),hue_gamma=2)
        self.assertAlmostEqual(single['hue_degrees'],180)
    def test_low_scores_keep_gamma_weights(self):
        source=row({'happy':.3,'energetic':.001})
        for g in [.1,2,5]:
            out=segment_color(source,hue_gamma=g)
            self.assertEqual(set(out['hue_weights']),{'happy','energetic'})
            self.assertAlmostEqual(out['hue_weights']['happy'],.3**g)
            self.assertAlmostEqual(out['hue_weights']['energetic'],.001**g)
    def test_gamma_parameters_in_export(self):
        mapped=map_colors({'segments':[row({'happy':1})]},hue_gamma=2,saturation_gamma=3,value_gamma=4)
        self.assertEqual(mapped['mapping']['parameters']['hue_gamma'],2)
        self.assertEqual(mapped['mapping']['parameters']['saturation_gamma'],3)
        self.assertEqual(mapped['mapping']['parameters']['value_gamma'],4)
        self.assertIn('value_before_gamma',color_csv(mapped).decode('utf-8-sig'))
    def test_song_means_weight_original_duration(self):
        first={**row({'happy':1},v=3,a=9),'start_seconds':0.,'end_seconds':1.,'prediction_duration_seconds':100}
        second={**row({'happy':1},v=7,a=1),'start_seconds':1.,'end_seconds':4.,'prediction_duration_seconds':1}
        result={'segments':[first,second]}
        means=song_mean_centers(result)
        self.assertEqual(means['valence_mean'],6.)
        self.assertEqual(means['arousal_mean'],3.)
        self.assertEqual(means['value_center'],.625)
        self.assertEqual(means['saturation_center'],.25)
        mapped=map_colors(result,use_song_mean_centers=True)
        self.assertEqual(mapped['mapping']['parameters']['saturation_center'],.25)
        self.assertEqual(mapped['mapping']['parameters']['value_center'],.625)
        self.assertEqual(mapped['mapping']['sigmoid_center_mode'],'song_duration_weighted_mean')
    def test_song_means_exclude_failure_but_include_valid_va_without_hue(self):
        good=row(v=5,a=7)
        error={**row({'happy':1},v=9,a=1),'status':'error'}
        invalid=row(v=math.nan,a=9)
        result={'segments':[good,error,invalid]}
        means=song_mean_centers(result)
        self.assertEqual(means['segment_count'],1)
        self.assertEqual(means['valence_mean'],5.)
        self.assertEqual(means['arousal_mean'],7.)
        self.assertEqual(means['saturation_center'],.75)
    def test_song_mean_off_keeps_manual_parameters(self):
        result={'segments':[row({'happy':1},v=9,a=1)]}
        mapped=map_colors(result,saturation_center=.4,value_center=.6,use_song_mean_centers=False)
        self.assertEqual(mapped['mapping']['parameters']['saturation_center'],.4)
        self.assertEqual(mapped['mapping']['parameters']['value_center'],.6)
        no_valid={'segments':[{**row(),'status':'error'}]}
        fallback=map_colors(no_valid,use_song_mean_centers=True)
        self.assertFalse(fallback['mapping']['song_means']['available'])
        self.assertEqual(fallback['mapping']['sigmoid_center_mode'],'manual_fallback_no_valid_predictions')
        json.dumps(fallback,allow_nan=False)
if __name__=='__main__':unittest.main()
