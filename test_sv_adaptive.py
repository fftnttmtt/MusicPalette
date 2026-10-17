import csv
import io
import json
import math
import unittest
from hsv_timeline import map_colors, normalized_sigmoid, color_csv
from sv_adaptive import adaptive_statistics, adapt_value


def row(i,a,v=None,duration=1.,status='ok'):
    return {'segment':i,'start_seconds':float(i),'end_seconds':i+duration,
            'status':status,'arousal':a,'valence':a if v is None else v,
            'scores':[{'tag':'happy','score':.8}]}


class AdaptiveChecks(unittest.TestCase):
    def test_duration_weighted_quantiles_ignore_short_outlier_and_failure(self):
        result={'segments':[row(0,5.,duration=100.),row(100,9.,duration=1.),row(101,1.,duration=500.,status='error')]}
        stat=adaptive_statistics(result)['arousal']
        self.assertEqual(stat['low'],.5)
        self.assertEqual(stat['high'],.5)
        self.assertEqual(stat['segment_count'],2)
        self.assertEqual(stat['valid_duration_seconds'],101.)
        self.assertFalse(stat['constant'])
        self.assertEqual(stat['effective_span'],.2)
        split={'segments':[row(i,5.) for i in range(100)]+[row(100,9.)]}
        other=adaptive_statistics(split)['arousal']
        for key in ['low','high','lower_bound','upper_bound','effective_span']:
            self.assertEqual(stat[key],other[key])

    def test_narrow_score_range_has_larger_sv_changes(self):
        result={'segments':[row(i,x) for i,x in enumerate([4.6,4.8,5.,5.2,5.4])]}
        options=dict(saturation_k=5.,value_k=5.,hue_gamma=1.5,use_song_mean_centers=True)
        baseline=map_colors(result,**options)
        enhanced=map_colors(result,**options,enhance_sv=True,value_range=(.25,1.))
        for dimension in ['saturation','value']:
            before=baseline['segments'][-1][dimension]-baseline['segments'][0][dimension]
            after=enhanced['segments'][-1][dimension]-enhanced['segments'][0][dimension]
            self.assertGreater(after,2*before)
        self.assertEqual([r['hue_degrees'] for r in baseline['segments']],
                         [r['hue_degrees'] for r in enhanced['segments']])

    def test_constant_and_tiny_variation_do_not_expand_to_full_range(self):
        constant=adaptive_statistics({'segments':[row(0,7.),row(1,7.)]})['arousal']
        self.assertTrue(constant['constant'])
        self.assertEqual(adapt_value(.75,constant,1.),.75)
        tiny=adaptive_statistics({'segments':[row(0,5.),row(1,5.0008)]},max_gain=5.)['arousal']
        difference=adapt_value(.5001,tiny,1.)-adapt_value(.5,tiny,1.)
        self.assertLessEqual(difference,.00050000001)

    def test_zero_strength_and_disabled_are_identity_before_output_range(self):
        result={'segments':[row(0,3.),row(1,7.)]}
        base=map_colors(result)
        zero=map_colors(result,enhance_sv=True,enhancement_strength=0.)
        disabled=map_colors(result,enhance_sv=False,enhancement_strength=1.)
        self.assertEqual(base['segments'],zero['segments'])
        self.assertEqual(base['segments'],disabled['segments'])

    def test_center_tracks_transformed_mean_and_gamma_precedes_range(self):
        result={'segments':[row(0,3.,v=4.),row(1,7.,v=6.,duration=3.)]}
        out=map_colors(result,enhance_sv=True,use_song_mean_centers=True,
                       saturation_gamma=2.,value_gamma=1.,value_range=(.25,1.))
        params=out['mapping']['parameters'];adaptation=out['mapping']['sv_adaptation']
        self.assertAlmostEqual(params['saturation_center'],adapt_value(.625,adaptation['statistics']['arousal'],.5))
        for r in out['segments']:
            self.assertAlmostEqual(r['saturation_after_gamma'],r['saturation_before_gamma']**2)
            self.assertAlmostEqual(r['value'],.25+.75*r['value_before_gamma'])
        manual=map_colors(result,enhance_sv=True,saturation_center=.33,value_center=.72)
        self.assertEqual(manual['mapping']['parameters']['saturation_center'],.33)
        self.assertEqual(manual['mapping']['parameters']['value_center'],.72)

    def test_exports_and_missing_data_remain_finite(self):
        result={'segments':[row(0,3.),row(1,7.),row(2,math.nan)]}
        out=map_colors(result,enhance_sv=True)
        json.dumps(out,allow_nan=False)
        data=list(csv.DictReader(io.StringIO(color_csv(out).decode('utf-8-sig'))))
        self.assertIn('adapted_arousal',data[0])
        self.assertAlmostEqual(float(data[0]['adapted_arousal']),out['segments'][0]['adapted_arousal'])
        empty=map_colors({'segments':[]},enhance_sv=True)
        self.assertFalse(empty['mapping']['sv_adaptation']['statistics']['arousal']['available'])
        for args in [dict(max_gain=0.),dict(percentile_low=90.,percentile_high=10.),
                     dict(enhancement_strength=2.),dict(value_range=(.8,.2))]:
            with self.assertRaises(ValueError):map_colors(result,**args)


if __name__=='__main__':unittest.main()
