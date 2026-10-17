import csv
import io
import json
import math
from pathlib import Path
import shutil
import subprocess
import unittest

from color_transition import build_transition, integral, transition_color, transition_csv
from hsv_timeline import color_svg


def mapped_rows(offset=0):
    return {'segments': [
        {'segment': i+1, 'start_seconds': offset+i*2., 'end_seconds': offset+(i+1)*2.,
         'status': 'ok', 'hex': color, 'rgb': rgb, 'hue_degrees': h, 'saturation': 1.,
         'value': 1., 'valence': 9., 'arousal': 9.}
        for i, (color, rgb, h) in enumerate([('#ff0000', [255, 0, 0], 0.), ('#0000ff', [0, 0, 255], 240.)])]}


class TransitionChecks(unittest.TestCase):
    def test_exact_anchors_constant_edges_and_continuity(self):
        curve = build_transition(mapped_rows(), {'diagnostic': {'time_s': [0., 2., 4.], 'novelty': [0., 1., 0.]}})
        self.assertEqual(transition_color(curve, 0.)['hex'], '#ff0000')
        for t in [2., 3., 4.]: self.assertEqual(transition_color(curve, t)['hex'], '#0000ff')
        self.assertEqual(transition_color(curve, 2.-1e-8), transition_color(curve, 2.+1e-8))
        self.assertNotEqual(transition_color(curve, 2.)['hex'], '#ff0000')
        self.assertIsNone(transition_color(curve, -.01))

    def test_exponential_speed_and_exact_piecewise_linear_integral(self):
        curve = build_transition(mapped_rows(), {'diagnostic': {'time_s': [0., 1., 2., 3., 4.], 'novelty': [0., 0., 1., 0., 0.]}},strength=4.)
        self.assertAlmostEqual(curve['rate'][curve['grid'].index(2.)], .2+math.expm1(4))
        # The interval 1..2 has a linear speed, integrated exactly at its midpoint.
        a = .2; b = .2+math.expm1(4)
        self.assertAlmostEqual(integral(curve, 1.5)-integral(curve, 1.), a*.5+.5*(b-a)*.5**2)
        self.assertGreater(integral(curve, 2.01)-integral(curve, 1.99),
                           integral(curve, 1.02)-integral(curve, 1.))
        samples = [transition_color(curve, t) for t in [1.1, 1.7, 2.9]]
        for t in [1.13, 1.14, 1.15]: transition_color(curve, t)
        self.assertEqual(samples, [transition_color(curve, t) for t in [1.1, 1.7, 2.9]])

    def test_absolute_time_offset_and_zero_strength_fallback(self):
        a = build_transition(mapped_rows(), {'diagnostic': {'time_s': [0., 2., 4.], 'novelty': [0., 1., 0.]}})
        b = build_transition(mapped_rows(90), {'offset_sample': 9000, 'sample_rate': 100,
                            'diagnostic': {'time_s': [0., 2., 4.], 'novelty': [0., 1., 0.]}})
        for t in [1.25, 1.8, 2.75]: self.assertEqual(transition_color(a, t), transition_color(b, t+90))
        no_curve = build_transition(mapped_rows(), {})
        zero_strength = build_transition(mapped_rows(), {'diagnostic': {'time_s': [0., 2., 4.], 'novelty': [0., 1., 0.]}}, strength=0)
        self.assertEqual(no_curve['mode'], 'smooth_fallback_no_foote')
        for t in [1.2, 2., 2.7]: self.assertEqual(transition_color(no_curve, t), transition_color(zero_strength, t))

    def test_failures_and_gaps_do_not_bridge(self):
        mapped = mapped_rows()
        bad = {**mapped['segments'][0], 'segment': 3, 'start_seconds': 4., 'end_seconds': 6., 'hex': None, 'rgb': None, 'status': 'prediction_error'}
        last = {**mapped['segments'][0], 'segment': 4, 'start_seconds': 6., 'end_seconds': 8.}
        mapped['segments'] += [bad, last]
        curve = build_transition(mapped, {})
        self.assertIsNone(transition_color(curve, 5.))
        self.assertEqual(transition_color(curve, 3.8)['hex'], '#0000ff')
        self.assertEqual(transition_color(curve, 6.)['hex'], '#ff0000')
        mapped['segments'].pop(2)
        self.assertIsNone(transition_color(build_transition(mapped, {}), 5.))

    def test_svg_csv_and_json_share_curve(self):
        mapped = mapped_rows()
        mapped['transition'] = build_transition(mapped, {})
        rows = list(csv.DictReader(io.StringIO(transition_csv(mapped).decode('utf-8-sig'))))
        self.assertEqual(len(rows), 121)
        self.assertEqual(rows[60]['hex'], transition_color(mapped['transition'], 2.)['hex'])
        self.assertIn('linearGradient', color_svg(mapped))
        self.assertIn('anchor_area', json.dumps(mapped, allow_nan=False))
        mapped['transition']['enabled'] = False
        rows = list(csv.DictReader(io.StringIO(transition_csv(mapped).decode('utf-8-sig'))))
        self.assertEqual(rows[60]['hex'], '#0000ff')
        self.assertNotIn('linearGradient', color_svg(mapped))

    @unittest.skipUnless(shutil.which('node'), 'Node is optional for the browser/Python parity check')
    def test_browser_matches_python_at_arbitrary_times(self):
        curve = build_transition(mapped_rows(11), {'offset_sample': 110, 'sample_rate': 10,
                    'diagnostic': {'time_s': [0., .5, 1., 1.5, 2., 3., 4.], 'novelty': [0., .1, 0., .8, .7, 0., 0.]}})
        times = [11., 12., 12.13, 12.6, 13., 13.43, 14., 15.]
        source = Path(__file__).with_name('color_player.js').read_text(encoding='utf-8')
        script = source+'\nconst d='+json.dumps({'curve': curve, 'times': times})+';\nprocess.stdout.write(JSON.stringify(d.times.map(t=>transitionColor(d.curve,t))));'
        result = json.loads(subprocess.check_output(['node', '-e', script], text=True))
        for t, actual in zip(times, result):
            self.assertEqual(actual['rgb'], transition_color(curve, t)['rgb'])
            self.assertEqual(actual['hex'], transition_color(curve, t)['hex'])


if __name__ == '__main__': unittest.main()
