import json
import unittest
from tag_scores import correct_prediction, correct_result
from hsv_timeline import segment_color, TAG_HUES
from tag_charts import tag_timeline_rows, load_timeline_json, TIMELINE_TAGS


def target(scores):
    return {'segment':1,'start_seconds':0.,'end_seconds':2.,'status':'ok','valence':5.,'arousal':5.,
            'scores':[{'tag':t,'score':p,'中文':t} for t,p in scores.items()]}


class DisabledTagChecks(unittest.TestCase):
    def test_funny_removed_and_other_scores_unchanged(self):
        original=target({'funny':1.,'happy':.6,'sad':.2})
        corrected=correct_prediction(original)
        self.assertEqual({r['tag']:r['score'] for r in corrected['scores']},{'happy':.6,'sad':.2})
        self.assertEqual(corrected['predicted_tags'],['happy'])
        self.assertEqual(corrected['disabled_tags'],['funny'])
        self.assertEqual(correct_prediction(corrected),corrected)
        self.assertEqual(original['scores'][0]['tag'],'funny')

    def test_funny_cannot_change_weighted_or_fallback_hue(self):
        for scores in [{'sad':.28,'happy':.1},{'sad':.6,'happy':.8}]:
            expected=segment_color(target(scores),hue_gamma=5)
            for value in [0.,.2,.9,1.]:
                self.assertEqual(segment_color(target({**scores,'funny':value}),hue_gamma=5),expected)
        self.assertNotIn('funny',TAG_HUES)
        self.assertEqual(segment_color(target({'funny':1.}))['status'],'missing_or_invalid_tag_scores')

    def test_import_old_corrected_funny_is_disabled(self):
        original=target({'funny':.9,'happy':.2})
        original['scores'][0].update(raw_score=1.,score_adjustment=-.1)
        original['score_adjustments']={'funny':{'offset':-.1}}
        original['raw_prediction']=target({'funny':.95,'sad':.3})
        result=load_timeline_json(json.dumps({'segments':[original]}))
        row=result['segments'][0]
        self.assertEqual([s['tag'] for s in row['scores']],['happy'])
        self.assertEqual([s['tag'] for s in row['raw_prediction']['scores']],['sad'])
        self.assertNotIn('funny',row['score_adjustments'])
        self.assertEqual(correct_result(result),result)
        self.assertEqual(load_timeline_json(json.dumps(result)),result)

    def test_nine_tag_heatmap(self):
        rows=tag_timeline_rows({'segments':[target({'funny':1.,'happy':.2,'energetic':.7})]})
        self.assertEqual(len(rows),9)
        self.assertEqual(TIMELINE_TAGS,list(TAG_HUES))
        self.assertEqual(next(r['score'] for r in rows if r['tag']=='energetic'),.7)
        self.assertNotIn('funny',TIMELINE_TAGS)
        self.assertNotIn('funny',[r['tag'] for r in rows])


if __name__=='__main__':unittest.main()
