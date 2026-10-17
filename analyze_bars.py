"""CLI: preview or predict the adaptive bar contexts."""
import argparse
import json
from pathlib import Path
from beat_core import load_audio,get_tracker
from segmented import select_audio
from bar_context import make_bar_plan,predict_bar_plan,csv_bytes

def main():
    p=argparse.ArgumentParser()
    p.add_argument('audio');p.add_argument('--a',type=float,default=6.,help='Maximum target segment length and target prediction duration in seconds, >=3; overlong bars are split')
    p.add_argument('--boundary',type=float,default=.7)
    p.add_argument('--foote-context',type=float,default=2.)
    p.add_argument('--start',type=float,default=0.)
    p.add_argument('--seconds',type=float,default=0.)
    p.add_argument('--model',choices=['small0','final0'],default='final0')
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--preview',action='store_true')
    p.add_argument('--no-raw',action='store_true')
    p.add_argument('--per-beat',action='store_true',help='Predict each detected beat interval instead of bars (slower)')
    p.add_argument('--output',default='bar_emotions.json')
    args=p.parse_args()
    audio,sr=load_audio(args.audio);full=audio.mean(axis=1)
    y,offset=select_audio(full,sr,args.start,args.seconds)
    tracker,device=get_tracker(args.model,args.device)
    plan=make_bar_plan(y,sr,tracker,target_seconds=args.a,boundary_threshold=args.boundary,
            context_seconds=args.foote_context,offset_sample=offset,model=args.model,device=device,
            prediction_unit='beat' if args.per_beat else 'bar')
    if args.preview:result=plan
    else:
        from backend import Tester
        result=predict_bar_plan(y,sr,plan,Tester(device),compare_raw=not args.no_raw,
                               whole_audio=full,
                               progress=lambda done,total,row:print(f'{done}/{total}'))
        result['audio_name']=Path(args.audio).name
        Path(args.output).with_suffix('.csv').write_bytes(csv_bytes(result))
    Path(args.output).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(args.output)
if __name__=='__main__':main()
