"""Bar or beat targets with Foote-guided context; hard minimum of 3 seconds."""
import csv
import io
import math
import tempfile
import time
import traceback
from pathlib import Path
import numpy as np
from acoustic_features import extract_features, foote_novelty
from beat_core import build_analysis, get_tracker

MINIMUM_SECONDS = 3.0
LEADING_INHERIT_SECONDS = 1.0


def build_beat_targets(analysis):
    """Use detected beat events as exact sample boundaries, preserving edges."""
    sr,n=analysis['sample_rate'],analysis['total_samples']
    beats=np.asarray(analysis['beat_samples'],dtype=int)
    edges=np.unique(np.r_[0,beats,n]).astype(int)
    detected=set(map(int,beats))
    numbers={int(sample):i+1 for i,sample in enumerate(beats[beats<n])}
    bars=analysis['segments'];starts=[s['start_sample'] for s in bars]
    rows=[]
    for index,(left,right) in enumerate(zip(edges[:-1],edges[1:]),1):
        left,right=int(left),int(right)
        parent=bars[max(0,int(np.searchsorted(starts,left,side='right'))-1)]
        number=numbers.get(left)
        if not len(beats):kind,label='unmetered','未检测到节拍边界'
        elif number is None:kind,label='beat_leading','开头剩余片段'
        elif right in detected:kind,label='beat',f'节拍 {number}'
        else:kind,label='beat_tail',f'节拍 {number}（结尾剩余片段）'
        in_bar=(int(np.searchsorted(beats,left,side='left')-
                    np.searchsorted(beats,parent['start_sample'],side='left'))+1
                if number is not None and parent['kind']=='bar' else None)
        rows.append({'index':index,'kind':kind,'label':label,'bar_number':parent['bar_number'],
            'beat_number':number,'beat_in_bar':in_bar,'start_sample':left,'end_sample':right,
            'start':left/sr,'end':right/sr,'duration':(right-left)/sr,
            'beat_count':int(number is not None),'bar_source_segment_index':parent['index'],
            'bar_source_kind':parent['kind'],'bar_start_sample':parent['start_sample'],
            'bar_end_sample':parent['end_sample']})
    return rows


def split_long_targets(segments, sr, target_seconds, beats):
    """Keep source boundaries; subdivide long targets near balanced beat positions.

    A feasible range at each cut guarantees positive parts no longer than a.
    If no detected beat fits, use the nearest balanced sample without padding.
    """
    if not math.isfinite(sr) or sr<1:
        raise ValueError('采样率必须至少为1。')
    if not math.isfinite(target_seconds) or target_seconds<MINIMUM_SECONDS:
        raise ValueError('a不能小于模型最短输入3秒。')
    maximum=math.floor(target_seconds*sr)
    beat_samples=np.unique(np.asarray(beats,dtype=int))
    rows=[]
    for source in segments:
        start,end=int(source['start_sample']),int(source['end_sample'])
        count=(end-start+maximum-1)//maximum
        edges=[start];reasons=['source_boundary']
        for part in range(1,count):
            remaining=count-part
            low=max(edges[-1]+1,end-remaining*maximum)
            high=min(edges[-1]+maximum,end-remaining)
            ideal=start+round((end-start)*part/count)
            first=np.searchsorted(beat_samples,low,side='left')
            last=np.searchsorted(beat_samples,high,side='right')
            candidates=beat_samples[first:last]
            if len(candidates):
                cut=int(candidates[np.argmin(np.abs(candidates-ideal))])
                reason='detected_beat'
            else:
                cut=max(low,min(high,ideal));reason='duration_limit_sample'
            edges.append(cut);reasons.append(reason)
        edges.append(end)
        for part,(a,b) in enumerate(zip(edges,edges[1:]),start=1):
            row={**source,'index':len(rows)+1,'start_sample':a,'end_sample':b,
                 'start':a/sr,'end':b/sr,'duration':(b-a)/sr,
                 'beat_count':int(np.searchsorted(beat_samples,b,side='left')-np.searchsorted(beat_samples,a,side='left')),
                 'source_segment_index':source['index'],'source_kind':source['kind'],
                 'source_start_sample':start,'source_end_sample':end,
                 'source_start_seconds':start/sr,'source_end_seconds':end/sr,
                 'forced_split':count>1,'split_part':part,'split_count':count,
                 'split_start_reason':reasons[part-1]}
            if count>1:
                row['kind']=source['kind']+'_part'
                row['label']=f"{source['label']} · 子段 {part}/{count}（超长强制切分）"
            rows.append(row)
    return rows


def inherit_short_leading(rows, sr):
    """Retain the opening interval, but reuse the first bar's actual input.

    A single detected downbeat does not establish a complete first bar. Long
    first bars use their first split target, which starts at that downbeat.
    """
    if len(rows)<2:
        return
    leading,first=rows[:2]
    if (leading.get('source_kind',leading.get('kind'))!='leading'
            or first.get('source_kind',first.get('kind'))!='bar'
            or leading['end_sample']!=first['start_sample']
            or not 0<leading['end_sample']-leading['start_sample']<LEADING_INHERIT_SECONDS*sr):
        return
    _inherit_input(leading,first,'first_bar','不足1秒，采用第一小节')


def _inherit_input(leading,first,source,description):
    # These fields describe the donor input, not an independently extended
    # opening window. Context lengths relative to the ignored target are N/A.
    for key in ['prediction_start_sample','prediction_end_sample','prediction_start_seconds',
                'prediction_end_seconds','prediction_duration_seconds','requested_target_seconds',
                'target_duration_seconds','prediction_beat_count','detected_prediction_beat_count',
                'beat_count_source','reached_target']:
        leading[key]=first[key]
    leading.update(prediction_source=source,inherited_from_segment=first['segment'],
                   independent_prediction_skipped=True,stop_reason='inherited_'+source,
                   left_context_seconds=None,right_context_seconds=None,added_duration_seconds=None,
                   added_beat_count=None,forced_boundary_crossings=[],forced_expansion=False,
                   minimum_extension_required=False,
                   label=leading['label']+'（'+description+'）')


def inherit_short_beat_leading(rows,sr,analysis,offset_sample):
    """Keep the existing <1s pre-bar rule, using that bar's first beat result."""
    sources=analysis['segments']
    if (len(sources)<2 or sources[0]['kind']!='leading' or sources[1]['kind']!='bar'
            or not 0<sources[0]['end_sample']-sources[0]['start_sample']<LEADING_INHERIT_SECONDS*sr):
        return
    first_start=sources[1]['start_sample']+offset_sample
    first=next((r for r in rows if r['start_sample']==first_start),None)
    if first is None:return  # Never move a beat boundary to manufacture a donor.
    for row in rows:
        if row['end_sample']>first_start:break
        _inherit_input(row,first,'first_bar_first_beat','开头不足1秒，采用第一小节首拍')


def extend_interval(left, right, n, sr, beats, times, novelty, target_seconds=6., boundary_threshold=.7):
    if not math.isfinite(target_seconds) or target_seconds < MINIMUM_SECONDS:
        raise ValueError('预测目标时长a不能小于3秒。')
    if not math.isfinite(sr) or sr<=0:
        raise ValueError('采样率必须为正数。')
    if not 0 < boundary_threshold <= 1:
        raise ValueError('边界阈值须在 (0,1] 内。')
    if n < math.ceil(MINIMUM_SECONDS * sr):
        raise ValueError('所选音频不足3秒，无法通过扩展满足模型最短输入。')
    if not 0 <= left < right <= n:
        raise ValueError('目标边界无效。')
    maximum=math.floor(target_seconds*sr)
    if right-left>maximum:
        raise ValueError('目标段超过a秒，请先强制切分超长小节。')
    original_left, original_right = left, right
    times=np.asarray(times,dtype=float);novelty=np.asarray(novelty,dtype=float)
    if times.ndim!=1 or not len(times) or novelty.shape!=times.shape or not np.isfinite(times).all() or not np.isfinite(novelty).all() or np.any(np.diff(times)<=0):
        raise ValueError('Foote曲线须为有限、时间严格递增的等长序列。')
    # Beats still define the original bars; context extension is now measured in
    # seconds on the acoustic feature grid, with an exact final sample endpoint.
    detected_beats = np.unique(np.asarray(beats, dtype=int))
    detected_beats = detected_beats[(detected_beats >= 0) & (detected_beats < n)]
    grid=np.unique(np.r_[0,np.clip(np.rint(times*sr),0,n).astype(int),n]).astype(int)
    def beat_count(a, b):
        # Count onset events inside [a,b), excluding the next interval's first beat.
        return int(np.searchsorted(detected_beats, b, side='left') -
                   np.searchsorted(detected_beats, a, side='left'))
    original_beat_count = beat_count(original_left, original_right)
    hard = math.ceil(MINIMUM_SECONDS * sr)
    goal=max(hard,maximum)
    blocked = [False, False]
    crossed = []
    def cost(a, b):
        # Include both ends so crossing a peak at the original bar edge is visible.
        lo = np.searchsorted(times, a/sr, side='left')
        hi = np.searchsorted(times, b/sr, side='right')
        values = np.r_[np.interp([a/sr, b/sr], times, novelty), novelty[lo:hi]]
        return float(values.mean() + .5*values.max()), float(values.max())
    while right-left < goal and (left > 0 or right < n):
        # Reach the hard minimum first; do not inspect/block audio past the
        # endpoint actually being borrowed in this step.
        needed=(hard if right-left<hard else goal)-(right-left)
        proposals = []
        if left > 0:
            idx = np.searchsorted(grid, left, side='left')-1
            a = max(int(grid[max(0,idx)]),left-needed)
            score, peak = cost(a,left)
            proposals.append((0,a,score,peak))
        if right < n:
            idx = np.searchsorted(grid, right, side='right')
            b = min(int(grid[min(idx,len(grid)-1)]),right+needed)
            score, peak = cost(right,b)
            proposals.append((1,b,score,peak))
        for side, endpoint, score, peak in proposals:
            if peak >= boundary_threshold:
                blocked[side] = True
        forced = right-left < hard
        choices = proposals if forced else [p for p in proposals if not blocked[p[0]]]
        if not choices:
            break
        # Lower novelty first; ties expand the side with less borrowed context.
        side, endpoint, score, peak = min(choices, key=lambda p:(round(p[2],8),
            original_left-left if p[0]==0 else right-original_right, p[0]))
        if forced and blocked[side]:
            # Cross only far enough to satisfy the hard 3s requirement.
            needed=hard-(right-left)
            endpoint=max(endpoint,left-needed) if side==0 else min(endpoint,right+needed)
            actual_peak=cost(endpoint,left)[1] if side==0 else cost(right,endpoint)[1]
            crossed.append({'side':'left' if side==0 else 'right', 'peak':actual_peak,
                            'from_sample':left if side==0 else right, 'to_sample':endpoint})
        if side==0: left=endpoint
        else: right=endpoint
    assert right-left >= hard
    return {'prediction_start_sample':int(left), 'prediction_end_sample':int(right),
            'prediction_duration_seconds':(right-left)/sr,
            'left_context_seconds':(original_left-left)/sr,
            'right_context_seconds':(right-original_right)/sr,
            'forced_boundary_crossings':crossed, 'forced_expansion':bool(crossed),
            'minimum_extension_required':original_right-original_left<hard,
            'requested_target_seconds':float(target_seconds),
            'target_duration_seconds':goal/sr,
            'added_duration_seconds':(right-left-original_right+original_left)/sr,
            'original_beat_count':original_beat_count,
            'added_beat_count':beat_count(left,right)-original_beat_count,
            'prediction_beat_count':beat_count(left,right),
            'detected_prediction_beat_count':int(np.searchsorted(detected_beats,right,side='left') -
                                                  np.searchsorted(detected_beats,left,side='left')),
            'beat_count_source':'detected',
            'reached_target':right-left>=goal,
            'stop_reason':'target_reached' if right-left>=goal else
                          ('audio_edges' if left==0 and right==n else 'strong_boundary')}


def make_bar_plan(y, sr, tracker, *, target_seconds=6., boundary_threshold=.7,
                  context_seconds=2., offset_sample=0, model='final0', device='cuda', progress=None,
                  prediction_unit='bar'):
    y=np.asarray(y,dtype=np.float32)
    if y.ndim!=1 or not np.isfinite(y).all() or sr<=0:
        raise ValueError('需要有限数值的单声道音频。')
    if len(y)<math.ceil(3*sr):
        raise ValueError('所选音频不足3秒，请扩大分析范围。')
    if not math.isfinite(context_seconds) or context_seconds<=0:
        raise ValueError('Foote上下文必须为正数。')
    if not math.isfinite(target_seconds) or target_seconds<MINIMUM_SECONDS or not 0<boundary_threshold<=1:
        raise ValueError('a须至少3秒，强边界阈值须在(0,1]。')
    if prediction_unit not in ['bar','beat']:
        raise ValueError('预测单位须为bar或beat。')
    beats, downbeats=tracker(y,sr)
    analysis=build_analysis(y,sr,beats,downbeats,model=model,device=device)
    sources=build_beat_targets(analysis) if prediction_unit=='beat' else analysis['segments']
    if prediction_unit=='beat' and not any(t<len(y) for t in analysis['beat_samples']):
        analysis['warnings'].append('未检测到可用节拍，未生成虚构拍点；保留完整音频，并按a秒上限划分后预测。')
    targets=split_long_targets(sources,sr,target_seconds,analysis['beat_samples'])
    features,times,step=extract_features(y,sr,progress=progress)
    curve=sum(w*foote_novelty(features[k],step,context_seconds) for k,w in
              [('timbre',.45),('chroma',.35),('energy',.20)])
    rows=[]
    for segment in targets:
        row=dict(segment)
        a,b=row['start_sample'],row['end_sample']
        extension=extend_interval(a,b,len(y),sr,analysis['beat_samples'],times,curve,
                                  target_seconds,boundary_threshold)
        row.update(extension)
        row['segment']=row['index']
        row['start_seconds']=(a+offset_sample)/sr
        row['end_seconds']=(b+offset_sample)/sr
        row['duration_seconds']=(b-a)/sr
        row['start_sample']=a+offset_sample
        row['end_sample']=b+offset_sample
        row['source_start_sample']+=offset_sample;row['source_end_sample']+=offset_sample
        row['source_start_seconds']=row['source_start_sample']/sr
        row['source_end_seconds']=row['source_end_sample']/sr
        if prediction_unit=='beat':
            for key in ['bar_start_sample','bar_end_sample']:row[key]+=offset_sample
            row['bar_start_seconds']=row['bar_start_sample']/sr
            row['bar_end_seconds']=row['bar_end_sample']/sr
        for k in ['prediction_start_sample','prediction_end_sample']:
            row[k]+=offset_sample
        for crossing in row['forced_boundary_crossings']:
            crossing['from_sample']+=offset_sample;crossing['to_sample']+=offset_sample
        row['prediction_start_seconds']=row['prediction_start_sample']/sr
        row['prediction_end_seconds']=row['prediction_end_sample']/sr
        row.update(prediction_source='context',inherited_from_segment=None,independent_prediction_skipped=False,
                   prediction_unit=prediction_unit)
        rows.append(row)
    if prediction_unit=='beat':inherit_short_beat_leading(rows,sr,analysis,offset_sample)
    else:inherit_short_leading(rows,sr)
    return {'mode':'beat_context' if prediction_unit=='beat' else 'bar_context',
            'prediction_unit':prediction_unit,'sample_rate':int(sr),'num_samples':len(y),
            'offset_sample':int(offset_sample),'duration_seconds':len(y)/sr,
            'parameters':{'target_seconds':float(target_seconds),'extension_unit':'total_seconds','minimum_seconds':3.,
                          'force_split_long_targets':True,'maximum_target_samples':math.floor(target_seconds*sr),
                          'split_strategy':'balanced_nearest_feasible_beat_with_sample_fallback',
                          'short_leading_inherit_seconds':LEADING_INHERIT_SECONDS,
                          'boundary_threshold':boundary_threshold,'context_seconds':context_seconds,
                          'beat_model':model,'prediction_unit':prediction_unit},'segment_rows':rows,
            'diagnostic':{'time_s':times.tolist(),'novelty':curve.tolist(),
                          'threshold':[boundary_threshold]*len(times)},
            'split_summary':{'source_segment_count':len(sources),'target_segment_count':len(rows),
                             'long_source_count':sum(s['end_sample']-s['start_sample']>math.floor(target_seconds*sr) for s in sources),
                             'added_target_count':len(rows)-len(sources)},
            'beat_analysis':analysis}


def predict_bar_plan(y,sr,plan,tester,threshold=.5,progress=None,compare_raw=True,*,whole_audio=None):
    import soundfile as sf
    if len(y)!=plan['num_samples'] or sr!=plan['sample_rate']:
        raise ValueError('音频与计划不匹配。')
    if whole_audio is not None:
        whole_audio=np.asarray(whole_audio,dtype=np.float32)
        if whole_audio.ndim!=1 or not np.isfinite(whole_audio).all() or len(whole_audio)<math.ceil(3*sr):
            raise ValueError('整曲预测需要至少3秒的有限数值单声道音频。')
    started=time.perf_counter(); rows=[]; cache={};offset=plan['offset_sample']
    total=len(plan['segment_rows'])+int(whole_audio is not None)
    whole_prediction=None;whole_inferences=0
    with tempfile.TemporaryDirectory(prefix='bar_context_') as tmp:
        path=Path(tmp)/'input.wav'
        def run(a,b,signal=y,namespace='selected'):
            key=(namespace,a,b)
            if key not in cache:
                if b-a<math.ceil(3*sr) or not 0<=a<b<=len(signal):
                    raise ValueError('预测范围必须至少3秒且位于音频内。')
                try:
                    sf.write(path,signal[a:b],sr,subtype='FLOAT')
                    cache[key]={**tester.predict(path,threshold=threshold),'status':'ok'}
                except Exception as exc:
                    cache[key]={'status':'error','error':f'{type(exc).__name__}: {exc}',
                                'error_type':type(exc).__name__,'error_traceback':traceback.format_exc(),
                                'valence':None,'arousal':None,'scores':[],'predicted_tags':[]}
            return dict(cache[key])
        for i,item in enumerate(plan['segment_rows']):
            row=dict(item)
            a=row['prediction_start_sample']-offset;b=row['prediction_end_sample']-offset
            row.update(run(a,b));row['padding_seconds']=0.
            raw_a=row['start_sample']-offset;raw_b=row['end_sample']-offset
            if row.get('independent_prediction_skipped'):
                raw={'status':'skipped_inherited','inherited_from_segment':row['inherited_from_segment'],
                     'reason':'开头片段不足1秒，沿用第一小节或其首拍的预测，不单独推理'}
            elif not compare_raw:
                raw={'status':'disabled'}
            elif raw_b-raw_a<math.ceil(3*sr):
                raw={'status':'skipped_minimum','reason':'未扩展目标段不足3秒，未送入模型'}
            else:
                raw=run(raw_a,raw_b)
            row['raw_prediction']=raw
            if row['status']=='ok' and raw.get('status')=='ok':
                row['context_valence_delta']=row['valence']-raw['valence']
                row['context_arousal_delta']=row['arousal']-raw['arousal']
            rows.append(row)
            if progress:progress(i+1,total,row)
        bar_inferences=len(cache)
        if whole_audio is not None:
            # Reuse only when the full waveform really equals the selected
            # waveform; never mistake a same-length crop for the entire song.
            same=offset==0 and len(whole_audio)==len(y) and np.array_equal(whole_audio,y)
            whole_prediction={**run(0,len(whole_audio),whole_audio,'selected' if same else 'whole_track'),
                'segment':0,'kind':'whole_song','analysis_scope':'whole_track',
                'prediction_method':'full_audio_model_inference',
                'start_sample':0,'end_sample':len(whole_audio),'start_seconds':0.,
                'end_seconds':len(whole_audio)/sr,'duration_seconds':len(whole_audio)/sr,
                'sample_rate':int(sr),'padding_seconds':0.}
            whole_inferences=len(cache)-bar_inferences
            if progress:progress(total,total,whole_prediction)
    result={**plan,'segments':rows,'segment_count':len(rows),
            'success_count':sum(r['status']=='ok' for r in rows),'threshold':threshold,
            'elapsed_seconds':round(time.perf_counter()-started,3),'unique_inferences':len(cache),
            'bar_unique_inferences':bar_inferences,'whole_song_inferences':whole_inferences,
            'score_semantics':'independent sigmoid scores; not emotion proportions',
            'prediction_semantics':'expanded audio prediction assigned to target '+plan.get('prediction_unit','bar')+'; approximate local emotion'}
    if whole_prediction is not None:
        result['whole_song_prediction']=whole_prediction
    return result


def csv_bytes(result):
    from backend import ZH
    from tag_scores import correct_result
    base=['segment','kind','prediction_unit','bar_number','beat_number','beat_in_bar',
          'bar_source_segment_index','bar_source_kind','bar_start_seconds','bar_end_seconds',
          'source_segment_index','source_kind','source_start_seconds','source_end_seconds',
          'forced_split','split_part','split_count','split_start_reason','prediction_source',
          'inherited_from_segment','independent_prediction_skipped','start_seconds','end_seconds','duration_seconds',
          'prediction_start_seconds','prediction_end_seconds','prediction_duration_seconds',
          'requested_target_seconds','target_duration_seconds','added_duration_seconds',
          'left_context_seconds','right_context_seconds','original_beat_count','added_beat_count','prediction_beat_count',
          'detected_prediction_beat_count','beat_count_source','forced_expansion','reached_target',
          'stop_reason','valence','arousal','status','raw_status','raw_valence','raw_arousal',
          'context_valence_delta','context_arousal_delta','error','error_type','error_traceback']
    out=io.StringIO(newline='');writer=csv.DictWriter(out,fieldnames=base+[t for t in ZH if t!='funny']);writer.writeheader()
    for r in correct_result(result)['segments']:
        row={k:r.get(k,'') for k in base};raw=r.get('raw_prediction',{})
        row.update(raw_status=raw.get('status',''),raw_valence=raw.get('valence',''),raw_arousal=raw.get('arousal',''))
        row.update({s['tag']:s['score'] for s in r.get('scores',[])})
        writer.writerow(row)
    return out.getvalue().encode('utf-8-sig')
