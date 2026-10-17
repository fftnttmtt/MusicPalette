"""Multi-feature, local Foote novelty segmentation. No learned weights required."""
from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Callable
import numpy as np
from scipy import fft, signal, ndimage
from scipy.io import wavfile


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Return float32 frames x channels, preserving source rate for playback/export."""
    try:
        import soundfile as sf
    except ImportError:
        if Path(path).suffix.lower() != '.wav':
            raise RuntimeError('请安装 requirements.txt 中的 SoundFile，或先用 WAV 文件测试。')
        sr, audio = wavfile.read(path)
        if np.issubdtype(audio.dtype, np.integer):
            info = np.iinfo(audio.dtype)
            audio = ((audio.astype(np.float32)-128)/128 if audio.dtype==np.uint8
                     else audio.astype(np.float32)/max(abs(info.min),info.max))
        else:
            audio = audio.astype(np.float32)
    else:
        audio, sr = sf.read(path, dtype='float32', always_2d=True)
    if audio.ndim == 1:
        audio = audio[:, None]
    if not len(audio) or sr < 1000 or not np.isfinite(audio).all():
        raise ValueError('音频为空、采样率无效或包含 NaN/Inf。')
    return np.ascontiguousarray(audio, dtype=np.float32), int(sr)


def write_wav(path, audio, sr):
    pcm = (np.clip(audio, -1, 1)*32767).round().astype(np.int16)
    if pcm.ndim==2 and pcm.shape[1]==1:
        pcm=pcm[:,0]
    wavfile.write(path, sr, pcm)


def _mel_bank(sr, n_fft, bands=40):
    hz = fft.rfftfreq(n_fft, 1/sr)
    to_mel=lambda v:2595*np.log10(1+np.asarray(v)/700)
    points=700*(10**(np.linspace(to_mel(40),to_mel(min(sr/2,10000)),bands+2)/2595)-1)
    bank=np.zeros((bands,len(hz)),dtype=np.float32)
    for i in range(bands):
        left,center,right=points[i:i+3]
        bank[i]=np.maximum(0,np.minimum((hz-left)/(center-left),(right-hz)/(right-center)))
        bank[i]/=max(float(bank[i].sum()),1e-12)
    return bank


def _chroma_bank(sr,n_fft):
    hz=fft.rfftfreq(n_fft,1/sr)
    bank=np.zeros((12,len(hz)),dtype=np.float32)
    valid=np.flatnonzero((hz>=65)&(hz<=5000))
    midi=69+12*np.log2(hz[valid]/440)
    nearest=np.rint(midi).astype(int)
    weight=np.exp(-0.5*((midi-nearest)/0.35)**2)
    bank[nearest%12,valid]=weight
    return bank


def _unit_rows(x):
    return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-8)


def extract_features(audio, sr, step_seconds=0.05, progress=None, cancel=None):
    y=audio.mean(axis=1,dtype=np.float32) if audio.ndim==2 else np.asarray(audio,dtype=np.float32)
    target=22050
    divisor=math.gcd(sr,target)
    if sr!=target:
        y=signal.resample_poly(y,target//divisor,sr//divisor).astype(np.float32)
    sr=target; n_fft=2048; hop=max(1,round(sr*step_seconds))
    # Center frames on the original timeline. Reflect-padding avoids artificial zero edges.
    y=np.pad(y,(n_fft//2,n_fft//2),mode='reflect' if len(y)>1 else 'edge')
    frames=np.lib.stride_tricks.sliding_window_view(y,n_fft)[::hop]
    count=len(frames)
    mel_bank=_mel_bank(sr,n_fft); chroma_bank=_chroma_bank(sr,n_fft)
    window=signal.windows.hann(n_fft,sym=False).astype(np.float32)
    timbre=[]; chroma=[]; rms=[]
    for a in range(0,count,512):
        if cancel and cancel():raise InterruptedError('已取消分析。')
        block=np.ascontiguousarray(frames[a:a+512])
        spectrum=fft.rfft(block*window,axis=1)
        power=(spectrum.real**2+spectrum.imag**2).astype(np.float32)
        mel=np.maximum(power@mel_bank.T,1e-10)
        coeff=fft.dct(np.log(mel),type=2,norm='ortho',axis=1)[:,1:14]
        timbre.append(coeff)
        c=np.sqrt(np.maximum(power@chroma_bank.T,0))
        chroma.append(c)
        rms.append(np.sqrt(np.mean(block**2,axis=1)))
        if progress:progress(0.12+0.43*min(a+512,count)/count,'提取音色、和声和响度特征')
    timbre=np.concatenate(timbre); chroma=np.concatenate(chroma); rms=np.concatenate(rms)
    # Do not amplify almost-constant coefficients by dividing by a tiny standard deviation.
    timbre=(timbre-np.mean(timbre,axis=0))/np.maximum(np.std(timbre,axis=0),3.0)
    timbre=_unit_rows(np.column_stack([timbre,np.ones(count,dtype=np.float32)]))
    chroma=_unit_rows(chroma)
    db=20*np.log10(np.maximum(rms,1e-6))
    theta=np.clip((db+80)/80,0,1)*np.pi
    energy=np.column_stack([np.cos(theta),np.sin(theta)]).astype(np.float32)
    return {'timbre':timbre,'chroma':chroma,'energy':energy}, np.arange(count)*hop/sr, hop/sr


def foote_novelty(features, step_seconds, context_seconds=2.0):
    """Exact separable checkerboard novelty for cosine self-similarity, without a full similarity matrix.

    The sum of the signed outer-product kernel times F F^T equals the squared
    norm of the signed weighted feature sum. No full T x T matrix is allocated.
    """
    n=len(features)
    if n<4:return np.zeros(n,dtype=np.float32)
    half=max(1,min(round(context_seconds/step_seconds),max(1,(n-1)//3)))
    distances=(np.arange(half,dtype=np.float32)+0.5)*step_seconds
    weights=np.exp(-0.5*(distances/max(context_seconds/2,step_seconds))**2)
    weights/=weights.sum()
    kernel=np.concatenate([-weights[::-1],weights])[:,None]
    # Correlation yields comparison of windows before/after each candidate frame.
    padded=np.pad(features,((half,half),(0,0)),mode='edge')
    sums=signal.correlate(padded,kernel,mode='valid',method='fft')[:n]
    novelty=np.sum(sums*sums,axis=1)/4
    novelty=ndimage.gaussian_filter1d(novelty,max(0.5,0.12/step_seconds))
    # A stable feature sequence should not become a strong curve merely by normalization.
    scale=max(float(np.quantile(novelty,0.99)),0.015)
    return np.clip(novelty/scale,0,1)


def select_boundaries(duration, candidates, min_duration=1.0, max_duration=10.0,
                      novelty_times=None, novelty_values=None, cancel=None):
    """Globally select natural boundaries and insert fallback points if necessary.

    candidates: sequence of (time, score). max_duration=None disables the cap.
    Fallback points are explicitly marked and never called detected changes.
    """
    if not math.isfinite(duration) or duration<=0:raise ValueError('音频时长必须大于 0。')
    if not math.isfinite(min_duration) or min_duration<=0:raise ValueError('最短段长必须大于 0。')
    if max_duration is not None and (not math.isfinite(max_duration) or max_duration<min_duration):
        raise ValueError('最长段长不能小于最短段长。')
    if duration<min_duration:
        return [0.0,duration],{duration:'audio_end'},['整段音频短于最短段长，保留为一个片段。']
    valid={}
    for time_,score in candidates:
        t=float(time_); score=float(score)
        if math.isfinite(t) and math.isfinite(score) and 0<t<duration:
            valid[t]=max(valid.get(t,0),max(0,min(score,1)))
    points={0.0,duration,*valid}
    if max_duration is not None:
        k=math.ceil((duration-1e-9)/max_duration)
        if k*min_duration>duration+1e-8:
            raise ValueError('总时长无法同时满足这些段长限制，请降低最短段长或提高最长段长。')
        # A guaranteed feasible path, including fractional end durations.
        points.update(duration*i/k for i in range(1,k))
        spacing=max(0.1,min(0.25,min_duration/3))
        points.update(np.arange(spacing,duration,spacing).tolist())
    points=np.array(sorted(points),dtype=np.float64)
    reward=np.array([1.05+2*valid[t] if t in valid else 0 for t in points])
    change=(np.interp(points,novelty_times,novelty_values) if novelty_times is not None and len(novelty_times)>0
            else np.zeros(len(points)))
    costs=np.full(len(points),np.inf);costs[0]=0;previous=np.full(len(points),-1,dtype=int)
    target=(min_duration+max_duration)/2 if max_duration else None
    for j in range(1,len(points)):
        if cancel and j%100==0 and cancel():raise InterruptedError('已取消分析。')
        lo=np.searchsorted(points,points[j]-max_duration-1e-8) if max_duration else 0
        hi=np.searchsorted(points,points[j]-min_duration+1e-8,side='right')
        hi=min(hi,j)
        if hi<=lo:continue
        lengths=points[j]-points[lo:hi]
        regularizer=0.12*((lengths-target)/target)**2 if target else 0
        # Soft preference for stronger changes when a cap forces a new boundary.
        boundary_reward=reward[j] if reward[j]>0 else (0.35*change[j] if j<len(points)-1 else 0)
        choices=costs[lo:hi]+1+regularizer-boundary_reward
        best=int(np.argmin(choices))
        if np.isfinite(choices[best]):costs[j]=choices[best];previous[j]=lo+best
    if previous[-1]<0:raise ValueError('未找到满足段长约束的切分，请稍微放宽参数。')
    indices=[];j=len(points)-1
    while j>=0:
        indices.append(j)
        if j==0:break
        j=int(previous[j])
    chosen=points[indices[::-1]].tolist()
    reasons={float(t):('audio_end' if t==duration else 'acoustic_change' if t in valid else 'length_limit')
             for t in chosen[1:]}
    return chosen,reasons,[]


def segment_audio(audio, sr, *, min_duration=1.0, max_duration=10.0, sensitivity=0.8,
                  context_seconds=2.0, timbre_weight=0.45,chroma_weight=0.35,energy_weight=0.20,
                  progress:Callable|None=None,cancel:Callable|None=None):
    if not 0<=sensitivity<=1:raise ValueError('灵敏度应在 0～1 之间。')
    if not math.isfinite(context_seconds) or context_seconds<=0:raise ValueError('比较窗口必须大于 0。')
    weights=np.array([timbre_weight,chroma_weight,energy_weight],dtype=float)
    if not np.isfinite(weights).all() or np.min(weights)<0 or weights.sum()<=0:
        raise ValueError('特征权重需非负，且至少一种权重大于 0。')
    weights/=weights.sum();duration=len(audio)/sr
    if progress:progress(0.05,'准备音频')
    groups,times,step=extract_features(audio,sr,progress=progress,cancel=cancel)
    curves={name:foote_novelty(groups[name],step,context_seconds) for name in ['timbre','chroma','energy']}
    novelty=sum(w*curves[name] for w,name in zip(weights,curves))
    median_width=max(3,round(8/step)|1)
    baseline=ndimage.median_filter(novelty,size=median_width,mode='nearest')
    threshold=baseline+(0.06+0.30*(1-sensitivity))
    prominence=0.04+0.14*(1-sensitivity)
    peaks,_=signal.find_peaks(novelty,prominence=prominence,distance=max(1,round(0.25/step)))
    candidates=[(float(times[i]),float(novelty[i])) for i in peaks
                if times[i]<duration and novelty[i]>threshold[i]]
    if progress:progress(0.75,'选择变化边界并满足段长约束')
    bounds,reasons,exceptions=select_boundaries(duration,candidates,min_duration,max_duration,times,novelty,cancel)
    segments=[]
    for i,(a,b) in enumerate(zip(bounds[:-1],bounds[1:]),1):
        segments.append({'index':i,'start':a,'end':b,'duration':b-a,
                         'boundary_reason':reasons[b],'acoustic_change':float(np.interp(b,times,novelty))})
    # Boundary values and curve samples are diagnostic scores, not emotion probabilities.
    result={'algorithm':'multi_feature_foote_v1','duration':duration,'source_sample_rate':sr,
            'settings':{'min_duration':min_duration,'max_duration':max_duration,'sensitivity':sensitivity,
                        'context_seconds':context_seconds,'weights':dict(zip(['timbre','chroma','energy'],weights.tolist()))},
            'segments':segments,'boundaries':bounds,'constraint_exceptions':exceptions,
            'candidate_boundaries':[{'time':t,'score':s} for t,s in candidates],
            'novelty':{'times':times.tolist(),'combined':novelty.tolist(),'threshold':threshold.tolist(),
                       **{name:values.tolist() for name,values in curves.items()}}}
    if progress:progress(1.0,'分析完成')
    return result


def export_segments(audio,sr,result,directory):
    target=Path(directory);target.mkdir(parents=True,exist_ok=True)
    for s in result['segments']:
        a=round(s['start']*sr);b=round(s['end']*sr)
        write_wav(target/f'{s["index"]:03d}_{s["start"]:.2f}-{s["end"]:.2f}s.wav',audio[a:b],sr)
    (target/'segments.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
