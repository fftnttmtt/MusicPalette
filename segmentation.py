"""Feature-driven, non-overlapping musical segmentation (no learned weights)."""
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import find_peaks


def _contrast(x, radius):
    """Euclidean distance between the mean features on each side of a boundary."""
    x = np.asarray(x, dtype=float)
    n = x.shape[1]
    k = np.arange(n)
    left, right = np.maximum(0, k-radius), np.minimum(n, k+radius)
    cumulative = np.pad(np.cumsum(x, axis=1), ((0,0),(1,0)))
    before = (cumulative[:,k]-cumulative[:,left])/np.maximum(1,k-left)
    after = (cumulative[:,right]-cumulative[:,k])/np.maximum(1,right-k)
    d = np.sqrt(np.mean((after-before)**2,axis=0))
    d[:min(radius,n)] = 0
    d[max(0,n-radius):] = 0
    return d


def _unit_curve(curve):
    # Absolute floor prevents nearly constant signals being amplified into structure.
    scale = max(float(np.percentile(curve,95)), .15)
    return np.clip(curve/scale,0,3)


def select_boundaries(n_samples, sr, frame_times, novelty, threshold, beats,
                      min_duration=1., max_duration=None, snap=.25):
    """Select novelty peaks, enforce lengths, snap only strong nearby beat positions.

    Return exact sample boundaries and an explanation of every boundary. The last
    endpoint always equals n_samples. Long homogeneous spans use marked fallbacks.
    """
    if n_samples<1 or sr<=0 or min_duration<=0 or (max_duration is not None and (not np.isfinite(max_duration) or max_duration<2*min_duration)):
        raise ValueError('必须有音频，最短段长 > 0，最长段长关闭或 ≥ 最短段长的 2 倍')
    t = np.asarray(frame_times)
    novelty = np.asarray(novelty)
    threshold = np.asarray(threshold)
    beat_samples = np.asarray(np.rint(np.asarray(beats)*sr),dtype=int)
    minimum = max(1,round(min_duration*sr))
    maximum = round(max_duration*sr) if max_duration is not None else n_samples
    peaks,_ = find_peaks(novelty, height=threshold, prominence=.08)
    chosen = {0:('start',0.), n_samples:('end',0.)}
    # Strong boundaries take priority over weak ones when candidates are too close.
    for k in sorted(peaks,key=lambda k:float(novelty[k]-threshold[k]),reverse=True):
        original = int(round(t[k]*sr))
        candidate,reason = original,'acoustic_change'
        if len(beat_samples) and snap>0:
            b = int(beat_samples[np.argmin(np.abs(beat_samples-original))])
            if abs(b-original)<=snap*sr and np.interp(b/sr,t,novelty)>=.65*novelty[k]:
                if all(abs(b-p)>=minimum for p in chosen):
                    candidate,reason = b,'acoustic_change_beat'
        if 0<candidate<n_samples and all(abs(candidate-p)>=minimum for p in chosen):
            chosen[candidate]=(reason,float(novelty[k]))
    # Keep detected boundaries; split only remaining overlong gaps. Interior cuts
    # leave at least minimum samples on both sides, preventing a tiny final tail.
    initial = sorted(chosen)
    for left,right in zip(initial[:-1],initial[1:]):
        while right-left>maximum:
            low,high = left+minimum,min(left+maximum,right-minimum)
            target=min(left+round(.8*maximum), high)
            all_peaks,_=find_peaks(novelty,prominence=.03)
            peak_samples=np.rint(t[all_peaks]*sr).astype(int)
            options = np.unique(np.concatenate([peak_samples,beat_samples]))
            options = options[(options>=low)&(options<=high)]
            if len(options):
                scores=np.interp(options/sr,t,novelty)-.15*np.abs(options-target)/maximum
                cut=int(options[np.argmax(scores)])
                reason='max_duration_acoustic_fallback'
            else:
                cut=int(high)
                reason='max_duration_time_fallback'
            chosen[cut]=(reason,float(np.interp(cut/sr,t,novelty)))
            left=cut
    boundaries=sorted(chosen)
    rows=[{'sample':p,'time_s':p/sr,'reason':chosen[p][0],
           'novelty':chosen[p][1]} for p in boundaries]
    return list(zip(boundaries[:-1],boundaries[1:])),rows


def acoustic_segments(y, sr=22050, min_duration=1., max_duration=None,
                      sensitivity=.5, snap_to_beat=True):
    """MFCC/chroma/RMS multi-scale change detection, then optional beat snapping."""
    import librosa
    y=np.asarray(y,dtype=np.float32)
    if y.ndim!=1 or len(y)==0 or not np.isfinite(y).all():
        raise ValueError('声学分段需要非空、有限数值的单声道音频')
    if not 0<=sensitivity<=1:
        raise ValueError('灵敏度范围必须为 0–1')
    hop=512
    duration=len(y)/sr
    if len(y)<2048 or np.max(np.abs(y))<1e-5:
        t=np.arange(0,len(y),hop)/sr
        novelty=np.zeros(len(t)); threshold=np.ones(len(t))
        beats=np.array([])
    else:
        power=np.abs(librosa.stft(y,n_fft=2048,hop_length=hop))**2
        mel=librosa.feature.melspectrogram(S=power,sr=sr,n_mels=64)
        db=librosa.power_to_db(mel,ref=np.max)
        mfcc=librosa.feature.mfcc(S=db,n_mfcc=14)[1:]
        mfcc=(mfcc-np.median(mfcc,axis=1,keepdims=True))/np.maximum(np.std(mfcc,axis=1,keepdims=True),5.)
        chroma=librosa.feature.chroma_stft(S=power,sr=sr,hop_length=hop,tuning=0.)
        chroma/=np.maximum(np.linalg.norm(chroma,axis=0,keepdims=True),1e-6)
        rms=librosa.feature.rms(S=np.sqrt(power))[0]
        energy=20*np.log10(np.maximum(rms,1e-6))
        energy=((energy-np.median(energy))/max(float(np.std(energy)),6.))[None,:]
        t=np.arange(power.shape[1])*hop/sr
        curves=[]
        for x in (mfcc,chroma,energy):
            # Compare neighborhoods at two scales, rather than cutting at every onset.
            x=gaussian_filter1d(x,sigma=.10*sr/hop,axis=1,mode='nearest')
            curve=.4*_contrast(x,max(1,round(.5*sr/hop)))+.6*_contrast(x,max(1,round(1.5*sr/hop)))
            curves.append(_unit_curve(curve))
        novelty=gaussian_filter1d(.45*curves[0]+.35*curves[1]+.20*curves[2],sigma=.10*sr/hop)
        local=median_filter(novelty,size=2*round(4*sr/hop)+1,mode='nearest')
        threshold=local+(.45-.35*sensitivity)
        beats=np.array([])
        if snap_to_beat:
            onset=librosa.onset.onset_strength(S=db,sr=sr,hop_length=hop)
            if np.max(onset)>1e-6:
                _,beats=librosa.beat.beat_track(onset_envelope=onset,sr=sr,hop_length=hop,units='time')
    spans,rows=select_boundaries(len(y),sr,t,novelty,threshold,beats,
                               min_duration,max_duration,.25 if snap_to_beat else 0.)
    diagnostic={'time_s':t.tolist(),'novelty':novelty.tolist(),
                'threshold':threshold.tolist(),'beats_s':np.asarray(beats).tolist(),
                'duration_s':duration,'boundaries':rows}
    return spans,diagnostic
