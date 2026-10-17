"""Duration-weighted song-relative contrast before sigmoid and gamma."""
import math


def finite(value):
    return isinstance(value,(int,float)) and math.isfinite(value)


def clip(value):
    return max(0.,min(1.,value))


def weighted_quantile(values, quantile):
    ordered=sorted(values)
    total=math.fsum(weight for _,weight in ordered)
    threshold=quantile*total
    cumulative=0.
    for value,weight in ordered:
        cumulative+=weight
        if cumulative>=threshold:return value
    return ordered[-1][0]


def adaptive_statistics(result, percentile_low=10., percentile_high=90., max_gain=5.):
    if not all(finite(v) for v in [percentile_low,percentile_high,max_gain]) or not 0<=percentile_low<percentile_high<=100 or max_gain<1:
        raise ValueError('分位数须满足0≤低值<高值≤100，最大放大倍数须≥1。')
    rows=[]
    for row in result.get('segments',[]):
        if row.get('status')!='ok' or not all(finite(row.get(k)) for k in ['arousal','valence','start_seconds','end_seconds']):continue
        duration=row['end_seconds']-row['start_seconds']
        if duration>0:rows.append((row,duration))
    stats={}
    for dimension in ['arousal','valence']:
        values=[(clip((r[dimension]-1.)/8.),duration) for r,duration in rows]
        item={'available':bool(values),'segment_count':len(values),
              'valid_duration_seconds':math.fsum(d for _,d in values),
              'percentile_low':percentile_low,'percentile_high':percentile_high,
              'max_gain':max_gain,'weighting':'original_segment_duration',
              'scope':'available_prediction_intervals','constant':False,
              'low':None,'high':None,'lower_bound':None,'upper_bound':None,'effective_span':None}
        if values:
            low=weighted_quantile(values,percentile_low/100.)
            high=weighted_quantile(values,percentile_high/100.)
            span=max(high-low,1./max_gain)
            midpoint=(low+high)/2.
            item.update(low=low,high=high,lower_bound=midpoint-span/2.,
                        upper_bound=midpoint+span/2.,effective_span=span,
                        constant=max(x for x,_ in values)-min(x for x,_ in values)<=1e-9)
        stats[dimension]=item
    return stats


def adapt_value(x, statistics, strength):
    if not finite(strength) or not 0<=strength<=1:
        raise ValueError('S/V增强强度须在0–1内。')
    x=clip(x)
    if not statistics or not statistics['available'] or statistics['constant'] or strength==0:return x
    relative=clip((x-statistics['lower_bound'])/statistics['effective_span'])
    return clip((1-strength)*x+strength*relative)


def validate_output_range(bounds):
    if len(bounds)!=2 or not all(finite(x) for x in bounds) or not 0<=bounds[0]<=bounds[1]<=1:
        raise ValueError('S/V输出范围须满足0≤下限≤上限≤1。')


def output_range(value, bounds):
    validate_output_range(bounds)
    return bounds[0]+(bounds[1]-bounds[0])*value
