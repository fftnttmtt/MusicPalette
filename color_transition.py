"""Boundary holds/fades and the original Foote-warped OKLab interpolation.

Boundary mode uses absolute Foote level to select fade duration. Legacy mode
integrates a speed field on the feature grid plus color anchors. Both are
independent of display/export sampling; failed targets and gaps split the runs.
"""
from bisect import bisect_right
import colorsys
import csv
import io
import math
from copy import deepcopy


def _finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def _rgb_to_lab(rgb):
    r, g, b = [v / 255. for v in rgb]
    r, g, b = [v / 12.92 if v <= .04045 else ((v + .055) / 1.055)**2.4 for v in (r, g, b)]
    l = (.4122214708*r + .5363325363*g + .0514459929*b)**(1/3)
    m = (.2119034982*r + .6806995451*g + .1073969566*b)**(1/3)
    s = (.0883024619*r + .2817188376*g + .6299787005*b)**(1/3)
    return [.2104542553*l + .7936177850*m - .0040720468*s,
            1.9779984951*l - 2.4285922050*m + .4505937099*s,
            .0259040371*l + .7827717662*m - .8086757660*s]


def _lab_to_rgb(lab):
    L, a, b = lab
    l = (L + .3963377774*a + .2158037573*b)**3
    m = (L - .1055613458*a - .0638541728*b)**3
    s = (L - .0894841775*a - 1.291485548*b)**3
    linear = [4.0767416621*l - 3.3077115913*m + .2309699292*s,
              -1.2684380046*l + 2.6097574011*m - .3413193965*s,
              -.0041960863*l - .7034186147*m + 1.707614701*s]
    rgb = []
    for v in linear:
        v = max(0., min(1., v))
        v = 12.92*v if v <= .0031308 else 1.055*v**(1/2.4) - .055
        rgb.append(int(math.floor(max(0., min(1., v))*255 + .5)))
    return rgb


def _interp(times, values, t):
    i = bisect_right(times, t) - 1
    if i < 0: return values[0]
    if i >= len(times)-1: return values[-1]
    u = (t-times[i])/(times[i+1]-times[i])
    return values[i]*(1-u)+values[i+1]*u


def integral(curve, t):
    grid, rate, area = curve['grid'], curve['rate'], curve['area']
    if len(grid) == 1: return 0.
    t = max(grid[0], min(grid[-1], t))
    j = max(0, min(len(grid)-2, bisect_right(grid, t)-1))
    d = t-grid[j]
    slope = (rate[j+1]-rate[j])/(grid[j+1]-grid[j])
    return area[j]+rate[j]*d+.5*slope*d*d


def build_transition(mapped, result, strength=8., enabled=True, *, mode='foote_exponential',
                     boundary_radius=.25,max_transition_fraction=1.,boundary_statistic='mean'):
    if not _finite(strength) or not 0 <= strength <= 10:
        raise ValueError('过渡响应强度须在0–10内。')
    if mode not in ['foote_exponential','boundary_hold']:
        raise ValueError('未知颜色过渡模式。')
    if not _finite(boundary_radius) or boundary_radius<0:
        raise ValueError('边界比较半径须为非负数。')
    if not _finite(max_transition_fraction) or not 0<max_transition_fraction<=1:
        raise ValueError('过渡占新片段的最大比例须在(0,1]内。')
    if boundary_statistic not in ['mean','peak']:
        raise ValueError('边界Foote统计方式须为mean或peak。')
    rows = sorted(mapped['segments'], key=lambda r: r['start_seconds'])
    runs = []
    previous = None
    for row in rows:
        if row.get('hex') is None or row.get('status') != 'ok':
            previous = None
            continue
        if previous is None or abs(previous['end_seconds']-row['start_seconds']) > 1e-6:
            runs.append({'start': row['start_seconds'], 'end': row['end_seconds'],
                         'times': [], 'labs': [], 'anchor_hex': [], 'segments': []})
        run = runs[-1]
        run['end'] = row['end_seconds']
        # Original bar starts are immutable color anchors in every smooth mode.
        run['times'].append(row['start_seconds'])
        run['labs'].append(_rgb_to_lab(row['rgb']))
        run['anchor_hex'].append(row['hex'])
        run['segments'].append({'segment':row['segment'],'start':row['start_seconds'],'end':row['end_seconds']})
        previous = row
    diagnostic = result.get('diagnostic') or {}
    at, novelty = diagnostic.get('time_s', []), diagnostic.get('novelty', [])
    valid = (isinstance(at, list) and isinstance(novelty, list) and len(at) > 0
             and len(at) == len(novelty) and all(_finite(t) for t in at)
             and all(_finite(n) for n in novelty) and all(b > a for a, b in zip(at, at[1:])))
    offset = 0.
    if valid:
        sr, sample = result.get('sample_rate'), result.get('offset_sample', 0)
        if _finite(sr) and sr > 0 and _finite(sample): offset = sample/sr
        at = [float(t)+offset for t in at]
    if mode=='boundary_hold':
        # Arrive at the next target's color exactly at its original start.
        # Fade only inside the interval between consecutive start anchors;
        # no delay, global shift, overlap, or movement of beat/bar timestamps.
        for run in runs:
            events=[]
            for j,(left,right) in enumerate(zip(run['segments'],run['segments'][1:])):
                boundary=right['start']
                receiving_duration=right['end']-right['start']
                interval_duration=boundary-left['start']
                cap=max_transition_fraction*interval_duration
                radius=min(boundary_radius,.5*min(left['end']-left['start'],right['end']-right['start']))
                lo=max(boundary-radius,at[0]) if valid else boundary
                hi=min(boundary+radius,at[-1]) if valid else boundary
                available=valid and hi>=lo
                novelty_value=None
                novelty_peak=None;novelty_mean=None
                if available:
                    a=bisect_right(at,lo);b=bisect_right(at,hi)
                    points=sorted(set([lo,hi]+at[a:b]))
                    values=[max(0.,min(1.,_interp(at,novelty,t))) for t in points]
                    novelty_peak=max(values)
                    novelty_mean=(math.fsum(.5*(x+y)*(tb-ta) for ta,tb,x,y in
                                  zip(points,points[1:],values,values[1:]))/(hi-lo)) if hi>lo else values[0]
                    novelty_value=novelty_mean if boundary_statistic=='mean' else novelty_peak
                    # No fixed limits in seconds: scale the available interval
                    # between the original starts by the novelty response.
                    response=novelty_value if strength<1e-6 else math.expm1(strength*novelty_value)/math.expm1(strength)
                    requested=interval_duration*max(0.,min(1.,1.-response))
                else:
                    requested=interval_duration
                duration=min(requested,cap)
                fade_start=max(left['start'],boundary-duration)
                duration=boundary-fade_start
                events.append({'boundary_seconds':boundary,'start_seconds':fade_start,
                               'end_seconds':boundary,'duration_seconds':duration,
                               'requested_duration_seconds':requested,'duration_cap_seconds':cap,
                               'receiving_duration_seconds':receiving_duration,
                               'anchor_interval_seconds':interval_duration,'anchor_time_error_seconds':0.,
                               'duration_capped':requested>cap+1e-9,'novelty':novelty_value,
                               'novelty_mean':novelty_mean,'novelty_peak':novelty_peak,
                               'boundary_statistic':boundary_statistic,
                               'novelty_available':bool(available),'measurement_start_seconds':lo if available else None,
                               'measurement_end_seconds':hi if available else None,
                               'from_segment':left['segment'],'to_segment':right['segment'],
                               'from_index':j,'to_index':j+1})
            run['events']=events
        return {'enabled':bool(enabled),'mode':'boundary_hold','strength':strength,
                'interpolation_space':'OKLab','anchor':'original_segment_start',
                'duration_limits_seconds':None,'boundary_radius_seconds':boundary_radius,
                'max_segment_fraction':max_transition_fraction,'duration_cap_basis':'interval_between_original_segment_starts',
                'boundary_statistic':boundary_statistic,'foote_available':bool(valid),'diagnostic_offset_seconds':offset,
                'duration_formula':'anchor_interval*(1-expm1(strength*novelty)/expm1(strength)); strength=0: anchor_interval*(1-novelty)',
                'novelty_measurement':'time-weighted mean or peak Foote in boundary window, clipped to 0..1',
                'timing':'end exactly at original next segment start; start=end-duration within preceding anchor interval',
                'progress':'smoothstep((time-start)/duration); hold outside fade',
                'missing_foote':'use interval between original segment starts, subject only to configured percentage',
                'maximum_novelty':'zero duration means an instantaneous color change','runs':runs}
    anchors = [t for run in runs for t in run['times']]
    grid = sorted(set((at if valid else []) + anchors)) or [0.]
    rate = [.2+math.expm1(min(20., strength*max(0., min(3., _interp(at, novelty, t)))))
            if valid else .2 for t in grid]
    area = [0.]
    for i in range(len(grid)-1):
        area.append(area[-1]+.5*(rate[i]+rate[i+1])*(grid[i+1]-grid[i]))
    out = {'enabled': bool(enabled), 'mode': 'foote_exponential' if valid else 'smooth_fallback_no_foote',
           'strength': strength, 'interpolation_space': 'OKLab', 'anchor': 'original_segment_start',
           'speed': '0.2 + expm1(clip(strength * clip(novelty, 0, 3), 0, 20))',
           'progress': 'smoothstep(normalized integral of piecewise-linear speed)',
           'diagnostic_offset_seconds': offset, 'grid': grid, 'rate': rate, 'area': area, 'runs': runs}
    for run in runs:
        run['anchor_area'] = [integral(out, t) for t in run['times']]
    return out


def start_aligned_transition(curve):
    """Causal playback: begin each fade at its target's original start.

    The legacy boundary_hold curve reaches the next color AT that start,
    which necessarily reveals it early. Keep legacy exports readable while
    making SPECTRA playback wait for the corresponding musical boundary.
    """
    if curve.get('mode') != 'boundary_hold':
        return deepcopy(curve)
    out = deepcopy(curve)
    fraction = out.get('max_segment_fraction', .15)
    for run in out['runs']:
        for event in run['events']:
            interval = event['receiving_duration_seconds']
            old_interval = event['anchor_interval_seconds']
            ratio = event['requested_duration_seconds']/old_interval if old_interval > 0 else 0.
            requested = interval * max(0., min(1., ratio))
            cap = interval * fraction
            duration = min(requested, cap)
            boundary = event['boundary_seconds']
            event.update(start_seconds=boundary, end_seconds=boundary+duration,
                         duration_seconds=duration, requested_duration_seconds=requested,
                         duration_cap_seconds=cap, duration_capped=requested > cap+1e-9,
                         anchor_time_error_seconds=None, transition_start_error_seconds=0.)
    out.update(mode='boundary_start', anchor='transition_start_at_original_segment_start',
               duration_cap_basis='receiving_segment_start_interval',
               duration_formula='receiving_interval*(1-expm1(strength*novelty)/expm1(strength)); capped by configured fraction of receiving interval',
               timing='start exactly at original next segment start; never show next color before boundary')
    return out


def transition_color(curve, t):
    runs = curve['runs']
    i = bisect_right(runs, t, key=lambda run: run['start'])-1
    if i < 0 or t > runs[i]['end']: return None
    run = runs[i]
    if t == run['end'] and i+1 < len(runs) and runs[i+1]['start'] == t:
        run = runs[i+1]
    times, labs = run['times'], run['labs']
    anchor_index=bisect_right(times,t)-1
    if curve.get('mode')!='boundary_start' and anchor_index>=0 and times[anchor_index]==t:
        rgb=_lab_to_rgb(labs[anchor_index])
    elif curve.get('mode') in ['boundary_hold','boundary_start']:
        events=run['events']
        j=bisect_right(events,t,key=lambda event:event['start_seconds'])-1
        if j<0:lab=labs[0]
        else:
            event=events[j]
            u=1. if event['duration_seconds']<=0 else max(0.,min(1.,(t-event['start_seconds'])/event['duration_seconds']))
            u=u*u*(3-2*u)
            lab=[a*(1-u)+b*u for a,b in zip(labs[event['from_index']],labs[event['to_index']])]
        rgb=_lab_to_rgb(lab)
    elif len(times) == 1 or t <= times[0]: rgb = _lab_to_rgb(labs[0])
    elif t >= times[-1]: rgb = _lab_to_rgb(labs[-1])
    else:
        j = bisect_right(times, t)-1
        denominator = run['anchor_area'][j+1]-run['anchor_area'][j]
        u = max(0., min(1., (integral(curve, t)-run['anchor_area'][j])/denominator))
        u = u*u*(3-2*u)
        rgb = _lab_to_rgb([a*(1-u)+b*u for a, b in zip(labs[j], labs[j+1])])
    h, s, v = colorsys.rgb_to_hsv(*(v/255. for v in rgb))
    return {'rgb': rgb, 'hex': '#'+''.join(f'{v:02x}' for v in rgb),
            'hue_degrees': h*360, 'saturation': s, 'value': v}


def transition_csv(mapped, fps=30):
    """30 Hz base grid plus exact anchors/fade endpoints; never round bar times."""
    curve = mapped['transition']
    rows = sorted(mapped['segments'], key=lambda r: r['start_seconds'])
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=['time_seconds', 'segment', 'hex', 'R', 'G', 'B',
                                               'hue_degrees', 'saturation', 'value', 'status'])
    writer.writeheader()
    if not rows: return stream.getvalue().encode('utf-8-sig')
    start, end = rows[0]['start_seconds'], max(r['end_seconds'] for r in rows)
    times = [start+i/fps for i in range(math.ceil((end-start)*fps))]+[end]
    times += [r[key] for r in rows for key in ['start_seconds','end_seconds']]
    if curve.get('mode')=='boundary_hold':
        times += [event[key] for run in curve['runs'] for event in run['events']
                  for key in ['start_seconds','end_seconds']]
    times=sorted(set(times))
    j = 0
    for t in times:
        while j+1 < len(rows) and rows[j+1]['start_seconds'] <= t: j += 1
        row = rows[j]
        inside=row['start_seconds']<=t<row['end_seconds']
        at_eof=j==len(rows)-1 and t==end and t==row['end_seconds']
        valid=row.get('hex') and (inside or at_eof)
        color = (transition_color(curve, t) if curve['enabled'] else row) if valid else None
        out = {'time_seconds': t, 'segment': row['segment'], 'status': 'ok' if color else 'unmapped'}
        if color:
            out.update({k: color[k] for k in ['hex', 'hue_degrees', 'saturation', 'value']})
            out.update(zip(['R', 'G', 'B'], color['rgb']))
        writer.writerow(out)
    return stream.getvalue().encode('utf-8-sig')
