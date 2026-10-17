"""User-defined HSV mapping: nine-tag circular hue mean; funny disabled."""
import colorsys
import csv
from html import escape
import io
import json
import math
from tag_scores import corrected_score_items, valid_score
from sv_adaptive import adaptive_statistics, adapt_value, output_range, validate_output_range

TAG_HUES = {'calm':180, 'romantic':310, 'dream':270, 'sad':240, 'epic':40,
            'happy':60, 'love':330, 'nature':110, 'energetic':10}


def _finite(value):
    return isinstance(value,(int,float)) and math.isfinite(value)


def normalized_sigmoid(x, k=10., center=.5):
    if not all(_finite(v) for v in [x,k,center]) or k<0 or not 0<=center<=1:
        raise ValueError('Sigmoid要求有限数值，斜率k≥0，中心在0–1内。')
    x=max(0.,min(1.,x))
    if k<1e-6 or x==0 or x==1:return x
    if x==.5 and center==.5:return .5
    def sigmoid(z):
        if z>=0:return 1./(1.+math.exp(-z))
        e=math.exp(z);return e/(1.+e)
    low=sigmoid(-k*center);high=sigmoid(k*(1-center))
    return max(0.,min(1.,(sigmoid(k*(x-center))-low)/(high-low)))


def gamma_transform(x, gamma=1.):
    if not _finite(x) or not _finite(gamma) or gamma<=0:
        raise ValueError('伽马指数γ必须为有限正数。')
    return max(0.,min(1.,x))**gamma


def segment_color(segment, *, saturation_k=10., value_k=10., saturation_center=.5, value_center=.5,
                  hue_gamma=2., saturation_gamma=1., value_gamma=1., sv_adaptation=None,
                  saturation_range=(0.,1.), value_range=(0.,1.)):
    out={'segment':segment['segment'], 'start_seconds':segment['start_seconds'],
         'end_seconds':segment['end_seconds'], 'valence':segment.get('valence') if _finite(segment.get('valence')) else None,
         'arousal':segment.get('arousal') if _finite(segment.get('arousal')) else None, 'hue_degrees':None, 'saturation':None,
         'value':None, 'rgb':None, 'hex':None, 'status':'prediction_error'}
    for key in ['prediction_source','inherited_from_segment','independent_prediction_skipped']:
        if key in segment:out[key]=segment[key]
    if segment.get('status')!='ok': return out
    valence,arousal=segment.get('valence'),segment.get('arousal')
    if not _finite(valence) or not _finite(arousal):
        out['status']='invalid_valence_arousal';return out
    out['normalized_valence']=max(0.,min(1.,(valence-1.)/8.))
    out['normalized_arousal']=max(0.,min(1.,(arousal-1.)/8.))
    adaptation=sv_adaptation or {'enabled':False}
    stats=adaptation.get('statistics',{}) if adaptation['enabled'] else {}
    strength=adaptation.get('strength',0.)
    out['adapted_arousal']=adapt_value(out['normalized_arousal'],stats.get('arousal'),strength)
    out['adapted_valence']=adapt_value(out['normalized_valence'],stats.get('valence'),strength)
    gamma_transform(.5,hue_gamma)
    out['value_before_gamma']=normalized_sigmoid(out['adapted_valence'],value_k,value_center)
    out['saturation_before_gamma']=normalized_sigmoid(out['adapted_arousal'],saturation_k,saturation_center)
    out['value_after_gamma']=gamma_transform(out['value_before_gamma'],value_gamma)
    out['saturation_after_gamma']=gamma_transform(out['saturation_before_gamma'],saturation_gamma)
    out['value']=output_range(out['value_after_gamma'],value_range)
    out['saturation']=output_range(out['saturation_after_gamma'],saturation_range)
    out['va_clipped']=not (1<=valence<=9 and 1<=arousal<=9)
    scores={item['tag']:item.get('score') for item in corrected_score_items(segment.get('scores',[])) if item.get('tag') in TAG_HUES}
    if not any(valid_score(scores.get(t)) for t in TAG_HUES):
        out['status']='missing_or_invalid_tag_scores';return out
    out['tag_scores']={t:float(scores[t]) for t in TAG_HUES if valid_score(scores.get(t))}
    out['invalid_or_missing_tags']=[t for t in TAG_HUES if not valid_score(scores.get(t))]
    # A zero score contributes zero; every positive valid score participates,
    # including values below the former 0.3 cutoff.
    weights={tag:gamma_transform(score,hue_gamma) for tag,score in out['tag_scores'].items() if score>0}
    out['hue_selection']='all_valid_weighted_mean'
    out['hue_fallback_tag']=None
    if not weights or sum(weights.values())<=0:
        winner=max(out['tag_scores'],key=out['tag_scores'].get)
        weights={winner:1.}
        out['hue_selection']='highest_valid_tag_fallback'
        out['hue_fallback_tag']=winner
    out['hue_weights']=weights
    out['hue_tags_used']=','.join(weights)
    out['hue_tag_count']=len(weights)
    total=sum(weights.values())
    x=sum(score*math.cos(math.radians(TAG_HUES[tag])) for tag,score in weights.items())
    y=sum(score*math.sin(math.radians(TAG_HUES[tag])) for tag,score in weights.items())
    magnitude=math.hypot(x,y)
    out['hue_resultant_length']=magnitude/total if total>0 else 0.
    if total<=0 or magnitude<=1e-12*total:
        out['status']='undefined_hue';return out
    h=math.degrees(math.atan2(y,x))%360
    rgb_float=colorsys.hsv_to_rgb(h/360.,out['saturation'],out['value'])
    rgb=[round(v*255) for v in rgb_float]
    out.update(hue_degrees=h,rgb=rgb,hex='#'+''.join(f'{v:02x}' for v in rgb),status='ok')
    return out


def song_mean_centers(result):
    valid=[]
    for segment in result.get('segments',[]):
        if segment.get('status')!='ok' or not all(_finite(segment.get(k)) for k in ['valence','arousal','start_seconds','end_seconds']):
            continue
        duration=segment['end_seconds']-segment['start_seconds']
        if duration>0:valid.append((segment,duration))
    duration=math.fsum(d for _,d in valid)
    if not duration:
        return {'available':False,'segment_count':0,'valid_duration_seconds':0.,
                'arousal_mean':None,'valence_mean':None,'saturation_center':None,'value_center':None,
                'weighting':'original_segment_duration','scope':'available_prediction_intervals'}
    arousal=math.fsum(s['arousal']*(d/duration) for s,d in valid)
    valence=math.fsum(s['valence']*(d/duration) for s,d in valid)
    return {'available':True,'segment_count':len(valid),'valid_duration_seconds':duration,
            'arousal_mean':arousal,'valence_mean':valence,
            'saturation_center':max(0.,min(1.,(arousal-1.)/8.)),
            'value_center':max(0.,min(1.,(valence-1.)/8.)),
            'weighting':'original_segment_duration','scope':'available_prediction_intervals'}


def map_colors(result, *, saturation_k=10., value_k=10., saturation_center=.5, value_center=.5,
               hue_gamma=2., saturation_gamma=1., value_gamma=1., use_song_mean_centers=False,
               enhance_sv=False, enhancement_strength=.5, percentile_low=10., percentile_high=90., max_gain=5.,
               saturation_range=(0.,1.), value_range=(0.,1.)):
    manual_saturation_center,manual_value_center=saturation_center,value_center
    means=song_mean_centers(result)
    statistics=adaptive_statistics(result,percentile_low,percentile_high,max_gain)
    adapt_value(.5,None,enhancement_strength)
    validate_output_range(saturation_range);validate_output_range(value_range)
    adaptation={'enabled':bool(enhance_sv),'strength':enhancement_strength,'statistics':statistics,
                'blend':'(1-strength)*normalized_score + strength*clip((normalized_score-lower_bound)/effective_span,0,1)',
                'minimum_effective_span':'1/max_gain','constant_song':'identity',
                'quantile_definition':'duration-weighted empirical inverse CDF',
                'order':'normalize -> adaptive blend -> normalized sigmoid -> gamma -> output range'}
    center_mode='manual'
    if use_song_mean_centers:
        if means['available']:
            saturation_center=means['saturation_center'];value_center=means['value_center']
            if enhance_sv:
                saturation_center=adapt_value(saturation_center,statistics['arousal'],enhancement_strength)
                value_center=adapt_value(value_center,statistics['valence'],enhancement_strength)
            center_mode='song_duration_weighted_mean'
        else:
            center_mode='manual_fallback_no_valid_predictions'
    params=dict(saturation_k=saturation_k,value_k=value_k,saturation_center=saturation_center,
                value_center=value_center,hue_gamma=hue_gamma,
                saturation_gamma=saturation_gamma,value_gamma=value_gamma,
                sv_adaptation=adaptation,saturation_range=saturation_range,value_range=value_range)
    for g in [hue_gamma,saturation_gamma,value_gamma]:gamma_transform(.5,g)
    # Validate parameters even when results contain only failed predictions.
    normalized_sigmoid(.5,saturation_k,saturation_center)
    normalized_sigmoid(.5,value_k,value_center)
    mapped={'prediction_unit':result.get('prediction_unit','bar'),
            'mapping':{'color_space':'HSV', 'tag_hues_degrees':TAG_HUES,
             'hue':'atan2(sum(weight*sin(h)), sum(weight*cos(h))) modulo 360; all valid mapped scores participate; weight=score**hue_gamma; zero scores contribute zero',
             'disabled_tags':['funny'],
             'hue_fallback':'only when all weights are zero: highest valid score among the nine mapped tags; ties use tag_hues_degrees order; unit weight even when score is zero',
             'saturation':'range_S(normalized_sigmoid(adapt_S(clip((arousal-1)/8,0,1)), saturation_k, saturation_center)**saturation_gamma)',
             'value':'range_V(normalized_sigmoid(adapt_V(clip((valence-1)/8,0,1)), value_k, value_center)**value_gamma)',
             'normalized_sigmoid':'(sigmoid(k*(x-center))-sigmoid(-k*center))/(sigmoid(k*(1-center))-sigmoid(-k*center)); k=0: x',
             'gamma_convention':'output=input**gamma; hue gamma modifies weights, not hue angles',
             'parameters':{k:v for k,v in {**params,'use_song_mean_centers':bool(use_song_mean_centers)}.items() if k!='sv_adaptation'},
             'sv_adaptation':adaptation,
             'sigmoid_center_mode':center_mode,'song_means':means,'input_va_range':[1,9], 'score_threshold_applied':False,
             'hue_threshold':None,
             'undefined_hue':'no mapped color when no valid mapped tags exist or circular vectors cancel'},
            'segments':[segment_color(segment,**params) for segment in result.get('segments',[])]}
    whole=result.get('whole_song_prediction')
    if isinstance(whole,dict):
        # Reuse the same mapping with an independent whole-song context. A
        # single global VA pair has no internal spread, so adaptive stretching
        # is identity. Auto centers come from this full-audio result, not bars.
        whole_mapped=map_colors({'segments':[{**whole,'segment':0}]},
            saturation_k=saturation_k,value_k=value_k,
            saturation_center=manual_saturation_center,value_center=manual_value_center,
            hue_gamma=hue_gamma,saturation_gamma=saturation_gamma,value_gamma=value_gamma,
            use_song_mean_centers=use_song_mean_centers,enhance_sv=enhance_sv,
            enhancement_strength=enhancement_strength,percentile_low=percentile_low,
            percentile_high=percentile_high,max_gain=max_gain,
            saturation_range=saturation_range,value_range=value_range)
        mapped['whole_song_color']=whole_mapped['segments'][0]
        mapped['whole_song_color'].update(analysis_scope='whole_track',
            prediction_method='full_audio_model_inference',
            duration_seconds=whole['end_seconds']-whole['start_seconds'])
        mapped['whole_song_mapping']=whole_mapped['mapping']
        mapped['whole_song_mapping']['statistics_scope']='whole_song_prediction_only'
        mapped['mapping']['whole_song_color']='independent full-audio prediction and independent whole-song mapping statistics; excludes local bar predictions'
    return mapped


def render_whole_song_color(mapped,result,key_prefix):
    import streamlit as st
    st.subheader('整曲色彩标签')
    st.caption('曲目：'+str(result.get('audio_name') or '未记录曲目名'))
    song=mapped.get('whole_song_color')
    if song is None:
        st.info('此结果尚无整曲预测。重新运行预测后会自动生成整曲色彩标签。')
        return
    if song['status']!='ok':
        prediction=result.get('whole_song_prediction',{})
        st.warning('整曲色彩标签暂不可用：'+str(prediction.get('error') or song['status']))
        return
    swatch,detail=st.columns([1,3])
    swatch.markdown(f'<div role="img" aria-label="整曲颜色 {song["hex"]}" style="height:105px;border-radius:12px;background:{song["hex"]};border:1px solid #80808040"></div>',unsafe_allow_html=True)
    with detail:
        st.metric('乐曲色彩标签',song['hex'])
        st.write(f'H {song["hue_degrees"]:.2f}° · S {song["saturation"]:.3f} · V {song["value"]:.3f}　｜　valence {song["valence"]:.3f} · arousal {song["arousal"]:.3f}')
    st.caption(f'完整音频 0–{song["end_seconds"]:.3f} 秒独立预测；使用当前九标签、Sigmoid、伽马及输出范围，色相不设得分阈值。整曲统计独立于小节分析范围；单个整曲预测值不作相对拉伸，自动中心使用整曲VA。')
    with st.expander('核查整曲预测与色相权重'):
        prediction=result.get('whole_song_prediction',{})
        st.write(f'输入范围：{song["start_seconds"]:.3f}–{song["end_seconds"]:.3f} 秒；整曲 valence={song["valence"]:.6f}，arousal={song["arousal"]:.6f}。')
        if prediction.get('inference_pipeline_version')=='duration_weighted_mert_v1':
            st.caption('MERT分段特征已按实际音频时长加权，包含最后不足30秒的尾段。')
            durations=prediction.get('mert_chunk_durations_seconds',[])
            if durations:
                total=math.fsum(durations)
                st.dataframe([{'MERT段':i+1,'实际时长（秒）':duration,'汇总权重':duration/total}
                              for i,duration in enumerate(durations)],hide_index=True,use_container_width=True)
        else:
            st.info('这是旧版整曲预测；重新运行预测可应用MERT尾段时长加权修正。')
        st.dataframe([{'tag':tag,'score':score,'H':TAG_HUES[tag],
                       'weight':song['hue_weights'].get(tag,0.)} for tag,score in song['tag_scores'].items()],
                     hide_index=True,use_container_width=True)
        st.write(f'色相合成向量长度 R={song["hue_resultant_length"]:.6f}；越接近0，标签色相方向越相互抵消。')
        st.json(mapped['whole_song_mapping'],expanded=False)
    label={'audio_name':result.get('audio_name'),'audio_sha256':result.get('audio_sha256'),
           'whole_song_color':song,'whole_song_prediction':result.get('whole_song_prediction'),
           'mapping':mapped['whole_song_mapping']}
    st.download_button('下载整曲色彩标签 JSON',json.dumps(label,ensure_ascii=False,indent=2,allow_nan=False),
                       file_name='song_color_label.json',mime='application/json',key=key_prefix+'_song_label')


def color_svg(mapped):
    rows=mapped['segments']
    if not rows:return ''
    start=min(r['start_seconds'] for r in rows);end=max(r['end_seconds'] for r in rows)
    span=max(end-start,1e-9);left=35;plot_width=1130
    def x(t):return left+(t-start)/span*plot_width
    svg=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 170" role="img" aria-label="音乐片段颜色时间轴" style="width:100%;min-width:650px;background:#11151c;border-radius:10px;font-family:Microsoft YaHei,Arial,sans-serif">',
         '<defs><pattern id="unmapped" width="8" height="8" patternUnits="userSpaceOnUse"><rect width="8" height="8" fill="#525965"/><path d="M-2,2 L2,-2 M0,8 L8,0 M6,10 L10,6" stroke="#929aa6" stroke-width="1"/></pattern></defs>']
    for row in rows:
        if row['status']=='ok':
            fill=row['hex']
            tip=(f"片段 {row['segment']} | {row['start_seconds']:.3f}–{row['end_seconds']:.3f}s | {fill} | "
                 f"H {row['hue_degrees']:.2f}° · S {row['saturation']:.3f} · V {row['value']:.3f} | "
                 f"valence {row['valence']:.3f} · arousal {row['arousal']:.3f}")
        else:
            fill='url(#unmapped)';tip=f"片段 {row['segment']} | {row['start_seconds']:.3f}–{row['end_seconds']:.3f}s | 未映射：{row['status']}"
        curve=mapped.get('transition')
        if row['status']=='ok' and curve and curve['enabled']:
            from color_transition import transition_color
            a,b=row['start_seconds'],row['end_seconds']
            count=max(2,math.ceil((b-a)/span*1800))
            samples=[a+(b-a)*i/count for i in range(count+1)]
            # Original starts/endpoints are included even in very long songs.
            samples=sorted(set(samples))
            if curve.get('mode')=='boundary_hold':
                # Include holds and narrow fades explicitly, independent of the
                # overview resolution and CSV sampling rate.
                for run in curve['runs']:
                    for event in run['events']:
                        if event['end_seconds']<a or event['start_seconds']>b:continue
                        samples.extend(event['start_seconds']+event['duration_seconds']*i/16 for i in range(17)
                                       if a<=event['start_seconds']+event['duration_seconds']*i/16<=b)
                samples=sorted(set(samples))
            gradient='transition_'+str(len(svg))
            stops=[]
            for t in samples:
                # Use the left limit at a rectangle's right edge, so an instant
                # change at the next boundary cannot create an early SVG fade.
                color=transition_color(curve,math.nextafter(t,a) if t==b else t)
                stops.append(f'<stop offset="{(t-a)/(b-a)*100:.8f}%" stop-color="{color["hex"] if color else row["hex"]}"/>')
            svg.append(f'<defs><linearGradient id="{gradient}" color-interpolation="sRGB">'+''.join(stops)+'</linearGradient></defs>')
            fill=f'url(#{gradient})'
        svg.append(f'<rect x="{x(row["start_seconds"]):.4f}" y="24" width="{max(0,x(row["end_seconds"])-x(row["start_seconds"])):.4f}" height="80" fill="{fill}" stroke="#c1c8d2" stroke-opacity="0.2" stroke-width="0.5"><title>{escape(tip)}</title></rect>')
    for i in range(9):
        t=start+span*i/8;px=x(t)
        svg.append(f'<line x1="{px}" y1="109" x2="{px}" y2="114" stroke="#9daab9"/><text x="{px}" y="132" fill="#c9d3df" text-anchor="middle" font-size="12">{t:.1f}s</text>')
    curve=mapped.get('transition',{})
    unit_name='节拍' if mapped.get('prediction_unit')=='beat' else '小节'
    description=(f'边界持色过渡；原{unit_name}起点准确呈现对应颜色；斜线 = 未映射' if curve.get('mode')=='boundary_hold'
                 else f'Foote指数过渡；原{unit_name}起点为颜色节点；斜线 = 未映射') if curve.get('enabled') else '色块宽度 = 片段时长；斜线 = 未映射'
    svg.append(f'<text x="35" y="155" fill="#9daab9" font-size="11">{description}</text></svg>')
    return ''.join(svg)


def color_csv(mapped):
    stream=io.StringIO(newline='')
    fields=['segment','start_seconds','end_seconds','valence','arousal','hue_degrees',
            'normalized_arousal','normalized_valence','adapted_arousal','adapted_valence','saturation_before_gamma','value_before_gamma','saturation_after_gamma','value_after_gamma','saturation','value','hex','status','va_clipped','hue_resultant_length','hue_tag_count','hue_tags_used','hue_selection','hue_fallback_tag']
    writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader()
    writer.writerows(mapped['segments'])
    return stream.getvalue().encode('utf-8-sig')


def render_color_timeline(result,key_prefix='colors',audio_data=None,audio_name=None):
    import streamlit as st
    import streamlit.components.v1 as components
    st.subheader('音乐颜色时间轴')
    unit_name='节拍' if result.get('prediction_unit')=='beat' else '小节'
    whole_panel=st.container()
    means=song_mean_centers(result)
    whole=result.get('whole_song_prediction')
    whole_means=song_mean_centers({'segments':[whole]}) if isinstance(whole,dict) else {'available':False}
    auto_centers_available=means['available'] or whole_means['available']
    with st.expander('S/V 变化增强（调整无需重新预测）',expanded=True):
        enhance_sv=st.toggle('启用歌曲内自适应增强',value=True,key=key_prefix+'_enhance_sv')
        enhancement_strength=st.slider('增强强度',0.,1.,.5,.05,disabled=not enhance_sv,key=key_prefix+'_enhance_strength',
            help='0保留原始得分范围，1完全使用歌曲内分位数范围；最大放大倍数限制微小波动的放大。')
        p_left,p_right=st.columns(2)
        percentile_low=p_left.number_input('低分位点（%）',min_value=0,max_value=49,value=10,step=1,
                                           disabled=not enhance_sv,key=key_prefix+'_p_low')
        percentile_high=p_right.number_input('高分位点（%）',min_value=51,max_value=100,value=90,step=1,
                                             disabled=not enhance_sv,key=key_prefix+'_p_high')
        max_gain=st.slider('最大放大倍数',1.,10.,5.,.5,disabled=not enhance_sv,key=key_prefix+'_max_gain')
        saturation_range=st.slider('S 输出范围',0.,1.,(.2,1.),.01,disabled=not enhance_sv,key=key_prefix+'_s_range')
        value_range=st.slider('V 输出范围',0.,1.,(.25,1.),.01,disabled=not enhance_sv,key=key_prefix+'_v_range',
                              help='适当提高下限，让暗段仍能看见颜色与饱和度差异。')
        st.caption('先按原始片段时长计算整曲分位数，再拉伸并混合；增强后更强调歌曲内部差异。部分分析仅使用已有结果范围。恒定得分不作拉伸。')
    statistics=adaptive_statistics(result,percentile_low,percentile_high,max_gain)
    if not enhance_sv:saturation_range=(0.,1.);value_range=(0.,1.)
    if enhance_sv and means['available']:
        a_stats,v_stats=statistics['arousal'],statistics['valence']
        st.caption(f"增强参考范围（原始1–9得分）：arousal {1+8*a_stats['low']:.3f}–{1+8*a_stats['high']:.3f}；valence {1+8*v_stats['low']:.3f}–{1+8*v_stats['high']:.3f}。")
    with st.expander('Sigmoid 映射参数（调整无需重新预测）'):
        use_song_means=st.toggle('用歌曲均值作为 S/V 敏感中心',value=auto_centers_available,
            disabled=not auto_centers_available,key=key_prefix+'_mean_centers',
            help='按原始片段时长加权：S用arousal均值，V用valence均值，先归一化至0–1。')
        if means['available']:
            st.caption(f"arousal均值 {means['arousal_mean']:.4f}，valence均值 {means['valence_mean']:.4f}；按当前结果覆盖范围计算，完整分析时即整首歌曲均值。")
        else:
            st.caption('片段没有有效VA，片段映射使用手动中心；有效整曲标签可独立使用整曲VA作为自动中心。')
        left,right=st.columns(2)
        with left:
            st.caption('S · 饱和度：来自 arousal')
            saturation_k=st.slider('S 斜率 k',0.,40.,0.,.5,key=key_prefix+'_s_k')
            if use_song_means and means['available']:
                saturation_center=means['saturation_center']
                effective_s_center=adapt_value(saturation_center,statistics['arousal'],enhancement_strength) if enhance_sv else saturation_center
                st.metric('S 敏感中心（自动）',f'{effective_s_center:.4f}')
            else:
                saturation_center=st.slider('S 敏感中心',0.,1.,st.session_state.get(key_prefix+'_saved_s_center',.5),.01,key=key_prefix+'_s_center')
                st.session_state[key_prefix+'_saved_s_center']=saturation_center
        with right:
            st.caption('V · 明度：来自 valence')
            value_k=st.slider('V 斜率 k',0.,40.,0.,.5,key=key_prefix+'_v_k')
            if use_song_means and means['available']:
                value_center=means['value_center']
                effective_v_center=adapt_value(value_center,statistics['valence'],enhancement_strength) if enhance_sv else value_center
                st.metric('V 敏感中心（自动）',f'{effective_v_center:.4f}')
            else:
                value_center=st.slider('V 敏感中心',0.,1.,st.session_state.get(key_prefix+'_saved_v_center',.5),.01,key=key_prefix+'_v_center')
                st.session_state[key_prefix+'_saved_v_center']=value_center
        st.caption('k越大，中心附近变化越敏感；k=0为线性。自动中心随得分使用相同的增强变换；手动中心作用于增强后的0–1尺度。关闭增强时，中心0.5对应原始情绪值5。')
    with st.expander('伽马变换参数（调整无需重新预测）',expanded=True):
        h_col,s_col,v_col=st.columns(3)
        with h_col:
            hue_gamma=st.slider('H 标签权重 γ',.1,5.,2.,.05,key=key_prefix+'_h_gamma')
        with s_col:
            saturation_gamma=st.slider('S 饱和度 γ',.1,5.,1.,.05,key=key_prefix+'_s_gamma')
        with v_col:
            value_gamma=st.slider('V 明度 γ',.1,5.,1.,.05,key=key_prefix+'_v_gamma')
        st.caption('采用 x^γ。H的γ>1更强调高分标签；S/V的γ>1降低输出，γ<1提高输出。γ=1保持原映射。')
    mapped=map_colors(result,saturation_k=saturation_k,value_k=value_k,
                      saturation_center=saturation_center,value_center=value_center,
                      hue_gamma=hue_gamma,saturation_gamma=saturation_gamma,value_gamma=value_gamma,
                      use_song_mean_centers=use_song_means,enhance_sv=enhance_sv,
                      enhancement_strength=enhancement_strength,percentile_low=percentile_low,
                      percentile_high=percentile_high,max_gain=max_gain,
                      saturation_range=saturation_range,value_range=value_range)
    with whole_panel:
        render_whole_song_color(mapped,result,key_prefix)
    from color_transition import build_transition, transition_csv
    with st.expander('颜色过渡（调整无需重新预测）'):
        transition_enabled=st.toggle('启用颜色过渡',value=True,key=key_prefix+'_transition')
        transition_mode=st.selectbox('过渡模式',['边界持色过渡','原指数连续渐变'],
                                    disabled=not transition_enabled,key=key_prefix+'_transition_mode')
        boundary_mode=transition_mode=='边界持色过渡'
        strength=st.slider('过渡对声学变化的响应强度',0.,10.,8.,.5,
                           disabled=not transition_enabled,key=key_prefix+'_transition_strength',
                           help='边界模式：值越大，越集中在强变化处缩短过渡，弱、中等变化保留较长过渡；0为线性时长响应。原模式：0为普通平滑渐变。')
        boundary_radius=.25
        max_transition_fraction=.15;boundary_statistic='mean'
        if boundary_mode:
            boundary_radius=st.number_input('边界 Foote 比较半径（秒）',min_value=0.,max_value=1.,value=.25,step=.05,
                                           disabled=not transition_enabled,key=key_prefix+'_transition_radius')
            statistic_label=st.selectbox('边界 Foote 取值',['局部平均','局部峰值'],disabled=not transition_enabled,
                                         key=key_prefix+'_transition_statistic')
            boundary_statistic='mean' if statistic_label=='局部平均' else 'peak'
            fraction_percent=st.slider(f'过渡最多占{unit_name}起点间隔（%）',10,100,15,5,
                                      disabled=not transition_enabled,key=key_prefix+'_transition_fraction')
            max_transition_fraction=fraction_percent/100.
            st.caption(f'每个原始{unit_name}起点都是固定颜色节点，到点即达到对应颜色。过渡在前后两个起点之间完成：先持色，再渐变，到下个起点结束。Foote只调节这段渐变的长短，不延后节点；无固定秒数上下限，Foote=1时在边界直接切色。')
        else:
            st.caption(f'以原始{unit_name}起点为固定颜色节点，通过指数速度场积分在相邻起点之间渐变，到点准确呈现对应颜色。')
    mapped['transition']=build_transition(mapped,result,strength,transition_enabled,
                        mode='boundary_hold' if boundary_mode else 'foote_exponential',
                        boundary_radius=boundary_radius,
                        max_transition_fraction=max_transition_fraction,boundary_statistic=boundary_statistic)
    curve=mapped['transition']
    if transition_enabled and curve['mode']=='smooth_fallback_no_foote':
        st.caption('该结果未包含有效Foote曲线，暂用普通平滑渐变；新预测的完整结果会保存Foote曲线。')
    if transition_enabled and boundary_mode:
        events=[event for run in curve['runs'] for event in run['events']]
        if any(not event['novelty_available'] for event in events):
            st.caption(f'部分边界没有可用Foote曲线，采用相邻{unit_name}起点间隔计算过渡，并遵守最大占比。')
        if events:
            lengths=sorted(event['duration_seconds'] for event in events)
            median=lengths[len(lengths)//2] if len(lengths)%2 else (lengths[len(lengths)//2-1]+lengths[len(lengths)//2])/2
            st.caption(f"实际过渡时长：最短 {lengths[0]:.3f}秒，中位数 {median:.3f}秒，最长 {lengths[-1]:.3f}秒；{sum(event['duration_capped'] for event in events)}/{len(events)} 次受{unit_name}起点间隔限制。所有过渡在原始边界结束。")
        with st.expander('查看边界与实际过渡时长'):
            if events:
                st.dataframe([{k:event[k] for k in ['boundary_seconds','from_segment','to_segment','novelty','novelty_mean','novelty_peak',
                              'start_seconds','end_seconds','requested_duration_seconds','duration_seconds','duration_capped','anchor_time_error_seconds']}
                              for event in events],hide_index=True,use_container_width=True)
            else:st.caption('没有相邻且颜色有效的片段，无需过渡。')
    svg=color_svg(mapped)
    if not svg:
        st.info('暂无片段结果。');return
    if audio_data is None:
        playback=st.file_uploader('选择与结果对应的音乐（仅用于同步播放，无需重新预测）',
                    type=['mp3','wav','flac','ogg','m4a'],key=key_prefix+'_audio')
        if playback is not None:
            audio_data=playback.getvalue();audio_name=playback.name
    if audio_data is not None:
        from color_player import build_color_player
        components.html(build_color_player(mapped,audio_data,audio_name or 'audio.mp3'),height=455,scrolling=True)
    else:
        st.caption('选择对应音乐后，这里会显示随播放时间变化的颜色块。')
    components.html('<!doctype html><meta charset="utf-8"><style>body{margin:0}div{overflow-x:auto}</style><div>'+svg+'</div>',height=210,scrolling=True)
    st.caption('funny已禁用。H使用指定九标签（含energetic，H=10°）的全部有效得分：权重伽马 → 圆周平均，不设0.3筛选阈值；零分贡献为0，权重全为0时取最高分有效标签的色相。S/V：归一化 → 自适应增强 → Sigmoid → 伽马 → 输出范围。')
    failed=sum(row['status']!='ok' for row in mapped['segments'])
    if failed:
        st.caption(f'{failed} 个片段未映射，以斜线标记：预测失败、没有有效标签得分或色相向量抵消。')
    a,b,c=st.columns(3)
    a.download_button('下载颜色条 SVG',svg,file_name='music_colors.svg',mime='image/svg+xml',key=key_prefix+'_svg')
    b.download_button('下载 HSV / HEX CSV',color_csv(mapped),file_name='music_colors.csv',mime='text/csv',key=key_prefix+'_csv')
    c.download_button('下载颜色 JSON',json.dumps(mapped,ensure_ascii=False,indent=2,allow_nan=False),file_name='music_colors.json',mime='application/json',key=key_prefix+'_json')
    st.download_button(f'下载连续颜色 CSV（含原始{unit_name}起点）',transition_csv(mapped),
                       file_name='music_color_transition.csv',mime='text/csv',key=key_prefix+'_continuous_csv')
    with st.expander('颜色数值与映射规则'):
        st.dataframe([{k:v for k,v in row.items() if k not in ['tag_scores','hue_weights','rgb']} for row in mapped['segments']],hide_index=True,use_container_width=True)
        st.write('先把情绪值1–9归一化至0–1，按歌曲分位数自适应增强，再用Sigmoid和伽马映射，最后缩放到各自输出范围。')
        st.write('f(x)=[σ(k(x−c))−σ(−kc)]/[σ(k(1−c))−σ(−kc)]。')
        st.write('自动中心使用原始片段时长加权的有效arousal/valence均值，归一化后应用同一增强变换；失败段不参与统计。')
        st.json(mapped['mapping']['sv_adaptation'],expanded=False)
        st.write('Sigmoid之后，S=Smin+(Smax−Smin)×S₀^γS，V=Vmin+(Vmax−Vmin)×V₀^γV。funny完全禁用，色相使用指定9标签的全部有效得分，权重w=p^γH；权重全为0时取最高分有效标签色相。不对色相角本身作伽马。')
        st.write('H = atan2(Σ wᵢ sin hᵢ, Σ wᵢ cos hᵢ)，转换为0–360°。')
        st.dataframe([{'tag':tag,'色相（°）':h} for tag,h in TAG_HUES.items()],hide_index=True,use_container_width=True)
