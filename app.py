from pathlib import Path
import hashlib
import io
import json
import tempfile
import traceback
import zipfile
import altair as alt
import pandas as pd
import soundfile as sf
import streamlit as st
from backend import Tester,missing_files,ZH
from beat_core import get_tracker,load_audio
from segmented import select_audio
from bar_context import make_bar_plan,predict_bar_plan,csv_bytes
from tag_charts import render_tag_timelines, load_timeline_json
from hsv_timeline import render_color_timeline
from tag_scores import correct_result

st.set_page_config(page_title='Music2Emo · 节拍/小节情绪',page_icon='🎵',layout='wide')
st.title('Music2Emo · funny 已禁用')
st.caption('Beat This! 小节或节拍 → 自适应上下文 → 情绪时间轴与整曲色彩标签 · 原始起点对齐')
tag_panel=st.container()

@st.cache_resource(show_spinner=False)
def tracker_resource(model,device):
    return get_tracker(model,device)[0]
@st.cache_resource(show_spinner=False)
def emotion_resource(device):
    return Tester(device)
@st.cache_data(show_spinner=False,max_entries=2)
def decode(data,suffix):
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/('input'+suffix);path.write_bytes(data)
        audio,sr=load_audio(path)
    return audio.mean(axis=1),sr

def audio_bytes(y,sr):
    out=io.BytesIO();sf.write(out,y,sr,format='WAV',subtype='PCM_16');return out.getvalue()

def export_windows(y,sr,plan):
    out=io.BytesIO();offset=plan['offset_sample']
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('plan.json',json.dumps(plan,ensure_ascii=False,indent=2))
        for r in plan['segment_rows']:
            for folder,left,right in [('targets',r['start_sample'],r['end_sample']),
                    ('prediction_windows',r['prediction_start_sample'],r['prediction_end_sample'])]:
                archive.writestr(f"{folder}/{r['segment']:04d}.wav",audio_bytes(y[left-offset:right-offset],sr))
    return out.getvalue()

with st.sidebar:
    device=st.selectbox('计算设备',['cpu','cuda'],index=1)
    if st.button('检测 CUDA 兼容性'):
        from gpu_compat import isolated_probe
        with st.spinner('正在独立进程检测 CUDA 内核…'):
            gpu_report=isolated_probe()
        if gpu_report.get('ok'):st.success('CUDA实际运算自检通过')
        else:st.error('CUDA自检失败，请关闭程序后运行 repair_cuda.bat')
        st.json(gpu_report,expanded=False)
    beat_model=st.selectbox('Beat This! 模型',['small0','final0'],index=1)
    st.caption('small0 已附带；final0 首次使用需下载权重。')
    per_beat=st.toggle('按节拍预测（较慢）',value=False,key='predict_per_beat',
        help='关闭时按小节；开启时以相邻节拍为目标段，计算次数通常更多。模型仍使用扩展至至少3秒的上下文。')
    prediction_unit='beat' if per_beat else 'bar'
    unit_name='节拍' if per_beat else '小节'
    start=st.number_input('开始位置（秒）',min_value=0.,value=0.,step=1.)
    seconds=st.number_input('分析时长（0为到结尾）',min_value=0.,value=0.,step=5.)
    target=st.number_input('预测目标总时长 a（秒）',min_value=3.,value=6.,step=.5)
    boundary=st.slider('强边界阈值',.05,1.,.7,.05)
    context=st.number_input('Foote比较上下文（秒）',min_value=.2,value=2.,step=.2)
    compare_raw=st.checkbox('同时预测未扩展目标段（至少3秒）',value=True)
    st.caption(f'超过a秒的{unit_name}先强制切分为不超过a秒的目标段，优先在检测拍上切；不足3秒强制扩展。预测目标总长a秒，达到3秒后尊重强边界，可能提前停止。')
    st.caption('第一小节前不足1秒的片段采用第一小节预测（逐拍时采用该小节首拍）。整曲另用完整音频预测一次，不受选段范围及a秒上限限制。')
    threshold=st.slider('标签显示阈值',0.,1.,.5,.01)
missing=missing_files()
if missing:
    st.info('可先预览小节与预测范围；情绪预测需要现有 Music2Emo 权重。')
    with st.expander('缺少的模型文件'):st.code('\n'.join(missing))
upload=st.file_uploader('选择音乐',type=['mp3','wav','flac','ogg','m4a'])
audio_preview=st.empty()
identity=None
if upload:
    data=upload.getvalue();audio_preview.audio(data)
    identity=(hashlib.sha256(data).hexdigest(),start,seconds,device,beat_model,target,boundary,context,compare_raw,prediction_unit,'total_seconds_bar_or_beat_v4')
col1,col2=st.columns(2)
preview=col1.button(f'预览{unit_name}与扩展范围',disabled=not upload,key='preview_targets')
predict=col2.button(f'逐{unit_name}预测',type='primary',disabled=not upload or bool(missing),key='predict_targets')
if preview or predict:
    try:
        with st.spinner(f'检测{unit_name}并计算整曲声学变化…'):
            full,sr=decode(data,Path(upload.name).suffix)
            y,offset=select_audio(full,sr,start,seconds)
            plan=make_bar_plan(y,sr,tracker_resource(beat_model,device),target_seconds=target,
                    boundary_threshold=boundary,context_seconds=context,offset_sample=offset,
                    model=beat_model,device=device,prediction_unit=prediction_unit)
        st.session_state['bar_plan']=plan;st.session_state['bar_plan_id']=identity
        st.session_state['selected_audio']=y
        if predict:
            with st.spinner('加载情绪模型…'):tester=emotion_resource(device)
            bar=st.progress(0.)
            def report(done,total,row):
                bar.progress(done/total,text='整曲预测完成' if row.get('kind')=='whole_song' else f'已处理{unit_name}目标 {done}/{total-1}；随后预测整曲')
            result=predict_bar_plan(y,sr,plan,tester,threshold,report,compare_raw,whole_audio=full)
            result['audio_name']=upload.name
            result['audio_sha256']=identity[0]
            st.session_state['bar_result']=result;st.session_state['bar_result_id']=identity
            st.session_state['bar_result_audio']={'data':data,'name':upload.name}
            bar.empty()
    except Exception as exc:
        st.error(str(exc))
        with st.expander('详细错误'):st.code(traceback.format_exc())
with tag_panel:
    imported=st.file_uploader('已有预测结果？导入完整结果 JSON 查看颜色与标签',type=['json'],key='tag_result_import')
    chart_result=st.session_state.get('bar_result')
    playback_audio=st.session_state.get('bar_result_audio')
    if upload is not None:
        result_id=st.session_state.get('bar_result_id')
        if imported is not None or (result_id and result_id[0]==hashlib.sha256(data).hexdigest()):
            playback_audio={'data':data,'name':upload.name}
    if imported is not None:
        try:
            chart_result=load_timeline_json(imported.getvalue())
            if upload is None or (chart_result.get('audio_name') and chart_result['audio_name']!=upload.name):
                playback_audio=None
            elif chart_result.get('audio_sha256') and chart_result['audio_sha256']!=hashlib.sha256(data).hexdigest():
                playback_audio=None
            st.caption('正在显示导入的结果：'+imported.name+'；清除该文件可返回本次预测。')
        except (ValueError,TypeError,KeyError) as exc:
            st.error(str(exc))
    if chart_result is not None:
        chart_result=correct_result(chart_result)
        if chart_result.get('beat_analysis'):
            from spectra_prediction import make_spectra_bundle
            st.download_button('导出 SPECTRA 预测 JSON（当前默认配色）',
                json.dumps(make_spectra_bundle(chart_result),ensure_ascii=False,allow_nan=False),
                'prediction.spectra.json','application/json',key='spectra_export')
        result_unit=chart_result.get('prediction_unit','bar')
        st.caption('当前结果单位：'+('节拍' if result_unit=='beat' else '小节'))
        if imported is None and result_unit!=prediction_unit:
            st.info('预测单位开关已改变；点击新的预测按钮后更新结果。当前仍显示上次结果。')
        if upload is not None and chart_result.get('audio_sha256') and chart_result['audio_sha256']!=hashlib.sha256(data).hexdigest():
            st.warning('当前显示的是“'+str(chart_result.get('audio_name') or '上次音乐')+'”的预测结果，与新选择的音乐不同；请重新预测或导入对应结果。')
        audio_preview.empty()
        render_color_timeline(chart_result,audio_data=playback_audio['data'] if playback_audio else None,
                              audio_name=playback_audio['name'] if playback_audio else None)
        render_tag_timelines(chart_result)
    else:
        st.subheader('音乐颜色时间轴 / 九标签热力图 · funny 已禁用')
        st.info(f'完成“逐{unit_name}预测”后这里会显示颜色条与热力图。也可以在上方导入之前导出的完整结果 JSON，无需重新分析。')

if 'bar_plan' in st.session_state:
    plan=st.session_state['bar_plan'];rows=plan['segment_rows']
    plan_unit='节拍' if plan.get('prediction_unit','bar')=='beat' else '小节'
    if identity!=st.session_state['bar_plan_id']:st.info('预览来自上次分析，当前设置或音乐已改变。')
    for warning in plan['beat_analysis']['warnings']:st.warning(warning)
    forced=sum(r['forced_expansion'] for r in rows)
    a,b,c=st.columns(3)
    if plan_unit=='节拍':a.metric('检测节拍数',sum(t<plan['num_samples'] for t in plan['beat_analysis']['beat_samples']))
    else:a.metric('完整小节',plan['beat_analysis']['bar_count'])
    b.metric('含首尾的目标数',len(rows));c.metric('强制跨边界',forced)
    split_summary=plan.get('split_summary',{})
    if split_summary.get('long_source_count'):
        st.caption(f"{split_summary['long_source_count']} 个超长原始段已强制切分，新增 {split_summary['added_target_count']} 个目标段；每个目标段不超过a秒，原始小节编号保留。")
    if any(r.get('independent_prediction_skipped') for r in rows):
        st.caption('开头片段不足1秒，已忽略其独立预测并采用第一小节的结果（逐拍时采用该小节首拍）；保留原时间范围，开头即可显示同一颜色。')
    with st.expander(f'{plan_unit}与实际预测范围',expanded=False):
        fields=['segment','kind','prediction_unit','bar_number','beat_number','beat_in_bar',
                'source_segment_index','forced_split','split_part','split_count','split_start_reason',
                'prediction_source','inherited_from_segment','independent_prediction_skipped',
                'start_seconds','end_seconds','duration_seconds',
                'prediction_start_seconds','prediction_end_seconds','prediction_duration_seconds',
                'requested_target_seconds','target_duration_seconds','added_duration_seconds',
                'original_beat_count','added_beat_count','prediction_beat_count','beat_count_source',
                'forced_expansion','stop_reason']
        st.dataframe(pd.DataFrame(rows).reindex(columns=fields),hide_index=True,use_container_width=True)
        diagnostic=plan['diagnostic'];stride=max(1,len(diagnostic['time_s'])//2000)
        st.line_chart(pd.DataFrame({'时间（秒）':[t+plan['offset_sample']/plan['sample_rate'] for t in diagnostic['time_s'][::stride]],
                      'Foote变化':diagnostic['novelty'][::stride],'强边界阈值':diagnostic['threshold'][::stride]}),x='时间（秒）')
        chosen=st.selectbox('试听目标与上下文',range(len(rows)),format_func=lambda i:f"{rows[i]['label']} · {rows[i]['start_seconds']:.2f}–{rows[i]['end_seconds']:.2f}s")
        row=rows[chosen];y=st.session_state['selected_audio'];sr=plan['sample_rate'];offset=plan['offset_sample']
        l,r=st.columns(2)
        l.caption('未扩展目标段');l.audio(audio_bytes(y[row['start_sample']-offset:row['end_sample']-offset],sr))
        r.caption('实际模型输入');r.audio(audio_bytes(y[row['prediction_start_sample']-offset:row['prediction_end_sample']-offset],sr))
        if row.get('independent_prediction_skipped'):st.caption('本开头片段不单独推理；右侧是第一小节或其首拍采用的模型输入。')
        if row['forced_expansion']:st.caption('为满足3秒最短输入，本段跨过了强边界。')
        file_prefix='beat' if plan_unit=='节拍' else 'bar'
        st.download_button('下载范围 JSON',json.dumps(plan,ensure_ascii=False,indent=2),file_name=file_prefix+'_context_plan.json')
        if st.button(f'准备导出{plan_unit}与预测音频 ZIP',key='prepare_window_zip'):
            st.session_state['window_zip']=export_windows(y,sr,plan)
            st.session_state['window_zip_id']=st.session_state['bar_plan_id']
        if st.session_state.get('window_zip_id')==st.session_state['bar_plan_id']:
            st.download_button('下载音频 ZIP',st.session_state['window_zip'],file_name=file_prefix+'_context_audio.zip')
if 'bar_result' in st.session_state:
    result=correct_result(st.session_state['bar_result'])
    if identity!=st.session_state['bar_result_id']:st.info('预测来自上次分析，请重新预测以更新。')
    st.caption(f"成功 {result['success_count']}/{result['segment_count']} · {result['elapsed_seconds']:.1f}s · 实际推理 {result['unique_inferences']} 次")
    timeline=[]
    for row in result['segments']:
        if row['status']=='ok':
            for dimension in ['valence','arousal']:
                timeline.append({'start':row['start_seconds'],'end':row['end_seconds'],
                    'segment':row['segment'],'dimension':dimension,'value':row[dimension],
                    'input_start':row['prediction_start_seconds'],'input_end':row['prediction_end_seconds']})
    if timeline:
        chart=alt.Chart(pd.DataFrame(timeline)).mark_rule(strokeWidth=4).encode(
            x=alt.X('start:Q',title='时间（秒）'),x2='end:Q',y=alt.Y('value:Q',title='情绪数值'),
            color='dimension:N',tooltip=['segment','start','end','input_start','input_end','dimension','value']).properties(height=260)
        st.altair_chart(chart,use_container_width=True)
    result_unit='节拍' if result.get('prediction_unit','bar')=='beat' else '小节'
    st.caption(f'当前按{result_unit}显示实际目标段；数值来自扩展上下文，是对目标段的近似。超长段的子段保留源编号，第一小节前不足1秒的片段继承第一小节或其首拍的结果。')
    overview=[]
    for row in result['segments']:
        raw=row['raw_prediction'];overview.append({k:row.get(k) for k in ['segment','start_seconds','end_seconds','valence','arousal','forced_expansion','status']}|{'原始预测状态':raw['status'],'原始valence':raw.get('valence'),'原始arousal':raw.get('arousal')})
    st.dataframe(pd.DataFrame(overview),hide_index=True,use_container_width=True)
    index=st.selectbox('查看目标的标签与对比',range(len(result['segments'])),format_func=lambda i:f"第{i+1}段",key='prediction_choice')
    row=result['segments'][index]
    if row['status']=='error':
        st.error(row['error'])
        if row.get('error_traceback'):
            with st.expander('本段详细错误'):st.code(row['error_traceback'])
    else:
        tags=[s for s in row['scores'] if s['score']>threshold]
        st.write('、'.join(s['中文'] for s in tags) or '没有超过显示阈值的标签')
        st.dataframe(pd.DataFrame(row['scores']),hide_index=True,use_container_width=True)
    st.json({'raw_prediction':row['raw_prediction'],'forced_boundary_crossings':row['forced_boundary_crossings']},expanded=False)
    a,b=st.columns(2)
    file_prefix='beat' if result_unit=='节拍' else 'bar'
    a.download_button('下载全部情绪 CSV',csv_bytes(result),file_name=file_prefix+'_emotions.csv',mime='text/csv')
    b.download_button('下载完整结果 JSON',json.dumps(result,ensure_ascii=False,indent=2),file_name=file_prefix+'_emotions.json')
