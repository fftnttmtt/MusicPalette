from pathlib import Path
import hashlib
import io
import json
import traceback
import altair as alt
import pandas as pd
import soundfile as sf
import streamlit as st
from backend import Tester, missing_files, csv_bytes, ZH
from tag_charts import render_tag_timelines
from hsv_timeline import render_color_timeline
from segmented import select_audio, make_plan, predict_plan, wide_csv_bytes

st.set_page_config(page_title='Music2Emo · 分段测试', page_icon='🎵', layout='wide')
st.title('Music2Emo')
st.caption('声学自动分段 → 逐段情绪预测')

@st.cache_resource(show_spinner=False)
def load_tester(device):
    return Tester(device)

@st.cache_data(show_spinner=False, max_entries=2)
def decode_audio(data):
    audio, sr = sf.read(io.BytesIO(data), dtype='float32', always_2d=True)
    return audio.mean(axis=1), sr

with st.sidebar:
    st.subheader('分析范围')
    device = st.selectbox('计算设备', ['cpu', 'cuda'], index=1, key='device_choice', format_func=lambda x: 'CPU' if x == 'cpu' else 'NVIDIA GPU（CUDA）')
    start = st.number_input('开始位置（秒）', min_value=0.0, value=0.0, step=1.0)
    seconds = st.number_input('分析时长（秒，0表示到结尾）', min_value=0.0, value=0.0, step=5.0)
    st.subheader('声学分段')
    minimum = st.number_input('最短段长（秒）', min_value=0.1, value=1.0, step=0.1)
    limit_enabled = st.checkbox('限制最长段长', value=True)
    maximum = st.number_input('最长段长（秒）', min_value=0.2, value=10.0, step=1.0, disabled=not limit_enabled)
    sensitivity = st.slider('声学变化灵敏度', 0.0, 1.0, 0.8, 0.05)
    snap = st.checkbox('边界对齐邻近节拍', value=True)
    st.caption('按上传代码的MFCC、色度与能量变化选取边界。最长段长至少为最短段长的2倍。')
    st.subheader('结果显示')
    threshold = st.slider('标签显示阈值', 0.0, 1.0, 0.5, 0.01, key='score_threshold')
    top_n = st.slider('单段图表标签数量', 5, 55, 20)

config = dict(min_duration=minimum, max_duration=maximum if limit_enabled else None,
              sensitivity=sensitivity, snap_to_beat=snap)
missing = missing_files()
if missing:
    st.warning('缺少模型文件时仍可预览分段；预测前请准备权重。')
    with st.expander('缺少的文件'):
        st.code('\n'.join(missing))
upload = st.file_uploader('选择音乐', type=['mp3', 'wav', 'flac', 'ogg'])
identity = None
if upload:
    audio_bytes = upload.getvalue()
    st.audio(audio_bytes)
    identity = (hashlib.sha256(audio_bytes).hexdigest(), start, seconds, device,
                minimum, config['max_duration'], sensitivity, snap)

left, right = st.columns(2)
preview = left.button('预览分段', disabled=upload is None)
predict = right.button('分段并预测', type='primary', disabled=upload is None or bool(missing))
if preview or predict:
    try:
        with st.spinner('正在分析声学变化并选取分段边界…'):
            full_audio, sr = decode_audio(audio_bytes)
            selected, offset = select_audio(full_audio, sr, start, seconds)
            plan = make_plan(selected, sr, config, offset)
        st.session_state['plan'] = plan
        st.session_state['plan_identity'] = identity
        if predict:
            with st.spinner('正在加载模型…'):
                tester = load_tester(device)
            bar = st.progress(0.0, text=f"准备分析 {len(plan['spans'])} 段…")
            def report(done, total, row):
                bar.progress(done / total, text=f"已处理 {done}/{total} 段 · {row['start_seconds']:.2f}–{row['end_seconds']:.2f}s")
            result = predict_plan(selected, sr, plan, tester, threshold, report)
            result['audio_name'] = upload.name
            result['device'] = device
            st.session_state['result'] = result
            st.session_state['identity'] = identity
            bar.empty()
    except Exception as exc:
        st.error(str(exc))
        with st.expander('详细错误'):
            st.code(traceback.format_exc())

if 'plan' in st.session_state:
    plan = st.session_state['plan']
    if identity != st.session_state['plan_identity']:
        st.info('分段预览来自上一次分析；当前音乐或设置已改变。')
    with st.expander(f"分段预览 · {len(plan['spans'])} 段", expanded='result' not in st.session_state):
        st.dataframe(pd.DataFrame(plan['segment_rows']), hide_index=True, use_container_width=True)
        diagnostic = plan['diagnostic']
        stride = max(1, len(diagnostic['time_s']) // 2000)
        offset_s = plan['offset_sample'] / plan['sample_rate']
        curve = pd.DataFrame({'时间（秒）': [v + offset_s for v in diagnostic['time_s'][::stride]],
                              '变化强度': diagnostic['novelty'][::stride],
                              '边界阈值': diagnostic['threshold'][::stride]})
        st.line_chart(curve, x='时间（秒）', y=['变化强度', '边界阈值'])
        st.download_button('下载分段边界 JSON', json.dumps(plan, ensure_ascii=False, indent=2),
                           file_name='music2emo_segments.json', mime='application/json')

if 'result' in st.session_state:
    from tag_scores import correct_result
    stored = st.session_state['result']
    result = {**stored, 'threshold': threshold,
              'segments': [{**r, 'predicted_tags': [s['tag'] for s in r['scores'] if s['score'] > threshold]}
                           for r in stored['segments']]}
    result=correct_result(result)
    if identity != st.session_state['identity']:
        st.info('以下预测来自上一次分析。当前音乐或分段设置已改变，点击“分段并预测”更新。')
    render_color_timeline(result, key_prefix='acoustic_colors')
    render_tag_timelines(result, key_prefix='acoustic')
    a, b, c = st.columns(3)
    a.metric('总段数', result['segment_count'])
    b.metric('预测成功', f"{result['success_count']}/{result['segment_count']}")
    c.metric('逐段预测耗时', f"{result['elapsed_seconds']:.1f} 秒")
    st.caption(f"{result['audio_name']} · 每段独立预测 · 调整显示阈值不会重新推理")
    if result['success_count'] != result['segment_count']:
        st.warning('部分段落预测失败。失败段保留在表格与导出文件中，可查看详细错误。')
    timeline = []
    for row in result['segments']:
        if row['status'] == 'ok':
            for dimension in ['valence', 'arousal']:
                timeline.append({'segment': row['segment'], 'start': row['start_seconds'],
                                 'end': row['end_seconds'], 'dimension': dimension,
                                 'value': row[dimension]})
    if timeline:
        st.subheader('情绪时间轴')
        chart = alt.Chart(pd.DataFrame(timeline)).mark_rule(strokeWidth=4).encode(
            x=alt.X('start:Q', title='时间（秒）'), x2='end:Q',
            y=alt.Y('value:Q', title='情绪数值'), color=alt.Color('dimension:N', title=None),
            tooltip=['segment', 'start', 'end', 'dimension', alt.Tooltip('value:Q', format='.3f')]
        ).properties(height=250)
        st.altair_chart(chart, use_container_width=True)
        st.caption('每条水平线对应一个分段的预测；失败段留空。模型没有预测段内的连续变化。')
        selected_tags = st.multiselect('查看标签随时间的变化', [t for t in ZH if t!='funny'],
            default=['happy', 'sad', 'relaxing', 'energetic', 'dark', 'romantic'],
            format_func=lambda tag: f'{ZH[tag]} · {tag}')
        heat = []
        for row in result['segments']:
            if row['status'] == 'ok':
                for score in row['scores']:
                    if score['tag'] in selected_tags:
                        heat.append({'segment': row['segment'], 'start': row['start_seconds'],
                                     'end': row['end_seconds'], 'label': score['中文'] + ' · ' + score['tag'],
                                     'score': score['score']})
        if heat:
            chart = alt.Chart(pd.DataFrame(heat)).mark_rect().encode(
                x=alt.X('start:Q', title='时间（秒）'), x2='end:Q',
                y=alt.Y('label:N', title=None),
                color=alt.Color('score:Q', scale=alt.Scale(domain=[0, 1], scheme='viridis'), title='标签分数'),
                tooltip=['segment', 'start', 'end', 'label', alt.Tooltip('score:Q', format='.4f')]
            ).properties(height=max(160, 25 * len(selected_tags)))
            st.altair_chart(chart, use_container_width=True)
    overview = []
    for row in result['segments']:
        strongest = row['scores'][0] if row['scores'] else None
        overview.append({'段落': row['segment'], '开始秒': row['start_seconds'], '结束秒': row['end_seconds'],
                         '时长秒': row['duration_seconds'], 'Valence': row['valence'], 'Arousal': row['arousal'],
                         '最高分标签': strongest['中文'] + ' · ' + strongest['tag'] if strongest else '',
                         '最高分': strongest['score'] if strongest else None, '状态': row['status']})
    st.subheader('逐段结果')
    st.dataframe(pd.DataFrame(overview), hide_index=True, use_container_width=True)
    chosen = st.selectbox('查看某一段的全部55个启用标签', range(len(result['segments'])),
        key='segment_choice',
        format_func=lambda index: f"第{index + 1}段 · {result['segments'][index]['start_seconds']:.2f}–{result['segments'][index]['end_seconds']:.2f}s")
    row = result['segments'][chosen]
    if identity == st.session_state['identity'] and upload:
        full_audio, sr = decode_audio(audio_bytes)
        clip = full_audio[row['start_sample']:row['end_sample']]
        buffer = io.BytesIO()
        sf.write(buffer, clip, sr, format='WAV', subtype='PCM_16')
        st.audio(buffer.getvalue(), format='audio/wav')
    if row['status'] == 'error':
        st.error(row['error'])
        with st.expander('该段详细错误'):
            st.code(row.get('error_traceback', row['error']))
    else:
        tags = [f"{s['中文']}（{s['tag']}）" for s in row['scores'] if s['score'] > threshold]
        st.write('超过阈值的标签：' + ('、'.join(tags) if tags else '无'))
        if row['padding_seconds']:
            st.caption(f"原片段短于1秒，仅模型输入补零 {row['padding_seconds']:.3f} 秒；分段边界与试听保留原音频。")
        scores = pd.DataFrame(row['scores'])
        plot = scores.head(top_n).copy()
        plot['label'] = plot['中文'] + ' · ' + plot['tag']
        chart = alt.Chart(plot).mark_bar(color='#7779ed').encode(
            x=alt.X('score:Q', scale=alt.Scale(domain=[0, 1]), title='独立标签分数'),
            y=alt.Y('label:N', sort='-x', title=None),
            tooltip=['tag', '中文', alt.Tooltip('score:Q', format='.4f')]
        ).properties(height=max(250, top_n * 22))
        st.altair_chart(chart, use_container_width=True)
        st.dataframe(scores, hide_index=True, use_container_width=True,
                     column_config={'score': st.column_config.ProgressColumn('标签分数', min_value=0, max_value=1, format='%.4f')})
        st.download_button('下载当前段分数 CSV', csv_bytes(row['scores']),
                           file_name=f"music2emo_segment_{row['segment']:03d}.csv", mime='text/csv')
    st.caption('各标签分数独立，不是情感成分占比，总和不要求等于100%。短片段的情绪结果应结合试听判断。')
    d1, d2 = st.columns(2)
    d1.download_button('下载全部段落 CSV', wide_csv_bytes(result), file_name='music2emo_timeline.csv', mime='text/csv')
    d2.download_button('下载完整结果 JSON', json.dumps(result, ensure_ascii=False, indent=2),
                       file_name='music2emo_timeline.json', mime='application/json')
