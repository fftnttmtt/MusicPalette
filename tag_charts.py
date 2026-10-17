"""Music2Emo tag timelines use target spans and corrected independent scores."""
import math
from tag_scores import corrected_score_items, correct_result

TIMELINE_TAGS = ['calm', 'romantic', 'dream', 'sad', 'epic', 'happy', 'love', 'nature', 'energetic']
LABELS = {'calm': '平静', 'romantic': '浪漫', 'dream': '梦幻', 'sad': '悲伤',
          'epic': '史诗', 'happy': '快乐', 'love': '爱', 'nature': '自然', 'energetic': '活力充沛'}


def tag_timeline_rows(result, tags=TIMELINE_TAGS):
    rows = []
    for segment in result.get('segments', []):
        scores = {item['tag']: item.get('score') for item in corrected_score_items(segment.get('scores', []))}
        for tag in tags:
            value = scores.get(tag) if segment.get('status') == 'ok' else None
            valid = isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 1
            rows.append({'segment': segment['segment'], 'tag': tag,
                         'label': f'{tag} · {LABELS.get(tag, tag)}',
                         'start': segment['start_seconds'], 'end': segment['end_seconds'],
                         'score': float(value) if valid else None,
                         'status': '成功' if valid else ('预测失败' if segment.get('status') != 'ok' else '缺少分数'),
                         'input_start': segment.get('prediction_start_seconds', segment['start_seconds']),
                         'input_end': segment.get('prediction_end_seconds', segment['end_seconds'])})
    return rows


def tag_heatmap(rows, tags=TIMELINE_TAGS):
    import altair as alt
    labels = [f'{tag} · {LABELS.get(tag, tag)}' for tag in tags]
    base = alt.Chart(alt.InlineData(values=rows)).encode(
        x=alt.X('start:Q', title='时间（秒）', scale=alt.Scale(zero=False)), x2='end:Q',
        y=alt.Y('label:N', title=None, sort=labels, axis=alt.Axis(labelLimit=180)),
        tooltip=[alt.Tooltip('tag:N', title='标签'), alt.Tooltip('segment:Q', title='片段'),
                 alt.Tooltip('start:Q', title='片段开始（秒）', format='.3f'),
                 alt.Tooltip('end:Q', title='片段结束（秒）', format='.3f'),
                 alt.Tooltip('score:Q', title='得分', format='.4f'),
                 alt.Tooltip('status:N', title='状态'),
                 alt.Tooltip('input_start:Q', title='模型输入开始（秒）', format='.3f'),
                 alt.Tooltip('input_end:Q', title='模型输入结束（秒）', format='.3f')]
    )
    # Separate background prevents Vega's quantitative invalid-value filter from
    # dropping missing-score rectangles before the fallback color is evaluated.
    background = base.mark_rect(color='#444952')
    scored = base.transform_filter('isValid(datum.score)').mark_rect().encode(
        color=alt.Color('score:Q', title='标签得分', scale=alt.Scale(domain=[0, 1], scheme='viridis'),
                        legend=alt.Legend(format='.1f')))
    return alt.layer(background, scored).properties(height=max(120, 35 * len(tags)))


def tag_trends(rows, tags=TIMELINE_TAGS):
    import altair as alt
    points = []
    for tag in tags:
        previous_end = None; run = 0
        for row in (r for r in rows if r['tag'] == tag):
            if row['score'] is None:
                previous_end = None; continue
            if previous_end is None or abs(row['start'] - previous_end) > 1e-6:
                run += 1
            for time in [row['start'], row['end']]:
                points.append({**row, 'time': time, 'run': run, 'sequence': len(points)})
            previous_end = row['end']
    labels = [f'{tag} · {LABELS.get(tag, tag)}' for tag in tags]
    return alt.Chart(alt.InlineData(values=points)).mark_line(interpolate='linear', strokeWidth=2).encode(
        x=alt.X('time:Q', title='时间（秒）', scale=alt.Scale(zero=False)),
        y=alt.Y('score:Q', title=None, scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(values=[0, .5, 1])),
        color=alt.Color('tag:N', sort=list(tags), scale=alt.Scale(scheme='tableau10'), legend=None),
        detail='run:N', order=alt.Order('sequence:Q'),
        tooltip=[alt.Tooltip('tag:N', title='标签'), alt.Tooltip('segment:Q', title='片段'),
                 alt.Tooltip('start:Q', title='开始（秒）', format='.3f'),
                 alt.Tooltip('end:Q', title='结束（秒）', format='.3f'),
                 alt.Tooltip('score:Q', title='得分', format='.4f')]
    ).properties(height=65).facet(
        row=alt.Row('label:N', title=None, sort=labels,
                    header=alt.Header(labelAngle=0, labelAlign='left', labelLimit=160)),
        spacing=8).resolve_scale(x='shared', y='shared')


def render_tag_timelines(result, key_prefix="timeline"):
    import streamlit as st
    st.subheader('九标签热力图 · funny 已禁用')
    tags = st.multiselect('显示标签', TIMELINE_TAGS, default=TIMELINE_TAGS,
                         format_func=lambda tag: f'{tag} · {LABELS[tag]}', key=key_prefix+'_tag_selection')
    # Preserve the requested row order even when checkboxes are reselected.
    tags = [tag for tag in TIMELINE_TAGS if tag in tags]
    if not tags:
        st.info('选择至少一个标签即可查看图表。'); return
    rows = tag_timeline_rows(result, tags)
    mode=st.radio('显示方式',['兼容显示','交互显示'],horizontal=True,key=key_prefix+'_renderer')
    svg=heatmap_svg(rows,tags)
    if mode=='兼容显示':
        import streamlit.components.v1 as components
        components.html('<!doctype html><meta charset="utf-8"><style>body{margin:0}div{overflow-x:auto}</style><div>'+svg+'</div>',height=max(190,100+34*len(tags)),scrolling=True)
    else:
        st.altair_chart(tag_heatmap(rows, tags), use_container_width=True)
    st.download_button('下载热力图 SVG',svg,file_name='tag_heatmap.svg',mime='image/svg+xml',key=key_prefix+'_svg')
    st.caption('每列宽度对应原始片段时长；颜色越亮得分越高。灰色为失败或缺失。固定0–1色标；funny已禁用。')
    with st.expander('分标签阶梯趋势'):
        if any(row['score'] is not None for row in rows):
            st.altair_chart(tag_trends(rows, tags), use_container_width=True)
            st.caption('每个片段内保持同一得分；失败片段留空，不跨越缺失片段连线。各图使用相同的0–1纵轴。')
        else:
            st.info('暂无成功的预测分数。')


def heatmap_svg(rows, tags=TIMELINE_TAGS):
    """Browser-native SVG fallback; no Vega runtime or network dependency."""
    from html import escape
    if not rows:
        return ''
    left=165; top=35; plot_width=900; row_height=31
    height=top+row_height*len(tags)+78
    start=min(row['start'] for row in rows);end=max(row['end'] for row in rows)
    span=max(end-start,1e-9)
    def x(t):return left+(t-start)/span*plot_width
    palette=[(68,1,84),(59,82,139),(33,145,140),(94,201,98),(253,231,37)]
    def color(value):
        if value is None:return '#525965'
        pos=min(4.,max(0.,value*4));i=min(3,int(pos));f=pos-i
        rgb=[round(palette[i][c]*(1-f)+palette[i+1][c]*f) for c in range(3)]
        return '#'+''.join(f'{v:02x}' for v in rgb)
    svg=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1135 {height}" role="img" aria-label="标签得分随时间变化的热力图" style="width:100%;min-width:680px;font-family:Microsoft YaHei,Arial,sans-serif;background:#11151c;border-radius:10px">',
         '<defs><linearGradient id="scoreLegend">'+''.join(f'<stop offset="{i*25}%" stop-color="{color(i/4)}"/>' for i in range(5))+'</linearGradient></defs>']
    for j,tag in enumerate(tags):
        y=top+j*row_height
        svg.append(f'<text x="{left-12}" y="{y+20}" fill="#e7edf5" text-anchor="end" font-size="13">{escape(tag)} · {escape(LABELS.get(tag,tag))}</text>')
        for row in (r for r in rows if r['tag']==tag):
            value=row['score'];text='缺失' if value is None else f'{value:.4f}'
            tip=f"{tag} | 片段 {row['segment']} | {row['start']:.3f}–{row['end']:.3f}秒 | 得分 {text} | {row['status']}"
            svg.append(f'<rect x="{x(row["start"]):.3f}" y="{y}" width="{max(0,x(row["end"])-x(row["start"])):.3f}" height="{row_height-1}" fill="{color(value)}"><title>{escape(tip)}</title></rect>')
    bottom=top+len(tags)*row_height
    for i in range(7):
        t=start+span*i/6;px=x(t)
        svg.append(f'<line x1="{px}" y1="{bottom+3}" x2="{px}" y2="{bottom+8}" stroke="#9daab9"/><text x="{px}" y="{bottom+25}" fill="#bfcbd9" text-anchor="middle" font-size="12">{t:.1f}s</text>')
    svg.extend([f'<text x="{left}" y="{bottom+54}" fill="#bfcbd9" font-size="12">得分</text>',
        f'<rect x="{left+40}" y="{bottom+43}" width="180" height="12" fill="url(#scoreLegend)"/>',
        f'<text x="{left+32}" y="{bottom+54}" fill="#bfcbd9" text-anchor="end" font-size="12">0</text>',
        f'<text x="{left+228}" y="{bottom+54}" fill="#bfcbd9" font-size="12">1</text>',
        f'<rect x="{left+270}" y="{bottom+43}" width="18" height="12" fill="#525965"/><text x="{left+297}" y="{bottom+54}" fill="#bfcbd9" font-size="12">失败 / 缺失</text>','</svg>'])
    return ''.join(svg)


def load_timeline_json(data):
    import json
    result=json.loads(data)
    if not isinstance(result,dict) or not isinstance(result.get('segments'),list) or not result['segments']:
        raise ValueError('请选择“下载完整结果 JSON”导出的文件；分段计划不包含情绪得分。')
    count=0
    for row in result['segments']:
        if not isinstance(row,dict) or not all(k in row for k in ['segment','start_seconds','end_seconds']):
            raise ValueError('结果缺少片段编号或起止时间。')
        if not all(isinstance(row[k],(int,float)) and math.isfinite(row[k]) for k in ['start_seconds','end_seconds']) or row['end_seconds']<=row['start_seconds']:
            raise ValueError('结果中的片段起止时间无效。')
        if not isinstance(row.get('scores',[]),list):raise ValueError('结果中的标签分数格式无效。')
        for item in row.get('scores',[]):
            if not isinstance(item,dict) or not isinstance(item.get('tag'),str):
                raise ValueError('结果中的标签条目格式无效。')
        count+=bool(row.get('scores'))
    if not count:raise ValueError('该文件没有预测分数，请导入预测结果而非分段计划。')
    return correct_result(result)
