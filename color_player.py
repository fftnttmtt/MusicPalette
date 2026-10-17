"""One iframe owns the audio clock and the live HSV color block."""
import base64
from html import escape
import json
from pathlib import Path


def audio_mime(name):
    return {'.mp3':'audio/mpeg','.wav':'audio/wav','.flac':'audio/flac',
            '.ogg':'audio/ogg','.m4a':'audio/mp4'}.get(Path(name).suffix.lower(),'audio/mpeg')


def build_color_player(mapped, audio_data, audio_name='audio'):
    rows=sorted(mapped['segments'],key=lambda r:r['start_seconds'])
    payload={'segments':rows,'audioName':audio_name,'transition':mapped.get('transition')}
    data=json.dumps(payload,ensure_ascii=False,allow_nan=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    encoded=base64.b64encode(audio_data).decode('ascii')
    script=Path(__file__).with_suffix('.js').read_text(encoding='utf-8')
    return '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<style>
*{box-sizing:border-box}body{margin:0;color:#edf2fa;font-family:Microsoft YaHei,Arial,sans-serif;background:#11151c;border-radius:12px;padding:16px}
.header{display:flex;justify-content:space-between;gap:12px;font-size:13px;color:#aebbcf;margin-bottom:12px}.name{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
#colorBlock{height:215px;border-radius:9px;border:1px solid #ffffff25;background:#252c36}
.meta{display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap;font-variant-numeric:tabular-nums;margin:12px 0 8px;font-size:14px}
#details{color:#9daec5;font-size:12px;min-height:18px;margin-bottom:12px}
audio{width:100%;height:40px;display:block}#seekbar{position:relative;display:flex;height:20px;margin:12px 0 4px;cursor:pointer;border-radius:4px;overflow:hidden;background:#252c36}#colorTrack{position:absolute;width:100%;height:100%;pointer-events:none}
#playhead{position:absolute;top:0;bottom:0;width:2px;background:white;box-shadow:0 0 4px #000;pointer-events:none}
.foot{display:flex;justify-content:space-between;color:#899bb2;font-size:11px}button{border:1px solid #ffffff30;color:#cbd8eb;background:transparent;border-radius:4px;cursor:pointer;font-size:11px}
#warning{color:#ffc776;font-size:12px;min-height:16px;margin-top:7px}
</style>
<div class="header"><span>实时颜色 · 与下方播放器同步</span><span class="name">'''+escape(audio_name)+'''</span></div>
<div id="colorBlock" aria-label="当前音乐片段的颜色"></div>
<div class="meta"><span id="segmentLabel">准备播放</span><span id="timeLabel">0:00.00</span></div>
<div id="details"></div>
<audio id="music" controls preload="metadata" src="data:'''+audio_mime(audio_name)+''';base64,'''+encoded+'''"></audio>
<div id="seekbar" aria-label="点击颜色条跳转"><canvas id="colorTrack" width="1800" height="20"></canvas><span id="playhead"></span></div>
<div class="foot"><span>点击色条跳转；暂停保留当前颜色</span><button id="jumpStart">回到分析起点</button></div><div id="warning"></div>
<script id="colorData" type="application/json">'''+data+'''</script><script>'''+script+'''</script></html>'''
