/* Half-open segment intervals: seeking to a boundary selects the next segment. */
function findColorSegment(rows, seconds, ended) {
  let low=0, high=rows.length;
  while(low<high){ const mid=(low+high)>>1; if(rows[mid].start_seconds<=seconds)low=mid+1;else high=mid; }
  const row=rows[low-1];
  if(row && seconds<row.end_seconds)return row;
  const last=rows[rows.length-1];
  if(ended && last && Math.abs(seconds-last.end_seconds)<0.03)return last;
  return null;
}
function upperBound(values,t){
  let lo=0,hi=values.length;
  while(lo<hi){const m=(lo+hi)>>1;if(values[m]<=t)lo=m+1;else hi=m;}
  return lo;
}
function speedIntegral(curve,t){
  const {grid,rate,area}=curve;
  if(grid.length===1)return 0;
  t=Math.max(grid[0],Math.min(grid[grid.length-1],t));
  const j=Math.max(0,Math.min(grid.length-2,upperBound(grid,t)-1)),d=t-grid[j];
  const slope=(rate[j+1]-rate[j])/(grid[j+1]-grid[j]);
  return area[j]+rate[j]*d+.5*slope*d*d;
}
function labToColor(lab){
  const [L,a,b]=lab;
  const l=(L+.3963377774*a+.2158037573*b)**3;
  const m=(L-.1055613458*a-.0638541728*b)**3;
  const s=(L-.0894841775*a-1.291485548*b)**3;
  const linear=[4.0767416621*l-3.3077115913*m+.2309699292*s,
    -1.2684380046*l+2.6097574011*m-.3413193965*s,
    -.0041960863*l-.7034186147*m+1.707614701*s];
  const rgb=linear.map(v=>{v=Math.max(0,Math.min(1,v));return Math.round(255*(v<=.0031308?12.92*v:1.055*v**(1/2.4)-.055));});
  const [r,g,blue]=rgb.map(v=>v/255),max=Math.max(r,g,blue),min=Math.min(r,g,blue),d=max-min;
  let h=0;
  if(d){if(max===r)h=((g-blue)/d)%6;else if(max===g)h=(blue-r)/d+2;else h=(r-g)/d+4;h=(h*60+360)%360;}
  return {rgb,hex:'#'+rgb.map(v=>v.toString(16).padStart(2,'0')).join(''),hue_degrees:h,saturation:max?d/max:0,value:max};
}
function transitionColor(curve,t){
  if(!curve||!curve.enabled)return null;
  const runs=curve.runs;
  const i=upperBound(runs.map(r=>r.start),t)-1;
  if(i<0||t>runs[i].end)return null;
  const run=runs[i],times=run.times,labs=run.labs;
  const anchorIndex=upperBound(times,t)-1;
  if(anchorIndex>=0&&times[anchorIndex]===t)return labToColor(labs[anchorIndex]);
  if(curve.mode==='boundary_hold'){
    const events=run.events,j=upperBound(events.map(e=>e.start_seconds),t)-1;
    if(j<0)return labToColor(labs[0]);
    const event=events[j];
    let u=event.duration_seconds<=0?1:Math.max(0,Math.min(1,(t-event.start_seconds)/event.duration_seconds));
    u=u*u*(3-2*u);
    return labToColor(labs[event.from_index].map((v,k)=>v*(1-u)+labs[event.to_index][k]*u));
  }
  if(times.length===1||t<=times[0])return labToColor(labs[0]);
  if(t>=times[times.length-1])return labToColor(labs[labs.length-1]);
  const j=upperBound(times,t)-1;
  let u=Math.max(0,Math.min(1,(speedIntegral(curve,t)-run.anchor_area[j])/(run.anchor_area[j+1]-run.anchor_area[j])));
  u=u*u*(3-2*u);
  return labToColor(labs[j].map((v,k)=>v*(1-u)+labs[j+1][k]*u));
}
function initializeColorPlayer(){
  const data=JSON.parse(document.getElementById('colorData').textContent);
  const rows=data.segments, audio=document.getElementById('music');
  const block=document.getElementById('colorBlock'), label=document.getElementById('segmentLabel');
  const timeLabel=document.getElementById('timeLabel'), details=document.getElementById('details');
  const bar=document.getElementById('seekbar'), head=document.getElementById('playhead');
  const warning=document.getElementById('warning');
  const start=rows.length?rows[0].start_seconds:0;
  const end=rows.length?Math.max(...rows.map(r=>r.end_seconds)):0;
  const span=Math.max(end-start,0.001);
  let raf=0;
  const missing='repeating-linear-gradient(135deg,#525965 0,#525965 7px,#727c8b 7px,#727c8b 9px)';
  function format(t){ t=Math.max(0,t||0);return `${Math.floor(t/60)}:${(t%60).toFixed(2).padStart(5,'0')}`; }
  const canvas=document.getElementById('colorTrack'),ctx=canvas.getContext('2d');
  bar.style.background=missing;
  for(let x=0;x<canvas.width;x++){
    const t=start+span*x/Math.max(1,canvas.width-1),row=findColorSegment(rows,t,x===canvas.width-1);
    if(!row||!row.hex)continue;
    const color=transitionColor(data.transition,t)||row;
    ctx.fillStyle=color.hex;ctx.fillRect(x,0,1,canvas.height);
  }
  function update(){
    const t=audio.currentTime||0,row=findColorSegment(rows,t,audio.ended);
    timeLabel.textContent=format(t);
    head.style.left=`${Math.max(0,Math.min(100,100*(t-start)/span))}%`;
    if(row){
      const color=row.hex?(transitionColor(data.transition,t)||row):null;
      block.style.background=color?color.hex:missing;
      label.textContent=`片段 ${row.segment} · ${format(row.start_seconds)}–${format(row.end_seconds)}`;
      const prefix=data.transition&&data.transition.enabled?'当前过渡色 · ':'';
      details.textContent=color?`${prefix}${color.hex.toUpperCase()}   H ${color.hue_degrees.toFixed(1)}°   S ${(color.saturation*100).toFixed(1)}%   V ${(color.value*100).toFixed(1)}%`:'当前片段没有可用映射颜色';
    }else{
      block.style.background=missing;label.textContent='当前时间不在已分析范围内';details.textContent='';
    }
  }
  function loop(){update();if(!audio.paused&&!audio.ended)raf=requestAnimationFrame(loop);}
  audio.addEventListener('play',()=>{cancelAnimationFrame(raf);loop();});
  for(const event of ['timeupdate','seeking','seeked','pause','ended'])audio.addEventListener(event,update);
  audio.addEventListener('loadedmetadata',()=>{
    if(start>0)audio.currentTime=Math.min(start,audio.duration||start);
    if(Number.isFinite(audio.duration) && end>audio.duration+0.25)warning.textContent='音乐时长短于结果时间范围，请确认选择的是同一首音乐。';
    update();
  });
  audio.addEventListener('error',()=>{warning.textContent='音乐无法播放，请重新选择对应音频或尝试WAV/MP3格式。';});
  bar.addEventListener('click',event=>{
    if(!Number.isFinite(audio.duration))return;
    const rect=bar.getBoundingClientRect();
    audio.currentTime=Math.max(0,Math.min(audio.duration,start+span*(event.clientX-rect.left)/rect.width));update();
  });
  document.getElementById('jumpStart').addEventListener('click',()=>{
    if(Number.isFinite(audio.duration))audio.currentTime=Math.min(start,audio.duration);update();
  });
  update();
}
if(typeof document!=='undefined')initializeColorPlayer();
