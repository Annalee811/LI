'use strict';
// A separate, editable proposal layer. It never changes the four pixel-label layers.
let structure52=null, selectedSegment52=null, selectedNode52=null, selectedArms52=[];
let click52=[500,500], draggingNode52=false, undo52=[], viewToken52=0;
const caches52=new Map(), images52=new Map();
const types52={tip:'根尖候选',boundary_endpoint:'图像边界端点',crossing:'交叉候选',branch:'分叉候选',junction_uncertain:'连接待定'};
const signs52={tip:'T',boundary_endpoint:'E',crossing:'X',branch:'B',junction_uncertain:'?'};
const esc52=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function frame52(){return structure52?.periods.find(f=>f.name===active?.name&&f.image_sha256===active?.image_sha256)||null;}
function decodeOwners52(a,n){
 if(!Array.isArray(a)||a.length%2)throw Error('逐根像素编码无效');
 const out=new Int32Array(n);let offset=0;
 for(let i=0;i<a.length;i+=2){const id=a[i],count=a[i+1];if(!Number.isSafeInteger(id)||id<0||!Number.isSafeInteger(count)||count<=0||offset+count>n)throw Error('逐根像素编码越界');out.fill(id,offset,offset+count);offset+=count;}
 if(offset!==n)throw Error('逐根像素尺寸不匹配');return out;
}
function cache52(frame){
 if(!caches52.has(frame.name)){const owner=decodeOwners52(frame.owner_map_rle,frame.width*frame.height);caches52.set(frame.name,{owner,byIndex:new Map(frame.segments.map(s=>[s.index,s])),byId:new Map(frame.segments.map(s=>[s.id,s]))});}
 return caches52.get(frame.name);
}
function rebuild52(){caches52.clear();for(const f of structure52?.periods||[])cache52(f);}
function root52(frame,sid){return cache52(frame).byId.get(sid)?.root_id||'';}
function rootColor52(id){let h=0;for(const c of id)h=(h*31+c.charCodeAt(0))>>>0;return `hsl(${(h*137.508)%360},80%,65%)`;}
function rgbColor52(id){const hue=Array.from(id).reduce((n,c)=>(n*31+c.charCodeAt(0))>>>0,0)*137.508%360;const a=.8*.35;const f=n=>{const k=(n+hue/30)%12;return Math.round((.65-a*Math.max(-1,Math.min(k-3,9-k,1)))*255)};return [f(0),f(8),f(4)];}
function ownerAt52(frame,x,y){const i=Math.max(0,Math.min(frame.width-1,Math.round(x)))+Math.max(0,Math.min(frame.height-1,Math.round(y)))*frame.width;return cache52(frame).byIndex.get(cache52(frame).owner[i])||null;}
function distancePath52(points,x,y){let best=Infinity,at=0;for(let i=0;i<points.length;i++){const p=points[i],d=(p[0]-x)**2+(p[1]-y)**2;if(d<best){best=d;at=i}}return {d:best,at};}
function findSegment52(frame,x,y){let s=ownerAt52(frame,x,y);if(s)return s;let best=144;for(const q of frame.segments){const d=distancePath52(q.points,x,y).d;if(d<best){best=d;s=q}}return s;}
function findNode52(frame,x,y){const z=Number($('zoom').value)||1;let best=(12/Math.max(z,.3))**2,node=null;for(const n of frame.nodes){const d=(n.x-x)**2+(n.y-y)**2;if(d<best){best=d;node=n}}return node;}
function serialize52(){
 if(!structure52)return null;
 const obj=structuredClone(structure52);
 for(const f of obj.periods){const live=structure52.periods.find(v=>v.name===f.name),owner=cache52(live).owner.slice(),st=states.get(f.name+':'+f.image_sha256);if(st)for(let i=0;i<owner.length;i++)if(st.mask[i]!==255)owner[i]=0;f.owner_map_rle=rleEncode(owner);f.human_reviewed=f.segments.every(s=>s.review_status==='confirmed')&&f.nodes.every(n=>n.review_status==='confirmed');}
 obj.human_reviewed=obj.periods.every(f=>f.human_reviewed);obj.status=obj.human_reviewed?'human_reviewed':'proposal_under_review';
 for(const l of obj.cross_period_links){const p=obj.periods.find(f=>f.period===l.previous_period)?.segments.find(s=>s.id===l.previous_segment),c=obj.periods.find(f=>f.period===l.current_period)?.segments.find(s=>s.id===l.current_segment);if(!p||!c||p.root_id!==c.root_id){l.review_status='pending';l.kind='identity_link_needs_review'}else l.root_id=c.root_id;}
 obj.human_reviewed=obj.human_reviewed&&obj.cross_period_links.every(l=>l.review_status==='confirmed');obj.status=obj.human_reviewed?'human_reviewed':'proposal_under_review';
 return obj;
}
function snapshot52(){undo52.push(JSON.stringify(serialize52()));while(undo52.length>12)undo52.shift();}
function changed52(item){if(item)item.review_status='modified';if(structure52){structure52.human_reviewed=false;structure52.status='proposal_under_review'}if(active)active.modified=true;refresh52();render();}
function selected52(){const f=frame52();return f?cache52(f).byId.get(selectedSegment52):null;}
function node52(){return frame52()?.nodes.find(n=>n.id===selectedNode52)||null;}
function newRoot52(){let max=0;for(const f of structure52.periods)for(const s of f.segments)max=Math.max(max,Number(s.root_id.replace(/^R/,''))||0);return 'R'+String(max+1).padStart(4,'0');}
function readRoot52(v){v=String(v).trim().toUpperCase();if(/^\d+$/.test(v))v='R'+v.padStart(4,'0');if(!/^R\d{4,6}$/.test(v))throw Error('根编号请输入 R0001 这样的形式，也可直接输入数字。');return v;}
function act52(fn){return()=>{try{if(!frame52())throw Error('请先打开带结构标注的四期原图');fn();}catch(e){status(e.message)}}}

const host52=document.createElement('div');host52.id='structureControls52';
host52.innerHTML=`<h2>逐根结构复核</h2><div class="tip">自动初稿，全部待复核。编号表示候选根身份；短碎段可能仍需合并。同一根可跨期沿用编号。X交叉、B分叉、T根尖、E边界、?待定。</div>
<div id="periodButtons52">${['T1','T2','T3','T4'].map((t,i)=>`<button data-period52="${i}">${t}</button>`).join('')}</div>
<button id="restore52">恢复进度 JSON</button><button id="localImages52">选择本地原图</button>
<label>缩放<input id="zoom52" type="range" min="0.1" max="8" step="0.1" value="0.6"></label><button id="pan52">移动画面／空格拖动</button>
<label>操作<select id="tool52"><option value="select">选择根段／点位</option><option value="move">移动点位（拖动）</option><option value="tip">添加根尖</option><option value="branch">添加分叉点</option><option value="crossing">添加交叉点</option><option value="split">在点击处拆开根段</option><option value="pixel">原有像素标注画笔</option></select></label>
<label>显示<select id="view52"><option value="identity">逐根编号与颜色</option><option value="temporal">原时序类别颜色</option><option value="rgb">原图 RGB</option></select></label>
<div><label class="inline52"><input type="checkbox" id="labels52" checked>编号</label><label class="inline52"><input type="checkbox" id="points52" checked>点位</label><button id="fit52">全图适应</button></div>
<label class="inline52"><input type="checkbox" id="focusPoints52" checked>点位聚焦所选根</label><label class="inline52"><input type="checkbox" id="short52">列表显示短碎段</label>
<div id="summary52"></div><label>查找根编号／根段<input id="search52" type="text" placeholder="例如 R0021 或 T2-S0010"></label>
<select id="segments52" size="5"></select><div><button id="previousRoot52">上一根段</button><button id="nextRoot52">下一根段</button></div>
<details id="rootDetails52" open><summary>当前根段与跨期身份</summary><div id="rootInfo52"></div>
<label>逐根编号<input id="rootId52" type="text" list="rootIds52"><datalist id="rootIds52"></datalist></label>
<button id="assign52">此段归入该编号</button><button id="rename52">本期同编号整根改号</button><button id="newRoot52">此段另设编号</button>
<label>侧根的母根编号（可空）<input id="parent52" type="text" list="rootIds52"></label><button id="setParent52">保存母根</button>
<div id="links52"></div><button id="confirmLink52">确认与前一期同号根对应</button><button id="confirmRoot52">确认此根段</button></details>
<details id="nodeDetails52" open><summary>交叉／分叉／根尖</summary><div id="nodeInfo52">点击图上的点位查看</div>
<label>点位类型<select id="nodeType52">${Object.entries(types52).map(([k,v])=>`<option value="${k}">${v}</option>`).join('')}</select></label>
<label>交叉中心所属根段<select id="nodeOwner52"></select></label>
<div id="arms52"></div><button id="connect52">连接选中的两臂</button><button id="disconnect52">断开选中臂的连接</button>
<button id="saveNode52">保存点位修改</button><button id="confirmNode52">确认此点位</button><button id="deleteNode52">删除此点位</button>
<label>备注<input id="note52" type="text"></label></details>
<button id="nextPending52">下一个待复核交叉点</button><button id="undo52">撤销结构修改</button>
<button id="save52">保存全部进度 JSON</button><button id="saveHtml52">保存可编辑网页</button>
<small>首次联网读取固定原图；也可选择对应本地原图。先保存再关闭。恢复 JSON 会同时恢复四期像素标注、根编号、连接和点位。结构修改不自动改写时序类别，也不会用于当前训练。</small>`;
const controls52=document.querySelector('.controls');controls52.prepend(host52);
const strip52=document.createElement('div');strip52.id='comparison52';strip52.innerHTML='<div>同一位置四期 RGB（点击根段或点位更新）</div><div id="crops52"></div>';document.querySelector('.view').append(strip52);
for(const t of ['T1','T2','T3','T4']){const d=document.createElement('div');d.innerHTML=`<b>${t}</b><canvas width="160" height="160" data-period="${t}"></canvas>`;$('crops52').append(d)}
let pixelsWarning52=false;
function refresh52(){
 const f=frame52();if(!f){$('summary52').textContent='当前 JSON 没有匹配的逐根结构标注';return;}
 const s=selected52(),n=node52(),query=$('search52').value.trim().toLowerCase();
 $('summary52').textContent=`${f.period}：${f.segments.length} 条候选轨迹（含碎段），${f.nodes.length} 个候选点；已确认根段 ${f.segments.filter(s=>s.review_status==='confirmed').length}。`;
 for(const b of $('periodButtons52').querySelectorAll('button'))b.classList.toggle('active',Number(b.dataset.period52)===index);
 const options=f.segments.filter(s=>(!query||s.root_id.toLowerCase().includes(query)||s.id.toLowerCase().includes(query))&&($('short52').checked||s.length_px>=30||s.id===selectedSegment52));
 $('segments52').innerHTML=options.map(s=>`<option value="${s.id}" ${s.id===selectedSegment52?'selected':''}>${s.root_id} · ${s.id} · ${Math.round(s.length_px)}px ${s.review_status==='confirmed'?'✓':s.fragment_candidate?'碎段?':''}</option>`).join('');
 if(!selectedSegment52)$('segments52').selectedIndex=-1;
 const ids=[...new Set(structure52.periods.flatMap(f=>f.segments.map(s=>s.root_id)))].sort();$('rootIds52').innerHTML=ids.map(id=>`<option value="${id}">`).join('');
 $('rootInfo52').textContent=s?`${s.id}｜${s.length_px.toFixed(1)} px｜${s.review_status==='confirmed'?'已确认':'待复核'}。此编号可能仍是整根的一部分。`:'在图上点击根或从列表选择';
 $('rootId52').value=s?.root_id||'';$('parent52').value=s?.parent_root_id||'';
 $('links52').innerHTML=s?(s.previous_candidates?.length?'<small>上一期候选：'+s.previous_candidates.map(v=>`${esc52(v.previous_root_id)} (${v.score})`).join('；')+'。确认同根时可沿用它的编号。</small>':'<small>无可靠上一期自动对应；不代表一定是新生根。</small>'):'';
 if(s){const links=structure52.cross_period_links.filter(l=>l.current_period===f.period&&(l.current_segment===s.id||l.current_segments?.includes(s.id)));$('links52').innerHTML+='<small>跨期对应：'+(links.map(l=>`${l.previous_period} ${esc52(l.root_id)} · ${l.review_status==='confirmed'?'已确认':'待复核'}`).join('；')||'尚未建立')+'</small>';}
 $('nodeInfo52').textContent=n?`${n.id}｜(${n.x.toFixed(1)}, ${n.y.toFixed(1)})｜${n.review_status==='confirmed'?'已确认':'待复核'}`:'点击图上的点位查看';
 $('nodeType52').value=n?.type||'junction_uncertain';$('note52').value=n?.note||'';
 let incident=n?[...new Set([n.owner_segment_id,...n.arms.map(a=>a.segment_id)].filter(id=>cache52(f).byId.has(id)))]:[];
 if(s&&!incident.includes(s.id))incident.push(s.id);
 $('nodeOwner52').innerHTML=incident.map(id=>`<option value="${id}" ${id===n?.owner_segment_id?'selected':''}>${root52(f,id)} · ${id}</option>`).join('');
 $('arms52').innerHTML=n?n.arms.map(a=>`<button data-arm52="${a.id}" class="${selectedArms52.includes(a.id)?'active':''}">${a.id} ${root52(f,a.segment_id)}</button>`).join('')+'<small>连接：'+(n.connections.map(p=>p.join(' ↔ ')).join('；')||'尚未确认')+'</small>':'';
 for(const b of $('arms52').querySelectorAll('button'))b.onclick=()=>{const id=b.dataset.arm52;selectedArms52=selectedArms52.includes(id)?selectedArms52.filter(v=>v!==id):[...selectedArms52.slice(-1),id];refresh52();render()};
 for(const id of ['assign52','rename52','newRoot52','setParent52','confirmRoot52','confirmLink52'])$(id).disabled=!s;
 for(const id of ['saveNode52','deleteNode52','confirmNode52'])$(id).disabled=!n;
 $('undo52').disabled=!undo52.length;
}
function center52(x,y){const view=$('b3ImageViewport'),z=Number($('zoom').value)||1;view.scrollLeft=x*z-view.clientWidth/2;view.scrollTop=y*z-view.clientHeight/2;}
async function compare52(x,y){
 const token=++viewToken52;click52=[x,y];
 for(const f of structure52?.periods||[]){
  try{let im=images52.get(f.name);if(!im){const file=files.find(v=>v.name===f.name)||window.availableReferenceFilesV50.find(v=>v.name===f.name);if(!file)continue;im=await decodeImage(file);images52.set(f.name,im)}
   if(token!==viewToken52)return;
   const canvas=$('crops52').querySelector(`[data-period="${f.period}"]`),ctx=canvas.getContext('2d');ctx.fillStyle='#202427';ctx.fillRect(0,0,160,160);const left=x-80,top=y-80;ctx.drawImage(im,-left,-top);ctx.strokeStyle='#ffdc52';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(74,80);ctx.lineTo(86,80);ctx.moveTo(80,74);ctx.lineTo(80,86);ctx.stroke();
  }catch(e){status('局部对照读取失败：'+e.message)}
 }
}
function choose52(s,n=null,center=false){selectedSegment52=s?.id||null;selectedNode52=n?.id||null;selectedArms52=[];const p=n?[n.x,n.y]:s?.points[Math.floor(s.points.length/2)];if(p){compare52(...p);if(center)center52(...p)}refresh52();render();}

const renderBase52=render;
render=function(){
 const f=frame52(),view=$('view52').value;
 if(!f||$('tool52').value==='pixel'){renderBase52();return;}
 if(view==='temporal')renderBase52();else ovctx.clearRect(0,0,$('overlay').width,$('overlay').height);
 if(!active||!$('show').checked)return;
 const cache=cache52(f),s=selected52(),n=node52();
 if(view==='identity'){
  const im=ovctx.createImageData(W,H),colors=new Map(f.segments.map(q=>[q.index,rgbColor52(q.root_id)]));
  for(let i=0;i<N;i++){if(active.mask[i]!==255)continue;const color=colors.get(cache.owner[i]);if(!color)continue;const j=i*4;im.data.set(color,j);im.data[j+3]=s?(cache.byIndex.get(cache.owner[i])?.root_id===s.root_id?240:80):185;}
  // Junction ownership is a separate draft layer; original temporal colors stay intact.
  for(const p of f.nodes){if(!p.ownership_radius||!p.owner_segment_id)continue;const color=rgbColor52(root52(f,p.owner_segment_id)),r=p.ownership_radius;
   for(let y=Math.max(0,Math.floor(p.y-r));y<=Math.min(H-1,Math.ceil(p.y+r));y++)for(let x=Math.max(0,Math.floor(p.x-r));x<=Math.min(W-1,Math.ceil(p.x+r));x++){const i=y*W+x;if(active.mask[i]===255&&(x-p.x)**2+(y-p.y)**2<=r*r){im.data.set(color,i*4);im.data[i*4+3]=240;}}
  }
  ovctx.putImageData(im,0,0);
 }
 const z=Math.max(.1,Number($('zoom').value)||1),selectedRoot=s?.root_id;
 if($('labels52').checked){ovctx.font=`bold ${Math.min(24,12/z)}px Arial`;ovctx.textBaseline='middle';
  const labeled=new Set();for(const q of f.segments){if(labeled.has(q.root_id)||q.length_px<(z<.8?100:35)&&q.root_id!==selectedRoot)continue;labeled.add(q.root_id);const p=q.points[Math.floor(q.points.length/2)];if(!p)continue;const text=q.root_id,w=ovctx.measureText(text).width;ovctx.fillStyle='rgba(0,0,0,.8)';ovctx.fillRect(p[0]+3,p[1]-8/z,w+4,16/z);ovctx.fillStyle=q.root_id===selectedRoot?'#fff':rootColor52(q.root_id);ovctx.fillText(text,p[0]+5,p[1]);}
 }
 if($('points52').checked)for(const p of f.nodes){const isSelected=p.id===n?.id,belongs=s&&root52(f,p.owner_segment_id)===s.root_id;
  if($('focusPoints52').checked&&!isSelected){if(s&&!belongs&&!(n&&Math.hypot(p.x-n.x,p.y-n.y)<35))continue;if(!s&&['tip','boundary_endpoint'].includes(p.type))continue;}
  if(z<.75&&!isSelected&&!belongs)continue;if(z<1.1&&!belongs&&!isSelected&&['tip','boundary_endpoint'].includes(p.type))continue;
  const radius=(isSelected?7:4)/z;ovctx.beginPath();ovctx.arc(p.x,p.y,radius,0,2*Math.PI);ovctx.fillStyle=isSelected?'#fff':p.review_status==='confirmed'?'#56f0a0':'#ffcf50';ovctx.fill();ovctx.strokeStyle='#181818';ovctx.lineWidth=1/z;ovctx.stroke();
  if(z>=.75||isSelected){ovctx.font=`bold ${11/z}px Arial`;ovctx.fillStyle='#fff';ovctx.strokeStyle='#000';ovctx.lineWidth=2/z;ovctx.strokeText(signs52[p.type]||'?',p.x+7/z,p.y-4/z);ovctx.fillText(signs52[p.type]||'?',p.x+7/z,p.y-4/z);}
 }
 if(n){ovctx.font=`bold ${12/z}px Arial`;for(const a of n.arms){ovctx.beginPath();ovctx.moveTo(n.x,n.y);ovctx.lineTo(a.x,a.y);ovctx.strokeStyle=selectedArms52.includes(a.id)?'#fff':'#ffcf50';ovctx.lineWidth=2/z;ovctx.stroke();ovctx.fillStyle='#fff';ovctx.strokeStyle='#000';ovctx.lineWidth=3/z;ovctx.strokeText(a.id,a.x+3/z,a.y-3/z);ovctx.fillText(a.id,a.x+3/z,a.y-3/z);}
  for(const pair of n.connections){const a=n.arms.find(a=>a.id===pair[0]),b=n.arms.find(a=>a.id===pair[1]);if(!a||!b)continue;ovctx.beginPath();ovctx.moveTo(a.x,a.y);ovctx.quadraticCurveTo(n.x,n.y,b.x,b.y);ovctx.lineWidth=2/z;ovctx.strokeStyle='#55e1ff';ovctx.stroke();}
 }
};

const pointerDownBase52=$('overlay').onpointerdown,pointerMoveBase52=$('overlay').onpointermove,pointerUpBase52=$('overlay').onpointerup;
$('overlay').onpointerdown=function(e){
 if($('tool52').value==='pixel')return pointerDownBase52(e);
 if(!active||e.button!==0||!frame52())return;e.preventDefault();$('overlay').focus();
 const [x,y]=xy(e),f=frame52(),tool=$('tool52').value;click52=[x,y];
 if(tool==='split'){try{split52(x,y)}catch(e){status(e.message)}return;}
 let n=findNode52(f,x,y),s=n?cache52(f).byId.get(n.owner_segment_id):findSegment52(f,x,y);
 if(['tip','branch','crossing'].includes(tool)){
  snapshot52();const id=f.period+'-M'+String(Date.now());s=s||selected52();n={id,type:tool,x:Math.max(0,Math.min(W-1,x)),y:Math.max(0,Math.min(H-1,y)),owner_segment_id:s?.id||null,arms:[],connections:[],review_status:'modified',proposed:false,note:'',ownership_radius:tool==='crossing'||tool==='branch'?2:0};
  if(tool!=='tip')for(const q of f.segments){const near=distancePath52(q.points,x,y);if(near.d>18**2)continue;const choices=[Math.max(0,near.at-12),Math.min(q.points.length-1,near.at+12)];for(const index of choices){const p=q.points[index];if(Math.hypot(p[0]-x,p[1]-y)<4)continue;if(n.arms.some(a=>a.segment_id===q.id&&Math.hypot(a.x-p[0],a.y-p[1])<4))continue;n.arms.push({id:'A'+(n.arms.length+1),segment_id:q.id,x:p[0],y:p[1]});}}
  f.nodes.push(n);choose52(s,n);changed52(n);return;
 }
 choose52(s,n);compare52(x,y);
 if(tool==='move'&&n){snapshot52();draggingNode52=true;$('overlay').setPointerCapture(e.pointerId);}
};
$('overlay').onpointermove=function(e){if($('tool52').value==='pixel')return pointerMoveBase52(e);if(!draggingNode52)return;const n=node52();if(!n)return;const [x,y]=xy(e);n.x=Math.max(0,Math.min(W-1,x));n.y=Math.max(0,Math.min(H-1,y));render();};
$('overlay').onpointerup=function(e){if($('tool52').value==='pixel')return pointerUpBase52(e);if(draggingNode52){draggingNode52=false;changed52(node52());compare52(node52().x,node52().y)}};
$('overlay').onpointercancel=()=>{if(draggingNode52){draggingNode52=false;changed52(node52())}};

function split52(x,y){
 const f=frame52(),s=findSegment52(f,x,y);if(!s)throw Error('请点击要拆开的根段');const at=distancePath52(s.points,x,y).at;
 if(at<3||at>s.points.length-4)throw Error('点击位置离根段末端太近，请在根段中部拆分');
 snapshot52();const cache=cache52(f),oldId=s.id,second=structuredClone(s),nextIndex=Math.max(...f.segments.map(s=>s.index))+1;
 second.index=nextIndex;second.id=f.period+'-S'+String(nextIndex).padStart(4,'0');second.root_id=newRoot52();second.points=s.points.slice(at);s.points=s.points.slice(0,at);second.previous_candidates=[];second.parent_root_id=null;second.note='手动拆分，跨期关系需复核';
 for(let i=0;i<cache.owner.length;i++)if(cache.owner[i]===s.index){const xx=i%W,yy=Math.floor(i/W);if(distancePath52(second.points,xx,yy).d<distancePath52(s.points,xx,yy).d)cache.owner[i]=second.index;}
 function length(p){let n=0;for(let i=1;i<p.length;i++)n+=Math.min(5,Math.hypot(p[i][0]-p[i-1][0],p[i][1]-p[i-1][1]));return n;}
 s.length_px=length(s.points);second.length_px=length(second.points);s.review_status=second.review_status='modified';f.segments.push(second);cache.byId.set(second.id,second);cache.byIndex.set(second.index,second);
 for(const n of f.nodes){for(const a of n.arms)if(a.segment_id===oldId&&distancePath52(second.points,a.x,a.y).d<distancePath52(s.points,a.x,a.y).d)a.segment_id=second.id;if(n.owner_segment_id===oldId&&distancePath52(second.points,n.x,n.y).d<distancePath52(s.points,n.x,n.y).d)n.owner_segment_id=second.id;}
 f.nodes.push({id:f.period+'-M'+Date.now(),type:'junction_uncertain',x,y,owner_segment_id:s.id,arms:[],connections:[],review_status:'modified',proposed:false,ownership_radius:0,note:'手动拆根位置；不自动认定为根尖'});
 choose52(second);changed52(s);status('根段已拆开并给后半段新编号。请复核两段编号、交叉连接与跨期关系。');
}

$('assign52').onclick=act52(()=>{const s=selected52();if(!s)return;const id=readRoot52($('rootId52').value);snapshot52();s.root_id=id;changed52(s)});
$('rename52').onclick=act52(()=>{const s=selected52();if(!s)return;const id=readRoot52($('rootId52').value),old=s.root_id;snapshot52();for(const q of frame52().segments)if(q.root_id===old){q.root_id=id;q.review_status='modified'}changed52(s)});
$('newRoot52').onclick=act52(()=>{const s=selected52();if(!s)return;snapshot52();s.root_id=newRoot52();changed52(s)});
$('setParent52').onclick=act52(()=>{const s=selected52();if(!s)return;const v=$('parent52').value.trim(),parent=v?readRoot52(v):null;if(parent===s.root_id)throw Error('母根不能是根自身');snapshot52();s.parent_root_id=parent;changed52(s)});
$('confirmLink52').onclick=act52(()=>{const f=frame52(),s=selected52(),at=structure52.periods.findIndex(p=>p.name===f.name);if(!s||at<1)throw Error('T1 是基础期，没有前一期对应');const previous=structure52.periods[at-1],old=previous.segments.filter(q=>q.root_id===s.root_id),current=f.segments.filter(q=>q.root_id===s.root_id);if(!old.length)throw Error('前一期没有同编号根。请先检查四期 RGB，再把本期根段归入对应编号。');snapshot52();structure52.cross_period_links=structure52.cross_period_links.filter(l=>!(l.current_period===f.period&&(l.current_segments||[l.current_segment]).some(id=>current.some(s=>s.id===id))));structure52.cross_period_links.push({id:'L-M'+Date.now(),previous_period:previous.period,current_period:f.period,previous_segment:old[0].id,current_segment:current[0].id,previous_segments:old.map(s=>s.id),current_segments:current.map(s=>s.id),root_id:s.root_id,kind:'same_root',review_status:'confirmed',score:null});changed52();status('已确认本期同号根与前一期的对应关系。')});
$('connect52').onclick=act52(()=>{const n=node52(),f=frame52();if(!n||selectedArms52.length!==2)throw Error('请先选中两条连接臂');const pair=[...selectedArms52],a=n.arms.find(a=>a.id===pair[0]),b=n.arms.find(a=>a.id===pair[1]);snapshot52();n.connections=n.connections.filter(p=>!p.some(v=>pair.includes(v)));n.connections.push(pair);const sa=cache52(f).byId.get(a.segment_id),sb=cache52(f).byId.get(b.segment_id);if(sa&&sb){sb.root_id=sa.root_id;sa.review_status=sb.review_status='modified'}changed52(n);status('已连接两臂，并沿用同一根编号；原时序像素类别未改变。')});
$('disconnect52').onclick=act52(()=>{const n=node52();if(!n||!selectedArms52.length)throw Error('请先选择要断开的臂');snapshot52();n.connections=n.connections.filter(p=>!p.some(v=>selectedArms52.includes(v)));changed52(n);status('连接已删除。若两根仍同号，请再改编号或拆开根段。')});
$('saveNode52').onclick=act52(()=>{const n=node52();if(!n)return;snapshot52();n.type=$('nodeType52').value;n.owner_segment_id=$('nodeOwner52').value||null;n.note=$('note52').value;n.ownership_radius=['crossing','branch','junction_uncertain'].includes(n.type)?2:0;changed52(n)});
$('confirmNode52').onclick=act52(()=>{const n=node52();if(!n)return;snapshot52();n.type=$('nodeType52').value;n.owner_segment_id=$('nodeOwner52').value||null;n.note=$('note52').value;n.ownership_radius=['crossing','branch','junction_uncertain'].includes(n.type)?2:0;n.review_status='confirmed';n.proposed=false;if(active)active.modified=true;refresh52();render()});
$('confirmRoot52').onclick=act52(()=>{const s=selected52();if(!s)return;snapshot52();s.review_status='confirmed';if(active)active.modified=true;refresh52();render()});
$('deleteNode52').onclick=act52(()=>{const f=frame52();if(!node52())return;snapshot52();f.nodes=f.nodes.filter(n=>n.id!==selectedNode52);selectedNode52=null;changed52()});
$('undo52').onclick=()=>{if(!undo52.length)return;structure52=JSON.parse(undo52.pop());rebuild52();selectedSegment52=selectedNode52=null;selectedArms52=[];refresh52();render();if(active)active.modified=true;status('已撤销上一步结构修改；像素标注使用原画笔的撤销按钮。')};
$('nextPending52').onclick=act52(()=>{const f=frame52(),list=f.nodes.filter(n=>n.review_status!=='confirmed'&&!['tip','boundary_endpoint'].includes(n.type));if(!list.length){status('本期交叉与分叉点均已确认');return;}const i=list.findIndex(n=>n.id===selectedNode52),n=list[(i+1)%list.length];choose52(cache52(f).byId.get(n.owner_segment_id),n,true)});
$('segments52').onchange=()=>{const f=frame52();if(f)choose52(cache52(f).byId.get($('segments52').value),null,true)};
for(const [id,step] of [['previousRoot52',-1],['nextRoot52',1]])$(id).onclick=act52(()=>{const f=frame52(),i=f.segments.findIndex(s=>s.id===selectedSegment52),next=Math.max(0,Math.min(f.segments.length-1,i+step));choose52(f.segments[next],null,true)});
$('search52').oninput=refresh52;$('short52').onchange=refresh52;
$('restore52').onclick=()=>$('projectfile').click();$('localImages52').onclick=()=>$('imagefiles').click();
for(const b of $('periodButtons52').querySelectorAll('button'))b.onclick=()=>select(Number(b.dataset.period52));
$('zoom52').oninput=()=>{window.B3Zoom60Installed.setZoom(Number($('zoom52').value));render()};
$('pan52').onclick=()=>{$('b3Pan').click();$('pan52').classList.toggle('active',document.body.classList.contains('b3-panning'))};
for(const id of ['labels52','points52','view52','focusPoints52'])$(id).onchange=render;
$('tool52').onchange=()=>{draggingNode52=false;document.body.classList.toggle('structure-pixel52',$('tool52').value==='pixel');if($('tool52').value==='pixel'){pixelsWarning52=true;status('已切换像素画笔。改动像素后，新增区域的逐根归属需重新复核。')}render()};
$('fit52').onclick=()=>{const vp=$('b3ImageViewport');if(!vp)return;window.B3Zoom60Installed.setZoom(Math.min((vp.clientWidth-12)/W,(vp.clientHeight-12)/H));render()};
const originalSize52=sizeStage;sizeStage=function(){originalSize52();if(structure52)render()};

function validateStructure52(obj,project){
 if(!obj)return null;if(obj.version!==1||!Array.isArray(obj.periods))throw Error('逐根结构版本无效');
 const frames=new Set();for(const f of obj.periods){const im=project.images.find(v=>v.name===f.name&&v.image_sha256===f.image_sha256);if(!im||im.width!==f.width||im.height!==f.height||frames.has(f.name))throw Error('逐根结构与原图不匹配');frames.add(f.name);
  if(!Array.isArray(f.segments)||!Array.isArray(f.nodes))throw Error('逐根结构不完整');const ids=new Set(),indices=new Set();
  for(const s of f.segments){if(ids.has(s.id)||indices.has(s.index)||!Number.isSafeInteger(s.index)||s.index<=0||!/^R\d{4,6}$/.test(s.root_id)||!Array.isArray(s.points)||!s.points.length)throw Error('根编号或轨迹无效');ids.add(s.id);indices.add(s.index);for(const p of s.points)if(p.length!==2||!p.every(Number.isFinite)||p[0]<0||p[0]>=f.width||p[1]<0||p[1]>=f.height)throw Error('根轨迹坐标越界');}
  const owner=decodeOwners52(f.owner_map_rle,f.width*f.height);for(const id of new Set(owner))if(id!==0&&!indices.has(id))throw Error('逐根像素引用了不存在的根段');
  const nodeIds=new Set();for(const n of f.nodes){if(nodeIds.has(n.id)||!types52[n.type]||![n.x,n.y].every(Number.isFinite)||n.x<0||n.x>=f.width||n.y<0||n.y>=f.height||n.owner_segment_id&&!ids.has(n.owner_segment_id))throw Error('结构点位无效');nodeIds.add(n.id);const arms=new Set();for(const a of n.arms){if(arms.has(a.id)||!ids.has(a.segment_id)||![a.x,a.y].every(Number.isFinite))throw Error('连接臂无效');arms.add(a.id)}const used=new Set();for(const p of n.connections){if(p.length!==2||p[0]===p[1]||p.some(v=>!arms.has(v)||used.has(v)))throw Error('每个连接臂只能连接另一条臂');p.forEach(v=>used.add(v));}}
 }
 return structuredClone(obj);
}
const projectBase52=projectObject;
projectObject=function(){const obj=projectBase52();const st=serialize52();if(st)obj.root_structure_annotation=st;else delete obj.root_structure_annotation;return obj;};
const importBase52=importProject;
importProject=async function(file,options={}){const obj=JSON.parse(await file.text()),pending=validateStructure52(obj.root_structure_annotation,obj);if(structure52)window.restoredProjectMetadataV50.root_structure_annotation=serialize52();const result=await importBase52(file,options);structure52=pending;undo52=[];rebuild52();selectedSegment52=selectedNode52=null;selectedArms52=[];refresh52();render();return result;};
const selectBase52=select;
select=async function(i){const root=selected52()?.root_id;await selectBase52(i);selectedSegment52=selectedNode52=null;selectedArms52=[];const frame=frame52(),same=frame?.segments.filter(s=>s.root_id===root).sort((a,b)=>b.length_px-a.length_px)[0];if(same)selectedSegment52=same.id;refresh52();render();};
$('save52').onclick=()=>saveProject();
$('saveHtml52').onclick=()=>{
 const obj=projectObject();if(!obj.images.length)return;
 const clone=document.documentElement.cloneNode(true),tag=clone.querySelector('#reference-original-versions-v50'),versions=JSON.parse(tag.textContent);
 versions.original=obj;tag.textContent=JSON.stringify(versions).replace(/<\//g,'<\\/');
 clone.querySelector('#structureControls52')?.remove();clone.querySelector('#comparison52')?.remove();
 // Remove runtime-built controls before replaying the original setup scripts.
 const controls=clone.querySelector('.controls'),quick=clone.querySelector('#b3QuickTools'),other=clone.querySelector('#b3OtherControls');
 for(const el of [...(quick?.children||[]),...(other?.children||[])])controls.append(el);quick?.remove();other?.remove();
 clone.querySelector('#b3PeriodNav')?.remove();const movement=clone.querySelector('#b3Pan')?.parentElement;
 if(movement){if(movement.nextElementSibling?.tagName==='SMALL')movement.nextElementSibling.remove();movement.remove();}
 const zoomLabel=clone.querySelector('#zoom')?.closest('label');
 if(zoomLabel)zoomLabel.innerHTML='显示放大倍数<input id="zoom" type="range" min="0.25" max="60" step="0.25" value="0.5">';
 const viewport=clone.querySelector('#b3ImageViewport'),stage=clone.querySelector('#stage');if(viewport){viewport.before(stage);viewport.remove();}
 clone.querySelector('#b3ViewInfo')?.remove();clone.querySelector('#b3-zoom60-style')?.remove();clone.querySelector('#palette')?.replaceChildren();
 clone.querySelector('body')?.classList.remove('structure-pixel52','b3-panning','b3-dragging');
 download(new Blob(['<!doctype html>\n'+clone.outerHTML],{type:'text/html'}),'annotate_root_structure.html');
 status('已保存含当前标注的可编辑网页；原图仍按固定地址读取，也可选择匹配的本地原图。');
};
window.addEventListener('keydown',e=>{if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='z'&&$('tool52').value!=='pixel'&&!(e.target instanceof Element&&e.target.closest('input,textarea,select'))){e.preventDefault();e.stopImmediatePropagation();$('undo52').click()}},true);
window.structureReady52=(async()=>{await window.restoreEditorReadyV50;if(!structure52){const version=JSON.parse($('reference-original-versions-v50').textContent).original;structure52=validateStructure52(version.root_structure_annotation,version);rebuild52();}const undoImport=$('restoreUndoV50').onclick;$('restoreUndoV50').onclick=async()=>{await undoImport();structure52=structuredClone(window.restoredProjectMetadataV50.root_structure_annotation||null);rebuild52();selectedSegment52=selectedNode52=null;refresh52();render()};document.querySelector('h1').textContent='0353-2-B3 · 四期逐根身份与连接复核';document.title='四期逐根结构标注复核';refresh52();render();$('fit52').click();await compare52(1400,700);return {periods:structure52?.periods.length||0};})();
