import fs from 'node:fs';
import crypto from 'node:crypto';
const out='/workspace/LI/results/B3_root_cut_repair_v61_20261026';
const expected=JSON.parse(fs.readFileSync(out+'/source_progress_15.json','utf8'));
const digest=v=>crypto.createHash('sha256').update(JSON.stringify(v)).digest('hex');
const expectedLayers=expected.images.map(e=>({name:e.name,layers:Object.fromEntries(Object.entries(e.layers).map(([k,v])=>[k,digest(v)]))}));
let session;
const exceptions=[];const checks=[];
async function openPage(filename){
 const target=await(await fetch('http://127.0.0.1:8893/json/new?'+encodeURIComponent('http://127.0.0.1:8894/'+filename),{method:'PUT'})).json();
 const ws=new WebSocket(target.webSocketDebuggerUrl);await new Promise(r=>ws.onopen=r);let id=0;const pending=new Map();
 ws.onmessage=e=>{const d=JSON.parse(e.data);if(d.id){const h=pending.get(d.id);pending.delete(d.id);d.error?h.reject(Error(JSON.stringify(d.error))):h.resolve(d.result)}else if(d.method==='Runtime.exceptionThrown')exceptions.push(d.params.exceptionDetails)};
 const call=(method,params={})=>new Promise((resolve,reject)=>{const n=++id;pending.set(n,{resolve,reject});ws.send(JSON.stringify({id:n,method,params}))});
 const ev=async(expression)=>{const d=await call('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(d.exceptionDetails)throw Error(JSON.stringify(d.exceptionDetails));return d.result.value};
 await call('Runtime.enable');await call('Page.enable');await call('DOM.enable');await call('Emulation.setDeviceMetricsOverride',{width:1600,height:1050,deviceScaleFactor:1,mobile:false});
 for(let i=0;i<120;i++){if(await ev("typeof window.structureReady52!=='undefined'"))break;await new Promise(r=>setTimeout(r,500));}
 await ev('window.structureReady52');await ev('window.hierarchyReady54');await ev('window.rootCutReady55');await ev('window.continuityReady56');await ev('window.readableReady57');await ev('window.parentageReady58');await ev('window.temporalReady59');await ev('window.confidenceReady60');
 return {call,ev,ws,target};
}
function check(name,pass,detail){checks.push({name,pass:!!pass,...(detail?{detail}:{})});if(!pass)throw Error(name);console.log('PASS',name)}
async function layersExact(){return session.ev(`(async()=>{const expected=${JSON.stringify(expectedLayers)};const project=projectObject();return project.images.length===expected.length&&(await Promise.all(expected.map(async e=>{const v=project.images.find(x=>x.name===e.name);return v&&(await Promise.all(Object.entries(e.layers).map(async([k,h])=>(await sha(new TextEncoder().encode(JSON.stringify(v.layers[k]))))===h))).every(Boolean)}))).every(Boolean)})()`)}
async function restoreFile(path){
 const doc=await session.call('DOM.getDocument');const node=await session.call('DOM.querySelector',{nodeId:doc.root.nodeId,selector:'#projectfile'});await session.call('DOM.setFileInputFiles',{nodeId:node.nodeId,files:[path]});
 for(let n=0;n<120;n++){const s=await session.ev("({busy:$('projectfile').disabled,value:$('projectfile').value,status:$('status').textContent})");if(!s.busy&&!s.value&&!s.status.startsWith('正在核对'))return s;await new Promise(r=>setTimeout(r,100));}throw Error('Import did not finish');
}
async function clickNative(x,y,dragTo=null){
 const pos=await session.ev(`(()=>{const b=$('overlay').getBoundingClientRect();return {x:b.left+${x}*b.width/W,y:b.top+${y}*b.height/H}})()`);
 await session.call('Input.dispatchMouseEvent',{type:'mousePressed',button:'left',buttons:1,clickCount:1,...pos});
 let end=pos;
 if(dragTo){end=await session.ev(`(()=>{const b=$('overlay').getBoundingClientRect();return {x:b.left+${dragTo[0]}*b.width/W,y:b.top+${dragTo[1]}*b.height/H}})()`);await session.call('Input.dispatchMouseEvent',{type:'mouseMoved',button:'left',buttons:1,...end});}
 await session.call('Input.dispatchMouseEvent',{type:'mouseReleased',button:'left',buttons:0,clickCount:1,...end});
}
async function typeField(id,text){await session.ev(`$('${id}').focus();$('${id}').select()`);await session.call('Input.insertText',{text});}
async function clickButton(id){const p=await session.ev(`(()=>{const b=$('${id}');b.scrollIntoView({block:'nearest'});const r=b.getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2}})()`);await session.call('Input.dispatchMouseEvent',{type:'mousePressed',button:'left',buttons:1,clickCount:1,...p});await session.call('Input.dispatchMouseEvent',{type:'mouseReleased',button:'left',buttons:0,clickCount:1,...p});}
async function clean(){await restoreFile(out+'/source_progress_15.json');await session.ev('select(0)');}
async function makeCut(role='pending',target='',sync=false,frontMother=false){
 return session.ev(`(()=>{const f=frame52();let picked=null;for(const s of [...f.segments].sort((a,b)=>b.points.length-a.points.length)){if(s.points.length<80)continue;const a=s.points[Math.floor(s.points.length*.3)],b=s.points[Math.floor(s.points.length*.6)];if(a[0]<150||a[0]>2300||a[1]<100||a[1]>1000||b[0]<150||b[0]>2300||b[1]<100||b[1]>1000)continue;try{const plan=pathBetween55(f,s.root_id,a,b);if(plan.keys.length>10){picked={s,plan};break}}catch(e){}}if(!picked)throw Error('No native test root path');choose52(picked.s);cutRoot55=picked.s.root_id;cutSelection55=picked.plan.points[0];cutPreview55=picked.plan;$('cutRole55').value=${JSON.stringify(role)};$('syncCut55').checked=${sync};$('cutTarget55').value=${JSON.stringify(frontMother?'':target)};if(${frontMother}){$('motherLabel54').value=${JSON.stringify(target)};$('motherLabel54').dispatchEvent(new Event('input'));}window.cutOld55=picked.s.root_id;const result=commitCut55();return {result,oldRoot:window.cutOld55,id:selected52()?.root_id,label:selected52()?label54(selected52().root_id):null,status:$('visibleStatus53').textContent}})()`);
}
try{session=await openPage('diagnostic_editor.html');const result=await session.ev(`(()=>{const issues=[];for(const f of structure52.periods){for(const h of rawHierarchy54().values()){if(!assigned60(h.root_id)||!f.segments.some(s=>s.root_id===h.root_id))continue;const g=graph55(f,h.root_id),seen=new Set(),components=[];for(const k of g.nodes.keys()){if(seen.has(k))continue;const component=[],todo=[k];seen.add(k);while(todo.length){const q=todo.pop();component.push(q);for(const n of g.edges.get(q).keys())if(!seen.has(n)){seen.add(n);todo.push(n)}}if(component.length>=4)components.push(component);}components.sort((a,b)=>b.length-a.length);if(components.length>1){const a=g.nodes.get(components[0][Math.floor(components[0].length/2)]),b=g.nodes.get(components[1][Math.floor(components[1].length/2)]);let error=null;try{pathBetween55(f,h.root_id,a,b)}catch(e){error=e.message;}issues.push({period:f.period,root_id:h.root_id,label:h.label,component_sizes:components.map(c=>c.length),a,b,error});}}}return {disconnected_formal_root_periods:issues.length,issues}})()`);fs.writeFileSync(out+'/cut_failure_diagnosis.json',JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify({count:result.disconnected_formal_root_periods,examples:result.issues.slice(0,5)},null,2));}finally{if(session){session.ws.close();await fetch('http://127.0.0.1:8893/json/close/'+session.target.id)}}