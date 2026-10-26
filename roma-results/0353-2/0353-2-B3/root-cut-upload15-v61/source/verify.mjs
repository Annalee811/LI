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
 await ev('window.structureReady52');await ev('window.hierarchyReady54');await ev('window.rootCutReady55');await ev('window.continuityReady56');await ev('window.readableReady57');await ev('window.parentageReady58');await ev('window.temporalReady59');await ev('window.confidenceReady60');await ev('window.detachReady61');
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
const uploaded=JSON.parse(fs.readFileSync(out+'/source_progress_15.json','utf8'));const originalDefinitions=uploaded.root_structure_annotation.root_track_hierarchy.filter(h=>h.annotation_status_v60!=='unassigned').map(h=>({root_id:h.root_id,label:h.label,parent_root_id:h.parent_root_id,definition:h.human_identity_definition_v60||null}));
async function definitionsExact(){return session.ev(`(()=>{const expected=${JSON.stringify(originalDefinitions)},map=rawHierarchy54();return expected.every(e=>{const h=map.get(e.root_id);return h&&h.label===e.label&&h.parent_root_id===e.parent_root_id&&JSON.stringify(h.human_identity_definition_v60||null)===JSON.stringify(e.definition)})})()`)}
async function ownerMaps(){return session.ev("JSON.stringify(serialize52().periods.map(f=>f.owner_map_rle))")}
async function selectIssue(){const issue=JSON.parse(fs.readFileSync(out+'/cut_failure_diagnosis.json','utf8')).issues.find(e=>e.period==='T1'&&e.component_sizes[1]>25);await session.ev(`(async()=>{await select(0);choose52(frame52().segments.find(s=>s.root_id===${JSON.stringify(issue.root_id)}));$('fit52').click()})()`);return issue;}
try{
 session=await openPage('annotate_test.html');
 check('upload15 editor initializes with all 62 user definitions and their parents preserved',await definitionsExact()&&await layersExact()&&await session.ev("[...rawHierarchy54().values()].filter(h=>h.annotation_status_v60==='user_defined').length===62"));
 check('new path-independent detach controls are available',await session.ev("!!$('startDetach61')&&!!$('detachScope61')&&!!$('commitDetach61')"));
 const originalOwners=await ownerMaps(),issue=await selectIssue();
 await session.ev("$('startReviewCut60').closest('details').open=true");await clickButton('startReviewCut60');await clickNative(...issue.a);await clickNative(...issue.b);
 check('actual disconnected-root two-point failure remains visible with correct remedy',await session.ev("!cutPreview55&&$('reviewStatus60').textContent.includes('连续路径')&&$('reviewStatus60').textContent.includes('点选分离')"));
 check('failed cut has not changed user definitions or foreground pixels',await definitionsExact()&&await layersExact()&&await ownerMaps()===originalOwners);
 await clickButton('startDetach61');await session.ev("$('detachScope61').value='component';$('detachScope61').dispatchEvent(new Event('change'))");
 const pick=await session.ev(`(()=>{const f=frame52(),r=${JSON.stringify(issue.root_id)},g=graph55(f,r),seen=new Set(),comp=[];for(const key of g.nodes.keys()){if(seen.has(key))continue;const keys=[],todo=[key];seen.add(key);while(todo.length){const k=todo.pop();keys.push(k);for(const n of g.edges.get(k).keys())if(!seen.has(n)){seen.add(n);todo.push(n)}}if(keys.length>20)comp.push(keys);}comp.sort((a,b)=>b.length-a.length);for(const k of comp[1]){const p=g.nodes.get(k),s=ownerAt52(f,...p);if(s?.root_id===r&&p[0]>50&&p[0]<2400&&p[1]>50&&p[1]<1050)return {point:p,segment:s.id};}throw Error('no owned point in second component')})()`);await clickNative(...pick.point);
 check('one real click previews disconnected parts without requiring a continuous path',await session.ev("detach61.parts.size>0&&!$('commitDetach61').disabled"));
 const preview=await session.ev('JSON.stringify([...detach61.parts])');await clickNative(...pick.point);
 check('clicking the same component again deselects it without mutation',await session.ev("detach61.parts.size===0&&$('commitDetach61').disabled")&&await layersExact());await clickNative(...pick.point);
 const partIds=JSON.parse(await session.ev('JSON.stringify([...detach61.parts])'));const rootBefore=issue.root_id;await clickButton('commitDetach61');
 check('confirmation actually separates those previously uncuttable parts',await session.ev(`selectedRoot60()!==${JSON.stringify(rootBefore)}&&!assigned60(selectedRoot60())&&label54(selectedRoot60()).startsWith('U-')`));
 const detached=await session.ev('selectedRoot60()');
 check('every selected segment is assigned to the new undefined candidate',await session.ev(`frame52().segments.filter(s=>${JSON.stringify(partIds)}.includes(s.id)).every(s=>s.root_id===${JSON.stringify(detached)})`));
 check('point-detach preserves foreground width pixel classes and single owner maps',await layersExact()&&await ownerMaps()===originalOwners);
 check('all existing manual numbers and mothers stay unchanged after detach',await definitionsExact());
 check('other three periods retain all their original segment identity assignments',await session.ev(`(()=>{const source=${JSON.stringify(uploaded.root_structure_annotation.periods.slice(1).map(f=>f.segments.map(s=>[s.id,s.root_id])))};return structure52.periods.slice(1).every((f,i)=>JSON.stringify(f.segments.map(s=>[s.id,s.root_id]))===JSON.stringify(source[i]))})()`));
 check('detached project validates as complete editable data',await session.ev("(()=>{const p=projectObject();validateStructure52(p.root_structure_annotation,p);return true})()"));
 await session.ev("$('defineKind60').value='side';$('defineMother60').value='0001';$('defineLabel60').value=''");await clickButton('defineApply60');
 check('separated piece can enter the existing mother-root numbering system',await session.ev(`assigned60(${JSON.stringify(detached)})&&rawHierarchy54().get(${JSON.stringify(detached)}).parent_root_id===resolve54('0001')&&label54(${JSON.stringify(detached)}).startsWith('0001-')`));
 await session.ev("window.download61=[];download=async(blob,name)=>window.download61.push({name,text:await blob.text()});$('saveReview60').click()");let saved;for(let i=0;i<150;i++){saved=await session.ev('window.download61[0]||null');if(saved)break;await new Promise(r=>setTimeout(r,100));}check('corrected progress exports a complete JSON file',!!saved?.text);fs.writeFileSync(out+'/detached_test.json',saved.text);await restoreFile(out+'/detached_test.json');
 check('JSON restore keeps split identities and the user-assigned mother',await session.ev(`rawHierarchy54().get(${JSON.stringify(detached)}).parent_root_id===resolve54('0001')&&structure52.manual_detachments_v61.length>0`)&&await layersExact());
 await clean();await selectIssue();await clickButton('startDetach61');await session.ev("$('detachScope61').value='segment';$('detachScope61').dispatchEvent(new Event('change'))");await clickNative(...pick.point);
 check('single-original-segment mode supports precise multi-part separation',await session.ev('detach61.parts.size===1'));
 const other=await session.ev("(()=>{const r=detach61.root,s=frame52().segments.find(s=>s.root_id!==r&&s.length_px>40&&ownerAt52(frame52(),...s.points[Math.floor(s.points.length/2)])?.root_id===s.root_id);return s.points[Math.floor(s.points.length/2)]})()");const selectedCount=await session.ev('detach61.parts.size');await clickNative(...other);
 check('clicking a different root is rejected without accidentally moving it',await session.ev(`detach61.parts.size===${selectedCount}&&$('reviewStatus60').textContent.includes('另一条根')`));
 await session.ev("$('detachScope61').value='route';$('detachScope61').dispatchEvent(new Event('change'))");await clickNative(...pick.point);check('direction-based preview only selects parts of the chosen root',await session.ev("detach61.parts.size>0&&frame52().segments.filter(s=>detach61.parts.has(s.id)).every(s=>s.root_id===detach61.root)"));
 await clickButton('cancelDetach61');check('cancellation discards preview without data changes',await session.ev('detach61===null')&&await definitionsExact()&&await layersExact());
 await clickButton('startDetach61');await session.ev('select(1)');check('changing period cancels unfinished selection and cut state',await session.ev('detach61===null&&cutPreview55===null'));
 await clean();await selectIssue();await clickButton('startDetach61');await session.ev("$('detachScope61').value='component';$('detachScope61').dispatchEvent(new Event('change'))");await clickNative(...pick.point);await clickButton('commitDetach61');await clickButton('undoPath60');
 check('undo restores the original upload15 root assignments and definitions',await definitionsExact()&&await layersExact()&&await ownerMaps()===originalOwners&&await session.ev(`JSON.stringify(structure52.periods.map(f=>f.segments.map(s=>[s.id,s.root_id])))===${JSON.stringify(JSON.stringify(uploaded.root_structure_annotation.periods.map(f=>f.segments.map(s=>[s.id,s.root_id]))))}`));
 await clean();
 const connected=await session.ev("(()=>{for(const s of frame52().segments){if(s.points.length<100)continue;const a=s.points[Math.floor(s.points.length*.3)],b=s.points[Math.floor(s.points.length*.6)];if(a[0]<100||a[0]>2300||a[1]<80||a[1]>1000)continue;try{const p=pathBetween55(frame52(),s.root_id,a,b);if(p.keys.length>10){choose52(s);$('fit52').click();return {a,b,root:s.root_id};}}catch(e){}}throw Error('no connected path')})()");await session.ev("$('startReviewCut60').closest('details').open=true");await clickButton('startReviewCut60');await clickNative(...connected.a);await clickNative(...connected.b);
 check('continuous interior two-point cut still previews successfully',await session.ev("!!cutPreview55&&!$('applyReviewCut60').disabled"));await clickButton('applyReviewCut60');
 check('continuous cut still creates an undefined candidate while preserving pixel layers',await session.ev(`selectedRoot60()!==${JSON.stringify(connected.root)}&&!assigned60(selectedRoot60())`)&&await layersExact());
 await clean();await selectIssue();await clickButton('startDetach61');await session.ev("$('detachScope61').value='component';$('detachScope61').dispatchEvent(new Event('change'))");await clickNative(...pick.point);await clickButton('commitDetach61');const splitId=await session.ev('selectedRoot60()');await clickButton('startDetach61');await session.ev("window.download61=[];download=async(blob,name)=>window.download61.push({name,text:await blob.text()});$('saveReviewHtml60').click()");let html;for(let i=0;i<150;i++){html=await session.ev('window.download61[0]||null');if(html)break;await new Promise(r=>setTimeout(r,100));}check('split project exports an editable standalone HTML',!!html?.text);fs.writeFileSync(out+'/saved_editor_test.html',html.text);
 session.ws.close();await fetch('http://127.0.0.1:8893/json/close/'+session.target.id);session=await openPage('saved_editor_test.html');
 check('saved HTML reopens the actual split and all original manual definitions',await definitionsExact()&&await layersExact()&&await session.ev(`rawHierarchy54().has(${JSON.stringify(splitId)})&&!assigned60(${JSON.stringify(splitId)})&&document.querySelectorAll('#detachPanel61').length===1`));
 check('reopened saved preview never leaves root labels hidden by an unfinished selection',await session.ev("detach61===null&&!document.body.classList.contains('detaching61')"));
 check('no browser runtime exceptions',exceptions.length===0,exceptions);fs.writeFileSync(out+'/browser_verification.json',JSON.stringify({passed:true,checks,exceptions,source_upload:'root_annotation_progress (15).json',actual_browser:'Chromium real mouse, keyboard, file inputs and HTML reopening',tested_editor_SHA256:crypto.createHash('sha256').update(fs.readFileSync(out+'/annotate.html')).digest('hex')},null,2)+'\n');console.log(JSON.stringify({passed:true,checks:checks.length}));
}catch(error){fs.writeFileSync(out+'/browser_verification_failed.json',JSON.stringify({checks,exceptions,error:error.stack},null,2)+'\n');console.error(error);process.exitCode=1;}
finally{if(session){session.ws.close();await fetch('http://127.0.0.1:8893/json/close/'+session.target.id).catch(()=>{});}}
