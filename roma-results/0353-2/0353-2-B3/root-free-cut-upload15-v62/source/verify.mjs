import fs from 'node:fs';
import crypto from 'node:crypto';
const out='/workspace/LI/results/B3_root_free_cut_v62_20261027';
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
 await ev('window.structureReady52');await ev('window.hierarchyReady54');await ev('window.rootCutReady55');await ev('window.continuityReady56');await ev('window.readableReady57');await ev('window.parentageReady58');await ev('window.temporalReady59');await ev('window.confidenceReady60');await ev('window.detachReady61');await ev('window.freeCutReady62');
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
 check('upload15 layers and all formal definitions are unchanged on load',await layersExact()&&await definitionsExact());
 check('arbitrary cut controls are present',await session.ev("!!$('startFreeCut62')&&!!$('cutAndDetach62')&&!!$('detachPieces62')"));
 await session.ev("window.initial62=serialize52();window.initialRootOwners62=structure52.periods.map(f=>{const c=cache52(f);return Array.from(c.owner,i=>i?c.byIndex.get(i).root_id:null)});$('fit52').click()");
 const picked=await session.ev("(()=>{const f=frame52();for(const s of [...f.segments].sort((a,b)=>b.points.length-a.points.length)){if(s.points.length<100)continue;const aa=Math.floor(s.points.length*.3),bb=Math.floor(s.points.length*.65),a=s.points[aa],b=s.points[bb];if([a,b].some(p=>p[0]<60||p[0]>2400||p[1]<60||p[1]>1030))continue;if(ownerAt52(f,...a)?.id===s.id&&ownerAt52(f,...b)?.id===s.id){choose52(s);return {id:s.id,root:s.root_id,a,b,aa,bb,count:f.segments.length,points:s.points}}}throw Error('no test segment')})()");
 await clickButton('startFreeCut62');await clickNative(...picked.a);
 check('real mouse click previews an interior cut without requiring two endpoints',await session.ev('freeCut62.cuts.length===1&&!$("commitFreeCut62").disabled'));
 await clickNative(...picked.a);check('same cut click toggles preview off without editing',await session.ev('freeCut62.cuts.length===0')&&await ownerMaps()===JSON.stringify(uploaded.root_structure_annotation.periods.map(f=>f.owner_map_rle)));
 await clickNative(...picked.a);await clickNative(...picked.b);
 check('two arbitrary locations on one native segment can both be previewed',await session.ev('freeCut62.cuts.length===2'));
 await clickButton('cutAndDetach62');
 check('two interior cuts create three individually selectable pieces',await session.ev(`frame52().segments.length===${picked.count+2}&&frame52().segments.filter(s=>s.manual_cut_source_v62===${JSON.stringify(picked.id)}).length===3`));
 check('cut and detach button enters precise single-piece mode',await session.ev("freeCut62===null&&!!detach61&&$('detachScope61').value==='segment'"));
 check('native centerline vertices are conserved exactly in order',await session.ev(`JSON.stringify(frame52().segments.filter(s=>s.manual_cut_source_v62===${JSON.stringify(picked.id)}).flatMap(s=>s.points))===${JSON.stringify(JSON.stringify(picked.points))}`));
 check('cutting keeps every pixel on its original biological root identity',await session.ev('structure52.periods.every((f,k)=>{const c=cache52(f);return c.owner.every((i,j)=>(i?c.byIndex.get(i).root_id:null)===window.initialRootOwners62[k][j])})')&&await layersExact()&&await definitionsExact());
 check('the three other periods remain structurally byte-exact',await session.ev('JSON.stringify(serialize52().periods.slice(1))===JSON.stringify(window.initial62.periods.slice(1))'));
 const middle=await session.ev(`(()=>{const ss=frame52().segments.filter(s=>s.manual_cut_source_v62===${JSON.stringify(picked.id)}),s=ss[1];window.middle62=s.id;window.middleIndex62=s.index;window.siblingIds62=ss.filter(q=>q!==s).map(q=>q.id);return {point:s.points[Math.floor(s.points.length*.5)],id:s.id}})()`);
 await clickNative(...middle.point);
 check('clicking middle selects only that newly cut piece',await session.ev(`detach61.parts.size===1&&detach61.parts.has(${JSON.stringify(middle.id)})`));
 await clickButton('commitDetach61');const newRoot=await session.ev('selectedRoot60()');
 check('confirm detaches only the middle as undefined while both ends keep original identity',await session.ev(`!assigned60(${JSON.stringify(newRoot)})&&cache52(frame52()).byId.get(window.middle62).root_id===${JSON.stringify(newRoot)}&&window.siblingIds62.every(id=>cache52(frame52()).byId.get(id).root_id===${JSON.stringify(picked.root)})`));
 check('foreground and pixel classes remain exact after cutting and detaching',await layersExact());
 check('pixel ownership has no holes or duplicate owners',await session.ev('structure52.periods.every((f,k)=>{const c=cache52(f);return c.owner.every((i,j)=>(i>0)===(window.initialRootOwners62[k][j]!==null)&&(!i||c.byIndex.has(i)))})'));
 check('full edited project validates',await session.ev('(()=>{const p=projectObject();validateStructure52(p.root_structure_annotation,p);return true})()'));
 await session.ev("$('defineKind60').value='side';$('defineMother60').value='0001';$('defineLabel60').value=''");await clickButton('defineApply60');
 check('cut piece can receive a mother-root side number',await session.ev(`rawHierarchy54().get(${JSON.stringify(newRoot)}).parent_root_id===resolve54('0001')&&assigned60(${JSON.stringify(newRoot)})`));
 // Export real JSON via its browser control, then restore through actual file input.
 await session.ev("window.download61=[];download=async(blob,name)=>window.download61.push({name,text:await blob.text()});$('saveReview60').click()");let saved;
 for(let i=0;i<150;i++){saved=await session.ev('window.download61[0]||null');if(saved)break;await new Promise(r=>setTimeout(r,100));}
 check('cut and detached data exports',!!saved?.text);fs.writeFileSync(out+'/cut_detach_test.json',saved.text);await restoreFile(out+'/cut_detach_test.json');
 check('JSON restore retains cuts separated identity and assigned mother',await session.ev(`frame52().manual_free_cuts_v62.length===1&&cache52(frame52()).byId.get(${JSON.stringify(middle.id)}).root_id===${JSON.stringify(newRoot)}&&rawHierarchy54().get(${JSON.stringify(newRoot)}).parent_root_id===resolve54('0001')`)&&await layersExact());
 await clean();await session.ev(`choose52(cache52(frame52()).byId.get(${JSON.stringify(picked.id)}));$('fit52').click()`);await clickButton('startFreeCut62');await clickNative(...picked.a);await clickButton('commitFreeCut62');
 check('single-point cut works and does not assign new formal identities',await session.ev(`frame52().segments.length===${picked.count+1}`)&&await definitionsExact());
 await clickButton('undoPath60');
 check('one undo restores exact original segment geometry and owner map',await session.ev(`JSON.stringify(serialize52().periods.map(f=>({segments:f.segments,nodes:f.nodes,owners:f.owner_map_rle})))===${JSON.stringify(JSON.stringify(uploaded.root_structure_annotation.periods.map(f=>({segments:f.segments,nodes:f.nodes,owners:f.owner_map_rle}))))}`));
 await clean();await session.ev("$('fit52').click()");await clickButton('startFreeCut62');
 const issue=JSON.parse(fs.readFileSync(out+'/cut_failure_diagnosis.json')).issues.find(e=>e.period==='T1'&&e.component_sizes[1]>25);
 await clickNative(...issue.a);await clickNative(...issue.b);check('disconnected locations accept two independent cuts without a path search',await session.ev('freeCut62.cuts.length===2'));
 await clickButton('commitFreeCut62');check('disconnected-location cuts commit and retain all formal definitions',await session.ev("freeCut62===null&&frame52().manual_free_cuts_v62.length===1&&!$('freeCutInfo62').textContent.includes('失败')")&&await definitionsExact()&&await layersExact());
 await clean();await session.ev("$('fit52').click()");await clickButton('startFreeCut62');await clickNative(...picked.a);await session.ev('select(1)');check('switching periods cancels cut preview',await session.ev('freeCut62===null'));
 await clean();await session.ev("$('fit52').click()");await clickButton('startFreeCut62');await clickNative(...picked.a);await clickButton('cancelFreeCut62');check('cancel leaves original data unchanged',await session.ev('freeCut62===null')&&await definitionsExact()&&await layersExact());
 // Endpoint cuts must work even for already short segments and remove only their connections.
 const endCase=await session.ev("(()=>{for(const s of frame52().segments){for(const at of [0,s.points.length-1]){const p=s.points[at];if(p[0]>60&&p[0]<2400&&p[1]>60&&p[1]<1030&&ownerAt52(frame52(),...p)?.id===s.id)return {id:s.id,p,at,count:frame52().segments.length}}}throw Error('no endpoint')})()");
 await clickButton('startFreeCut62');await clickNative(...endCase.p);await session.ev(`(()=>{const c=freeCut62.cuts.at(-1);Object.assign(c,c.candidates.find(q=>q.segment_id===${JSON.stringify(endCase.id)}));})()`);await clickButton('commitFreeCut62');
 check('existing root endpoint accepts a topological cut without deleting a segment',await session.ev(`frame52().segments.length===${endCase.count}&&frame52().manual_free_cuts_v62.at(-1).endpoints.some(e=>e.segment_id===${JSON.stringify(endCase.id)})`)&&await layersExact());
 // Crossing ownership choice and selective endpoint connection removal.
 await clean();await session.ev("$('fit52').click()");
 const junction=await session.ev("(()=>{for(const n of frame52().nodes){if(n.connections.length<2)continue;for(const pair of n.connections)for(const armid of pair){const arm=n.arms.find(a=>a.id===armid),s=cache52(frame52()).byId.get(arm.segment_id);if(!s)continue;for(const at of [0,s.points.length-1]){const p=s.points[at];if(p[0]<60||p[0]>2400||p[1]<60||p[1]>1030||Math.hypot(p[0]-arm.x,p[1]-arm.y)>=5)continue;return {node:n.id,segment:s.id,p,before:structuredClone(n.connections),remove:n.arms.filter(a=>a.segment_id===s.id&&Math.hypot(a.x-p[0],a.y-p[1])<10).map(a=>a.id)};}}}throw Error('no crossing endpoint')})()");
 await clickButton('startFreeCut62');await clickNative(...junction.p);await session.ev(`$('freeCutChoice62').value=${JSON.stringify(junction.segment)};$('freeCutChoice62').dispatchEvent(new Event('change'))`);
 check('crossing dropdown can explicitly choose the intended root segment',await session.ev(`freeCut62.cuts.at(-1).segment_id===${JSON.stringify(junction.segment)}`));
 await clickButton('commitFreeCut62');
 check('endpoint cut removes only selected-arm connections and retains the other crossing root',await session.ev(`JSON.stringify(frame52().nodes.find(n=>n.id===${JSON.stringify(junction.node)}).connections)===${JSON.stringify(JSON.stringify(junction.before.filter(pair=>!pair.some(id=>junction.remove.includes(id)))))}`)&&await layersExact());
 // Native minimum-length pieces are valid; no old 3-point/4-point exclusion.
 await clean();await session.ev("$('fit52').click()");await clickButton('startFreeCut62');
 await session.ev(`(()=>{const s=cache52(frame52()).byId.get(${JSON.stringify(picked.id)});freeCut62.cuts=[{...cutCandidate62(s,...s.points[1]),candidates:[cutCandidate62(s,...s.points[1])]},{...cutCandidate62(s,...s.points[2]),candidates:[cutCandidate62(s,...s.points[2])]}];updateFreeCut62();})()`);await clickButton('commitFreeCut62');
 check('adjacent interior cuts can isolate a one-point piece without losing pixels',await session.ev(`frame52().segments.filter(s=>s.manual_cut_source_v62===${JSON.stringify(picked.id)}).some(s=>s.points.length===1)`)&&await layersExact());
 const shortId=await session.ev(`(()=>{const s=frame52().segments.find(s=>s.manual_cut_source_v62===${JSON.stringify(picked.id)}&&s.points.length===1);choose52(s);return s.id})()`);await clickButton('detachPieces62');await session.ev(`$('pieceSelect62').value=${JSON.stringify(shortId)}`);await clickButton('togglePiece62');
 check('single-point or occluded piece can be selected by segment list without hitting its pixels',await session.ev(`detach61.parts.size===1&&detach61.parts.has(${JSON.stringify(shortId)})`));await clickButton('cancelDetach61');
 // Saved editor must replay new code, preserve confirmed cuts, and discard unfinished previews.
 await clickButton('startFreeCut62');await session.ev("window.download61=[];download=async(blob,name)=>window.download61.push({name,text:await blob.text()});$('saveReviewHtml60').click()");let html;
 for(let i=0;i<150;i++){html=await session.ev('window.download61[0]||null');if(html)break;await new Promise(r=>setTimeout(r,100));}
 check('modified editor can export standalone HTML',!!html?.text);fs.writeFileSync(out+'/saved_freecut_test.html',html.text);session.ws.close();await fetch('http://127.0.0.1:8893/json/close/'+session.target.id);session=await openPage('saved_freecut_test.html');
 check('saved HTML restores confirmed cuts and discards preview-only modes',await session.ev("frame52().manual_free_cuts_v62.length===1&&freeCut62===null&&!document.body.classList.contains('freeCutting62')&&document.querySelectorAll('#freeCutPanel62').length===1")&&await definitionsExact()&&await layersExact());
 check('no browser runtime exceptions',exceptions.length===0,exceptions);
 fs.writeFileSync(out+'/browser_verification.json',JSON.stringify({passed:true,checks,exceptions,actual_browser:'Chromium real mouse + actual file restore + saved HTML reopening',tested_editor_SHA256:crypto.createHash('sha256').update(fs.readFileSync(out+'/annotate.html')).digest('hex')},null,2)+'\n');console.log(JSON.stringify({passed:true,checks:checks.length}));
}catch(error){fs.writeFileSync(out+'/browser_verification_failed.json',JSON.stringify({checks,exceptions,error:error.stack},null,2)+'\n');console.error(error);process.exitCode=1;}
finally{if(session){session.ws.close();await fetch('http://127.0.0.1:8893/json/close/'+session.target.id).catch(()=>{});}}
