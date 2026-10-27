import fs from 'node:fs';
import crypto from 'node:crypto';
const out='/workspace/LI/results/B3_left_region_review_v63_20261027';
const expected=JSON.parse(fs.readFileSync(out+'/source_progress_16.json','utf8'));
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
 await ev('window.structureReady52');await ev('window.hierarchyReady54');await ev('window.rootCutReady55');await ev('window.continuityReady56');await ev('window.readableReady57');await ev('window.parentageReady58');await ev('window.temporalReady59');await ev('window.confidenceReady60');await ev('window.detachReady61');await ev('window.freeCutReady62');await ev('window.regionReady63');
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
async function clean(){await restoreFile(out+'/source_progress_16.json');await session.ev('select(0)');}
async function makeCut(role='pending',target='',sync=false,frontMother=false){
 return session.ev(`(()=>{const f=frame52();let picked=null;for(const s of [...f.segments].sort((a,b)=>b.points.length-a.points.length)){if(s.points.length<80)continue;const a=s.points[Math.floor(s.points.length*.3)],b=s.points[Math.floor(s.points.length*.6)];if(a[0]<150||a[0]>2300||a[1]<100||a[1]>1000||b[0]<150||b[0]>2300||b[1]<100||b[1]>1000)continue;try{const plan=pathBetween55(f,s.root_id,a,b);if(plan.keys.length>10){picked={s,plan};break}}catch(e){}}if(!picked)throw Error('No native test root path');choose52(picked.s);cutRoot55=picked.s.root_id;cutSelection55=picked.plan.points[0];cutPreview55=picked.plan;$('cutRole55').value=${JSON.stringify(role)};$('syncCut55').checked=${sync};$('cutTarget55').value=${JSON.stringify(frontMother?'':target)};if(${frontMother}){$('motherLabel54').value=${JSON.stringify(target)};$('motherLabel54').dispatchEvent(new Event('input'));}window.cutOld55=picked.s.root_id;const result=commitCut55();return {result,oldRoot:window.cutOld55,id:selected52()?.root_id,label:selected52()?label54(selected52().root_id):null,status:$('visibleStatus53').textContent}})()`);
}
const uploaded=JSON.parse(fs.readFileSync(out+'/source_progress_16.json','utf8'));const originalDefinitions=uploaded.root_structure_annotation.root_track_hierarchy.filter(h=>h.annotation_status_v60!=='unassigned').map(h=>({root_id:h.root_id,label:h.label,parent_root_id:h.parent_root_id,definition:h.human_identity_definition_v60||null}));
async function definitionsExact(){return session.ev(`(()=>{const expected=${JSON.stringify(originalDefinitions)},map=rawHierarchy54();return expected.every(e=>{const h=map.get(e.root_id);return h&&h.label===e.label&&h.parent_root_id===e.parent_root_id&&JSON.stringify(h.human_identity_definition_v60||null)===JSON.stringify(e.definition)})})()`)}
async function ownerMaps(){return session.ev("JSON.stringify(serialize52().periods.map(f=>f.owner_map_rle))")}
async function selectIssue(){const issue=JSON.parse(fs.readFileSync(out+'/cut_failure_diagnosis.json','utf8')).issues.find(e=>e.period==='T1'&&e.component_sizes[1]>25);await session.ev(`(async()=>{await select(0);choose52(frame52().segments.find(s=>s.root_id===${JSON.stringify(issue.root_id)}));$('fit52').click()})()`);return issue;}
try{
 session=await openPage('annotate_test.html');
 const project=JSON.parse(fs.readFileSync(out+'/review_project.json','utf8')),proposals=project.root_structure_annotation.regional_review_v63.proposals;
 check('all 135 uploaded human definitions and their mothers/history are preserved',await definitionsExact()&&await session.ev("[...rawHierarchy54().values()].filter(h=>h.annotation_status_v60==='user_defined').length===135"));
 check('original four period root pixel labels are unchanged',await layersExact());
 const initialOwners=JSON.stringify(uploaded.root_structure_annotation.periods.map(f=>f.owner_map_rle));
 check('all four single-owner maps are exact before editing',await ownerMaps()===initialOwners);
 check('all 12 proposals load visibly as assistant proposals rather than human confirmations',await session.ev("proposals63().length===12&&proposals63().every(p=>rawHierarchy54().get(p.root_id).annotation_status_v60==='assistant_proposed_v63')&&$('proposalSelect63').options.length===12"));
 check('two proposed laterals use actual pre-existing mothers',await session.ev("resolve54('0004-1')&&resolve54('0007-7')&&rawHierarchy54().get(resolve54('0004-1')).parent_root_id===resolve54('0004')&&rawHierarchy54().get(resolve54('0007-7')).parent_root_id===resolve54('0007')"));
 const first=proposals[0];await session.ev(`$('proposalSelect63').value=${JSON.stringify(first.root_id)}`);await clickButton('focusProposal63');
 check('real focus button selects the full reviewed axis with unconfirmed wording',await session.ev(`selectedRoot60()===${JSON.stringify(first.root_id)}&&$('selection60').textContent.includes('待你复核')`));
 await new Promise(r=>setTimeout(r,400));
 check('proposed root labels have distinct dashed frames',await session.ev(`!!labelSvg57.querySelector('g[data-root57="${first.root_id}"][data-proposal63="true"] rect[stroke-dasharray]')`));
 await clickButton('acceptProposal63');
 check('explicit user confirmation records a real user action without changing earlier definitions',await session.ev(`rawHierarchy54().get(${JSON.stringify(first.root_id)}).annotation_status_v60==='user_defined'&&rawHierarchy54().get(${JSON.stringify(first.root_id)}).human_identity_definition_v60.user_action===true`)&&await definitionsExact()&&await layersExact());
 await clickButton('undoPath60');
 check('undo restores an accepted proposal to its original pending status',await session.ev(`rawHierarchy54().get(${JSON.stringify(first.root_id)}).annotation_status_v60==='assistant_proposed_v63'`));
 const second=proposals[1];await session.ev(`$('proposalSelect63').value=${JSON.stringify(second.root_id)};updateRegion63()`);await clickButton('rejectProposal63');
 check('reject preserves the complete cross-period trace as unassigned rather than deleting pixels',await session.ev(`rawHierarchy54().get(${JSON.stringify(second.root_id)}).annotation_status_v60==='unassigned'&&label54(${JSON.stringify(second.root_id)}).startsWith('U-')&&structure52.periods.every(f=>f.segments.some(s=>s.root_id===${JSON.stringify(second.root_id)}))`)&&await layersExact()&&await ownerMaps()===initialOwners);
 await clickButton('undoPath60');check('rejected proposal can be restored by undo',await session.ev(`rawHierarchy54().get(${JSON.stringify(second.root_id)}).label==='0007-7'&&rawHierarchy54().get(${JSON.stringify(second.root_id)}).annotation_status_v60==='assistant_proposed_v63'`));
 const uncertain=proposals.find(p=>p.label==='0033');await session.ev(`$('proposalSelect63').value=${JSON.stringify(uncertain.root_id)};updateRegion63()`);await session.ev('select(3)');await clickButton('focusProposal63');
 check('unresolved T4 correspondence remains absent and explicitly marked',await session.ev(`!frame52().segments.some(s=>s.root_id===${JSON.stringify(uncertain.root_id)})&&$('proposalInfo63').textContent.includes('尚未确定')`));
 // Actual four-period controls retain a proposal ID and never invent missing geometry.
 const full=proposals.find(p=>p.label==='0032');await session.ev(`$('proposalSelect63').value=${JSON.stringify(full.root_id)};updateRegion63()`);await clickButton('focusProposal63');
 const seen=[];for(let i=0;i<4;i++){await session.ev(`select(${i})`);await clickButton('focusProposal63');seen.push(await session.ev(`selectedRoot60()===${JSON.stringify(full.root_id)}&&frame52().segments.some(s=>s.root_id===${JSON.stringify(full.root_id)})`));}
 check('same proposed root can be selected with the same identity in all four periods',seen.every(Boolean));
 await session.ev('select(0)');await session.ev(`$('proposalSelect63').value=${JSON.stringify(first.root_id)};updateRegion63()`);await clickButton('focusProposal63');await clickButton('acceptProposal63');
 await session.ev("window.download61=[];download=async(blob,name)=>window.download61.push({name,text:await blob.text()});$('saveReview60').click()");let saved;for(let i=0;i<150;i++){saved=await session.ev('window.download61[0]||null');if(saved)break;await new Promise(r=>setTimeout(r,100));}check('regional work exports as full editable JSON',!!saved?.text);fs.writeFileSync(out+'/regional_test.json',saved.text);await restoreFile(out+'/regional_test.json');
 check('JSON restore retains the explicit decision and all original human annotations',await session.ev(`rawHierarchy54().get(${JSON.stringify(first.root_id)}).annotation_status_v60==='user_defined'&&structure52.regional_review_v63.user_decisions.length>0`)&&await definitionsExact()&&await layersExact());
 // Existing arbitrary cutter is present and works on the new annotation file.
 await restoreFile(out+'/review_project.json');await session.ev('select(0)');await session.ev(`$('proposalSelect63').value=${JSON.stringify(full.root_id)};updateRegion63()`);await clickButton('focusProposal63');await clickButton('startFreeCut62');
 const point=await session.ev(`(()=>{const s=frame52().segments.find(s=>s.root_id===${JSON.stringify(full.root_id)});return s.points[Math.floor(s.points.length/2)]})()`);await clickNative(...point);check('arbitrary cut preview works on a newly proposed root',await session.ev('freeCut62.cuts.length===1'));await clickButton('commitFreeCut62');
 check('new regional data remains compatible with actual arbitrary cutting',await session.ev("frame52().manual_free_cuts_v62.length>0&&!$('freeCutInfo62').textContent.includes('失败')")&&await definitionsExact()&&await layersExact());
 await restoreFile(out+'/review_project.json');
 await session.ev("window.download61=[];download=async(blob,name)=>window.download61.push({name,text:await blob.text()});$('saveReviewHtml60').click()");let html;for(let i=0;i<150;i++){html=await session.ev('window.download61[0]||null');if(html)break;await new Promise(r=>setTimeout(r,100));}check('standalone editable HTML exports with the regional review',!!html?.text);fs.writeFileSync(out+'/saved_regional_test.html',html.text);
 session.ws.close();await fetch('http://127.0.0.1:8893/json/close/'+session.target.id);session=await openPage('saved_regional_test.html');
 check('saved HTML reopens one review panel and preserves all proposals and original definitions',await session.ev("document.querySelectorAll('#regionPanel63').length===1&&proposals63().length===12&&proposals63().every(p=>rawHierarchy54().get(p.root_id).annotation_status_v60==='assistant_proposed_v63')")&&await definitionsExact()&&await layersExact());
 check('full project validates after save and restore',await session.ev('(()=>{const p=projectObject();validateStructure52(p.root_structure_annotation,p);return true})()'));
 check('no browser runtime exceptions',exceptions.length===0,exceptions);
 fs.writeFileSync(out+'/browser_verification.json',JSON.stringify({passed:true,checks,exceptions,actual_browser:'Chromium real buttons and native pointer, JSON file input and saved HTML reopening',tested_editor_SHA256:crypto.createHash('sha256').update(fs.readFileSync(out+'/annotate.html')).digest('hex')},null,2)+'\n');console.log(JSON.stringify({passed:true,checks:checks.length}));
}catch(error){fs.writeFileSync(out+'/browser_verification_failed.json',JSON.stringify({checks,exceptions,error:error.stack},null,2)+'\n');console.error(error);process.exitCode=1;}
finally{if(session){session.ws.close();await fetch('http://127.0.0.1:8893/json/close/'+session.target.id).catch(()=>{});}}
