import fs from 'node:fs';
import crypto from 'node:crypto';
const out='/workspace/LI/results/B3_root_whole_trace_v60_20261026';
const expected=JSON.parse(fs.readFileSync(out+'/review_project.json','utf8'));
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
async function clean(){await restoreFile(out+'/review_project.json');await session.ev('select(0)');}
async function makeCut(role='pending',target='',sync=false,frontMother=false){
 return session.ev(`(()=>{const f=frame52();let picked=null;for(const s of [...f.segments].sort((a,b)=>b.points.length-a.points.length)){if(s.points.length<80)continue;const a=s.points[Math.floor(s.points.length*.3)],b=s.points[Math.floor(s.points.length*.6)];if(a[0]<150||a[0]>2300||a[1]<100||a[1]>1000||b[0]<150||b[0]>2300||b[1]<100||b[1]>1000)continue;try{const plan=pathBetween55(f,s.root_id,a,b);if(plan.keys.length>10){picked={s,plan};break}}catch(e){}}if(!picked)throw Error('No native test root path');choose52(picked.s);cutRoot55=picked.s.root_id;cutSelection55=picked.plan.points[0];cutPreview55=picked.plan;$('cutRole55').value=${JSON.stringify(role)};$('syncCut55').checked=${sync};$('cutTarget55').value=${JSON.stringify(frontMother?'':target)};if(${frontMother}){$('motherLabel54').value=${JSON.stringify(target)};$('motherLabel54').dispatchEvent(new Event('input'));}window.cutOld55=picked.s.root_id;const result=commitCut55();return {result,oldRoot:window.cutOld55,id:selected52()?.root_id,label:selected52()?label54(selected52().root_id):null,status:$('visibleStatus53').textContent}})()`);
}
try{
 session=await openPage('annotate_test.html');
 check('confidence editor initializes four native RGB periods',await session.ev("files.length===4&&!!$('confidencePanel60')")&&await layersExact());
 check('only two visible-axis labels are formally assigned',await session.ev("[...rawHierarchy54().values()].filter(h=>assigned60(h.root_id)).map(h=>h.label).sort().join(',')==='0030,0059'"));
 await session.ev("$('fit52').click();drawLabels57()");check('unassigned paths have no on-image U or invented numeric labels',await session.ev("[...labelSvg57.querySelectorAll('g')].every(g=>assigned60(g.dataset.root57)&&['0030','0059'].includes(g.dataset.label57))"));
 check('isolated short foreground candidates remain selectable rather than silently discarded',await session.ev("(()=>{const s=frame52().segments.find(s=>s.length_px<10&&s.points.length>0&&nearPaths60(...s.points[0]).some(q=>q.s.root_id===s.root_id));if(!s)return false;choosePath60(s);return selectedRoot60()===s.root_id&&!assigned60(s.root_id)})()"));
 check('unassigned native root pixels remain available without automatic numbers',await session.ev("structure52.periods.every(f=>cache52(f).owner.some(v=>v>0))&&[...rawHierarchy54().values()].some(h=>h.annotation_status_v60==='unassigned')")&&await layersExact());
 await session.ev("select(3)");await session.ev("$('fit52').click()");await clickNative(850.5,1075.5);
 check('real click selects whole horizontal root across previously split IDs',await session.ev("selectedRoot60()==='R0148'&&frame52().segments.filter(s=>s.root_id==='R0148').length>=6"));
 check('the same horizontal root is numbered 0059 in all four periods',await session.ev("structure52.periods.every(f=>f.segments.some(s=>s.root_id==='R0148'))&&label54('R0148')==='0059'"));
 check('click renders four actual-dated whole-path previews',await session.ev("$('wholePreviews60').querySelectorAll('canvas').length===4&&[...$('wholePreviews60').querySelectorAll('canvas')].every(c=>c.getContext('2d').getImageData(0,0,360,180).data.some(v=>v>0))"));
 check('legacy auto-numbered preview and root-number controls are hidden',await session.ev("getComputedStyle($('comparison52')).display==='none'&&getComputedStyle($('editingToolbar53')).display==='none'"));
 await restoreFile(out+'/source_v59.json');check('legacy automatic numbers import as unassigned rather than confirmed roots',await session.ev("[...rawHierarchy54().values()].every(h=>!assigned60(h.root_id)&&h.label.startsWith('U-'))")&&await layersExact());
 await clean();
 const first=await session.ev("(()=>{const h=[...rawHierarchy54().values()].find(h=>!assigned60(h.root_id)&&h.observed_periods.includes('T1')&&h.observed_periods.length>=3&&structure52.periods[0].segments.some(s=>s.root_id===h.root_id&&s.length_px>70));const s=frame52().segments.find(s=>s.root_id===h.root_id);choose52(s);return {id:s.root_id,p:s.points[Math.floor(s.points.length/2)]}})()");
 await session.ev("$('fit52').click()");await clickNative(...first.p);
 check('real click identifies an unnumbered whole-axis candidate',await session.ev("!assigned60(selectedRoot60())&&$('selection60').textContent.includes('未定义')"));
 await session.ev("window.firstCandidate60=selectedRoot60();$('defineKind60').value='existing'");await typeField('defineLabel60','0001');await clickButton('defineApply60');
 check('user can define an existing root as 0001',await session.ev("label54(window.firstCandidate60)==='0001'&&assigned60(window.firstCandidate60)&&rawHierarchy54().get(window.firstCandidate60).annotation_status_v60==='user_defined'"));
 check('definition automatically enrolls every matched period under the same ID',await session.ev("(()=>{const p=projectObject(),id=window.firstCandidate60,h=p.root_structure_annotation.root_track_hierarchy.find(h=>h.root_id===id);validateStructure52(p.root_structure_annotation,p);return p.root_structure_annotation.periods.every(f=>f.segments.filter(s=>s.root_id===id).every(s=>s.display_label===h.label))})()")&&await layersExact());
 await clickButton('undoPath60');check('undo returns the whole candidate to unassigned',await session.ev("!assigned60(window.firstCandidate60)")&&await layersExact());
 await session.ev("choose52(frame52().segments.find(s=>s.root_id===window.firstCandidate60));$('defineKind60').value='new'");await clickButton('defineApply60');
 check('a T1-present candidate cannot silently become a new root',await session.ev("!assigned60(window.firstCandidate60)&&$('reviewStatus60').textContent.includes('T1')"));
 await session.ev("$('defineKind60').value='existing'");await typeField('defineLabel60','1118');await clickButton('defineApply60');
 check('unexplained bare post-0312 numbering is rejected',await session.ev("!assigned60(window.firstCandidate60)&&$('reviewStatus60').textContent.includes('0312')"));
 await session.ev("$('defineLabel60').value='0001'");await clickButton('defineApply60');
 await session.ev("(async()=>{await select(1);const h=[...rawHierarchy54().values()].find(h=>!assigned60(h.root_id)&&!h.observed_periods.includes('T1')&&h.observed_periods.includes('T2')&&frame52().segments.some(s=>s.root_id===h.root_id&&s.length_px>40));choose52(frame52().segments.find(s=>s.root_id===h.root_id));window.newCandidate60=h.root_id;$('defineKind60').value='new';$('defineLabel60').value=''})()");await clickButton('defineApply60');
 check('user-defined newly appearing independent root receives N-prefix',await session.ev("assigned60(window.newCandidate60)&&/^N-\\d{4}$/.test(label54(window.newCandidate60))"));
 await session.ev("window.newFormalLabel60=label54(window.newCandidate60)");await clickButton('clearDefinition60');
 check('formal definition can be withdrawn without erasing native roots',await session.ev("!assigned60(window.newCandidate60)")&&await layersExact());
 await session.ev("$('defineKind60').value='newside';$('defineMother60').value='0001';$('defineLabel60').value=''");await clickButton('defineApply60');
 check('user-defined new lateral enters the mother-Nx numbering system',await session.ev("label54(window.newCandidate60).startsWith('0001-N')&&rawHierarchy54().get(window.newCandidate60).parent_root_id===window.firstCandidate60"));
 check('mother and lateral retain separate identities with exact shared family color',await session.ev("window.newCandidate60!==window.firstCandidate60&&JSON.stringify(rgbColor52(window.newCandidate60))===JSON.stringify(rgbColor52(window.firstCandidate60))"));
 await clean();
 const merge=await session.ev("(()=>{const ss=frame52().segments.filter(s=>!assigned60(s.root_id)&&s.length_px>50);const a=ss[0],b=ss.find(s=>s.root_id!==a.root_id);choose52(a);return {a:a.root_id,b:b.root_id,sid:b.id}})()");await clickButton('joinPath60');await session.ev(`choosePath60(frame52().segments.find(s=>s.id===${JSON.stringify(merge.sid)}))`);
 check('join adds the other complete candidate across matched dates',await session.ev(`!structure52.periods.some(f=>f.segments.some(s=>s.root_id===${JSON.stringify(merge.b)}))&&selectedRoot60()===${JSON.stringify(merge.a)}`)&&await layersExact());
 await clickButton('undoPath60');check('whole-path joining can be undone',await session.ev(`structure52.periods.some(f=>f.segments.some(s=>s.root_id===${JSON.stringify(merge.b)}))`)&&await layersExact());
 await session.ev(`choose52(frame52().segments.find(s=>s.root_id===${JSON.stringify(merge.a)}))`);await clickButton('dropPath60');await session.ev(`choosePath60(frame52().segments.find(s=>s.root_id===${JSON.stringify(merge.a)}))`);
 check('wrong-path removal produces an unassigned selectable whole candidate',await session.ev(`selectedRoot60()!==${JSON.stringify(merge.a)}&&!assigned60(selectedRoot60())`)&&await layersExact());
 await clean();await session.ev("choose52(frame52().segments.find(s=>s.root_id==='R0151'));$('defineKind60').value='existing';$('defineLabel60').value='0031'");await clickButton('defineApply60');
 check('reviewed formal root can be renumbered consistently',await session.ev("label54('R0151')==='0031'")&&await layersExact());
 await session.ev("window.download60=[];download=async(blob,name)=>window.download60.push({name,text:await blob.text()});$('saveReview60').click()");let save;for(let i=0;i<150;i++){save=await session.ev('window.download60[0]||null');if(save)break;await new Promise(r=>setTimeout(r,100));}check('JSON progress exports complete editable candidate data',save&&JSON.parse(save.text).root_structure_annotation.confidence_review_v60);fs.writeFileSync(out+'/roundtrip_test.json',save.text);await restoreFile(out+'/roundtrip_test.json');
 check('JSON restore preserves formal definitions and unassigned paths',await session.ev("label54('R0151')==='0031'&&assigned60('R0151')&&[...rawHierarchy54().values()].some(h=>!assigned60(h.root_id)&&h.label.startsWith('U-'))")&&await layersExact());
 await session.ev("window.download60=[];$('saveReviewHtml60').click()");let html;for(let i=0;i<150;i++){html=await session.ev('window.download60[0]||null');if(html)break;await new Promise(r=>setTimeout(r,100));}check('editable standalone HTML can be saved',html?.name==='annotate_root_structure.html');fs.writeFileSync(out+'/saved_editor_test.html',html.text);
 session.ws.close();await fetch('http://127.0.0.1:8893/json/close/'+session.target.id);session=await openPage('saved_editor_test.html');
 check('saved HTML reopens one confidence panel with saved root definitions',await session.ev("document.querySelectorAll('#confidencePanel60').length===1&&label54('R0151')==='0031'&&assigned60('R0151')")&&await layersExact());
 await session.ev("$('fit52').click();drawLabels57()");check('reopened editor never displays technical unknown handles as root labels',await session.ev("[...labelSvg57.querySelectorAll('g')].every(g=>assigned60(g.dataset.root57)&&!g.dataset.label57.startsWith('U-'))"));
 const screenshot=await session.call('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});fs.writeFileSync(out+'/editor_verified.png',Buffer.from(screenshot.data,'base64'));
 check('no browser runtime exceptions',exceptions.length===0,exceptions);fs.writeFileSync(out+'/browser_verification.json',JSON.stringify({passed:true,checks,exceptions,actual_browser:'Chromium real mouse keyboard file inputs and save/reopen',tested_editor_SHA256:crypto.createHash('sha256').update(fs.readFileSync(out+'/annotate.html')).digest('hex'),candidate_numbers_not_biological_ids:true},null,2)+'\n');console.log(JSON.stringify({passed:true,checks:checks.length}));
}catch(error){fs.writeFileSync(out+'/browser_verification_failed.json',JSON.stringify({checks,exceptions,error:error.stack},null,2)+'\n');console.error(error);process.exitCode=1;}
finally{if(session){session.ws.close();await fetch('http://127.0.0.1:8893/json/close/'+session.target.id).catch(()=>{});}}
