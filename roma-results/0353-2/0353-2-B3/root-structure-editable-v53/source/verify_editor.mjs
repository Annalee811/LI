import fs from 'node:fs';
import crypto from 'node:crypto';
const out='/workspace/LI/results/B3_editor_interaction_v53_20261025';
const expected=JSON.parse(fs.readFileSync(out+'/source_progress_10.json','utf8'));
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
 await ev('window.structureReady52');
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
try{
 session=await openPage('annotate_test.html');
 const start=await session.ev("({periods:structure52.periods.length,names:files.map(f=>f.name),width:W,height:H,controls:document.querySelectorAll('#structureControls52').length,viewport:!!$('b3ImageViewport')})");
 check('four upper-period original images and single editor initialize',start.periods===4&&start.names.length===4&&start.height===1108&&start.width===2541&&start.controls===1&&start.viewport,start);
 check('original four pixel-label layers are byte-equivalent',await layersExact());
 const geometry=await session.ev("(()=>{const data=[];for(const f of structure52.periods){const s=states.get(f.name+':'+f.image_sha256),owner=cache52(f).owner;let changed=0;for(let i=0;i<owner.length;i++)if((owner[i]>0)!==(s.mask[i]===255))changed++;data.push({period:f.period,changed,segments:f.segments.length,pending:f.segments.every(s=>s.review_status==='pending')&&f.nodes.every(n=>n.review_status==='pending')})}return data})()");
 check('root ownership covers exactly the unchanged masks; all proposals unconfirmed',geometry.every(v=>v.changed===0&&v.pending),geometry);
 check('every crossing owner with visible arms belongs to an incident trace',await session.ev("structure52.periods.every(f=>f.nodes.every(n=>!n.arms.length||n.arms.some(a=>a.segment_id===n.owner_segment_id)))"));
 for(let i=0;i<4;i++){await session.ev(`select(${i})`);check('period navigation '+(i+1),await session.ev(`active.name==='T${i+1}_RGB.png'&&!!frame52()`));}
 await session.ev('select(1)');
 await session.ev("(()=>{const f=frame52(),s=f.segments.reduce((a,b)=>a.length_px>b.length_px?a:b);$('segments52').value=s.id;$('segments52').dispatchEvent(new Event('change'));window.selectedBefore52=s.id})()");
 check('root list selects a full stored trajectory',await session.ev('selectedSegment52===window.selectedBefore52&&!!selected52()'));
 const linked=await session.ev("(()=>{const previous=structure52.periods[0],s=frame52().segments.find(s=>previous.segments.some(p=>p.root_id===s.root_id));choose52(s);$('confirmLink52').click();return structure52.cross_period_links.some(l=>l.current_period==='T2'&&l.root_id===s.root_id&&l.review_status==='confirmed')})()");
 check('cross-period same-root association can be explicitly confirmed',linked);
 await session.ev("$('undo52').click();choose52(frame52().segments.find(s=>s.id===window.selectedBefore52))");
 const originalId=await session.ev('selected52().root_id');
 await session.ev("$('rootId52').value='R9999';$('assign52').click()");
 check('root ID can be edited without changing pixel labels',await session.ev("selected52().root_id==='R9999'")&&await layersExact());
 await session.ev("$('undo52').click()");
 check('structure undo restores root ID',await session.ev(`frame52().segments.find(s=>s.id===window.selectedBefore52).root_id===${JSON.stringify(originalId)}`));
 const node=await session.ev("(()=>{const n=frame52().nodes.find(n=>n.arms.length>=4&&new Set(n.arms.map(a=>a.segment_id)).size>=2);choose52(cache52(frame52()).byId.get(n.owner_segment_id),n);return {id:n.id,count:frame52().nodes.length}})()");
 const connect=await session.ev("(()=>{const n=node52(),a=n.arms[0],b=n.arms.find(b=>b.segment_id!==a.segment_id);document.querySelector('[data-arm52='+a.id+']').click();document.querySelector('[data-arm52='+b.id+']').click();$('connect52').click();const c=node52();return {pair:c.connections.some(p=>p.includes(a.id)&&p.includes(b.id)),oneToOne:new Set(c.connections.flat()).size===c.connections.flat().length,sameId:root52(frame52(),a.segment_id)===root52(frame52(),b.segment_id)}})()");
 check('crossing arms connect with unique partners and common root ID',connect.pair&&connect.oneToOne&&connect.sameId,connect);
 await session.ev("$('undo52').click()");
 await session.ev(`(()=>{const n=frame52().nodes.find(n=>n.id===${JSON.stringify(node.id)});choose52(cache52(frame52()).byId.get(n.owner_segment_id),n);const option=[...$('nodeOwner52').options].find(o=>o.value!==n.owner_segment_id);window.newOwner52=option.value;$('nodeOwner52').value=option.value;$('saveNode52').click()})()`);
 check('crossing owner is editable without recoloring temporal labels',await session.ev('node52().owner_segment_id===window.newOwner52')&&await layersExact());
 await session.ev("$('nodeType52').value='branch';$('note52').value='人工复核测试';$('confirmNode52').click()");
 check('branch type, review status and note are saved',await session.ev("node52().type==='branch'&&node52().review_status==='confirmed'&&node52().note==='人工复核测试'"));
 await session.ev("$('fit52').click();$('tool52').value='tip';$('tool52').dispatchEvent(new Event('change'))");
 const countBefore=await session.ev('frame52().nodes.length');await clickNative(1000,500);
 check('native canvas pointer creates editable tip',await session.ev(`frame52().nodes.length===${countBefore+1}&&node52().type==='tip'&&Math.abs(node52().x-1000)<1&&Math.abs(node52().y-500)<1`));
 await session.ev("$('tool52').value='move';$('tool52').dispatchEvent(new Event('change'))");
 await clickNative(1000,500,[1025,520]);
 check('real pointer drag moves point in original coordinates',await session.ev('Math.abs(node52().x-1025)<1&&Math.abs(node52().y-520)<1'));
 await session.ev("$('deleteNode52').click()");check('point deletion works',await session.ev(`frame52().nodes.length===${countBefore}`));
 await session.ev("$('undo52').click()");check('point deletion can be undone',await session.ev(`frame52().nodes.length===${countBefore+1}`));
 // Restore a clean draft before independently testing split and project export.
 await restoreFile(out+'/root_structure_draft.json');await session.ev('select(1)');
 const splitPoint=await session.ev("(()=>{const f=frame52(),s=f.segments.reduce((a,b)=>a.length_px>b.length_px?a:b),owner=cache52(f).owner;let at=Math.floor(s.points.length/2);while(at<s.points.length-5&&owner[s.points[at][1]*W+s.points[at][0]]!==s.index)at++;const p=s.points[at];$('fit52').click();$('tool52').value='split';$('tool52').dispatchEvent(new Event('change'));return {x:p[0],y:p[1],count:f.segments.length}})()");
 await clickNative(splitPoint.x,splitPoint.y);
 check('mistaken whole-root connection can be split with pixel geometry preserved',await session.ev(`frame52().segments.length===${splitPoint.count+1}`)&&await layersExact());
 await session.ev("$('undo52').click()");check('split is reversible',await session.ev(`frame52().segments.length===${splitPoint.count}`));
 await session.ev("(()=>{const s=frame52().segments[0];choose52(s);$('rootId52').value='R9998';$('assign52').click();window.savedIdentity52=s.id;window.download52=[];download=async(blob,name)=>{window.download52.push({name,text:await blob.text()})};$('save52').click()})()");
 let saved;
 for(let n=0;n<50;n++){saved=await session.ev('window.download52[0]||null');if(saved)break;await new Promise(r=>setTimeout(r,100));}
 check('save JSON button includes all root structure edits',saved&&JSON.parse(saved.text).root_structure_annotation.periods[1].segments[0].root_id==='R9998');
 fs.writeFileSync(out+'/edited_roundtrip_test.json',saved.text);
 await restoreFile(out+'/root_structure_draft.json');await restoreFile(out+'/edited_roundtrip_test.json');
 check('real file-input restore retains edited identities and unchanged pixel layers',await session.ev("structure52.periods[1].segments[0].root_id==='R9998'")&&await layersExact());
 const bad=JSON.parse(saved.text);bad.root_structure_annotation.periods[0].image_sha256='0'.repeat(64);fs.writeFileSync(out+'/invalid_structure_test.json',JSON.stringify(bad));
 const rejected=await restoreFile(out+'/invalid_structure_test.json');
 check('mismatched structural source is rejected atomically',rejected.status.includes('失败')&&await session.ev("structure52.periods[1].segments[0].root_id==='R9998'"));
 await session.ev("window.download52=[];$('saveHtml52').click()");
 let html;
 for(let n=0;n<50;n++){html=await session.ev('window.download52[0]||null');if(html)break;await new Promise(r=>setTimeout(r,100));}
 check('editable HTML export carries current annotation',html?.name==='annotate_root_structure.html'&&html.text.includes('R9998'));
 fs.writeFileSync(out+'/saved_editor_test.html',html.text);
 session.ws.close();await fetch('http://127.0.0.1:8893/json/close/'+session.target.id);
 session=await openPage('saved_editor_test.html');
 check('saved editable HTML reopens with edits and no duplicate controls',await session.ev("structure52.periods[1].segments[0].root_id==='R9998'&&document.querySelectorAll('#b3QuickTools').length===1&&document.querySelectorAll('#structureControls52').length===1&&document.querySelectorAll('#b3ImageViewport').length===1"));
 check('reopened HTML retains all original pixel labels',await layersExact());
 await session.ev("select(2)");await session.ev("(()=>{const n=frame52().nodes.find(n=>n.type==='crossing'&&n.x>600&&n.x<2000&&n.y>300&&n.y<900);choose52(cache52(frame52()).byId.get(n.owner_segment_id),n);window.B3Zoom60Installed.setZoom(1.2);center52(n.x,n.y);$('structureControls52').scrollTop=0})()");
 const screenshot=await session.call('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});fs.writeFileSync(out+'/editor_preview.png',Buffer.from(screenshot.data,'base64'));
 await session.ev('window.interactionReady53');
 check('editing instructions and error status remain visible',await session.ev("$('visibleStatus53').getBoundingClientRect().height>0&&$('editHelp53').getBoundingClientRect().height>0"));
 await restoreFile(out+'/root_structure_draft.json');
 await session.ev("$('fit52').click();$('pan52').click();$('tool52').value='tip';$('tool52').dispatchEvent(new Event('change'))");
 const countPan=await session.ev('frame52().nodes.length');
 check('selecting an edit tool releases viewport pan capture',await session.ev('!window.B3Zoom60Installed.pan'));
 await clickNative(1100,550);
 check('a real click edits after switching from pan',await session.ev(`frame52().nodes.length===${countPan+1}`));await session.ev("$('undo52').click()");
 await session.ev("$('rootEdit53').click();choose52(frame52().segments.find(s=>s.root_id==='R0001'))");
 await session.ev("$('rootQuickId53').focus();$('rootQuickId53').select()");
 await session.call('Input.insertText',{text:'9997'});
 const apply=await session.ev("(()=>{const r=$('applyRoot53').getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2}})()");
 await session.call('Input.dispatchMouseEvent',{type:'mousePressed',button:'left',buttons:1,clickCount:1,...apply});await session.call('Input.dispatchMouseEvent',{type:'mouseReleased',button:'left',buttons:0,clickCount:1,...apply});
 check('typing a root number and clicking apply renames the entire four-period track',await session.ev("structure52.periods.every(f=>f.segments.some(s=>s.root_id==='R9997'))&&!structure52.periods.some(f=>f.segments.some(s=>s.root_id==='R0001'))"));
 await session.ev("$('undo52').click()");
 check('global rename is reversible across all periods',await session.ev("structure52.periods.every(f=>f.segments.some(s=>s.root_id==='R0001'))"));
 await session.ev("choose52(frame52().segments.find(s=>s.root_id==='R0001'));$('mergeRoot53').click();choose52(frame52().segments.find(s=>s.root_id==='R0002'))");
 check('two mistaken root fragments can merge with four-period IDs synchronized',await session.ev("!structure52.periods.some(f=>f.segments.some(s=>s.root_id==='R0002'))&&structure52.periods.every(f=>f.segments.some(s=>s.root_id==='R0001'))&&mergeTarget53===null")&&await layersExact());
 const association=await session.ev("(()=>{const s=serialize52(),t=s.root_tracks.find(t=>t.root_id==='R0001');return {allFour:t.observed_periods.length===4,links:s.cross_period_links.filter(l=>l.root_id==='R0001').length,oldLinks:s.cross_period_links.some(l=>l.root_id==='R0002')}})()");
 check('saved cross-period relationships follow the edited whole-root identity',association.allFour&&association.links===3&&!association.oldLinks,association);
 await session.ev("$('undo52').click()");
 for(let i=0;i<4;i++){await session.ev(`select(${i})`);check('period switch retains selected whole-root identity '+(i+1),await session.ev("selected52()?.root_id==='R0001'"));}
 await session.ev("$('rootQuickId53').value='INVALID';$('applyRoot53').click()");
 check('invalid IDs produce a visible error rather than silent failure',await session.ev("$('visibleStatus53').textContent.includes('编号')&&$('visibleStatus53').getBoundingClientRect().height>0&&selected52().root_id==='R0001'"));
 await session.ev('select(1)');
 const pixel=await session.ev("(()=>{$('brush').value='1';$('pan52').click();document.querySelector('[data-code53=\"2\"]').click();$('fit52').click();const i=active.temporal.findIndex((v,i)=>v===3&&active.mask[i]===255&&i%W>800&&i%W<1600&&Math.floor(i/W)>300&&Math.floor(i/W)<700);return {index:i,x:i%W+.5,y:Math.floor(i/W)+.5}})()");
 check('temporal paint shortcut exits pan and exposes a usable brush',await session.ev("!window.B3Zoom60Installed.pan&&$('tool52').value==='pixel'&&$('palette').getBoundingClientRect().height>0"));
 await clickNative(pixel.x,pixel.y);
 check('real pointer drawing changes temporal labels',await session.ev(`active.temporal[${pixel.index}]===2`));
 await session.ev("$('undo').click();$('rootEdit53').click()");
 check('temporal edits can be undone without altering original labels',await layersExact());
 const validTracks=await session.ev("(()=>{const r=registry53();return {allFour:r.filter(t=>t.observed_periods.length===4).length,newContinues:r.filter(t=>t.first_seen_period!=='T1'&&t.observed_periods.length>1).length}})()");
 check('draft includes persistent four-period IDs and later continuation of new roots',validTracks.allFour>0&&validTracks.newContinues>0,validTracks);
 check('saved HTML recreates exactly one quick editing toolbar',await session.ev("document.querySelectorAll('#editingToolbar53').length===1"));
 check('no browser script exceptions',exceptions.length===0,exceptions);
 const report={passed:true,checks,exceptions,actual_browser:'Chromium with real DOM file input and pointer events',network_fixture:'Only original-image URLs remapped to exact local PNG bytes',training_modified:false};fs.writeFileSync(out+'/browser_verification.json',JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify({passed:true,checks:checks.length}));
}catch(error){fs.writeFileSync(out+'/browser_verification_failed.json',JSON.stringify({checks,exceptions,error:error.stack},null,2)+'\n');console.error(error);process.exitCode=1;}
finally{if(session){session.ws.close();await fetch('http://127.0.0.1:8893/json/close/'+session.target.id).catch(()=>{});}}
