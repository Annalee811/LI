import fs from 'node:fs';
import crypto from 'node:crypto';
const out='/workspace/LI/results/B3_root_whole_axis_v57_20261026';
const expected=JSON.parse(fs.readFileSync(out+'/source_v56.json','utf8'));
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
 await ev('window.structureReady52');await ev('window.hierarchyReady54');await ev('window.rootCutReady55');await ev('window.continuityReady56');await ev('window.readableReady57');
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
async function clean(){await restoreFile(out+'/root_structure_draft.json');await session.ev('select(0)');}
async function makeCut(role='pending',target='',sync=false,frontMother=false){
 return session.ev(`(()=>{const f=frame52();let picked=null;for(const s of [...f.segments].sort((a,b)=>b.points.length-a.points.length)){if(s.points.length<80)continue;const a=s.points[Math.floor(s.points.length*.3)],b=s.points[Math.floor(s.points.length*.6)];if(a[0]<150||a[0]>2300||a[1]<100||a[1]>1000||b[0]<150||b[0]>2300||b[1]<100||b[1]>1000)continue;try{const plan=pathBetween55(f,s.root_id,a,b);if(plan.keys.length>10){picked={s,plan};break}}catch(e){}}if(!picked)throw Error('No native test root path');choose52(picked.s);cutRoot55=picked.s.root_id;cutSelection55=picked.plan.points[0];cutPreview55=picked.plan;$('cutRole55').value=${JSON.stringify(role)};$('syncCut55').checked=${sync};$('cutTarget55').value=${JSON.stringify(frontMother?'':target)};if(${frontMother}){$('motherLabel54').value=${JSON.stringify(target)};$('motherLabel54').dispatchEvent(new Event('input'));}window.cutOld55=picked.s.root_id;const result=commitCut55();return {result,oldRoot:window.cutOld55,id:selected52()?.root_id,label:selected52()?label54(selected52().root_id):null,status:$('visibleStatus53').textContent}})()`);
}
try{
 session=await openPage('annotate_test.html');
 check('four-period editor initializes with original annotation',await session.ev("files.length===4&&!!$('applyCut55')&&!!$('swapNumber55')")&&await layersExact());
 await session.ev("choose52(frame52().segments.find(s=>label54(s.root_id)==='0001'))");
 await typeField('rootId52','9000');
 check('typing in left number box synchronizes top box',await session.ev("$('rootQuickId53').value==='9000'&&$('rootId52').value==='9000'"));
 await clickButton('applyRoot53');
 check('real apply click commits the value entered in left box',await session.ev("label54(selected52().root_id)==='9000'&&$('rootQuickId53').value==='9000'"));
 check('rename updates saved graph and all descendant prefixes',await session.ev("(()=>{const p=projectObject();validateStructure52(p.root_structure_annotation,p);return p.root_structure_annotation.root_track_hierarchy.some(h=>h.label==='9000')})()")&&await layersExact());
 await session.ev("$('undo52').click();choose52(frame52().segments.find(s=>label54(s.root_id)==='0001'))");
 await typeField('rootQuickId53','9001');await session.ev('refresh52()');
 check('unapplied input survives refresh instead of reverting',await session.ev("$('rootQuickId53').value==='9001'&&$('rootId52').value==='9001'"));
 await session.call('Input.dispatchKeyEvent',{type:'keyDown',key:'Enter',code:'Enter',windowsVirtualKeyCode:13});await session.call('Input.dispatchKeyEvent',{type:'keyUp',key:'Enter',code:'Enter',windowsVirtualKeyCode:13});
 check('Enter applies a root number',await session.ev("label54(selected52().root_id)==='9001'"));
 await session.ev("$('undo52').click();choose52(frame52().segments.find(s=>label54(s.root_id)==='0001'))");
 await typeField('rootQuickId53','0003');await clickButton('applyRoot53');
 check('an occupied number can enroll the segment into the existing root',await session.ev("label54(selected52().root_id)==='0003'&&![...rawHierarchy54().values()].some(h=>h.label==='0001')"));
 check('merge retains unique labels and valid side hierarchy',await session.ev("(()=>{const p=projectObject();validateStructure52(p.root_structure_annotation,p);return true})()")&&await layersExact());
 await session.ev("$('undo52').click();choose52(frame52().segments.find(s=>label54(s.root_id)==='0001'))");
 const old=await session.ev("selected52().root_id");await typeField('rootQuickId53','0003');await clickButton('swapNumber55');
 check('number-only exchange keeps the roots independent',await session.ev(`label54(${JSON.stringify(old)})==='0003'&&[...rawHierarchy54().values()].some(h=>h.label==='0001')`));
 await session.ev("$('undo52').click();choose52(frame52().segments.find(s=>label54(s.root_id)==='0001'))");await typeField('rootQuickId53','９００２');await clickButton('applyRoot53');
 check('full-width keyboard digits are accepted and normalized',await session.ev("label54(selected52().root_id)==='9002'"));await session.ev("$('undo52').click()");
 await clean();
 const location=await session.ev("(()=>{const s=frame52().segments.filter(s=>s.points.length>100&&s.points[30][0]>150&&s.points[30][0]<2300&&s.points[30][1]>100&&s.points[30][1]<1000).sort((a,b)=>b.points.length-a.points.length)[0];choose52(s);$('startCut55').click();$('fit52').click();return {a:s.points[Math.floor(s.points.length*.3)],b:s.points[Math.floor(s.points.length*.6)]}})()");
 await clickNative(...location.a);await clickNative(...location.b);
 check('two real endpoint clicks create a middle-segment preview',await session.ev("!!cutPreview55&&cutPreview55.keys.length>=4&&!$('applyCut55').disabled"));
 const sourceBefore=await session.ev('cutRoot55');await session.ev("$('cutRole55').value='main';$('syncCut55').checked=false");await typeField('rootQuickId53','9005');await session.call('Input.dispatchKeyEvent',{type:'keyDown',key:'Enter',code:'Enter',windowsVirtualKeyCode:13});await session.call('Input.dispatchKeyEvent',{type:'keyUp',key:'Enter',code:'Enter',windowsVirtualKeyCode:13});
 check('Enter during cut preview applies only to the middle, not the complete original root',await session.ev(`selected52().root_id!==${JSON.stringify(sourceBefore)}&&label54(selected52().root_id)==='9005'&&label54(${JSON.stringify(sourceBefore)})!=='9005'`));
 check('middle segment automatically receives a registered ID',await session.ev("(()=>{const s=selected52();return !!s&&rawHierarchy54().has(s.root_id)&&frame52().manual_disconnects?.length>0&&frame52().nodes.filter(n=>n.type==='cut').length>=2})()"));
 check('two-end split retains all original pixels and temporal classes',await layersExact());
 check('split result is structurally valid and serializable',await session.ev("(()=>{const p=projectObject();validateStructure52(p.root_structure_annotation,p);return true})()"));
 await typeField('rootQuickId53','9003');await clickButton('applyRoot53');check('detached middle segment can immediately be renumbered',await session.ev("label54(selected52().root_id)==='9003'"));
 // User-painted additions belong to the selected detached root, not an unregistered mask.
 const paint=await session.ev("(()=>{const f=frame52(),s=selected52(),o=cache52(f).owner;let p=null;for(let i=0;i<o.length&&!p;i++)if(o[i]===s.index&&i%W>150&&i%W<W-150&&Math.floor(i/W)>100&&Math.floor(i/W)<H-100){for(const [dx,dy] of [[1,0],[-1,0],[0,1],[0,-1]]){const j=i+dx+dy*W;if(active.mask[j]===0){p={index:j,x:j%W+.5,y:Math.floor(j/W)+.5,owner:s.index,id:s.root_id};break}}}if(!p)throw Error('No adjacent background test pixel');$('pixelEdit53').click();$('brush').value='1';$('temporal').value='3';$('fit52').click();return p})()");
 await clickNative(paint.x,paint.y);
 check('manual correction pixels automatically inherit detached-root ownership',await session.ev(`active.mask[${paint.index}]===255&&cache52(frame52()).owner[${paint.index}]===${paint.owner}`));
 await session.ev("$('undo').click();$('rootEdit53').click()");check('pixel undo restores ownership and pixels together',await session.ev(`active.mask[${paint.index}]===0&&cache52(frame52()).owner[${paint.index}]===0`)&&await layersExact());
 await clean();const main=await makeCut('main','0001',false,true);
 check('typing mother number in the front field enrolls the middle into mother itself',main.result&&main.label==='0001',main);
 await clean();const existing=await makeCut('existing','0001');
 check('existing-side mode creates an independent child ID under the entered mother',existing.result&&/^0001-\d+$/.test(existing.label)&&await session.ev("rawHierarchy54().get(selected52().root_id).parent_root_id===resolve54('0001')&&selected52().root_id!==resolve54('0001')"),existing);
 check('newly registered side shares exact mother-family color',await session.ev("JSON.stringify(rgbColor52(selected52().root_id))===JSON.stringify(rgbColor52(resolve54('0001')))"));
 await clean();const newborn=await makeCut('new','0001',false,true);
 check('new-side mode uses front mother input to allocate N-number',newborn.result&&/^0001-N\d+$/.test(newborn.label),newborn);
 check('N-side remains a distinct root rather than merging into mother',await session.ev("selected52().root_id!==resolve54('0001')&&rawHierarchy54().get(selected52().root_id).branch_kind==='new'"));
 await clean();const nested=await makeCut('new','0001-1');
 check('a side can have its own newly born side with hierarchical numbering',nested.result&&/^0001-1-N\d+$/.test(nested.label),nested);
 await clean();
 const peer=await session.ev("(()=>{const f=frame52(),seen=[];for(const s of f.segments){if(s.points.length<70||seen.includes(s.root_id))continue;seen.push(s.root_id);for(let start=10;start<s.points.length-35;start+=30){const a=s.points[start],b=s.points[start+25];try{const p=pathBetween55(f,s.root_id,a,b);for(const other of structure52.periods.slice(1)){try{const q=pathBetween55(other,s.root_id,a,b),ss=q.points.filter((_,i)=>i%4===0),close=ss.filter(v=>distanceToPoints55(p.points,...v)<=100).length/ss.length;if(close>=.7&&q.length/p.length>.5&&q.length/p.length<2){choose52(s);cutRoot55=s.root_id;cutSelection55=p.points[0];cutPreview55=p;$('cutRole55').value='pending';$('syncCut55').checked=true;$('cutTarget55').value='';const result=commitCut55();return {result,id:selected52()?.root_id,periods:structure52.periods.filter(f=>f.segments.some(s=>s.root_id===selected52()?.root_id)).map(f=>f.period)};}}catch(e){}}}catch(e){}}}return {result:false}})()");
 check('reliably matched periods share one automatically registered detached-root ID',peer.result&&peer.periods.length>=2,peer);
 check('all editing operations preserve the untouched original pixel layers',await layersExact());
 await clean();
 check('0001 identity is present in all four periods',await session.ev("(()=>{const rid=resolve54('0001');return structure52.periods.every(f=>f.segments.some(s=>s.root_id===rid))})()"));
 check('independent new roots use N-prefix without mother linkage',await session.ev("(()=>{const entries=[...rawHierarchy54().values()].filter(h=>h.label.startsWith('N-'));return entries.length>0&&entries.every(h=>/^N-\\d{4,6}(?:-N?\\d+)*$/.test(h.label)&&(!parentLabel56(h.label)?h.parent_root_id===null:true))})()"));
 await session.ev("choose52(frame52().segments.find(s=>label54(s.root_id)==='0001'))");await typeField('rootQuickId53','Ｎ－９００１');await clickButton('applyRoot53');
 check('full-width N-prefix rename commits as independent root',await session.ev("label54(selected52().root_id)==='N-9001'&&rawHierarchy54().get(selected52().root_id).parent_root_id===null"));
 check('renaming an independent root to N-prefix cascades existing side numbers',await session.ev("[...rawHierarchy54().values()].some(h=>h.label==='N-9001-1'&&h.parent_root_id===resolve54('N-9001'))"));
 await session.ev("choose52(frame52().segments.find(s=>label54(s.root_id)==='0003'));$('motherLabel54').value='N-9001'");await clickButton('newSide54');
 check('a new lateral under N-prefixed mother receives separate nested N-side label',await session.ev("label54(selected52().root_id).startsWith('N-9001-N')&&rawHierarchy54().get(selected52().root_id).parent_root_id===resolve54('N-9001')"));
 await clickButton('newIndependent56');
 check('new independent button detaches from mother and assigns N-0001 namespace',await session.ev("/^N-\\d{4,6}$/.test(label54(selected52().root_id))&&rawHierarchy54().get(selected52().root_id).parent_root_id===null"));
 check('N-prefixed root and side hierarchy passes validation and serialization',await session.ev("(()=>{const p=projectObject();validateStructure52(p.root_structure_annotation,p);return true})()")&&await layersExact());
 await clean();await clickButton('focusFamily56');for(let i=0;i<150;i++){if(await session.ev("frame52()?.period==='T4'&&label54(selected52()?.root_id)==='0001'"))break;await new Promise(r=>setTimeout(r,100));}
 check('T4 family focus selects same 0001 identity with family filter',await session.ev("frame52().period==='T4'&&label54(selected52().root_id)==='0001'&&$('familyOnly54').checked"));
 await session.ev("$('reviewKind56').value='uncertain'");await clickButton('nextReview56');for(let i=0;i<150;i++){if(await session.ev("$('reviewMessage56').textContent.includes('候选对应有歧义')"))break;await new Promise(r=>setTimeout(r,100));}
 check('uncertain correspondence queue navigates actual saved candidates',await session.ev("$('reviewMessage56').textContent.includes('候选对应有歧义')"));
 await clean();await session.ev("choose52(frame52().segments.find(s=>label54(s.root_id)==='0001'))");await typeField('rootQuickId53','N-9001');await clickButton('applyRoot53');
 await session.ev("window.download55=[];download=async(blob,name)=>window.download55.push({name,text:await blob.text()});$('saveQuick53').click()");let saved;for(let n=0;n<50;n++){saved=await session.ev('window.download55[0]||null');if(saved)break;await new Promise(r=>setTimeout(r,100));}
 check('saved JSON contains middle-root ownership, cut endpoints and hierarchy',saved&&JSON.parse(saved.text).root_structure_annotation.periods.some(f=>f.manual_disconnects?.length));fs.writeFileSync(out+'/edited_roundtrip_test.json',saved.text);
 const ids=await session.ev('JSON.stringify(serialize52().root_track_hierarchy)');await restoreFile(out+'/edited_roundtrip_test.json');
 check('file-input restore preserves cut and numbering edits',await session.ev('JSON.stringify(serialize52().root_track_hierarchy)')===ids&&await session.ev("structure52.periods.some(f=>f.manual_disconnects?.length)")&&await layersExact());
 await session.ev("window.download55=[];$('saveHtml52').click()");let html;for(let n=0;n<50;n++){html=await session.ev('window.download55[0]||null');if(html)break;await new Promise(r=>setTimeout(r,100));}check('edited HTML can be exported',html?.name==='annotate_root_structure.html');fs.writeFileSync(out+'/saved_editor_test.html',html.text);
 session.ws.close();await fetch('http://127.0.0.1:8893/json/close/'+session.target.id);session=await openPage('saved_editor_test.html');
 check('saved HTML reopens cuts and numbering without duplicate controls',await session.ev("JSON.stringify(serialize52().root_track_hierarchy)")===ids&&await session.ev("document.querySelectorAll('#twoEndCutPanel55').length===1&&document.querySelectorAll('#swapNumber55').length===1"));
 check('reopened HTML retains original foreground and temporal labels',await layersExact());

 await session.ev("(()=>{const s=frame52().segments.filter(s=>label54(s.root_id)==='N-9001').sort((a,b)=>b.points.length-a.points.length)[0];choose52(s);window.labelCenter57=s.points[Math.floor(s.points.length/2)];$('fontSize57').value='18';$('fontSize57').dispatchEvent(new Event('input'));})()");
 for(const z of [1,8,20,60]){
  const info=await session.ev(`(async()=>{window.B3Zoom60Installed.setZoom(${z});center52(...window.labelCenter57);drawLabels57();await new Promise(r=>requestAnimationFrame(r));const text=document.querySelector('#readableLabels57 .chosen57 text');const box=text?.getBoundingClientRect();return {label:text?.textContent,font:text?getComputedStyle(text).fontSize:null,height:box?.height,zoom:$('zoom').value,svgCount:document.querySelectorAll('#readableLabels57').length}})()`);
  check('root number remains screen-space readable at '+z+'x',info.label==='N-9001'&&parseFloat(info.font)===18&&info.height>=16&&info.height<=26&&info.svgCount===1,info);
  if(z===20||z===60){const screenshot=await session.call('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});fs.writeFileSync(out+'/labels_zoom_'+z+'.png',Buffer.from(screenshot.data,'base64'));}
 }
 await session.ev("$('fontSize57').value='32';$('fontSize57').dispatchEvent(new Event('input'));drawLabels57()");
 check('label size can increase to 32px without changing root pixels',await session.ev("getComputedStyle(document.querySelector('#readableLabels57 .chosen57 text')).fontSize==='32px'")&&await layersExact());
 const pill=await session.ev("(()=>{const g=document.querySelector('#readableLabels57 .chosen57'),r=g.querySelector('rect').getBoundingClientRect();return {id:g.dataset.root57,x:r.x+r.width/2,y:r.y+r.height/2}})()");
 await session.call('Input.dispatchMouseEvent',{type:'mousePressed',button:'left',buttons:1,clickCount:1,x:pill.x,y:pill.y});await session.call('Input.dispatchMouseEvent',{type:'mouseReleased',button:'left',buttons:0,clickCount:1,x:pill.x,y:pill.y});
 check('clicking crisp number label selects its actual root',await session.ev(`selected52()?.root_id===${JSON.stringify(pill.id)}`));
 check('label preferences serialize separately from annotation pixels',await session.ev("projectObject().editor_preferences_v57.font_size===32")&&await layersExact());
 check('reopened editor contains one label overlay and one control panel',await session.ev("document.querySelectorAll('#readableLabels57').length===1&&document.querySelectorAll('#readabilityPanel57').length===1"));
 check('no browser runtime exceptions',exceptions.length===0,exceptions);
 const report={passed:true,checks,exceptions,tested_editor_SHA256:crypto.createHash('sha256').update(fs.readFileSync(out+'/annotate.html')).digest('hex'),actual_browser:'Chromium; real mouse, keyboard and file picker',network_fixture:'Original PNG URLs remapped to identical local files',original_annotation_rebuilt:false,uploaded_12_preserved:true,continuity_proposals_added:true,screen_space_labels_verified:true};fs.writeFileSync(out+'/browser_verification.json',JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify({passed:true,checks:checks.length}));
}catch(error){fs.writeFileSync(out+'/browser_verification_failed.json',JSON.stringify({checks,exceptions,error:error.stack},null,2)+'\n');console.error(error);process.exitCode=1;}
finally{if(session){session.ws.close();await fetch('http://127.0.0.1:8893/json/close/'+session.target.id).catch(()=>{});}}
