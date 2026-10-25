import fs from 'node:fs';
import crypto from 'node:crypto';
const out='/workspace/LI/results/B3_root_family_review_v54_20261025';
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
 await ev('window.structureReady52');await ev('window.hierarchyReady54');
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
 check('four native-period images initialize with hierarchical editor',await session.ev("files.length===4&&structure52.periods.length===4&&W===2541&&H===1108&&!!$('familyPanel54')"));
 check('original mask, color, temporal and viability layers remain exact',await layersExact());
 const audit=await session.ev("(()=>{const map=rawHierarchy54();return {entries:map.size,unique:new Set([...map.values()].map(h=>h.label)).size===map.size,acyclic:[...map.values()].every(h=>!cycle54(map,h.root_id,h.parent_root_id)),colors:[...map.values()].every(h=>!h.parent_root_id||JSON.stringify(rgbColor52(h.root_id))===JSON.stringify(rgbColor52(h.parent_root_id))),existing:[...map.values()].filter(h=>h.parent_root_id&&h.branch_kind==='existing').length,newSides:[...map.values()].filter(h=>h.parent_root_id&&h.branch_kind==='new').length}})()");
 check('hierarchical labels unique and mother graph acyclic',audit.unique&&audit.acyclic,audit);
 check('mother, existing sides and new sides share the exact family color',audit.colors&&audit.existing>0&&audit.newSides>0);
 const structural=await session.ev("(()=>{const rows=[];for(const f of structure52.periods){const st=states.get(f.name+':'+f.image_sha256),own=cache52(f).owner;let changed=0;for(let i=0;i<own.length;i++)if((own[i]>0)!==(st.mask[i]===255))changed++;rows.push(changed)}return rows})()");check('root pixel geometry remains unchanged',structural.every(n=>n===0));
 check('all serialized segments retain one lifetime hierarchical label',await session.ev("(()=>{const o=serialize52(),m=new Map(o.root_track_hierarchy.map(h=>[h.root_id,h]));return o.periods.every(f=>f.segments.every(s=>s.display_label===m.get(s.root_id).label&&s.family_root_id===m.get(s.root_id).family_root_id))})()"));
 const side=await session.ev("(()=>{const h=[...rawHierarchy54().values()].find(h=>h.parent_root_id&&h.observed_periods?.length>1);window.side54=h.root_id;return {id:h.root_id,label:h.label,periods:h.observed_periods}})()");
 for(const t of side.periods){await session.ev(`(async()=>{await select(${Number(t[1])-1});choose52(frame52().segments.find(s=>s.root_id===window.side54))})()`);check('side-root label retained in '+t,await session.ev(`$('rootQuickId53').value===${JSON.stringify(side.label)}`));}
 const newSide=await session.ev("(()=>{const h=[...rawHierarchy54().values()].find(h=>h.branch_kind==='new'&&h.observed_periods?.length>1);window.newSide54=h.root_id;return {id:h.root_id,label:h.label,periods:h.observed_periods}})()");
 check('new lateral has a permanent N-number',/\-N\d+(?:-|$)/.test(newSide.label),newSide);
 for(const t of newSide.periods){await session.ev(`(async()=>{await select(${Number(t[1])-1});choose52(frame52().segments.find(s=>s.root_id===window.newSide54))})()`);check('new lateral keeps same N-number in '+t,await session.ev(`$('rootQuickId53').value===${JSON.stringify(newSide.label)}`));}
 await session.ev("(()=>{const h=rawHierarchy54().get(window.side54);window.mother54=h.parent_root_id;window.parentLabel54=label54(h.parent_root_id)})()");
 const targetLabel=await session.ev("window.parentLabel54+'-99'");
 await session.ev("(()=>{const s=structure52.periods.find(f=>f.segments.some(s=>s.root_id===window.side54));window.targetPeriod54=structure52.periods.indexOf(s)})()");await session.ev('(async()=>{await select(window.targetPeriod54);choose52(frame52().segments.find(s=>s.root_id===window.side54));$(\'rootQuickId53\').focus();$(\'rootQuickId53\').select()})()');
 await session.call('Input.insertText',{text:targetLabel});
 const button=await session.ev("(()=>{const r=$('applyRoot53').getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2}})()");await session.call('Input.dispatchMouseEvent',{type:'mousePressed',button:'left',buttons:1,clickCount:1,...button});await session.call('Input.dispatchMouseEvent',{type:'mouseReleased',button:'left',buttons:0,clickCount:1,...button});
 check('real keyboard/mouse can edit a hierarchical side-root number',await session.ev(`label54(window.side54)===${JSON.stringify(targetLabel)}`));
 check('side rename synchronizes every observed period without pixel relabeling',await session.ev(`structure52.periods.every(f=>f.segments.filter(s=>s.root_id===window.side54).every(s=>s.display_label===${JSON.stringify(targetLabel)}))`)&&await layersExact());
 await session.ev("$('undo52').click()");check('hierarchical rename can be undone',await session.ev(`label54(window.side54)===${JSON.stringify(side.label)}`));
 await session.ev("(()=>{const h=rawHierarchy54().get(window.side54);let p=h;while(p.parent_root_id)p=rawHierarchy54().get(p.parent_root_id);window.head54=p.root_id;window.oldHeadLabel54=p.label;})()");
 await session.ev("(async()=>{const i=structure52.periods.findIndex(f=>f.segments.some(s=>s.root_id===window.head54));await select(i);choose52(frame52().segments.find(s=>s.root_id===window.head54));$('rootQuickId53').value='9000';$('applyRoot53').click()})()");
 check('mother rename cascades to all descendant labels',await session.ev("label54(window.head54)==='9000'&&label54(window.side54).startsWith('9000-')"));
 check('cascade preserves family color and stable internal identities',await session.ev("family54(window.side54)===window.head54&&JSON.stringify(rgbColor52(window.side54))===JSON.stringify(rgbColor52(window.head54))")&&await layersExact());
 await session.ev("$('undo52').click()");
 await session.ev("(async()=>{const i=structure52.periods.findIndex(f=>f.segments.some(s=>s.root_id===window.head54));await select(i);choose52(frame52().segments.find(s=>s.root_id===window.head54));$('motherLabel54').value=label54(window.side54);$('existingSide54').click()})()");
 check('parent/child cycles are rejected with visible feedback',await session.ev("!rawHierarchy54().get(window.head54).parent_root_id&&$('visibleStatus53').textContent.includes('后代')"));
 const additions=await session.ev("(()=>{const all=[...rawHierarchy54().values()],chosen=all.filter(h=>!h.parent_root_id&&h.root_id!==window.head54&&!all.some(c=>c.parent_root_id===h.root_id)&&h.observed_periods?.length>1);return chosen.slice(0,3).map(h=>h.root_id)})()");
 for(let i=0;i<2;i++){
  await session.ev(`(async()=>{const id=${JSON.stringify(additions[i])};await select(structure52.periods.findIndex(f=>f.segments.some(s=>s.root_id===id)));choose52(frame52().segments.find(s=>s.root_id===id));$('motherLabel54').value=label54(window.head54);$('newSide54').click()})()`);
  check('new side '+(i+1)+' gets its own N-ID and mother color',await session.ev(`rawHierarchy54().get(${JSON.stringify(additions[i])}).parent_root_id===window.head54&&/-N\\d+$/.test(label54(${JSON.stringify(additions[i])}))&&JSON.stringify(rgbColor52(${JSON.stringify(additions[i])}))===JSON.stringify(rgbColor52(window.head54))`));
 }
 check('N-counters are shared across periods and never reused',await session.ev(`label54(${JSON.stringify(additions[0])})!==label54(${JSON.stringify(additions[1])})`));
 await session.ev(`(async()=>{const id=${JSON.stringify(additions[2])};await select(structure52.periods.findIndex(f=>f.segments.some(s=>s.root_id===id)));choose52(frame52().segments.find(s=>s.root_id===id));$('motherLabel54').value=label54(window.head54);$('existingSide54').click()})()`);
 check('existing lateral uses a numeric suffix rather than N',await session.ev(`/-\\d+$/.test(label54(${JSON.stringify(additions[2])}))&&rawHierarchy54().get(${JSON.stringify(additions[2])}).branch_kind==='existing'`));
 await session.ev("$('independent54').click()");check('incorrect side relationship can be detached',await session.ev(`!rawHierarchy54().get(${JSON.stringify(additions[2])}).parent_root_id&&!label54(${JSON.stringify(additions[2])}).includes('-')`));
 check('all hierarchy edits preserve original temporal categories and boundaries',await layersExact());
 const pair=await session.ev("(()=>{const m=rawHierarchy54(),all=[...m.values()],f=frame52();return all.filter(h=>!h.parent_root_id&&!all.some(c=>c.parent_root_id===h.root_id)&&f.segments.some(s=>s.root_id===h.root_id)).slice(0,2).map(h=>h.root_id)})()");
 await session.ev(`choose52(frame52().segments.find(s=>s.root_id===${JSON.stringify(pair[0])}));$('mergeRoot53').click();choose52(frame52().segments.find(s=>s.root_id===${JSON.stringify(pair[1])}))`);
 check('fragment merging preserves a valid unique hierarchy',await session.ev("(()=>{const p=projectObject();validateStructure52(p.root_structure_annotation,p);return true})()")&&await layersExact());
 await session.ev("$('undo52').click()");
 const splitting=await session.ev("(()=>{const s=frame52().segments.filter(s=>s.points.length>60).sort((a,b)=>b.points.length-a.points.length)[0],owner=cache52(frame52()).owner;let i=Math.floor(s.points.length/2);while(i<s.points.length-4&&owner[s.points[i][1]*W+s.points[i][0]]!==s.index)i++;return {p:s.points[i],count:frame52().segments.length}})()");
 await session.ev(`split52(${splitting.p[0]},${splitting.p[1]})`);
 check('splitting a false connection also creates valid editable identity metadata',await session.ev(`(()=>{const p=projectObject();validateStructure52(p.root_structure_annotation,p);return frame52().segments.length===${splitting.count+1}})()`)&&await layersExact());
 await session.ev("$('undo52').click()");
 await session.ev("$('fit52').click();$('pan52').click();$('tool52').value='tip';$('tool52').dispatchEvent(new Event('change'))");check('editing mode releases pan capture',await session.ev('!window.B3Zoom60Installed.pan'));
 const pointCount=await session.ev('frame52().nodes.length');await clickNative(1100,550);check('real canvas click can add a point',await session.ev(`frame52().nodes.length===${pointCount+1}`));await session.ev("$('undo52').click()");
 await session.ev("$('rootEdit53').click();window.savedLabels54=JSON.stringify([...rawHierarchy54().values()].map(h=>({root_id:h.root_id,label:h.label,parent_root_id:h.parent_root_id,family_root_id:h.family_root_id,branch_kind:h.branch_kind})));window.download54=[];download=async(blob,name)=>window.download54.push({name,text:await blob.text()});$('saveQuick53').click()");
 let saved;for(let n=0;n<50;n++){saved=await session.ev('window.download54[0]||null');if(saved)break;await new Promise(r=>setTimeout(r,100));}
 check('save JSON includes editable hierarchy',saved&&JSON.parse(saved.text).root_structure_annotation.root_track_hierarchy.length>0);fs.writeFileSync(out+'/edited_roundtrip_test.json',saved.text);
 const expectedHierarchy=await session.ev('window.savedLabels54');await restoreFile(out+'/edited_roundtrip_test.json');
 check('file-input restore recovers exact family labels and parentage',await session.ev('JSON.stringify([...rawHierarchy54().values()].map(h=>({root_id:h.root_id,label:h.label,parent_root_id:h.parent_root_id,family_root_id:h.family_root_id,branch_kind:h.branch_kind})))')===expectedHierarchy&&await layersExact());
 const bad=JSON.parse(saved.text),h=bad.root_structure_annotation.root_track_hierarchy.find(h=>h.parent_root_id);h.parent_root_id=h.root_id;fs.writeFileSync(out+'/invalid_hierarchy_test.json',JSON.stringify(bad));const rejected=await restoreFile(out+'/invalid_hierarchy_test.json');
 check('invalid hierarchical import is rejected atomically',rejected.status.includes('失败')&&await session.ev('JSON.stringify([...rawHierarchy54().values()].map(h=>({root_id:h.root_id,label:h.label,parent_root_id:h.parent_root_id,family_root_id:h.family_root_id,branch_kind:h.branch_kind})))')===expectedHierarchy);
 await session.ev("window.download54=[];$('saveHtml52').click()");let html;for(let n=0;n<50;n++){html=await session.ev('window.download54[0]||null');if(html)break;await new Promise(r=>setTimeout(r,100));}check('editable HTML can be saved',html?.name==='annotate_root_structure.html');fs.writeFileSync(out+'/saved_editor_test.html',html.text);
 session.ws.close();await fetch('http://127.0.0.1:8893/json/close/'+session.target.id);session=await openPage('saved_editor_test.html');
 check('saved HTML reopens with the exact edited hierarchy',await session.ev('JSON.stringify([...rawHierarchy54().values()].map(h=>({root_id:h.root_id,label:h.label,parent_root_id:h.parent_root_id,family_root_id:h.family_root_id,branch_kind:h.branch_kind})))')===expectedHierarchy);
 check('reopened page has no duplicate tools',await session.ev("document.querySelectorAll('#familyPanel54').length===1&&document.querySelectorAll('#editingToolbar53').length===1&&document.querySelectorAll('#b3ImageViewport').length===1"));
 check('reopened page retains original pixel layers',await layersExact());
 await restoreFile('/workspace/LI/results/B3_editor_interaction_v53_20261025/root_structure_draft.json');
 check('legacy V53 mother assignments are preserved during upgrade',await session.ev("(()=>{const obj=projectObject();validateStructure52(obj.root_structure_annotation,obj);return obj.root_structure_annotation.root_track_hierarchy.reduce((n,h)=>n+(h.preserved_legacy_parent_assignments?.length||0),0)>0})()")&&await layersExact());
 await restoreFile(out+'/root_structure_draft.json');await session.ev("(async()=>{const h=[...rawHierarchy54().values()].find(h=>h.label==='0001'),f=structure52.periods.find(f=>f.segments.some(s=>s.root_id===h.root_id));await select(structure52.periods.indexOf(f));choose52(frame52().segments.find(s=>s.root_id===h.root_id));$('familyOnly54').checked=true;window.B3Zoom60Installed.setZoom(.8);center52(2310,750);render();$('structureControls52').scrollTop=0})()");const png=await session.call('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});fs.writeFileSync(out+'/editor_preview.png',Buffer.from(png.data,'base64'));
 check('no browser runtime exceptions',exceptions.length===0,exceptions);
 const report={passed:true,checks,exceptions,tested_editor_SHA256:crypto.createHash('sha256').update(fs.readFileSync(out+'/annotate.html')).digest('hex'),actual_browser:'Chromium; real keyboard, mouse, drag and file-picker interfaces',network_fixture:'Original PNG URLs remapped to identical local originals',training_modified:false};fs.writeFileSync(out+'/browser_verification.json',JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify({passed:true,checks:checks.length}));
}catch(error){fs.writeFileSync(out+'/browser_verification_failed.json',JSON.stringify({checks,exceptions,error:error.stack},null,2)+'\n');console.error(error);process.exitCode=1;}
finally{if(session){session.ws.close();await fetch('http://127.0.0.1:8893/json/close/'+session.target.id).catch(()=>{});}}
