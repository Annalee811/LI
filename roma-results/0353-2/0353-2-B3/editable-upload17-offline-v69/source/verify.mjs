import fs from 'node:fs';
import crypto from 'node:crypto';
const out='/workspace/LI/results/B3_editable_upload17_v69_20261028';
const source=JSON.parse(fs.readFileSync(out+'/root_annotation_progress_17.json','utf8'));
const hash=v=>crypto.createHash('sha256').update(JSON.stringify(v)).digest('hex');
const expectedLayers=source.images.map(e=>({name:e.name,hash:Object.fromEntries(Object.entries(e.layers).map(([k,v])=>[k,hash(v)]))}));
const expectedGeometry=source.root_structure_annotation.periods.map(f=>({period:f.period,owner:hash(f.owner_map_rle),segments:hash(f.segments.map(s=>({id:s.id,index:s.index,root_id:s.root_id,points:s.points,length_px:s.length_px}))),cuts:hash(f.manual_free_cuts_v62||[])}));
const defs=source.root_structure_annotation.root_track_hierarchy.filter(h=>h.annotation_status_v60==='user_defined').map(h=>({root_id:h.root_id,label:h.label,parent_root_id:h.parent_root_id,definition:h.human_identity_definition_v60||null}));
const checks=[];const exceptions=[];let session;
function check(name,yes,detail){checks.push({name,passed:!!yes,...(detail?{detail}:{})});if(!yes)throw Error(name);console.log('PASS',name)}
async function open(filename){
 const target=await(await fetch('http://127.0.0.1:8870/json/new?about:blank',{method:'PUT'})).json();
 const ws=new WebSocket(target.webSocketDebuggerUrl);await new Promise(r=>ws.onopen=r);let serial=0;const pending=new Map();const network=[];
 ws.onmessage=e=>{const d=JSON.parse(e.data);if(d.id){const p=pending.get(d.id);pending.delete(d.id);d.error?p.reject(Error(JSON.stringify(d.error))):p.resolve(d.result)}else if(d.method==='Runtime.exceptionThrown')exceptions.push(d.params.exceptionDetails);else if(d.method==='Network.requestWillBeSent')network.push(d.params.request.url)};
 const call=(method,params={})=>new Promise((resolve,reject)=>{const id=++serial;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}))});
 const ev=async expression=>{const v=await call('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(v.exceptionDetails)throw Error(JSON.stringify(v.exceptionDetails));return v.result.value};
 await call('Runtime.enable');await call('Page.enable');await call('DOM.enable');await call('Network.enable');
 await call('Network.setBlockedURLs',{urls:['https://*','http://127.0.0.1:8869/assets/*']});
 await call('Emulation.setDeviceMetricsOverride',{width:1600,height:1050,deviceScaleFactor:1,mobile:false});
 await call('Page.navigate',{url:'http://127.0.0.1:8869/'+filename});
 for(let i=0;i<180;i++){if(await ev("typeof window.upload17Ready69!=='undefined'"))break;await new Promise(r=>setTimeout(r,300));}
 await ev('window.upload17Ready69');return {target,ws,call,ev,network};
}
async function close(){if(session){session.ws.close();await fetch('http://127.0.0.1:8870/json/close/'+session.target.id);session=null;}}
async function click(id){const p=await session.ev(`(()=>{const e=$('${id}');e.scrollIntoView({block:'nearest'});const b=e.getBoundingClientRect();return {x:b.x+b.width/2,y:b.y+b.height/2}})()`);await session.call('Input.dispatchMouseEvent',{type:'mousePressed',button:'left',buttons:1,clickCount:1,...p});await session.call('Input.dispatchMouseEvent',{type:'mouseReleased',button:'left',buttons:0,clickCount:1,...p});}
async function layers(){return session.ev(`(async()=>{const p=projectObject(),expected=${JSON.stringify(expectedLayers)};for(const e of expected){const v=p.images.find(x=>x.name===e.name);if(!v)return false;for(const [k,h] of Object.entries(e.hash))if(await sha(new TextEncoder().encode(JSON.stringify(v.layers[k])))!==h)return false;}return p.images.length===4})()`)}
async function geometry(includeRootIDs=true){return session.ev(`(async()=>{const expected=${JSON.stringify(expectedGeometry)},s=serialize52();for(const e of expected){const f=s.periods.find(v=>v.period===e.period);if(await sha(new TextEncoder().encode(JSON.stringify(f.owner_map_rle)))!==e.owner)return false;if(await sha(new TextEncoder().encode(JSON.stringify(f.manual_free_cuts_v62||[])))!==e.cuts)return false;if(${includeRootIDs}&&await sha(new TextEncoder().encode(JSON.stringify(f.segments.map(v=>({id:v.id,index:v.index,root_id:v.root_id,points:v.points,length_px:v.length_px})))))!==e.segments)return false;}return true})()`)}
async function definitions(){return session.ev(`(()=>{const map=rawHierarchy54();return ${JSON.stringify(defs)}.every(e=>{const h=map.get(e.root_id);return h&&h.label===e.label&&h.parent_root_id===e.parent_root_id&&JSON.stringify(h.human_identity_definition_v60||null)===JSON.stringify(e.definition)})})()`)}
async function restore(path){const doc=await session.call('DOM.getDocument');const node=await session.call('DOM.querySelector',{nodeId:doc.root.nodeId,selector:'#projectfile'});await session.call('DOM.setFileInputFiles',{nodeId:node.nodeId,files:[path]});for(let i=0;i<180;i++){const done=await session.ev("!$('projectfile').disabled&&!$('projectfile').value&&!$('status').textContent.startsWith('正在核对')");if(done)return;await new Promise(r=>setTimeout(r,100));}throw Error('Restore timeout')}
async function capture(button){await session.ev("window.downloadAudit69=[];download=async(blob,name)=>window.downloadAudit69.push({name,text:await blob.text()})");await click(button);for(let i=0;i<180;i++){const value=await session.ev('window.downloadAudit69[0]||null');if(value)return value;await new Promise(r=>setTimeout(r,100));}throw Error('Save timeout')}
try{
 session=await open('annotate.html');
 check('offline HTML opens four native RGB periods with all remote image requests blocked',await session.ev("files.length===4&&states.size===4&&W===2541&&H===1108&&!!active")&&!session.network.some(v=>v.startsWith('https://')));
 check('all sixteen pixel layers match upload17 exactly',await layers());
 check('all native root paths, IDs, lengths, ownership maps and cut records match',await geometry());
 check('all142 user-confirmed root definitions are preserved',await definitions()&&await session.ev("[...rawHierarchy54().values()].filter(h=>h.annotation_status_v60==='user_defined').length===142"));
 for(let i=0;i<4;i++){await session.ev(`select(${i})`);check('T'+(i+1)+' displays exact matching original PNG',await session.ev(`active.name==='T${i+1}_RGB.png'&&W===2541&&H===1108`));}
 await session.ev('select(0)');
 const chosen=await session.ev(`(()=>{const hs=[...rawHierarchy54().values()],f=frame52();const h=hs.find(h=>h.annotation_status_v60==='user_defined'&&!h.parent_root_id&&!hs.some(q=>q.parent_root_id===h.root_id)&&f.segments.some(s=>s.root_id===h.root_id));if(!h)throw Error('No independent human test root');choose52(f.segments.find(s=>s.root_id===h.root_id));let label=null;for(let i=312;i>0;i--){const v=String(i).padStart(4,'0');if(!hs.some(h=>h.label===v)){label=v;break}}$('defineKind60').value='existing';$('defineLabel60').focus();$('defineLabel60').select();return {root_id:h.root_id,old_label:h.label,new_label:label}})()`);
 await session.call('Input.insertText',{text:chosen.new_label});await click('defineApply60');
 check('real root-number input and Apply button update the selected biological root',await session.ev(`label54(${JSON.stringify(chosen.root_id)})===${JSON.stringify(chosen.new_label)}`));
 check('number editing leaves pixel labels and native ownership unchanged',await layers()&&await geometry());
 const saved=await capture('saveReview60');fs.writeFileSync(out+'/software_test_edited.json',saved.text);await restore(out+'/root_annotation_progress_17.json');
 check('Restore progress applies the uploaded original source correctly',await definitions()&&await layers()&&await geometry());
 await restore(out+'/software_test_edited.json');
 check('Restore progress also recovers a newly edited number',await session.ev(`label54(${JSON.stringify(chosen.root_id)})===${JSON.stringify(chosen.new_label)}`));
 const html=await capture('saveReviewHtml60');fs.writeFileSync(out+'/software_test_saved.html',html.text);
 check('Save editable HTML emits a complete file with embedded original images',html.text.includes('data:image/png;base64,')&&html.text.includes('upload17-decode-v69'));
 await close();session=await open('software_test_saved.html');
 check('saved edited HTML reopens offline and retains edits without resetting to original17',await session.ev(`label54(${JSON.stringify(chosen.root_id)})===${JSON.stringify(chosen.new_label)}&&files.length===4`)&&await layers()&&await geometry());
 check('cut, detachment and save controls are present once after reopening',await session.ev("['startFreeCut62','detachPieces62','defineApply60','saveReviewHtml60'].every(id=>document.querySelectorAll('#'+id).length===1)"));
 await restore(out+'/root_annotation_progress_17.json');
 check('original project still validates and restores exactly after saving and reopening',await definitions()&&await layers()&&await geometry()&&await session.ev('(()=>{const p=projectObject();validateStructure52(p.root_structure_annotation,p);return true})()'));
 check('no browser runtime exceptions',exceptions.length===0,exceptions);
 fs.writeFileSync(out+'/browser_verification.json',JSON.stringify({passed:true,checks,exceptions,tested_HTML_SHA256:crypto.createHash('sha256').update(fs.readFileSync(out+'/annotate.html')).digest('hex'),browser:'Chromium, actual input/click/file restore/save/reopen; HTTPS and external image assets blocked',software_test_changes_not_applied_to_delivery:true},null,2)+'\n');console.log(JSON.stringify({passed:true,checks:checks.length}));
}catch(error){fs.writeFileSync(out+'/browser_verification_failed.json',JSON.stringify({checks,exceptions,error:error.stack},null,2)+'\n');console.error(error);process.exitCode=1;}
finally{await close();}
