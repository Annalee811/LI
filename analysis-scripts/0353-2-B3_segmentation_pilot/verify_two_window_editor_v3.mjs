import fs from 'node:fs';
const output=process.argv[2]||'/workspace/LI/results/B3_two_groups_review';
const pages=await(await fetch('http://127.0.0.1:8793/json/list')).json();
const page=pages.find(p=>p.url==='http://127.0.0.1:8794/annotate.html');
if(!page)throw Error('Review page not open');
const ws=new WebSocket(page.webSocketDebuggerUrl);await new Promise(r=>ws.onopen=r);
let sequence=0;const pending=new Map();
ws.onmessage=e=>{const r=JSON.parse(e.data);if(r.id){const p=pending.get(r.id);pending.delete(r.id);r.error?p.reject(Error(JSON.stringify(r.error))):p.resolve(r.result)}};
const call=(method,params={})=>new Promise((resolve,reject)=>{const id=++sequence;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}))});
const evaluate=async expression=>{const r=await call('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value};
await call('Page.enable');
const loaded=new Promise(resolve=>{const f=e=>{if(JSON.parse(e.data).method==='Page.loadEventFired'){ws.removeEventListener('message',f);resolve()}};ws.addEventListener('message',f)});
await evaluate('states.clear();active=null');await call('Page.reload');await loaded;
try{
 const report=await evaluate(`(async()=>{
 await window.datasetReady;const checks=[],check=(name,ok)=>{checks.push({name,pass:!!ok});if(!ok)throw Error(name)};
 const keys=['mask','color','temporal','viability','cohort','scene','instance'];
 check('exactly two windows, eight original images and seven annotation layers',files.length===8&&states.size===8&&[...states.values()].every(s=>keys.every(k=>s[k].length===N)));
 check('all T1 roots are baseline, with no newborn or persistent claim',[...states].filter(([n])=>n.endsWith('_T1.png')).every(([n,s])=>s.mask.every((v,i)=>v!==255||(s.cohort[i]===1&&s.temporal[i]===0))));
 check('T1 controls prohibit newborn, elongation and persistence',$('temporal').disabled&&$('cohort').disabled&&['new','growing','persistent'].every(k=>document.querySelector('[data-category="'+k+'"]').disabled));
 check('root and scene labels agree, non-root pixels have no root attributes',[...states.values()].every(s=>s.mask.every((v,i)=>(s.scene[i]===1)===(v===255)&&(v===255||['color','temporal','viability','cohort','instance'].every(k=>s[k][i]===0)))));
 check('unpainted background remains unreviewed',[...states.values()].every(s=>s.mask.includes(64)&&!s.reviewed&&!s.annotator));
 check('no biological death or birth inferred in supplied labels',[...states.values()].every(s=>!s.viability.some(Boolean)&&!s.temporal.includes(1)));
 check('uncertainty hidden by default and opacity adjustable',!$('showUncertain').checked&&Number($('opacity').value)===175);
 const rgba=[],maskHashes=[];
 for(let i=0;i<files.length;i++){await select(i);rgba.push({name:files[i].name,sha256:await sha(imgctx.getImageData(0,0,W,W).data)});maskHashes.push(await sha(active.mask))}
 check('all eight originals open and every period has a distinct root mask',rgba.length===8&&new Set(maskHashes).size===8);
 await select(0);$('focusRoot').value='1';await $('trackRoot').onclick();
 check('root ID one has four-period panels in first window',$('trackingGrid').querySelectorAll('canvas').length===4&&[...$('trackingGrid').children].every(x=>x.textContent.includes('此编号已标出')));
 await select(4);await $('trackRoot').onclick();
 check('four-period tracking switches to second window',$('trackingGrid').querySelectorAll('canvas').length===4&&$('trackingNote').textContent.startsWith('B3_A0_C1'));
 $('focusRoot').value='0';render();
 for(const layer of ['overview','cohort','scene','color','temporal','viability']){$('displayLayer').value=layer;$('displayLayer').onchange();check('render independent layer '+layer,ovctx.getImageData(0,0,W,W).data.length===N*4)}
 $('displayLayer').value='overview';$('brush').value='1';$('overlay').setPointerCapture=()=>{};
 function paint(x,y){const r=$('overlay').getBoundingClientRect();$('overlay').onpointerdown({preventDefault(){},pointerId:1,clientX:r.left+(x+.5)*r.width/W,clientY:r.top+(y+.5)*r.height/W});end()}
 const p=150*W+150;const before=Object.fromEntries(keys.map(k=>[k,active[k][p]]));$('cohort').value='4';$('temporal').value='1';$('fg').click();paint(150,150);
 check('T1 paint forces baseline despite stale attribute controls',active.mask[p]===255&&active.cohort[p]===1&&active.temporal[p]===0);
 $('overlay').dispatchEvent(new KeyboardEvent('keydown',{key:'z',ctrlKey:true,bubbles:true,cancelable:true}));
 check('Ctrl+Z restores all seven arrays',keys.every(k=>active[k][p]===before[k]));
 for(const value of [2,3,4,0]){document.querySelector('[data-scene="'+value+'"]').click();paint(150,150);check('object brush '+value+' clears root classification',active.scene[p]===value&&active.mask[p]===(value===0?64:0)&&['cohort','temporal','instance'].every(k=>active[k][p]===0));$('undo').click()}
 const backupData=serializeProject(),restored=parseProject(backupData);
 check('all eight images round-trip every label array',states.size===8&&[...states].every(([n,s])=>keys.every(k=>s[k].every((v,i)=>v===restored.get(n)[k][i]))));
 let savedBlob;const oldURL=URL.createObjectURL,oldClick=HTMLAnchorElement.prototype.click;
 URL.createObjectURL=b=>{savedBlob=b;return 'blob:software-test'};HTMLAnchorElement.prototype.click=function(){};
 $('saveProject').onclick();const saved=JSON.parse(await savedBlob.text());check('save project includes all eight without human review',saved.records.length===8&&saved.records.every(r=>!r.reviewed&&!r.annotator));
 await select(5);$('fg').click();$('rootId').value='1';$('cohort').value='1';$('temporal').value='3';paint(150,150);check('persistent root and T1 baseline origin are independent',active.temporal[p]===3&&active.cohort[p]===1&&active.instance[p]===1);$('undo').click();
 await select(0);$('annotator').value='SOFTWARE_TEST_ONLY';$('reviewed').checked=true;await $('export').onclick();const bytes=new Uint8Array(await savedBlob.arrayBuffer());check('reviewed software fixture exports ZIP',bytes[0]===80&&bytes[1]===75);
 let raw='';for(let i=0;i<bytes.length;i+=16384)raw+=String.fromCharCode(...bytes.slice(i,i+16384));const exportFixture=btoa(raw);
 $('annotator').value='';$('reviewed').checked=false;active.reviewed=false;URL.createObjectURL=oldURL;HTMLAnchorElement.prototype.click=oldClick;
 states.clear();active=null;loadPrefills();await select(4);$('focusRoot').value='1';await $('trackRoot').onclick();
 return {checks,rgba_hashes:rgba,exportFixture,software_only:true,scientific_accuracy_not_evaluated:true};
})()`);
 fs.writeFileSync('/tmp/B3_two_window_export_fixture.zip',Buffer.from(report.exportFixture,'base64'));delete report.exportFixture;
 fs.writeFileSync(output+'/browser_validation.json',JSON.stringify(report,null,2)+'\n');
 const screenshot=await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:true});fs.writeFileSync('/tmp/B3_two_window_editor.png',Buffer.from(screenshot.data,'base64'));
 console.log(JSON.stringify({passed:report.checks.length,failed:report.checks.filter(c=>!c.pass).length}));
}finally{ws.close()}
