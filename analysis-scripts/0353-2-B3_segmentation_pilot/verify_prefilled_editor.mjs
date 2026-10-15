import fs from 'node:fs';
const pages=await (await fetch('http://127.0.0.1:8793/json/list')).json();const p=pages.find(p=>p.url.includes('8794/annotate.html'));const ws=new WebSocket(p.webSocketDebuggerUrl);await new Promise(r=>ws.onopen=r);let id=0;const pending=new Map();ws.onmessage=e=>{const v=JSON.parse(e.data);if(v.id){const h=pending.get(v.id);pending.delete(v.id);v.error?h.reject(Error(JSON.stringify(v.error))):h.resolve(v.result)}};
const call=(method,params={})=>new Promise((resolve,reject)=>{const n=++id;pending.set(n,{resolve,reject});ws.send(JSON.stringify({id:n,method,params}))});
const ev=async expression=>{const r=await call('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value};
await call('Page.enable');const loaded=new Promise(resolve=>{const listener=e=>{if(JSON.parse(e.data).method==='Page.loadEventFired'){ws.removeEventListener('message',listener);resolve()}};ws.addEventListener('message',listener)});await ev('states.clear();active=null');await call('Page.reload');await loaded;
const report=await ev(`(async()=>{
 await window.datasetReady;const checks=[];function check(name,ok){checks.push({name,pass:!!ok});if(!ok)throw Error(name)};
 check('60 native source images and 60 nonempty prefills load',files.length===60&&states.size===60&&[...states.values()].every(s=>s.mask.includes(255)));
 check('all prefilled images await person review', [...states.values()].every(s=>s.reviewed===false&&!s.annotator&&!!s.prefill_origin));
 check('neutral pending-review heading',document.querySelector('h1').textContent.includes('待复核')&&!document.querySelector('h1').textContent.includes('AI'));
 check('verified original image remains editable',active&&!$('fg').disabled&&$('image').width===448&&$('image').height===448);
 const hashes=[];for(let i=0;i<files.length;i++){await select(i);hashes.push({name:files[i].name,rgba_sha256:await sha(imgctx.getImageData(0,0,W,W).data)});check('prefill remains present after switching '+files[i].name,active.mask.includes(255)&&active.prefill_origin)}
 await select(0);const before=await sha(active.mask),oldColor=await sha(active.color),oldTemporal=await sha(active.temporal),oldViability=await sha(active.viability);
 $('brush').value='1';document.querySelector('[data-category="new"]').click();$('overlay').setPointerCapture=()=>{};const rect=$('overlay').getBoundingClientRect();$('overlay').onpointerdown({preventDefault(){},pointerId:1,clientX:rect.left+100.5*rect.width/W,clientY:rect.top+100.5*rect.height/W});end();
 check('existing prefill can be edited with class and pixel width',active.mask[100*W+100]===255&&active.temporal[100*W+100]===1);
 const e=new KeyboardEvent('keydown',{key:'z',ctrlKey:true,bubbles:true,cancelable:true});$('overlay').dispatchEvent(e);check('Ctrl+Z restores original prefill and all attributes',e.defaultPrevented&&await sha(active.mask)===before&&await sha(active.color)===oldColor&&await sha(active.temporal)===oldTemporal&&await sha(active.viability)===oldViability);
 document.querySelector('[data-period="4"]').click();await select(files.findIndex(f=>f.name==='B3_A0_C0_T4.png'));check('same-window period navigation retains T4 annotations',files[index].name==='B3_A0_C0_T4.png'&&active.mask.includes(255));
 await $('export').onclick();check('unreviewed prefill cannot be exported as completed human truth',$('status').textContent.includes('保存全部修改'));
 let blob;const realURL=URL.createObjectURL;URL.createObjectURL=b=>{blob=b;return 'blob:project-software-test'};const nativeClick=HTMLAnchorElement.prototype.click;HTMLAnchorElement.prototype.click=function(){};$('saveProject').click();const saved=JSON.parse(await blob.text());
 check('saving all 60 images does not require whole-image review',saved.records.length===60&&saved.records.every(r=>!r.reviewed)&&saved.not_training_ground_truth===true);
 const parsed=parseProject(saved);check('project restores all 4 layers byte-for-byte', [...states].every(([n,s])=>['mask','color','temporal','viability'].every(k=>s[k].every((v,i)=>v===parsed.get(n)[k][i]))));
 check('source provenance retained in saved project',saved.records.every(r=>r.prefill_origin&&r.image_sha256===expectedImages[r.name]));
 const bad=structuredClone(saved);bad.records[1].image_sha256='bad';let rejected=false;try{parseProject(bad)}catch{rejected=true}check('wrong-source project rejected',rejected);
 const id=files[index].name.replace(/\\.png$/,'');$('annotator').value='SOFTWARE_UI_TEST_NOT_HUMAN_ANNOTATION';$('reviewed').checked=true;await $('export').onclick();const bytes=new Uint8Array(await blob.arrayBuffer());check('reviewed software fixture generates a ZIP',bytes[0]===80&&bytes[1]===75);
 let raw='';for(let i=0;i<bytes.length;i+=16384)raw+=String.fromCharCode(...bytes.slice(i,i+16384));const zipBase64=btoa(raw);states.clear();active=null;return {checks,rgba_hashes:hashes,zipBase64,scientific_metrics_not_evaluated:true};
})()`);
fs.writeFileSync('/tmp/B3_prefilled_export_fixture.zip',Buffer.from(report.zipBase64,'base64'));delete report.zipBase64;fs.writeFileSync('/tmp/B3_prefilled_browser_validation.json',JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify({passed:report.checks.length,failed:report.checks.filter(x=>!x.pass).length}));ws.close();
