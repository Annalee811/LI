import fs from 'node:fs';
const pages=await (await fetch('http://127.0.0.1:8793/json/list')).json();const p=pages.find(p=>p.url.includes('8794/annotate.html'));if(!p)throw Error('missing editor tab');
const ws=new WebSocket(p.webSocketDebuggerUrl);await new Promise(r=>ws.onopen=r);let id=0;const pending=new Map();ws.onmessage=e=>{const v=JSON.parse(e.data);if(v.id){const h=pending.get(v.id);pending.delete(v.id);v.error?h.reject(Error(JSON.stringify(v.error))):h.resolve(v.result)}};
const call=(method,params={})=>new Promise((resolve,reject)=>{const n=++id;pending.set(n,{resolve,reject});ws.send(JSON.stringify({id:n,method,params}))});
const evaluate=async expression=>{const r=await call('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value};
await call('Page.enable');const loaded=new Promise(resolve=>{const listener=e=>{if(JSON.parse(e.data).method==='Page.loadEventFired'){ws.removeEventListener('message',listener);resolve()}};ws.addEventListener('message',listener)});await evaluate('states.clear();active=null');await call('Page.reload');await loaded;
const report=await evaluate(`(async()=>{
 await window.datasetReady;states.clear();active=null;await loadEmbedded();const checks=[];function check(name,pass){checks.push({name,pass:!!pass});if(!pass)throw Error(name)};
 check('60 embedded images load and verified editing enabled',files.length===60&&active&&!$('export').disabled&&$('image').width===448);
 $('brush').value='1';const before=await sha(imgctx.getImageData(0,0,W,W).data);
 function paint(x,y){const r=$('overlay').getBoundingClientRect(),cx=r.left+(x+.5)*r.width/W,cy=r.top+(y+.5)*r.height/W;$('overlay').onpointerdown({preventDefault(){},pointerId:1,clientX:cx,clientY:cy});end()}
 // A synthetic pointer calls the same handlers; pointer capture needs a real pointer, so only stub capture for this software test.
 $('overlay').setPointerCapture=()=>{};
 let x=20;const marks=[];for(const key of Object.keys(classes).filter(k=>k!=='ignore')){document.querySelector('[data-category="'+key+'"]').click();paint(x,20);const i=20*W+x,rgb=Array.from(ovctx.getImageData(x,20,1,1).data);check('paint '+key+' color and attribute encoding',rgb.slice(0,3).every((v,i)=>Math.abs(v-classes[key].rgb[i])<=2)&&active.mask[i]===255&&active.color[i]===(classes[key].color||0)&&active.temporal[i]===(classes[key].temporal||0)&&active.viability[i]===(classes[key].viability||0));marks.push({key,x,y:20,rgb});x+=20}
 check('different colors coexist',marks.every(m=>Array.from(ovctx.getImageData(m.x,m.y,1,1).data).slice(0,3).every((v,i)=>Math.abs(v-classes[m.key].rgb[i])<=2)));
 const originalMask=active.mask.slice(),attrsBefore=await sha(active.color);
 $('color').value='2';$('temporal').value='3';$('viability').value='3';$('color').onchange();paint(400,400);
 for(const [layer,key] of [['overview','dead'],['color','brown'],['temporal','persistent'],['viability','dead']]){$('displayLayer').value=layer;$('displayLayer').onchange();check('mixed attributes display '+layer,Array.from(ovctx.getImageData(400,400,1,1).data).slice(0,3).every((v,i)=>Math.abs(v-classes[key].rgb[i])<=2))}
 check('changing view preserves existing annotations',active.mask.slice(0,100*W).every((v,i)=>v===originalMask[i])&&active.color.slice(0,100*W).every((v,i)=>v===((i===20*W+40)?1:(i===20*W+60)?2:0)));
 $('bg').click();paint(400,400);check('erase clears all attributes',active.mask[400*W+400]===0&&active.color[400*W+400]===0&&active.temporal[400*W+400]===0&&active.viability[400*W+400]===0);
 $('undo').click();check('undo restores all attributes',active.mask[400*W+400]===255&&active.color[400*W+400]===2&&active.temporal[400*W+400]===3&&active.viability[400*W+400]===3);
 $('ignore').click();paint(11,11);check('uncertainty is distinct gray, not a death label',active.mask[11*W+11]===128&&Array.from(ovctx.getImageData(11,11,1,1).data).slice(0,3).every((v,i)=>Math.abs(v-classes.ignore.rgb[i])<=2));
 check('original RGB pixels unchanged by colored annotation',before===await sha(imgctx.getImageData(0,0,W,W).data));
 function key(options={},target=window){const e=new KeyboardEvent('keydown',{key:'z',ctrlKey:true,bubbles:true,cancelable:true,...options});target.dispatchEvent(e);return e}
 $('annotator').focus();$('fg').click();paint(390,390);check('drawing focuses canvas so Ctrl+Z works after editing a text field',document.activeElement===$('overlay'));let n=active.undo.length;const ctrl=key();check('Ctrl+Z undoes exactly one stroke and restores every layer',ctrl.defaultPrevented&&active.undo.length===n-1&&active.mask[390*W+390]===64&&active.color[390*W+390]===0);
 paint(390,390);n=active.undo.length;const cmd=key({ctrlKey:false,metaKey:true});check('Mac Command+Z invokes the same undo',cmd.defaultPrevented&&active.undo.length===n-1&&active.mask[390*W+390]===64);
 paint(390,390);n=active.undo.length;for(const [name,options] of [['Shift+Ctrl+Z',{shiftKey:true}],['Alt+Ctrl+Z',{altKey:true}],['key repeat',{repeat:true}],['IME composition',{isComposing:true}],['plain z',{ctrlKey:false}]]){const e=key(options);check(name+' does not invoke canvas undo',!e.defaultPrevented&&active.undo.length===n)}
 for(const id of ['annotator','evidence','color','reviewed']){const e=key({},$(id));check('native input undo preserved: '+id,!e.defaultPrevented&&active.undo.length===n)}
 const editable=document.createElement('div');editable.contentEditable='true';const child=document.createElement('span');editable.append(child);document.body.append(editable);const ce=key({},child);check('contenteditable descendants preserve native undo',!ce.defaultPrevented&&active.undo.length===n);editable.remove();
 key();check('undo invalidates whole-image review',!$('reviewed').checked&&!active.reviewed);
 const saved=active;active=null;const readonly=key();check('readonly preview ignores undo shortcut',!readonly.defaultPrevented);active=saved;
 const history=active.undo;active.undo=[];const empty=key();check('empty history is a safe no-op',empty.defaultPrevented&&active.mask[11*W+11]===128);active.undo=history;
 $('annotator').value='SOFTWARE_UI_TEST_NOT_HUMAN_ANNOTATION';await $('export').onclick();check('unreviewed export rejected',$('status').textContent.includes('确认已检查'));
 $('reviewed').checked=true;await $('export').onclick();check('confirmed death requires evidence',$('status').textContent.includes('判断依据'));
 $('evidence').value='Synthetic software test only; not a biological death judgment';let blob;URL.createObjectURL=b=>{blob=b;return 'blob:software-test'};HTMLAnchorElement.prototype.click=function(){};await $('export').onclick();check('reviewed synthetic export succeeds',blob instanceof Blob&&$('status').textContent.includes('已导出'));
 const bytes=new Uint8Array(await blob.arrayBuffer());let raw='';for(let i=0;i<bytes.length;i+=16384)raw+=String.fromCharCode(...bytes.slice(i,i+16384));
 const exportBase64=btoa(raw);states.clear();active=null;return {editor:'v4',synthetic_software_checks_only:true,checks,marks,exportBase64};
})()`);
fs.writeFileSync('/tmp/B3_v4_export_fixture.zip',Buffer.from(report.exportBase64,'base64'));delete report.exportBase64;fs.writeFileSync('/tmp/B3_v4_browser_validation.json',JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify({passed:report.checks.length,failed:report.checks.filter(x=>!x.pass).length}));ws.close();
