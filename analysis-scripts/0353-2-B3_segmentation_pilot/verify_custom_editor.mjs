import fs from 'node:fs';
const page=(await(await fetch('http://127.0.0.1:8793/json/list')).json()).find(x=>x.type==='page');
const ws=new WebSocket(page.webSocketDebuggerUrl);await new Promise(r=>ws.onopen=r);let id=0;const jobs=new Map();ws.onmessage=e=>{const d=JSON.parse(e.data);if(d.id){const j=jobs.get(d.id);jobs.delete(d.id);d.error?j.reject(Error(JSON.stringify(d.error))):j.resolve(d.result)}};
const call=(method,params={})=>new Promise((resolve,reject)=>{const n=++id;jobs.set(n,{resolve,reject});ws.send(JSON.stringify({id:n,method,params}))});
const ev=async expression=>{const r=await call('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true,replMode:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value};
const wait=async expression=>{for(let i=0;i<100;i++){if(await ev(expression))return;await new Promise(r=>setTimeout(r,100))}throw Error('timeout: '+expression)};
const checks=[];const check=async(name,expression)=>{const pass=await ev(expression);checks.push({name,pass:!!pass});if(!pass)throw Error(name)};
async function setFiles(selector,paths){const {root}=await call('DOM.getDocument');const {nodeId}=await call('DOM.querySelector',{nodeId:root.nodeId,selector});await call('DOM.setFileInputFiles',{nodeId,files:paths})}
async function mouse(type,x,y,extra={}){return call('Input.dispatchMouseEvent',{type,x,y,button:'left',...extra})}
try{
 await call('Emulation.setDeviceMetricsOverride',{width:1400,height:1000,deviceScaleFactor:1,mobile:false});
 await call('Browser.setDownloadBehavior',{behavior:'allow',downloadPath:'/tmp/B3_custom_test/downloads'});
 await call('Page.navigate',{url:'http://127.0.0.1:8794/annotate.html'});await wait('!!window.B3Zoom60Installed');
 await check('starts empty with no fixed 60-patch dataset',"files.length===0&&!active&&!document.getElementById('builtin')");
 const base='/tmp/B3_custom_test/images/';await setFiles('#imagefiles',['T1_full.png','T2_full.png','T3_full.jpg','T4_full.webp'].map(n=>base+n));
 await wait('!!active&&files.length===4');
 await check('actual file picker enables full 2541x1867 RGB annotation',"W===2541&&H===1867&&N===2541*1867&&active.mask.length===N&&!$('fg').disabled&&!$('export').disabled");
 await ev("window.rgbBefore=await sha(imgctx.getImageData(0,0,W,H).data);window.ui=B3Zoom60Installed;ui.setZoom(60);ui.viewport.scrollLeft=1100*60;ui.viewport.scrollTop=900*60;document.querySelector('[data-category=\"persistent\"]').click();$('brush').value='3';");
 await check('60x keeps rectangular native-size canvases',"ui.zoom===60&&$('image').width===2541&&$('image').height===1867&&$('stage').offsetHeight===1867*60");
 const pos=await ev("(()=>{const b=ui.viewport.getBoundingClientRect();const x=b.left+b.width/2,y=b.top+b.height/2,c=$('overlay').getBoundingClientRect();window.testPX=Math.floor((x-c.left)*W/c.width);window.testPY=Math.floor((y-c.top)*H/c.height);window.testIndex=testPY*W+testPX;return{x,y}})()");
 await mouse('mousePressed',pos.x,pos.y,{clickCount:1});await mouse('mouseMoved',pos.x+10,pos.y+10,{buttons:1});await mouse('mouseReleased',pos.x+10,pos.y+10,{clickCount:1});
 await check('real pointer stroke paints the correct native pixel at 60x',"active.mask[testIndex]===255&&active.temporal[testIndex]===3&&active.undo.length===1");
 await check('root annotation leaves original RGB unchanged',"await sha(imgctx.getImageData(0,0,W,H).data)===rgbBefore");
 await call('Input.dispatchKeyEvent',{type:'keyDown',key:'z',code:'KeyZ',modifiers:2});await call('Input.dispatchKeyEvent',{type:'keyUp',key:'z',code:'KeyZ',modifiers:2});
 await check('Ctrl+Z restores all brush attribute layers',"active.mask[testIndex]===64&&active.temporal[testIndex]===0&&ui.zoom===60");
 await mouse('mousePressed',pos.x,pos.y,{clickCount:1});await mouse('mouseReleased',pos.x,pos.y,{clickCount:1});
 await ev('window.focusBefore=ui.captureCenter();window.t1Mask=await sha(active.mask);');
 for(const i of [1,2,3,0]){
  await ev(`await select(${i})`);
  await check(`period ${i+1} editable at same 60x and native view center`,"active&&!$('fg').disabled&&ui.zoom===60&&Math.abs(ui.captureCenter().x-focusBefore.x)<.04&&Math.abs(ui.captureCenter().y-focusBefore.y)<.04");
 }
 await check('switching periods preserves prior annotations',"await sha(active.mask)===t1Mask&&active.temporal[testIndex]===3");
 await ev("$('annotator').value='browser-test';saveProject();window.projectText=JSON.stringify(projectObject());");
 await check('unfinished annotations can be saved without whole-image review',"!active.reviewed&&JSON.parse(projectText).images.length===4");
 // Export while immediately switching to another-sized image: the ZIP must keep the source snapshot.
 await ev("$('reviewed').checked=true;window.exportPromise=$('export').onclick();");
 await setFiles('#imagefiles',[base+'rectangle.png']);await wait('!!active&&W===613');await ev('await exportPromise');
 await check('arbitrary non-square image is editable without a known filename',"W===613&&H===913&&active.mask.length===613*913&&!$('fg').disabled");
 await ev("ui.setZoom(2);ui.viewport.scrollLeft=400;ui.viewport.scrollTop=900;document.querySelector('[data-category=\"brown\"]').click();$('brush').value='1';");
 const rect=await ev("(()=>{const b=ui.viewport.getBoundingClientRect(),x=b.left+b.width/2,y=b.top+b.height/2,c=$('overlay').getBoundingClientRect();window.rx=Math.floor((x-c.left)*W/c.width);window.ry=Math.floor((y-c.top)*H/c.height);return{x,y}})()");
 await mouse('mousePressed',rect.x,rect.y,{clickCount:1});await mouse('mouseReleased',rect.x,rect.y,{clickCount:1});
 await check('rectangular brush uses height correctly, including y beyond 448',"ry>448&&active.mask[ry*W+rx]===255&&active.color[ry*W+rx]===2");
 await ev('window.rectKey=active.key;window.rectMask=await sha(active.mask)');
 await setFiles('#imagefiles',[base+'other/rectangle.png']);await wait('!!active&&active.key!==rectKey');
 await check('different-content same-name images never share annotations',"!active.mask.includes(255)&&states.has(rectKey)");
 await setFiles('#imagefiles',[base+'rectangle.png']);await wait('!!active&&active.key===rectKey');
 await check('reselecting identical original RGB restores in-page labels',"await sha(active.mask)===rectMask");
 await check('fixed tools remain beside the image at high zoom',"(()=>{ui.setZoom(60);const a=document.querySelector('.controls').getBoundingClientRect(),b=ui.viewport.getBoundingClientRect();return a.right<b.left&&a.bottom<=innerHeight+1&&b.right<=innerWidth+1})()");
 await ev("window.fullSave=JSON.stringify(projectObject());window.allLayerHashes=Object.fromEntries(await Promise.all([...states].map(async([k,s])=>[k,await Promise.all(['mask','color','temporal','viability'].map(layer=>sha(s[layer])))])));states.clear();active=null;await importProject(new File([fullSave],'progress.json',{type:'application/json'}));");
 await check('save/import round-trip preserves every annotation layer and dimension',"(await Promise.all([...states].map(async([k,s])=>(await Promise.all(['mask','color','temporal','viability'].map(layer=>sha(s[layer])))).every((v,i)=>v===allLayerHashes[k][i])))).every(Boolean)&&active.width===613&&active.height===913");
 await ev("window.keysBefore=[...states.keys()].join('|');try{const bad=JSON.parse(fullSave);bad.images[1].layers.mask=[999,1];await importProject(new File([JSON.stringify(bad)],'bad.json'))}catch(e){window.rejectedBad=true}");
 await check('invalid project import rejects atomically without erasing labels',"rejectedBad&&[...states.keys()].join('|')===keysBefore");
 const report={checks,passed:checks.length,failed:checks.filter(c=>!c.pass).length,scope:'Browser UI and pixel/annotation integrity tests using synthetic RGB; no scientific root-identification claim.'};
 fs.writeFileSync('results/custom_image_editor_v1/browser_validation.json',JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify({passed:report.passed,failed:report.failed}));
}finally{ws.close()}
