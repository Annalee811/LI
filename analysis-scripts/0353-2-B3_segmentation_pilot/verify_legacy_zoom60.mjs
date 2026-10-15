import fs from 'node:fs';
const pages=await(await fetch('http://127.0.0.1:8793/json/list')).json(),p=pages.find(p=>p.url.endsWith('/index.html'));
const ws=new WebSocket(p.webSocketDebuggerUrl);await new Promise(r=>ws.onopen=r);let id=0;const jobs=new Map();ws.onmessage=e=>{const d=JSON.parse(e.data);if(d.id){const j=jobs.get(d.id);jobs.delete(d.id);d.error?j.reject(Error(JSON.stringify(d.error))):j.resolve(d.result)}};
const call=(method,params={})=>new Promise((resolve,reject)=>{const n=++id;jobs.set(n,{resolve,reject});ws.send(JSON.stringify({id:n,method,params}))});const ev=async expression=>{const r=await call('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value};
try{
 const report=await ev(`(async()=>{
 await window.datasetReady;await window.zoom60Ready;const checks=[],check=(name,ok)=>{checks.push({name,pass:!!ok});if(!ok)throw Error(name)},ui=window.B3Zoom60Installed,vp=ui.viewport;
 check('same original 60 images load for fresh manual annotation',files.length===60&&active&&!active.mask.includes(255));
 check('maximum zoom input and slider are 60',Number($('zoom').max)===60&&Number($('b3ZoomRange').max)===60);
 ui.setZoom(60);check('60x uses original-sized RGB and label buffers',ui.zoom===60&&$('stage').offsetWidth===448*60&&$('image').width===448&&$('overlay').width===448);
 ui.setZoom(100);check('zoom cannot exceed 60',ui.zoom===60);ui.setZoom(10);vp.scrollLeft=183.25*10-vp.clientWidth/2;vp.scrollTop=208.125*10-vp.clientHeight/2;
 const center=ui.captureCenter();for(const j of [1,2,3,0]){await select(j);const next=ui.captureCenter();check('period '+(j+1)+' keeps 10x and native center',ui.zoom===10&&Math.abs(next.x-center.x)<.11&&Math.abs(next.y-center.y)<.11)}
 ui.setZoom(60);const center60=ui.captureCenter();await select(4);const next60=ui.captureCenter();check('another image keeps 60x and native center',ui.zoom===60&&Math.abs(next60.x-center60.x)<.03&&Math.abs(next60.y-center60.y)<.03);
 const side=document.querySelector('.controls').getBoundingClientRect(),view=vp.getBoundingClientRect();check('tools stay beside zoomed viewport',side.left>=0&&side.top>=0&&side.bottom<=innerHeight+1&&side.right<view.left&&view.right<=innerWidth+1);
 const beforeMask=await sha(active.mask),beforeRGB=await sha(imgctx.getImageData(0,0,W,W).data);$('b3Pan').click();vp.setPointerCapture=()=>{};
 function pointer(type,x,y){$('overlay').dispatchEvent(new PointerEvent(type,{clientX:view.left+view.width/2+x,clientY:view.top+view.height/2+y,button:0,pointerId:1,bubbles:true,cancelable:true}))}
 const oldLeft=vp.scrollLeft,oldTop=vp.scrollTop;pointer('pointerdown',0,0);pointer('pointermove',100,120);pointer('pointerup',100,120);
 check('pan moves viewport without painting or changing RGB',vp.scrollLeft===oldLeft-100&&vp.scrollTop===oldTop-120&&beforeMask===await sha(active.mask)&&beforeRGB===await sha(imgctx.getImageData(0,0,W,W).data));
 document.querySelector('[data-category="brown"]').click();check('choosing a class leaves pan mode and keeps magnification',!ui.pan&&ui.zoom===60&&$('color').value==='2');
 $('brush').value='1';$('overlay').setPointerCapture=()=>{};const b=$('overlay').getBoundingClientRect(),x=view.left+view.width/2,y=view.top+view.height/2,px=Math.floor((x-b.left)*W/b.width),py=Math.floor((y-b.top)*W/b.height),i=py*W+px;
 $('overlay').onpointerdown({clientX:x,clientY:y,pointerId:1,preventDefault(){}});end();check('60x stroke hits correct native pixel and class',active.mask[i]===255&&active.color[i]===2);$('undo').click();check('undo at 60x restores labels without changing view',await sha(active.mask)===beforeMask&&ui.zoom===60);
 const focus=ui.captureCenter();document.querySelector('[data-period="4"]').click();await new Promise(r=>setTimeout(r,200));check('same-window period shortcut keeps 60x and position',files[index].name.endsWith('_T4.png')&&ui.zoom===60&&Math.abs(ui.captureCenter().x-focus.x)<.03);
 $('zoom').value='20';$('zoom').dispatchEvent(new Event('input'));check('typed zoom applies correctly',ui.zoom===20);$('b3ZoomRange').value='40';$('b3ZoomRange').dispatchEvent(new Event('input'));check('slider supports high zoom',ui.zoom===40);
 const rgba=[];for(const j of [0,1,2,3]){await select(j);rgba.push({name:files[j].name,rgba_sha256:await sha(imgctx.getImageData(0,0,W,W).data)})}
 return {checks,rgba,scope:'UI only; original images and label schema',no_existing_annotations_cleared:true,software_only:true};
 })()`);
 fs.writeFileSync('results/B3_legacy_zoom60_v1/browser_validation.json',JSON.stringify(report,null,2)+'\n');
 const shot=await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});fs.writeFileSync('results/B3_legacy_zoom60_v1/toolbar_preview.png',Buffer.from(shot.data,'base64'));
 console.log(JSON.stringify({passed:report.checks.length,failed:report.checks.filter(c=>!c.pass).length}));
}finally{ws.close()}
