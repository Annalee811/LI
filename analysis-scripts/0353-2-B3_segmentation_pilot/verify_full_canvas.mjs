import fs from 'node:fs';
const output=process.argv[2]||'/workspace/LI/results/B3_full_canvas_v1';
const scriptDir=new URL('.',import.meta.url);
async function connect(target){const ws=new WebSocket(target.webSocketDebuggerUrl);await new Promise(r=>ws.onopen=r);let id=0;const pending=new Map();ws.onmessage=e=>{const r=JSON.parse(e.data);if(r.id){const p=pending.get(r.id);pending.delete(r.id);r.error?p.reject(Error(JSON.stringify(r.error))):p.resolve(r.result)}};const call=(method,params={})=>new Promise((resolve,reject)=>{const n=++id;pending.set(n,{resolve,reject});ws.send(JSON.stringify({id:n,method,params}))});const evaluate=async expression=>{const r=await call('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value};return{ws,call,evaluate}}
const pages=await(await fetch('http://127.0.0.1:8793/json/list')).json();
const old=await connect(pages.find(p=>p.url.endsWith('/old_v4.html')));
for(const p of pages.filter(p=>p.url.endsWith('/annotate.html'))){const previous=await connect(p);await previous.evaluate('modified=false');await old.call('Target.closeTarget',{targetId:p.id});previous.ws.close()}
const oldHashes=await old.evaluate(`(async()=>{await window.datasetReady;active=null;states.clear();const names=['B3_A0_C0_T1.png','B3_A0_C0_T2.png','B3_A0_C1_T3.png','B3_A0_C1_T4.png','B3_B0_C0_T1.png','B3_B0_C0_T2.png','B3_B0_C4_T3.png','B3_B0_C4_T4.png'];for(let j=0;j<names.length;j++){await select(files.findIndex(f=>f.name===names[j]));for(let x=80;x<105;x++){const i=(60+j)*W+x;active.mask[i]=255;active.color[i]=j%3;active.temporal[i]=/_T1/.test(names[j])?3:j%4;active.viability[i]=0}active.mask[12*W+12]=0;active.mask[14*W+14]=128;active.modified=true}remember();return await Promise.all([...states].map(async([n,s])=>[n,await Promise.all(['mask','color','temporal','viability'].map(k=>sha(s[k])))]))})()`);
const recovered=await old.evaluate(fs.readFileSync(new URL('preserve_open_editor.js',scriptDir),'utf8'));
if(recovered.images!==8)throw Error('Old-page recovery did not capture exactly eight visited images');
const backup=await old.evaluate('window.B3_PRESERVED_PROJECT_JSON');fs.writeFileSync('/tmp/B3_old8_recovered_fixture.json',backup);
const afterHashes=await old.evaluate(`Promise.all([...states].map(async([n,s])=>[n,await Promise.all(['mask','color','temporal','viability'].map(k=>sha(s[k])))]))`);
if(JSON.stringify(oldHashes)!==JSON.stringify(afterHashes))throw Error('Recovery altered old annotation arrays');
await old.call('Target.createTarget',{url:'about:blank'});
const pages2=await(await fetch('http://127.0.0.1:8793/json/list')).json();const app=await connect(pages2.find(p=>p.url==='about:blank'));
await app.call('Page.enable');const loaded=new Promise(resolve=>{const listener=e=>{if(JSON.parse(e.data).method==='Page.loadEventFired'){app.ws.removeEventListener('message',listener);resolve()}};app.ws.addEventListener('message',listener)});await app.call('Page.navigate',{url:'http://127.0.0.1:8794/annotate.html'});await loaded;
await app.evaluate('window.datasetReady');
await app.evaluate('window.fixtureBackup='+JSON.stringify(backup));
const report=await app.evaluate(`(async()=>{
 const checks=[],check=(name,ok)=>{checks.push({name,pass:!!ok});if(!ok)throw Error(name)};
 check('four full-size registered RGB images loaded separately',tiles.length===4&&tiles.every(t=>t.width===2541&&t.height===1867&&t.x===0));
 check('no old assistant annotations prefilled',tiles.every(t=>!t.state.mask.includes(255)));
 const migrated=await importProject(window.fixtureBackup,'SOFTWARE_OLD8_FIXTURE.json');check('all eight old images import without conflicts or scaling',migrated.imported_images===8&&migrated.conflicts_kept_current===0&&!migrated.scaled);
 const old=JSON.parse(window.fixtureBackup);for(const rec of old.records){const m=mappingByName.get(rec.name),t=tiles.find(t=>t.name===m.full_name);for(const key of keys){const a=decode(rec[key],448*448);let same=true;for(let y=0;y<448&&same;y++)for(let x=0;x<448;x++)if(a[y*448+x]!==t.state[key][(m.full_y+y)*t.width+m.full_x+x]){same=false;break}check('exact old pixels preserved '+rec.name+'/'+key,same)}}
 check('original user backup text retained verbatim',archives[0].original_text===window.fixtureBackup);
 const duplicate=await importProject(window.fixtureBackup,'same_backup.json');check('reimport is idempotent',duplicate.written_pixels===0&&duplicate.conflicts_kept_current===0&&archives.length===1);
 const modifiedLegacy=JSON.parse(window.fixtureBackup),row=modifiedLegacy.records[0];let array=decode(row.color,448*448);array[60*448+80]=2;row.color=encode(array);const conflict=await importProject(JSON.stringify(modifiedLegacy),'SOFTWARE_CONFLICT.json');check('conflicting pixels retained rather than overwritten',conflict.conflicts_kept_current===1);
 const beforeBad=tiles.reduce((n,t)=>n+t.state.mask.reduce((a,v)=>a+(v===255),0),0);const bad=JSON.parse(window.fixtureBackup);bad.records.at(-1).image_sha256='bad';let rejected=false;try{await importProject(JSON.stringify(bad))}catch(e){rejected=true}check('invalid source hash rejected before any mutation',rejected&&beforeBad===tiles.reduce((n,t)=>n+t.state.mask.reduce((a,v)=>a+(v===255),0),0));
 const saved=serialize();let count=0;for(const row of saved.sources){const t=tiles.find(t=>t.name===row.name);for(const k of keys){const a=decode(row[k],t.width*t.height);check('full project exact roundtrip '+row.name+'/'+k,a.every((v,i)=>v===t.state[k][i]));count++}}
 check('legacy T1 temporal attributes not silently relabelled',tiles[0].state.temporal[(mappingByName.get('B3_A0_C0_T1.png').full_y+60)*2541+mappingByName.get('B3_A0_C0_T1.png').full_x+80]===3);
 const beforeRestore=await Promise.all(tiles.map(async t=>await Promise.all(keys.map(k=>sha(t.state[k])))));await importProject(JSON.stringify(saved),'SOFTWARE_FULL_RESTORE.json');check('full project restore preserves all source masks and attributes',JSON.stringify(beforeRestore)===JSON.stringify(await Promise.all(tiles.map(async t=>await Promise.all(keys.map(k=>sha(t.state[k])))))));
 const originalMasks=await Promise.all(tiles.map(t=>sha(t.state.mask)));const oldScale=camera.s;$('expand').click();const side=$('drawingTools').getBoundingClientRect(),canvas=stage.getBoundingClientRect();check('expanded image keeps tools next to canvas',$('workspace').classList.contains('expanded')&&side.right<=canvas.left+4&&side.top>=0&&side.bottom<=innerHeight&&camera.s>oldScale);
 check('zoom does not alter annotation masks',JSON.stringify(originalMasks)===JSON.stringify(await Promise.all(tiles.map(t=>sha(t.state.mask)))));
 const scale=camera.s,x=camera.x,y=camera.y;selectPeriod(2);check('period change keeps local view and zoom',camera.s===scale&&camera.x===x&&camera.y===y);
 $('exitExpanded').click();check('normal view restored',!$('workspace').classList.contains('expanded'));
 function paint(x,y){const t=tiles[selected],r=stage.getBoundingClientRect();stage.setPointerCapture=()=>{};stage.onpointerdown({clientX:r.left+camera.x+(t.x+x+.5)*camera.s,clientY:r.top+camera.y+(t.y+y+.5)*camera.s,button:0,pointerId:1,preventDefault(){}});endStroke()}
 $('brush').value='1';$('root').click();$('cohort').value='1';$('temporal').value='3';const t2=tiles[1],index2=800*2541+1200,oldValues=keys.map(k=>t2.state[k][index2]);paint(1200,800);
 check('later-root brush retains chosen origin and persistence',t2.state.mask[index2]===255&&t2.state.cohort[index2]===1&&t2.state.temporal[index2]===3);
 stage.dispatchEvent(new KeyboardEvent('keydown',{key:'z',ctrlKey:true,bubbles:true,cancelable:true}));check('Ctrl+Z restores every label layer',keys.every((k,i)=>t2.state[k][index2]===oldValues[i]));
 selectPeriod(1);$('temporal').value='3';$('cohort').value='4';paint(1200,800);check('newly painted T1 roots are forced to baseline',tiles[0].state.cohort[index2]===1&&tiles[0].state.temporal[index2]===0);undo();
 const separatorBefore=await sha(tiles[0].state.mask);paint(1200,1118);check('separator cannot receive root strokes',separatorBefore===await sha(tiles[0].state.mask));
 check('no full human review created by migration',tiles.every(t=>!t.state.reviewed&&!t.state.annotator));
 await saveDraft();const draft=await readDraft();check('automatic backup writes matching arrays',tiles.every(t=>keys.every(k=>t.state[k].every((v,i)=>v===draft[t.name].state[k][i]))));
 const draftHashes=await Promise.all(tiles.map(async t=>await Promise.all(keys.map(k=>sha(t.state[k])))));for(const t of tiles)t.state=fresh(t);await $('restoreDraft').onclick();check('restore-draft action recovers every image and attribute',JSON.stringify(draftHashes)===JSON.stringify(await Promise.all(tiles.map(async t=>await Promise.all(keys.map(k=>sha(t.state[k])))))));
 const oldClick=HTMLAnchorElement.prototype.click;HTMLAnchorElement.prototype.click=function(){};$('save').click();const blobText=await(await fetch($('download').href)).text();HTMLAnchorElement.prototype.click=oldClick;const project=JSON.parse(blobText);check('JSON download retains four full images and original eight-image backup',project.sources.length===4&&project.archives.some(a=>a.original_text===window.fixtureBackup));
 const rgba=[];for(const t of tiles){const c=document.createElement('canvas');c.width=t.width;c.height=t.height;const cx=c.getContext('2d');cx.drawImage(t.image,0,0);rgba.push({name:t.name,sha256:await sha(cx.getImageData(0,0,t.width,t.height).data)})}
 return {checks,rgba_hashes:rgba,software_only:true,actual_user_annotations_received:false,fullProject:blobText};
})()`);
fs.writeFileSync('/tmp/B3_full_project_software_fixture.json',report.fullProject);delete report.fullProject;
report.old_page_recovery={images:8,original_four_arrays_unchanged:true,software_fixture_only:true};
fs.writeFileSync(output+'/browser_validation.json',JSON.stringify(report,null,2)+'\n');
await app.evaluate("$('expand').click()");const shot=await app.call('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});fs.writeFileSync('/tmp/B3_full_editor_toolbar.png',Buffer.from(shot.data,'base64'));
console.log(JSON.stringify({passed:report.checks.length,failed:report.checks.filter(c=>!c.pass).length,old_page_states_preserved:8}));
old.ws.close();app.ws.close();
