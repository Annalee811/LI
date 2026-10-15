/* Run ONLY in the already-open 0353-2-B3 annotate.html console.
   No network requests, refreshes, deletion, relabelling, or changes to RGB pixels.
   Adds a download panel and snapshots every loaded annotation state, including current strokes. */
(async()=>{
 'use strict';
 if(typeof states==='undefined'||!(states instanceof Map)||typeof active==='undefined'||typeof files==='undefined')throw Error('请在仍显示已标注结果的旧 annotate.html 页面中运行，不是在新标签页运行。');
 if(typeof remember==='function')remember();
 const snapshot=new Map(states);if(active&&typeof index==='number'&&files[index])snapshot.set(files[index].name,active);
 const length=typeof N==='number'?N:448*448;
 const runs=a=>{const out=[];let value=a[0],count=1;for(let i=1;i<a.length;i++){if(a[i]===value)count++;else{out.push(value,count);value=a[i];count=1}}out.push(value,count);return out};
 const hash=async bytes=>[...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))].map(v=>v.toString(16).padStart(2,'0')).join('');
 const records=[];
 for(const [name,s] of snapshot){
  if(!(s.mask instanceof Uint8Array)||s.mask.length!==length)throw Error('标注数组尺寸异常：'+name+'；未改动原标注。');
  const f=[...files].find(x=>x.name===name);let digest=null,hashBasis='unavailable_annotations_still_preserved';
  if(f)try{digest=await hash(await f.arrayBuffer());hashBasis='fresh_file_sha256'}catch(e){/* A removed temporary image must not prevent recovering its annotations. */}
  if(!digest&&typeof expectedImages!=='undefined'&&expectedImages[name]){digest=expectedImages[name];hashBasis='legacy_expected_image_hash'}
  const arrays={};for(const k of ['mask','color','temporal','viability','cohort','scene','instance']){
   if(s[k]){if(s[k].length!==length)throw Error('属性数组尺寸异常：'+name+'/'+k);arrays[k]=s[k].slice()}
   else arrays[k]=new Uint8Array(length);
  }
  if(!s.scene)for(let i=0;i<length;i++)arrays.scene[i]=s.mask[i]===255?1:s.mask[i]===128?5:s.mask[i]===0?4:0;
  if(!s.cohort&&/_T1\.png$/i.test(name))for(let i=0;i<length;i++)if(s.mask[i]===255)arrays.cohort[i]=1;
  records.push({name,image_sha256:digest,image_hash_basis:hashBasis,...Object.fromEntries(Object.entries(arrays).map(([k,a])=>[k,runs(a)])),
   reviewed:s.reviewed===true,evidence:s.evidence||'',annotator:s.annotator||'',prefill_origin:s.prefill_origin||'existing_local_editor_state',
   review_status:s.review_status||'preserved_without_reclassification',was_modified:s.modified===true});
 }
 if(!records.length)throw Error('旧页面没有可备份的标注状态；请不要刷新。');
 const data={kind:'B3_reviewable_annotations',version:2,encoding:'rle_value_count',not_training_ground_truth:true,
  saved_at:new Date().toISOString(),recovery:{from:'already_open_local_editor',all_original_mask_values_preserved:true,all_original_attributes_preserved:true,new_review_status_not_created:true},records};
 const text=JSON.stringify(data);window.B3_PRESERVED_PROJECT=data;window.B3_PRESERVED_PROJECT_JSON=text;
 const previous=document.getElementById('b3-preservation-panel');if(previous)previous.remove();
 const panel=document.createElement('section');panel.id='b3-preservation-panel';panel.style.cssText='position:fixed;right:12px;top:12px;z-index:2147483647;width:350px;max-height:85vh;overflow:auto;background:#fff8df;color:#202d28;border:3px solid #376b50;padding:18px;font:15px/1.6 sans-serif;box-shadow:0 3px 20px #5558';
 const title=document.createElement('strong');title.textContent='已整理 '+records.length+' 张页面标注，等待你保存';panel.append(title);
 const note=document.createElement('p');note.textContent='原页面和标注没有被清除或重载。点击下面保存一个JSON；确认下载后，把它导入完整RGB大画布，或上传回聊天。未确认文件保存成功前，不要关闭旧页面。';panel.append(note);
 const link=document.createElement('a');link.textContent='点击保存全部已有标注 JSON';link.download='0353-2-B3_existing_annotations_backup_'+Date.now()+'.json';link.href=URL.createObjectURL(new Blob([text],{type:'application/json'}));link.style.cssText='display:block;padding:12px;background:#376b50;color:white;border-radius:6px;text-align:center';panel.append(link);
 const detail=document.createElement('details'),summary=document.createElement('summary');summary.textContent='下载失败时：显示备份文本';detail.append(summary);
 const area=document.createElement('textarea');area.value=text;area.readOnly=true;area.style.cssText='width:100%;height:180px';detail.append(area);panel.append(detail);
 const close=document.createElement('button');close.textContent='收起此面板（不清除标注）';close.onclick=()=>panel.remove();panel.append(close);document.body.append(panel);
 console.log('B3_BACKUP_READY',{images:records.length,bytes:new Blob([text]).size,download:'请点击页面右上角新出现的绿色保存按钮'});
 return {images:records.length,bytes:new Blob([text]).size};
})()
