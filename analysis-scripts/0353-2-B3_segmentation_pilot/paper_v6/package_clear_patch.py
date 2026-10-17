#!/usr/bin/env python3
"""Lossless screened-data export and CRC/SHA-verified trial delivery."""
import argparse,csv,json,random,shutil,zipfile
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import protocol as p
from clear_patch_trial import ScreenedSampler,read,csvout

def copy(source,dest):dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)

def dataset(data,run):
 out=run/'data_package';out.mkdir(exist_ok=True)
 for f in (data.source/'dataset/upper').glob('*.png'):copy(f,out/'dataset/upper'/f.name)
 copy(data.source/'original_human_progress.json',out/'original_human_progress.json');copy(data.source/'dataset/provenance.json',out/'source_v5_provenance.json')
 with (run/'final_training_selection.csv').open(encoding='utf-8-sig') as f:rootrows=list(csv.DictReader(f))
 sampler=ScreenedSampler(data,run);selection=[dict(r,partition='train') for r in sampler.pool+sampler.bg]
 for split,col in [('val',3),('test',4)]:
  for t in p.PERIODS:
   for i,(x,y) in enumerate(p.XY):selection.append(dict(id=f'{t}_C{col}_W{i:02d}',period=t,column=col,x=p.COLS[col]+x,y=y,side=384,partition=split))
 manifest=[];checks=0
 for r in selection:
  x,y,side,col=(int(r[k]) for k in ['x','y','side','column']);t=r['period'];split=r['partition'];entry={k:r[k] for k in ['id','period','partition']};entry.update(column=col,x=x,y=y,side=side,spatial_group=f'C{col}_X{x}_Y{y}_S{side}')
  assert (col<3)==(split=='train') and (col==3)==(split=='val') and (col==4)==(split=='test')
  for layer in ['RGB','mask','color','temporal','viability']:
   source=np.array(Image.open(data.source/'dataset/upper'/(t+'_'+layer+'.png')));patch=source[y:y+side,x:x+side]
   path=out/'dataset/patches'/split/(r['id']+'_'+layer+'.png');path.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(patch).save(path)
   assert np.array_equal(np.array(Image.open(path)),patch);checks+=1;entry[layer]=str(path.relative_to(out))
  manifest.append(entry)
 csvout(out/'dataset/manifest.csv',manifest)
 files={str(f.relative_to(out)):p.sha(f) for f in sorted((out/'dataset/upper').glob('*.png'))};files['dataset/manifest.csv']=p.sha(out/'dataset/manifest.csv')
 p.js(out/'dataset/provenance.json',dict(source_progress_sha256=p.sha(out/'original_human_progress.json'),files=files))
 # Verify sampling cannot read evaluation pixels, even if they are overwritten in memory.
 random.seed(p.SEED);before=[sampler.sample(i) for i in range(32)]
 for t in p.PERIODS:
  data.rgb[t][:,1558:]=0;data.gt[t][:,1558:]=255
 random.seed(p.SEED);after=[sampler.sample(i) for i in range(32)]
 assert all(np.array_equal(a,b) for s1,s2 in zip(before,after) for a,b in zip(s1,s2))
 assert sum(s[0].shape[0]==128 for s in before)>0
 p.js(out/'dataset/verification.json',dict(passed=True,lossless_layer_crop_checks=checks,training_samples_independent_of_validation_and_C4_pixels=True,
  original_source_progress_SHA256=p.sha(out/'original_human_progress.json'),native_sizes=[384,128],ground_truth_edited=False))
 p.js(out/'dataset/summary.json',dict(source_RGB_images=4,positions=1,train_root_patch_images=len(sampler.pool),train_background_patch_images=len(sampler.bg),
  validation_patch_images=40,repeated_benchmark_patch_images=40,unique_selected_root_window_geometries=len({r['spatial_group'] for r in manifest if r['partition']=='train' and r['side']==384}),
  source_and_full_temporal_layers_preserved=True,death_not_trained=True,external_independent_test_images=0,
  grouping='Same spatial columns across all4 periods; no period crosses split. Selected training windows can differ by period. Binary training only.',
  overlap='Exported windows overlap; not independent biological samples; official evaluation fuses each complete column and counts pixels once.'))
 # Updated RGB sheets display the actual final70 inclusion decisions.
 original=p.Data(data.source);review=run/'review_final';review.mkdir(exist_ok=True);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
 for t in p.PERIODS:
  rr=[r for r in rootrows if r['period']==t];canvas=Image.new('RGB',(6*384,5*414),'white');draw=ImageDraw.Draw(canvas)
  for i,r in enumerate(rr):
   x,y=int(r['x']),int(r['y']);left=i%6*384;top=i//6*414;canvas.paste(Image.fromarray(original.rgb[t][y:y+384,x:x+384]),(left,top+30))
   keep=r['final_selected']=='True';draw.text((left+3,top+5),r['id']+(' KEEP' if keep else ' OMIT'),font=font,fill='#006E65' if keep else '#A44820')
  canvas.save(review/(t+'_final_training_review.jpg'),quality=96)
 for name in ['training_visibility.csv','final_training_selection.csv','background_candidates.csv','visual_review.json','screening_protocol.json','evaluation_protocol.json']:copy(run/name,out/'screening'/name)
 print('CLEAR_DATA_VERIFIED',len(manifest),checks,flush=True)

def zipout(path,entries):
 with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for source,name in entries:
   if source.is_dir():
    for f in sorted(source.rglob('*')):
     if f.is_file() and '__pycache__' not in f.parts:z.write(f,str(name/f.relative_to(source)))
   else:z.write(source,str(name))
 with zipfile.ZipFile(path) as z:assert z.testzip() is None
 assert path.stat().st_size<100*1024**2
 return dict(file=path.name,bytes=path.stat().st_size,SHA256=p.sha(path),CRC_verified=True)

def main():
 a=argparse.ArgumentParser();a.add_argument('--data',type=Path,required=True);a.add_argument('--run',type=Path,required=True);a.add_argument('--prepare-data',action='store_true');a.add_argument('--output',type=Path);args=a.parse_args()
 if args.prepare_data:dataset(p.Data(args.data),args.run);return
 run=args.run;out=args.output;out.mkdir(parents=True,exist_ok=True);src=Path(__file__).resolve().parent
 reports=['Results_CN.md','metrics.csv','all_metrics.json','root_length_agreement.json','root_length_pairs.csv','frozen_selection.json','verification.json','stage_gate.json','screening_protocol.json','visual_review.json','evaluation_protocol.json','final_training_selection.csv','training_visibility.csv','background_candidates.csv']
 base=Path('0353-2-B3_clear_patch_v7');common=[(run/f,base/'evaluation'/f) for f in reports]
 source=[(src/f,base/'source/paper_v6'/f) for f in ['clear_patch_trial.py','evaluate_clear_patch.py','package_clear_patch.py','train.py','protocol.py','figures.py','requirements.txt']]
 source += [(src.parent/'root_background_v5'/f,base/'source/root_background_v5'/f) for f in ['evaluate.py','common.py']]
 model_entries=common+source
 for arm in ['control','screened']:
  for f in ['best.pt','training_run.json','history.csv','initial_validation.json','validation_selection.json','thresholds_identity.csv','thresholds_tta.csv']:model_entries.append((run/arm/f,base/arm/f))
 result_entries=common+[(run/'figures',base/'figures'),(run/'images',base/'images'),(run/'review_final',base/'training_review')]
 score_entries=common+[(run/'evaluation',base/'evaluation')]
 data_entries=[(run/'data_package',base),(run/'review_final',base/'training_review')]+source
 manifest=[]
 for name,entries in [('results',result_entries),('models',model_entries),('data',data_entries),('scores',score_entries)]:
  manifest.append(zipout(out/('0353-2-B3_clear_patch_v7_'+name+'.zip'),entries))
 p.js(out/'SHA256.json',dict(archives=manifest,ground_truth_unchanged=True,original_progress_SHA256=p.sha(args.data/'original_human_progress.json')))
 text=(run/'Results_CN.md').read_text()+'\n## 下载 ZIP\n\n'+''.join('- ['+label+'](0353-2-B3_clear_patch_v7_'+name+'.zip)\n' for name,label in [('results','筛选对照结果、论文图及四期分割'),('models','两组验证择优权重、训练记录及脚本'),('data','筛选后的原像素图块与完整原标注'),('scores','三组原始概率数组与评价掩膜')])
 text+='\n复现训练时的起始权重来自已发布的 [v6模型包](https://raw.githubusercontent.com/Annalee811/LI/aa84f9a64f48d252b2e41e770915bfa64f440e55/roma-results/0353-2/0353-2-B3/segmentation-pilot-annotation/paper-refinement-v6/0353-2-B3_refined_models_v6.zip) 中的 unet_pretrained/best.pt。本包保存两组验证择优权重；若 best_epoch 为0，其参数与起始模型相同，不代表新增训练带来提升。若要重复微调，需要原始起始权重。用原v6依赖环境，把四个新 ZIP 解压合并到同一个目录；将该目录作为 DATA 和 RUN，将完整 v6 模型目录作为 PREVIOUS。从 screening/复制筛选清单至 RUN，原始标签保持不变。\n'
 text+='\n```bash\npython source/paper_v6/clear_patch_trial.py --data "$DATA" --run "$RUN" --initial "$PREVIOUS/unet_pretrained/best.pt" --arm control --epochs 20 --threads 3\npython source/paper_v6/clear_patch_trial.py --data "$DATA" --run "$RUN" --initial "$PREVIOUS/unet_pretrained/best.pt" --arm screened --epochs 20 --threads 3\npython source/paper_v6/evaluate_clear_patch.py --data "$DATA" --run "$RUN" --previous "$PREVIOUS" --threads 3\n```\n'
 (out/'README.md').write_text(text);copy(run/'metrics.csv',out/'metrics.csv');copy(run/'verification.json',out/'verification.json')
 for m in manifest:print('CLEAR_TRIAL_ARCHIVE_VERIFIED',m,flush=True)
if __name__=='__main__':main()
