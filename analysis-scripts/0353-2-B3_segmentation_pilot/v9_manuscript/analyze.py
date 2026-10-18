"""Recompute native metrics and lengths, with explicit non-interchangeable means."""
import argparse,collections
from pathlib import Path
import numpy as np
from PIL import Image
import common as c

def ap(score,gt):
 order=np.argsort(-score,kind='stable');s=score[order];g=gt[order].astype(np.int64);tp=g.cumsum();last=np.r_[np.flatnonzero(s[1:]!=s[:-1]),len(s)-1];tp=tp[last];fp=last+1-tp;recall=tp/max(1,int(gt.sum()));precision=tp/(tp+fp)
 return float((np.diff(np.r_[0.,recall])*precision).sum()),recall,precision

def main():
 p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=True);inp=a.inputs;source=c.read(inp/'records/all_metrics.json');oldlength=c.csvread(inp/'records/root_length_pairs.csv');aggregate=[];per=[];lens=[];lengthstats=[];density=[];prstats=[];truth={};checks=[]
 for region,col in [('C3',3),('C4',4)]:
  for t in c.PERIODS:
   gt=np.array(Image.open(inp/'upper'/f'{t}_mask.png'))[:,c.COLS[col]:c.COLS[col]+448];truth[region,t]=gt;density.append(dict(region=region,period=t,valid_pixels=int((gt!=128).sum()),human_root_pixels=int((gt==255).sum()),root_prevalence=float((gt==255).sum()/(gt!=128).sum())))
 for n in c.MODELS:
  byregion={}
  for region,split in [('C3','val'),('C4','test')]:
   cm=np.zeros(4,np.int64);periods=[];scores=[];gts=[]
   for t in c.PERIODS:
    gt=truth[region,t];pred=np.array(Image.open(inp/'predictions'/n/region/f'{t}_mask.png'))>0;cc=c.counts(gt,pred);cm+=cc;row=dict(model=n,region=region,period=t,**c.metrics(cc));per.append(row);periods.append(row)
    for metric,value in c.metrics(cc).items():assert np.isclose(value,source[n][split]['after']['per_period'][t][metric],atol=1e-12,rtol=0)
    lf=c.length_field(gt==255);lp=c.length_field(pred)
    for yi in range(4):
     for xi in range(2):
      sl=(slice(yi*277,(yi+1)*277),slice(xi*224,(xi+1)*224));r=dict(model=n,region=region,period=t,block=f'X{xi}Y{yi}',human_length_px=float(lf[sl].sum()),predicted_length_px=float(lp[sl].sum()));ref=next(q for q in oldlength if q['model']==n and q['partition']==split and q['mode']=='after' and q['period']==t and q['block']==r['block'])
      assert np.isclose(r['human_length_px'],float(ref['human_length_px']),atol=1e-9) and np.isclose(r['predicted_length_px'],float(ref['predicted_length_px']),atol=1e-9);lens.append(r)
    score=np.load(inp/'predictions'/n/region/f'{t}_probability.npy');valid=gt!=128;scores.append(score[valid]);gts.append((gt==255)[valid])
   metrics=c.metrics(cm);byregion[region]=(cm,periods)
   for key,value in metrics.items():assert np.isclose(value,source[n][split]['after']['root'][key],atol=1e-12,rtol=0)
   aggregate.append(dict(model=n,aggregation=region+'_pooled4dates',recommended_use='main_same_position_benchmark' if region=='C4' else 'validation_selected_supplement',**metrics))
   for statistic in ['mean','SD']:
    vals={key:float(np.mean([r[key] for r in periods]) if statistic=='mean' else np.std([r[key] for r in periods],ddof=1)) for key in c.METRICS};aggregate.append(dict(model=n,aggregation=region+'_period_macro_'+statistic,recommended_use='descriptive4correlated_dates_not_CI',TP=None,FP=None,FN=None,TN=None,**vals))
   subset=[r for r in lens if r['model']==n and r['region']==region];ls=c.agreement([r['human_length_px'] for r in subset],[r['predicted_length_px'] for r in subset]);lengthstats.append(dict(model=n,region=region,**ls))
   src=c.read(inp/'records/root_length_agreement.json')[n][split]['after'];assert np.isclose(ls['R2_agreement'],src['R2_agreement'],atol=1e-12) and np.isclose(ls['RMSE_px'],src['RMSE_px'],atol=1e-12)
   score=np.concatenate(scores);gt=np.concatenate(gts);average_precision,recall,precision=ap(score,gt);prstats.append(dict(model=n,region=region,Average_precision_exact=average_precision,prevalence=float(gt.mean()),valid_pixels=len(gt),score_scope='raw probability of selected network; final connector mask is a separate operating point'))
   # Downsample exact PR only for plotting, preserve all slope changes for AP.
   ix=np.unique(np.r_[0,np.linspace(0,len(recall)-1,3000,dtype=int),len(recall)-1]);np.savez_compressed(out/f'{n}_{region}_PR.npz',recall=recall[ix],precision=precision[ix]);checks.append(dict(model=n,region=region,pooled_counts_and_all_period_metrics_exact=True,length_block_pairs_and_R2_RMSE_exact=True))
  combined=c.metrics(byregion['C3'][0]+byregion['C4'][0]);aggregate.append(dict(model=n,aggregation='C3_C4_count_pooled8views',recommended_use='exploratory_mixed_validation_and_benchmark_not_primary',**combined))
  m={key:float(np.mean([c.metrics(byregion[r][0])[key] for r in ['C3','C4']])) for key in c.METRICS};aggregate.append(dict(model=n,aggregation='C3_C4_arithmetic_region_mean',recommended_use='exploratory_only_not_primary',TP=None,FP=None,FN=None,TN=None,**m))
  m={key:float(np.mean([r[key] for region in ['C3','C4'] for r in byregion[region][1]])) for key in c.METRICS};aggregate.append(dict(model=n,aggregation='C3_C4_macro8dateviews',recommended_use='exploratory8correlated_views_not_primary',TP=None,FP=None,FN=None,TN=None,**m))
  subset=[r for r in lens if r['model']==n];lengthstats.append(dict(model=n,region='C3_C4_exploratory',**c.agreement([r['human_length_px'] for r in subset],[r['predicted_length_px'] for r in subset])))
 for f,rows in [('aggregate_metrics.csv',aggregate),('per_period_metrics.csv',per),('root_length_pairs.csv',lens),('root_length_summary.csv',lengthstats),('root_prevalence.csv',density),('precision_recall_summary.csv',prstats)]:c.csvout(out/f,rows)
 for region in ['C3','C4']:c.csvout(out/f'Table_{region}_strict_pooled4dates.csv',[r for r in aggregate if r['aggregation']==region+'_pooled4dates'])
 c.js(out/'summary.json',dict(aggregate=aggregate,length_agreement=lengthstats,root_prevalence=density,precision_recall=prstats))
 manifest=c.csvread(inp/'records/training_patch_manifest.csv');train=[r for r in manifest if r['partition']=='train'];counts=collections.Counter((r['partition'],r['side']) for r in manifest)
 c.js(out/'study_design.json',dict(original_RGB_images=4,original_annotated_location=1,train_date_region_views=12,validation_C3_date_region_views=4,repeated_benchmark_C4_date_region_views=4,external_independent_test_RGB_images=0,training_draws_per_model=len(train),unique_training_date_xy_side_windows=len({(r['period'],r['x'],r['y'],r['side']) for r in train}),patch_manifest_counts=[dict(partition=k[0],side=int(k[1]),count=v) for k,v in counts.items()],native_upper_size=[2541,1108],columns=[dict(name=f'C{i}',left=x,right_exclusive=x+448,width=448,role='train' if i<3 else 'validation_selection' if i==3 else 'repeated_same_position_benchmark') for i,x in enumerate(c.COLS)],guard_width=64,grouping='All dates of a spatial column share the same group; no random splitting of temporal neighbours or overlapping crops',primary='C4 four-period pixel-count pooled descriptive benchmark; C3 separately reported; not independent external validation',mean_policy='Never use mixed C3+C4 mean as the primary reported performance',uncertainty='No p-values or biological confidence intervals; dates,spatial blocks,pixels,crops are correlated'))
 for f,digest in c.read(inp/'integrity.json')['files'].items():assert c.sha(inp/f)==digest
 c.js(out/'verification.json',dict(passed=True,original_inputs_unchanged=True,original_v9_archive_metrics_reproduced=True,all24_native_masks_evaluated=True,all192_block_date_model_lengths_reproduced=True,checks=checks,original_annotation_SHA256=c.sha(inp/'original_human_progress.json'),new_training=False,temporal_v13_not_used=True))
 print('V9_REANALYSIS_COMPLETE',flush=True)
if __name__=='__main__':main()
