"""Publication-size scientific figure from archived results only; no new fitting.

Final size 190 x 145 mm. All raster data and vector marks are derived from fixed
CSV/JSON results. Error bars are optimization-seed sample SD, not confidence intervals.
"""
from pathlib import Path
import argparse,csv,json,hashlib,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

HERE=Path(__file__).resolve().parent
def read_csv(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--results',type=Path,default=Path('/workspace/work/V90_validation_20261123/reports'))
 ap.add_argument('--fonts',type=Path,default=Path('/workspace/work/root_workflow_real_assets_20261109/fonts'))
 ap.add_argument('--output',type=Path,default=HERE)
 a=ap.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=True)
 for n in ['Times New Roman.ttf','Times New Roman Bold.ttf']:font_manager.fontManager.addfont(str(a.fonts/n))
 plt.rcParams.update({'font.family':'Times New Roman','font.size':8,'axes.titlesize':8.8,'axes.titleweight':'bold','axes.labelsize':8,
  'axes.labelweight':'bold','text.color':'black','axes.labelcolor':'black','xtick.color':'black','ytick.color':'black',
  'axes.edgecolor':'black','axes.linewidth':.65,'xtick.major.width':.6,'ytick.major.width':.6,
  'xtick.major.size':2.5,'ytick.major.size':2.5,'xtick.labelsize':7.3,'ytick.labelsize':7.3,
  'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
  'savefig.facecolor':'white','figure.facecolor':'white','hatch.linewidth':.4})
 runs=read_csv(a.results/'Internal_results_per_run.csv');classes=read_csv(a.results/'Internal_results_per_class.csv')
 dec=read_csv(a.results/'Paired_decoder_stage_results.csv');frozen=json.loads((a.results/'frozen_references.json').read_text())['V66_pairwise_frozen']['metrics_by_split']['2']
 seeds=sorted(set(int(r['seed'])for r in runs));assert len(seeds)==3
 order=['V66','current_only','no_lstm','no_gat','no_gates','full']
 names=['Frozen V66','Frame-only','w/o LSTM','w/o GAT','w/o gates','Full']
 stages=['raw_head','axis_constraints','axis_and_junction_constraints'];stage_labels=['Raw\nhead','Axis','Axis +\njunction']
 color={'full':'#84445A','ablated':'#557C99','ref':'#777F86','persistent':'#326B94','new':'#CC9842','extension':'#A44879'}
 records=[]
 def vals(v,k):return np.array([float(next(r[k]for r in runs if r['variant']==v and int(r['seed'])==s))*100 for s in seeds])
 def dv(stage,k):return np.array([float(next(r[k]for r in dec if r['stage']==stage and int(r['seed'])==s))*100 for s in seeds])
 def stats(v):return dict(mean=float(v.mean()),SD=float(v.std(ddof=1))if len(v)>1 else None,n_seeds=len(v))
 fig=plt.figure(figsize=(190/25.4,145/25.4))
 positions=[(.135,.605,.195,.275),(.445,.605,.210,.275),(.792,.605,.190,.275),(.100,.120,.230,.280),(.435,.120,.230,.280),(.770,.120,.212,.280)]
 axes=[fig.add_axes(p)for p in positions]
 for ax in axes:ax.tick_params(pad=3)
 fig.text(.02,.970,'I   Controlled neural modules',fontsize=9.1,fontweight='bold',va='center')
 fig.text(.982,.970,'Partition-isolated context',fontsize=7.8,ha='right',va='center')
 fig.text(.02,.480,'II   Paired structural decoding',fontsize=9.1,fontweight='bold',va='center')
 fig.text(.982,.480,'Same whole-field neural output within each seed',fontsize=7.8,ha='right',va='center')
 for ax,letter,x,y in zip(axes,'ABCDEF',[.02,.36,.70,.02,.36,.70],[.916,.916,.916,.434,.434,.434]):
  fig.text(x,y,letter,fontweight='bold',fontsize=12,va='center')
 # A: same masked reference-pixel comparison; fixed historical V66 is one reference.
 ax=axes[0]
 for i,v in enumerate(order):
  x=np.array([100*frozen['macro_Dice']])if v=='V66'else vals(v,'test_macro_Dice')
  c=color['full']if v=='full'else color['ref']if v in ['V66','current_only']else color['ablated']
  if len(x)>1:
   ax.errorbar(x.mean(),i,xerr=x.std(ddof=1),fmt='none',ecolor=c,elinewidth=1.2,capsize=2.3,zorder=3)
   ax.scatter(x,i+np.array([-.20,0,.20]),s=9,color='black',alpha=.72,zorder=4)
  ax.scatter(x.mean(),i,s=35,marker='D'if v=='V66'else's'if v=='full'else'o',facecolor='white'if v=='V66'else c,edgecolor=c,lw=.8,zorder=5)
  ax.text(x.mean()+3.5,i,f'{x.mean():.2f}',va='center',fontsize=7.5,fontweight='bold'if v=='full'else'normal')
  records.append(dict(panel='A',method=v,metric='macro_Dice_percent',**stats(x)))
 ax.set_yticks(range(6),names);ax.set_ylim(5.6,-.6);ax.set_xlim(0,85);ax.set_xticks([0,25,50,75]);ax.set_xlabel('Macro Dice (%)',labelpad=4)
 ax.set_title('Internal performance',pad=7);ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0,pad=5)
 # B: class means; all text remains black against a light quantitative palette.
 ax=axes[1];mat=[]
 for v in order:
  rr=[]
  for k in ['persistent','new','extension']:
   x=np.array([float(next(c['Dice']for c in frozen['classes']if c['name']==k))*100])if v=='V66'else np.array([float(r['Dice'])*100 for r in classes if r['variant']==v and r['class_name']==k])
   rr.append(x.mean());records.append(dict(panel='B',method=v,metric=k+'_Dice_percent',**stats(x)))
  mat.append(rr)
 mat=np.array(mat);cmap=LinearSegmentedColormap.from_list('pale_blue',['#FFFFFF','#DDE7EE','#94B5CC'])
 ax.imshow(mat,vmin=0,vmax=100,cmap=cmap,aspect='auto',interpolation='nearest')
 for i in range(6):
  for j in range(3):ax.text(j,i,f'{mat[i,j]:.1f}',ha='center',va='center',fontsize=7.7,fontweight='bold'if i==5 else'normal')
 ax.set_yticks(range(6),names);ax.tick_params(axis='both',length=0,pad=4);ax.set_xticks(range(3),['Persistent','New','Extending']);ax.set_title('Class Dice (%)',pad=7)
 for spine in ax.spines.values():spine.set_visible(False)
 # C: paired seed differences; standard deviations are not CIs or biological replicates.
 ax=axes[2];effects={}
 for i,(v,label) in enumerate([('no_lstm','+ ConvLSTM'),('no_gat','+ GAT'),('no_gates','+ ID gates')]):
  x=vals('full','test_macro_Dice')-vals(v,'test_macro_Dice');effects[v]=dict(paired_seed_differences_pp=x.tolist(),**stats(x))
  ax.errorbar(x.mean(),i,xerr=x.std(ddof=1),fmt='s',color=color['full'],markersize=4.6,capsize=3,lw=1.2,zorder=4)
  ax.scatter(x,i+np.array([-.10,0,.10]),s=11,c='black',alpha=.65,zorder=5)
  ax.text(x.mean(),i-.30,f'{x.mean():+.2f}',ha='center',fontsize=7.8,fontweight='bold')
  records.append(dict(panel='C',method=label,metric='full_minus_ablated_macro_Dice_pp',**stats(x)))
 ax.axvline(0,color='#888888',lw=.65,ls=(0,(3,3)));ax.set_yticks(range(3),['+ ConvLSTM','+ GAT','+ ID gates']);ax.set_ylim(2.6,-.65);ax.set_xlim(-.5,1.45)
 ax.set_xticks([-.5,0,.5,1]);ax.set_xlabel('Change in Dice (pp)',labelpad=4);ax.set_title('Module contribution',pad=7);ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0,pad=5)
 # D: a separate whole-field context, paired within each seed.
 ax=axes[3];macro=np.array([dv(s,'macro_Dice')for s in stages]);means=macro.mean(1);sds=macro.std(1,ddof=1)
 for j in range(3):ax.plot(range(3),macro[:,j],color='#ADB3B8',lw=.75,marker='o',ms=2.7,alpha=.85)
 ax.errorbar(range(3),means,yerr=sds,color=color['full'],fmt='s-',lw=1.65,ms=5,capsize=3,zorder=4)
 for i,v in enumerate(means):
  ax.text(i,v+sds[i]+3,f'{v:.2f}',ha='center',fontsize=8,fontweight='bold');records.append(dict(panel='D',method=stages[i],metric='macro_Dice_percent',**stats(macro[i])))
 ax.set(ylim=(0,100),xlim=(-.2,2.2));ax.set_xticks(range(3),stage_labels);ax.set_ylabel('Macro Dice (%)',labelpad=3);ax.set_title('Paired decoder stages',pad=7)
 # E: rare extending roots drive the macro improvement.
 ax=axes[4];decoder_class={}
 for k,label,marker in [('persistent','Persistent','o'),('new','New','s'),('extension','Extending','^')]:
  x=np.array([dv(s,k+'_Dice')for s in stages]);decoder_class[k]={s:stats(x[i])for i,s in enumerate(stages)}
  ax.errorbar(range(3),x.mean(1),yerr=x.std(1,ddof=1),fmt=marker+'-',ms=4,color=color[k],lw=1.3,capsize=2,label=label)
  for i,s in enumerate(stages):records.append(dict(panel='E',method=s,metric=k+'_Dice_percent',**stats(x[i])))
  # Endpoints are directly labelled; labels in the legend retain the class mapping.
  ax.text(2.15,x[-1].mean(),f'{x[-1].mean():.1f}',va='center',fontsize=7.4)
 ax.set(ylim=(0,105),xlim=(-.2,2.68));ax.set_xticks(range(3),stage_labels);ax.set_ylabel('Class Dice (%)',labelpad=3);ax.set_title('Class-specific changes',pad=7)
 ax.legend(frameon=False,fontsize=7,loc='center left',bbox_to_anchor=(.01,.50),handlelength=1.4,labelspacing=.3)
 # F: pairwise instance metric, not the graph edge identity-link F1.
 ax=axes[5];instance={}
 for j,(s,marker,c,label)in enumerate([('axis_constraints','o','#71869A','Axis'),('axis_and_junction_constraints','s',color['full'],'Axis + junction')]):
  m=[];sd=[];instance[s]={}
  for i,key in enumerate(['pairwise_instance_precision','pairwise_instance_recall','pairwise_instance_F1']):
   x=dv(s,key);m.append(x.mean());sd.append(x.std(ddof=1));instance[s][key]=stats(x);records.append(dict(panel='F',method=s,metric=key+'_percent',**stats(x)))
  xx=np.arange(3)+(j-.5)*.23;ax.errorbar(xx,m,yerr=sd,fmt=marker,ms=4.3,mfc='white'if j==0 else c,mec=c,ecolor=c,capsize=2,label=label,lw=1,zorder=4)
  if j==1:
   for xx0,v,e in zip(np.arange(3),m,sd):ax.text(xx0,v+e+3,f'{v:.2f}',ha='center',fontsize=7.3,fontweight='bold')
 ax.set(ylim=(0,107),xlim=(-.42,2.4));ax.set_xticks(range(3),['Precision','Recall','F1']);ax.set_ylabel('Instance score (%)',labelpad=3);ax.set_title('Root ownership',pad=7)
 ax.legend(frameon=False,fontsize=7,loc='upper center',bbox_to_anchor=(.47,.81),handlelength=1.3,labelspacing=.3)
 for ax in [axes[0],axes[2],axes[3],axes[4],axes[5]]:
  ax.set_axisbelow(True);ax.grid(axis='x'if ax in [axes[0],axes[2]]else'y',color='#E9ECEF',lw=.4)
 fig.text(.02,.018,'Mean ± optimization SD (3 seeds); seed points shown in A, C and D. Internal B3 agreement; cross-tube accuracy awaits human labels.',fontsize=7.0)
 fig.canvas.draw();renderer=fig.canvas.get_renderer();overflow=[]
 for text in fig.findobj(matplotlib.text.Text):
  if not text.get_visible()or not text.get_text():continue
  bb=text.get_window_extent(renderer)
  if bb.x0<-.5 or bb.y0<-.5 or bb.x1>fig.bbox.width+.5 or bb.y1>fig.bbox.height+.5:overflow.append(text.get_text())
 assert not overflow,('Text outside final canvas',overflow)
 for ext in ['pdf','svg','png','jpg','tif']:
  kw={'dpi':600,'facecolor':'white'}
  if ext=='jpg':kw['pil_kwargs']={'quality':98,'subsampling':0}
  if ext=='tif':kw['pil_kwargs']={'compression':'tiff_lzw'}
  fig.savefig(out/('V90_controlled_experiments_SCI.'+ext),**kw)
 fig.savefig(out/'figure_preview.png',dpi=170,facecolor='white');plt.close(fig)
 with(out/'Figure_source_values.csv').open('w',encoding='utf-8-sig',newline='')as f:
  w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
 inputs=['Internal_results_per_run.csv','Internal_results_per_class.csv','Internal_results_mean_SD.csv','Paired_decoder_stage_results.csv','frozen_references.json','data_audit.json','locked_protocol.json','independent_protocol.json']
 for n in inputs:shutil.copy2(a.results/n,out/('source_'+n))
 summary=dict(effect_direction='full minus ablated, percentage points',paired_module_effects=effects,
  paired_decoder_macro={s:stats(macro[i])for i,s in enumerate(stages)},paired_decoder_classes=decoder_class,
  pairwise_instance=instance,internal_full=stats(vals('full','test_macro_Dice')),
  frozen_V66_macro_Dice_percent=frozen['macro_Dice']*100,
  figure_size_mm=[190,145],font='Times New Roman',axis_font_pt=8,panel_letter_font_pt=12,raster_dpi=600,
  sources_SHA256={n:hashlib.sha256((a.results/n).read_bytes()).hexdigest()for n in inputs},
  independent_accuracy_available=False,read_existing_results_only=True,training_or_inference_run=False,
  all_visible_text_inside_canvas=True)
 (out/'Figure_metrics_and_checks.json').write_text(json.dumps(summary,indent=2))
 print('SCI_FIGURE_COMPLETE',out,flush=True)
if __name__=='__main__':main()
