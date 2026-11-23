from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from openpyxl import Workbook
from openpyxl.styles import Font,PatternFill
W=Path(__file__).resolve().parents[1];R=W/'reports';F=Path('/workspace/work/root_workflow_real_assets_20261109/fonts');
for n in ['Times New Roman.ttf','Times New Roman Bold.ttf']:font_manager.fontManager.addfont(str(F/n))
plt.rcParams.update({'font.family':'Times New Roman','font.size':9,'axes.labelcolor':'black','text.color':'black','xtick.color':'black','ytick.color':'black','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
def rows(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
summ=rows(R/'Internal_results_mean_SD.csv');per=rows(R/'Internal_results_per_run.csv');cl=rows(R/'Internal_results_per_class.csv');frozen=json.loads((R/'frozen_references.json').read_text());order=['current_only','no_lstm','no_gat','no_gates','full'];names={'current_only':'Frame-only\nclassifier','no_lstm':'w/o ConvLSTM','no_gat':'ConvLSTM only\n(w/o GAT)','no_gates':'w/o identity gates','full':'Full'};colors={'current_only':'#6F7780','no_lstm':'#2A7F86','no_gat':'#5275A0','no_gates':'#8A6696','full':'#9E435C'}
fig,axs=plt.subplots(2,2,figsize=(11,7.4));fig.subplots_adjust(left=.075,right=.985,bottom=.12,top=.91,wspace=.29,hspace=.55)
for ax,letter in zip(axs.ravel(),'ABCD'):ax.text(-.14,1.08,letter,transform=ax.transAxes,fontsize=15,fontweight='bold');ax.tick_params(labelsize=8)
a=axs[0,0]
for v in order:
 data=[]
 for seed in [20261123,20261124,20261125]:
  h=rows(W/'experiments'/v/str(seed)/'training_history.csv');data.append([float(r['validation_macro_Dice'])*100 for r in h])
 data=np.array(data);x=np.arange(1,101);mean=data.mean(0);sd=data.std(0,ddof=1);a.plot(x,mean,color=colors[v],lw=1.4,label=names[v].replace('\n',' '));a.fill_between(x,mean-sd,mean+sd,color=colors[v],alpha=.12,lw=0)
a.axhline(frozen['V66_pairwise_frozen']['metrics_by_split']['1']['macro_Dice']*100,color='#A88A60',ls='--',lw=1,label='Frozen V66');a.set(xlabel='Optimization step',ylabel='Validation macro Dice (%)',xlim=(1,100));a.set_title('Training stability',fontweight='bold',fontsize=11);a.legend(frameon=False,fontsize=7,loc='lower right',ncol=2)
a=axs[0,1];methods=['V66']+order;labels=['Frozen V66']+[names[v]for v in order]
for i,v in enumerate(methods):
 if v=='V66':vals=np.array([frozen['V66_pairwise_frozen']['metrics_by_split']['2']['macro_Dice']*100]);c='#A88A60'
 else:vals=np.array([float(r['test_macro_Dice'])*100 for r in per if r['variant']==v]);c=colors[v]
 a.bar(i,vals.mean(),width=.62,color=c,alpha=.9);a.errorbar(i,vals.mean(),yerr=vals.std(ddof=1)if len(vals)>1 else 0,color='black',capsize=3,fmt='none',lw=1);a.scatter(i+np.linspace(-.09,.09,len(vals)),vals,c='black',s=12,zorder=4);a.text(i,max(vals)+2.5,f'{vals.mean():.1f}',ha='center',fontsize=8,fontweight='bold')
a.set_xticks(range(6),labels,rotation=23,ha='right',fontsize=7.5);a.set(ylabel='Test macro Dice (%)',ylim=(0,100));a.set_title('Same internal held-out pixels',fontweight='bold',fontsize=11)
a=axs[1,0];classcolors=['#356FB0','#E7A835','#AE4082'];classnames=['persistent','new','extension'];display=['Persistent','Newly observed','Extending'];x=np.arange(6)
for k,c in enumerate(classnames):
 means=[];sds=[]
 for v in methods:
  vals=[frozen['V66_pairwise_frozen']['metrics_by_split']['2']['classes'][k]['Dice']*100]if v=='V66'else[float(r['Dice'])*100 for r in cl if r['variant']==v and r['class_name']==c];means.append(np.mean(vals));sds.append(np.std(vals,ddof=1)if len(vals)>1 else 0)
 a.bar(x+(k-1)*.24,means,width=.22,color=classcolors[k],label=display[k],yerr=sds,capsize=2,error_kw={'lw':.7,'ecolor':'black'})
a.set_xticks(x,labels,rotation=23,ha='right',fontsize=7.5);a.set(ylabel='Class Dice (%)',ylim=(0,105));a.set_title('Temporal root classes',fontweight='bold',fontsize=11);a.legend(frameon=False,fontsize=7.5,ncol=3,loc='upper center')
a=axs[1,1];x=np.arange(5)
for key,label,c,off in [('test_identity_link_F1','Identity-link F1','#526F91',-.17),('test_junction_assignment','Junction assignment','#92617E',.17)]:
 means=[];sds=[]
 for v in order:
  vals=np.array([float(r[key])*100 for r in per if r['variant']==v]);means.append(vals.mean());sds.append(vals.std(ddof=1))
 a.bar(x+off,means,width=.3,yerr=sds,capsize=2,label=label,color=c,error_kw={'lw':.8})
a.set_xticks(x,[names[v]for v in order],rotation=23,ha='right',fontsize=7.5);a.set(ylabel='Internal relation score (%)',ylim=(0,105));a.set_title('Root relation supervision',fontweight='bold',fontsize=11);a.legend(frameon=False,ncol=2,fontsize=7.5,loc='upper center')
fig.suptitle('B3 controlled module study: fixed upstream prior, freshly trained modules',fontsize=13,fontweight='bold');fig.text(.075,.018,'Mean ± sample SD over three optimization seeds; frozen V66 is a single reference. These are internal results, not independent-tube accuracy.',fontsize=9)
fig.savefig(R/'Baseline_ablation_SCI.png',dpi=600,facecolor='white');fig.savefig(R/'Baseline_ablation_SCI.jpg',dpi=600,facecolor='white',pil_kwargs={'quality':97});fig.savefig(R/'Baseline_ablation_SCI.pdf',facecolor='white');plt.close(fig)
wb=Workbook();wb.remove(wb.active)
for name,file in [('Controlled_mean_SD','Internal_results_mean_SD.csv'),('Controlled_each_run','Internal_results_per_run.csv'),('Classwise_results','Internal_results_per_class.csv'),('Decoder_paired','Paired_decoder_stage_results.csv'),('Frozen_references','Historical_reference_and_decoder_stage_scores.csv'),('Independent_cohort','annotation_checklist.csv')]:
 p=R/file
 if not p.exists():continue
 rr=rows(p);ws=wb.create_sheet(name)
 if not rr:continue
 keys=list(rr[0]);ws.append(keys)
 for r in rr:
  vals=[]
  for key in keys:
   v=r[key]
   try:v=float(v)if any(c in v for c in '.eE')else int(v)
   except(ValueError,TypeError):pass
   vals.append(v)
  ws.append(vals)
 for col in ws.columns:
  ws.column_dimensions[col[0].column_letter].width=min(48,max(14,max(len(str(c.value or''))for c in col)+2))
 for c in ws[1]:c.font=Font(name='Times New Roman',bold=True,color='FFFFFF');c.fill=PatternFill('solid',fgColor='28495C')
 ws.freeze_panes='C2';ws.auto_filter.ref=ws.dimensions
 for i,key in enumerate(keys,1):
  if any(word in key.lower()for word in ['dice','iou','precision','recall','f1','assignment'])and'events'not in key and'pixels'not in key:
   for cells in ws.iter_cols(min_col=i,max_col=i,min_row=2):
    for c in cells:
     if isinstance(c.value,(float,int)):c.number_format='0.00%'
ws=wb.create_sheet('Read_me');notes=['These are internal B3 feature-backbone experiments; not external generalization proof.','Three seeds measure optimization variability, not independent biological replication.','All retrained variants: 100 full-graph updates, 8192 sampled training root pixels per update, validation-only checkpoint selection.','Current-frame-only removes prior logits/history inputs; no-LSTM retains frame CNN and graph/history cues.','No learned identity gates retains the geometric distance prior and GAT attention.','Pure-node partition masks and cross-partition edge removal are common to all controlled models.','Pixel scores cover eligible typed reference-root pixels at T2–T4; fixed foreground, background not scored in these internal class metrics.','Temporal class labels use current reference B; identity labels use the original released identity supervision.','Historical frozen results and whole-field decoder stage comparisons are separate protocols.','Independent test: 3 tubes, 7 positions, 26 frames; human labels unavailable; no independent Dice/IoU computed.','Test RGBs had prior unlabelled prediction exposure; not a never-seen prospective blind-image study.']
for note in notes:ws.append([note])
ws.column_dimensions['A'].width=135;wb.save(R/'Baseline_ablation_and_test_index.xlsx');print('PLOTS_AND_TABLES_READY')
