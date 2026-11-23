from pathlib import Path
import csv,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
W=Path(__file__).resolve().parents[1];R=W/'reports';F=Path('/workspace/work/root_workflow_real_assets_20261109/fonts')
for n in ['Times New Roman.ttf','Times New Roman Bold.ttf']:font_manager.fontManager.addfont(str(F/n))
plt.rcParams.update({'font.family':'Times New Roman','font.size':9,'text.color':'black','axes.labelcolor':'black','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False});rows=list(csv.DictReader((R/'Paired_decoder_stage_results.csv').open(encoding='utf-8-sig')));stages=['raw_head','axis_constraints','axis_and_junction_constraints'];names=['Raw head','Axis','Axis + junction'];fig,axes=plt.subplots(1,3,figsize=(11,3.7));fig.subplots_adjust(left=.06,right=.985,top=.77,bottom=.23,wspace=.4)
for ax,letter in zip(axes,'ABC'):ax.text(-.16,1.18,letter,transform=ax.transAxes,fontweight='bold',fontsize=14);ax.tick_params(labelsize=8)
a=axes[0]
for seed in [20261123,20261124,20261125]:
 vals=[float(next(r for r in rows if r['seed']==str(seed)and r['stage']==s)['macro_Dice'])*100 for s in stages];a.plot(range(3),vals,c='#999999',ls='--',lw=.8,marker='o',ms=3)
means=[np.mean([float(r['macro_Dice'])*100 for r in rows if r['stage']==s])for s in stages];sd=[np.std([float(r['macro_Dice'])*100 for r in rows if r['stage']==s],ddof=1)for s in stages];a.errorbar(range(3),means,yerr=sd,color='#9E435C',lw=2,marker='s',ms=5,capsize=3)
for i,v in enumerate(means):a.text(i,v+3,f'{v:.2f}',ha='center',fontweight='bold',fontsize=8)
a.set_xticks(range(3),names,rotation=18,ha='right');a.set(ylabel='Macro Dice (%)',ylim=(0,100),title='Paired decoder intervention')
a=axes[1]
for k,name,c in [('persistent','Persistent','#356FB0'),('new','Newly observed','#E7A835'),('extension','Extending','#AE4082')]:
 vals=[[float(r[k+'_Dice'])*100 for r in rows if r['stage']==s]for s in stages];a.errorbar(range(3),[np.mean(x)for x in vals],yerr=[np.std(x,ddof=1)for x in vals],lw=1.4,marker='o',ms=4,capsize=2,color=c,label=name)
a.set_xticks(range(3),names,rotation=18,ha='right');a.set(ylabel='Class Dice (%)',ylim=(0,105),title='Root class consistency');a.legend(frameon=False,fontsize=7,loc='center right')
a=axes[2];x=np.arange(2)
for off,key,label,c in [(-.24,'pairwise_instance_precision','Precision','#5275A0'),(0,'pairwise_instance_recall','Recall','#BD905D'),(.24,'pairwise_instance_F1','F1','#8F637F')]:
 vals=[[float(r[key])*100 for r in rows if r['stage']==s]for s in stages[1:]];a.bar(x+off,[np.mean(v)for v in vals],width=.21,yerr=[np.std(v,ddof=1)for v in vals],capsize=2,color=c,label=label,error_kw={'lw':.8})
a.set_xticks(x,names[1:],rotation=18,ha='right');a.set(ylabel='Pairwise instance score (%)',ylim=(0,105),title='Root ownership agreement');a.legend(frameon=False,fontsize=7,loc='upper center',ncol=3)
fig.suptitle('Whole-field structure decoding: identical neural output within each seed',fontweight='bold',fontsize=12);fig.text(.06,.03,'B3 internal reference agreement; mean ± optimization SD (3 seeds). Separate inference context from the primary partition-isolated module study.',fontsize=8.5)
for ext in ['png','jpg','pdf']:
 kwargs=dict(dpi=600,facecolor='white')if ext!='pdf'else dict(facecolor='white')
 if ext=='jpg':kwargs['pil_kwargs']={'quality':97}
 fig.savefig(R/('Structured_decoder_SCI.'+ext),**kwargs)
plt.close(fig);print('DECODER_PLOT_READY')
