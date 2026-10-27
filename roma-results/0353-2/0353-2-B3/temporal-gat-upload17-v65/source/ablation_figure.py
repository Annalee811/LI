import json
import numpy as np
import matplotlib.pyplot as plt
from prepare import O,MODELS,js
from figures import plot_save,CLASS_NAMES,NAMES

def main():
 s=json.loads((O/'metrics_summary.json').read_text());rows=s['temporal_classes'];plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False});fig,axs=plt.subplots(1,3,figsize=(11.5,4.3),sharey=True)
 for ax,k in zip(axs,[2,1,3]):
  for j,m in enumerate(MODELS):
   for offset,variant,color,name in [(-.24,'archived','#bbc1c7','Archived'),(0,'mlp','#927da5','New MLP control'),(.24,'gat','#348b97','New GAT')]:
    value=next(r['Dice'] for r in rows if r['model']==m and r['variant']==variant and r['class_code']==k);ax.bar(j+offset,value,.22,color=color,label=name if j==0 else None);ax.text(j+offset,value+.014,f'{value:.3f}',fontsize=6,ha='center',rotation=90)
  ax.set(title=CLASS_NAMES[k],xticks=np.arange(3),xticklabels=[NAMES[m] for m in MODELS],ylim=(0,1.12),ylabel='Dice' if k==2 else '');ax.tick_params(axis='x',labelrotation=15);ax.legend(frameon=False,fontsize=6.5);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
 fig.suptitle('Ablation: matched labels, features and family split; include the no-GAT control',fontsize=11);fig.tight_layout(rect=(0,0,1,.92));files=[plot_save(fig,'Fig_GAT_MLP_temporal_ablation')]
 fig,axs=plt.subplots(1,2,figsize=(9,4.1));idrows=s['identity']
 for ax,metric,title in zip(axs,['same_root_F1','heldout_post_seed_name_accuracy'],['Held-out same-root candidate-pair F1','Post-seed root-date name agreement']):
  for j,m in enumerate(MODELS):
   for off,kind,color,name in [(-.18,'mlp','#927da5','No-GAT control'),(.18,'gat','#348b97','GAT')]:
    value=next(r[metric] for r in idrows if r['model']==m and r['variant']==kind);ax.bar(j+off,value,.34,color=color,label=name if j==0 else None);ax.text(j+off,value+.025,f'{value:.3f}',ha='center',fontsize=8)
  ax.set(title=title,xticks=np.arange(3),xticklabels=[NAMES[m] for m in MODELS],ylim=(0,1.05),ylabel='Score');ax.tick_params(axis='x',labelrotation=12);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);ax.legend(frameon=False,fontsize=8)
 fig.suptitle('Identity evaluation: held-out families; initial seed names excluded from name agreement',fontsize=11);fig.tight_layout(rect=(0,0,1,.9));files.append(plot_save(fig,'Fig_Root_ID_tracking_ablation'))
 js(O/'ablation_figure_manifest.json',dict(passed=True,figures=files));print('ABLATION_FIGURES_COMPLETE',flush=True)
if __name__=='__main__':main()
