"""Three publication-size figures; all marks come from fixed existing outputs."""
from pathlib import Path
import argparse,csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
W=Path(__file__).resolve().parent
COL={'existing':'#3F7091','new':'#C58D31','macro':'#826071','iou':'#7B8795'}
def read(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def n(v):return None if v==''else float(v)
def save(fig,stem):
 fig.canvas.draw();r=fig.canvas.get_renderer();outside=[]
 for t in fig.findobj(matplotlib.text.Text):
  if not t.get_visible()or not t.get_text():continue
  b=t.get_window_extent(r)
  if b.x0<-.5 or b.y0<-.5 or b.x1>fig.bbox.width+.5 or b.y1>fig.bbox.height+.5:outside.append(t.get_text())
 assert not outside,('Clipped text',stem,outside)
 for ext in ['png','jpg','pdf','svg','tif']:
  kw=dict(dpi=600,facecolor='white')
  if ext=='jpg':kw['pil_kwargs']=dict(quality=98,subsampling=0)
  if ext=='tif':kw['pil_kwargs']=dict(compression='tiff_lzw')
  fig.savefig(W/(stem+'.'+ext),**kw)
 fig.savefig(W/(stem+'_preview.png'),dpi=170,facecolor='white');plt.close(fig)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--fonts',type=Path,default=Path('/workspace/work/root_workflow_real_assets_20261109/fonts'));a=ap.parse_args()
 for f in ['Times New Roman.ttf','Times New Roman Bold.ttf']:font_manager.fontManager.addfont(str(a.fonts/f))
 plt.rcParams.update({'font.family':'Times New Roman','font.size':8,'axes.labelsize':8,'axes.labelweight':'bold','axes.titlesize':9,'axes.titleweight':'bold',
  'text.color':'black','axes.labelcolor':'black','xtick.color':'black','ytick.color':'black','axes.edgecolor':'black','axes.linewidth':.65,
  'xtick.major.width':.6,'ytick.major.width':.6,'xtick.labelsize':7.5,'ytick.labelsize':7.5,'axes.spines.top':False,'axes.spines.right':False,
  'pdf.fonttype':42,'svg.fonttype':'none'})
 runs=read(W/'metrics/Internal_two_class_runs.csv');classes=read(W/'metrics/Internal_two_class_classwise.csv');refs=read(W/'metrics/Historical_two_class_references.csv')
 effects=read(W/'metrics/Paired_module_two_class_effects.csv');dec=read(W/'metrics/Paired_decoder_two_class.csv');dc=read(W/'metrics/Paired_decoder_two_class_classwise.csv')
 seeds=[20261123,20261124,20261125]
 def rv(model,key):
  rr=[r for r in refs if r['model']==model][:1]if model=='V66_pairwise_frozen'else[r for r in runs if r['variant']==model]
  return np.array([float(r[key])*100 for r in rr])
 def cv(model,cl,key):
  rr=[r for r in refs if r['model']==model and r['class_name']==cl]if model=='V66_pairwise_frozen'else[r for r in classes if r['variant']==model and r['class_name']==cl]
  return np.array([float(r[key])*100 for r in rr if r[key]!=''])
 baselines=['current_only','V66_pairwise_frozen','no_gat','full'];labels=['Frame-only control','Frozen V66 prior','ConvLSTM (no GAT)','Full temporal graph']
 short=['Frame-only\ncontrol','Frozen\nV66','ConvLSTM\n(no GAT)','Full\nmodel']
 fig=plt.figure(figsize=(190/25.4,135/25.4));a1=fig.add_axes((.175,.55,.34,.30));b=fig.add_axes((.65,.55,.32,.30));c=fig.add_axes((.245,.145,.70,.245))
 fig.text(.02,.974,'Temporal baselines: two-class internal re-evaluation',fontsize=11,fontweight='bold')
 for x,y,letter in [(.02,.90,'A'),(.56,.90,'B'),(.02,.43,'C')]:fig.text(x,y,letter,fontsize=12,fontweight='bold')
 for i,model in enumerate(baselines):
  for cl,dy,marker in [('existing',-.12,'s'),('new',.12,'o')]:
   x=cv(model,cl,'Dice');mean=x.mean();sd=x.std(ddof=1)if len(x)>1 else 0
   a1.errorbar(mean,i+dy,xerr=sd,fmt=marker,color=COL[cl],ms=4.8,capsize=2,lw=1.1)
   if len(x)>1:a1.scatter(x,i+dy+np.array([-.06,0,.06]),s=8,c='black',alpha=.65,zorder=4)
   a1.text(mean,i+(-.34 if cl=='existing'else .37),f'{mean:.2f}',ha='center',va='center',fontsize=7.5)
 a1.set_yticks(range(4),labels);a1.set_ylim(3.65,-.6);a1.set_xlim(-5,108);a1.set_xticks([0,25,50,75,100]);a1.set_xlabel('Class Dice (%)');a1.set_title('New and existing roots',pad=7);a1.spines['left'].set_visible(False);a1.tick_params(axis='y',length=0,pad=4);a1.grid(axis='x',color='#E9ECEF',lw=.45);a1.set_axisbelow(True)
 fig.legend(handles=[Line2D([],[],color=COL['new'],marker='o',lw=1.1,label='New'),Line2D([],[],color=COL['existing'],marker='s',lw=1.1,label='Existing')],frameon=False,ncol=2,bbox_to_anchor=(.34,.933),loc='center',prop={'size':8,'weight':'bold'},handlelength=1.5)
 for j,(key,name,col)in enumerate([('macro_Dice','Macro Dice',COL['macro']),('macro_IoU','Macro IoU',COL['iou'])]):
  for i,model in enumerate(baselines):
   x=rv(model,key);xx=i+(j-.5)*.31;b.bar(xx,x.mean(),width=.28,color=col,label=name if i==0 else None)
   b.errorbar(xx,x.mean(),yerr=x.std(ddof=1)if len(x)>1 else 0,fmt='none',color='black',capsize=2,lw=.7)
   if key=='macro_Dice':b.text(xx,x.mean()+3,f'{x.mean():.2f}',ha='center',fontsize=7.2,fontweight='bold')
 b.set_xticks(range(4),short,fontsize=7.1);b.set(ylim=(0,105),ylabel='Two-class macro score (%)');b.set_title('Same fixed test pixels',pad=7)
 handles,names=b.get_legend_handles_labels();fig.legend(handles,names,frameon=False,loc='center',bbox_to_anchor=(.81,.933),ncol=2,fontsize=7.3,handlelength=1.1,columnspacing=1.1)
 b.grid(axis='y',color='#E9ECEF',lw=.45);b.set_axisbelow(True)
 mat=[]
 for model in baselines:
  row=[]
  for cl,key in [('existing','Precision'),('existing','Recall'),('new','Precision'),('new','Recall')]:
   x=cv(model,cl,key);row.append(x.mean()if len(x)else np.nan)
  mat.append(row)
 mat=np.array(mat);cmap=LinearSegmentedColormap.from_list('pale',['#FFFFFF','#DCE7ED','#95B5C9']);cmap.set_bad('#F3F3F3');c.imshow(mat,vmin=0,vmax=100,cmap=cmap,aspect='auto')
 for i in range(4):
  for j in range(4):c.text(j,i,'—'if np.isnan(mat[i,j])else f'{mat[i,j]:.2f}',ha='center',va='center',fontsize=8,fontweight='bold'if i==3 else'normal')
 c.set_yticks(range(4),labels);c.set_xticks(range(4),['Existing\nprecision','Existing\nrecall','New\nprecision','New\nrecall']);c.tick_params(axis='both',length=0,pad=4);c.set_title('Class-specific precision and recall (%)',pad=9)
 for spine in c.spines.values():spine.set_visible(False)
 fig.text(.02,.047,'B3 held-out typed root pixels only; fixed foreground, background excluded. Mean ± optimization SD over 3 seeds; frozen V66 is one reference.',fontsize=7)
 fig.text(.02,.016,'Original checkpoints retained. Frame-only collapsed to existing-only; its New precision is undefined (—). Two-class scores are not three-class scores.',fontsize=7)
 save(fig,'Figure_1_Temporal_baselines_two_classes')
 # Figure2 separates matched module effects from paired decoder inference.
 fig=plt.figure(figsize=(190/25.4,145/25.4));ax1=fig.add_axes((.17,.61,.31,.24));ax2=fig.add_axes((.70,.61,.27,.24));ax3=fig.add_axes((.105,.15,.365,.28));ax4=fig.add_axes((.63,.15,.34,.28))
 fig.text(.02,.970,'Module ablations: two-class re-evaluation',fontsize=11,fontweight='bold')
 fig.text(.02,.914,'Matched neural modules',fontweight='bold',fontsize=8.5);fig.text(.02,.492,'Separate paired whole-field structural decoding',fontweight='bold',fontsize=8.5)
 for x,y,letter in [(.02,.868,'A'),(.57,.868,'B'),(.02,.445,'C'),(.53,.445,'D')]:fig.text(x,y,letter,fontsize=12,fontweight='bold')
 matrix=[[1,1,1],[0,1,1],[1,0,None],[1,1,0]];methodnames=['Full','w/o ConvLSTM','w/o GAT','w/o learned gates']
 ax1.imshow(np.array([[.9 if v==1 else .2 for v in r]for r in matrix]),cmap=LinearSegmentedColormap.from_list('config',['#FFFFFF','#E2E8ED']),vmin=0,vmax=1,aspect='auto')
 for i,row in enumerate(matrix):
  for j,val in enumerate(row):ax1.text(j,i,'n/a'if val is None else'+'if val else'−',ha='center',va='center',fontsize=9,fontweight='bold')
 ax1.set_yticks(range(4),methodnames);ax1.set_xticks(range(3),['ConvLSTM','GAT','Learned gate']);ax1.tick_params(length=0,pad=4);ax1.set_title('Module usage',pad=7)
 for sp in ax1.spines.values():sp.set_visible(False)
 for i,(module,label)in enumerate([('ConvLSTM','+ ConvLSTM'),('GAT','+ GAT'),('Learned identity gates','+ ID gate')]):
  x=np.array([float(r['delta_macro_Dice_pp'])for r in effects if r['module']==module]);ax2.errorbar(x.mean(),i,xerr=x.std(ddof=1),fmt='s',color=COL['macro'],ms=4.5,capsize=3,lw=1.1);ax2.scatter(x,i+np.array([-.12,0,.12]),s=13,c='black',alpha=.7,zorder=4);ax2.text(x.mean(),i-.30,f'{x.mean():+.3f}',ha='center',fontsize=7.7,fontweight='bold')
 ax2.axvline(0,color='#777777',lw=.65,ls=(0,(3,3)));ax2.set_yticks(range(3),['+ ConvLSTM','+ GAT','+ ID gate']);ax2.set_ylim(2.6,-.65);ax2.set_xlim(-.5,.5);ax2.set_xticks([-.4,-.2,0,.2,.4]);ax2.set_xlabel('Full − ablated Dice (pp)');ax2.set_title('Paired module effects',pad=7);ax2.tick_params(axis='y',length=0);ax2.spines['left'].set_visible(False);ax2.grid(axis='x',color='#E9ECEF',lw=.45);ax2.set_axisbelow(True)
 stages=['raw_head','axis_constraints','axis_and_junction_constraints'];sl=['Raw head','Axis','Axis + junction']
 mat=np.array([[float(next(r['macro_Dice']for r in dec if r['stage']==s and int(r['seed'])==seed))*100 for seed in seeds]for s in stages]);mean=mat.mean(1);sd=mat.std(1,ddof=1)
 for j in range(3):ax3.plot(range(3),mat[:,j],color='#B1B7BB',lw=.8,marker='o',ms=2.6)
 ax3.errorbar(range(3),mean,yerr=sd,fmt='s-',color=COL['macro'],ms=4.8,capsize=3,lw=1.5)
 for i,v in enumerate(mean):ax3.text(i,v+sd[i]+3,f'{v:.2f}',ha='center',fontsize=8,fontweight='bold')
 ax3.set(ylim=(0,105),xlim=(-.15,2.2),ylabel='Two-class macro Dice (%)');ax3.set_xticks(range(3),sl,fontsize=7.2);ax3.set_title('Paired decoding intervention',pad=7)
 for cl,marker,name in [('existing','s','Existing'),('new','o','New')]:
  mat=np.array([[float(next(r['Dice']for r in dc if r['stage']==s and int(r['seed'])==seed and r['class_name']==cl))*100 for seed in seeds]for s in stages]);mean=mat.mean(1);sd=mat.std(1,ddof=1)
  ax4.errorbar(range(3),mean,yerr=sd,fmt=marker+'-',color=COL[cl],ms=4.3,capsize=2,lw=1.3,label=name);ax4.text(2.16,mean[-1],f'{mean[-1]:.2f}',va='center',fontsize=7.5)
 ax4.set(ylim=(0,105),xlim=(-.15,2.55),ylabel='Class Dice (%)');ax4.set_xticks(range(3),sl,fontsize=7.1);ax4.set_title('New and existing roots',pad=7);ax4.legend(frameon=False,loc='lower left',bbox_to_anchor=(.02,.28),fontsize=7.5,ncol=2,handlelength=1.3)
 for ax in [ax3,ax4]:ax.grid(axis='y',color='#E9ECEF',lw=.45);ax.set_axisbelow(True)
 fig.text(.02,.058,'A–B: same internal partition protocol. C–D: identical whole-field neural outputs per seed; a separate inference context.',fontsize=7)
 fig.text(.02,.028,'Existing = persistent + extension. Means ± optimization SD; 3 seeds. Grouping does not merge root identities or establish identity accuracy.',fontsize=7)
 save(fig,'Figure_2_Module_ablations_two_classes')
 # Figure3: locked cohort coverage and same-field prediction examples, no invented scores.
 manifest=read(W/'Independent_manifest.csv');examples=json.loads((W/'Independent_examples.json').read_text());fig=plt.figure(figsize=(190/25.4,160/25.4))
 fig.text(.02,.975,'Cross-tube transfer: two-class predictions, ground truth pending',fontsize=10.5,fontweight='bold')
 fig.text(.02,.928,'A',fontsize=12,fontweight='bold');coverage=fig.add_axes((.21,.715,.33,.19));positions=sorted(set(r['position_id']for r in manifest))
 coverage.set(xlim=(-.5,3.5),ylim=(6.6,-.7));coverage.set_xticks(range(4),['T1','T2','T3','T4']);coverage.xaxis.tick_top();coverage.set_yticks(range(7),positions)
 for i,pos in enumerate(positions):
  present={r['period']for r in manifest if r['position_id']==pos}
  for j,t in enumerate(['T1','T2','T3','T4']):
   if t not in present:coverage.text(j,i,'—',ha='center',va='center',fontsize=8)
   elif t=='T1':coverage.scatter(j,i,marker='s',s=24,facecolor='white',edgecolor='#777777',lw=.7)
   else:coverage.scatter(j,i,marker='s',s=24,facecolor='#575F65',edgecolor='none')
 coverage.tick_params(length=0,pad=4)
 for sp in coverage.spines.values():sp.set_visible(False)
 fig.text(.625,.884,'Training tube excluded: 0353-2',fontsize=8.1,fontweight='bold')
 fig.text(.625,.847,'3 test tubes  |  7 positions  |  26 images',fontsize=8)
 fig.text(.625,.810,'7 baseline + 19 temporal observations',fontsize=8)
 fig.text(.625,.771,'Human temporal labels unavailable',fontsize=8,fontweight='bold')
 fig.text(.625,.735,'Independent scores: not evaluated',fontsize=8,fontweight='bold')
 fig.text(.21,.689,'Filled: temporal image   Open: baseline   —: not acquired',fontsize=7)
 fig.text(.02,.653,'B',fontsize=12,fontweight='bold')
 left=.13;colw=.205;gap=.012
 for j,t in enumerate(['T1','T2','T3','T4']):fig.text(left+j*(colw+gap)+colw/2,.649,t,ha='center',fontweight='bold',fontsize=8.5)
 for i,ex in enumerate(examples):
  pos=ex['position'];data=np.load(W/'figure3_image_cache'/(pos+'.npz'));y=.49-i*.155
  fig.text(.02,y+.055,pos,ha='left',va='center',fontsize=7.5,fontweight='bold',rotation=90)
  for j,t in enumerate(['T1','T2','T3','T4']):
   rgb=data[t+'_RGB'];pred=data[t+'_prediction'];over=rgb.copy()
   for code,cl in [(1,'existing'),(2,'new')]:
    sel=pred==code;col=np.array([int(COL[cl][k:k+2],16)for k in [1,3,5]]);over[sel]=np.rint(.92*col+.08*rgb[sel]).astype(np.uint8)
   over[pred==255]=np.rint(.75*np.array([205,205,205])+.25*rgb[pred==255]).astype(np.uint8)
   joint=np.concatenate([rgb,np.full((rgb.shape[0],3,3),255,np.uint8),over],axis=1)
   ax=fig.add_axes((left+j*(colw+gap),y,colw,.125));ax.imshow(joint,aspect='auto');ax.set_axis_off()
   if j==0:
    ax.plot([.035,.035+50/joint.shape[1]],[.07,.07],transform=ax.transAxes,color='white',lw=2)
    ax.text(.035,.13,'50 px',transform=ax.transAxes,fontsize=6.7,color='black',bbox=dict(facecolor='white',edgecolor='none',pad=.8,alpha=.9))
   if t=='T1':ax.text(.76,.04,'Baseline',transform=ax.transAxes,ha='center',fontsize=6.7,color='black',bbox=dict(facecolor='white',edgecolor='none',pad=.8,alpha=.85))
 fig.legend(handles=[Line2D([],[],color=COL['new'],lw=2.6,label='New'),Line2D([],[],color=COL['existing'],lw=2.6,label='Existing')],loc='center',bbox_to_anchor=(.68,.132),ncol=2,frameon=False,prop={'size':8,'weight':'bold'})
 fig.text(.13,.132,'Each tile: RGB | Full neural head',fontsize=7.7,va='center')
 fig.text(.02,.083,'Retrained Full neural head, fixed seed 20261123; foreground-based crop selection without GT or performance ranking. Same crop at T1–T4.',fontsize=7)
 fig.text(.02,.052,'Frozen transfer-feature approximation and prior unlabelled image exposure are documented. Examples demonstrate predictions, not accuracy.',fontsize=7)
 fig.text(.02,.021,'Existing combines persistent and elongating segments. Gray baseline roots are unassigned. Native predictions for all 26 images are provided.',fontsize=7)
 save(fig,'Figure_3_Independent_transfer_two_classes')
 (W/'Figure_design.json').write_text(json.dumps(dict(font='Times New Roman, black',raster_dpi=600,figures=[
  dict(name='Figure1',mm=[190,135],purpose='temporal baselines; common B3 fixed-root domain'),dict(name='Figure2',mm=[190,145],purpose='matched module effects and separate paired structure decoder'),
  dict(name='Figure3',mm=[190,160],purpose='locked cross-tube cohort coverage and actual two-class prediction examples; no accuracy values')],independent_scores_computed=False),indent=2));print('THREE_TWO_CLASS_KEY_FIGURES_COMPLETE',flush=True)
if __name__=='__main__':main()
