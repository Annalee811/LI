"""Publication panels from measured outputs; no smoothing or invented samples."""
import argparse,json,string
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch,Rectangle
import common as c
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();r=a.run;o=r/'paper_figures';o.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
colors=['#8866AA','#009E73','#E6AB02'];names=['Persistent','Newly observed','Tip growth'];model_names=['U-Net','DeepLabV3+','SegFormer-B0'];mc=['#0072B2','#D55E00','#009E73']
legend=[Patch(color=c.PALETTE[k]/255,label=n) for k,n in [(4,'T1 baseline'),(1,'Persistent'),(2,'Newly observed'),(3,'Tip growth')]]
def save(fig,name,vector=False):
 fig.savefig(o/(name+'.tif'),dpi=600,pil_kwargs={'compression':'tiff_lzw'})
 fig.savefig(o/(name+'.png'),dpi=130)
 fig.savefig(o/(name+'.pdf'),dpi=300)
 if vector:fig.savefig(o/(name+'.svg'))
 plt.close(fig);print('FIGURE',name,flush=True)
def load(path):return np.array(Image.open(path))
rgb={t:load(r/'data/upper'/f'{t}_RGB.png') for t in c.PERIODS}
refs={ref:{t:load(r/'data'/ref/f'{t}_classes.png') for t in c.PERIODS} for ref in ['original','revised']}
preds={m:{t:load(r/m/'four_period_prediction'/f'{t}_temporal_classes.png') for t in c.PERIODS} for m in c.MODELS}
roi=(1200,220,1800,820);sl=np.s_[roi[1]:roi[3],roi[0]:roi[2]]
def overlay(im,cl):
 z=im.copy();v=(cl>0)&(cl<255);z[v]=np.rint(.25*im[v]+.75*c.PALETTE[cl[v]]).astype('uint8');return z
def panels(name,mode='overlay',local=False,original=False):
 columns=['Original RGB']+(['Original user reference'] if original else [])+['U-Net-derived reference']+model_names
 if mode=='isolated':columns=columns[1:]
 fig,ax=plt.subplots(4,len(columns),figsize=(11.69,8.0 if local else 6.8),squeeze=False)
 for i,t in enumerate(c.PERIODS):
  cls=([refs['original'][t]] if original else [])+[refs['revised'][t]]+[preds[m][t] for m in c.MODELS]
  views=([rgb[t]] if mode!='isolated' else [])+[(overlay(rgb[t],cl) if mode=='overlay' else c.PALETTE[np.where(cl==255,0,cl)]) for cl in cls]
  for j,(axis,im) in enumerate(zip(ax[i],views)):
   axis.imshow(im[sl] if local else im,interpolation='none');axis.set_xticks([]);axis.set_yticks([])
   for sp in axis.spines.values():sp.set_visible(False)
   if i==0:axis.set_title(columns[j],fontsize=8,pad=6)
   if j==0:axis.set_ylabel(t,rotation=0,labelpad=14,fontweight='bold')
   if not local and mode=='overlay' and j==0:axis.add_patch(Rectangle((roi[0],roi[1]),roi[2]-roi[0],roi[3]-roi[1],fill=False,edgecolor='#F0E442',lw=.7))
 fig.legend(handles=legend,ncol=4,loc='lower center',frameon=False,bbox_to_anchor=(.5,.005))
 fig.subplots_adjust(left=.035,right=.995,top=.94,bottom=.065,wspace=.035,hspace=.08)
 save(fig,name)
panels('Fig3_roots_without_background',mode='isolated')
panels('Fig4_local_roots_without_background',mode='isolated',local=True)
# Paired measured training trajectories; do not interpolate missing validation epochs.
fig,axes=plt.subplots(2,3,figsize=(10.8,6),sharex='col')
for j,m in enumerate(c.MODELS):
 rows=c.rows(r/m/'history.csv');ep=[int(z['epoch']) for z in rows]
 axes[0,j].plot(ep,[float(z['train_loss']) for z in rows],color=mc[j],lw=1.4);axes[0,j].set_title(model_names[j]);axes[0,j].set_ylabel('Training objective');axes[0,j].grid(alpha=.2)
 for ref,co in [('revised','#0072B2'),('original','#D55E00')]:
  rr=[z for z in rows if z['validation_'+ref+'_macro_Dice']];baseline=__import__('json').loads((r/m/'baseline_validation_whole_axis.json').read_text())[ref]['macro_Dice'];axes[1,j].plot([0]+[int(z['epoch']) for z in rr],[baseline]+[float(z['validation_'+ref+'_macro_Dice']) for z in rr],'-o',ms=3,color=co,label='U-Net-derived reference' if ref=='revised' else 'Original reference')
 axes[1,j].set_xlabel('Epoch');axes[1,j].set_ylabel('Validation macro Dice');axes[1,j].set_ylim(0,1);axes[1,j].grid(alpha=.2)
axes[1,0].legend(frameon=False,fontsize=7);fig.tight_layout();save(fig,'FigS1_learning_curves',True)
metrics={m:json.loads((r/m/'metrics.json').read_text()) for m in c.MODELS}
fig,axes=plt.subplots(1,2,figsize=(10.8,4.2),sharey=True)
for ax,ref in zip(axes,['original','revised']):
 for j,m in enumerate(c.MODELS):
  val=[q['Dice'] for q in metrics[m]['test'][ref]['end_to_end']['classes']];ax.bar(np.arange(3)+(j-1)*.25,val,.24,color=mc[j],label=model_names[j])
 ax.set_xticks(range(3),names);ax.set_ylim(0,1);ax.set_ylabel('Class Dice');ax.set_title('Original user reference' if ref=='original' else 'U-Net-derived reference');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
axes[0].legend(frameon=False);fig.tight_layout();save(fig,'Fig5_temporal_class_Dice',True)
period=c.rows(r/'tables/per_transition_metrics.csv');fig,axes=plt.subplots(1,3,figsize=(10.8,3.8),sharey=True)
for j,m in enumerate(c.MODELS):
 for k in [1,2,3]:
  rr=[z for z in period if z['model']==m and z['split']=='test' and z['reference']=='revised' and int(z['class_id'])==k];axes[j].plot([z['current'] for z in rr],[float(z['Dice']) if z['Dice'] else np.nan for z in rr],'-o',color=colors[k-1],label=names[k-1],ms=4)
 axes[j].set_title(model_names[j]);axes[j].set_ylim(0,1);axes[j].set_ylabel('Class Dice vs U-Net-derived reference');axes[j].grid(alpha=.2)
axes[0].legend(frameon=False);fig.tight_layout();save(fig,'FigS2_transition_Dice',True)
length=c.rows(r/'tables/skeleton_length_pairs.csv');summary=c.rows(r/'tables/skeleton_length_agreement.csv')
fig,axes=plt.subplots(2,3,figsize=(10.8,6.5))
for i,ref in enumerate(['original','revised']):
 for j,m in enumerate(c.MODELS):
  rr=[z for z in length if z['model']==m and z['reference']==ref and z['class_id']=='0'];x=np.array([float(z['reference_skeleton_pixels']) for z in rr]);y=np.array([float(z['predicted_skeleton_pixels']) for z in rr]);ss=next(z for z in summary if z['model']==m and z['reference']==ref and z['class_id']=='0');ax=axes[i,j];mx=max(x.max(),y.max())*1.04
  ax.scatter(x,y,s=20,facecolor=mc[j],edgecolor='white',lw=.3,alpha=.85);ax.plot([0,mx],[0,mx],'--',color='#777777',lw=.8)
  xx=np.array([0,mx]);ax.plot(xx,float(ss['slope'])*xx+float(ss['intercept']),color=mc[j],lw=1)
  ax.text(.03,.96,f"Identity R² = {float(ss['R2_identity']):.3f}\nOLS R² = {float(ss['R2_OLS']):.3f}\nRMSE = {float(ss['RMSE_skeleton_pixels']):.1f} px",transform=ax.transAxes,va='top',fontsize=7)
  ax.set_xlim(0,mx);ax.set_ylim(0,mx);ax.set_xlabel('Reference skeleton pixels');ax.set_ylabel('Predicted skeleton pixels');ax.set_title(model_names[j]+(' | original' if ref=='original' else ' | revised'));ax.grid(alpha=.15)
fig.tight_layout();save(fig,'FigS3_skeleton_length_agreement',True)
fig,axes=plt.subplots(1,3,figsize=(11.0,3.8))
for j,m in enumerate(c.MODELS):
 cm=np.array(metrics[m]['test']['revised']['end_to_end']['confusion_matrix']);den=cm.sum(1,keepdims=True);norm=np.divide(cm,den,out=np.zeros_like(cm,dtype=float),where=den>0);ax=axes[j];im=ax.imshow(norm,vmin=0,vmax=1,cmap='Blues')
 for y in range(4):
  for x in range(4):ax.text(x,y,f'{norm[y,x]:.2f}',ha='center',va='center',fontsize=7,color='white' if norm[y,x]>.55 else '#222222')
 ax.set_xticks(range(4),['BG','Persist.','New','Growth'],rotation=35,ha='right');ax.set_yticks(range(4),['BG','Persist.','New','Growth']);ax.set_xlabel('Prediction');ax.set_ylabel('U-Net-derived reference');ax.set_title(model_names[j])
fig.colorbar(im,ax=axes,shrink=.7,pad=.02,label='Row proportion');fig.subplots_adjust(left=.07,right=.88,bottom=.23,top=.88,wspace=.5);save(fig,'FigS6_confusion_matrices',True)
# Choose tile by number of reference growth pixels, not by any prediction score.
test=[z for z in c.rows(r/'data/paired_manifest.csv') if z['split']=='test'];chosen=max(test,key=lambda z:int((refs['revised'][z['current']][int(z['y']):int(z['y'])+256,int(z['x']):int(z['x'])+256]==3).sum()));x,y=int(chosen['x']),int(chosen['y']);ss=np.s_[y:y+256,x:x+256];gt=refs['revised'][chosen['current']][ss]
fig,axes=plt.subplots(3,4,figsize=(10.5,8));errcolors=['#009E73','#CC79A7','#E69F00']
for i,m in enumerate(c.MODELS):
 pred=load(r/m/'evaluation/test'/(chosen['pair_id']+'_prediction.png'));axes[i,0].imshow(overlay(rgb[chosen['current']][ss],gt),interpolation='none');axes[i,0].set_ylabel(model_names[i]);
 for k in [1,2,3]:
  out=np.full((*gt.shape,3),255,np.uint8);truth=gt==k;guess=pred==k;out[truth&guess]=[0,158,115];out[guess&~truth]=[204,121,167];out[truth&~guess]=[230,159,0];axes[i,k].imshow(out,interpolation='none')
 for j in range(4):
  axes[i,j].set_xticks([]);axes[i,j].set_yticks([])
  if i==0:axes[i,j].set_title('U-Net-derived reference' if j==0 else names[j-1])
fig.legend(handles=[Patch(color=z,label=n) for z,n in zip(errcolors,['Correct root class','False positive','False negative'])],loc='lower center',ncol=3,frameon=False);fig.subplots_adjust(left=.08,right=.99,top=.94,bottom=.065,wspace=.03,hspace=.03);save(fig,'FigS5_class_error_localization')
c.js(o/'figure_protocol.json',dict(references_disclosed=True,local_RGB_ROI=list(roi),error_example=chosen,error_example_selection='Maximum revised-reference growth pixels among test pairs; no predictions used',all_upper_panels_include_training_pixels=True,quantitative_panels='18 same-field date-pairs, not 18 independent plants or tubes',native_RGB_not_enhanced=True,TIF_dpi=600,raw_data_in_tables=True))
(o/'CAPTIONS.md').write_text('''# U-Net-reference / whole-axis v16 figure descriptions

All panels show 0353-2-B3, upper region only. Blue: T1 baseline; purple: persistent segment; green: newly observed root/side root; gold: tip growth. T2–T4 are each relative to the immediately previous period. No death classification is made. Root identity at crossings and occlusions is not independently established.

- Integrated TIFF: Four-date 72-panel atlas of original RGB, U-Net-derived reference and all three outputs, with full/local overlays and background-removed temporal roots. Original RGB controls have the matching ROI boxes. See figure_protocol.json for the exact ROI and layout.
- Fig3 / Fig4: Temporal classes with the background removed, complete upper region / common local window. Raw RGB-only root cutouts are also in each model result package.
- Fig5: Three temporal class Dice values on the 18 internal tile/date pairs, reported against both references. T1 is excluded from macro Dice. Original human classes are retained separately; both morphology and temporal classes derive from the latest U-Net output.
- S1: Actual 24-epoch training objectives and measured validation macro Dice. Validation whole-axis outputs were evaluated at epoch zero and every four epochs; points are connected without curve fitting. Decoder calibration occurs after these recorded trajectories.
- S2: Class Dice for T1→T2, T2→T3 and T3→T4 against the revised reference, pooling confusion counts within each transition rather than averaging ratios.
- S3: Skeleton pixel-count agreement for all typed roots, separately against original and revised references. Dashed line: identity. Solid line: OLS fit. Identity R², OLS R² and RMSE are distinct. There are no independent manual physical-length measurements.
- S6: Row-normalized four-class confusion matrices against the revised reference, including background errors from the frozen root detectors.
- S5: One-vs-rest class error maps on the test example containing the most revised-reference growth pixels. Green: true positive; pink: false positive; orange: false negative. Selection uses only reference content, not model performance.

The revised reference includes both temporal classes and morphology from the starting v15 U-Net. U-Net agreement with its own reference is circular, not independent accuracy. Original human labels remain separate. These are same-position development results. The frozen root models and historical warm starts have prior same-field exposure. Whole-axis decoding also uses full-image prediction context across tile boundaries. No independent external test image is available. Do not describe the results as an unbiased cross-tube test or biological root-identity validation.
''',encoding='utf-8')
