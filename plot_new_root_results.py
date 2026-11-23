"""New-class Precision/Recall figures only, from existing predictions."""
from pathlib import Path
import csv,json,sys,hashlib
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
W=Path(__file__).resolve().parent
STUDY=Path('/workspace/work/V90_validation_20261123')
O=Path('/workspace/work/V90_all_microtubes_20261118')
REF=Path('/workspace/work/V66_V90_same_field_SCI_20261113/sources/reference')
F=Path('/workspace/work/root_workflow_real_assets_20261109/fonts')
COLOR=np.array([216,153,49])
def read(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def number(v):return None if v in ['',None]else float(v)
def setup():
 for name in ['Times New Roman.ttf','Times New Roman Bold.ttf']:font_manager.fontManager.addfont(str(F/name))
 plt.rcParams.update({'font.family':'Times New Roman','font.size':8,'axes.labelsize':8,'axes.labelweight':'bold','axes.titleweight':'bold','axes.titlesize':9,
  'text.color':'black','axes.labelcolor':'black','xtick.color':'black','ytick.color':'black','axes.linewidth':.65,'xtick.major.width':.6,'ytick.major.width':.6,
  'xtick.labelsize':7.5,'ytick.labelsize':7.5,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
def export(fig,name):
 fig.canvas.draw();renderer=fig.canvas.get_renderer()
 for t in fig.findobj(matplotlib.text.Text):
  if not t.get_text()or not t.get_visible():continue
  b=t.get_window_extent(renderer)
  assert b.x0>=-1 and b.y0>=-1 and b.x1<=fig.bbox.width+1 and b.y1<=fig.bbox.height+1,('Text clipped',t.get_text())
 for ext in ['png','jpg','pdf','svg','tif']:
  kw=dict(dpi=600,facecolor='white')
  if ext=='jpg':kw['pil_kwargs']=dict(quality=98,subsampling=0)
  if ext=='tif':kw['pil_kwargs']=dict(compression='tiff_lzw')
  fig.savefig(W/(name+'.'+ext),**kw)
 fig.savefig(W/(name+'_preview.png'),dpi=170,facecolor='white');plt.close(fig)
def main():
 setup();runs=read(W/'metrics/New_root_internal_each_run.csv');dec=read(W/'metrics/New_root_paired_decoder.csv');refs=read(W/'metrics/New_root_frozen_heldout_reference.csv')
 order=['V66_frozen','current_only','no_lstm','no_gat','no_gates','full'];names=['Frozen V66','Frame-only','w/o LSTM','w/o GAT','w/o gates','Full']
 fig=plt.figure(figsize=(180/25.4,126/25.4));positions=[(.16,.60,.30,.29),(.66,.60,.30,.29),(.13,.13,.33,.28),(.63,.13,.33,.28)]
 axes=[fig.add_axes(p)for p in positions]
 fig.text(.02,.97,'I   Internal module controls: newly observed roots only',fontweight='bold',fontsize=9)
 fig.text(.02,.49,'II   Paired whole-field decoding: newly observed roots only',fontweight='bold',fontsize=9)
 for ax,letter,x,y in zip(axes,'ABCD',[.02,.53,.02,.53],[.92,.92,.44,.44]):fig.text(x,y,letter,fontweight='bold',fontsize=12)
 for ax,key in zip(axes[:2],['Precision','Recall']):
  for i,v in enumerate(order):
   rr=[r for r in refs if r['model']==v]if v=='V66_frozen'else[r for r in runs if r['variant']==v]
   x=np.array([number(r[key])*100 for r in rr if number(r[key])is not None]);c='#8B465A'if v=='full'else'#587C95'if v!='V66_frozen'else'#7C8389'
   if len(x):
    ax.errorbar(x.mean(),i,xerr=x.std(ddof=1)if len(x)>1 else 0,fmt='s'if v=='full'else'o',ms=4.5,capsize=2,color=c,zorder=4)
    if len(x)>1:ax.scatter(x,i+np.array([-.18,0,.18]),s=9,c='black',alpha=.65,zorder=3)
    ax.text(x.mean()+3.2,i,f'{x.mean():.2f}',va='center',fontsize=7.7,fontweight='bold'if v=='full'else'normal')
   else:ax.text(12,i,'Undefined*',fontsize=7.5,va='center')
  ax.set_yticks(range(6),names);ax.set_ylim(5.6,-.6);ax.set_xlim(0,110);ax.set_xticks([0,25,50,75,100]);ax.set_xlabel(key+' (%)');ax.set_title('New-root '+key.lower(),pad=7);ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0);ax.grid(axis='x',color='#ECEFF1',lw=.45);ax.set_axisbelow(True)
 stages=['raw_head','axis_constraints','axis_and_junction_constraints'];labels=['Raw head','Axis','Axis + junction']
 for ax,key in zip(axes[2:],['Precision','Recall']):
  m=np.array([[float(r[key])*100 for r in dec if r['stage']==s]for s in stages]);mean=m.mean(1);sd=m.std(1,ddof=1)
  for k in range(3):ax.plot(range(3),m[:,k],color='#AEB5BB',lw=.8,marker='o',ms=2.5)
  ax.errorbar(range(3),mean,yerr=sd,fmt='s-',color='#8B465A',capsize=3,ms=4.8,lw=1.4,zorder=4)
  for i,v in enumerate(mean):ax.text(i,v+sd[i]+3,f'{v:.2f}',ha='center',fontsize=8,fontweight='bold')
  ax.set(ylim=(0,105),xlim=(-.22,2.22));ax.set_xticks(range(3),labels,fontsize=7.3);ax.set_ylabel(key+' (%)');ax.set_title('Paired new-root '+key.lower(),pad=7);ax.grid(axis='y',color='#ECEFF1',lw=.45);ax.set_axisbelow(True)
 fig.text(.02,.038,'Mean ± optimization SD (3 seeds). *No positive predictions: precision is undefined. Internal B3 root-pixel evaluation.',fontsize=7)
 fig.text(.02,.012,'Rows use different inference contexts. Cross-tube precision and recall await human labels.',fontsize=7)
 export(fig,'New_root_precision_recall_SCI')
 # Historical B3 comparison, with a matched crop at all four dates.
 hist=read(W/'metrics/New_root_B3_full_image_historical_agreement.csv');fig=plt.figure(figsize=(180/25.4,166/25.4))
 axp=fig.add_axes((.11,.75,.35,.18));axr=fig.add_axes((.60,.75,.35,.18))
 for ax,key,letter,x in [(axp,'Precision','A',.02),(axr,'Recall','B',.51)]:
  fig.text(x,.97,letter,fontweight='bold',fontsize=12)
  for i,(model,name,c)in enumerate([('V66_frozen','V66','#71869A'),('V90_automatic_historical','V90','#BA8844')]):
   r=next(r for r in hist if r['model']==model and r['period']=='T2-T4_pooled');v=float(r[key])*100
   ax.bar(i,v,width=.47,color=c);ax.text(i,v+2.2,f'{v:.2f}',ha='center',fontweight='bold',fontsize=8.5)
  ax.set_xticks([0,1],['V66','V90']);ax.set_ylim(0,108);ax.set_yticks([0,25,50,75,100]);ax.set_ylabel(key+' (%)');ax.set_title('Newly observed roots',pad=7)
  ax.spines['bottom'].set_visible(False);ax.tick_params(axis='x',length=0);ax.grid(axis='y',color='#ECEFF1',lw=.4);ax.set_axisbelow(True)
 fig.text(.02,.68,'C',fontweight='bold',fontsize=12)
 # Crop selection uses reference-new coverage across dates, never model disagreement.
 masks=[np.array(Image.open(REF/(t+'_classes.png')))==2 for t in ['T2','T3','T4']]
 H,WW=masks[0].shape;cw,ch=560,320;best=None
 integral=[np.pad(m.astype(np.int64),((1,0),(1,0))).cumsum(0).cumsum(1)for m in masks]
 for yy in range(0,H-ch+1,24):
  for xx in range(0,WW-cw+1,24):
   counts=[int(z[yy+ch,xx+cw]-z[yy,xx+cw]-z[yy+ch,xx]+z[yy,xx])for z in integral]
   score=sum(np.log1p(c)for c in counts)
   if best is None or score>best[0]:best=(score,xx,yy,counts)
 _,x0,y0,counts=best;sl=np.s_[y0:y0+ch,x0:x0+cw]
 periods=['T1','T2','T3','T4'];left=.105;bottom=.10;width=.855;row_h=.126;gap=.008;col_w=(width-3*gap)/4
 sources=[None,REF,O/'model_source/assets/base/unet/full',O/'B3_verified_inference/junction/full'];rownames=['RGB','Human','V66','V90']
 for i,(src,label)in enumerate(zip(sources,rownames)):
  y=bottom+(3-i)*(row_h+.009)
  fig.text(.095,y+row_h/2,label,ha='right',va='center',fontsize=8,fontweight='bold')
  for j,t in enumerate(periods):
   rgb=np.array(Image.open(O/'model_source/assets/base/data/upper'/(t+'_RGB.png')).convert('RGB'))[sl]
   if src is not None:
    n=np.array(Image.open(src/(t+'_classes.png')))[sl]==2;rgb=rgb.copy();rgb[n]=np.rint(.92*COLOR+.08*rgb[n]).astype(np.uint8)
   ax=fig.add_axes((left+j*(col_w+gap),y,col_w,row_h));ax.imshow(rgb,aspect='auto');ax.set_axis_off()
   if i==0 and j==0:
    ax.plot([.045,.045+100/cw],[.10,.10],transform=ax.transAxes,color='white',lw=2.5)
    ax.text(.045,.15,'100 px',transform=ax.transAxes,fontsize=6.8,color='black',bbox=dict(facecolor='white',edgecolor='none',alpha=.9,pad=1))
   if i==0:ax.set_title(t,fontsize=8.5,fontweight='bold',pad=4)
   if t=='T1'and i>0:ax.text(.5,.05,'Baseline',transform=ax.transAxes,ha='center',va='bottom',fontsize=7,color='black',bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=1.2))
 fig.text(.105,.064,'Newly observed roots only',fontsize=7.7,fontweight='bold')
 fig.text(.02,.035,'B3 full-image historical human-label agreement; metrics use T2–T4 and include valid background. No error bars: one sequence.',fontsize=7)
 fig.text(.02,.012,'Same crop at all dates; crop selected from human new-root coverage. These are not independent-tube accuracy estimates.',fontsize=7)
 (W/'B3_crop_selection.json').write_text(json.dumps(dict(position='0353-2-B3',x=x0,y=y0,width=cw,height=ch,
  criterion='Maximum sum log(1 + human new-root pixels) over T2–T4, grid step 24; no prediction-quality selection',new_pixels_T2_T3_T4=counts),indent=2))
 export(fig,'B3_V66_V90_new_roots_only_SCI')
 print('NEW_ONLY_SCI_FIGURES_COMPLETE',flush=True)
if __name__=='__main__':main()
