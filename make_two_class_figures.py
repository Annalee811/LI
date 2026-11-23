from pathlib import Path
import argparse,csv,json,os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.collections import PolyCollection
W=Path(__file__).resolve().parent
def read(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def num(v):return None if v==''else float(v)
def save(fig,stem):
 fig.canvas.draw();r=fig.canvas.get_renderer()
 for t in fig.findobj(matplotlib.text.Text):
  if not t.get_visible()or not t.get_text():continue
  b=t.get_window_extent(r)
  assert b.x0>=-1 and b.y0>=-1 and b.x1<=fig.bbox.width+1 and b.y1<=fig.bbox.height+1,('Text clipped',stem,t.get_text())
 for ext in ['png','jpg','pdf','svg','tif']:
  kw=dict(dpi=600,facecolor='white')
  if ext=='jpg':kw['pil_kwargs']=dict(quality=98,subsampling=0)
  if ext=='tif':kw['pil_kwargs']=dict(compression='tiff_lzw')
  fig.savefig(W/(stem+'.'+ext),**kw)
 fig.savefig(W/(stem+'_preview.png'),dpi=170,facecolor='white');plt.close(fig)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--fonts',type=Path,default=Path('/workspace/work/root_workflow_real_assets_20261109/fonts'));a=ap.parse_args()
 for f in ['Times New Roman.ttf','Times New Roman Bold.ttf']:font_manager.fontManager.addfont(str(a.fonts/f))
 plt.rcParams.update({'font.family':'Times New Roman','font.size':8,'axes.labelsize':8.5,'axes.labelweight':'bold','axes.titlesize':10,'axes.titleweight':'bold',
  'text.color':'black','axes.labelcolor':'black','xtick.color':'black','ytick.color':'black','axes.edgecolor':'black','axes.linewidth':.65,
  'xtick.labelsize':7.7,'ytick.labelsize':7.7,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
 colors={'new':'#C58D31','existing':'#3F7091'};spec=[('new','New roots','o'),('existing','Existing roots','s')]
 for csvname,stem,label in [('Series_mean_adjacent.csv','V90_three_series_two_class_rates_SCI','True adjacent intervals'),('Series_mean_previous_observed.csv','V90_two_class_previous_observed_intervals','Previous observed intervals (includes gaps)')]:
  rows=read(W/csvname);fig,axes=plt.subplots(1,3,figsize=(190/25.4,87/25.4),sharey=True)
  fig.subplots_adjust(left=.075,right=.99,bottom=.25,top=.86,wspace=.18)
  for i,(material,ax)in enumerate(zip(['0353','A017','Q1'],axes)):
   rr=[r for r in rows if r['material']==material and r['period']!='T1'];x=np.array([int(r['period'][1:])for r in rr])
   for key,title,marker in spec:
    mean=np.array([float(r[key+'_mean_pct'])for r in rr]);sd=np.array([num(r[key+'_SD_pct'])or 0 for r in rr])
    ax.fill_between(x,np.maximum(0,mean-sd),np.minimum(100,mean+sd),color=colors[key],alpha=.13,lw=0)
    ax.plot(x,mean,color=colors[key],marker=marker,ms=4.3,lw=1.6,label=title)
    other=np.array([float(r[('existing'if key=='new'else'new')+'_mean_pct'])for r in rr])
    for xx,yy,oo in zip(x,mean,other):
     dy=6 if yy>=oo else-7
     if yy>91:dy=-7
     if yy<9:dy=6
     ax.annotate(f'{yy:.1f}',(xx,yy),xytext=(0,dy),textcoords='offset points',ha='center',va='bottom'if dy>0 else'top',fontsize=7.6,fontweight='bold')
   ax.set(xlim=(.8,4.18),ylim=(0,100));ax.set_xticks([1,2,3,4],['T1','T2','T3','T4']);ax.set_yticks([0,25,50,75,100]);ax.set_xlabel('Period',labelpad=3);ax.set_title(material,pad=7)
   ax.text(-.16,1.11,'ABC'[i],transform=ax.transAxes,fontweight='bold',fontsize=12)
   ax.text(1,50,'Baseline',rotation=90,ha='center',va='center',fontsize=7.3)
   ax.grid(axis='y',color='#E9ECEF',lw=.5);ax.set_axisbelow(True)
   ns=', '.join(r['n_positions']for r in rr);ax.text(.5,-.30,'Fields (T2–T4): '+ns,transform=ax.transAxes,ha='center',fontsize=7.1)
  axes[0].set_ylabel('Predicted length share (%)',labelpad=4)
  handles=[Line2D([],[],color=colors[k],lw=1.6,marker=m,ms=4,label=n)for k,n,m in spec]
  fig.legend(handles=handles,loc='center',bbox_to_anchor=(.54,.10),ncol=2,frameon=False,prop={'weight':'bold','size':8.3},columnspacing=2)
  fig.text(.075,.018,label+'; mean ± field SD. Existing = persistent + extension. Descriptive model predictions.',fontsize=7.1)
  save(fig,stem)
 # A compact two-layer version matching the user's earlier filled 3-D layout.
 rows=read(W/'Series_mean_adjacent.csv');fig=plt.figure(figsize=(210/25.4,100/25.4))
 for i,material in enumerate(['0353','A017','Q1']):
  ax=fig.add_axes((.025+i*.325,.19,.275,.67),projection='3d');ax.set_proj_type('ortho');ax.view_init(elev=24,azim=-58);ax.set_box_aspect((1.25,.70,.9))
  rr=[r for r in rows if r['material']==material and r['period']!='T1'];xx=np.array([2,3,4])
  for depth,(key,label,marker)in enumerate(spec):
   zz=np.array([float(r[key+'_mean_pct'])for r in rr]);poly=PolyCollection([[(2,0),*zip(xx,zz),(4,0)]],facecolors=[colors[key]],edgecolors='none',alpha=.55)
   ax.add_collection3d(poly,zs=depth,zdir='y');ax.plot(xx,np.full(3,depth),zz,color='black',ls=(0,(3,2)),marker=marker,mfc='white',ms=3.4,lw=.8)
   for x,z in zip(xx,zz):ax.text(x,depth,min(100,z+4),f'{z:.1f}',ha='center',fontsize=8,fontweight='bold')
  ax.set(xlim=(1,4.2),ylim=(-.2,1.25),zlim=(0,105));ax.set_xticks([1,2,3,4],['T1','T2','T3','T4']);ax.set_yticks([0,1],['New','Existing']);ax.set_zticks([0,25,50,75,100]);ax.set_xlabel('Period',fontsize=8.5,fontweight='bold',labelpad=1)
  ax.tick_params(axis='x',pad=0,labelsize=7.6);ax.tick_params(axis='y',pad=0,labelsize=7.6);ax.tick_params(axis='z',pad=1,labelsize=7.6)
  for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:axis.set_pane_color((1,1,1,1));axis._axinfo['grid'].update(color=(.85,.87,.89,1),linewidth=.45)
  fig.text(.165+i*.325,.92,material,ha='center',fontsize=10,fontweight='bold');fig.text(.02+i*.325,.92,'ABC'[i],fontsize=12,fontweight='bold')
 fig.text(.006,.54,'Mean length share (%)',rotation=90,fontsize=8.5,fontweight='bold',va='center')
 fig.legend(handles=[Patch(facecolor=colors[k],alpha=.7,label=l)for k,l,_ in spec],loc='center',bbox_to_anchor=(.53,.10),ncol=2,frameon=False,prop={'size':8.5,'weight':'bold'})
 fig.text(.03,.022,'V90-derived predictions; actual adjacent intervals only. T1 unassigned. Filled polygons are display closures, not extra observations.',fontsize=7)
 # 3-D tick extents are not reliable before projection; the full saved canvas is inspected visually.
 for ext in ['png','jpg','pdf','svg','tif']:
  kw=dict(dpi=600,facecolor='white')
  if ext=='jpg':kw['pil_kwargs']=dict(quality=98,subsampling=0)
  if ext=='tif':kw['pil_kwargs']=dict(compression='tiff_lzw')
  fig.savefig(W/('V90_three_series_two_class_3D_SCI.'+ext),**kw)
 fig.savefig(W/'V90_three_series_two_class_3D_SCI_preview.png',dpi=170,facecolor='white');plt.close(fig)
 (W/'Figure_metadata.json').write_text(json.dumps(dict(primary_mm=[190,87],primary_source='Series_mean_adjacent.csv',font='Black Times New Roman',raster_dpi=600,
  root_groups=['new','existing'],new_label_unchanged=True,existing_is_persistent_plus_extension=True,uncertainty='Shading is field SD clipped to 0–100 for display; full numeric SD retained',
  T1_assigned_rate=False,rate_denominator='Current predicted root skeleton length',no_independent_accuracy_claim=True),indent=2))
 print('TWO_CLASS_FIGURES_COMPLETE',flush=True)
if __name__=='__main__':main()
