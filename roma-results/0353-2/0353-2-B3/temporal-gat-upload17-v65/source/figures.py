from pathlib import Path
import colorsys,csv,hashlib,io,json,sys
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageCms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.text import Text
from prepare import O,R,BASE,PERIODS,MODELS,RGBROOT,js,sha
FONT=R/'results/B3_fixed_100_epochs_v35_20261023/assets/TimesNewRoman-Bold.ttf';NAMES={'unet':'U-Net','deeplab':'DeepLabV3+','segformer':'SegFormer-B0'};CLASS_NAMES={1:'Persistent roots',2:'Newly observed roots',3:'Growing root tips (extension)'};COLORS={1:(136,86,167),2:(0,158,115),3:(230,171,2)}

def text(draw,x,y,value,size=50,anchor='mt'):
 font=ImageFont.truetype(str(FONT),size);box=draw.textbbox((x,y),value,font=font,anchor=anchor);draw.text((x,y),value,font=font,fill='black',anchor=anchor);return box
def save(im,name):
 path=O/(name+'.tif');im.save(path,compression='tiff_adobe_deflate',tiffinfo={317:2},dpi=(600,600),icc_profile=ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes());preview=im.copy();preview.thumbnail((1500,1700));preview.save(O/(name+'_preview.png'))
 with Image.open(path) as q:assert q.mode=='RGB' and tuple(round(v) for v in q.info['dpi'])==(600,600)
 return dict(file=path.name,bytes=path.stat().st_size,SHA256=sha(path),dimensions=list(im.size),dpi=600)
def plot_save(fig,name):
 fig.canvas.draw()
 for t in fig.findobj(Text):t.set_fontproperties(FontProperties(fname=str(FONT),size=t.get_fontsize(),weight='bold'));t.set_color('black')
 bio=io.BytesIO();fig.savefig(bio,format='png',dpi=600,facecolor='white');plt.close(fig);bio.seek(0)
 with Image.open(bio) as im:return save(im.convert('RGB'),name)
def class_figures(geo):
 files=[];roi=[1440,120,1824,504];x,y,x1,y1=roi;sl=np.s_[y:y1,x:x1];panels=[]
 cols=[('reference','Reference')]+[(m+'_'+v,NAMES[m]+(' | Before' if v=='archived' else ' | +GAT')) for m in MODELS for v in ['archived','gat']]
 for k in [2,1,3]:
  im=Image.new('RGB',(7000,3800),'white');dr=ImageDraw.Draw(im);text(dr,3500,30,CLASS_NAMES[k]+': archived branch versus GAT refinement',92);text(dr,3500,145,'Same 384 × 384 px illustrative window | Frozen root outlines | Uploaded17 pixel reference',46)
  for j,(_,name) in enumerate(cols):text(dr,240+j*950+425,250,name,53)
  for ti,t in enumerate(PERIODS[1:]):
   yy=350+ti*1080;text(dr,90,yy+425,t,75,anchor='mm');rgb=np.array(Image.open(RGBROOT/(t+'_RGB.png')).convert('RGB'))[sl]
   for j,(key,_) in enumerate(cols):
    xx=240+j*950
    if key=='reference':cl=geo['gt_'+t][sl]
    elif key.endswith('_archived'):cl=np.array(Image.open(BASE/key[:-9]/'full'/(t+'_classes.png')))[sl]
    else:cl=np.array(Image.open(O/key/'full'/(t+'_classes.png')))[sl]
    out=rgb.copy();other=(cl>0)&(cl!=k);focus=cl==k;out[other]=np.rint(.72*rgb[other]+.28*np.array([145,145,145])).astype(np.uint8);out[focus]=np.rint(.12*rgb[focus]+.88*np.array(COLORS[k])).astype(np.uint8);im.paste(Image.fromarray(out).resize((850,850),Image.Resampling.LANCZOS),(xx,yy));dr.rectangle((xx,yy,xx+849,yy+849),outline='#d5dce2',width=2);panels.append(dict(class_code=k,period=t,method=key,ROI_xyxy=roi,class_pixels=int(focus.sum())))
  dr.rectangle((550,3600,610,3660),fill=COLORS[k]);text(dr,640,3600,'Color: '+CLASS_NAMES[k]+'; gray: other root classes',50,anchor='lt');text(dr,3500,3700,'Display window is illustrative; quantitative comparison uses the same held-out manual families.',45)
  name={1:'Fig_Persistent_roots_before_after_GAT',2:'Fig_New_roots_before_after_GAT',3:'Fig_Growing_tips_before_after_GAT'}[k];files.append(save(im,name))
 js(O/'class_figure_panels.json',panels);return files

def metrics_figures(summary):
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.8})
 rows=summary['temporal_classes'];fig,axs=plt.subplots(1,3,figsize=(11.5,4.2),sharey=True)
 for ax,k in zip(axs,[2,1,3]):
  for j,m in enumerate(MODELS):
   for off,v,col,label in [(-.18,'archived','#aeb6bf','Before'),(.18,'gat','#3a8397','+GAT')]:
    value=next(r['Dice'] for r in rows if r['model']==m and r['variant']==v and r['class_code']==k);ax.bar(j+off,value,.34,color=col,label=label if j==0 else None);ax.text(j+off,value+.018,f'{value:.3f}',ha='center',fontsize=8)
  ax.set(title=CLASS_NAMES[k],xticks=np.arange(3),xticklabels=[NAMES[m] for m in MODELS],ylim=(0,1.05),ylabel='Dice' if k==2 else '');ax.tick_params(axis='x',labelrotation=15);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);ax.legend(frameon=False,fontsize=8)
 fig.suptitle('Class-specific temporal agreement on held-out manually numbered root families',fontsize=12);fig.tight_layout(rect=(0,.02,1,.92));files=[plot_save(fig,'Fig_Three_class_Dice_before_after_GAT')]
 im=Image.new('RGB',(6000,3400),'white');dr=ImageDraw.Draw(im);text(dr,3000,35,'Three-class temporal results: same held-out root families',88);xs=[90,1450,2440,3300,4050,4800,5520];headers=['Class','Model','Variant','Precision','Recall','Dice','IoU'];yy=220
 for xx,h in zip(xs,headers):text(dr,xx,yy,h,60,anchor='lt')
 dr.line((70,yy+95,5910,yy+95),fill='black',width=6);yy+=125
 for k in [2,1,3]:
  for mi,m in enumerate(MODELS):
   for v in ['archived','gat']:
    row=next(r for r in rows if r['class_code']==k and r['model']==m and r['variant']==v);vals=[CLASS_NAMES[k] if mi==0 and v=='archived' else '',NAMES[m], 'Before' if v=='archived' else '+GAT',*[f"{row[key]:.4f}" for key in ['Precision','Recall','Dice','IoU']]]
    for xx,val in zip(xs,vals):text(dr,xx,yy,val,53,anchor='lt')
    yy+=135
  dr.line((70,yy-15,5910,yy-15),fill='#999999',width=2);yy+=30
 text(dr,3000,3205,'Reference: uploaded17 temporal colors on manually numbered roots; background errors included.',43);text(dr,3000,3280,'Single B3 field, model-assisted temporal reference; no independent external test.',43);files.append(save(im,'Table_Three_classes_before_after_GAT'))
 fig,axs=plt.subplots(2,3,figsize=(11,6.2))
 for j,m in enumerate(MODELS):
  with (O/(m+'_gat')/'history.csv').open(encoding='utf-8-sig') as fd:hist=list(csv.DictReader(fd))
  ep=[int(r['epoch']) for r in hist];assert ep==list(range(1,101));axs[0,j].plot(ep,[float(r['training_loss']) for r in hist],color='#0072b2',lw=1);axs[0,j].set(title=NAMES[m]+' +GAT',ylabel='Training objective' if j==0 else '',xlabel='GAT refinement epoch',xlim=(1,100));axs[1,j].plot(ep,[float(r['validation_macro_Dice']) for r in hist],color='#009e73',lw=1,label='Validation class Dice');axs[1,j].plot(ep,[float(r['validation_same_root_F1']) for r in hist],color='#d55e00',lw=1,label='Validation same-root F1');axs[1,j].set(ylabel='Validation score' if j==0 else '',xlabel='GAT refinement epoch',ylim=(0,1),xlim=(1,100));axs[1,j].legend(frameon=False,fontsize=7)
  for i in range(2):axs[i,j].grid(alpha=.18)
 fig.suptitle('100 real refinement epochs; archived image branches remain frozen',fontsize=12);fig.tight_layout(rect=(0,0,1,.95));files.append(plot_save(fig,'Fig_GAT_training_monitoring'))
 return files

def identity_figure(nodes,g,geo,source):
 first={(n['period'],n['segment_id']):n['node'] for n in reversed(nodes)};display_nodes=[]
 for f in source['root_structure_annotation']['periods']:
  for seg in f['segments']:
   nid=first[f['period'],seg['id']];display_nodes.append(dict(period=f['period'],points=seg['points'],node=nid,manual_label=nodes[nid]['manual_label']))
 mask=np.zeros((1108,2541),np.int32)
 for t in PERIODS:
  nm=geo['node_map_'+t];live=nm>=0;sel=np.zeros(nm.shape,bool);sel[live]=g['node_split'][nm[live]]==2;mask+=sel
 ii=np.pad(mask.cumsum(0).cumsum(1),((1,0),(1,0)));candidates=[];size=512
 for y in range(0,1108-size+1,48):
  for x in range(0,2541-size+1,48):candidates.append((int(ii[y+size,x+size]-ii[y,x+size]-ii[y+size,x]+ii[y,x]),x,y))
 _,x,y=max(candidates);roi=[x,y,x+512,y+512];hs={h['root_id']:h for h in source['root_structure_annotation']['root_track_hierarchy']};family={h['label']:hs[h['family_root_id']]['label'] for h in hs.values() if h.get('annotation_status_v60')=='user_defined'};testnames={n['manual_label'] for n in nodes if n['root_target'] and g['node_split'][n['node']]==2};names={}
 for m in MODELS:
  registry=json.loads((O/(m+'_gat')/'track_registry.json').read_text());tr=np.load(O/(m+'_gat')/'node_track.npy');names[m]=[registry[int(c)]['display_label'] for c in tr]
 def color(label):
  if not label or label.startswith('P-'):return (150,150,150)
  key=family.get(label,label);h=int(hashlib.sha256(key.encode()).hexdigest()[:8],16);return tuple(int(v*255) for v in colorsys.hsv_to_rgb((h*.6180339887)%1,.78,.9))
 im=Image.new('RGB',(6200,6500),'white');dr=ImageDraw.Draw(im);text(dr,3100,40,'Root identity comparison: human numbering and learned GAT tracks',86);text(dr,3100,155,'First-observation names are anchors; later correspondence is predicted. Centerlines are not width measurements.',42)
 cols=[('human','Human IDs')]+[(m,NAMES[m]+' +GAT') for m in MODELS];z=2.55;tile=round(512*z)
 for j,(_,name) in enumerate(cols):text(dr,240+j*1480+tile/2,260,name,65)
 label_count=0
 for ti,t in enumerate(PERIODS):
  yy=380+ti*1470;text(dr,100,yy+tile/2,t,80,anchor='mm');rgb=Image.open(RGBROOT/(t+'_RGB.png')).convert('RGB').crop(tuple(roi)).resize((tile,tile),Image.Resampling.LANCZOS)
  for j,(m,_) in enumerate(cols):
   xx=240+j*1480;image=rgb.copy();draw=ImageDraw.Draw(image);groups={}
   for n in display_nodes:
    if n['period']!=t:continue
    pts=[p for p in n['points'] if x<=p[0]<x+512 and y<=p[1]<y+512]
    if not pts:continue
    label=n['manual_label'] if m=='human' else names[m][n['node']];draw.line([((p[0]-x)*z,(p[1]-y)*z) for p in n['points']],fill=color(label),width=4 if label and not label.startswith('P-') else 2)
    if label and (not label.startswith('P-') or n['manual_label']):groups.setdefault(label,[]).append((n,pts))
   occupied=[];font=ImageFont.truetype(str(FONT),31)
   for label,rows in groups.items():
    n,pts=max(rows,key=lambda v:len(v[1]));p=pts[len(pts)//2];w=draw.textlength(label,font=font)+14;h=39;lx=max(1,min(tile-w-1,(p[0]-x)*z));ly=max(1,min(tile-h-1,(p[1]-y)*z-42))
    if any(lx<a+c and lx+w>a and ly<b+d and ly+h>b for a,b,c,d in occupied):continue
    occupied.append((lx,ly,w,h));draw.rectangle((lx,ly,lx+w,ly+h),fill='white',outline=color(label),width=3);draw.text((lx+6,ly+2),label,font=font,fill='black');label_count+=1
   im.paste(image,(xx,yy));dr.rectangle((xx,yy,xx+tile-1,yy+tile-1),outline='#d6dce1',width=2)
 text(dr,3100,6335,'Names and fragments for every root are available in the interactive viewer and per-root CSV.',43);text(dr,3100,6410,'Window selected from held-out root geometry only; not selected by model accuracy. Unnamed tracks remain P-xxxx.',42)
 js(O/'ID_figure_ROI.json',dict(ROI_xyxy=roi,selection='Maximum held-out manually numbered root pixels across four periods; no model predictions used',labels_drawn=label_count));return save(im,'Fig_Root_ID_comparison')

def main():
 assert (O/'evaluation_complete.json').exists();summary=json.loads((O/'metrics_summary.json').read_text());geo=np.load(O/'data/geometry.npz');g={k:v for k,v in np.load(O/'data/graph.npz').items()};nodes=json.loads((O/'data/nodes.json').read_text());source=json.loads((O/'data/source_progress_17.json').read_text());files=class_figures(geo)+metrics_figures(summary)+[identity_figure(nodes,g,geo,source)];js(O/'figure_manifest.json',dict(passed=True,figures=files,typography='Times New Roman Bold, all text black',dpi=600));print('FIGURES_COMPLETE',len(files),flush=True)
if __name__=='__main__':main()
