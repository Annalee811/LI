#!/usr/bin/env python3
"""Three-model root-length figures from the unchanged v6 masks and paired data."""
import argparse,csv,json,hashlib,sys,zipfile
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import protocol as p
from figures import save,TEAL,DARK,GREY,panel

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'root_background_v5'))
from evaluate import length_field,agreement

MODELS={'unet':'U-Net','deeplab':'DeepLabV3+','segformer':'SegFormer-B0'}
MARKERS=['o','s','^','D']

def read(path):return json.loads(Path(path).read_text())

def table(path,rows):
 with path.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)

def fit(x,y):
 design=np.column_stack([x,np.ones_like(x)]);slope,intercept=np.linalg.lstsq(design,y,rcond=None)[0]
 fitted=design@np.array([slope,intercept]);sst=np.sum((y-y.mean())**2)
 r2=float(1-np.sum((y-fitted)**2)/sst)
 assert np.isclose(r2,np.corrcoef(x,y)[0,1]**2,atol=1e-12)
 return dict(OLS_slope=float(slope),OLS_intercept_px=float(intercept),R2_regression=r2,**agreement(x,y))

def scatter(ax,x,y,periods,stats,limit,title,label):
 for t,marker in zip(p.PERIODS,MARKERS):
  hit=np.array(periods)==t
  ax.scatter(x[hit],y[hit],s=22,c=TEAL,marker=marker,lw=.35,edgecolors='white',alpha=.88,zorder=3)
 line=np.linspace(0,limit,201)
 ax.plot(line,line,c=GREY,lw=.85,ls='--',zorder=1)
 ax.plot(line,stats['OLS_slope']*line+stats['OLS_intercept_px'],c=TEAL,lw=1,zorder=2)
 ax.set(xlim=(0,limit),ylim=(0,limit),xlabel='Human-mask skeleton length (px)',ylabel='Predicted skeleton length (px)',title=title)
 ax.set_aspect('equal',adjustable='box');ax.locator_params(axis='both',nbins=4)
 text='R² agreement = {:.3f}\nR² regression = {:.3f}\nRMSE = {:.1f} px\ny = {:.3f}x {:+.1f}'.format(stats['R2_agreement'],stats['R2_regression'],stats['RMSE_px'],stats['OLS_slope'],stats['OLS_intercept_px'])
 ax.text(.045,.965,text,transform=ax.transAxes,va='top',fontsize=6.3,linespacing=1.4,bbox=dict(facecolor='white',edgecolor='none',alpha=.88,pad=1))
 panel(ax,label)

def legends(fig):
 handles=[Line2D([],[],marker=m,ls='none',mfc=TEAL,mec='white',label=t) for t,m in zip(p.PERIODS,MARKERS)]
 handles += [Line2D([],[],c=TEAL,lw=1,label='OLS fit'),Line2D([],[],c=GREY,ls='--',lw=.85,label='Identity (y = x)')]
 fig.legend(handles=handles,loc='outside lower center',ncol=6,frameon=False,fontsize=6.5,columnspacing=1.1,handlelength=1.8)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
 root=args.run;out=args.output;out.mkdir(parents=True,exist_ok=True);figdir=out/'figures';figdir.mkdir(exist_ok=True)
 with (root/'root_length_pairs.csv').open(encoding='utf-8-sig') as f:rows=[r for r in csv.DictReader(f) if r['candidate'] in MODELS]
 stored=read(root/'root_length_agreement.json');points={};stats={};source_files={};checks=0
 for model in MODELS:
  for split in ['val','test']:
   rr=[r for r in rows if r['candidate']==model and r['split']==split]
   assert len(rr)==32 and len({(r['period'],r['block']) for r in rr})==32
   for t in p.PERIODS:
    folder=root/'candidates'/model/split
    human_path=folder/(t+'_human.png');pred_path=folder/(t+'_cleaned.png')
    for path in [human_path,pred_path]:source_files[str(path.relative_to(root))]=p.sha(path)
    human=np.array(Image.open(human_path))==255;pred=np.array(Image.open(pred_path))==255
    lf=length_field(human);lp=length_field(pred)
    for r in [r for r in rr if r['period']==t]:
     xi=int(r['block'][1]);yi=int(r['block'][3]);sl=(slice(yi*277,(yi+1)*277),slice(xi*224,(xi+1)*224))
     assert np.isclose(lf[sl].sum(),float(r['human_length_px']),atol=1e-8,rtol=0)
     assert np.isclose(lp[sl].sum(),float(r['predicted_length_px']),atol=1e-8,rtol=0);checks+=2
   x=np.array([float(r['human_length_px']) for r in rr]);y=np.array([float(r['predicted_length_px']) for r in rr])
   m=fit(x,y)
   for key,value in stored[model][split].items():
    if isinstance(value,(int,float)):assert np.isclose(m[key],value,rtol=0,atol=1e-10)
   points[model,split]=(x,y,[r['period'] for r in rr]);stats[model,split]=m
 # Models share exactly the same reference block/date pairs in each partition.
 for split in ['val','test']:
  for model in MODELS:assert np.array_equal(points[model,split][0],points['unet',split][0])
 summary=[dict(model=MODELS[m],partition=s,**stats[m,s]) for s in ['val','test'] for m in MODELS]
 table(out/'three_model_root_length_metrics.csv',summary);table(out/'three_model_root_length_pairs.csv',rows)
 limits={s:float(np.ceil(max(float(max(x.max(),y.max())) for (m,sp),(x,y,_) in points.items() if sp==s)*1.08/100)*100) for s in ['val','test']}
 for split in ['val','test']:
  fig,axes=plt.subplots(2,3,figsize=(7.2,5.35),layout='constrained')
  difference=np.concatenate([points[m,split][1]-points[m,split][0] for m in MODELS])
  lines=[stats[m,split][k] for m in MODELS for k in ['Bland_Altman_lower_px','Bland_Altman_upper_px']]
  low=min(difference.min(),min(lines));high=max(difference.max(),max(lines));pad=(high-low)*.1
  for j,model in enumerate(MODELS):
   x,y,periods=points[model,split];m=stats[model,split]
   scatter(axes[0,j],x,y,periods,m,limits[split],MODELS[model],chr(97+j))
   ax=axes[1,j]
   for t,marker in zip(p.PERIODS,MARKERS):
    hit=np.array(periods)==t;ax.scatter(((x+y)/2)[hit],(y-x)[hit],s=22,c=TEAL,marker=marker,lw=.35,edgecolors='white',alpha=.88)
   for key,style in [('mean_bias_px','-'),('Bland_Altman_lower_px','--'),('Bland_Altman_upper_px','--')]:ax.axhline(m[key],c=GREY,lw=.8,ls=style)
   ax.set(xlim=(0,limits[split]),ylim=(low-pad,high+pad),xlabel='Mean skeleton length (px)',ylabel='Prediction − reference (px)');panel(ax,chr(100+j));ax.locator_params(axis='both',nbins=4)
  legends(fig);save(fig,figdir/('Fig4_three_models_'+('validation' if split=='val' else 'repeated_C4')))
 for model in MODELS:
  fig,ax=plt.subplots(figsize=(3.2,3.35),layout='constrained');x,y,periods=points[model,'test']
  scatter(ax,x,y,periods,stats[model,'test'],limits['test'],MODELS[model],'a')
  handles=[Line2D([],[],marker=m,ls='none',mfc=TEAL,mec='white',label=t) for t,m in zip(p.PERIODS,MARKERS)]
  fig.legend(handles=handles,loc='outside lower center',ncol=4,frameon=False,fontsize=7);save(fig,figdir/(model+'_root_length_regression_repeated_C4'))
 # Preserve measured points; regression is descriptive and never changes predictions.
 table(out/'OLS_fitted_lines.csv',[dict(model=MODELS[m],partition=s,human_length_px=float(x),fitted_length_px=float(stats[m,s]['OLS_slope']*x+stats[m,s]['OLS_intercept_px'])) for s in ['val','test'] for m in MODELS for x in np.linspace(0,limits[s],201)])
 p.js(out/'verification.json',dict(passed=True,mask_length_checks=checks,reference_pairs_identical_across_models=True,
  existing_v6_agreement_statistics_reproduced=True,OLS_R2_equals_squared_Pearson=True,source_masks_SHA256=source_files,
  source_pairs_SHA256=p.sha(root/'root_length_pairs.csv'),source_length_statistics_SHA256=p.sha(root/'root_length_agreement.json'),
  fitted_lines_are_descriptive=True,models_retrained=False,forecasts_modified=False,confidence_intervals_or_P_values_added=False))
 rows_text='|模型|一致性 R²|回归拟合 R²|RMSE（原像素）|\n|---|---:|---:|---:|\n'
 for m in MODELS:
  v=stats[m,'test'];rows_text+=f"|{MODELS[m]}|{v['R2_agreement']:.4f}|{v['R2_regression']:.4f}|{v['RMSE_px']:.2f}|\n"
 description=f'''# 三模型根长回归与一致性补充图

补齐 U-Net、上传的 DeepLabV3+、SegFormer-B0 三模型的根长图。主图为相同重复空间留出 C4 的三模型回归散点及 Bland–Altman 综合图，另附同版验证集图与三张独立模型回归图。原v6模型、阈值、后处理和人工标注保持不变，没有重新拟合分割模型或挑选不同的有利样本。

{rows_text}
图中实线为普通最小二乘回归 y=ax+b；虚线为 y=x。两种 R² 明确分开：**一致性 R²＝1−Σ(预测−参考)²/Σ(参考−参考均值)²**，与之前的报告定义相同；**回归拟合 R²＝1−Σ(预测−拟合值)²/Σ(预测−预测均值)²**，等于带截距单变量线性回归的 Pearson r²。高相关性不等同于无偏一致，不能用较高的回归 R² 替换原一致性 R²。RMSE 始终比较原始预测长度与参考长度，不使用回归校正后的值。

参考根长由人工笔刷掩膜骨架计算，不是另行人工测量；单位为原始像素，无物理标尺。每模型每分区32个区块—时期对，来自8个相同空间区块×4时期，存在相关性，不是32个独立生物样本。C4为同位置已多次查看的探索性空间留出，不是新独立测试。回归线仅作描述，不添加独立样本显著性或置信区间。所有点均显示，长度和已有指标已从保存的掩膜重新核对。

综合图每分区内三模型共享坐标范围；Bland–Altman 图也共享纵轴。白底、青绿色、灰色，四期用点形区分；600dpi PNG、LZW TIFF、PDF和SVG。三张独立回归图均使用相同 C4 坐标。原始成对数据、拟合曲线数据、完整脚本和逐项核对记录同时保存。
'''
 (out/'README.md').write_text(description,encoding='utf-8')
 (out/'Figure_caption_EN.md').write_text('Root-length regression and agreement for U-Net, DeepLabV3+ and SegFormer-B0. Upper panels: all32 correlated block–date pairs in the same partition, ordinary least-squares line (solid), and identity line (dashed). R² agreement uses1−SSE/SST of the uncorrected prediction versus reference; R² regression describes the OLS fit and is reported separately. RMSE compares uncorrected prediction with reference. Lower panels: Bland–Altman differences with mean bias and descriptive±1.96 SD lines. Reference length is derived from supplied human brush-mask skeletons in native pixels, not independent manual length measurement. Eight spatial blocks recur over four dates. C4 is a repeatedly inspected within-position benchmark, not a new independent test. Identical axes are used across models within each partition. No points were removed or predictions altered.\n')
 (out/'README.md').write_text(description+'\n复现：在解压后的图包根目录，使用原v6绘图环境运行 `python source/paper_v6/three_model_regression.py --run run_snapshot --output /tmp/B3_regression_repeat/contents`。图包保留三模型二值掩膜及原始长度表，脚本先核对再绘图；不需要训练权重或重新训练。\n',encoding='utf-8')
 for f in [Path(__file__),Path(__file__).with_name('protocol.py'),Path(__file__).with_name('figures.py')]:
  source=out/'source/paper_v6'/f.name;source.parent.mkdir(parents=True,exist_ok=True);source.write_bytes(f.read_bytes())
 # One additional helper suffices for exact reference skeleton reconstruction.
 helper=out/'source/root_background_v5';helper.mkdir(exist_ok=True)
 for f in ['evaluate.py','common.py']:(helper/f).write_bytes((Path(__file__).resolve().parents[1]/'root_background_v5'/f).read_bytes())
 snapshot=out/'run_snapshot';snapshot.mkdir(exist_ok=True)
 for f in ['root_length_pairs.csv','root_length_agreement.json']:(snapshot/f).write_bytes((root/f).read_bytes())
 for f in source_files:
  dest=snapshot/f;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((root/f).read_bytes())
 archive=out.parent/'0353-2-B3_three_model_root_length_regression_v6.zip'
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for f in sorted(out.rglob('*')):
   if f.is_file():z.write(f,str(Path('0353-2-B3_three_model_root_length_regression_v6')/f.relative_to(out)))
 with zipfile.ZipFile(archive) as z:assert z.testzip() is None
 p.js(out.parent/'SHA256.json',dict(file=archive.name,bytes=archive.stat().st_size,SHA256=p.sha(archive),CRC_verified=True))
 print(rows_text,flush=True);print('REGRESSION_ZIP_VERIFIED',archive,archive.stat().st_size,p.sha(archive),flush=True)
if __name__=='__main__':main()
