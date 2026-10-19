"""Reference/prediction/absolute-error 2x2 figure from final native test data."""
import argparse,csv,hashlib,json,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image

MODELS=['unet','deeplab','segformer']
NAMES=['U-Net','DeepLabV3+','SegFormer-B0']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def csvout(path,rows):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    source=a.run/'revised_report/tables/root_length_pairs.csv';a.out.mkdir(parents=True,exist_ok=True)
    with source.open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    keys=sorted({r['image_id'] for r in rows});assert len(keys)==24
    groups=sorted({r['tile_id'] for r in rows});assert len(groups)==6
    assert all(sum(r['model']==m for r in rows)==24 for m in MODELS)
    references=[];predictions=[];errors=[];summaries=[];export=[];weights=[]
    for model,name in zip(MODELS,NAMES):
        rr={r['image_id']:r for r in rows if r['model']==model};assert set(rr)==set(keys)
        ref=np.array([float(rr[k]['reference_mask_skeleton_length_px']) for k in keys]);pred=np.array([float(rr[k]['predicted_skeleton_length_px']) for k in keys]);ae=abs(pred-ref)
        assert np.isfinite(ref).all() and np.isfinite(pred).all()
        references.append(ref);predictions.append(pred);errors.append(ae)
        summaries.append(dict(model=model,name=name,n_samples=24,MAE_px=float(ae.mean()),RMSE_px=float(np.sqrt(np.mean(ae**2))),median_AE_px=float(np.median(ae))))
        sel=json.loads((a.run/model/'selection.json').read_text());ck=a.run/model/('best_weights_fp16.npz' if model=='deeplab' else 'best.pt')
        assert sha(ck)==sel['checkpoint_SHA256']
        training=json.loads((a.run/model/'training_run.json').read_text());assert training['epochs_completed']==20 and training['unique_training_images']==200
        weights.append(dict(model=model,selected_epoch=sel['selected_epoch'],checkpoint_SHA256=sel['checkpoint_SHA256']))
        for index,k in enumerate(keys,1):export.append(dict(model=model,sample_index=index,image_id=k,tile_id=rr[k]['tile_id'],period=rr[k]['period'],reference_length_px=float(ref[index-1]),predicted_length_px=float(pred[index-1]),absolute_error_px=float(ae[index-1])))
    assert all(np.array_equal(references[0],r) for r in references)
    # Check against the final run's published table, not any older model run.
    with (a.run/'revised_report/tables/root_length_agreement.csv').open(encoding='utf-8-sig') as f:recorded={r['model']:r for r in csv.DictReader(f)}
    for r in summaries:assert np.isclose(r['MAE_px'],float(recorded[r['model']]['MAE_px']),atol=1e-10,rtol=0)
    csvout(a.out/'prediction_error_data.csv',export);csvout(a.out/'model_MAE_summary.csv',summaries)
    shutil.copyfile(source,a.out/'source_root_length_pairs.csv')
    plt.rcParams.update({'font.family':'Liberation Serif','font.size':10,'axes.titlesize':11,'axes.labelsize':10,
        'xtick.labelsize':9,'ytick.labelsize':9,'axes.linewidth':.65,'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
    fig,axs=plt.subplots(2,2,figsize=(8.8,6.5));fig.subplots_adjust(left=.085,right=.91,bottom=.16,top=.94,wspace=.35,hspace=.36)
    x=np.arange(1,25);ylim=max(max(r.max() for r in references),max(p.max() for p in predictions))*1.14
    emax=max(e.max() for e in errors)*1.12
    for i,(ax,name,ref,pred,ae) in enumerate(zip(axs.flat,NAMES,references,predictions,errors)):
        right=ax.twinx();right.set_zorder(0);ax.set_zorder(1);ax.patch.set_visible(False)
        # Lines join dates ONLY within a tile. No implied continuity across tiles.
        for group in groups:
            idx=np.array([j for j,k in enumerate(keys) if k.startswith(group+'_')]);xx=x[idx]
            right.fill_between(xx,0,ae[idx],color='#AAAAAA',alpha=.20,lw=0)
            right.plot(xx,ae[idx],color='#999999',lw=.75,alpha=.8)
            ax.plot(xx,ref[idx],color='#E52020',lw=1.25)
            ax.plot(xx,pred[idx],color='#164DDE',lw=1.25)
        right.axhline(ae.mean(),color='#D9AE00',lw=1.25,ls=(0,(5,3)))
        for boundary in [4.5,8.5,12.5,16.5,20.5]:ax.axvline(boundary,color='#DDDDDD',lw=.6,ls=':',zorder=0)
        ax.set(xlim=(.5,24.5),ylim=(0,ylim),xticks=[1,4,8,12,16,20,24],xlabel='Sample index',ylabel='Root length (px)',title='('+chr(97+i)+') '+name)
        right.set(ylim=(0,emax),ylabel='Absolute error (px)');right.tick_params(axis='y',colors='#777777');right.yaxis.label.set_color('#777777')
        for spine in right.spines.values():spine.set_linewidth(.55);spine.set_color('#B0B0B0')
        ax.grid(axis='y',color='#E8E8E8',lw=.55);ax.set_axisbelow(True)
        ax.text(.025,.94,'MAE = %.2f px'%ae.mean(),transform=ax.transAxes,va='top',fontsize=9,color='#555555',bbox=dict(facecolor='white',edgecolor='none',alpha=.75,pad=1.3))
        ax.legend(handles=[Line2D([],[],color='#E52020',lw=1.25,label='Reference'),Line2D([],[],color='#164DDE',lw=1.25,label='Prediction')],loc='upper right',frameon=True,framealpha=.9,fontsize=8.5,borderpad=.35)
    ax=axs[1,1];means=np.array([r['MAE_px'] for r in summaries]);colors=['#2760BF','#6995D6','#B1CAE9']
    ax.bar(np.arange(3),means,color=colors,width=.58,edgecolor='#526A89',lw=.65,zorder=2)
    offsets=np.linspace(-.19,.19,24)
    for i,ae in enumerate(errors):
        ax.scatter(i+offsets,ae,s=10,facecolors='white',edgecolors='#555555',lw=.45,alpha=.75,zorder=3)
        ax.text(i,means[i]+emax*.035,'%.2f'%means[i],ha='center',va='bottom',fontsize=9,fontweight='bold',bbox=dict(facecolor='white',edgecolor='none',alpha=.8,pad=.8))
    ax.set(xticks=range(3),xticklabels=NAMES,ylim=(0,emax),ylabel='Absolute error (px)',title='(d) MAE comparison')
    ax.grid(axis='y',color='#E8E8E8',lw=.55);ax.set_axisbelow(True)
    ax.legend(handles=[Line2D([],[],marker='o',mfc='white',mec='#555555',lw=0,ms=3,label='Individual samples')],loc='upper left',frameon=False,fontsize=8.5)
    fig.legend(handles=[Line2D([],[],color='#999999',lw=1,label='Absolute error'),Line2D([],[],color='#D9AE00',lw=1.25,ls=(0,(5,3)),label='Mean absolute error (MAE)')],loc='lower center',bbox_to_anchor=(.5,.058),ncol=2,frameon=False,fontsize=9)
    fig.text(.5,.027,'24 paired samples = 6 test tiles × 4 dates; gaps separate different tiles.',ha='center',fontsize=9,color='#555555')
    stem='Fig_prediction_MAE_2x2'
    fig.savefig(a.out/(stem+'.png'),dpi=600);fig.savefig(a.out/(stem+'.pdf'));fig.savefig(a.out/(stem+'.svg'))
    with Image.open(a.out/(stem+'.png')) as im:im.save(a.out/(stem+'.tif'),compression='tiff_lzw',dpi=(600,600))
    plt.close(fig)
    caption='Reference and predicted root lengths with absolute errors for the three final models on the same24test tile-date samples. Red:reference; blue:prediction; gray:per-sample absolute error(right axis); dashed yellow:MAE. Lines connect dates within the same tile only; gaps separate tiles. The fourth panel compares MAE,with dots showing individual absolute errors. Lengths are mask-skeleton lengths in native pixels; reference definitions follow the final experiment Methods. Samples are correlated observations,not independent biological replicates.'
    (a.out/'Figure_caption_EN.txt').write_text(caption+'\n')
    (a.out/'README.md').write_text('# 三模型预测与MAE：2×2对比图\n\n使用最后一轮三个模型的数据：200训练视图、20验证、24内部测试，三个模型均微调20轮。\n\n前三格：红色参考根长、蓝色预测根长，灰色逐样本绝对误差AE（右轴），黄色虚线为该模型MAE。第四格：三模型MAE与各样本绝对误差。\n\n横轴为样本编号，按同一图块T1–T4排列；不同图块之间断开，不能把它当作连续时间步。长度来自参考/预测掩膜骨架，单位像素。参考构建与来源沿用最终实验的方法说明，不是额外取得的独立人工尺量。\n\n这是根长预测误差图，不是训练损失图；原S1训练曲线另外保留。数据、指标、模型权重均未更改。附600dpiTIFF/PNG、PDF/SVG、CSV和完整绘图脚本。\n')
    provenance=dict(source_run=str(a.run),source_pairs_SHA256=sha(source),final_model_weights=weights,only_final_models=True,
        sample_count=24,original_source_images=4,source_location_count=1,reference_is_independent_manual_truth=False,
        reference_provenance='../annotation-refinement-v1/revision_provenance.json',measures='Mask-skeleton lengths in native pixels',
        gray_curve='Absolute error per sample, not moving MAE',yellow_line='Arithmetic mean absolute error over24samples',
        not_a_learning_curve=True,metrics_match_final_run=True,summaries=summaries)
    (a.out/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summaries,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
