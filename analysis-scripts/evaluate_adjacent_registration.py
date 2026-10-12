#!/usr/bin/env python3
"""Audit saved RoMa adjacent registrations; independent RootSIFT validation.

No registration field is refitted. SIFT acceptance uses descriptors, mutual ratio
matching, a similarity RANSAC and spatial thinning, never RoMa residual/confidence.
Errors are automatic validation-point errors, not human landmark TRE.
"""
from pathlib import Path
import csv, json, shutil, hashlib, gc, argparse
import cv2
import numpy as np
from PIL import Image

REPO = Path.cwd().resolve()
ROOT = REPO / "results/microtube_registration"
PAIRS = ['T1_to_T2', 'T2_to_T3', 'T3_to_T4']
RULES = dict(min_effective_target_fraction=.10, min_roma_reliable_samples=100,
             min_automatic_validation_points=20, max_automatic_median_px_at_width1400=3.,
             max_confident_cycle_p95_target_px=3.)
COLORS = np.array([[210,210,210], [48,154,96], [232,162,47], [150,112,170]], np.uint8)


def rgb(path):
    with Image.open(path) as im: return np.array(im.convert('RGB'))


def sample(array, points, nearest=False):
    return cv2.remap(array, points[None,:,0].astype(np.float32),
                     points[None,:,1].astype(np.float32),
                     cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=-1)[0]


def stats(error):
    if not len(error): return dict(n=0, mean=None, median=None, rmse=None, p95=None)
    return dict(n=len(error), mean=float(error.mean()), median=float(np.median(error)),
                rmse=float(np.sqrt(np.mean(error**2))), p95=float(np.percentile(error,95)))


def features(path):
    a=rgb(path); h,w=a.shape[:2]; nw=1400; nh=round(nw*h/w)
    a=cv2.resize(a,(nw,nh),interpolation=cv2.INTER_AREA if w>nw else cv2.INTER_CUBIC)
    gray=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY)
    sift=cv2.SIFT_create(nfeatures=0,contrastThreshold=.02,edgeThreshold=10)
    keys,d=sift.detectAndCompute(gray,None)
    d=np.sqrt(d/(d.sum(1,keepdims=True)+1e-7))
    return keys,d,np.array([w/nw,h/nh]),(nw,nh)


def independent_points(meta):
    ka,da,sa,_=features(REPO/meta['source_file'])
    kb,db,sb,analysis_size=features(REPO/meta['target_file'])
    matcher=cv2.BFMatcher(cv2.NORM_L2)
    ab=matcher.knnMatch(da,db,2);ba=matcher.knnMatch(db,da,2)
    reverse={m.queryIdx:m.trainIdx for m,n in ba if m.distance<.75*n.distance}
    matches=[m for m,n in ab if m.distance<.75*n.distance and reverse.get(m.trainIdx)==m.queryIdx]
    pa=np.array([ka[m.queryIdx].pt for m in matches],np.float32)
    pb=np.array([kb[m.trainIdx].pt for m in matches],np.float32)
    if len(matches)<4:
        return np.empty((0,2)),np.empty((0,2)),dict(mutual_matches=len(matches),ransac_inliers=0),sb
    affine,mask=cv2.estimateAffinePartial2D(pa,pb,method=cv2.RANSAC,
        ransacReprojThreshold=2.5,maxIters=20000,confidence=.999,refineIters=10)
    good=mask.ravel().astype(bool) if mask is not None else np.zeros(len(matches),bool)
    ids=[]
    # Prefer descriptor distance; spatial thinning uses only observed target points.
    for j in sorted(np.flatnonzero(good),key=lambda j:matches[j].distance):
        if not ids or np.linalg.norm(pb[j]-pb[ids],axis=1).min()>=12: ids.append(j)
    return pa[ids]*sa,pb[ids]*sb,dict(source_keypoints=len(ka),target_keypoints=len(kb),
        mutual_matches=len(matches),ransac_inliers=int(good.sum()),
        spatially_thinned_points=len(ids),analysis_target_size_wh=list(analysis_size),
        similarity_in_analysis_coordinates=affine.tolist() if affine is not None else None),sb


def save_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def evaluate(name,source_root,out):
    folder=source_root/name;dest=out/name;dest.mkdir(parents=True,exist_ok=True)
    meta=json.loads((folder/'metrics.json').read_text())
    for filename in ['registered_rgb.png','overlay_50pct.png','checkerboard.png',
                     'confidence_16bit.png','valid_overlap_mask.png','geometric_overlap_mask.png']:
        shutil.copy2(folder/filename,dest/filename)
    geometry=np.array(Image.open(folder/'geometric_overlap_mask.png'))>0
    valid=np.array(Image.open(folder/'valid_overlap_mask.png'))>0
    confidence=np.load(folder/'confidence_float32.npy',mmap_mode='r')
    low=geometry&(confidence<.5)
    other=geometry&~valid&~low
    total=int(geometry.size);g=int(geometry.sum());e=int(valid.sum());l=int(low.sum())
    assert not (valid&~geometry).any() and not (valid&low).any()
    assert e+l+int(other.sum())+int((~geometry).sum())==total
    Image.fromarray(low.astype(np.uint8)*255).save(dest/'low_confidence_mask.png')
    state=np.zeros(geometry.shape,np.uint8);state[valid]=1;state[low]=2;state[other]=3
    Image.fromarray(COLORS[state]).save(dest/'overlap_quality_map.png')
    with np.load(folder/'dense_transform.npz') as fields:
        forward=fields['source_to_target_xy_px']
        pa,pb,match_info,sb=independent_points(meta)
        pred=sample(forward,pa);errors=np.linalg.norm(pred-pb,axis=1)
        normalized_errors=np.linalg.norm((pred-pb)/sb,axis=1)
        valid_points=sample(valid.astype(np.uint8),pb,True)>0
        inside=(np.isfinite(pred).all(1)&(pred[:,0]>=0)&(pred[:,0]<=meta['target_size_wh'][0]-1)&
                (pred[:,1]>=0)&(pred[:,1]<=meta['target_size_wh'][1]-1))
        # All independently accepted points enter the headline error, including outliers.
        all_stats=stats(errors);effective_stats=stats(errors[valid_points]);norm_stats=stats(normalized_errors)
        if len(pa):
            point_rows=[]
            for k in range(len(pa)):
                point_rows.append(dict(point_id=k+1,source_x=float(pa[k,0]),source_y=float(pa[k,1]),
                    observed_target_x=float(pb[k,0]),observed_target_y=float(pb[k,1]),
                    roma_predicted_target_x=float(pred[k,0]),roma_predicted_target_y=float(pred[k,1]),
                    error_target_px=float(errors[k]),error_at_target_width1400=float(normalized_errors[k]),
                    in_effective_overlap=bool(valid_points[k]),prediction_in_target_bounds=bool(inside[k])))
            save_csv(dest/'automatic_validation_points.csv',point_rows)
    checks=dict(effective_area=e/total>=RULES['min_effective_target_fraction'],
                reliable_sample_count=meta['reliable_sampled_matches']>=RULES['min_roma_reliable_samples'],
                independent_point_count=all_stats['n']>=RULES['min_automatic_validation_points'],
                independent_point_error=norm_stats['median'] is not None and norm_stats['median']<=RULES['max_automatic_median_px_at_width1400'],
                confident_cycle=meta['cycle_error_confident_target_px']['p95']<=RULES['max_confident_cycle_p95_target_px'])
    row=dict(pair=name.replace('_to_','→'),target_width=meta['target_size_wh'][0],target_height=meta['target_size_wh'][1],
        automatic_qc_pass=all(checks.values()),human_landmark_TRE='未评估（无人工标志点）',
        automatic_validation_points=all_stats['n'],automatic_point_mean_target_px=all_stats['mean'],
        automatic_point_median_target_px=all_stats['median'],automatic_point_rmse_target_px=all_stats['rmse'],
        automatic_point_p95_target_px=all_stats['p95'],automatic_point_median_at_target_width1400=norm_stats['median'],
        geometric_overlap_pixels=g,geometric_overlap_pct_target=100*g/total,
        effective_overlap_pixels=e,effective_overlap_pct_target=100*e/total,
        low_confidence_overlap_pixels=l,low_confidence_overlap_pct_target=100*l/total,
        low_confidence_pct_geometric_overlap=100*l/g if g else None,
        low_confidence_full_canvas_pixels=int((confidence<.5).sum()),
        low_confidence_full_canvas_pct=100*float((confidence<.5).mean()),
        other_unreliable_overlap_pixels=int(other.sum()),other_unreliable_overlap_pct_target=100*float(other.mean()),
        nonoverlap_pixels=total-g,nonoverlap_pct_target=100*(total-g)/total,
        reliable_roma_samples=meta['reliable_sampled_matches'],sampled_roma_matches=meta['sampled_matches'],
        cycle_median_effective_target_px=meta['cycle_error_effective_target_px']['median'],
        cycle_p95_effective_target_px=meta['cycle_error_effective_target_px']['p95'])
    details=dict(summary=row,automatic_qc_checks=checks,independent_matching=match_info,
                 automatic_validation_all_points=all_stats,
                 automatic_validation_points_in_effective_area=effective_stats,
                 automatic_validation_width1400=norm_stats,
                 roma_original_metrics=meta)
    (dest/'metrics.json').write_text(json.dumps(details,ensure_ascii=False,indent=2))
    print(json.dumps(row,ensure_ascii=False),flush=True)
    gc.collect()
    return row,details


def create_report(out,rows,report):
    import os
    os.environ.setdefault('MPLCONFIGDIR','/workspace/.cache/matplotlib')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.backends.backend_pdf import PdfPages
    font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'font.size':11,'pdf.fonttype':42})
    fig=plt.figure(figsize=(15,9.4),facecolor='white')
    fig.suptitle('0353-2-B3  相邻时期 RoMa 配准评估',fontsize=21,y=.975)
    fig.text(.04,.918,f"自动质控通过率：{report['automatic_qc_passed_pairs']}/3 = {report['automatic_qc_pass_rate_pct']:.1f}%     人工标志点误差：未评估",fontsize=13)
    headers=['图像对','自动验证点数','平均误差\n目标像素','中位误差\n目标像素','几何重叠\n% 全画布','有效重叠\n% 全画布','低置信度面积\n% 全画布','自动质控']
    cells=[]
    for r in rows:
        cells.append([r['pair'],r['automatic_validation_points'],f"{r['automatic_point_mean_target_px']:.3f}",
                      f"{r['automatic_point_median_target_px']:.3f}",f"{r['geometric_overlap_pct_target']:.2f}",
                      f"{r['effective_overlap_pct_target']:.2f}",f"{r['low_confidence_overlap_pct_target']:.2f}",
                      '通过' if r['automatic_qc_pass'] else '未通过'])
    ax=fig.add_axes([.035,.675,.93,.20]);ax.axis('off')
    table=ax.table(cellText=cells,colLabels=headers,cellLoc='center',loc='center',bbox=[0,0,1,1]);table.auto_set_font_size(False);table.set_fontsize(11)
    for (r,c),cell in table.get_celld().items():
        cell.set_edgecolor('#d9e0e5')
        cell.set_facecolor('#e8eff5' if r==0 else ('#f6f8fa' if r%2==0 else 'white'))
    ax=fig.add_axes([.07,.345,.41,.255])
    x=np.arange(3)
    parts=[('effective_overlap_pct_target','有效重叠','#309a60'),('low_confidence_overlap_pct_target','低置信度','#e8a22f'),('other_unreliable_overlap_pct_target','其他不可靠重叠','#9670aa'),('nonoverlap_pct_target','无几何重叠','#d2d2d2')]
    base=np.zeros(3)
    for key,label,color in parts:
        vals=np.array([r[key] for r in rows]);ax.bar(x,vals,bottom=base,color=color,label=label,width=.57);base+=vals
    ax.set_xticks(x,[r['pair'] for r in rows]);ax.set_ylim(0,100);ax.set_ylabel('后一期完整画布面积 (%)');ax.set_title('面积分类（四类互斥）',fontsize=12)
    ax.legend(ncol=2,fontsize=9,loc='upper center',bbox_to_anchor=(.5,-.15),frameon=False)
    ax.spines[['top','right']].set_visible(False)
    ax2=fig.add_axes([.58,.345,.34,.255])
    ax2.bar(x,[r['automatic_point_median_at_target_width1400'] for r in rows],color='#3978a9',width=.57)
    ax2.set_xticks(x,[r['pair'] for r in rows]);ax2.set_ylabel('中位误差（宽度统一 1400 px）');ax2.set_title('自动验证点误差（统一分析尺度）',fontsize=12);ax2.spines[['top','right']].set_visible(False)
    notes=[
      '有效重叠：来源坐标在图内，双向 RoMa 置信度 ≥ 0.5，循环误差 ≤ 3 个目标像素，局部 Jacobian > 0。',
      '低置信度面积：几何重叠内目标侧置信度 < 0.5；此表比例分母为后一期完整画布。无重叠区域单独计数。',
      '自动验证：独立 RootSIFT 特征匹配 + 互选比例检验 0.75 + 相似变换 RANSAC + 空间抽稀；不按 RoMa 误差筛点。',
      '自动点误差使用后一期原始像素；T4 为 5100×4547，原始像素误差与 T2/T3 不能直接作同尺度比较。',
      '通过条件：有效面积 ≥10%，可靠 RoMa 采样 ≥100，自动点 ≥20，统一尺度中位误差 ≤3 px，置信区循环误差 P95 ≤3 px。',
      '自动质控通过不代表人工确认成功。自动特征点不是人工真值；未标定像素物理尺寸，面积以 px² 报告。',
      '结果仅涵盖此位置的 3 个相邻图像对，不代表整个 19 位置数据集。'
    ]
    for j,text in enumerate(notes):fig.text(.045,.235-j*.026,text,fontsize=9,color='#39434b')
    fig.savefig(out/'registration_metrics.png',dpi=240,bbox_inches='tight')
    fig2,axes=plt.subplots(3,3,figsize=(13.5,12),facecolor='white')
    for i,name in enumerate(PAIRS):
        for j,filename in enumerate(['registered_rgb.png','checkerboard.png','overlap_quality_map.png']):
            im=Image.open(out/name/filename);im.thumbnail((1000,950));axes[i,j].imshow(im);axes[i,j].axis('off')
            if i==0:axes[i,j].set_title(['前一期 → 后一期配准图','棋盘格质控','面积质量分类'][j],fontsize=13)
        axes[i,0].text(0,1.015,rows[i]['pair'],transform=axes[i,0].transAxes,fontsize=12,color='#172934')
    fig2.suptitle('相邻配准结果与重叠质控',fontsize=18,y=.985)
    fig2.text(.12,.012,'绿色：有效重叠  |  橙色：低置信度  |  紫色：其他不可靠重叠  |  灰色：无几何重叠',fontsize=11)
    fig2.tight_layout(rect=[0,.03,1,.96]);fig2.savefig(out/'registration_visual_qc.png',dpi=220)
    with PdfPages(out/'registration_evaluation_report.pdf') as pdf:pdf.savefig(fig);pdf.savefig(fig2)
    plt.close('all')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'0353-2-B3')
    parser.add_argument('--output',type=Path,default=ROOT/'0353-2-B3_registration_evaluation')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    cv2.setNumThreads(4);cv2.setRNGSeed(20261012)
    rows=[];details=[]
    for name in PAIRS:
        row,d=evaluate(name,args.source,args.output);rows.append(row);details.append(d)
    passed=sum(r['automatic_qc_pass'] for r in rows)
    report=dict(position='0353-2-B3',evaluated_pairs=3,automatic_qc_passed_pairs=passed,
        automatic_qc_pass_rate_pct=100*passed/3,successful_execution_pairs=3,
        successful_execution_rate_pct=100,manual_landmark_TRE=None,
        manual_confirmed_registration_success_rate=None,
        confidence_threshold=.5,automatic_qc_rules=RULES,
        area_units='native target pixels squared (px^2); no physical calibration available',
        low_confidence_definition='geometric overlap AND target confidence <0.5; primary denominator full target canvas',
        error_definition='Euclidean distance between independent RootSIFT target point and RoMa source-to-target prediction, native target pixels',
        limitations=['Automatic quality pass is not independently confirmed registration success.',
                    'SIFT verification is feature-rich-location sampling and may include false matches.',
                    'Manual landmark TRE unavailable; cycle error is internal consistency only.',
                    'Pixel errors and areas have different scales across periods; no mm calibration.',
                    '3 adjacent pairs from one position, not 19-position dataset.'],pairs=details)
    (args.output/'evaluation_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    save_csv(args.output/'registration_metrics.csv',rows)
    md=['# 0353-2-B3 相邻配准结果', '',f'自动质控通过：{passed}/3（{100*passed/3:.1f}%）。运行完成：3/3。人工确认成功率及人工标志点 TRE：未评估。','',
        '| 图像对 | 自动验证点 | 平均误差(px) | 中位误差(px) | 有效面积(px²) | 有效面积比例 | 低置信度面积(px²) | 低置信度比例 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        md.append(f"| {r['pair']} | {r['automatic_validation_points']} | {r['automatic_point_mean_target_px']:.3f} | {r['automatic_point_median_target_px']:.3f} | {r['effective_overlap_pixels']:,} | {r['effective_overlap_pct_target']:.2f}% | {r['low_confidence_overlap_pixels']:,} | {r['low_confidence_overlap_pct_target']:.2f}% |")
    md += ['', '面积与误差采用各对后一期原始像素坐标；所有表中面积比例分母为后一期完整画布。',
           '有效面积同时要求图内来源、双向置信度≥0.5、循环误差≤3目标像素、正Jacobian。',
           '低置信度面积为几何重叠内目标侧置信度<0.5，无重叠面积单独列入CSV。',
           '自动验证点误差不是人工标志点TRE。RootSIFT检验使用独立描述子、双向互选比例0.75、相似变换RANSAC阈值2.5分析像素、空间间隔12分析像素。',
           '分析宽度统一为1400像素；保留所有筛选后的对应点，不按RoMa置信度或预测误差删点。',
           '自动质控规则：'+json.dumps(RULES,ensure_ascii=False),
           'T4画布显著大于T2/T3，原始误差与面积值不能直接解释为物理差异。没有像素尺寸标定，无法换算毫米或平方毫米。',
           '仅此位置3对，不代表19位置整体成功率。']
    (args.output/'results_summary.md').write_text('\n'.join(md),encoding='utf-8')
    create_report(args.output,rows,report)
    print('AUTOMATIC QC',passed,'/',len(rows),flush=True)

if __name__=='__main__':main()
