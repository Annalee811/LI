"""Finalize one-row-per-position coverage and transparent delivery notes."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from render_results import aggregate,csvwrite

def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);a=p.parse_args();root=a.work/'deliverable';rows,confidence=aggregate(a.work);summary=[]
    assert (a.work/'inference_complete.json').exists()
    for f in sorted(root.glob('*/geometry_metadata.json')):
        m=json.loads(f.read_text());pos=m['position'];ar=m['area_rows'];n=len(m['regions']);cs=[r for r in confidence if r['position']==pos];vals=[r['common_area_percent'] for r in ar if r['common_area_percent'] is not None]
        if n:assert len(cs)==len(m['periods'])-1
        latest=max(cs,key=lambda r:int(r['target_period'][1:])) if cs else None
        record=dict(position=pos,observed_periods=';'.join('T'+str(t) for t in m['periods']),missing_periods=';'.join('T'+str(t) for t in range(1,5) if t not in m['periods']),original_images=len(m['periods']),independent_RGB_images=len(m['periods']) if n else 0,common_region_count=n,review_required_regions=';'.join(r['id'] for r in m['regions'] if r['review_required']),status=ar[0]['localization_status'],output_scope=m['scope'])
        for t in range(1,5):record[f'T{t}_common_area_percent']=next((r['common_area_percent'] for r in ar if r['period']==f'T{t}'),None)
        record.update(mean_available_period_common_area_percent=float(np.mean(vals)) if vals else None,min_available_period_common_area_percent=float(np.min(vals)) if vals else None,max_available_period_common_area_percent=float(np.max(vals)) if vals else None,final_observed_target=latest['target_period'] if latest else None,final_mean_cumulative_confidence=latest['mean_cumulative_confidence_on_common'] if latest else None,final_confident_correspondence_percent_of_common=latest['effective_overlap_percent_of_common'] if latest else None,final_low_cumulative_confidence_percent_of_common=latest['low_cumulative_confidence_percent_of_common'] if latest else None)
        summary.append(record)
    assert len(summary)==19 and sum(r['original_images'] for r in summary)==69 and sum(r['independent_RGB_images'] for r in summary)==66
    assert len(confidence)==48
    csvwrite(root/'all_series_position_summary.csv',summary)
    devices={json.loads(f.read_text())['device'] for f in (a.work/'posed_roma_evidence').glob('*/*/completed.json')}
    counts=dict(positions=19,source_images=69,complete_four_date_positions=sum(r['original_images']==4 for r in summary),three_date_positions=sum(r['original_images']==3 for r in summary),positions_with_localized_regions=sum(r['common_region_count']>0 for r in summary),positions_without_review_flag=sum(r['status']=='region_localized' for r in summary),positions_with_review_flag=sum(r['status']=='review_required' for r in summary),unresolved_positions=[r['position'] for r in summary if not r['common_region_count']],independent_RGB_images=66,posed_RoMa_observed_date_pairs=48,CUDA_used=any(d.startswith('cuda') for d in devices),runtime_devices=sorted(devices),Q1_source_SHA256='de74de5504474af5ef4f6e1cc28265a87ac7039032593e465ff9128336d394e2',confidence_is_conditional_on_localization=True,manual_landmark_TRE_evaluated=False)
    (root/'delivery_summary.json').write_text(json.dumps(counts,ensure_ascii=False,indent=2))
    readme='''# 全系列原图、独立拼接RGB与定位质量

采用更正后的 Q1.zip：19个位置、69张输入原图，12个位置有四期、7个位置有三期。
18个位置导出66张独立拼接RGB；其中8个位置含待复核的观察段，保留原标记，不按已验证成功计算。
Q1-3-Z3的T1未连通：保留三期原图，整序列共同面积记为未确定；另附T2/T3局部对应结果。

原图保持原始文件字节。按要求，0353-2-B3仅保留上方A观察区在四期源图中的原生像素包围框；
其独立拼接RGB固定2541×1108，已与此前人工标注使用的上半幅逐像素核对。
原图哈希记录的是完整输入；实际交付文件哈希以file_manifest.json为准。

共同面积比例 = 各段映射回该期完整原图后的多边形并集面积 / 该期完整原图宽×高。
已处理各期分辨率和尺度差异、重复覆盖及白色接缝。B3分母仍为完整原图，面积较小源于只交付上半区。
面积表提供逐期原尺寸、共同面积(px²)、占比和待复核状态；位置汇总提供已有时期的均值/最小/最大值。

所有RGB保持原色；每段只从原图做一次整体相似定位，不用稠密场扭曲根系形状。
白色分隔带是未观察间隔的非数据接缝，不应跨接缝认定根系连续。

累计置信度在已定位的RGB上重新运行完整RoMa后，从T1沿相邻实际观测日期的对应链累乘。
这些是以定位为条件的内部对应分数，不能作为定位正确性的独立证明；待复核标记不会因此取消。
原始T1坐标的16位置信度PNG除以65535为0–1数值，累计C的float32 NPZ完整保留，瓶颈均值列于CSV。
总包仅省去辅助矩阵及重复图像副本，累计C的float32数值逐位一致；全部矩阵可用脚本重建。
缺期时记录实际观测边，例如T2→T4；不补造图像或中间时期。

用于观察根系变化的是几何共同观察掩膜effective_common_observation_mask。
confidence_filtered_correspondence_mask（亦保留effective_overlap旧命名）用于逐点配准质控：
各步分数≥0.5、位移≤15个T1像素、循环误差≤10个T1像素，且路径不离开本观察段。
低累计置信度面积按累乘值<0.5统计；它与逐步筛选支持面积采用不同定义，两者不必互补。
真实根系生长、土层移动或反光可能降低对应置信度，不能因此直接删除这些生物变化像素。

含全系列及各位置定位总览、面积/质量汇总图、原图、各期独立RGB、累计置信度、掩膜和完整Python脚本。
脚本及冻结几何可复现既有非Q1定位，Q1另包含新的结构搜索与路线检验；RoMa源码和权重校验值随包保存。
模型权重使用官方缓存/下载地址，不重复打包。自动检测CUDA，本轮实际为CPU。
没有人工对应标志点，人工TRE未评估；覆盖面积和置信度分别报告，不把面积占比当成逐点准确率。
原图、RGB采样、面积并集、置信度数值及掩膜计数均由verify_results.py核验；ZIP另做CRC及SHA256校验。
无Word文件。进一步复现说明在scripts/README.md。
'''
    if devices!={'cpu'}:readme=readme.replace('本轮实际为CPU','本轮实际设备为'+', '.join(sorted(devices)))
    (root/'README.md').write_text(readme)
    print(json.dumps(counts,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
