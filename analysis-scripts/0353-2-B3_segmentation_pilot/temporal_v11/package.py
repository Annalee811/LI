"""Package v11 outputs; unchanged v9 root features remain immutable v10 downloads."""
import argparse,json,shutil,zipfile
from pathlib import Path
import common as c

DATA_COMMIT='685ae232bdcc29806944947d658a3d6779313b1c'
ROOT='roma-results/0353-2/0353-2-B3/segmentation-pilot-annotation/'

def tree(base,prefix):
 return [(f,str(Path(prefix)/f.relative_to(base))) for f in base.rglob('*') if f.is_file() and '__pycache__' not in f.parts]
def writezip(path,pairs):
 assert len(pairs)==len({n for _,n in pairs})
 with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for f,n in sorted(pairs,key=lambda r:r[1]):z.write(f,n)
 with zipfile.ZipFile(path) as z:assert z.testzip() is None
 assert path.stat().st_size<99*1024*1024,(path,path.stat().st_size)
 print('V11_ZIP',path.name,path.stat().st_size,flush=True)
 return dict(file=path.name,bytes=path.stat().st_size,sha256=c.sha(path))
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--roots',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);p.add_argument('--publish',type=Path,required=True);a=p.parse_args();run=a.run;out=a.publish;out.mkdir(parents=True,exist_ok=True);src=Path(__file__).resolve().parent;repo=src.parents[2];records=[]
 stats=['all_metrics.json','temporal_metrics.csv','per_transition_metrics.csv','root_length_agreement.json','root_length_pairs.csv','selection.json','verification.json','standalone_verification.json','before_after_class_metrics.csv','before_after_confusions.csv','comparison_protocol.json','junction_visual_ROI.json','geometry_verification.json','root_support_by_class.json','dataset_summary.json','pair_improvement_verification.json']
 for f in stats:shutil.copyfile(run/f,out/f)
 shutil.copyfile(src/'README.md',out/'README_method_and_reproduce.md');common=[(src/'README.md','README_method_and_reproduce.md')]
 records.append(writezip(out/'0353-2-B3_temporal_v11_paper_figures.zip',tree(run/'figures','figures')+[(run/f,f) for f in stats]+common))
 for n in c.MODELS:
  files=tree(run/n/'four_period_final_fit',n+'/four_period_final_fit')+tree(run/n/'evaluation',n+'/evaluation')
  files += [(f,n+'/'+f.name) for f in (run/n).iterdir() if f.is_file() and f.suffix in ['.json','.csv']]
  records.append(writezip(out/f'0353-2-B3_temporal_v11_results_{n}.zip',files+[(run/f,f) for f in ['before_after_class_metrics.csv','before_after_confusions.csv']]+common))
 files=[]
 for name in ['upper','targets','root_gates']:files+=tree(run/'data'/name,'data/'+name)
 for name in ['original_human_progress.json','source_integrity.json','human_class_counts.csv','paired_patch_manifest.csv']:files.append((run/'data'/name,'data/'+name))
 files += tree(run/'focused_training','focused_training')
 for n in c.MODELS:
  for name in ['evaluation_training_draws.csv','final_training_draws.csv']:files.append((run/n/name,'draws/'+n+'/'+name))
 records.append(writezip(out/'0353-2-B3_temporal_v11_training_data.zip',files+common))
 sources=[]
 for name in ['temporal_v11','temporal_v10','paper_v6','root_background_v5']:
  sources += [(f,arc) for f,arc in tree(src.parent/name,'source/'+name) if f.suffix in ['.py','.md','.txt','.json']]
 sources += [(f,arc) for f,arc in tree(repo/'results/B3_uploaded_deeplab_v1/vendor','vendor') if f.suffix not in ['.png','.jpg']]
 sources += [(repo/'results/B3_uploaded_deeplab_v1/Dockerfile','DeepLab_Dockerfile')]
 for n in c.MODELS:
  sources += [(f,n+'/'+f.name) for f in (run/n).iterdir() if f.is_file() and f.suffix in ['.pt','.json','.csv']]
  for name in ['evaluation_head.pt','final_head.pt']:sources.append((a.previous/n/name,'previous_v10/'+n+'/'+name))
 for n in ['unet','segformer']:
  for name in ['best.pt','postprocessing_selection.json','link_classifier.json']:sources.append((a.roots/n/name,n+'/'+name))
 records.append(writezip(out/'0353-2-B3_temporal_v11_weights_and_scripts.zip',sources+common))
 old=json.loads((repo/ROOT/'three-model-temporal-v10/download_manifest.json').read_text());external=[]
 for r in old['archives']:
  if 'frozen_features' in r['file'] or 'frozen_DeepLab_weights' in r['file'] or 'results_' in r['file']:
   external.append(dict(**r,url='https://raw.githubusercontent.com/Annalee811/LI/'+DATA_COMMIT+'/'+ROOT+'three-model-temporal-v10/'+r['file']))
 c.js(out/'download_manifest.json',dict(archives=records,unchanged_external_inputs=external,all_zip_crc_verified=True,annotation_sha256=c.sha(run/'data/original_human_progress.json')))
 report=json.loads((run/'all_metrics.json').read_text());oldreport=json.loads((a.previous/'all_metrics.json').read_text());text='# B3 三模型连接点与时序类别优化 v11\n\n在v10三个时序头上继续训练：新生、持续、生长全部纳入；根背景模型与人工标注不变。没有将物理连通或交叉直接作为同一类别的规则。死亡不纳入。\n\n|模型|持续Dice 旧→新|新生Dice 旧→新|生长Dice 旧→新|宏Dice 旧→新|\n|---|---:|---:|---:|---:|\n'
 for n in c.MODELS:
  before=oldreport[n]['validation']['end_to_end'];after=report[n]['validation']['end_to_end'];cells=[f'{b["Dice"]:.4f} → {v["Dice"]:.4f}' for b,v in zip(before['classes'],after['classes'])];cells.append(f'{before["macro_Dice"]:.4f} → {after["macro_Dice"]:.4f}');text+='|'+c.MODEL_NAMES[n]+'|'+'|'.join(cells)+'|\n'
 text+='\n本轮最终接受U-Net的更新；DeepLab与SegFormer未同时满足六项互误保护条件，部署头保留旧版（训练试验与未采用权重仍保留）。这是同位置C3验证指标，C3参与选优。C4重复区域、连接点及前期根尖区的所有三类变化和三对混淆均在CSV和图中列出，包括退步，不作为跨根管独立测试。完整四期图是全标注最终头的拟合展示；模型仍不等于已验证的根对象身份追踪。\n\n|ZIP下载|大小MB|\n|---|---:|\n'
 for r in records:text+='|[%s](%s)|%.1f|\n'%(r['file'],r['file'],r['bytes']/1e6)
 text+='\n[方法与复现](README_method_and_reproduce.md)。训练数据包包含原始人工JSON、四期上半部、未改动标签、实际连接区训练裁剪及剔背景根RGB；增强与完整抽样记录保留。权重脚本包包含三模型新旧时序权重和完整源码、U-Net/SegFormer根网络权重。冻结第一阶段输入和原始DeepLab根权重复用已校验的v10包：\n\n'
 for r in external:
  if 'frozen_' in r['file']:text+='- [%s](%s)\n'%(r['file'],r['url'])
 text+='\n预测始终使用前期模型根掩膜，不使用人工前期掩膜。每个ZIP低于GitHub文件限制，SHA256见download_manifest.json。\n';(out/'README.md').write_text(text)
 print('ALL_V11_PACKAGES_READY',flush=True)
if __name__=='__main__':main()
