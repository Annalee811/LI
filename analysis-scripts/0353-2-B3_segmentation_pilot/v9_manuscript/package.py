"""Re-save original v9 archives unchanged and add manuscript/analysis bundles."""
import argparse,shutil,zipfile,json
from pathlib import Path
import common as c

def tree(root,prefix):return [(f,str(Path(prefix)/f.relative_to(root))) for f in root.rglob('*') if f.is_file() and '__pycache__' not in f.parts]
def archive(path,files):
 assert len(files)==len({n for _,n in files})
 with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for f,name in sorted(files,key=lambda x:x[1]):z.write(f,name)
 with zipfile.ZipFile(path) as z:assert z.testzip() is None
 assert path.stat().st_size<99*1024*1024
 print('NEW_ARCHIVE_READY',path.name,path.stat().st_size,flush=True);return dict(file=path.name,bytes=path.stat().st_size,sha256=c.sha(path))
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--original-archives',type=Path,required=True);p.add_argument('--publish',type=Path,required=True);a=p.parse_args();run=a.run;pub=a.publish;pub.mkdir(parents=True,exist_ok=True);old=pub/'original_v9_archives';old.mkdir(exist_ok=True);src=Path(__file__).resolve().parent;new=[];saved=[]
 for r in c.read(run/'inputs/original_archives.json')['archives']:
  f=old/r['file'];shutil.copyfile(a.original_archives/r['file'],f);assert c.sha(f)==r['SHA256'];saved.append(dict(file='original_v9_archives/'+r['file'],bytes=f.stat().st_size,sha256=r['SHA256'],unchanged=True))
 originals=run/'original_v9_source';originals.mkdir(exist_ok=True)
 for name in ['0353-2-B3_v9_models_torch.zip','0353-2-B3_v9_model_deeplab.zip']:
  with zipfile.ZipFile(a.original_archives/name) as z:
   for member in z.namelist():
    parts=Path(member).parts
    if len(parts)<3 or parts[1] not in ['source','vendor','runtime']:continue
    if Path(member).suffix not in ['.py','.txt','.md','.json'] and not Path(member).name.startswith('Dockerfile') and Path(member).name not in ['LICENSE','NOTICE']:continue
    target=originals/Path(*parts[1:]);target.parent.mkdir(parents=True,exist_ok=True);data=z.read(member)
    if target.exists():assert target.read_bytes()==data
    else:target.write_bytes(data)
 root='0353-2-B3_v9_manuscript_reanalysis';base=[(src/'README.md',root+'/README.md')]
 new.append(archive(pub/'0353-2-B3_v9_PlantPhenomics_figures_tables.zip',tree(run/'figures',root+'/figures')+tree(run/'tables',root+'/tables')+tree(run/'manuscript',root+'/manuscript')+base))
 new.append(archive(pub/'0353-2-B3_v9_PlantPhenomics_guide.zip',tree(run/'manuscript',root+'/manuscript')+tree(run/'tables',root+'/tables')+base))
 sources=[(f,name) for f,name in tree(src,root+'/source/v9_manuscript') if f.suffix in ['.py','.md','.txt']]
 new.append(archive(pub/'0353-2-B3_v9_analysis_data_full_scripts.zip',tree(run/'inputs',root+'/inputs')+tree(originals,root+'/source/original_v9')+sources+base))
 for name in ['Table_C4_strict_pooled4dates.csv','Table_C3_strict_pooled4dates.csv','aggregate_metrics.csv','root_length_summary.csv','study_design.json','verification.json','source_archive_exact_verification.json']:shutil.copyfile(run/'tables'/name,pub/name)
 shutil.copyfile(run/'manuscript/论文图表使用与分析_CN.md',pub/'论文图表使用与分析_CN.md');shutil.copyfile(src/'README.md',pub/'README_reproduce.md')
 c.js(pub/'download_manifest.json',dict(new_archives=new,resaved_original_archives=saved,source_commit=c.ORIGINAL_COMMIT,original_annotation_sha256=c.sha(run/'inputs/original_human_progress.json'),original_v9_outputs_unchanged=True,reanalysis_only_no_training_or_threshold_changes=True,all_archive_CRC_verified=True))
 text='# 0353-2-B3 v9：Plant Phenomics 论文图表重整\n\n已按原字节重新保存用户指定v9的七个成果ZIP，不改人工标注、权重或预测。新增10组论文图、均值/合并汇总、三模型根长回归、英文图注、可编辑英文Methods/Results和中文投稿说明、完整复算脚本及native输入。\n\nC3是选优验证区；C4是反复查看过的同位置benchmark，外部独立测试图像为0。建议正文采用C4四期合并指标、C3单列补充；混合两区的均值不作为主要成绩。该刊实时作者指南受环境域名限制未核实，具体投稿要求需以官网为准。本次未投稿。\n\n|新文件|大小MB|\n|---|---:|\n'
 for r in new:text+=f'|[{r["file"]}]({r["file"]})|{r["bytes"]/1e6:.1f}|\n'
 text+='\n[论文图表使用与分析](论文图表使用与分析_CN.md) · [复现说明](README_reproduce.md)。\n\n原v9完整成果按原字节重新保存：\n\n|原始ZIP|大小MB|\n|---|---:|\n'
 for r in saved:text+=f'|[{Path(r["file"]).name}]({r["file"]})|{r["bytes"]/1e6:.1f}|\n'
 (pub/'README.md').write_text(text);print('ALL_V9_MANUSCRIPT_ARCHIVES_READY',flush=True)
if __name__=='__main__':main()
