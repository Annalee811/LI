#!/usr/bin/env python3
"""Report and package the anchored sequential pilot with honest, comparable metrics."""
from pathlib import Path
import argparse,csv,hashlib,json,shutil,zipfile
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--previous-run',type=Path,required=True);p.add_argument('--editor',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    def read(path):return json.loads(path.read_text())
    selection=read(a.run/'selection.json');metrics=read(a.run/'evaluation/metrics.json');summary=read(a.run/'four_period_summary.json');verification=read(a.run/'verification.json');assert verification['failed']==0
    metric_rows=[]
    for split,result in metrics.items():
        metric_rows.append({'split':split,'task':'root/background','class':'root',**{k:result['root'][k] for k in ['Precision','Recall','Dice','IoU']}})
        for row in result['temporal']['per_class']:metric_rows.append({'split':split,'task':'temporal','class':row['name'],**{k:row[k] for k in ['Precision','Recall','Dice','IoU']}})
    with (a.run/'evaluation_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(metric_rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(metric_rows)
    classes=['Persistent','Newly visible','Tip elongation'];names={'previous_temporal_unet':'Previous classifier','sequential_joint_unet':'Joint candidate','balanced_temporal_unet':'Balanced classifier'};fig,ax=plt.subplots(figsize=(7,4));x=np.arange(3)
    for i,row in enumerate(selection['temporal_candidates']):ax.bar(x+(i-1)*.24,[r['Dice'] for r in row['per_class']],width=.24,label=names[row['model']])
    ax.set_xticks(x,classes);ax.set_ylim(0,1);ax.set_ylabel('Validation Dice');ax.set_title('Same manual T1 anchor, same T2 validation pixels');ax.legend(frameon=False,fontsize=8);fig.tight_layout();fig.savefig(a.run/'validation_class_comparison.png',dpi=600);fig.savefig(a.run/'validation_class_comparison.pdf');plt.close(fig)
    # Exact native held-out label maps, with a separate root segmentation error panel.
    rgb=np.array(Image.open(a.dataset/'dataset/upper/T2_RGB.png'))[:,1558:2006];gt=np.array(Image.open(a.run/'evaluation/val/T2_human_temporal.png'));pred=np.array(Image.open(a.run/'evaluation/val/T2_temporal_prediction.png'));palette={0:[255,255,255],1:[147,51,234],2:[22,163,74],3:[234,179,8],255:[148,163,184]};colored=[]
    for mask in [gt,pred]:
        im=np.full_like(rgb,255)
        for key,color in palette.items():im[mask==key]=color
        colored.append(im)
    human_root=np.array(Image.open(a.dataset/'dataset/upper/T2_mask.png'))[:,1558:2006]==255;model_root=np.array(Image.open(a.run/'evaluation/val/T2_root_prediction.png'))==255;error=np.full_like(rgb,255);error[human_root&model_root]=[40,110,70];error[human_root&~model_root]=[37,99,235];error[~human_root&model_root]=[220,38,38]
    canvas=Image.new('RGB',(1792,1168),'white');draw=ImageDraw.Draw(canvas);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',22)
    for i,(im,label) in enumerate(zip([rgb,*colored,error],['T2 native RGB','Human classes','Raw model classes','Root TP green / FN blue / FP red'])):canvas.paste(Image.fromarray(im),(448*i,60));draw.text((448*i+5,10),label,font=font,fill='#20352d')
    canvas.save(a.run/'validation_native_comparison.png',dpi=(600,600))
    source=Path(__file__).resolve().parent
    inventory={}
    progress=read(a.previous_run/'original_human_progress.json')
    for item in progress['images']:
        if item['name'][:2] not in ['T1','T2']:continue
        def decode(key):return np.repeat(np.asarray(item['layers'][key][::2],np.uint8),item['layers'][key][1::2]).reshape(item['height'],item['width'])[:1108]
        root=decode('mask')==255;inventory[item['name'][:2]]={key:{str(v):int((decode(key)[root]==v).sum()) for v in np.unique(decode(key)[root])} for key in ['color','viability']}
    (a.run/'browning_death_label_inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
    v=metrics['val'];t=metrics['test'];lines=['0353-2-B3 连续根系分割与时序分类预实验 v3','',
      '连续输出：T1原人工标注 → T2原人工标注 → T3预测（参考T2人工根）→ T4预测（参考T3预测根）。',
      'T1全部已有根为基础根；各后续时期相对前一期，不把全部持续根都认为起源于T1。',
      '前期未见、本期可见统一属于新增；保留原人工分类：新生根（包括新侧根）与原有根尖延伸段。新增总类为这两类的并集；不能分类者单列未判定。',
      'T1/T2人工标签没有重新生成或覆盖。原始进度完整备份；T3/T4原笔迹不使用、不合并。下部及白色分隔带排除，只处理上部2541×1108原始像素，无缩放、色彩增强或宽度修改。','',
      '本轮实际训练：联合分割与时序U-Net 59轮（最佳第34轮）；类别均衡时序U-Net 72轮（最佳第47轮）。',
      '联合模型未在T2验证根分割上胜过原U-Net，因此最终根分割继续使用原U-Net；根分割没有显著改善。时序分类选择新增的均衡分支。',
      f"同一锚定协议下，验证根Dice={v['root']['Dice']:.4f}，空间留出根Dice={t['root']['Dice']:.4f}。",'']
    for row in v['temporal']['per_class']:lines.append(f"验证 {row['name']}：Precision={row['Precision']:.4f}，Recall={row['Recall']:.4f}，Dice={row['Dice']:.4f}，IoU={row['IoU']:.4f}。")
    lines.extend([f"验证三类平均Dice={v['temporal']['macro_Dice_supported_classes']:.4f}；同一协议旧分类器={selection['temporal_candidates'][0]['macro_Dice_supported_classes']:.4f}。",
      '延伸类验证Dice约0.24，仍不能可靠识别，不能据此声称准确追踪全部根尖。独立空间留出不含延伸真值，其Recall/Dice为NA；不能将其宣称为测试成功。',
      '本轮验证只评估T2，输入含前一期T1人工根；不同于v2合并T1/T2根分割及预测T1参考的协议，不能将跨协议差异当作性能提升。',
      '仍只有1个位置、2个已标注时期；采用原空间C0-C2训练、C3验证、C4留出，64像素隔离，同区T1/T2不跨组。原固定256图块计训练60、验证20、留出20；时序对应30/10/10个空间配对。',
      '均衡分支每轮从三类人工笔迹分别抽20个128×128原像素小块，合计60次动态采样；这不是新增独立图像。上一期参考50%采用人工T1、50%采用预测T1，适配人工起点及后续递推。',
      '多次探索复用同一验证与留出区域，不构成新的独立确认性研究。T3/T4没有完整真值，跨期推断准确率未评估，误差可向后传播。','',
      '褐变和死亡：T1/T2上部所有根的color及viability属性均为0（未分类），没有该类监督例证，本轮不预测也不记作0个死亡。',
      'previous_root_not_observed_candidate_mask是未再观察到的候选，采用8像素局部位置容差并排除边缘8像素；遮挡、位移、漏检均可导致此信号，绝不等同死亡。',
      'first_visible_geometry_candidate_mask是前期已标/检出根附近8像素之外的首次可见候选，不证明生物学新生；new_total_lateral_and_tip_mask为原始新生+延伸类别之并集。',
      '综合图与可编辑结果采用含未判定版本；原始模型分类另存。低分（最大条件概率<0.55或前两类差<0.15）保留未判定。',
      '另对几何冲突保留未判定：持续段距前期根>12像素，新增段落在前期根2像素以内，或延伸连通段附近24像素内没有前期预测根尖。此为探索性显示检查，未验证其生物学准确性；不改根掩膜，不用其回避错误抬高报告指标。',
      '主图蓝色基础、紫色持续、绿色新生、黄色延伸、灰色未判定。未再观察到候选的单独叠加图使用红色，红色不代表已确认死亡。没有建立逐条根身份ID。','',
      '结果包内annotate.html：选择images中的四张RGB，导入editable_upper_RGB_sequential.json，即可继续修改；最高60倍，切图保留倍率，Ctrl+Z撤销，Ctrl+S保存全部进度。',
      'editable_T1T2_manual_T3T4_sequential.json适用于原来四张完整2541×1867图；original_human_progress.json是未经修改的原始进度。',
      '原RGB独立保存在images中，T1/T2分类图是人工标注，T3/T4才是预测。人工参考没有伪装成模型成绩。',
      '复现：合并解压结果包与模型包（同一顶层目录），安装torch2.8.0、numpy2.2.6、Pillow11.3.0、scipy1.16.2、scikit-image0.25.2、transformers4.57.1、matplotlib3.10.6。',
      'python source/predict_sequential_anchored.py --dataset data_run --previous-run previous_run --run .',
      'python source/train_sequential_joint.py --dataset data_run --previous-run previous_run --output joint_retrained --epochs 100',
      'python source/train_balanced_temporal.py --dataset data_run --previous-run previous_run --output balanced_retrained --epochs 100',
      '自动检测CUDA，本轮实际CPU。原三模型二分类比较保留在上一版包；本轮增加两次U-Net训练，不冒充重新完成DeepLab/SegFormer训练。'])
    (a.run/'结果说明.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8-sig')
    stage=a.run/'packages';result=stage/'results';model=stage/'models';result.mkdir(parents=True,exist_ok=True);model.mkdir(exist_ok=True)
    sources=['train_uploaded_upper.py','train_pair_temporal.py','train_segformer_progress.py','train_sequential_joint.py','train_balanced_temporal.py','predict_sequential_anchored.py','package_sequential_results.py']
    common=['selection.json','verification.json','four_period_summary.json','evaluation_metrics.csv','browning_death_label_inventory.json','结果说明.txt']
    for folder in [result,model]:
        (folder/'source').mkdir(exist_ok=True)
        for name in sources:shutil.copyfile(source/name,folder/'source'/name)
        for name in common:shutil.copyfile(a.run/name,folder/name)
        shutil.copyfile(a.previous_run/'original_human_progress.json',folder/'original_human_progress.json')
        shutil.copytree(a.run/'evaluation',folder/'evaluation',dirs_exist_ok=True)
        for name in ['validation_class_comparison.png','validation_class_comparison.pdf','validation_native_comparison.png']:shutil.copyfile(a.run/name,folder/name)
    for name in ['four_period_results','images']:shutil.copytree(a.run/name,result/name,dirs_exist_ok=True)
    for name in ['editable_upper_RGB_sequential.json','editable_T1T2_manual_T3T4_sequential.json','four_period_root_classes.png','four_period_RGB_classes.png']:shutil.copyfile(a.run/name,result/name)
    shutil.copyfile(a.editor,result/'annotate.html')
    for name in ['training','balanced_temporal']:shutil.copytree(a.run/name,model/name,dirs_exist_ok=True)
    shutil.copytree(a.dataset/'dataset',model/'data_run/dataset',dirs_exist_ok=True);(model/'data_run/training').mkdir(parents=True,exist_ok=True);shutil.copyfile(a.dataset/'training/best.pt',model/'data_run/training/best.pt')
    (model/'previous_run/temporal').mkdir(parents=True,exist_ok=True);shutil.copyfile(a.previous_run/'temporal/best.pt',model/'previous_run/temporal/best.pt');shutil.copyfile(a.previous_run/'original_human_progress.json',model/'previous_run/original_human_progress.json')
    records=[]
    for folder,filename in [(result,'0353-2-B3_sequential_T1T2_manual_T3T4_results_v3.zip'),(model,'0353-2-B3_sequential_models_and_training_v3.zip')]:
        files=sorted(f for f in folder.rglob('*') if f.is_file() and f.name!='SHA256.json' and '__pycache__' not in f.parts and not (folder==result and f.suffix=='.npy'))
        checksums={str(f.relative_to(folder)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files};(folder/'SHA256.json').write_text(json.dumps(checksums,ensure_ascii=False,indent=2)+'\n');destination=a.output/filename
        with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for f in [*files,folder/'SHA256.json']:z.write(f,Path('0353-2-B3_sequential_v3')/f.relative_to(folder))
        assert destination.stat().st_size<100*1024**2,(destination,destination.stat().st_size)
        with zipfile.ZipFile(destination) as z:assert z.testzip() is None
        records.append({'file':filename,'bytes':destination.stat().st_size,'sha256':hashlib.sha256(destination.read_bytes()).hexdigest()})
    for name in common:shutil.copyfile(a.run/name,a.output/name)
    for name in sources:shutil.copyfile(source/name,a.output/name)
    (a.output/'README.md').write_text('# 0353-2-B3 连续根系分割与分类 v3\n\n[下载连续图像与可编辑结果](0353-2-B3_sequential_T1T2_manual_T3T4_results_v3.zip) · [下载模型与训练记录](0353-2-B3_sequential_models_and_training_v3.zip)\n\nT1/T2为原人工标签，T3参考T2人工根，T4参考T3预测根。新增=新生+延伸；未再观察到候选不等于死亡。褐变/死亡没有监督标签，未训练。\n\n验证T2根Dice %.4f；持续、新生、延伸Dice分别%.4f、%.4f、%.4f。原U-Net根分割仍优于新联合候选，最终使用原根分割与新均衡时序分支。延伸识别仍弱，T3/T4类别未经定量验证。\n\n详细协议、限制、原始指标和使用方法见结果说明.txt；所有指标使用原始未筛选预测，不通过删去低置信度像素提高成绩。\n' % (v['root']['Dice'],*[row['Dice'] for row in v['temporal']['per_class']]))
    (a.output/'artifact_sha256.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(records),flush=True)

if __name__=='__main__':main()
