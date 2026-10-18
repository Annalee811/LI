"""Final-model-only revised-reference report and Markdown submission guide."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import common as c
from comparison import overlay,root_only

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args();view=a.run
    summary=c.read(view/'tables/summary.json');comp=view/'comparison_RGB';docs=view/'manuscript';docs.mkdir(exist_ok=True)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',22)
    rows=[]
    for t in c.PERIODS:
        rgb=np.array(Image.open(view/'data/upper'/(t+'_RGB.png')));ref=np.array(Image.open(view/'data/upper'/(t+'_mask.png')))==255
        masks=[ref]+[np.array(Image.open(view/m/'upper'/(t+'_root_mask.png')))>0 for m in c.MODELS]
        names=['Reference annotation']+[c.NAMES[m] for m in c.MODELS]
        row=Image.new('RGB',(5*512,552),'white');dd=ImageDraw.Draw(row)
        images=[rgb]+[overlay(rgb,m) for m in masks]
        for j,im in enumerate(images):
            row.paste(Image.fromarray(im[40:552,1000:1512]),(j*512,40));dd.text((j*512+7,8),(['Original RGB']+names)[j],font=font,fill='#222222')
        row.save(comp/(t+'_512_RGB_annotated_comparison.png'));rows.append(row)
        for style in ['RGB','black_root']:
            local=Image.new('RGB',(4*512,552),'white');d=ImageDraw.Draw(local)
            for j,(name,m) in enumerate(zip(names,masks)):
                im=root_only(rgb,m) if style=='RGB' else np.repeat((~m).astype(np.uint8)[:,:,None]*255,3,axis=2)
                local.paste(Image.fromarray(im[40:552,1000:1512]),(j*512,40));d.text((j*512+7,8),name,font=font,fill='#222222')
            local.save(comp/(t+'_512_background_removed_'+style+'.png'))
        full=Image.new('RGB',(2541,5*1146),'white');d=ImageDraw.Draw(full)
        for j,im in enumerate(images):
            full.paste(Image.fromarray(im),(0,j*1146+38));d.text((9,j*1146+5),t+' | '+(['Original RGB']+names)[j]+' | Qualitative upper field',font=font,fill='#222222')
        full.save(comp/(t+'_full_upper_RGB_annotated_comparison.png'))
    atlas=Image.new('RGB',(2560,4*552),'white')
    for i,row in enumerate(rows):atlas.paste(row,(0,i*552))
    atlas.save(comp/'Four_period_512_RGB_annotated_comparison.png',dpi=(300,300))
    table=['|Model|Precision|Recall|Dice|IoU|','|---|---:|---:|---:|---:|']
    for r in summary['aggregate']:
        if r['split']=='test':table.append('|'+c.NAMES[r['model']]+'|'+'|'.join('%.4f'%r[k] for k in ['Precision','Recall','Dice','IoU'])+'|')
    methods='''# Final three-model results against revised root references

## Methods

Only the four human-annotated upper RGB fields of0353-2-B3(2541x1108px)were used. The preceding fixed random allocation of36native256px base positions into25training,5validation and6internal-test positions was retained. Each training position received a second native256px view translated12px in x/y, inward at boundaries, yielding200training views,20validation and24internal-test images. All four dates at one coordinate share one split. Added windows exclude every held-out pixel; within-train views overlap and are not independent acquisitions. No easy-tile/root-density filtering was used.

Original human strokes were retained separately. Revised binary references supplement narrow/missing strokes using previous U-Net probability, bright tubular RGB support and elongated-component constraints. These are model-assisted revised references, not new independent human ground truth. Original RGB pixels and colors remain unchanged; annotations/visualizations are separate files. No temporal or death classifier is fitted here.

All three previous matching models were fine-tuned for20epochs,64augmented draws/epoch,batch2,initialLR0.00008. The models are ResNet18U-Net,the supplied original TensorFlow1.12ResNet-v2-50DeepLabV3+,andSegFormer-B0with nativeRGB refinement. Epoch0can be retained. Checkpoints and thresholds were selected using ORIGINAL validation labels, not the revised reporting references. Original retained-mask metrics remain in the audit table.

The requested final figures compare only these final trained models with revised references. Precision/Recall/Dice/IoU are revised-reference AGREEMENT, pooled over the24internal-test tile-date views. They are not independent manual accuracy. Root-length references are skeletons derived from revised masks, in pixels; they are not independent manual measurements. The teacher-influenced reference can favor U-Net. External independently annotated test images:0.

## Revised-reference internal-test agreement

'''
    methods+='\n'.join(table)+'\n\nWhole-upper images include fitted pixels and are qualitative. No inference dilation, root reconnection or test-chosen cleanup is applied. ForegroundRGB and black-root background-removed versions are both supplied.\n'
    (docs/'Methods_Results_EN.md').write_text(methods)
    caps=c.read(view/'figures/captions.json');(docs/'Figure_captions_EN.txt').write_text('\n\n'.join(k+'\n'+v for k,v in caps.items())+'\n')
    chinese='''# Plant Phenomics 投稿说明

本包仅展示本轮最终模型与修订标注，不混入旧模型图。训练200张、验证20张、内部测试24张；训练200张是有重叠的原像素视图，不是200个独立生物学样本。同一坐标四期始终同组，新增训练窗口没有进入验证/测试像素。

图表参考是U-Net与RGB结构辅助修订标注。因此主表与S2/S3/S4/S5应写为“与修订参考的一致性”，不能写为独立人工准确率。原始人工标注和原始保留测试参考的数字另存供核查。模型与阈值实际由原始验证参考选定；S1的验证曲线如实保留这一真实训练记录，没有回算或改写历史分数。

主文建议Fig2（最终三模型修订参考一致性）、Fig3（四期原始RGB、修订参考及三模型标记对比）、Fig4（三模型根长回归）。Fig1可放方法；Fig5展示剔除背景后的局部根系。若论文需证明独立人工准确率，应使用独立复核并冻结的参考，而不能只依赖本轮模型辅助参考。

S1：本轮训练损失与实际选模验证Dice曲线，虚线为最终权重轮次。
S2：四期分别的Dice热图，区分验证/测试坐标；均对修订参考计算。
S3：根长差值Bland–Altman图；灰线平均偏差，虚线偏差±1.96SD。
S4：同一测试图块在三模型下的Dice差异，四期像素汇总；线条不是独立根管重复。
S5：相对修订参考的误差定位，绿色TP（根像素重合）、粉色FP（参考之外的预测根）、橙色FN（未被预测覆盖的参考根）。

绿色在二分类对比图中表示全部根系，不表示时序“新生根”。本轮没有训练死亡、新生、持续或延伸分类。根长单位为像素，来自掩膜骨架，不是独立人工尺量；回归图同时报告1:1一致性R²、OLS回归R²、RMSE和CCC。

只标注了一个位置的四期上半幅。不同日期/相邻切块存在相关性，外部独立根管测试图像为0，不能据此直接声称对其他根管的泛化准确率。原始/修订标注、模型权重、脚本、随机种子、固定划分、各图块明细应一起开放。

图像以600dpiTIFF/PNG和PDF保存，统计图另有可编辑SVG。提交主文图时使用单独图文件和英文图注；逐图块CSV和完整对比图作补充材料。源代码/数据建议保存到带永久版本的仓储，需要时申请DOI。以投稿时Plant Phenomics最新要求为准。没有Word文件。
'''
    (docs/'论文使用说明_中文.md').write_text(chinese)
    (docs/'Data_Code_Availability_EN.md').write_text('Original and model-assisted revised references, native RGBs, fixed splits, final weights, complete scripts and reference-relative evaluations are distributed together with SHA256 verification. The revised reference is teacher-influenced and is not independent manual truth. See the immutable repository commit recorded in DATA_COMMIT.txt.\n')
    c.js(docs/'submission_map.json',dict(final_models_only=True,reference='Model-assisted revised masks',
        main=['Fig2','Fig3','Fig4'],supplement=['S1','S2','S3','S4','S5'],no_Word=True,
        train_images=200,validation_images=20,internal_test_images=24,external_independent_test_images=0))
    print('FINAL_REVISED_ONLY_COMPARISONS_AND_MARKDOWN_DONE',flush=True)

if __name__=='__main__':main()
