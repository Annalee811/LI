"""Saved-map evaluation: teacher agreement and preserved annotation audit separated."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import common as c
from predict import evaluate

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--roots',type=Path,required=True);a=p.parse_args()
    rows=[];summary=[];period_rows=[];comparisons=[];figures=a.run/'figures';figures.mkdir(exist_ok=True)
    prep=json.loads((a.run/'preparation.json').read_text());previous=Path(prep['previous'])
    for m in c.MODELS:
        out=a.run/m;data=c.Data(a.run,a.roots,m);training=json.loads((out/'training_run.json').read_text())
        maps={t:np.array(Image.open(out/'four_period_prediction'/(t+'_temporal_classes.png'))) for t in c.PERIODS[1:]}
        targets=data.targets['current'];audit={}
        for reference in ['U_Net_derived_proposal','preserved_original_annotation']:
            data.targets['current']=targets if reference=='U_Net_derived_proposal' else {t:np.array(Image.open(a.run/'data/original'/(t+'_classes.png'))) for t in c.PERIODS}
            for split in ['val','test']:
                score,per=evaluate(data,maps,split)
                audit.setdefault(reference,{})[split]=score
                c.csvout(out/(split+'_'+reference+'_per_pair_metrics.csv'),per)
                for row in score['classes']:rows.append(dict(model=c.MODEL_NAMES[m],reference=reference,split=split,macro_Dice=score['macro_Dice'],**row))
                mean=lambda key:float(np.mean([r[key] for r in score['classes'] if r[key] is not None]))
                summary.append(dict(model=c.MODEL_NAMES[m],reference=reference,split=split,Precision=mean('Precision'),Recall=mean('Recall'),Dice=score['macro_Dice'],IoU=mean('IoU'),selected_epoch=training['selected_epoch'],interpretation='Teacher consistency, not independent accuracy' if reference=='U_Net_derived_proposal' else 'Original-label agreement, historical same-field exposure'))
                for t in c.PERIODS[1:]:
                    cm=np.zeros((4,4),np.int64);n=0
                    for r in data.pairs:
                        if r['split']!=split or r['current']!=t:continue
                        x,y=int(r['x']),int(r['y']);cm+=c.old.confusion(data.targets['current'][t][y:y+256,x:x+256],maps[t][y:y+256,x:x+256]);n+=1
                    metric=c.old.metrics(cm)
                    for row in metric['classes']:period_rows.append(dict(model=c.MODEL_NAMES[m],reference=reference,split=split,period=t,pair_count=n,macro_Dice=metric['macro_Dice'],**row))
                oldmaps={t:np.array(Image.open(previous/m/'four_period_prediction'/(t+'_temporal_classes.png'))) for t in c.PERIODS[1:]}
                old,_=evaluate(data,oldmaps,split)
                comparisons.append(dict(model=c.MODEL_NAMES[m],reference=reference,split=split,previous_V19_Dice=old['macro_Dice'],current_V21_Dice=score['macro_Dice'],difference=score['macro_Dice']-old['macro_Dice']))
        c.js(out/'separate_reference_evaluation.json',audit);data.targets['current']=targets
        # Clarify the inherited metadata field without altering predictions.
        inference_path=out/'four_period_prediction/inference.json';inference=json.loads(inference_path.read_text())
        inference.pop('manual_labels_used_for_training_and_validation_selection',None)
        inference.update(proposal_used_for_student_training_and_validation=True,original_manual_labels_used_for_student_selection=False,teacher_previously_fitted_source_annotations=True,proposal_SHA256=prep['training_reference_SHA256'])
        c.js(inference_path,inference)
        for t in c.PERIODS:
            rgb=data.rgb[t];gate=data.gate[t];rgba=np.concatenate([rgb,gate[...,None].astype('uint8')*255],2);rgba[~gate,:3]=0
            Image.fromarray(rgba).save(out/'four_period_prediction'/(t+'_original_roots_RGBA.png'))
    c.csvout(a.run/'three_model_class_metrics.csv',rows);c.csvout(a.run/'three_model_metrics_summary.csv',summary);c.csvout(a.run/'three_model_period_metrics.csv',period_rows);c.csvout(a.run/'before_after_two_references.csv',comparisons)
    # Native RGB is unwarped. Panels use one aspect ratio and one shared crop.
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',46);small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',34)
    headers=['RGB','Original annotation','U-Net reference','U-Net','DeepLabV3+','SegFormer-B0'];w=640;h=round(w*1108/2541);gap=16;margin=28;top=90
    full=Image.new('RGB',(margin*2+len(headers)*w+(len(headers)-1)*gap,top+4*(h+52)+70),'white');d=ImageDraw.Draw(full)
    for j,title in enumerate(headers):d.text((margin+j*(w+gap),20),title,font=font,fill='#222222')
    group=next(r for r in c.rows(a.run/'data/spatial_manifest.csv') if r['split']=='val');x,y=int(group['x']),int(group['y'])
    lw=448;localheads=['U-Net reference','U-Net','DeepLabV3+','SegFormer-B0'];local=Image.new('RGB',(2*margin+4*lw+3*gap,top+4*(2*lw+70)+70),'white');ld=ImageDraw.Draw(local)
    for j,title in enumerate(localheads):ld.text((margin+j*(lw+gap),20),title,font=font,fill='#222222')
    for i,t in enumerate(c.PERIODS):
        rgb=np.array(Image.open(a.run/'data/upper'/(t+'_RGB.png')));orig=np.array(Image.open(a.run/'data/original'/(t+'_classes.png')));revised=np.array(Image.open(a.run/'data/current'/(t+'_classes.png')))
        maps=[orig,revised]+[np.array(Image.open(a.run/m/'four_period_prediction'/(t+'_temporal_classes.png'))) for m in c.MODELS]
        overlays=[]
        for target in maps:
            overlay=rgb.copy();known=(target>0);overlay[known]=np.rint(.35*rgb[known]+.65*c.PALETTE[target[known]]).astype('uint8');overlays.append(overlay)
        for j,panel in enumerate([rgb]+overlays):
            xx=margin+j*(w+gap);yy=top+i*(h+52);d.text((xx,yy),t,font=small,fill='#222222')
            img=Image.fromarray(panel).resize((w,h),Image.Resampling.LANCZOS);full.paste(img,(xx,yy+45));d.rectangle((xx+round(x/2541*w),yy+45+round(y/1108*h),xx+round((x+256)/2541*w),yy+45+round((y+256)/1108*h)),outline='#D55E00',width=2)
        for j,(target,overlay) in enumerate(zip(maps[1:],overlays[1:])):
            xx=margin+j*(lw+gap);yy=top+i*(2*lw+70);ld.text((xx,yy),t+' / '+group['tile_id'],font=small,fill='#222222')
            local.paste(Image.fromarray(overlay[y:y+256,x:x+256]).resize((lw,lw),Image.Resampling.NEAREST),(xx,yy+45));local.paste(Image.fromarray(c.PALETTE[target[y:y+256,x:x+256]]).resize((lw,lw),Image.Resampling.NEAREST),(xx,yy+45+lw+8))
        ref=rgb.copy();root=revised>0;ref[root]=np.rint(.35*rgb[root]+.65*c.PALETTE[revised[root]]).astype('uint8');Image.fromarray(ref).save(a.run/'data/revision'/(t+'_RGB_with_annotation.png'))
    legend='Blue: T1 baseline | Purple: persistent | Green: newly observed | Yellow: terminal extension | Gray: untyped review'
    d.text((margin,full.height-48),legend,font=small,fill='#333333')
    ld.text((margin,local.height-78),'Blue: T1 baseline | Purple: persistent | Green: newly observed',font=small,fill='#333333')
    ld.text((margin,local.height-38),'Yellow: terminal extension | Gray: untyped review',font=small,fill='#333333')
    for name,canvas in [('four_period_RGB_original_reference_three_models',full),('same_validation_window_background_and_roots',local)]:
        canvas.save(figures/(name+'.png'),dpi=(300,300));canvas.save(figures/(name+'.tif'),compression='tiff_lzw',dpi=(600,600))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,3,figsize=(10,3.1),constrained_layout=True)
    for ax,m in zip(axs,c.MODELS):
        history=c.rows(a.run/m/'history.csv');xx=[int(r['epoch']) for r in history];ax.plot(xx,[float(r['train_loss']) for r in history],color='#0072B2',label='Training loss');right=ax.twinx();vv=[r for r in history if r['validation_macro_Dice']];right.plot([int(r['epoch']) for r in vv],[float(r['validation_macro_Dice']) for r in vv],color='#D55E00',marker='o',ms=3,label='Validation consistency');right.set_ylim(0,1);ax.set(title=c.MODEL_NAMES[m],xlabel='Epoch',ylabel='Training objective');right.set_ylabel('Teacher-reference macro Dice');ax.grid(alpha=.15)
    fig.savefig(figures/'learning_curves.png',dpi=300);fig.savefig(figures/'learning_curves.tif',dpi=600,pil_kwargs={'compression':'tiff_lzw'});plt.close(fig)
    lines=['# Four-period U-Net-aligned temporal proposal and three-model continuation','','Original source annotations are preserved byte-for-byte. The full upper-frame proposal follows the fixed V19 U-Net root mask and structured temporal classes, except weak ambiguous junction pixels kept temporally untyped. No artificial label error or random perturbation was introduced.','','Three previously trained binary detectors stay frozen. Each branch continues its paired 31-channel temporal head in PyTorch for 16 epochs. This is not end-to-end retraining of three backbone architectures. Geometry and RGB-only features plus the train-only V19 routers are reused unchanged.','','150 training / 15 validation / 18 test adjacent-date pairs (50 / 5 / 6 spatial views). The same position and dates retain the same spatial partition. Train views include overlapping shifted views; these are not independent plants or tubes. T1 baseline is displayed but excluded from the three-class temporal metric average.','','Checkpoint and continuity-strength selection use validation agreement with the U-Net-derived proposal. This measures teacher consistency, not independent segmentation or biological accuracy. Original annotations are audited separately; neither set supplies an external independent test. External independent images: 0. Historical same-field exposure applies.','','|Model|Internal test teacher consistency Dice|Internal test original-label agreement Dice|Selected epoch|','|---|---:|---:|---:|']
    for m in c.MODELS:
        rr=[r for r in summary if r['model']==c.MODEL_NAMES[m] and r['split']=='test'];a1=next(r for r in rr if r['reference']=='U_Net_derived_proposal');a2=next(r for r in rr if r['reference']=='preserved_original_annotation');lines.append(f"|{c.MODEL_NAMES[m]}|{a1['Dice']:.4f}|{a2['Dice']:.4f}|{a1['selected_epoch']}|")
    lines+=['','Full Precision/Recall/Dice/IoU: three_model_metrics_summary.csv and three_model_class_metrics.csv. Before/after values use identical references and spatial windows. Figures show all dates at identical native aspect ratio and one fixed validation crop. Full-frame visualizations include training regions. Single categorical ownership is enforced in each raster; candidate-root grammar restricts NEW vs PERSISTENT-with-terminal-EXTENSION. Putative axes and ambiguous crossings do not establish uniquely verified biological root identities.','','The standalone annotation_review/annotate.html embeds four RGB frames and the proposal; review, undo, 60x zoom and JSON saving remain available. Source provenance and original annotation are retained.']
    (a.run/'RESULTS.md').write_text('\n'.join(lines)+'\n');print('REPORT_SAVED',json.dumps(summary),flush=True)

if __name__=='__main__':main()
