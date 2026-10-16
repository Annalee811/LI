#!/usr/bin/env python3
"""Validate an adjacent-period pipeline, then anchor rollout on manual T1/T2.

T3 uses T2 manual roots; T4 uses T3 predicted probability. No T3 strokes are used.
Only validation scores select components; all test scores are reported afterwards.
"""
from pathlib import Path
import argparse,copy,csv,json,hashlib
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy import ndimage as ndi
import torch
import train_uploaded_upper as core
from train_pair_temporal import paired_model,metrics as temporal_metrics
from train_sequential_joint import make_model,features,infer_tiles,target_map,topology,ENCODING
from train_balanced_temporal import make_temporal

PALETTE={0:[255,255,255],1:[147,51,234],2:[22,163,74],3:[234,179,8],4:[37,99,235],255:[148,163,184]}

def old_conditional(model,previous,current,prior,root,device,coordinates=None,blend=False):
    image=np.concatenate([previous.astype(np.float32)/255,current.astype(np.float32)/255,prior[...,None],root[...,None]],2);h,w=prior.shape
    if coordinates is None:
        xs=list(range(0,w-255,192));ys=list(range(0,h-255,192))
        if xs[-1]!=w-256:xs.append(w-256)
        if ys[-1]!=h-256:ys.append(h-256)
        coordinates=[(x,y) for y in ys for x in xs]
    total=np.zeros((4,h,w),np.float32);count=np.zeros((h,w),np.float32);v=np.hanning(256).astype(np.float32);weight=np.maximum(v[:,None]*v[None,:],.02) if blend else np.ones((256,256),np.float32)
    model.eval()
    with torch.inference_mode():
        for i in range(0,len(coordinates),4):
            rr=coordinates[i:i+4];xx=torch.from_numpy(np.stack([image[y:y+256,x:x+256].transpose(2,0,1) for x,y in rr]).copy()).to(device)
            for (x,y),prob in zip(rr,torch.softmax(model(xx),1).cpu().numpy()):total[:,y:y+256,x:x+256]+=prob*weight;count[y:y+256,x:x+256]+=weight
    assert (count>0).all();prob=total/count[None];return prob[1:]/np.maximum(prob[1:].sum(0),1e-8)

def rle(array):
    data=array.ravel();starts=np.r_[0,np.flatnonzero(data[1:]!=data[:-1])+1];ends=np.r_[starts[1:],len(data)];return np.column_stack((data[starts],ends-starts)).ravel().tolist()

def decode(item,key):return np.repeat(np.asarray(item['layers'][key][::2],np.uint8),item['layers'][key][1::2]).reshape(item['height'],item['width'])

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--previous-run',type=Path,required=True);p.add_argument('--run',type=Path,required=True);a=p.parse_args();a.run.mkdir(exist_ok=True,parents=True);torch.set_num_threads(4);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    binary=a.dataset/'training/best.pt';baseline=core.SmallUNet(12).to(device);baseline.load_state_dict(torch.load(binary,map_location=device,weights_only=True)['state_dict']);baseline.eval()
    joint=make_model(binary).to(device);joint.load_state_dict(torch.load(a.run/'training/best.pt',map_location=device,weights_only=True)['state_dict']);joint.eval()
    old=paired_model('unet',binary).to(device);old.load_state_dict(torch.load(a.previous_run/'temporal/best.pt',map_location=device,weights_only=True)['state_dict']);old.eval()
    balanced=make_temporal(binary,a.previous_run/'temporal/best.pt').to(device);balanced.load_state_dict(torch.load(a.run/'balanced_temporal/best.pt',map_location=device,weights_only=True)['state_dict']);balanced.eval()
    original_path=a.previous_run/'original_human_progress.json';original=json.loads(original_path.read_text());by_period={i['name'][:2]:i for i in original['images']}
    with (a.dataset/'dataset/manifest.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    for r in rows:
        for k in ['column','x','y']:r[k]=int(r[k])
    rgb={t:np.array(Image.open(a.dataset/f'dataset/upper/{t}_RGB.png')) for t in ['T1','T2']};masks={t:np.array(Image.open(a.dataset/f'dataset/upper/{t}_mask.png')) for t in rgb};human_temporal=np.array(Image.open(a.dataset/'dataset/upper/T2_temporal.png'));target=target_map(masks['T2'],human_temporal);arrays={t:(rgb[t],{'mask':masks[t]}) for t in rgb}
    def column_inputs(split,col):
        seeds={t:prob for (t,c),(prob,_) in core.fused_columns(baseline,rows,arrays,split,device).items()};sl=slice(core.COLS[col],core.COLS[col]+448)
        coords=[(r['x']-core.COLS[col],r['y']) for r in rows if r['column']==col and r['period']=='T2'];prior=(masks['T1'][:,sl]==255).astype(np.float32)
        result=infer_tiles(joint,features(rgb['T1'][:,sl],rgb['T2'][:,sl],prior,seeds['T2']),device,coords)
        return seeds,sl,coords,prior,result
    seeds,sl,coords,prior,joint_result=column_inputs('val',3)
    # Both root alternatives receive an equal validation-only threshold search.
    root_candidates=[]
    for name,prob in [('single_frame_unet',seeds['T2']),('sequential_joint_unet',joint_result[0])]:
        options=[]
        for thr in np.arange(.1,.901,.05):
            thr=float(round(thr,2));options.append({'threshold':thr,**core.metrics(core.confusion(masks['T2'][:,sl],prob>=thr))})
        best=max(options,key=lambda r:(r['Dice'],-abs(r['threshold']-.5)));root_candidates.append({'model':name,**best})
    root_choice=max(root_candidates,key=lambda r:r['Dice']);root_name=root_choice['model'];threshold=root_choice['threshold'];chosen_root=seeds['T2'] if root_name=='single_frame_unet' else joint_result[0]
    cond_old=old_conditional(old,rgb['T1'][:,sl],rgb['T2'][:,sl],prior,chosen_root,device,coords)
    balanced_conditional=infer_tiles(balanced,features(rgb['T1'][:,sl],rgb['T2'][:,sl],prior,seeds['T2']),device,coords)[1:]
    conditionals={'previous_temporal_unet':cond_old,'sequential_joint_unet':joint_result[1:],'balanced_temporal_unet':balanced_conditional};class_candidates=[]
    for name,conditional in conditionals.items():
        pred=np.where(chosen_root>=threshold,conditional.argmax(0)+1,0).astype(np.uint8);class_candidates.append({'model':name,**temporal_metrics(target[:,sl],pred)})
    class_choice=max(class_candidates,key=lambda r:r['macro_Dice_supported_classes']);class_name=class_choice['model']
    # This is a functional input check, not an accuracy score or a simulated label.
    no_prior=infer_tiles(joint,features(rgb['T1'][:,sl],rgb['T2'][:,sl],np.zeros_like(prior),seeds['T2']),device,coords)
    dependency={'validation_joint_root_mean_absolute_change_when_previous_root_removed':float(np.abs(joint_result[0]-no_prior[0]).mean()),'validation_joint_class_mean_absolute_change_when_previous_root_removed':float(np.abs(joint_result[1:]-no_prior[1:]).mean())}
    balanced_no_prior=infer_tiles(balanced,features(rgb['T1'][:,sl],rgb['T2'][:,sl],np.zeros_like(prior),seeds['T2']),device,coords)[1:]
    dependency['validation_balanced_class_mean_absolute_change_when_previous_root_removed']=float(np.abs(balanced_conditional-balanced_no_prior).mean())
    selection={'selection_data':'validation T1 to T2 only, same C3 spatial column as earlier pilot','selected_root_model':root_name,'selected_temporal_model':class_name,'frozen_root_threshold':threshold,'root_candidates':root_candidates,'temporal_candidates':class_candidates,'input_check':dependency,'manual_anchor_evaluation':'T1 manual mask is provided as an earlier-period input; no T2 validation annotation enters model input','comparison_note':'This anchored protocol differs from v2, which used a predicted T1 prior and a threshold chosen from both periods. Compare v3 alternatives within this protocol. Repeated exploratory use of the same validation split does not provide independent confirmation.','T3_T4_used_for_model_selection':False,'class_definition':ENCODING,'browning_or_death_inferred':False}
    core.save_json(a.run/'selection.json',selection)
    reports={}
    for split,col in [('val',3),('test',4)]:
        if split=='val':ss,section,cc,pr,jr=seeds,sl,coords,prior,joint_result
        else:ss,section,cc,pr,jr=column_inputs(split,col)
        prob=ss['T2'] if root_name=='single_frame_unet' else jr[0]
        if class_name=='previous_temporal_unet':conditional=old_conditional(old,rgb['T1'][:,section],rgb['T2'][:,section],pr,prob,device,cc)
        elif class_name=='balanced_temporal_unet':conditional=infer_tiles(balanced,features(rgb['T1'][:,section],rgb['T2'][:,section],pr,ss['T2']),device,cc)[1:]
        else:conditional=jr[1:]
        pred=np.where(prob>=threshold,conditional.argmax(0)+1,0).astype(np.uint8);gt=target[:,section]
        reports[split]={'root':core.metrics(core.confusion(masks['T2'][:,section],prob>=threshold)),'temporal':temporal_metrics(gt,pred),'unique_evaluated_pixels':int((masks['T2'][:,section]!=128).sum()),'reference_mask':'earlier T1 manual annotation in same spatial split','target_period':'T2'}
        folder=a.run/'evaluation'/split;folder.mkdir(parents=True,exist_ok=True);Image.fromarray(pred).save(folder/'T2_temporal_prediction.png');Image.fromarray(gt).save(folder/'T2_human_temporal.png');Image.fromarray((prob>=threshold).astype(np.uint8)*255).save(folder/'T2_root_prediction.png');np.save(folder/'T2_root_probability.npy',prob.astype(np.float32));np.save(folder/'T2_conditional_probability.npy',conditional.astype(np.float32))
    core.save_json(a.run/'evaluation/metrics.json',reports)
    # Deploy with both reviewed periods fixed; do not overwrite the user's manual work.
    previous_rgb=None;previous_probability=None;previous_root=None;periods=[];panels=[];overlays=[];project=copy.deepcopy(original)
    for i in range(1,5):
        phase=f'T{i}';item=by_period[phase];source=a.run/'images'/f'{phase}_RGB_upper.png'
        if not source.exists():source=a.previous_run/'four_period_predictions'/phase/'original_RGB_upper.png'
        current=np.array(Image.open(source));assert current.shape==(1108,2541,3)
        out=a.run/'four_period_results'/phase;out.mkdir(parents=True,exist_ok=True);rgb_out=a.run/'images'/f'{phase}_RGB_upper.png';rgb_out.parent.mkdir(exist_ok=True);Image.fromarray(current).save(rgb_out);prior_sha=hashlib.sha256(previous_probability.tobytes()).hexdigest() if previous_probability is not None else None
        if i<=2:
            root=decode(item,'mask')[:1108]==255;prob=root.astype(np.float32);raw=np.zeros(root.shape,np.uint8)
            if i==1:raw[root]=4
            else:
                attrs=decode(item,'temporal')[:1108]
                for source_code,output_code in [(3,1),(1,2),(2,3)]:raw[root&(attrs==source_code)]=output_code
                raw[root&(attrs==0)]=255
            state=raw.copy();confidence=None;conditional=None;origin='original manual annotation, not model prediction'
        else:
            seed=core.whole_image(baseline,current,device);jr=infer_tiles(joint,features(previous_rgb,current,previous_probability,seed),device,blend=True)
            prob=seed if root_name=='single_frame_unet' else jr[0];root=prob>=threshold
            if class_name=='previous_temporal_unet':conditional=old_conditional(old,previous_rgb,current,previous_probability,prob,device,blend=True)
            elif class_name=='balanced_temporal_unet':conditional=infer_tiles(balanced,features(previous_rgb,current,previous_probability,seed),device,blend=True)[1:]
            else:conditional=jr[1:]
            rank=np.sort(conditional,axis=0);confidence=rank[-1];raw=np.where(root,conditional.argmax(0)+1,0).astype(np.uint8);state=raw.copy();state[root&((confidence<.55)|(rank[-1]-rank[-2]<.15))]=255;origin='model prediction, unreviewed'
            # Conservative display checks: retain the raw model output separately;
            # never invent an alternative class or claim improved scores by abstaining.
            prior_distance=ndi.distance_transform_edt(~previous_root)
            previous_tips=topology(previous_root.astype(np.float32))[0]
            tip_neighborhood=previous_tips>=np.exp(-(24.**2)/(2*10.**2))
            components,n=ndi.label(raw==3,np.ones((3,3),np.uint8));eligible=np.zeros(n+1,bool)
            eligible[np.unique(components[tip_neighborhood])]=True;eligible[0]=False
            unsupported_elongation=(raw==3)&~eligible[components]
            incompatible=(root&(raw==1)&(prior_distance>12))|(((raw==2)|(raw==3))&(prior_distance<=2))|unsupported_elongation
            state[incompatible]=255
            Image.fromarray(unsupported_elongation.astype(np.uint8)*255).save(out/'tip_elongation_without_prior_tip_support_mask.png')
            Image.fromarray((root&incompatible).astype(np.uint8)*255).save(out/'temporal_geometry_conflict_mask.png')
            Image.fromarray(np.round(prob*65535).astype(np.uint16)).save(out/'root_probability_16bit.png');Image.fromarray(np.round(confidence*255).astype(np.uint8)).save(out/'class_confidence_8bit.png')
            # Native root probability and maximum class confidence are saved. Full
            # class posteriors can be reproduced from the supplied weights/scripts.
            for name in ['persistent','new_lateral','tip_elongation']:(out/f'{name}_conditional_probability_16bit.png').unlink(missing_ok=True)
            dest=next(x for x in project['images'] if x['name']==item['name']);layers={k:np.full((item['height'],item['width']),64 if k=='mask' else 0,np.uint8) for k in ['mask','color','temporal','viability']};layers['mask'][:1108][root]=255
            for output_code,editor_code in [(1,3),(2,1),(3,2)]:layers['temporal'][:1108][state==output_code]=editor_code
            dest['layers']={k:rle(v) for k,v in layers.items()};dest['reviewed']=False;dest['evidence']='';dest['label_provenance']='Sequential model prediction; original T3/T4 strokes ignored'
        Image.fromarray(root.astype(np.uint8)*255).save(out/'root_mask.png');Image.fromarray(raw).save(out/'temporal_class_raw.png');Image.fromarray(state).save(out/'temporal_class_with_undetermined.png')
        colored=np.full_like(current,255)
        for code,color in PALETTE.items():colored[state==code]=color
        overlay=current.copy();overlay[root]=colored[root];Image.fromarray(colored).save(out/'root_classes_white.png',dpi=(600,600));Image.fromarray(overlay).save(out/'RGB_class_overlay.png',dpi=(300,300));panels.append(colored);overlays.append(overlay)
        observations=None
        if previous_root is not None:
            valid=np.ones(root.shape,bool);valid[:8]=False;valid[-8:]=False;valid[:,:8]=False;valid[:,-8:]=False
            novel=root&(ndi.distance_transform_edt(~previous_root)>8)&valid
            missing=previous_root&(ndi.distance_transform_edt(~root)>8)&valid
            model_added=(raw==2)|(raw==3)
            Image.fromarray(novel.astype(np.uint8)*255).save(out/'first_visible_geometry_candidate_mask.png')
            Image.fromarray(missing.astype(np.uint8)*255).save(out/'previous_root_not_observed_candidate_mask.png')
            Image.fromarray(model_added.astype(np.uint8)*255).save(out/'new_total_lateral_and_tip_mask.png')
            missing_overlay=previous_rgb.copy();missing_overlay[missing]=[220,70,70];Image.fromarray(missing_overlay).save(out/'previous_root_not_observed_overlay.png')
            observations={'new_total_by_labels_pixels':int(model_added.sum()),'new_total_definition':'new lateral plus tip elongation; classified segments only','first_visible_geometry_candidate_pixels':int(novel.sum()),'previous_not_observed_candidate_pixels':int(missing.sum()),'geometry_displacement_tolerance_pixels':8,'geometry_border_exclusion_pixels':8,'geometry_not_validated':True,'interpretation':'Image-coordinate candidates only. Displacement, occlusion, missed segmentation or appearance changes can create these signals; they do not prove birth or death.','browning_status':'not evaluated; no labelled examples','death_status':'not evaluated; missing roots are not labelled dead'}
        previous_rgb=current;previous_probability=prob.astype(np.float32);previous_root=root
        periods.append({'period':phase,'origin':origin,'previous_reference':None if i==1 else f'T{i-1}','previous_reference_origin':None if i==1 else ('manual' if i<=3 else 'model prediction'),'previous_probability_sha256':prior_sha,'current_probability_sha256':hashlib.sha256(previous_probability.tobytes()).hexdigest(),'root_pixels':int(root.sum()),'class_pixels':{str(k):int((state==k).sum()) for k in [1,2,3,4,255]},'upper_original_rgb_pixels_sha256':hashlib.sha256(current.tobytes()).hexdigest(),'uncertainty_rule':None if i<=2 else 'max conditional probability <0.55 or top-two margin <0.15; additionally abstain when persistent pixels >12px from previous support, new pixels <=2px from old support, or an elongating component has no previous tip within 24px. All thresholds are exploratory native-pixel display rules, not validated anatomical decisions; reported metrics use raw unfiltered predictions.','manual_labels_used_to_draw_this_period':i<=2,'change_observations':observations,'width':2541,'height':1108})
        print('SEQUENTIAL_SAVED',phase,origin,int(root.sum()),flush=True)
    for images,name in [(panels,'four_period_root_classes.png'),(overlays,'four_period_RGB_classes.png')]:
        canvas=Image.new('RGB',(5082,2486),'white');draw=ImageDraw.Draw(canvas);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',32)
        for j,img in enumerate(images):
            x=(j%2)*2541;y=(j//2)*1168;canvas.paste(Image.fromarray(img),(x,y+60));draw.text((x+20,y+10),f'T{j+1}: '+('human labels' if j<2 else f'predicted from T{j}'),font=font,fill='#20352d')
        x=20
        for code,text in [(4,'T1 baseline'),(1,'Persistent'),(2,'New root'),(3,'Tip elongation'),(255,'Undetermined')]:
            draw.rectangle((x,2390,x+30,2420),fill=tuple(PALETTE[code]));draw.text((x+45,2380),text,font=font,fill='#20352d');x+=650
        canvas.save(a.run/name,dpi=(600,600))
    project['model_prediction_provenance']={'manual_T1_T2_preserved_exactly':True,'T3_reference':'T2 original manual roots','T4_reference':'T3 predicted roots','previous_T3_T4_strokes_ignored':True,'reviewed':False,'root_model':root_name,'temporal_model':class_name,'browning_death_prediction':False};core.save_json(a.run/'editable_T1T2_manual_T3T4_sequential.json',project)
    upper_project=copy.deepcopy(project)
    for item in upper_project['images']:
        phase=item['name'][:2];layers={k:decode(item,k)[:1108].copy() for k in item['layers']};image_path=a.run/'images'/f'{phase}_RGB_upper.png'
        item['name']=image_path.name;item['width']=2541;item['height']=1108;item['image_sha256']=core.sha(image_path);item['layers']={k:rle(v) for k,v in layers.items()}
    upper_project['source_scope']='Native-pixel upper ROI; T1/T2 labels cropped without modification; lower region excluded';core.save_json(a.run/'editable_upper_RGB_sequential.json',upper_project)
    checks=[]
    def check(name,passed):
        assert passed,name;checks.append({'name':name,'passed':True})
    for phase in ['T1','T2']:check(phase+' complete original progress object unchanged',next(x for x in project['images'] if x['name']==by_period[phase]['name'])==by_period[phase])
    for item in upper_project['images']:
        phase=item['name'][:2];full=next(x for x in project['images'] if x['name']==by_period[phase]['name'])
        check(phase+' upper editable project matches native cropped layers',all(np.array_equal(decode(item,k),decode(full,k)[:1108]) for k in item['layers']))
    check('T3 reads T2 manual probability',periods[2]['previous_probability_sha256']==periods[1]['current_probability_sha256'] and periods[2]['previous_reference_origin']=='manual')
    check('T4 reads T3 predicted probability',periods[3]['previous_probability_sha256']==periods[2]['current_probability_sha256'] and periods[3]['previous_reference_origin']=='model prediction')
    for phase in ['T3','T4']:
        item=next(x for x in project['images'] if x['name']==by_period[phase]['name']);folder=a.run/'four_period_results'/phase;mask=np.array(Image.open(folder/'root_mask.png'));state=np.array(Image.open(folder/'temporal_class_with_undetermined.png'))
        check(phase+' editable roots equal predictions with no old strokes',np.array_equal(decode(item,'mask')[:1108]==255,mask==255));check(phase+' excluded lower region remains unknown',np.all(decode(item,'mask')[1108:]==64));check(phase+' classes contained inside roots',not ((state!=0)&(mask!=255)).any())
    for split,col in [('val',3),('test',4)]:
        pred=np.array(Image.open(a.run/f'evaluation/{split}/T2_root_prediction.png'))==255;gt=masks['T2'][:,core.COLS[col]:core.COLS[col]+448];metric=core.metrics(core.confusion(gt,pred));check(split+' root metrics reproduce on unique native pixels',metric==reports[split]['root'] and reports[split]['unique_evaluated_pixels']==496384)
    check('test elongation absent is NA, not zero',reports['test']['temporal']['per_class'][2]['support_pixels']==0 and reports['test']['temporal']['per_class'][2]['Dice'] is None)
    check('joint model responds to previous root input',dependency['validation_joint_class_mean_absolute_change_when_previous_root_removed']>1e-7)
    check('selected balanced model responds to previous root input',dependency['validation_balanced_class_mean_absolute_change_when_previous_root_removed']>1e-7)
    core.save_json(a.run/'verification.json',{'passed':len(checks),'failed':0,'checks':checks});core.save_json(a.run/'four_period_summary.json',{'periods':periods,'encoding':ENCODING,'palette_rgb':PALETTE,'scope':'Only upper 2541x1108; T1/T2 manual, T3/T4 predictions; lower region excluded','root_model':root_name,'temporal_model':class_name,'T3_T4_accuracy':'not measured; no complete reference labels','individual_root_identity_tracking':False,'phenotype_definition_limit':'Model learns user-labelled tip elongation versus new lateral segments; tip/branch cues are not proof of biological identity','browning_death_inferred':False})
    print('SEQUENTIAL_DONE',json.dumps({'selection':selection,'metrics':reports,'checks':len(checks)}),flush=True)

if __name__=='__main__':main()
