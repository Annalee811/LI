#!/usr/bin/env python3
"""Native-size four-period root segmentation and separately learned temporal state."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
import train_uploaded_upper as core
from train_pair_temporal import root_model,paired_model

COLORS={0:[255,255,255],1:[147,51,234],2:[22,163,74],3:[234,179,8],4:[0,0,0],255:[150,150,150]}
NAMES={0:'background',1:'persistent relative to previous period',2:'newly observed since previous period',3:'elongating since previous period',4:'T1 baseline roots',255:'undetermined'}

def pair_whole(model,previous,current,previous_probability,current_probability,device):
    h,w=current.shape[:2];xs=list(range(0,w-255,192));ys=list(range(0,h-255,192))
    if xs[-1]!=w-256:xs.append(w-256)
    if ys[-1]!=h-256:ys.append(h-256)
    coords=[(x,y) for y in ys for x in xs];value=np.zeros((4,h,w),np.float32);weight=np.zeros((h,w),np.float32);v=np.hanning(256).astype(np.float32);blend=np.maximum(v[:,None]*v[None,:],.02)
    model.eval()
    with torch.inference_mode():
        for i in range(0,len(coords),4):
            rr=coords[i:i+4];data=[np.concatenate([previous[y:y+256,x:x+256].astype(np.float32)/255,current[y:y+256,x:x+256].astype(np.float32)/255,previous_probability[y:y+256,x:x+256,None],current_probability[y:y+256,x:x+256,None]],2) for x,y in rr];xx=torch.from_numpy(np.stack(data).transpose(0,3,1,2).copy()).to(device).float()
            for (x,y),pr in zip(rr,torch.softmax(model(xx),1).cpu().numpy()):
                value[:,y:y+256,x:x+256]+=pr*blend[None];weight[y:y+256,x:x+256]+=blend
    assert (weight>0).all();return value/weight[None]

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--originals',type=Path,required=True);p.add_argument('--upper-only',action='store_true',help='Use exported per-period original_RGB_upper.png files');a=p.parse_args();torch.set_num_threads(4)
    selection=json.loads((a.run/'three_model_selection.json').read_text());arch=selection['winner'];checkpoint=(a.dataset if arch=='unet' else a.run/'segformer')/'training/best.pt'
    threshold=next(r['threshold'] for r in selection['scores'] if r['model_id']==arch);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root=root_model(arch,checkpoint).to(device).eval();pair=paired_model(arch,checkpoint).to(device);pc=torch.load(a.run/'temporal/best.pt',map_location=device,weights_only=True);pair.load_state_dict(pc['state_dict']);pair.eval()
    original=json.loads((a.run/'original_human_progress.json').read_text());expected={r['name']:r['image_sha256'] for r in original['images']}
    upper_hashes={r['period']:r['original_upper_rgb_file_sha256'] for r in json.loads((a.run/'four_period_prediction_summary.json').read_text())['periods']} if a.upper_only else {}
    destination=a.run/'four_period_predictions';destination.mkdir(exist_ok=True);arrays={};root_probs={};summary=[];panels=[];overlays=[]
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',32)
    for i in range(1,5):
        t=f'T{i}';src=a.originals/t/'original_RGB_upper.png' if a.upper_only else a.originals/f'{t}_multiregion_RGB.png';assert hashlib.sha256(src.read_bytes()).hexdigest()==(upper_hashes[t] if a.upper_only else expected[src.name])
        rgb=np.array(Image.open(src).convert('RGB'))[:1108].copy();arrays[t]=rgb;folder=destination/t;folder.mkdir(exist_ok=True);Image.fromarray(rgb).save(folder/'original_RGB_upper.png')
        prob=core.whole_image(root,rgb,device);root_probs[t]=prob;binary=prob>=threshold;Image.fromarray(binary.astype(np.uint8)*255).save(folder/'root_mask.png');Image.fromarray(np.round(np.clip(prob,0,1)*65535).astype(np.uint16)).save(folder/'root_probability_16bit.png')
        if i==1:
            raw=np.where(binary,4,0).astype(np.uint8);confident=raw.copy();confidence=prob
        else:
            pr=pair_whole(pair,arrays[f'T{i-1}'],rgb,root_probs[f'T{i-1}'],prob,device);conditional=pr[1:]/np.maximum(pr[1:].sum(0),1e-8);rank=np.sort(conditional,axis=0);confidence=rank[-1];raw=np.where(binary,conditional.argmax(0)+1,0).astype(np.uint8)
            confident=raw.copy();uncertain=binary&((confidence<.55)|((rank[-1]-rank[-2])<.15)|(pr[1:].sum(0)<.20));confident[uncertain]=255
        Image.fromarray(raw).save(folder/'temporal_class_raw.png');Image.fromarray(confident).save(folder/'temporal_class_with_undetermined.png');Image.fromarray(np.round(confidence*255).astype(np.uint8)).save(folder/'temporal_confidence_8bit.png')
        color=np.full((*binary.shape,3),255,np.uint8)
        for k,c in COLORS.items():color[confident==k]=c
        Image.fromarray(color).save(folder/'root_only_temporal_map.png',dpi=(600,600));overlay=rgb.copy();overlay[binary]=(overlay[binary]*.45+color[binary]*.55).astype(np.uint8);Image.fromarray(overlay).save(folder/'RGB_temporal_overlay.png',dpi=(300,300))
        panels.append(color);overlays.append(overlay)
        counts={str(k):int((confident==k).sum()) for k in [1,2,3,4,255]}
        summary.append(dict(period=t,original_upper_rgb_file_sha256=hashlib.sha256((folder/'original_RGB_upper.png').read_bytes()).hexdigest(),root_pixels=int(binary.sum()),root_area_fraction=float(binary.mean()),predicted_state_pixels=counts,undetermined_fraction_within_predicted_root=float((confident==255).sum()/max(1,binary.sum())),human_annotation_status='reviewed upper mask available' if i<=2 else 'not fully reviewed; prediction not quantitatively validated',comparison_reference='baseline' if i==1 else f'T{i-1} to {t}'))
        print('FOUR_PERIOD_PREDICTED',t,'ROOT_PIXELS',int(binary.sum()),flush=True)
    width=2541;height=1108;footer=90
    for images,name in [(panels,'four_period_root_changes.png'),(overlays,'four_period_RGB_overlays.png')]:
        canvas=Image.new('RGB',(4*width,height+60+footer),'white');draw=ImageDraw.Draw(canvas)
        for j,im in enumerate(images):canvas.paste(Image.fromarray(im),(j*width,60));draw.text((j*width+20,8),f'T{j+1}: model prediction',font=font,fill='#20352d')
        x=20
        for k,label in [(4,'T1 baseline'),(1,'Persistent'),(2,'Newly observed'),(3,'Elongating'),(255,'Undetermined')]:
            draw.rectangle((x,height+82,x+30,height+112),fill=tuple(COLORS[k]));draw.text((x+42,height+72),label,font=font,fill='#20352d');x+=600
        canvas.save(a.run/name,dpi=(600,600))
    core.save_json(a.run/'four_period_prediction_summary.json',dict(position='0353-2-B3',winner=arch,root_threshold=threshold,roi=[0,0,2541,1108],state_encoding=NAMES,periods=summary,temporal_model_trained_transition='T1 to T2 only',T3_T4_temporal_accuracy='not evaluated; no complete human labels',persistent_definition='Present relative to the preceding period; does not imply origin in T1',individual_root_ID_tracking=False,death_inference=False,temporal_uncertainty_rule='max conditional probability <0.55 or top two margin <0.15 or paired root mass <0.20',human_progress_preserved=True))
if __name__=='__main__':main()
