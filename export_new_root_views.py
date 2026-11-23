"""Lossless single-class derivation of existing native predictions and RGB views."""
from pathlib import Path
import csv,json,hashlib,io,sys
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import fitz
W=Path(__file__).resolve().parent
O=Path('/workspace/work/V90_all_microtubes_20261118')
STUDY=Path('/workspace/work/V90_validation_20261123')
RGBROOT=Path('/workspace/work/all_series_RGB_PDF_20261105/sources')
F=Path('/workspace/work/root_workflow_real_assets_20261109/fonts')
sys.path.insert(0,'/workspace/LI/analysis-scripts')
from dataset_identity import require_active_position,tube_id
COLOR=(216,153,49)
def font(n,bold=False):return ImageFont.truetype(str(F/('Times New Roman Bold.ttf'if bold else'Times New Roman.ttf')),n)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def csvout(p,rr):
 with p.open('w',encoding='utf-8-sig',newline='')as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def overlay(rgb,new):
 result=rgb.copy();result[new]=np.rint(.92*np.array(COLOR)+.08*rgb[new]).astype(np.uint8)
 assert np.array_equal(result[~new],rgb[~new]);return result
def page_image(pos,records):
 canvas=Image.new('RGB',(2600,1260),'white');d=ImageDraw.Draw(canvas);by_t={r['period']:r for r in records}
 d.text((35,20),pos+' | Newly observed roots',font=font(43,True),fill='black')
 for j,t in enumerate(['T1','T2','T3','T4']):d.text((480+j*570,90),t,font=font(32,True),fill='black')
 for i,label in enumerate(['Original RGB','New on RGB','New mask']):
  y=150+i*305;d.text((22,y+110),label,font=font(29,True),fill='black')
  for j,t in enumerate(['T1','T2','T3','T4']):
   x=285+j*570
   if t not in by_t:
    d.rectangle((x,y,x+550,y+285),outline='#DDDDDD');d.text((x+145,y+115),'Not acquired',font=font(29),fill='black');continue
   r=by_t[t]
   path=Path(r['RGB_path'])if i==0 else W/r['overlay_path']if i==1 else W/r['mask_path']
   if i==2:
    new=np.array(Image.open(path))==255;im=Image.fromarray(np.where(new,20,255).astype(np.uint8)).convert('RGB')
   else:im=Image.open(path).convert('RGB')
   im.thumbnail((550,280),Image.Resampling.LANCZOS);canvas.paste(im,(x+(550-im.width)//2,y+(285-im.height)//2))
   if r['baseline_period']:
    d.text((x+130,y+268),'Baseline; new class not assessed',font=font(21),fill='black')
 d.rectangle((35,1088,68,1114),fill=COLOR);d.text((83,1080),'Newly observed root',font=font(29,True),fill='black')
 kind=records[0]['prediction_kind'];caption='Original automatic B3 output; historical human-label agreement is reported separately.'if kind=='original_V90_B3_automatic'else'Frozen V90 transfer-adapter prediction; human temporal ground truth unavailable.'
 d.text((35,1130),caption,font=font(27),fill='black')
 d.text((35,1166),'Only original class 2 is highlighted. Native masks and root ownership are exported separately; no classes are merged.',font=font(26),fill='black')
 if pos=='Q1-1-Z2':bottom='Coarse localization at T2/T3: descriptive preview only; excluded from primary independent testing.'
 elif len(records)<4:bottom='Missing dates remain absent. New status is relative to the previous observed period.'
 else:bottom='T1 is a baseline observation, not evidence of absent root formation. Prediction display is not independent accuracy.'
 d.text((35,1202),bottom,font=font(26),fill='black');return canvas
def main():
 active=rows(Path('/workspace/LI/dataset/current/active_images.csv'));assert len(active)==66
 positions=sorted(set(require_active_position(r['position_id'])for r in active));assert len(positions)==18
 dest=W/'all_series_new_only';dest.mkdir(exist_ok=True);manifest=[];doc=fitz.open()
 for pos in positions:
  rr=sorted([r for r in active if r['position_id']==pos],key=lambda r:r['period']);folder=dest/pos;folder.mkdir(exist_ok=True)
  for ix,r in enumerate(rr):
   t=r['period'];rgbpath=RGBROOT/r['canonical_view_relative_path'];assert sha(rgbpath)==r['registered_RGB_SHA256']
   rgb=np.array(Image.open(rgbpath).convert('RGB'))
   src=O/'B3_verified_inference/junction/full'if pos=='0353-2-B3'else O/'delivery/V90_transfer_candidates'/pos
   p=src/(t+'_classes.png');original=np.array(Image.open(p));assert original.shape==rgb.shape[:2]
   new=original==2
   if ix==0:assert not new.any(),'First observation must remain a baseline'
   maskpath=folder/(t+'_new_mask.png');viewpath=folder/(t+'_new_RGB_overlay.png');whitepath=folder/(t+'_new_root_RGB_on_white.png')
   Image.fromarray(new.astype(np.uint8)*255).save(maskpath)
   Image.fromarray(overlay(rgb,new)).save(viewpath)
   white=np.full_like(rgb,255);white[new]=rgb[new];Image.fromarray(white).save(whitepath)
   ownpath=O/'B3_verified_inference/junction/ownership'/(t+'_axis_owner.npz')if pos=='0353-2-B3'else src/(t+'_ownership.npz')
   owner={};unknown=0
   if ownpath.exists():
    for k,v in dict(np.load(ownpath)).items():
     if v.dtype==np.bool_:owner[k]=v&new
     else:owner[k]=np.where(new,v,-1).astype(v.dtype)
    unknown=int((new&(owner['axis']<0)).sum());np.savez_compressed(folder/(t+'_new_ownership.npz'),**owner)
   row=dict(position_id=pos,tube_id=tube_id(pos),period=t,previous_observed_period=rr[ix-1]['period']if ix else'',baseline_period=ix==0,
    width=rgb.shape[1],height=rgb.shape[0],new_pixel_count=int(new.sum()),new_pixels_with_unassigned_axis=unknown,
    prediction_kind='original_V90_B3_automatic'if pos=='0353-2-B3'else'V90_transfer_adapter_prediction',
    accuracy_status='historical_B3_agreement_separate'if pos=='0353-2-B3'else'NOT_EVALUATED_NO_HUMAN_GT',
    coarse_localization_flag=pos=='Q1-1-Z2',RGB_path=str(rgbpath),RGB_SHA256=sha(rgbpath),
    source_class_path=str(p),source_class_SHA256=sha(p),new_mask_SHA256=sha(maskpath),
    mask_path=str(maskpath.relative_to(W)),overlay_path=str(viewpath.relative_to(W)),new_root_RGB_path=str(whitepath.relative_to(W)))
   manifest.append(row)
  image=page_image(pos,[r for r in manifest if r['position_id']==pos]);image.save(folder/'New_root_T1-T4.jpg',quality=97,subsampling=0,dpi=(300,300))
  b=io.BytesIO();image.save(b,format='JPEG',quality=96,subsampling=0);page=doc.new_page(width=780,height=378);page.insert_image(page.rect,stream=b.getvalue())
  print('NEW_ONLY_SERIES',pos,len(rr),flush=True)
 doc.save(W/'All_18_series_new_roots_only.pdf',deflate=True);doc.close()
 csvout(W/'All_66_new_root_prediction_manifest.csv',manifest)
 # Preserve the newly run baseline/ablation comparisons as single-class masks too.
 modelrows=[];mdest=W/'baseline_ablation_new_masks';mdest.mkdir(exist_ok=True)
 for p in sorted((STUDY/'independent_predictions').rglob('*_classes.png')):
  rel=p.relative_to(STUDY/'independent_predictions');out=mdest/rel.parent/(p.stem.replace('_classes','_new_mask')+'.png');out.parent.mkdir(parents=True,exist_ok=True)
  source=np.array(Image.open(p));new=source==2;Image.fromarray(new.astype(np.uint8)*255).save(out)
  modelrows.append(dict(source=str(rel),new_mask_path=str(out.relative_to(W)),new_pixel_count=int(new.sum()),source_SHA256=sha(p),mask_SHA256=sha(out),accuracy_status='NOT_EVALUATED_NO_HUMAN_GT'))
 assert len(modelrows)==442;csvout(W/'Baseline_ablation_442_new_masks_index.csv',modelrows)
 (W/'New_only_export_checks.json').write_text(json.dumps(dict(passed=True,series=len(positions),frames=len(manifest),comparison_masks=len(modelrows),
  mask_definition='original class == 2; 0/255 PNG',other_classes_not_merged=True,native_dimensions_preserved=True,
  RGB_bytes_verified_against_active_manifest=True,overlay_pixels_outside_new_mask_unchanged=True,
  root_identity_values_preserved_on_new_pixels=True,baseline_new_class_assessed=False,missing_periods_not_synthesized=True,
  independent_accuracy_computed=False,model_weights_or_annotations_modified=False,training_or_inference_run=False),indent=2))
 print('ALL_NEW_ONLY_VIEWS_COMPLETE',len(manifest),len(modelrows),flush=True)
if __name__=='__main__':main()
