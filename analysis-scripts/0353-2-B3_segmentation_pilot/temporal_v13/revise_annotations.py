"""Apply explicit image-reviewed temporal corrections; never use model predictions."""
import argparse,json,copy,shutil
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import common as c

def unpack(v,shape):return np.repeat(np.asarray(v[::2],np.uint8),np.asarray(v[1::2],np.int64)).reshape(shape)
def pack(a):
    a=a.ravel();starts=np.r_[0,np.flatnonzero(a[1:]!=a[:-1])+1];ends=np.r_[starts[1:],len(a)];return np.stack([a[starts].astype(np.int64),ends-starts],1).ravel().tolist()

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--review',type=Path,required=True);a=p.parse_args();data=a.data;original=json.loads((data/'original_human_progress.json').read_text());revised=copy.deepcopy(original);decisions=json.loads((a.review/'accepted_decisions.json').read_text());records=[]
    assert c.sha(data/'original_human_progress.json')=='27943e5f9d087bf718668bba0a39058dc9a4871de6c75c5ccc27febad4c06a09','Correction decisions apply only to the reviewed annotation version'
    for idx,t in enumerate(c.PERIODS):
        im=revised['images'][idx];shape=(im['height'],im['width']);mask=unpack(im['layers']['mask'],shape);temporal=unpack(im['layers']['temporal'],shape);before=temporal.copy();upper=temporal[:c.HEIGHT];root=mask[:c.HEIGHT]==255
        labels,_=ndi.label((upper==1)&root,structure=np.ones((3,3)))
        for d in [r for r in decisions if r['current']==t]:
            change=labels==d['component'];assert change.any();assert np.all(upper[change]==1);upper[change]=3
            yy,xx=np.where(change);r=dict(**d,from_raw_code=1,to_raw_code=3,from_class='newly_observed',to_class='persistent',pixels=int(change.sum()),bbox=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)],basis='Assistant visual review of adjacent original RGB and existing user traces; not model agreement');records.append(r)
            Image.fromarray(change.astype(np.uint8)*255).save(a.review/(d['review_id']+'_changed_pixels.png'))
        assert np.array_equal(before[c.HEIGHT:],temporal[c.HEIGHT:]);im['layers']['temporal']=pack(temporal)
        for layer in ['mask','color','viability']:assert im['layers'][layer]==original['images'][idx]['layers'][layer]
        Image.fromarray(upper).save(data/'upper'/f'{t}_temporal.png')
        if t!='T1':
            Image.fromarray(c.target(mask[:c.HEIGHT],upper)).save(data/'targets'/f'{t}_temporal_target.png')
            original_targets=data/'original_reference/targets';original_targets.mkdir(parents=True,exist_ok=True)
            Image.fromarray(c.target(mask[:c.HEIGHT],before[:c.HEIGHT])).save(original_targets/f'{t}_temporal_target.png')
    revised['annotation_revision']=dict(version='v13_temporal_review_1',original_sha256=c.sha(data/'original_human_progress.json'),review_origin='Assistant visual correction of user annotations, authorized by user; not an independent expert reference',changed_pixels=sum(r['pixels'] for r in records),regions=len(records),model_predictions_used=False,death_added=False,root_background_masks_changed=False,lower_part_changed=False)
    c.js(data/'active_progress.json',revised);c.js(a.review/'correction_log.json',dict(**revised['annotation_revision'],records=records,unresolved='R03,R05,R06,R07,R08 retained: crossings, nearby parallel roots or partial previous overlap do not prove persistence'))
    checks=json.loads((data/'source_integrity.json').read_text());checks['files']={str(f.relative_to(data)):c.sha(f) for f in sorted((data/'upper').glob('*.png'))};checks['active_progress_sha256']=c.sha(data/'active_progress.json');checks['labels_revised_by_visual_review']=True;c.js(data/'source_integrity.json',checks)
    rows=[]
    for t in c.PERIODS[1:]:
        y=np.array(Image.open(data/'targets'/f'{t}_temporal_target.png'));rows.append(dict(period=t,**{c.NAMES[k]:int((y==k).sum()) for k in [1,2,3]},untyped_root_pixels=int((y==255).sum())))
    c.csvout(data/'human_class_counts.csv',rows);shutil.copyfile(a.review/'correction_log.json',data/'correction_log.json');print('ANNOTATIONS_REVISED',revised['annotation_revision'],flush=True)
if __name__=='__main__':main()
