#!/usr/bin/env python3
"""Research checks: preserve thin roots, verify label-free inference inputs and source pixels."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import protocol as p

def main():
 a=argparse.ArgumentParser();a.add_argument('--data',type=Path,required=True);a.add_argument('--output',type=Path,required=True);args=a.parse_args();checks={}
 score=np.zeros((64,64),np.float32);score[10,10:30]=.6;score[40:42,40:42]=.6;score[48:51,48:51]=.95
 cfg=dict(threshold=.5,hysteresis_delta=0.,compact_area=20);mask,removed=p.clean(score,cfg,True)
 assert mask[10,10:30].all() and not mask[40:42,40:42].any() and mask[48:51,48:51].all() and removed.sum()==4
 checks['one_pixel_wide_long_root_preserved']=True;checks['low_confidence_compact_noise_removed']=True;checks['high_confidence_compact_candidate_preserved']=True
 score[:]=0;score[20,5:35]=.45;score[20,15:25]=.8;score[45,45]=.45
 mask=p.clean(score,dict(threshold=.6,hysteresis_delta=.2,compact_area=0));assert mask[20,5:35].all() and not mask[45,45]
 checks['weak_connected_root_reconstructed_without_isolated_weak_noise']=True
 data=p.Data(args.data);inputs=[]
 def capture(ims):inputs.append(hashlib.sha256(ims.tobytes()).hexdigest());return ims[:,:,:,0]
 before=data.column(capture,3);before_inputs=inputs.copy();inputs.clear()
 for t in p.PERIODS:data.gt[t][:]=255
 data.roots={};after=data.column(capture,3)
 assert inputs==before_inputs and all(np.array_equal(before[t],after[t]) for t in p.PERIODS)
 checks['inference_RGB_tensors_do_not_depend_on_human_masks']=True
 p.js(args.output/'method_verification.json',dict(passed=True,checks=checks,source_provenance_SHA256=p.sha(args.data/'dataset/provenance.json'),
  note='Synthetic filter checks and actual tensor hashes; not extra accuracy observations'))
 print('V6_METHOD_VERIFIED',len(checks),flush=True)
if __name__=='__main__':main()
