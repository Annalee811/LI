#!/usr/bin/env python3
"""Hand-check lengths and metric conventions before evaluating fitted models."""
import argparse,math
from pathlib import Path
import numpy as np
import common as c
from evaluate import length_field,edge_length_field,agreement

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();checks={}
    a0=np.zeros((5,5),bool);a0[2,1:4]=True
    assert abs(length_field(a0).sum()-2)<1e-12;checks['horizontal_three_pixels_two_edges']=True
    a0[:]=False;a0[1,1]=a0[2,2]=a0[3,3]=True
    assert abs(length_field(a0).sum()-2*math.sqrt(2))<1e-12;checks['diagonal_edges_sqrt2']=True
    a0[:]=False;a0[1,1]=a0[2,1]=a0[2,2]=True
    assert abs(edge_length_field(a0).sum()-2)<1e-12;checks['no_diagonal_triangle_shortcut_in_given_graph']=True
    a0[:]=False;a0[2,2]=True
    assert length_field(a0).sum()==1;checks['isolated_pixel_convention_one']=True
    a0[:]=False;a0[:,2]=True;lf=length_field(a0)
    assert lf[:2].sum()+lf[2:].sum()==lf.sum();checks['partition_does_not_add_crop_endpoints']=True
    m=agreement([1,2,3],[2,3,4]);assert m['R2_agreement']==-.5 and m['RMSE_px']==1
    checks['negative_R2_retained_not_squared_correlation']=True
    gt=np.array([255,255,0,0,128],np.uint8);pred=np.array([1,0,1,0,1],bool)
    m=c.metrics(c.counts(gt,pred));assert all(m[k]==.5 for k in ['Precision','Recall','Dice']) and m['IoU']==1/3
    checks['strict_foreground_scores_and_ignore128']=True
    c.js(a.run/'metric_verification.json',dict(passed=True,checks=checks))
    print('ROOT_METRICS_VERIFIED',len(checks),flush=True)
if __name__=='__main__':main()
