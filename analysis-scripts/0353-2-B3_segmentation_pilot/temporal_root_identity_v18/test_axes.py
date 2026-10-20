"""Nontrivial anatomical-grammar checks with crossings, laterals and tip growth."""
import numpy as np
from scipy import ndimage as ndi
import axes
from structured import decode

def make(lines):
 m=np.zeros((128,160),bool);rgb=np.full((128,160,3),65,np.uint8)
 for (y0,x0),(y1,x1),color in lines:
  n=max(abs(y1-y0),abs(x1-x0))+1;p=np.rint(np.linspace([y0,x0],[y1,x1],n)).astype(int);m[tuple(p.T)]=True;rgb[tuple(p.T)]=color
 m=ndi.binary_dilation(m,iterations=1);rgb[m&(rgb[:,:,0]==65)]=180
 return m,rgb
# Smooth horizontal pre-existing root, crossing a wholly new vertical root.
prev,prgb=make([((64,15),(64,140),210)]);mask,rgb=make([((64,15),(64,140),210),((15,80),(112,80),175)])
a=axes.build(prev,prgb);b=axes.build(mask,rgb);link=axes.match(a,b,prgb,rgb);prob=np.full((3,*mask.shape),.05,np.float32);prob[0]=.9;prob[1,:,78:83]=.9;prob[0,:,78:83]=.05;prob/=prob.sum(0)
pred,records=decode(prob,a,b,link,1.);assert pred[64,30]==1 and pred[30,80]==2
assert len([p for p in b['paths'] if len(p)>40])==2,'Crossing must have two long axes'
# A lateral arm at a T junction remains a distinct new axis.
mask,rgb=make([((64,15),(64,140),210),((20,80),(64,80),175)]);b=axes.build(mask,rgb);link=axes.match(a,b,prgb,rgb);q=np.zeros((3,*mask.shape),np.float32)+.05;q[0]=.85;q[1,:60,78:83]=.9;pred,rr=decode(q,a,b,link,1);assert pred[35,80]==2 and pred[64,35]==1
# Strongly contradictory local NEW/GROWTH logits cannot break one accepted axis.
prev,prgb=make([((64,20),(64,80),210)]);mask,rgb=make([((64,20),(64,130),210)]);a=axes.build(prev,prgb);b=axes.build(mask,rgb);link=axes.match(a,b,prgb,rgb);q=np.zeros((3,*mask.shape),np.float32)+.02;q[0]=.95;q[2,:,81:]=.96;q[0,:,81:]=.02;q[1,:,112:]=.97;q[2,:,112:]=.02;q/=q.sum(0);pred,rr=decode(q,a,b,link,1)
assert pred[64,40]==1 and pred[64,95]==3;assert not any(v['new_nodes'] and v['growth_nodes'] for v in rr);assert pred[64,120]!=2
# Parallel roots are distinct; no whole-component collapse.
mask,rgb=make([((35,15),(35,140),210),((43,15),(43,140),175)]);b=axes.build(mask,rgb);assert b['axis_map'][35,70]!=b['axis_map'][43,70]
print('PASS: crossing axes, new lateral, terminal-only growth, no NEW/GROWTH mixture, parallel separation')
# A short visibility gap/crossing can separate an extension from the old shaft.
prev,prgb=make([((64,20),(64,80),210)]);mask,rgb=make([((64,85),(64,130),210)]);a=axes.build(prev,prgb);b=axes.build(mask,rgb,a);link=axes.match(a,b,prgb,rgb);q=np.full((3,*mask.shape),.02,np.float32);q[2]=.96;pred,rr=decode(q,a,b,link,0.);assert pred[64,110]==3,'An outward old-tip continuation is growth even if its old shaft is outside this trace'
# A lateral rooted on the old shaft cannot be relabeled growth merely because
# the local network outputs GROWTH; no prior tip supports that interpretation.
mask,rgb=make([((59,50),(20,50),210)]);b=axes.build(mask,rgb,a);link=axes.match(a,b,prgb,rgb);pred,rr=decode(q,a,b,link,0.);assert pred[35,50]==2
print('PASS: disconnected tip continuation versus new lateral at old shaft')
# The final learned-emission decoder must keep a new lateral and old shaft
# separate, even when local growth logits are spuriously high at a crossing.
from continuity import decode as learned
prev,prgb=make([((64,15),(64,140),210)]);mask,rgb=make([((64,15),(64,140),210),((20,80),(64,80),175)]);a=axes.build(prev,prgb);b=axes.build(mask,rgb,a);link=axes.match(a,b,prgb,rgb);q=np.full((3,*mask.shape),.01,np.float32);q[0]=.9;q[1,:60,78:83]=.95;q[0,:60,78:83]=.01;q[2,40:45,78:83]=.99;pred,rr=learned(q,a,b,link,.5);assert pred[64,35]==1 and pred[30,80]==2 and pred[42,80]==2;assert not any(z['new_nodes'] and z['growth_nodes'] for z in rr)
# Preserve learned terminal growth without requiring a perfect binary old-tip.
prev,prgb=make([((64,20),(64,80),210)]);mask,rgb=make([((64,20),(64,130),210)]);a=axes.build(prev,prgb);b=axes.build(mask,rgb,a);link=axes.match(a,b,prgb,rgb);q=np.full((3,*mask.shape),.01,np.float32);q[0]=.95;q[2,:,85:]=.98;q[0,:,85:]=.01;pred,rr=learned(q,a,b,link,.5);assert pred[64,40]==1 and pred[64,110]==3
# Exercise the actual optional router interface on crossing geometry.
router=dict(mean=[0.]*12,scale=[1.]*12,coef=[0.]*12,intercept=0.);mask,rgb=make([((64,15),(64,140),210),((15,80),(112,80),175)]);b=axes.build(mask,rgb,a,router=router);assert len([p for p in b['paths'] if len(p)>40])==2
print('PASS: final learned decoder, NEW/GROWTH exclusion, terminal growth and learned-router crossing interface')
