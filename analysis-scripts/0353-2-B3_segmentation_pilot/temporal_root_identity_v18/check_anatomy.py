"""Functional crossings, birth-state and terminal extension checks."""
import numpy as np
from scipy import ndimage as ndi
import axes
from identity_decode import decode

def make(lines):
 mask=np.zeros((128,160),bool);rgb=np.full((128,160,3),65,np.uint8)
 for (y0,x0),(y1,x1),co in lines:
  xy=np.rint(np.linspace([y0,x0],[y1,x1],max(abs(y1-y0),abs(x1-x0))+1)).astype(int);mask[tuple(xy.T)]=True;rgb[tuple(xy.T)]=co
 mask=ndi.binary_dilation(mask,iterations=1);rgb[mask&(rgb[:,:,0]==65)]=180;return mask,rgb
prev,prgb=make([((64,15),(64,140),210)]);current,rgb=make([((64,15),(64,140),210),((15,80),(112,80),175)]);a=axes.build(prev,prgb);b=axes.build(current,rgb,a);link=axes.match(a,b,prgb,rgb);q=np.full((3,*current.shape),.02,np.float32);q[0]=.9;q[1,:,78:83]=.9;q[0,:,78:83]=.02;q[2,30:45,78:83]=.95;q[1,30:45,78:83]=.03;pred,records=decode(q,a,b,link,1)
assert pred[64,30]==1 and pred[20,80]==2 and pred[35,80]==2
assert all(not (v['new_nodes'] and (v['persistent_nodes'] or v['growth_nodes'])) for v in records)
prev,prgb=make([((64,20),(64,80),210)]);current,rgb=make([((64,20),(64,130),210)]);a=axes.build(prev,prgb);b=axes.build(current,rgb,a);link=axes.match(a,b,prgb,rgb);q=np.full((3,*current.shape),.02,np.float32);q[0]=.95;q[2,:,85:]=.98;q[0,:,85:]=.02;pred,records=decode(q,a,b,link,1);assert pred[64,40]==1 and pred[64,110]==3
# An isolated high-growth island inside the previous visible shaft cannot survive.
q[2,:,45:55]=.99;q[0,:,45:55]=.01;pred,records=decode(q,a,b,link,1);assert pred[64,50]==1
# A side root born on a shaft is not an extension of its old tip.
current,rgb=make([((20,50),(58,50),210)]);b=axes.build(current,rgb,a);link=axes.match(a,b,prgb,rgb);q=np.full((3,*current.shape),.02,np.float32);q[2]=.96;pred,records=decode(q,a,b,link,1);assert pred[30,50]==2
print('PASS: crossings, no NEW/PERSISTENT mixture, no NEW/GROWTH mixture, previous-shaft continuity, terminal-only growth, lateral versus old tip')
