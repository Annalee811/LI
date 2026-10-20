"""Join strongly aligned root fragments across junction cores without filling RGB/masks."""
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

def stitch(axis,rgb,mask):
 xy=axis['xy'];paths=axis['paths'];ends=axis['endkinds'];width=ndi.distance_transform_edt(mask);color=rgb.astype('float32')/255;items=[];near_foreground=ndi.maximum_filter(mask,size=3)
 for ai,path in enumerate(paths):
  if len(path)<8:continue
  for side,kind in enumerate(ends[ai]):
   if kind not in ['junction','unresolved','tip']:continue
   pp=xy[path if side==0 else path[::-1]];p0=pp[0].astype(float);dirs=[]
   for step in [8,16,28]:
    d=p0-pp[min(step,len(pp)-1)];dirs.append(d/max(np.linalg.norm(d),1e-6))
   sample=pp[min(2,len(pp)-1):min(18,len(pp))];appearance=np.median(color[tuple(sample.T)],0);wid=float(np.median(width[tuple(sample.T)]));items.append(dict(axis=ai,side=side,point=p0,dirs=np.array(dirs),color=appearance,width=wid,kind=kind))
 if not items:return axis
 tree=cKDTree(np.array([q['point'] for q in items]));candidates=[]
 for i,j in sorted(tree.query_pairs(18)):
  a,b=items[i],items[j]
  if a['axis']==b['axis']:continue
  delta=b['point']-a['point'];dist=float(np.linalg.norm(delta));alignment=-(a['dirs']*b['dirs']).sum(1);align=float(np.median(alignment))
  if align<.82 or alignment[0]<.55:continue
  if dist>2:
   u=delta/dist
   if max(a['dirs']@u)<.1 or max(b['dirs']@(-u))<.1:continue
  ratio=abs(np.log((a['width']+.5)/(b['width']+.5)))
  if ratio>1.0:continue
  ca=a['color']-a['color'].mean();cb=b['color']-b['color'].mean();chroma=float(np.linalg.norm(ca-cb))
  if chroma>.16:continue
  line=np.rint(a['point'][None]+np.linspace(0,1,max(3,int(dist)+1))[:,None]*delta).astype(int);support=float(near_foreground[tuple(line.T)].mean())
  if support<.85:continue
  # All tips may be virtually linked, but no root pixel or RGB value is added.
  da=a['dirs'][0];db=b['dirs'][0];lateral=(abs(da[0]*delta[1]-da[1]*delta[0])+abs(db[0]*delta[1]-db[1]*delta[0]))/2/(.5*(a['width']+b['width'])+1)
  cost=.55*(1-align)+.08*ratio+.07*min(chroma/.16,1)+.08*dist/18+.04*(1-support)+.10*min(lateral,3)
  if cost<.30:candidates.append(dict(i=i,j=j,cost=cost,distance=dist,alignment=align,gap_support=support))
 neighbors={i:[] for i in range(len(items))}
 for q in candidates:
  neighbors[q['i']].append(q);neighbors[q['j']].append(q)
 parent=list(range(len(paths)))
 def find(x):
  while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
  return x
 links={};accepted=[];occupied=set()
 for q in sorted(candidates,key=lambda q:(q['cost'],q['i'],q['j'])):
  i,j=q['i'],q['j'];a,b=items[i],items[j]
  if i in occupied or j in occupied or find(a['axis'])==find(b['axis']):continue
  alternatives=[]
  for endpoint in [i,j]:
   alt=[v['cost'] for v in neighbors[endpoint] if v is not q and v['i'] not in occupied and v['j'] not in occupied];alternatives.append(min(alt+[1.]))
  margin=min(alternatives)-q['cost']
  if margin<.035:continue
  aa=(a['axis'],a['side']);bb=(b['axis'],b['side']);links[aa]=bb;links[bb]=aa;occupied|={i,j};parent[find(a['axis'])]=find(b['axis']);accepted.append(dict(kind='post_junction_fragment_stitch',old_axes=[a['axis']+1,b['axis']+1],sides=[a['side'],b['side']],endpoints=[a['point'].tolist(),b['point'].tolist()],margin=float(margin),**{k:v for k,v in q.items() if k not in ['i','j']},root_pixels_filled=0,basis='Mutually separated multi-scale tangent, RGB/width and existing foreground support'))
 used=set();newpaths=[];newends=[];mapping=np.zeros(len(paths)+1,np.int32);groups=[]
 starts=[(i,s) for i in range(len(paths)) for s in [0,1] if (i,s) not in links]+[(i,0) for i in range(len(paths))]
 for ai,side in starts:
  if ai in used:continue
  cur,ss=ai,side;nodes=[];members=[];first=ends[cur][ss];last=first
  while cur not in used:
   used.add(cur);members.append(cur+1);mapping[cur+1]=len(newpaths)+1;nodes.extend(paths[cur] if ss==0 else paths[cur][::-1]);last=ends[cur][1-ss];other=links.get((cur,1-ss))
   if other is None:break
   cur,ss=other
  newpaths.append(np.array(nodes,np.int32));newends.append([first,last]);groups.append(members)
 out=dict(xy=xy,paths=newpaths,endkinds=newends,axis_map=mapping[axis['axis_map']],uncertain=axis['uncertain'].copy(),decisions=axis['decisions']+accepted,fragment_groups=groups)
 assert np.array_equal(out['axis_map']>0,mask);return out
