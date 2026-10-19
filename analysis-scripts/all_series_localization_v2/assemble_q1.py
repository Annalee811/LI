"""Select disjoint Q1 region hypotheses with explicit descriptor/route evidence."""
import argparse,itertools,json,math
from pathlib import Path
import cv2,numpy as np
from PIL import Image
from localize_q1 import modes,project
from render_results import sample,sha,grid

def largest_rect(mask):
    heights=np.zeros(mask.shape[1],int);best=(0,0,0,0,0)
    for y,row in enumerate(mask):
        heights=np.where(row,heights+1,0);stack=[]
        for x in range(len(heights)+1):
            cur=int(heights[x]) if x<len(heights) else 0;start=x
            while stack and stack[-1][1]>cur:
                sx,hh=stack.pop();ww=x-sx;start=sx
                if ww>=hh and ww*hh>best[0]:best=(ww*hh,sx,y-hh+1,x,y+1)
            if not stack or stack[-1][1]<cur:stack.append((start,cur))
    return list(best[1:])

def poses_from_tree(dates,tree,options):
    poses={1:np.eye(3)}
    for _ in dates:
        for (a,b),m in zip(tree,options):
            H=np.array(m['matrix_analysis1600'])
            if a in poses and b not in poses:poses[b]=H@poses[a]
            elif b in poses and a not in poses:poses[a]=np.linalg.inv(H)@poses[b]
    return poses if len(poses)==len(dates) else None

def build(work,pos):
    raw=work/'inputs/raw'/pos;files={int(p.stem[1:]):p for p in raw.iterdir() if p.is_file()};dates=sorted(files);wh={t:Image.open(p).size for t,p in files.items()};geo=work/'q1_geometry'/pos
    desc=json.loads((geo/'descriptor_search.json').read_text());edges={tuple(int(v[1:]) for v in k.split('_to_')):[dict(m,source='RootSIFT') for m in ms] for k,ms in desc['combined_hypotheses'].items()}
    for ep in sorted((work/'roma_evidence'/pos).glob('T*_to_T*')):
        if not (ep/'completed.json').exists():continue
        a,b=[int(x[1:]) for x in ep.name.split('_to_')]
        with np.load(ep/'dense_evidence.npz') as z:d={k:z[k].astype('float32') for k in z.files}
        with np.load(ep/'sampled_matches.npz') as z:pa=z['source_xy'];pb=z['target_xy'];c=z['source_confidence']
        cb=sample(d['target_confidence'],pb.reshape(1,-1,2),wh[b]).ravel();back=(sample(d['backward'],pb.reshape(1,-1,2),wh[b])[0]+1)*[wh[a][0]/2,wh[a][1]/2]-.5
        cyc=np.linalg.norm(back-pa,axis=1)*1600/wh[a][0];ok=(c>=.5)&(cb>=.5)&(cyc<=5)
        neural=[dict(m,source='RoMa') for m in modes(pa[ok]*1600/wh[a][0],pb[ok]*1600/wh[b][0],minimum=20)]
        edges[(a,b)]+=neural
    for key,ms in edges.items():
        ms.sort(key=lambda m:(m['region_evidence_pass'],m['source']=='RootSIFT',m['anchors']),reverse=True);unique=[]
        q=np.array([[200,200],[1400,200],[200,1100],[1400,1100],[800,600]])
        for m in ms:
            if any(np.median(np.linalg.norm(project(q,np.array(m['matrix_analysis1600']))-project(q,np.array(o['matrix_analysis1600'])),axis=1))<8 for o in unique):continue
            unique.append(m)
        edges[key]=unique[:4]
    w,h=wh[1];smallw=240;smallh=round(h*smallw/w);y,x=np.mgrid[:smallh,:smallw];g=np.stack([x,y],-1)*1600/smallw
    candidates=[]
    for tree in itertools.combinations([k for k,v in edges.items() if v],len(dates)-1):
        for options in itertools.product(*(edges[k] for k in tree)):
            poses=poses_from_tree(dates,tree,options)
            if poses is None:continue
            valid=np.ones(g.shape[:2],bool)
            for t,H in poses.items():
                q=project(g,H);valid&=(q[...,0]>=0)&(q[...,0]<(1600-1600/wh[t][0]))&(q[...,1]>=0)&(q[...,1]<((wh[t][1]-1)*1600/wh[t][0]))
            if valid.sum()<valid.size*.035:continue
            box=largest_rect(valid);x0,y0,x1,y1=box
            if x1-x0<smallw*.6 or y1-y0<smallh*.03:continue
            q=np.stack(np.meshgrid(np.linspace(x0,x1,7),np.linspace(y0,y1,5)),axis=-1).reshape(-1,2)*1600/smallw
            checks=[];connected={1};eligible=[]
            for (a,b),ms in edges.items():
                pa=project(q,poses[a]);pb=project(q,poses[b]);cs=[]
                for m in ms:
                    e=float(np.median(np.linalg.norm(project(pa,np.array(m['matrix_analysis1600']))-pb,axis=1)))
                    cs.append((e,m))
                if cs:
                    err,m=min(cs,key=lambda v:v[0]);agree=err<=8
                    checks.append(dict(pair=f'T{a}_to_T{b}',region_model_disagreement_analysis1600=err,source=m['source'],anchors=m['anchors'],evidence_pass=m['region_evidence_pass'],agrees=agree))
                    if agree and m['region_evidence_pass'] and m['source']=='RootSIFT':eligible.append((a,b))
            for _ in dates:
                for a,b in eligible:
                    if a in connected:connected.add(b)
                    if b in connected:connected.add(a)
            verified=sorted(connected)==dates and len(eligible)>=len(dates)
            score=100*sum(c['agrees'] and c['evidence_pass'] and c['source']=='RootSIFT' for c in checks)+20*sum(c['agrees'] for c in checks)+sum(math.log1p(c['anchors']) for c in checks if c['agrees'])
            candidates.append(dict(poses=poses,small_box=box,score=score,checks=checks,review=not verified,tree=[f'T{a}_to_T{b}' for a,b in tree]))
    candidates.sort(key=lambda c:(not c['review'],c['score'],(c['small_box'][2]-c['small_box'][0])*(c['small_box'][3]-c['small_box'][1])),reverse=True)
    # Reduce equivalent trees before native-resolution geometry.
    distinct=[]
    for c in candidates:
        b=c['small_box'];q=np.array([[(b[0]+b[2])/2,(b[1]+b[3])/2]])*1600/smallw
        if any(all(np.linalg.norm(project(q,c['poses'][t])-project(q,d['poses'][t]))<8 for t in dates) for d in distinct):continue
        distinct.append(c)
        if len(distinct)>=20:break
    original_grid=grid([0,0,w,h]);occupied=np.zeros((h,w),bool);regions=[]
    for c in distinct:
        matrices={t:np.diag([wh[t][0]/1600]*2+[1])@H@np.diag([1600/w]*2+[1]) for t,H in c['poses'].items()};valid=~occupied
        for t,H in matrices.items():
            q=project(original_grid,H);valid&=(q[...,0]>=0)&(q[...,0]<=wh[t][0]-1)&(q[...,1]>=0)&(q[...,1]<=wh[t][1]-1)
        box=largest_rect(valid);x0,y0,x1,y1=box
        if x1-x0<w*.6 or y1-y0<h*.035 or (x1-x0)*(y1-y0)<w*h*.025:continue
        regions.append(dict(id=chr(65+len(regions)),box=box,matrices={f'T{t}':H.tolist() for t,H in matrices.items()},review_required=c['review'],source='Corrected Q1: fresh full-RoMa evidence plus multi-mode RootSIFT geometry',evidence=dict(selected_tree=c['tree'],route_checks=c['checks'],selection_score=c['score'],manual_TRE=None,definition='Descriptor/route evidence is automatic and conditional, not manual accuracy. Dense fields never render RGB.')))
        occupied[y0:y1,x0:x1]=True
        if len(regions)>=4:break
    regions.sort(key=lambda r:r['box'][1])
    for i,r in enumerate(regions):r['id']=chr(65+i)
    (geo/'selection_audit.json').write_text(json.dumps(dict(position=pos,edge_modes={f'T{a}_to_T{b}':m for (a,b),m in edges.items()},tree_candidates=len(candidates),distinct_candidates=len(distinct),selected_regions=regions),ensure_ascii=False,indent=2))
    print('Q1_REGIONS',pos,[(r['id'],r['box'],r['review_required']) for r in regions],flush=True)
    return dict(position=pos,periods=dates,regions=regions,scope='Corrected Q1 release, fresh structural localization',input_sha256={f'T{t}':sha(p) for t,p in files.items()})

def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--positions',nargs='*');a=p.parse_args();cv2.setNumThreads(1);cv2.setRNGSeed(20261019)
    registry=json.loads((a.work/'registry_non_Q1.json').read_text());prior=a.work/'registry_Q1.json'
    q1=json.loads(prior.read_text()) if prior.exists() else {}
    for folder in sorted((a.work/'inputs/raw').glob('Q1*')):
        if a.positions and folder.name not in a.positions:continue
        q1[folder.name]=build(a.work,folder.name);prior.write_text(json.dumps(q1,ensure_ascii=False,indent=2))
    registry.update(q1);(a.work/'registry_all.json').write_text(json.dumps(registry,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
