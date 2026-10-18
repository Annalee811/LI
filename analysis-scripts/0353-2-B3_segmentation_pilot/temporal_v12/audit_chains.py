"""Along-chain smoothness and false switches, always accompanied by pixel scores."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import common as c
from coherence import chains,fragmentation,decode

def false_switches(pred,gt,mask):
    _,nodes,paths,_=chains(mask);wrong=same=0
    for path in paths:
        xy=nodes[path];g=gt[tuple(xy.T)];p=pred[tuple(xy.T)];valid=(g[:-1]>0)&(g[:-1]<255)&(g[:-1]==g[1:]);same+=int(valid.sum());wrong+=int((valid&(p[:-1]!=p[1:])).sum())
    return dict(same_human_class_chain_edges=same,false_switches_on_same_human_class=wrong)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);a=p.parse_args();rows=[]
    for n in c.MODELS:
        for split,col in [('validation',3),('repeated_benchmark',4)]:
            for version,root in [('v11',a.previous),('v12',a.run)]:
                for t in c.PERIODS[1:]:
                    base=root/n/'evaluation'/split;pred=np.array(Image.open(base/f'{t}_classes.png'));gt=np.array(Image.open(base/f'{t}_human_target.png'));gate=pred>0
                    rows.append(dict(model=n,split=split,version=version,period=t,**fragmentation(pred,gate),**false_switches(pred,gt,gate)))
    c.csvout(a.run/'chain_coherence.csv',rows);summary=[]
    for n in c.MODELS:
        for split in ['validation','repeated_benchmark']:
            for version in ['v11','v12']:
                rr=[r for r in rows if (r['model'],r['split'],r['version'])==(n,split,version)];counts={k:sum(r[k] for r in rr) for k in ['chain_count','chain_edges','class_switches','runs_shorter_than_5px','same_human_class_chain_edges','false_switches_on_same_human_class']};summary.append(dict(model=n,split=split,version=version,**counts,switches_per_1000_edges=1000*counts['class_switches']/max(1,counts['chain_edges'])))
    c.js(a.run/'chain_coherence_summary.json',dict(records=summary,switches_include_real_class_boundaries=True,false_switch_definition='Adjacent predicted skeleton nodes both on the same typed human class but with different predicted labels',cannot_prove_root_identity=True))
    # A genuine terminal change must survive while isolated weak flicker vanishes.
    mask=np.zeros((40,100),bool);mask[10,2:98]=True;mask[25,2:98]=True;prob=np.full((3,40,100),.025,np.float32);prob[0,10,:]=.95;prob[1,25,:]=.95;prob[:,10,40:43]=np.array([.4,.5,.1])[:,None];prob[:,10,80:98]=np.array([.02,.02,.96])[:,None];y=decode(prob,mask,4)
    assert (y[10,2:80]==1).all() and (y[10,80:98]==3).all() and (y[25,2:98]==2).all() and np.array_equal(y>0,mask)
    crossing=np.zeros((40,40),bool)
    for x in range(5,35):crossing[x,x]=True;crossing[x,39-x]=True
    _,_,paths,adj=chains(crossing);assert sum(len(v)>=3 for v in adj)>0 and len(paths)>=4
    p=np.full((3,40,40),.05,np.float32);p[0]=.9
    for x in range(5,35):p[:,x,39-x]=[.05,.9,.05]
    y=decode(p,crossing,12)
    assert all(y[x,x]==1 for x in range(5,15)) and all(y[x,39-x]==2 for x in range(5,15)) and np.array_equal(y>0,crossing)
    c.js(a.run/'geometry_verification.json',dict(passed=True,weak_short_flicker_removed=True,genuine_terminal_growth_retained=True,parallel_roots_not_merged=True,root_support_exact=True,junctions_split_chains=True,X_crossing_incident_classes_separate=True))
    support=[]
    for n in c.MODELS:
        for col in [3,4]:
            for k in [1,2,3]:
                total=detected=0
                for t in c.PERIODS[1:]:
                    gt=np.array(Image.open(a.run/'data/targets'/f'{t}_temporal_target.png'))[:,c.COLS[col]:c.COLS[col]+448];gate=np.array(Image.open(a.run/'data/root_gates'/n/f'C{col}'/f'{t}.png'))>0;total+=int((gt==k).sum());detected+=int(((gt==k)&gate).sum())
                support.append(dict(model=n,column=col,root_class=c.NAMES[k],human_pixels=total,detected_by_frozen_root_model=detected,root_detection_recall=detected/total if total else None))
    c.js(a.run/'root_support_by_class.json',support)
    print('CHAIN_AUDIT_READY',flush=True)
if __name__=='__main__':main()
