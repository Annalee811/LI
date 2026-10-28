"""Stratify frozen identity predictions by upload18 endpoint classes, no retraining."""
import csv,hashlib,json,sys,shutil
from pathlib import Path
import numpy as np
ROOT=Path('/workspace/LI');HERE=Path(__file__).resolve().parent
RUN=ROOT/'results/B3_root_association_GAT_v71_20261028'
GEOMETRY=ROOT/'results/B3_temporal_GAT_new_growth_v68_20261028/data'
OUT=ROOT/'results/B3_three_class_evaluation_v72_20261028'
MODELS=['unet','deeplab','segformer'];NAMES={'unet':'U-Net','deeplab':'DeepLabV3+','segformer':'SegFormer-B0'}
CLASSES=[(1,'newly_observed','新生根'),(2,'tip_extension','根尖延伸'),(0,'persistent','持续根')]
sys.path.insert(0,str(RUN/'source'));from metrics import binary_curve
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvout(p,data):
    with Path(p).open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
def score(y,p,threshold):
    pred=p>=threshold;tp=int((pred&(y==1)).sum());fp=int((pred&(y==0)).sum());fn=int((~pred&(y==1)).sum());tn=int((~pred&(y==0)).sum())
    curve=binary_curve(y==1,p) if len(y) else {}
    return dict(connection_Precision=tp/(tp+fp) if tp+fp else None,connection_Recall=tp/(tp+fn) if tp+fn else None,connection_F1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,connection_AP=curve.get('average_precision'),connection_AUC=curve.get('AUC'),TP=tp,FP=fp,FN=fn,TN=tn,positive_pairs=int((y==1).sum()),negative_pairs=int((y==0).sum()),candidate_pairs=len(y))
def main():
    sys.path.insert(0,str(ROOT/'analysis-scripts'));from dataset_identity import require_active_position
    assert require_active_position('0353-2-B3')=='0353-2-B3'
    OUT.mkdir(exist_ok=True);(OUT/'source').mkdir(exist_ok=True)
    allrows=[];periodrows=[];scopes=[];inputs=[]
    for m in MODELS:
        g=dict(np.load(RUN/'data'/m/'association.npz'));N=len(g['x']);counts=np.zeros((N,3),np.int64)
        for t in ['T2','T3','T4']:
            p=GEOMETRY/m/(t+'_pixels.npz');px=dict(np.load(p));valid=(px['split']==2)&(px['target']>=0)&(px['target']<3)
            np.add.at(counts,(px['node'][valid],px['target'][valid]),1);inputs.append(dict(path=str(p),sha256=sha(p)))
        totals=counts.sum(1);purity=counts.max(1)/np.maximum(totals,1);nodeclass=np.where((totals>=3)&(purity>=.9),counts.argmax(1),-1)
        a,b=g['pairs'].T;heldout=g['split']==2
        assert (g['target'][heldout]>=0).all()
        for kind in ['gat','mlp']:
            folder=RUN/(m+'_'+kind);done=json.loads((folder/'completion.json').read_text());assert done['epochs_completed']==100
            assert sha(folder/'best.pt')==done['checkpoint_sha256']
            probabilities=np.load(folder/'inference.npz')['probability'];threshold=done['test']['threshold']
            for cls,key,chinese in CLASSES:
                # An edge may enter more than one class stratum. Each edge is counted
                # ONCE within a stratum, even if both endpoints belong to that class.
                mask=heldout&((nodeclass[a]==cls)|(nodeclass[b]==cls))
                aa,bb=a[mask],b[mask];cross=(nodeclass[aa]!=nodeclass[bb])&(nodeclass[aa]>=0)&(nodeclass[bb]>=0)
                metric=score(g['target'][mask],probabilities[mask],threshold)
                allrows.append(dict(model=m,model_name=NAMES[m],variant=kind,category=key,category_Chinese=chinese,threshold=threshold,**metric,
                    cross_class_same_root_positive_pairs=int((cross&(g['target'][mask]==1)).sum()),category_anchor_nodes=int(((nodeclass==cls)&(g['node_split']==2)).sum()),
                    scope='Internal spatial holdout; unique candidate edges incident to a pure reference-class node; same-root labels from17; endpoint category from18'))
                for pair_scope,phase_condition in [('within_period',g['phase'][a]==g['phase'][b]),('across_adjacent_periods',g['phase'][a]!=g['phase'][b])]:
                    subset=mask&phase_condition
                    periodrows.append(dict(model=m,model_name=NAMES[m],variant=kind,category=key,category_Chinese=chinese,pair_scope=pair_scope,threshold=threshold,**score(g['target'][subset],probabilities[subset],threshold)))
            inputs.extend(dict(path=str(p),sha256=sha(p)) for p in [folder/'completion.json',folder/'inference.npz',folder/'best.pt'])
        scopes.append(dict(model=m,pure_class_nodes=int((nodeclass>=0).sum()),mixed_or_insufficient_nodes=int(((totals>0)&(nodeclass<0)).sum()),class_reference_pixel_counts=counts.sum(0).tolist(),all_heldout_identity_edges=int(heldout.sum())))
    csvout(OUT/'Three_classes_connection_metrics.csv',allrows)
    csvout(OUT/'Three_classes_connection_by_period.csv',periodrows)
    # Pixel classifier has not been retrained in V71. Report original frozen
    # full-frame classifier metrics in a separate explicitly labeled baseline table.
    old=ROOT/'results/B3_temporal_GAT_new_growth_v68_20261028/Temporal_class_metrics_before_after.csv'
    with old.open(encoding='utf-8-sig') as f:baseline=[dict(r,evaluation_status='Frozen V66 temporal baseline, full-frame inference; NOT a V71 GAT temporal result') for r in csv.DictReader(f) if r['variant']=='baseline']
    csvout(OUT/'Three_classes_original_temporal_baseline.csv',baseline);inputs.append(dict(path=str(old),sha256=sha(old)))
    report=dict(passed=True,trained_weights_or_thresholds_changed=False,new_training=False,classes_separately_evaluated=['newly_observed','tip_extension','persistent'],
        identity_reference='Uploaded17 user_defined142 IDs, unchanged',category_reference='Uploaded18 unchanged; training never used these classes in V71',
        class_definition='At least3 heldout annotated root pixels in a node and >=90% share one reference class; T1 baseline is not a change-class anchor',
        edge_definition='Both ends have scoreable heldout identity targets; at least one endpoint has the indicated reference category. Cross-category same-root positive links are retained.',
        strata_not_disjoint=True,strata_counts_must_not_be_added=True,threshold_definition='One validation-selected threshold per branch, reused unchanged for all three classes; no per-class test tuning',
        temporal_pixel_Dice_is_not_connection_F1=True,new_GAT_temporal_segmentation_metrics_available=False,
        evaluation_scope='One B3 development position; internal spatial holdout and transductive graph, not independent generalization',coverage=scopes,immutable_inputs=inputs)
    for x in inputs:assert sha(x['path'])==x['sha256']
    js(OUT/'verification.json',report)
    for p in HERE.glob('*.py'):shutil.copy2(p,OUT/'source'/p.name)
    print(json.dumps(allrows,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
