import numpy as np

def binary_curve(y,score):
 y=np.asarray(y,bool);score=np.asarray(score,float);order=np.argsort(-score,kind='stable');s=score[order];v=y[order];ends=np.r_[np.flatnonzero(s[1:]!=s[:-1]),len(s)-1];tp=np.cumsum(v)[ends].astype(float);fp=(ends+1)-tp;positive=int(y.sum());negative=len(y)-positive;recall=tp/max(positive,1);precision=tp/np.maximum(tp+fp,1);tpr=np.r_[0,recall];fpr=np.r_[0,fp/max(negative,1)];ap=float((np.diff(np.r_[0,recall])*precision).sum());auc=float(np.trapezoid(tpr,fpr));return dict(average_precision=ap if positive else None,AUC=auc if positive and negative else None,fpr=fpr,tpr=tpr,thresholds=np.r_[np.inf,s[ends]])
def adjusted_rand(truth,pred):
 _,a=np.unique(truth,return_inverse=True);_,b=np.unique(pred,return_inverse=True);ct=np.zeros((a.max()+1,b.max()+1),np.int64);np.add.at(ct,(a,b),1);choose=lambda x:(x*(x-1)/2).sum();n=len(a);total=n*(n-1)/2
 if not total:return 1.
 same=choose(ct);sa=choose(ct.sum(1));sb=choose(ct.sum(0));expect=sa*sb/total;den=(sa+sb)/2-expect
 return float((same-expect)/den) if den else 1.
if __name__=='__main__':
 assert binary_curve([0,1],[.1,.9])['AUC']==1
 assert binary_curve([0,1],[.9,.1])['AUC']==0
 assert binary_curve([0,1],[.5,.5])['AUC']==.5
 assert binary_curve([0,1],[.5,.5])['average_precision']==.5
 assert adjusted_rand([0,0,1,1],[2,2,3,3])==1
 assert abs(adjusted_rand([0,0,1,1],[0,1,0,1])+.5)<1e-7
 print('NUMERIC_METRICS_CHECKS_PASSED')
