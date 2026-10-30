import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score,roc_curve,brier_score_loss,log_loss
def metrics(y,p,threshold=.5):
    y=np.asarray(y,int);p=np.asarray(p,float);h=p>=threshold
    tp=int(((y==1)&h).sum());fn=int(((y==1)&~h).sum());fp=int(((y==0)&h).sum());tn=int(((y==0)&~h).sum())
    div=lambda a,b:float(a/b) if b else None
    den=((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**.5
    both=len(np.unique(y))==2
    return dict(n=len(y),positive=int(y.sum()),negative=int((1-y).sum()),tp=tp,fp=fp,tn=tn,fn=fn,
      AUROC=float(roc_auc_score(y,p)) if both else None,
      AP_active=float(average_precision_score(y,p)) if y.sum() else None,
      AP_inactive=float(average_precision_score(1-y,-p)) if (1-y).sum() else None,
      MCC=(tp*tn-fp*fn)/den if den else 0.,sensitivity=div(tp,tp+fn),specificity=div(tn,tn+fp),
      precision=div(tp,tp+fp),NPV=div(tn,tn+fn),F1=div(2*tp,2*tp+fp+fn),
      nonAMP_F1=div(2*tn,2*tn+fp+fn),Brier=float(brier_score_loss(y,p)),
      log_loss=float(log_loss(y,np.clip(p,1e-7,1-1e-7),labels=[0,1])),threshold=float(threshold))
def thresholds(y,p):
    y=np.asarray(y,int);p=np.asarray(p,float);u=np.unique(p)
    cuts=np.r_[0.,(u[:-1]+u[1:])/2,np.nextafter(1.,2.)]
    order=np.argsort(p,kind='stable');prefix=np.r_[0,np.cumsum(y[order])]
    i=np.searchsorted(p[order],cuts,side='left');fn=prefix[i];tn=i-fn;tp=y.sum()-fn;fp=len(y)-i-tp
    den=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)).astype(float)
    mcc=np.divide(tp*tn-fp*fn,den,out=np.zeros(len(den)),where=den>0)
    recall=tp/y.sum();spec=tn/(len(y)-y.sum())
    best=max(range(len(cuts)),key=lambda j:(mcc[j],spec[j],cuts[j]))
    out={'balanced':float(cuts[best]),'default':.5}
    for target in [.90,.95,.98]:
        eligible=np.flatnonzero(recall>=target)
        j=max(eligible,key=lambda j:(spec[j],mcc[j],cuts[j]))
        out['recall'+str(round(100*target))]=float(cuts[j])
    return out,{k:metrics(y,p,v) for k,v in out.items()}
def specificity_at_recall(y,score,target=.95):
    fpr,tpr,_=roc_curve(y,score,drop_intermediate=False)
    return float(np.max((1-fpr)[tpr>=target]))
def classwise(y,p,cut):
    r=metrics(y,p,cut)
    return [{'class':'AMP','AUROC':r['AUROC'],'AUPRC':r['AP_active'],'Precision':r['precision'],'Recall':r['sensitivity'],'F1_score':r['F1'],'support':r['positive'],
             'TP':r['tp'],'FP':r['fp'],'FN':r['fn'],'TN':r['tn']},
            {'class':'non-AMP','AUROC':r['AUROC'],'AUPRC':r['AP_inactive'],'Precision':r['NPV'],'Recall':r['specificity'],'F1_score':r['nonAMP_F1'],'support':r['negative'],
             'TP':r['tn'],'FP':r['fn'],'FN':r['fp'],'TN':r['tp']}]

