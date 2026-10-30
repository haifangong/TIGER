import numpy as np
from sklearn.metrics import (roc_auc_score,average_precision_score,confusion_matrix,matthews_corrcoef,
                            log_loss,brier_score_loss)

def metrics(y,p,threshold=.5):
    y=np.asarray(y,dtype=int);p=np.asarray(p,float);pred=p>=threshold
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    div=lambda a,b:float(a/b) if b else None
    sens=div(tp,tp+fn);spec=div(tn,tn+fp)
    return {'n':len(y),'positive':int(y.sum()),'negative':int(len(y)-y.sum()),
            'AUROC':float(roc_auc_score(y,p)) if len(set(y))==2 else None,
            'AP_active':float(average_precision_score(y,p)) if y.sum() else None,
            'AP_inactive':float(average_precision_score(1-y,1-p)) if (1-y).sum() else None,
            'MCC':float(matthews_corrcoef(y,pred)) if len(set(y))==2 else None,
            'sensitivity':sens,'specificity':spec,'balanced_accuracy':(sens+spec)/2 if sens is not None and spec is not None else None,
            'precision':div(tp,tp+fp),'NPV':div(tn,tn+fn),'F1':div(2*tp,2*tp+fp+fn),
            'accuracy':float(np.mean(y==pred)),'log_loss':float(log_loss(y,np.clip(p,1e-7,1-1e-7),labels=[0,1])),
            'Brier':float(brier_score_loss(y,p)),'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),'threshold':float(threshold)}

def thresholds(y,p):
    # Every distinct decision set, including predict-none and predict-all.
    unique=np.unique(p)
    candidates=np.r_[0.,(unique[:-1]+unique[1:])/2,np.nextafter(1.,2.)]
    rows=[metrics(y,p,t) for t in candidates]
    balanced=max(rows,key=lambda r:(r['MCC'],r['specificity'],r['threshold']))
    recall=max([r for r in rows if r['sensitivity']>=.9],key=lambda r:(r['specificity'],r['MCC'],r['threshold']))
    return {'balanced':balanced['threshold'],'recall90':recall['threshold'],'default':.5}, {'balanced':balanced,'recall90':recall,'default':metrics(y,p)}
