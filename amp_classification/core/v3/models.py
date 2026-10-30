import numpy as np
from scipy.special import logit
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier,GradientBoostingClassifier,AdaBoostClassifier

def build(name,c,seed,threads=3,rounds=None):
    n=int(rounds or c.get('max_iter',1000))
    if name.startswith('CatBoost'):
        from catboost import CatBoostClassifier
        params={k:v for k,v in c.items() if k not in ['feature','max_iter','patience','positive_weight']}
        if c.get('positive_weight',1.)!=1.:params['class_weights']=[1.,c['positive_weight']]
        return CatBoostClassifier(**params,iterations=n,loss_function='Logloss',eval_metric='AUC',thread_count=threads,
                   random_seed=seed,verbose=False,allow_writing_files=False)
    if name=='XGBoost':
        from xgboost import XGBClassifier
        params={k:v for k,v in c.items() if k not in ['feature','max_iter','depth']}
        return XGBClassifier(**params,max_depth=c['depth'],n_estimators=n,tree_method='hist',device='cpu',
                    objective='binary:logistic',eval_metric='auc',n_jobs=threads,random_state=seed)
    if name=='HistGradientBoosting':
        return HistGradientBoostingClassifier(**{k:v for k,v in c.items() if k not in ['feature','max_iter']},max_iter=n,early_stopping=False,random_state=seed)
    if name=='GBDT':
        return GradientBoostingClassifier(**{k:v for k,v in c.items() if k not in ['feature','max_iter','depth']},
                    max_depth=c['depth'],n_estimators=n,random_state=seed)
    if name=='AdaBoost':
        return AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=c['depth'],random_state=seed),
                    n_estimators=n,learning_rate=c['learning_rate'],algorithm='SAMME',random_state=seed)
    if name=='RBFSVM':return make_pipeline(StandardScaler(),SVC(C=c['C'],gamma=c['gamma'],kernel='rbf',
                    probability=False,cache_size=512,max_iter=-1,random_state=seed))
    if name=='EBM':
        from interpret.glassbox import ExplainableBoostingClassifier
        return ExplainableBoostingClassifier(**{k:v for k,v in c.items() if k!='feature'},
                   n_jobs=threads,random_state=seed,validation_size=.15,early_stopping_rounds=100)
    raise ValueError(name)

def score(model,X,threads=1):
    if model.__class__.__module__.startswith('catboost'):
        out=logit(np.clip(model.predict_proba(X,thread_count=threads)[:,1],1e-7,1-1e-7))
    elif hasattr(model,'decision_function'):out=model.decision_function(X)
    else:out=logit(np.clip(model.predict_proba(X)[:,1],1e-7,1-1e-7))
    out=np.asarray(out,float).reshape(-1);assert len(out)==len(X) and np.isfinite(out).all();return out

def choose_rounds(name,c,seed,X,y,Xv,yv,threads):
    if name in ['RBFSVM','EBM']:return None,[]
    model=build(name,c,seed,threads);trace=[]
    if name.startswith('CatBoost'):
        model.fit(X,y,eval_set=(Xv,yv),early_stopping_rounds=c.get('patience',100),use_best_model=True)
        hist=model.get_evals_result()['validation']['AUC']
        return int(model.tree_count_),[float(v) for v in hist]
    if name=='XGBoost':
        model.set_params(early_stopping_rounds=50)
        model.fit(X,y,eval_set=[(Xv,yv)],verbose=False)
        return int(model.best_iteration+1),[float(v) for v in model.evals_result()['validation_0']['auc']]
    model.fit(X,y)
    best=-np.inf;chosen=1
    for i,s in enumerate(model.staged_decision_function(Xv),1):
        auc=float(roc_auc_score(yv,np.asarray(s).reshape(-1)));trace.append(auc)
        if auc>best+1e-12:best=auc;chosen=i
        if i-chosen>=50:break
    return chosen,trace

def audit_params(name,c,model):
    if name.startswith('CatBoost'):
        p=model.get_all_params()
        for k,v in c.items():
            if k in ['feature','max_iter','patience','positive_weight']:continue
            assert k in p,(k,p)
            if isinstance(v,(int,float)):assert np.isclose(p[k],v,atol=1e-6,rtol=1e-6),(k,p[k],v)
            else:assert p[k]==v,(k,p[k],v)
        w=c.get('positive_weight',1.)
        if w!=1.:np.testing.assert_allclose(p['class_weights'],[1.,w])
        else:assert not p.get('class_weights') or p['class_weights']==[1,1]
        return p
    import json
    p=model.get_params(deep=False);out={}
    for k,v in p.items():
        try:json.dumps(v,allow_nan=False);out[k]=v
        except (TypeError,ValueError):out[k]=repr(v)
    return out

