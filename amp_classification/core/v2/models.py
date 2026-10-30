from common import *
import numpy as np
import time,copy,random
from scipy.special import expit,logit
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.svm import LinearSVC,LinearSVR
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,RandomForestRegressor,ExtraTreesRegressor
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score
NEURAL=['TinyCNN','Das','DeepAMP','APEX']
CLASSIC=['Logistic','LinearSVM','RF','ExtraTrees','CatBoost','LightGBM']
DEFAULT={
 'Logistic':{'feature':'F2','C':1.,'l1_ratio':.2},'LinearSVM':{'feature':'F2','C':.1},
 'RF':{'feature':'F2','trees':200,'depth':12,'min_leaf':3,'max_features':'sqrt'},
 'ExtraTrees':{'feature':'F2','trees':200,'depth':16,'min_leaf':3,'max_features':.7},
 'CatBoost':{'feature':'F2','depth':6,'learning_rate':.05,'l2_leaf_reg':5,'max_iter':1000},
 'LightGBM':{'feature':'F2','num_leaves':15,'min_child_samples':50,'learning_rate':.05,'max_iter':1000},
 'TinyCNN':{'feature':'tokens','channels':48,'dropout':.3,'lr':.001,'max_iter':100,'patience':15},
 'Das':{'feature':'tokens','hidden':100,'dropout':.3,'lr':.001,'max_iter':100,'patience':15},
 'DeepAMP':{'feature':'tokens','dropout':.5,'lr':.001,'max_iter':100,'patience':15},
 'APEX':{'feature':'tokens','hidden':128,'lr':.0005,'max_iter':100,'patience':15},
 'Cascade':{'feature':'AMP553','trees':50,'max_layers':3,'min_leaf':5},
 'Dummy':{'feature':'simple'},'SimpleLogistic':{'feature':'simple','C':1.,'l1_ratio':0.},
}

def sample_weights(y,balanced=False,base=None):
    w=np.ones(len(y)) if base is None else np.asarray(base,float).copy()
    if balanced:
        # Normalization uses only the actual fit partition, respecting weak weights.
        totals=np.bincount(y,weights=w,minlength=2)
        w*=w.sum()/(2*totals[y])
    return w

def build_classic(name,cfg,seed,rounds=None,reg=False):
    n=rounds or cfg.get('max_iter',1000)
    if name in ['Logistic','SimpleLogistic']:
        est=Ridge(alpha=1/cfg.get('C',1.)) if reg else LogisticRegression(penalty='elasticnet',solver='saga',C=cfg.get('C',1.),l1_ratio=cfg.get('l1_ratio',.2),max_iter=3000,tol=1e-4,random_state=seed,n_jobs=1)
        return make_pipeline(SimpleImputer(keep_empty_features=True),StandardScaler(),est)
    if name=='LinearSVM':
        est=LinearSVR(C=cfg.get('C',.1),dual='auto',random_state=seed,max_iter=10000) if reg else LinearSVC(C=cfg.get('C',.1),dual='auto',random_state=seed,max_iter=10000)
        return make_pipeline(SimpleImputer(keep_empty_features=True),StandardScaler(),est)
    if name in ['RF','ExtraTrees','RF_EIP']:
        cls=(ExtraTreesRegressor if reg else ExtraTreesClassifier) if name=='ExtraTrees' else (RandomForestRegressor if reg else RandomForestClassifier)
        return cls(n_estimators=cfg.get('trees',200),max_depth=cfg.get('depth',12),min_samples_leaf=cfg.get('min_leaf',3),max_features=cfg.get('max_features','sqrt'),n_jobs=4,random_state=seed)
    if name=='CatBoost':
        from catboost import CatBoostClassifier,CatBoostRegressor
        return (CatBoostRegressor if reg else CatBoostClassifier)(iterations=n,depth=cfg.get('depth',6),learning_rate=cfg.get('learning_rate',.05),l2_leaf_reg=cfg.get('l2_leaf_reg',5),
                loss_function='RMSE' if reg else 'Logloss',eval_metric='RMSE' if reg else 'AUC',thread_count=4,random_seed=seed,verbose=False,allow_writing_files=False)
    if name=='LightGBM':
        from lightgbm import LGBMClassifier,LGBMRegressor
        return (LGBMRegressor if reg else LGBMClassifier)(n_estimators=n,num_leaves=cfg.get('num_leaves',15),min_child_samples=cfg.get('min_child_samples',50),learning_rate=cfg.get('learning_rate',.05),
                reg_lambda=cfg.get('reg_lambda',1.),verbosity=-1,n_jobs=4,random_state=seed,deterministic=True,force_col_wise=True)
    if name=='Dummy':return DummyClassifier(strategy='prior',random_state=seed)
    if name=='Cascade':
        from deepforest import CascadeForestClassifier,CascadeForestRegressor
        return (CascadeForestRegressor if reg else CascadeForestClassifier)(max_layers=cfg.get('max_layers',3),n_estimators=2,n_trees=cfg.get('trees',50),max_depth=12,min_samples_leaf=cfg.get('min_leaf',5),n_jobs=4,random_state=seed,verbose=0)
    raise ValueError(name)

def score_model(model,X,reg=False,device='cpu'):
    if hasattr(model,'forward'):
        import torch
        model.eval();values=[]
        with torch.inference_mode():
            for i in range(0,len(X),256):values.append(model(torch.as_tensor(X[i:i+256],device=device)).cpu().numpy())
        score=np.concatenate(values)
    elif reg:score=model.predict(X)
    elif hasattr(model,'decision_function'):score=model.decision_function(X)
    else:score=logit(np.clip(model.predict_proba(X)[:,1],1e-7,1-1e-7))
    score=np.asarray(score).reshape(-1)
    assert len(score)==len(X) and np.isfinite(score).all()
    return -score if reg else score

def fit_classic(name,cfg,seed,X,y,weights,rounds=None,early=None,reg=False):
    model=build_classic(name,cfg,seed,rounds,reg);kw={}
    if hasattr(model,'steps'):kw[model.steps[-1][0]+'__sample_weight']=weights
    else:kw['sample_weight']=weights
    if early is not None and name=='CatBoost':kw.update(eval_set=early,early_stopping_rounds=50,use_best_model=True)
    if early is not None and name=='LightGBM':
        import lightgbm as lgb
        kw.update(eval_set=[early],eval_metric='l2' if reg else 'auc',callbacks=[lgb.early_stopping(50,first_metric_only=True,verbose=False)])
    model.fit(X,y,**kw)
    chosen=rounds
    if name=='CatBoost':chosen=int(model.tree_count_)
    if name=='LightGBM':chosen=int(model.best_iteration_ or model.n_estimators_)
    return model,chosen

def fit_neural(name,cfg,seed,X,y,weights,rounds=None,early=None,reg=False,device='cuda'):
    import torch
    from networks import SequenceModel
    torch.set_num_threads(2);torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    if device.startswith('cuda'):torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    model=SequenceModel(name,cfg).to(device);opt=torch.optim.AdamW(model.parameters(),lr=cfg.get('lr',.001),weight_decay=cfg.get('weight_decay',.0001))
    tx=torch.as_tensor(X,device=device);ty=torch.as_tensor(y,dtype=torch.float32,device=device);tw=torch.as_tensor(weights,dtype=torch.float32,device=device)
    rng=np.random.default_rng(seed);best=-np.inf;chosen=1;best_state=None;trace=[]
    epochs=rounds or cfg.get('max_iter',100)
    for ep in range(1,epochs+1):
        model.train();losses=[]
        for ids in np.array_split(rng.permutation(len(X)),int(np.ceil(len(X)/128))):
            ids=torch.as_tensor(ids,device=device);out=model(tx[ids])
            loss=((out-ty[ids])**2 if reg else torch.nn.functional.binary_cross_entropy_with_logits(out,ty[ids],reduction='none'))
            loss=(loss*tw[ids]).sum()/tw[ids].sum();opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();losses.append(loss.item())
        if early is not None:
            ex,ey=early;v=score_model(model,ex,reg,device);auc=roc_auc_score((ey<7).astype(int) if reg else ey,v)
            trace.append({'epoch':ep,'loss':float(np.mean(losses)),'early_AUROC':float(auc)})
            if auc>best+1e-5:best=auc;chosen=ep;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            if ep-chosen>=cfg.get('patience',15):break
    if early is not None:model.load_state_dict(best_state)
    else:chosen=epochs
    model.training_trace=trace
    return model,int(chosen)

def fit(name,cfg,seed,X,y,weights,rounds=None,early=None,reg=False,device='cpu'):
    return (fit_neural(name,cfg,seed,X,y,weights,rounds,early,reg,device) if name in NEURAL else fit_classic(name,cfg,seed,X,y,weights,rounds,early,reg))
