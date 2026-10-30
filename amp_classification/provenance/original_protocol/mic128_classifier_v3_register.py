from common import *
import numpy as np
def recipes():
    rng=np.random.default_rng(20261024);pick=lambda x:x[int(rng.integers(len(x)))];all_cfg={}
    for model in ['XGBoost','HistGradientBoosting','EBM','RBFSVM','GBDT','AdaBoost']:
        configs=[];seen=set()
        for trial in range(20):
            feature='F1' if model=='EBM' or trial<10 else 'F2'
            while True:
                c={'feature':feature}
                if model=='XGBoost':c.update(depth=pick([2,3,4,6]),learning_rate=pick([.02,.05,.1]),min_child_weight=pick([1,5,10]),subsample=pick([.7,1.]),colsample_bytree=pick([.7,1.]),reg_alpha=pick([0,.1,1.]),reg_lambda=pick([1.,5.,10.]),max_iter=1500)
                if model=='HistGradientBoosting':c.update(max_leaf_nodes=pick([7,15,31]),min_samples_leaf=pick([10,20,50]),l2_regularization=pick([0.,1.,10.]),learning_rate=pick([.03,.05,.1]),max_iter=1000)
                if model=='EBM':c.update(interactions=pick([0,5,10]),max_bins=pick([64,128,256]),learning_rate=pick([.01,.03,.05]),outer_bags=pick([4,8]),max_rounds=5000)
                if model=='RBFSVM':c.update(C=pick([.1,1.,10.,100.]),gamma=pick(['scale',.0001,.001,.01,.1]))
                if model=='GBDT':c.update(depth=pick([1,2,3]),min_samples_leaf=pick([5,10,20]),learning_rate=pick([.03,.05,.1]),subsample=pick([.7,1.]),max_iter=1000)
                if model=='AdaBoost':c.update(depth=pick([1,2,3]),learning_rate=pick([.03,.1,.3,1.]),max_iter=pick([100,300,600]))
                if trial in [0,10] and model!='EBM':
                    defaults={'XGBoost':dict(depth=3,learning_rate=.05,min_child_weight=1,subsample=1.,colsample_bytree=1.,reg_alpha=0.,reg_lambda=1.,max_iter=1500),
                      'HistGradientBoosting':dict(max_leaf_nodes=15,min_samples_leaf=20,l2_regularization=1.,learning_rate=.05,max_iter=1000),
                      'RBFSVM':dict(C=1.,gamma='scale'),'GBDT':dict(depth=2,min_samples_leaf=10,learning_rate=.05,subsample=1.,max_iter=1000),
                      'AdaBoost':dict(depth=1,learning_rate=.1,max_iter=300)}
                    c={'feature':feature,**defaults[model]}
                if model=='EBM' and trial==0:c.update(interactions=5,max_bins=128,learning_rate=.03,outer_bags=4)
                key=fingerprint(c)
                if key not in seen:seen.add(key);break
            configs.append(c)
        all_cfg[model]=configs
    plain=[]
    for i in range(8):
        c=dict(feature='F1' if i<4 else 'F2',boosting_type='Plain',grow_policy='SymmetricTree',depth=pick([4,6,8,10]),learning_rate=pick([.01,.02,.05,.1]),l2_leaf_reg=pick([1.,3.,10.,30.]),random_strength=pick([0.,.5,1.,3.]),rsm=pick([.7,1.]),border_count=pick([64,128,254]),max_iter=3000,patience=100)
        boot=pick(['MVS','Bernoulli','Bayesian']);c['bootstrap_type']=boot
        if boot=='Bayesian':c['bagging_temperature']=pick([0.,1.,3.])
        else:c['subsample']=pick([.6,.8,1.])
        if i in [0,4]:c.update(depth=8,learning_rate=.02,l2_leaf_reg=1.,random_strength=1.,rsm=1.,border_count=254,max_iter=2000,patience=50,bootstrap_type='MVS',subsample=.8);c.pop('bagging_temperature',None)
        plain.append(c)
    ordered=[dict(c,boosting_type='Ordered') for c in plain]
    depthwise=[]
    for i in [1,2,5,6]:
        depthwise.append(dict(plain[i],grow_policy='Depthwise',min_data_in_leaf=pick([5,20,50])))
    all_cfg['CatBoost-Tuned']=plain+ordered+depthwise
    assert all(len(c)==20 for c in all_cfg.values())
    return all_cfg
def main():
    cfg=recipes();jobs=[]
    # Interleave model families so the first running tranche tests every adapter.
    for trial in range(20):
        for fold in range(5):
            for model,configs in cfg.items():
                jobs.append(dict(id=f'cv_{model}_t{trial:02d}_f{fold}',stage='cv',model=model,trial=trial,fold=fold,seed=123,config=configs[trial]))
    assert len(jobs)==700
    dump(WORK/'search_registration.json',{'time':now(),'seed':20261024,'configs':cfg,'test_selection':False,'catboost_pairs':[[i,i+8] for i in range(8)]})
    dump(WORK/'cv_manifest.json',{'stage':'cv','jobs':jobs})
    print('Registered 700 CV jobs; 15 weighted jobs are conditional on Tuned selection.')
if __name__=='__main__':main()
