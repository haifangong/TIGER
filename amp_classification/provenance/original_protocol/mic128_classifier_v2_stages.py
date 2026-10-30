from common import *
from models import DEFAULT,NEURAL
import copy,argparse
import numpy as np
import pandas as pd

def summarize(prefix):
    rows=[]
    for p in sorted(RUNS.glob(prefix+'*/result.json')):
        r=read(p);j=read(p.parent/'job.json')
        if r['stage']!='cv':continue
        rows.append({'id':j['id'],'model':j['model'],'trial':j.get('trial',0),'fold':j['fold'],'AUROC':r['metrics']['AUROC'],
                     'AP_inactive':r['metrics']['AP_inactive'],'AP_active':r['metrics']['AP_active'],'MCC':r['metrics']['MCC'],'log_loss':r['metrics']['log_loss'],
                     'rounds':r['rounds'],'seconds':r['seconds'],'config':json.dumps(j['config'],sort_keys=True)})
    return pd.DataFrame(rows)

def screening():
    jobs=[]
    # All candidates share the same five outer folds and seed123 screening.
    for name,cfg in DEFAULT.items():
        for fold in range(5):jobs.append({'id':f'screen_{name}_f{fold}','stage':'cv','model':name,'seed':123,'fold':fold,'config':cfg.copy()})
    return {'stage':'screening','jobs':jobs}

def tuning():
    screen=summarize('screen_');screen.to_csv(WORK/'screening_folds.csv',index=False)
    scores=screen.groupby('model').AUROC.agg(['mean','std','count']).sort_values('mean',ascending=False);assert scores['count'].eq(5).all()
    scores.to_csv(WORK/'screening_summary.csv')
    neural=scores.loc[[n for n in scores.index if n in NEURAL]].index[0]
    chosen=['CatBoost','LightGBM',neural]
    chosen += [n for n in scores.index if n not in chosen and n not in NEURAL+['Dummy','SimpleLogistic']][:2]
    rng=np.random.default_rng(20261013);jobs=[];registry=[]
    def pick(values):return values[int(rng.integers(len(values)))]
    for name in chosen:
        n=12 if name in NEURAL else 20;seen=set()
        for trial in range(n):
            cfg=DEFAULT[name].copy()
            if trial:
                while True:
                    cfg=DEFAULT[name].copy();cfg['balanced']=pick([False,True])
                    if name not in NEURAL+['Cascade']:cfg['feature']=pick(['F1','F2'])
                    if name=='CatBoost':cfg.update(depth=pick([4,6,8]),learning_rate=pick([.02,.05,.1]),l2_leaf_reg=pick([1,5,10,20]),max_iter=2000)
                    elif name=='LightGBM':cfg.update(num_leaves=pick([7,15,31]),min_child_samples=pick([20,50,100]),learning_rate=pick([.02,.05,.1]),reg_lambda=pick([0,1,5,10]),max_iter=2000)
                    elif name in ['RF','ExtraTrees']:cfg.update(trees=pick([200,350,500]),depth=pick([8,12,16]),min_leaf=pick([2,5,10]),max_features=pick(['sqrt',.5,.8]))
                    elif name=='Cascade':cfg.update(trees=pick([30,50,100]),max_layers=pick([2,3,4]),min_leaf=pick([3,5,10]))
                    elif name in NEURAL:
                        cfg.update(lr=pick([.0003,.001,.003]),dropout=pick([.1,.3,.5]),weight_decay=pick([.0001,.001,.01]),max_iter=150,patience=15)
                        if name=='TinyCNN':cfg['channels']=pick([32,48,64])
                    else:cfg.update(C=pick([.01,.1,1.,10.]))
                    key=json.dumps(cfg,sort_keys=True)
                    if key not in seen:break
            seen.add(json.dumps(cfg,sort_keys=True));registry.append({'model':name,'trial':trial,'config':cfg})
            for fold in range(5):jobs.append({'id':f'tune_{name}_t{trial:02d}_f{fold}','stage':'cv','model':name,'seed':123,'fold':fold,'trial':trial,'config':cfg})
    dump(WORK/'search_registration.json',{'time':now(),'chosen_families':chosen,'screening_only':True,'trials':registry,'test_access':False,
         'final_three_policy':'Highest training-CV boosting model, highest training-CV forest model, highest training-CV sequence model; all other candidates retained in main benchmark.'})
    return {'stage':'tuning','jobs':jobs}

def locked_candidates():
    tune=summarize('tune_');tune.to_csv(WORK/'tuning_folds.csv',index=False)
    aggregate=tune.groupby(['model','trial']).AUROC.agg(['mean','std','count']).reset_index()
    assert aggregate['count'].eq(5).all()
    assert len(tune)==len(read(WORK/'tuning_manifest.json')['jobs'])
    aggregate.to_csv(WORK/'tuning_summary.csv',index=False)
    screen=summarize('screen_');selected={}
    for name,cfg in DEFAULT.items():
        part=aggregate[aggregate.model==name]
        if len(part):
            # Stable tie break: first registered trial.
            winner=part.sort_values(['mean','trial'],ascending=[False,True]).iloc[0]
            rows=tune[(tune.model==name)&(tune.trial==winner.trial)]
        else:rows=screen[screen.model==name]
        rounds=rows['rounds'].dropna()
        selected[name]={'config':json.loads(rows.config.iloc[0]),'rounds':max(1,int(np.median(rounds))) if len(rounds) else None,
                        'CV_AUROC':float(rows.AUROC.mean()),'CV_SE':float(rows.AUROC.std(ddof=1)/np.sqrt(5)),
                        'source_jobs':rows.id.tolist()}
    boost=max(['CatBoost','LightGBM'],key=lambda n:selected[n]['CV_AUROC'])
    forest=max(['Cascade','ExtraTrees','RF'],key=lambda n:selected[n]['CV_AUROC'])
    neural=max(NEURAL,key=lambda n:selected[n]['CV_AUROC'])
    return selected,[boost,forest,neural]

def controls_cv():
    selected,cores=locked_candidates();dump(WORK/'selected_configs.json',{'time':now(),'models':selected,'controls_core':cores,'test_access':False})
    jobs=[];variants=[]
    for name in cores:
        cfg=selected[name]['config']
        features=[cfg['feature']] if name in NEURAL else ['F1','F2']
        # Extra weight toggle for the native AMP553 cascade representation.
        if cfg['feature']=='AMP553':features.append('AMP553')
        for feature in features:
            for balanced in [False,True]:
                c={**cfg,'feature':feature,'balanced':balanced};tag=f'{feature}_{"balanced" if balanced else "unweighted"}'
                variants.append({'model':name,'variant':tag,'config':c,'regression':False})
    variants.append({'model':'RF_EIP','variant':'EIP_mRMR','config':{**DEFAULT['RF'],'feature':'EIP'},'regression':False})
    variants.append({'model':'Logistic','variant':'F1','config':{**DEFAULT['Logistic'],'feature':'F1'},'regression':False})
    for v in variants:
        for fold in range(5):jobs.append({'id':f'controlcv_{v["model"]}_{v["variant"]}_f{fold}','stage':'cv','seed':123,'fold':fold,**v})
    dump(WORK/'control_registration.json',{'time':now(),'variants':variants,'weak_ratios':[],'exclude_conflicts':True,
         'stress_models':[],'stress_configs':'Requested single random200test design; no alternate split experiments.',
         'test_access':False})
    return {'stage':'controls_cv','jobs':jobs}

def final_fits():
    selected=read(WORK/'selected_configs.json');jobs=[]
    for name,choice in selected['models'].items():
        for seed in SEEDS:jobs.append({'id':f'main_{name}_s{seed}','stage':'final','model':name,'seed':seed,'config':choice['config'],'rounds':choice['rounds'],'analysis':'main'})
    for v in read(WORK/'control_registration.json')['variants']:
        prefix=f'controlcv_{v["model"]}_{v["variant"]}_f'
        rows=[read(RUNS/f'{prefix}{k}'/'result.json') for k in range(5)]
        rounds=[r['rounds'] for r in rows if r['rounds'] is not None];rounds=int(np.median(rounds)) if rounds else None
        for seed in SEEDS:jobs.append({'id':f'control_{v["model"]}_{v["variant"]}_s{seed}','stage':'final','seed':seed,'rounds':rounds,'analysis':'control',**v})
    for name in selected['controls_core']:
        choice=selected['models'][name]
        for variant,extra in [('noConflict',{'exclude_conflicts':True})]:
            for seed in SEEDS:jobs.append({'id':f'control_{name}_{variant}_s{seed}','stage':'final','model':name,'seed':seed,'config':choice['config'],'rounds':choice['rounds'],'analysis':'control','variant':variant,**extra})
    return {'stage':'final_fits','jobs':jobs}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['screening','tuning','controls_cv','final_fits']);args=parser.parse_args()
    dump(WORK/(args.stage+'_manifest.json'),globals()[args.stage]())
