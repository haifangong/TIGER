from common import *
import numpy as np,pandas as pd
def choose(jobs,weighted=False):
    rows=[]
    for j in jobs:
        r=read(RUNS/j['id']/'result.json');assert r['status']=='PASS' and r['job_hash']==fingerprint(j)
        rows.append(dict(model=j['model'],trial=j['trial'],fold=j['fold'],AUROC=r['metrics']['AUROC'],AUPRC=r['metrics']['AP_active'],
                spec95=r['specificity_at_recall95'],rounds=r['rounds'],seconds=r['seconds']))
    frame=pd.DataFrame(rows);frame.to_csv(WORK/('weighted_cv_folds.csv' if weighted else 'cv_folds.csv'),index=False)
    selected={}
    for name,part in frame.groupby('model'):
        agg=part.groupby('trial').agg(AUROC=('AUROC','mean'),AUPRC=('AUPRC','mean'),spec95=('spec95','mean'),count=('fold','count'),AUROC_std=('AUROC','std')).reset_index()
        assert agg['count'].eq(5).all()
        # Lower configured structural complexity and trial index resolve ties.
        complexity={j['trial']:j['config'].get('depth',j['config'].get('max_leaf_nodes',0)) for j in jobs if j['model']==name}
        agg['complexity']=agg.trial.map(complexity)
        if weighted:order=['spec95','AUPRC','trial'];ascending=[False,False,True]
        else:order=['AUROC','AUPRC','complexity','trial'];ascending=[False,False,True,True]
        best=agg.sort_values(order,ascending=ascending).iloc[0];trial=int(best.trial)
        cohort=[j for j in jobs if j['model']==name and j['trial']==trial]
        rounds=part[part.trial.eq(trial)].rounds.dropna()
        selected[name]=dict(config=cohort[0]['config'],rounds=int(np.median(rounds)) if len(rounds) else None,
                CV_AUROC=float(best.AUROC),CV_AUPRC=float(best.AUPRC),CV_spec95=float(best.spec95),source_jobs=[j['id'] for j in cohort],trial=trial)
        agg.to_csv(WORK/f'search_{name}_summary.csv',index=False)
    return selected
def register_weights(selected):
    base=selected['CatBoost-Tuned'];jobs=[]
    for trial,w in enumerate([1.5,2.,3.],1):
        for f in range(5):jobs.append(dict(id=f'cv_CatBoost-Recall_w{w:g}_f{f}',stage='cv',model='CatBoost-Recall',trial=trial,seed=123,fold=f,config={**base['config'],'positive_weight':w}))
    dump(WORK/'weighted_manifest.json',{'stage':'weighted_cv','jobs':jobs});return jobs
def choose_weight(selected,jobs):
    base=selected['CatBoost-Tuned'];rows=[]
    for id in base['source_jobs']:
        r=read(RUNS/id/'result.json');rows.append(dict(weight=1.,fold=r['fold'],spec95=r['specificity_at_recall95'],AUPRC=r['metrics']['AP_active'],rounds=r['rounds'],job_id=id))
    for j in jobs:
        r=read(RUNS/j['id']/'result.json');assert r['status']=='PASS'
        rows.append(dict(weight=j['config']['positive_weight'],fold=j['fold'],spec95=r['specificity_at_recall95'],AUPRC=r['metrics']['AP_active'],rounds=r['rounds'],job_id=j['id']))
    d=pd.DataFrame(rows);d.to_csv(WORK/'catboost_weight_selection.csv',index=False)
    a=d.groupby('weight').agg(spec95=('spec95','mean'),AUPRC=('AUPRC','mean'),n=('fold','count')).reset_index();assert a.n.eq(5).all()
    w=float(a.sort_values(['spec95','AUPRC','weight'],ascending=[False,False,True]).iloc[0].weight)
    if w==1.:dump(WORK/'catboost_recall_alias.json',{'alias':'CatBoost-Recall','target':'CatBoost-Tuned','reason':'w=1 won preregistered CV comparison'})
    else:
        part=d[d.weight.eq(w)]
        selected['CatBoost-Recall']=dict(config={**base['config'],'positive_weight':w},rounds=int(part['rounds'].median()),
                CV_spec95=float(part.spec95.mean()),CV_AUPRC=float(part.AUPRC.mean()),source_jobs=part.job_id.tolist())
    dump(WORK/'selected_configs.json',selected);return selected
def final_jobs(selected):
    jobs=[dict(id=f'main_{name}_s{s}',stage='final',model=name,seed=s,config=choice['config'],rounds=choice['rounds']) for name,choice in selected.items() for s in SEEDS]
    dump(WORK/'final_manifest.json',{'stage':'final','jobs':jobs});return jobs

