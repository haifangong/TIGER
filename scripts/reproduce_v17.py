"""Replay all 420 frozen predictors and all 164 paired ablation tests on CPU."""
from pathlib import Path
import argparse
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'code'))
import numpy as np
import pandas as pd
from scipy import stats
from release_metrics import (METRICS,KEYS,require,read,sha256,aligned_prediction,
                                  family_metrics,holm,compare)


def run(output):
    output=Path(output).resolve()
    require(not output.is_relative_to(ROOT/'results/v17'), 'Frozen results cannot be overwritten')
    groups=json.loads((ROOT/'configs/ablation_groups.json').read_text())
    arms=[a for _,aa in groups for a in aa]
    require(len(arms)==len(set(arms))==42,'Expected 42 configurations')
    index=read(ROOT/'provenance/prediction_index.csv')
    expected={(a,s) for a in arms for s in range(120,130)}
    require(len(index)==420 and set(map(tuple,index[['arm','seed']].to_numpy()))==expected,
            'Missing, duplicated or unexpected arm/seed')
    pairs=read(ROOT/'data/v17/canonical_pairs.csv');peptides=read(ROOT/'data/v17/external_peptides.csv')
    require(len(pairs)==2211 and pairs.family.nunique()==11 and not pairs.duplicated(KEYS).any(),'Invalid endpoint')
    require(len(peptides)==208 and peptides.sequence.is_unique,'Invalid peptide dictionary')
    require(set(pairs.anchor_sequence)|set(pairs.query_sequence)==set(peptides.sequence),'Endpoint coverage mismatch')
    mic=peptides.set_index('sequence').MIC_E_coli_ug_mL
    require(np.allclose(pairs.y_true,np.log2(pairs.anchor_sequence.map(mic)/pairs.query_sequence.map(mic)),
                        atol=1e-12,rtol=0),'Wrong target units or direction')
    family_groups=[(f,g.index.to_numpy()) for f,g in pairs.groupby('family')]
    truth=pairs.y_true.to_numpy();family_rows=[];seed_rows=[]
    for row in index.itertuples():
        path=ROOT/row.path
        require(sha256(path)==row.sha256,f'Changed frozen prediction: {row.path}')
        pred=aligned_prediction(read(path),pairs);values=[]
        for f,ix in family_groups:
            met=family_metrics(truth[ix],pred[ix]);values.append(met)
            family_rows.append(dict(arm=row.arm,seed=row.seed,family=f,n_pairs=len(ix),**met))
        seed_rows.append(dict(arm=row.arm,seed=row.seed,**{m:np.mean([v[m] for v in values]) for m in METRICS}))
    family=pd.DataFrame(family_rows);seed=pd.DataFrame(seed_rows);summaries=[];tests=[]
    for arm,g in seed.groupby('arm'):
        for m in METRICS:
            v=g[m].to_numpy();mean=v.mean();sd=v.std(ddof=1);half=stats.t.ppf(.975,9)*sd/np.sqrt(10)
            summaries.append(dict(arm=arm,metric=m,n=10,n_defined=int(np.isfinite(v).sum()),
                                  mean=mean,sd=sd,ci_low=mean-half,ci_high=mean+half))
            if arm!='factor_P11':
                a=g.set_index('seed').sort_index()[m]
                b=seed[seed.arm=='factor_P11'].set_index('seed').sort_index()[m]
                delta=(a-b).to_numpy();half=stats.t.ppf(.975,9)*stats.sem(delta)
                p=1.0 if np.all(delta==0) else stats.wilcoxon(delta,method='auto').pvalue
                tests.append(dict(arm=arm,metric=m,delta_a_minus_b=delta.mean(),
                                  ci_low=delta.mean()-half,ci_high=delta.mean()+half,p_raw=p))
    summary=pd.DataFrame(summaries);tests=pd.DataFrame(tests);tests['p_holm']=holm(tests.p_raw)
    checks=0
    for frame,name,keys,cols in [
        (family,'tiger_family_per_seed',['arm','seed','family'],METRICS),
        (seed,'tiger_per_seed',['arm','seed'],METRICS),
        (summary,'tiger_summary',['arm','metric'],['mean','sd','ci_low','ci_high']),
        (tests,'ablation_contrasts',['arm','metric'],['delta_a_minus_b','ci_low','ci_high','p_raw','p_holm'])]:
        checks+=compare(frame,read(ROOT/f'results/v17/reference/{name}.csv'),keys,cols,atol=1e-10)
    output.mkdir(parents=True,exist_ok=True)
    for name,frame in [('tiger_family_per_seed',family),('tiger_per_seed',seed),('tiger_summary',summary),('ablation_contrasts',tests)]:
        frame.to_csv(output/f'{name}.csv',index=False)
    result=dict(status='PASS',configurations=42,seed_predictions=420,family_scores=4620,
                paired_tests=164,numeric_checks=checks,holm_significant_tests=int((tests.p_holm<.05).sum()))
    (output/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'outputs/replay')
    run(p.parse_args().output)
