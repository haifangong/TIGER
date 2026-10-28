"""Rebuild input graphs and reproduce the released seed123/fold1 pair hashes."""
from pathlib import Path
import hashlib
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from code.release import read_config,setup_torch

def main():
    setup_torch('cpu')
    import numpy as np
    from code.dataloader import load_or_build_cache,load_or_build_neg_cache,build_training_pairs,build_neg_pairs
    cfg=read_config('configs/TIGER_pair/seed123.json')
    payload=load_or_build_cache(cfg,ROOT);neg=load_or_build_neg_cache(cfg,ROOT)
    rows=payload['rows'];negative=neg['neg_rows']
    if len(rows)!=7217 or rows.sequence.nunique()!=6847 or len(negative)!=7860:
        raise ValueError('The released graph inputs changed')
    if set(rows.sequence)&set(negative.sequence):
        raise ValueError('Negative/activity overlap changed')
    run=ROOT/'checkpoints/V17/TIGER_pair/seed123/intermediate'
    from sklearn.model_selection import GroupKFold
    for fold,(train_ix,val_ix) in enumerate(GroupKFold(5).split(np.arange(len(rows)),groups=rows.sequence),1):
        archived=json.loads((run/f'fold{fold}_indices.json').read_text())
        if train_ix.tolist()!=archived['train'] or val_ix.tolist()!=archived['validation']:
            raise ValueError(f'Fold {fold} split identity changed')
    split=json.loads((run/'fold1_indices.json').read_text())
    tr,va=np.asarray(split['train']),np.asarray(split['validation'])
    if set(rows.iloc[tr].sequence)&set(rows.iloc[va].sequence):
        raise ValueError('Sequence overlap across training and validation folds')
    selected=rows.iloc[tr].reset_index(drop=True)
    positive_pairs=build_training_pairs(selected,selected.label.to_numpy(float),cfg,124)
    negative_pairs=build_neg_pairs(selected,negative,cfg)
    digest=lambda pairs:hashlib.sha256(np.asarray(pairs,dtype=np.int64).tobytes()).hexdigest()
    ref=json.loads((run/'fold1_pair_audit.json').read_text())
    for key,pairs in [('positive',positive_pairs),('negative',negative_pairs)]:
        if len(pairs)!=ref[key+'_pairs'] or digest(pairs)!=ref[key+'_pair_sha256']:
            raise ValueError(f'{key} training-pair identity changed')
    result=dict(status='PASS',training_rows=len(rows),training_sequences=rows.sequence.nunique(),
                negative_rows=len(negative),fold1_training_rows=len(tr),fold1_validation_rows=len(va),
                positive_pairs=len(positive_pairs),negative_pairs=len(negative_pairs),
                positive_pair_sha256=digest(positive_pairs),negative_pair_sha256=digest(negative_pairs),
                training_filter_stats=payload['train_stats'])
    out=ROOT/'outputs/training_input_validation.json';out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
