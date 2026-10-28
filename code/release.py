"""Train and evaluate the released V17 model with fold-specific transforms."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH_FIELDS = ('train_csv','test_csv','train_pdb_dir','test_pdb_dir','neg_csv',
               'neg_pdb_dir','feature_path','shared_cache_dir','out_dir')
KEYS = ['family','anchor_sequence','query_sequence']


def resolve(value):
    p = Path(value)
    return p.resolve() if p.is_absolute() else (ROOT/p).resolve()


def read_config(path):
    from .utils.config import load_config
    cfg = load_config(resolve(path))
    for key in PATH_FIELDS:
        value = getattr(cfg, key, None)
        if value:
            setattr(cfg, key, str(resolve(value)))
    return cfg


def setup_torch(device):
    # Match the original two-thread jobs; do not introduce RNG resets per fold.
    for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
        os.environ[key] = '2'
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Select --device cpu explicitly.')
    return torch


def train(args):
    setup_torch(args.device)
    from .train import train as frozen_train
    cfg = read_config(args.config)
    cfg.device = args.device
    if args.seed is not None:
        cfg.seed = args.seed
    if args.out_dir:
        cfg.out_dir = str(resolve(args.out_dir))
    out = Path(cfg.out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite a nonempty run: {out}')
    for name in ['train_csv','feature_path','train_pdb_dir']:
        if not Path(getattr(cfg,name)).exists():
            raise FileNotFoundError(f'{name}: {getattr(cfg,name)}; run scripts/prepare_structures.py first')
    if cfg.neg_csv and (cfg.cls_pretrain_epochs > 0 or cfg.neg_train_inject):
        for name in ['neg_csv','neg_pdb_dir']:
            if not Path(getattr(cfg,name)).exists():
                raise FileNotFoundError(getattr(cfg,name))
    summary = frozen_train(cfg, root=ROOT)
    print(json.dumps({'status':'PASS','out_dir':str(out),'seed':cfg.seed,
                      'folds':cfg.folds,'summary_written':True}))
    return summary


def predict(args):
    torch = setup_torch(args.device)
    import numpy as np
    import pandas as pd
    from .dataloader import build_graphs
    from .evaluation import eval_explicit_pairs, load_checkpoint_model
    from .utils.constants import TARGET_COL
    from .utils.metrics import apply_calibrator
    from .utils.scaling import apply_p_f_stats, apply_node_x_stats
    from .release_metrics import family_metrics

    run = resolve(args.run or f'checkpoints/V17/TIGER_pair/seed{args.seed}')
    cfg = read_config(run/'config.json')
    out = resolve(args.output)
    # Avoid overwriting immutable released assets even when called with bad paths.
    for protected in ['checkpoints','results/v17','data','configs','provenance']:
        if out.is_relative_to(ROOT/protected):
            raise ValueError('Select a writable output directory outside released assets')
    custom = any([args.peptides,args.pairs,args.pdb_dir])
    if custom and not all([args.peptides,args.pairs,args.pdb_dir]):
        raise ValueError('Custom prediction needs --peptides, --pairs and --pdb-dir together')
    pep = pd.read_csv(resolve(args.peptides or 'data/v17/external_peptides.csv'),float_precision='round_trip')
    pairs = pd.read_csv(resolve(args.pairs or 'data/v17/canonical_pairs.csv'),float_precision='round_trip')
    pdb_dir = resolve(args.pdb_dir or 'data/v17/external_pdb')
    if 'family' not in pairs:
        pairs['family'] = 'user_pairs'
    if not set(KEYS).issubset(pairs) or pairs.duplicated(KEYS).any():
        raise ValueError('Missing or duplicated pair identities')
    if not pep.sequence.is_unique or set(pairs.anchor_sequence)|set(pairs.query_sequence) != set(pep.sequence):
        raise ValueError('Peptide identities must exactly cover the supplied pair endpoint')
    if not custom and (len(pep)!=208 or len(pairs)!=2211 or pairs.family.nunique()!=11):
        raise ValueError('Invalid canonical 208-peptide/2,211-pair endpoint')
    rows = pep[['sequence']].copy()
    for name in ['n_terminus','c_terminus']:
        rows[name] = pep[name] if name in pep else ''
    # Replay the published actual inputs; metadata CFU is not the historical code.
    rows['cfu_group'] = (pep['cfu_group'] if custom and 'cfu_group' in pep else '1E5 - 1E6')
    has_truth = 'MIC_E_coli_ug_mL' in pep
    rows[TARGET_COL] = pep.MIC_E_coli_ug_mL if has_truth else 1.0
    if not np.isfinite(rows[TARGET_COL]).all() or (rows[TARGET_COL] <= 0).any():
        raise ValueError('MIC values must be finite positive mass concentrations')
    rows['pdb_path'] = [str(pdb_dir/f'{seq}.pdb') for seq in rows.sequence]
    if not all(Path(p).is_file() for p in rows.pdb_path):
        raise FileNotFoundError('Each peptide needs a {SEQUENCE}.pdb structure')
    graphs, kept, stats = build_graphs(rows,cfg)
    if len(graphs)!=len(rows) or set(kept.sequence)!=set(rows.sequence):
        raise ValueError(f'Graph construction dropped peptides: {stats}')
    frames=[]
    device=torch.device(args.device)
    for fold in range(1,6):
        read=lambda name: json.loads((run/'intermediate'/name).read_text())
        apply_p_f_stats(graphs,read(f'p_feature_zscore_fold{fold}.json'))
        if cfg.node_feature_scaling=='zscore':
            apply_node_x_stats(graphs,read(f'node_feature_zscore_fold{fold}.json'))
        model,_=load_checkpoint_model(run/f'checkpoints/fold{fold}_best.pt',cfg,device)
        with torch.inference_mode():
            frame,_=eval_explicit_pairs(model,graphs,kept,pairs,cfg,device)
        if frame.duplicated(KEYS).any() or len(frame)!=len(pairs):
            raise ValueError('Prediction endpoint mismatch')
        aligned=pairs[KEYS].merge(frame,on=KEYS,how='left',validate='one_to_one',sort=False)
        values=aligned.y_pred_delta_log2_anchor_minus_query.to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite fold predictions')
        frames.append(values)
        del model
    raw=np.mean(np.stack(frames),axis=0)
    pred=pairs[KEYS].copy()
    pred['y_pred_delta_log2_anchor_minus_query']=raw
    # This function returns a new frame; retaining its return is essential.
    pred=apply_calibrator(pred,json.loads((run/'results/calibrator.json').read_text()))
    pred=pred.rename(columns={'y_pred_delta_log2_anchor_minus_query':'y_pred'})
    pred['raw_fold_mean']=raw
    result={'n_peptides':len(rows),'n_pairs':len(pairs),'n_families':int(pairs.family.nunique()),
            'prediction_unit':'log2(MIC_anchor / MIC_query)','folds':5,'seed':cfg.seed,
            'input_cfu':'provided/default' if custom else 'historical uniform 1E5 - 1E6'}
    if has_truth:
        mic=pep.set_index('sequence').MIC_E_coli_ug_mL
        pred['y_true']=np.log2(pairs.anchor_sequence.map(mic)/pairs.query_sequence.map(mic)).to_numpy()
        if 'y_true' in pairs and not np.allclose(pred.y_true,pairs.y_true,rtol=0,atol=1e-10):
            raise ValueError('Supplied labels have a different pair direction or unit')
        family=pd.DataFrame([dict(family=f,n_pairs=len(g),**family_metrics(g.y_true,g.y_pred))
                             for f,g in pred.groupby('family')])
        result['macro']={m:float(np.mean(family[m].to_numpy())) for m in ['PCC','KCC','log10MAE','RSE']}
    if args.check_reference:
        if custom:
            raise ValueError('--check-reference requires the canonical endpoint')
        ref=pd.read_csv(ROOT/f'results/v17/predictions/factor_P11_seed{cfg.seed}.csv',float_precision='round_trip')
        aligned=pred.merge(ref[KEYS+['y_pred']],on=KEYS,validate='one_to_one',suffixes=('_new','_ref'))
        if len(aligned)!=2211:
            raise ValueError('Reference endpoint mismatch')
        maximum=float(np.max(np.abs(aligned.y_pred_new-aligned.y_pred_ref)))
        if maximum>args.atol:
            raise ValueError(f'Checkpoint replay differs from archive: max_abs={maximum}, atol={args.atol}')
        result['reference_check']={'status':'PASS','max_abs_prediction_difference':maximum,'atol':args.atol}
    out.mkdir(parents=True,exist_ok=True)
    pred.to_csv(out/'pair_predictions.csv',index=False)
    if has_truth:family.to_csv(out/'family_metrics.csv',index=False)
    (out/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('train',help='Train five sequential folds using the unchanged V17 core')
    p.add_argument('--config',default='configs/TIGER_pair/seed123.json')
    p.add_argument('--seed',type=int);p.add_argument('--out-dir')
    p.add_argument('--device',default='cuda:0');p.set_defaults(func=train)
    p=sub.add_parser('predict',help='Average five fold predictions and apply the saved calibrator')
    p.add_argument('--run');p.add_argument('--seed',type=int,default=123,choices=range(120,130))
    p.add_argument('--device',default='cpu');p.add_argument('--output',default='outputs/prediction')
    p.add_argument('--peptides');p.add_argument('--pairs');p.add_argument('--pdb-dir')
    p.add_argument('--check-reference',action='store_true');p.add_argument('--atol',type=float,default=2e-5)
    p.set_defaults(func=predict)
    args=parser.parse_args();args.func(args)

if __name__=='__main__':main()
