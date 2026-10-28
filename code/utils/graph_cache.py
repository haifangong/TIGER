"""Content-identified graph caches with process locks and atomic publication."""
from __future__ import annotations
import fcntl
import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import torch

SCHEMA = 'tiger_graph_V14_input_v15_1'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def cache_identity(cfg,root=None,negative=False):
    root=Path(root or '.')
    def resolve(p):
        p=Path(p);return p if p.is_absolute() else root/p
    csvs=[(cfg.neg_csv,cfg.neg_pdb_dir)] if negative else [(cfg.train_csv,cfg.train_pdb_dir)]
    if not negative and cfg.eval_external: csvs.append((cfg.test_csv,cfg.test_pdb_dir))
    source=[]
    for csv,pdb in csvs:
        csv,pdb=resolve(csv),resolve(pdb)
        seqs=sorted(set(pd.read_csv(csv).sequence.dropna().astype(str).str.strip().str.upper()))
        pdbs={}
        for s in seqs:
            if not s or any(a not in 'ACDEFGHIKLMNPQRSTVWY' for a in s) or not cfg.min_len<=len(s)<=cfg.max_len:
                continue
            path=pdb/f'{s}.pdb'
            pdbs[s]=digest(path) if path.exists() else None
        source.append({'csv':str(csv.resolve()),'csv_sha256':digest(csv),
                       'pdb_directory':str(pdb.resolve()),'pdb_contents':pdbs})
    names=['min_len','max_len','csv_encoding','target_col','use_cfu_protocol','use_cfu_feature',
           'p_feature_scaling','include_node_coords','edge_mode','p_feature_hmom','pair_hmom_delta']
    options={k:getattr(cfg,k,None) for k in names}
    if negative:options['neg_cap_log2']=cfg.neg_cap_log2
    utils=Path(__file__).resolve().parent
    code={p.name:digest(p) for p in [utils/'features.py',utils/'constants.py',utils.parent/'dataloader.py']}
    identity={'schema':SCHEMA,'negative':negative,'source':source,'options':options,
              'aa_features_sha256':digest(resolve(cfg.feature_path)),'code':code}
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    path=resolve(cfg.shared_cache_dir)/f'{"negative" if negative else "training"}_{SCHEMA}_{key}.pt'
    return path,identity


def load_cached_graphs(cfg,root=None,negative=False):
    if negative and not cfg.neg_csv:return None
    from ..dataloader import preprocess_dataset, build_graphs, build_negative_rows
    root=Path(root or '.')
    def resolve(p):
        p=Path(p);return str(p if p.is_absolute() else root/p)
    path,identity=cache_identity(cfg,root,negative)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if path.exists():
            payload=torch.load(path,map_location='cpu',weights_only=False)
            if payload.get('cache_identity')!=identity:raise RuntimeError(f'Cache identity mismatch: {path}')
            return payload
        if negative:
            rows,summary=build_negative_rows(cfg.neg_csv,cfg.neg_pdb_dir,cfg,root)
            if rows.empty:raise ValueError('Empty negative pool')
            graphs,kept,gs=build_graphs(rows,cfg);summary['graph_stats']=gs
            payload={'neg_graphs':graphs,'neg_rows':kept,'neg_stats':summary}
        else:
            rows,summary=preprocess_dataset(resolve(cfg.train_csv),resolve(cfg.train_pdb_dir),cfg,'train')
            graphs,kept,gs=build_graphs(rows,cfg);summary['graph_stats']=gs
            tgraphs,trows,tstats=[],pd.DataFrame(),{}
            if cfg.eval_external:
                tr,tstats=preprocess_dataset(resolve(cfg.test_csv),resolve(cfg.test_pdb_dir),cfg,'test')
                tgraphs,trows,tgs=build_graphs(tr,cfg);tstats['graph_stats']=tgs
            payload={'graphs':graphs,'rows':kept,'train_stats':summary,'test_graphs':tgraphs,
                     'test_rows':trows,'test_stats':tstats,'aa_encoding':'sequential_1_to_20_alphabetical'}
        payload['cache_identity']=identity
        tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
        torch.save(payload,tmp);os.replace(tmp,path)
        side=path.with_suffix('.json');temp=side.with_name(side.name+f'.{os.getpid()}.tmp')
        temp.write_text(json.dumps(identity,ensure_ascii=False,indent=2));os.replace(temp,side)
        return payload
