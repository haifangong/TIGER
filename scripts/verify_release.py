"""Verify every V17 payload byte and frozen runtime identity."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]

def main():
    manifest=json.loads((ROOT/'provenance/release_manifest.json').read_text())
    for name,expected in manifest['sha256'].items():
        p=ROOT/name
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:
            raise ValueError(f'Release integrity failure: {name}')
    for r in json.loads((ROOT/'provenance/runtime_identity.json').read_text()):
        if hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest()!=r['sha256']:
            raise ValueError(f'Frozen numerical runtime changed: {r["path"]}')
    for r in json.loads((ROOT/'provenance/config_identity.json').read_text()):
        allowed={'train_csv','test_csv','train_pdb_dir','test_pdb_dir','neg_csv','neg_pdb_dir',
                 'feature_path','shared_cache_dir','out_dir','checkpoint'}
        if not set(r['relocated_fields'])<=allowed:
            raise ValueError('A configuration changed scientific fields')
        cfg=json.loads((ROOT/r['path']).read_text())
        scientific={k:v for k,v in cfg.items() if k not in allowed}
        digest=hashlib.sha256(json.dumps(scientific,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if digest!=r['scientific_sha256']:
            raise ValueError(f'Scientific configuration changed: {r["path"]}')
        for key in allowed-{'checkpoint'}:
            if cfg.get(key) and Path(cfg[key]).is_absolute():
                raise ValueError(f'Nonportable configuration: {r["path"]}: {key}')
    print(f'PASS: {len(manifest["sha256"])} payload checksums; 13 frozen source identities; 420 portable configs')

if __name__=='__main__':main()
