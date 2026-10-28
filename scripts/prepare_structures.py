"""Extract and SHA256-check the exact V17 training structure archives."""
from pathlib import Path
import csv
import hashlib
import zipfile

ROOT=Path(__file__).resolve().parents[1]

def main():
    with (ROOT/'provenance/structure_archive_members.csv').open() as f:
        rows=list(csv.DictReader(f))
    groups={}
    for r in rows:groups.setdefault(r['archive'],[]).append(r)
    for archive,members in groups.items():
        with zipfile.ZipFile(ROOT/archive) as z:
            if set(z.namelist())!={r['member'] for r in members}:
                raise ValueError(f'Unexpected archive members: {archive}')
            for r in members:
                p=(ROOT/'data/v17/pdb'/r['member']).resolve()
                if not p.is_relative_to(ROOT/'data/v17/pdb'):
                    raise ValueError('Archive member escapes destination')
                data=z.read(r['member'])
                if hashlib.sha256(data).hexdigest()!=r['sha256']:
                    raise ValueError(f'Corrupt structure: {r["member"]}')
                p.parent.mkdir(parents=True,exist_ok=True)
                if p.exists() and p.read_bytes()!=data:
                    raise FileExistsError(f'Refusing to overwrite changed structure: {p}')
                if not p.exists():p.write_bytes(data)
        print(f'{archive}: {len(members)} structures verified')
    print(f'PASS: {len(rows)} exact source structures')

if __name__=='__main__':main()
