from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib,json,os
ROOT=Path(__file__).resolve().parents[2]
CODE=Path(__file__).resolve().parent
DATA=ROOT/'data'
WORK=Path(os.environ.get('TIGER_AMP_RUNS',str(ROOT.parent/'outputs/amp_training')))
RUNS=WORK
REPORT=WORK/'reports'
SEEDS=list(range(120,125))
AA='ACDEFGHIKLMNPQRSTVWY'
def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+f'.{os.getpid()}.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)
def fingerprint(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()

