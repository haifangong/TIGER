"""One registered CV or final fit. Tests are inaccessible in fit jobs."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='4'
from common import *
import sys,time,argparse,traceback,gc
import numpy as np
import pandas as pd
import joblib
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from models import fit,score_model,sample_weights,NEURAL
from metrics import metrics,thresholds

def load_data(feature):
    table=pd.read_csv(DATA/'samples.csv');index=pd.read_csv(DATA/'feature_index.csv')
    assert table.sample_id.tolist()==index.sample_id.iloc[:len(table)].tolist()
    path=DATA/('tokens.npy' if feature=='tokens' else f'features_{feature}.npy')
    X=np.load(path,mmap_mode='r');return table,X[:len(table)]

def save_model(model,out,name,config):
    if name in NEURAL:
        import torch
        torch.save({'state_dict':{k:v.cpu() for k,v in model.state_dict().items()},'name':name,'config':config},out/'model.pt')
    elif name=='Cascade':model.save(str(out/'cascade'))
    else:joblib.dump(model,out/'model.joblib',compress=3)

def run(job):
    out=RUNS/job['id'];out.mkdir(parents=True,exist_ok=True);fingerprint=hashlib.sha256(json.dumps(job,sort_keys=True).encode()).hexdigest()
    if (out/'result.json').exists():
        result=read(out/'result.json');assert result['job_hash']==fingerprint;return result
    dump(out/'job.json',job);start=time.time()
    table,X=load_data(job['config']['feature']);mapping={s:i for i,s in enumerate(table.sample_id)}
    name=job['model'];cfg=job['config'];seed=job['seed'];reg=job.get('regression',False);device=os.environ.get('TIGER_AMP_DEVICE','cpu') if name in NEURAL else 'cpu'
    y=table.log2_MIC.to_numpy() if reg else table.y.to_numpy();balanced=cfg.get('balanced',False)
    fold=job.get('fold');design=job.get('design','primary');selector=None
    if job['stage']=='cv':
        split=read(DATA/f'folds_{design}.json')[fold]
        fi=np.array([mapping[s] for s in split['fit_ids']]);si=np.array([mapping[s] for s in split['score_ids']])
        ef=np.array([mapping[s] for s in split['early_fit_ids']]);es=np.array([mapping[s] for s in split['early_stop_ids']])
        rounds=job.get('rounds');early_trace=[]
        if name in NEURAL+['CatBoost','LightGBM']:
            m,rounds=fit(name,cfg,seed,np.asarray(X[ef]),y[ef],sample_weights(table.y.iloc[ef].to_numpy(),balanced),early=(np.asarray(X[es]),y[es]),reg=reg,device=device)
            early_trace=getattr(m,'training_trace',[]);del m;gc.collect()
        if cfg['feature']=='EIP':
            from selection import MRMRSpearman
            selector=MRMRSpearman().fit(X[fi],table.y.iloc[fi].to_numpy());xf=selector.transform(X[fi]);xs=selector.transform(X[si])
            joblib.dump(selector,out/'selector.joblib',compress=3)
        else:xf=np.asarray(X[fi]);xs=np.asarray(X[si])
        model,rounds=fit(name,cfg,seed,xf,y[fi],sample_weights(table.y.iloc[fi].to_numpy(),balanced),rounds=rounds,reg=reg,device=device)
        scores=score_model(model,xs,reg,device);prob=expit(scores+7 if reg else scores)
        prediction=pd.DataFrame({'sample_id':table.sample_id.iloc[si].to_numpy(),'y':table.y.iloc[si].to_numpy(),'score':scores,'probability_uncalibrated':prob})
        prediction.to_csv(out/'cv_predictions.csv',index=False)
        result={'job_hash':fingerprint,'id':job['id'],'stage':'cv','model':name,'seed':seed,'fold':fold,'rounds':rounds,'seconds':time.time()-start,'metrics':metrics(prediction.y,prob),'early_trace':early_trace}
    else:
        assert job['stage']=='final'
        fi=np.flatnonzero(table['split_'+design].eq('T'));cal=np.flatnonzero(table['split_'+design].eq('Vcal'));op=np.flatnonzero(table['split_'+design].eq('Vop'))
        if job.get('exclude_conflicts'):fi=fi[~table.threshold_conflict.iloc[fi].to_numpy()]
        xf=np.asarray(X[fi]);yf=y[fi];binary=table.y.iloc[fi].to_numpy();base=np.ones(len(fi))
        if job.get('weak_ratio',0):
            assert not reg
            weak=np.load(DATA/('weak_tokens.npy' if cfg['feature']=='tokens' else f'weak_features_{cfg["feature"]}.npy'))
            per=job['weak_ratio']*(binary==0).sum()/len(weak)
            xf=np.concatenate([xf,weak]);yf=np.r_[yf,np.zeros(len(weak),int)];binary=np.r_[binary,np.zeros(len(weak),int)];base=np.r_[base,np.full(len(weak),per)]
        if cfg['feature']=='EIP':
            from selection import MRMRSpearman
            selector=MRMRSpearman().fit(xf,binary);xf=selector.transform(xf);joblib.dump(selector,out/'selector.joblib',compress=3)
        model,rounds=fit(name,cfg,seed,xf,yf,sample_weights(binary,balanced,base),rounds=job.get('rounds'),reg=reg,device=device)
        if name in NEURAL:
            model=model.cpu();device='cpu'  # Calibrate and choose thresholds in deployment precision/backend.
        def features(ids):return selector.transform(X[ids]) if selector is not None else np.asarray(X[ids])
        cscore=score_model(model,features(cal),reg,device)
        calibrator=make_pipeline(StandardScaler(),LogisticRegression(C=1e6,solver='lbfgs',max_iter=2000,random_state=seed))
        calibrator.fit(cscore[:,None],table.y.iloc[cal]);joblib.dump(calibrator,out/'calibrator.joblib',compress=3)
        oscore=score_model(model,features(op),reg,device);p=calibrator.predict_proba(oscore[:,None])[:,1]
        cut,opmetrics=thresholds(table.y.iloc[op],p)
        for label,ids,scores in [('Vcal',cal,cscore),('Vop',op,oscore)]:
            pd.DataFrame({'sample_id':table.sample_id.iloc[ids].to_numpy(),'y':table.y.iloc[ids].to_numpy(),'score':scores,'probability':calibrator.predict_proba(scores[:,None])[:,1]}).to_csv(out/f'{label}_predictions.csv',index=False)
        save_model(model,out,name,cfg)
        result={'job_hash':fingerprint,'id':job['id'],'stage':'final','model':name,'seed':seed,'rounds':rounds,'seconds':time.time()-start,'thresholds':cut,'Vop':opmetrics,
                'training_n':len(fi),'training_total_with_weak':len(yf),'calibration_n':len(cal),'operating_n':len(op),'fit_sample_hash':hashlib.sha256('\n'.join(table.sample_id.iloc[fi]).encode()).hexdigest(),
                'parameters':sum(p.numel() for p in model.parameters()) if name in NEURAL else None}
        result['model_bytes']=sum(p.stat().st_size for p in out.rglob('*') if p.is_file() and p.suffix not in ['.csv','.json'])
    result['status']='PASS'
    result['implementation_sha256']={p.name:sha(p) for p in Path(__file__).resolve().parent.glob('*.py')}
    if (out/'failure.txt').exists():(out/'failure.txt').rename(out/'resolved_previous_failure.txt')
    dump(out/'result.json',result);print(json.dumps({k:result[k] for k in ['id','model','seconds']}),flush=True);return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('job');args=parser.parse_args();job=read(args.job)
    try:run(job)
    except Exception:
        out=RUNS/job['id'];out.mkdir(parents=True,exist_ok=True);(out/'failure.txt').write_text(traceback.format_exc());raise
if __name__=='__main__':main()
