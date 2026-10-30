from common import *
import argparse,time,traceback,gc,resource
import numpy as np,pandas as pd,joblib
from scipy.special import expit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from models import build,score,choose_rounds,audit_params
from metrics import metrics,thresholds,specificity_at_recall

def run(j,threads):
    out=RUNS/j['id'];out.mkdir(parents=True,exist_ok=True);digest=fingerprint(j)
    if (out/'result.json').exists():
        r=read(out/'result.json');assert r['job_hash']==digest and r['status']=='PASS';return r
    dump(out/'job.json',j);start=time.time();name=j['model'];c=j['config'];seed=j['seed']
    t=pd.read_csv(DATA/'T.csv');X=np.load(DATA/f'T_{c["feature"]}.npy',mmap_mode='r');y=t.y.to_numpy(int)
    assert len(t)==8610 and np.isfinite(X).all();rounds=j.get('rounds');trace=[]
    with threadpool_limits(limits=threads if name=='HistGradientBoosting' else 1):
        if j['stage']=='cv':
            fold=read(DATA/'folds_primary.json')[j['fold']];mapping={s:i for i,s in enumerate(t.sample_id)}
            ids={k:np.array([mapping[s] for s in fold[k]]) for k in ['fit_ids','score_ids','early_fit_ids','early_stop_ids']}
            fit,validation=ids['fit_ids'],ids['score_ids'];ef,es=ids['early_fit_ids'],ids['early_stop_ids']
            assert not set(fit)&set(validation) and set(ef)|set(es)==set(fit)
            rounds,trace=choose_rounds(name,c,seed,X[ef],y[ef],X[es],y[es],threads)
            gc.collect();model=build(name,c,seed,threads,rounds);model.fit(X[fit],y[fit]);s=score(model,X[validation],threads)
            p=expit(s);pred=pd.DataFrame({'sample_id':t.sample_id.iloc[validation].to_numpy(),'y':y[validation],'score':s,'probability_uncalibrated':p})
            pred.to_csv(out/'cv_predictions.csv',index=False)
            r=dict(job_hash=digest,id=j['id'],stage='cv',model=name,seed=seed,trial=j.get('trial'),fold=j['fold'],rounds=rounds,
                metrics=metrics(y[validation],p),specificity_at_recall95=specificity_at_recall(y[validation],s),early_trace=trace,
                actual_parameters=audit_params(name,c,model))
        else:
            assert j['stage']=='final'
            model=build(name,c,seed,threads,rounds);model.fit(X,y)
            cal=pd.read_csv(DATA/'Vcal.csv');op=pd.read_csv(DATA/'Vop.csv')
            xc=np.load(DATA/f'Vcal_{c["feature"]}.npy');xo=np.load(DATA/f'Vop_{c["feature"]}.npy')
            cs=score(model,xc,threads);s=score(model,xo,threads)
            calibrator=make_pipeline(StandardScaler(),LogisticRegression(C=1e6,solver='lbfgs',max_iter=2000,random_state=seed))
            calibrator.fit(cs[:,None],cal.y);p=calibrator.predict_proba(s[:,None])[:,1];cut,opm=thresholds(op.y,p)
            for label,tab,sc in [('Vcal',cal,cs),('Vop',op,s)]:
                frame=tab[['sample_id','y']].copy();frame['score']=sc;frame['probability']=calibrator.predict_proba(sc[:,None])[:,1]
                frame.to_csv(out/f'{label}_predictions.csv',index=False)
            joblib.dump(model,out/'model.joblib',compress=3);joblib.dump(calibrator,out/'calibrator.joblib',compress=3)
            reloaded=joblib.load(out/'model.joblib');np.testing.assert_allclose(score(reloaded,xo[:20],threads),s[:20],atol=1e-10,rtol=1e-10)
            r=dict(job_hash=digest,id=j['id'],stage='final',model=name,seed=seed,rounds=rounds,thresholds=cut,Vop=opm,
                training_n=len(t),calibration_n=len(cal),operating_n=len(op),actual_parameters=audit_params(name,c,model),
                model_bytes=sum((out/n).stat().st_size for n in ['model.joblib','calibrator.joblib']),
                fit_sample_hash=hashlib.sha256('\n'.join(t.sample_id).encode()).hexdigest(),reload_score_check='PASS')
    r.update(status='PASS',time=now(),seconds=time.time()-start,threads=threads,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
             implementation_sha256={p.name:sha(p) for p in CODE.glob('*.py')})
    dump(out/'result.json',r)
    if (out/'failure.txt').exists():(out/'failure.txt').rename(out/'resolved_failure.txt')
    print(json.dumps({'id':j['id'],'seconds':r['seconds'],'RSS_MiB':r['peak_RSS_MiB']}),flush=True)
    return r
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('job');a.add_argument('--threads',type=int,default=3);args=a.parse_args()
    j=read(args.job)
    try:run(j,args.threads)
    except Exception:
        out=RUNS/j['id'];out.mkdir(exist_ok=True,parents=True);(out/'failure.txt').write_text(traceback.format_exc());raise
