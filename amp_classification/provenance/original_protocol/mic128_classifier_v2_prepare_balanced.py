"""User-requested balanced pooled cohort and exactly 200 random test peptides."""
from common import *
import numpy as np,pandas as pd
from sklearn.model_selection import train_test_split,StratifiedKFold

def main():
    DATA.mkdir(parents=True,exist_ok=True)
    if (DATA/'data_acceptance.json').exists():
        assert read(DATA/'data_acceptance.json')['status']=='PASS';return
    old=ROOT/'artifacts/mic128_classifier_v1/data'
    paths=[old/'all_sequence_labels.csv',old/'external.csv',ROOT/'artifacts/repair_v15/data/external_negatives_disjoint.csv']
    pool=pd.read_csv(paths[0]);external=pd.read_csv(paths[1]);negative=pd.read_csv(paths[2])
    assert len(pool)==6847 and len(external)==208
    # Restore all 30 formerly quarantined candidates: the historical external
    # set is explicitly merged in this protocol, not retained as a test set.
    pool['origin']='measured_training';pool['label_evidence']='measured_MIC'
    pool=pool.drop(columns=['group80','matrix_index','quarantined_external_homology'])
    external['origin']='historical_external';external['label_evidence']='measured_MIC'
    external['length']=external.sequence.str.len();external['MIC_ug_mL']=external.MIC_E_coli_ug_mL
    external['threshold_conflict']=False;external['n_observations']=1
    measured=pd.concat([pool,external],ignore_index=True,sort=False)
    assert measured.sequence.is_unique and measured.sample_id.is_unique
    assert np.array_equal(measured.y,(measured.log2_MIC<7).astype(int))
    positive=int(measured.y.sum());existing_negative=len(measured)-positive;needed=positive-existing_negative
    assert needed>0
    negative['source_row']=np.arange(len(negative));negative['valid']=negative.sequence.str.fullmatch('['+AA+']{6,50}')
    negative['duplicate_sequence']=negative.sequence.duplicated()
    negative['measured_overlap']=negative.sequence.isin(measured.sequence)
    valid=negative[negative.valid & ~negative.duplicate_sequence & ~negative.measured_overlap].copy()
    valid=valid.sort_values('sequence').reset_index(drop=True)
    assert len(valid)>=needed
    rng=np.random.default_rng(20261013)
    selected=valid.iloc[rng.choice(len(valid),size=needed,replace=False)].sort_values('sequence').copy()
    selected['sample_id']=['NEG_'+hashlib.sha256(s.encode()).hexdigest()[:12] for s in selected.sequence]
    selected['y']=0;selected['origin']='background_negative';selected['label_evidence']='inherited_negative_pool_label'
    selected['length']=selected.sequence.str.len();selected['log2_MIC']=np.nan;selected['MIC_ug_mL']=np.nan
    selected['threshold_conflict']=False;selected['n_observations']=0
    # Background labels are used at full unit weight under this new protocol.
    # Their unknown MIC is never replaced by a fabricated regression target.
    full=pd.concat([measured,selected],ignore_index=True,sort=False).sort_values('sample_id').reset_index(drop=True)
    assert full.sequence.is_unique and full.sample_id.is_unique
    assert full.y.value_counts().to_dict()=={0:positive,1:positive} or full.y.value_counts().to_dict()=={1:positive,0:positive}
    rng=np.random.default_rng(20261014)
    test=np.sort(rng.choice(len(full),size=200,replace=False));remaining=np.setdiff1d(np.arange(len(full)),test)
    full['split_primary']='unset';full.loc[test,'split_primary']='Test'
    stratum=full.origin+'__'+full.y.astype(str)
    train_n=int(round(.8*len(remaining)))
    train,validation=train_test_split(remaining,train_size=train_n,random_state=20261015,stratify=stratum.iloc[remaining])
    cal,op=train_test_split(validation,test_size=len(validation)//2,random_state=20261016,stratify=stratum.iloc[validation])
    for label,idx in [('T',train),('Vcal',cal),('Vop',op)]:full.loc[idx,'split_primary']=label
    full['fold_primary']=-1;train=np.sort(train);folds=[]
    cv=StratifiedKFold(5,shuffle=True,random_state=20261017)
    for k,(a,b) in enumerate(cv.split(train,full.y.iloc[train])):
        fi=train[a];si=train[b]
        ef,es=train_test_split(fi,test_size=.2,random_state=20261100+k,stratify=stratum.iloc[fi])
        folds.append({'fold':k,'fit_ids':full.sample_id.iloc[fi].tolist(),'score_ids':full.sample_id.iloc[si].tolist(),
                      'early_fit_ids':full.sample_id.iloc[ef].tolist(),'early_stop_ids':full.sample_id.iloc[es].tolist()})
        full.loc[si,'fold_primary']=k
        assert not set(fi)&set(si) and set(ef)|set(es)==set(fi) and not set(ef)&set(es)
    assert full.split_primary.value_counts().Test==200 and 'unset' not in set(full.split_primary)
    full.to_csv(DATA/'samples.csv',index=False);full.loc[test].to_csv(DATA/'test200.csv',index=False)
    selected.to_csv(DATA/'sampled_background_negatives.csv',index=False)
    negative['sampled']=negative.sequence.isin(selected.sequence)&~negative.duplicate_sequence
    negative.to_csv(DATA/'background_pool_audit.csv',index=False)
    dump(DATA/'folds_primary.json',folds)
    summary=full.groupby(['split_primary','origin','y']).size().rename('n').reset_index();summary.to_csv(DATA/'split_source_summary.csv',index=False)
    summary2=full.groupby('split_primary').y.agg(['size','sum']);summary2.columns=['n','positive'];summary2['negative']=summary2.n-summary2.positive;summary2.to_csv(DATA/'split_summary.csv')
    source_hashes={str(p):sha(p) for p in paths}
    dump(DATA/'data_acceptance.json',{'status':'PASS','time':now(),'source_hashes':source_hashes,'measured_total':len(measured),'measured_positive':positive,
        'measured_negative':existing_negative,'background_raw':len(negative),'background_unique_valid':int((negative.valid&~negative.duplicate_sequence).sum()),
        'background_exact_measured_conflicts':int((negative.valid&~negative.duplicate_sequence&negative.measured_overlap).sum()),
        'background_available':len(valid),'background_sampled':needed,'balanced_total':len(full),'class_counts':full.y.value_counts().to_dict(),
        'test_n':200,'split_counts':summary2.reset_index().to_dict('records'),'seeds':SEEDS,
        'historical_external_merged':True,'restored_prior_quarantine':30,
        'test_sampling':'Simple random without replacement, NOT forced to100:100; numpy default_rng seed20261014; fixed before model fits.',
        'random_seeds':{'background_sample':20261013,'test200':20261014,'train_validation':20261015,'calibration_operating':20261016,'CV':20261017},
        'remaining_split':'80% training T,20% validation; validation split equally Vcal/Vop; source-by-label stratification, random sequence allocation.',
        'label_definition':'Measured sequence median log2 MIC<7; historical external measured MIC<128; sampled background inherits pool label0, unknown numeric MIC stays missing.',
        'homology_policy':'Exact sequence duplicates removed. Random partitions as requested; no homology exclusion or cluster-disjoint claim. Test-to-training similarity audited separately.'})
    dump(WORK/'approved_protocol.json',{'status':'APPROVED','time':now(),'user_request':'Pool measured E.coli peptides including former external set with random pure-negative-pool samples to1:1; randomly hold out200; retrain and evaluate.',
        'seeds':SEEDS,'default_seed':123,'models':'11 candidates plus2 diagnostic baselines; compact feature/weight controls, RF EIP and LogisticF1; all fresh training.',
        'search':'5-fold training-only selection CV seed123; same92-trial maximum budget and prior family-selection rule.',
        'calibration':'Vcal sigmoid; Vop frozen balanced/recall90/default0.5 thresholds. Main and controls tested only after all primary fits and deployment choice sealed.',
        'controls_exclusions':'No MIC regression for background labels without measured MIC; no weak0.1/0.3 downweight arms because current user protocol treats sampled negatives as ordinary0 labels; no alternate resplits replacing requested200 test.',
        'metrics':'Class-specific AUROC, AP(AUPRC), precision,recall,F1; MCC,calibration; source-stratified200test and measured-only subset; fixed tests across seeds.',
        'v1_unchanged_manifest_sha256':sha(ROOT/'reports/mic128_classifier_v1/file_manifest.json')})
    update_state('data_frozen',n=len(full),sampled_background=needed,test200=200)
    print(json.dumps(read(DATA/'data_acceptance.json'),ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
