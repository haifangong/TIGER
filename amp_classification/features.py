"""Label-independent sequence features; no fitted population statistics."""
from pathlib import Path
AA='ACDEFGHIKLMNPQRSTVWY'
import numpy as np
import pandas as pd
from collections import Counter
HYDRO=dict(zip(AA,[1.8,2.5,-3.5,-3.5,2.8,-.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-.8,-.7,4.2,-.9,-1.3]))
EXTRA=['length','log_length','simple_charge','charge_density','hydropathy_mean','hydropathy_sd','hydrophobic_moment_100deg','aromatic_fraction','polar_fraction','basic_fraction','acidic_fraction','cysteine_fraction','proline_fraction','glycine_fraction','composition_entropy','aliphatic_index','helix_fraction','turn_fraction','sheet_fraction','adjacent_change_fraction']
F1_NAMES=['AAC_'+a for a in AA]+EXTRA
F2_NAMES=F1_NAMES+['DPC_'+a+b for a in AA for b in AA]

def validate(seqs):
    for i,s in enumerate(seqs):
        if not isinstance(s,str) or not 6<=len(s)<=50 or any(a not in AA for a in s):
            raise ValueError(f'Invalid sequence at row {i}: require 6–50 uppercase canonical amino acids; no truncation or silent symbol deletion.')

def compact(seqs,feature='F2'):
    validate(seqs);rows=[]
    for s in seqs:
        n=len(s);c=Counter(s);p=np.array([c[a]/n for a in AA]);h=np.array([HYDRO[a] for a in s])
        frac=lambda chars:sum(c[a] for a in chars)/n
        charge=c['K']+c['R']+.1*c['H']-c['D']-c['E']
        theta=np.arange(n)*np.deg2rad(100)
        e=[n,np.log(n),charge,charge/n,h.mean(),h.std(),abs(np.sum(h*np.exp(1j*theta)))/n,
           frac('FWY'),frac('STNQ'),frac('KRH'),frac('DE'),frac('C'),frac('P'),frac('G'),
           -sum(v*np.log2(v) for v in p if v),100*(frac('A')+2.9*frac('V')+3.9*frac('IL')),
           frac('EMALK'),frac('NPGSD'),frac('VIYFWT'),sum(a!=b for a,b in zip(s,s[1:]))/(n-1)]
        row=list(p)+e
        if feature=='F2':
            dp=Counter(s[i:i+2] for i in range(n-1));row += [dp[a+b]/(n-1) for a in AA for b in AA]
        if feature=='simple':row=[n,charge]
        rows.append(row)
    return np.asarray(rows,dtype=np.float32)

def tokens(seqs):
    validate(seqs);x=np.zeros((len(seqs),50),dtype=np.int64);lookup={a:i+1 for i,a in enumerate(AA)}
    for i,s in enumerate(seqs):x[i,:len(s)]=[lookup[a] for a in s]
    return x
