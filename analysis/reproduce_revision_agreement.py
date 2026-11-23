"""Supplementary Table S6: four fixed model groups, four agreement statistics, paired whole-match 95% CIs."""
import os,sys
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'outputs/revision2026/agreement'
HERE.mkdir(parents=True,exist_ok=True)
REV=ROOT/'data/reference/revision2026'
import numpy as np,pandas as pd,json,hashlib,time
from scipy.stats import rankdata,spearmanr

SELECTION_SEED=20261113
BOOTSTRAP_SEED=20261114
B=5000
NAMES=['LR','Ridge','Lasso','RF','GB','SVR','AdaB','DT','KNN','MLP','GNB','LDA']
protocol=dict(selection_seed=SELECTION_SEED,bootstrap_seed=BOOTSTRAP_SEED,bootstrap_draws=B,
    groups=['12 original single learners','40 uniform subsets from all4095','40 subsets with original top40 size histogram','40 frozen development-AUC-selected subsets'],
    random_selection='One fixed draw per random group, without replacement within group, no exclusion based on AUC and no resampling to obtain a favourable point estimate.',
    size_matching='Only the matched group preserves the exact top40 size histogram. Unrestricted group is uniform over all4095, including singletons.',
    metrics=['top10 Jaccard pairwise median','top10 Jaccard pairwise mean','full34feature Spearman pairwise median','full34feature Spearman pairwise mean'],
    importance='Original linear composition of calibrated-output member SHAP; average absolute ensemble SHAP over original three seeds and sampled sets.',
    confidence='Two-sided95% percentile intervals,5000 whole-match resamples of12 temporal-test matches; same draw for all groups. Groups and fitted learners are fixed.',
    scope='Intervals represent test-match composition uncertainty conditional on the chosen groups and fitted explanations; do not include random-group selection or model retraining uncertainty.')
(HERE/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
start=time.time()
archive=np.load(REV/'all4095_explanation_arrays.npz')
W=archive['weights'];sizes=(W>0).sum(axis=1)
assert W.shape==(4095,12)
single=np.array([np.flatnonzero((sizes==1)&(W[:,j]>0))[0] for j in range(12)])
selected=np.arange(40)
rng=np.random.default_rng(SELECTION_SEED)
uniform=rng.choice(4095,40,replace=False)
hist={int(k):int(np.sum(sizes[:40]==k)) for k in sorted(set(sizes[:40]))}
matched=np.concatenate([rng.choice(np.flatnonzero(sizes==k),n,replace=False) for k,n in hist.items()])
groups={'single12':single,'random40':uniform,'matched40':matched,'topAUC40':selected}
union=np.unique(np.concatenate(list(groups.values())))
local={k:np.searchsorted(union,v) for k,v in groups.items()}
assert all(len(set(v))==len(v) for v in groups.values())
assert {int(k):int(np.sum(sizes[matched]==k)) for k in sorted(set(sizes[matched]))}==hist
np.savez_compressed(HERE/'fixed_group_indices.npz',**groups)
sources={};values=[];features=None
for name in NAMES:
    vv=[]
    for seed in [42,142,242]:
        p=ROOT/f'data/reference/member_shap/shap_{name}_seed{seed}.npz'
        sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        z=np.load(p)
        if features is None:features=z['features'].astype(str)
        assert list(features)==list(z['features'].astype(str))
        vv.append(z['values'])
    values.append(vv)
values=np.asarray(values)
assert values.shape==(12,3,40,34)
phi=np.einsum('um,msnf->usnf',W[union],values)
absolute=np.abs(phi).mean(axis=1)
point=absolute.mean(axis=1)
assert np.allclose(point,archive['importance'][union],atol=1e-12)

METRICS=['Jaccard_median','Jaccard_mean','Spearman_median','Spearman_mean']
tri={k:np.triu_indices(len(v),1) for k,v in groups.items()}
def evaluate(imp):
    # batch x union x feature
    order=np.argsort(-imp,axis=2,kind='stable')
    mask=(np.uint64(1)<<order[:,:,:10].astype(np.uint64)).sum(axis=2,dtype=np.uint64)
    r=rankdata(imp,axis=2,method='average')
    r-=r.mean(axis=2,keepdims=True)
    norms=np.linalg.norm(r,axis=2,keepdims=True)
    assert np.all(norms>0)
    r/=norms
    result={}
    for name,idx in local.items():
        a,b=tri[name]
        m=mask[:,idx]
        inter=np.bitwise_count(m[:,a]&m[:,b]);j=inter/(20.-inter)
        rr=r[:,idx,:]
        corr=rr@rr.transpose(0,2,1)
        sp=corr[:,a,b]
        result[name]=np.column_stack([np.median(j,axis=1),j.mean(axis=1),np.median(sp,axis=1),sp.mean(axis=1)])
    return result

observed=evaluate(point[None])
assert np.allclose(observed['single12'][0],[.38095238095238093,.421824090941738,.5804131599081868,.5932438385384063],atol=1e-12)
assert np.allclose(observed['topAUC40'][0],[1.,.8441037594883749,.920122417750574,.9128566103623487],atol=1e-12)
# Independent, explicit pair checks against set intersection and scipy Spearman.
for name,idx in local.items():
    js=[];ss=[]
    for a,b in zip(*tri[name]):
        ia,ib=idx[a],idx[b]
        sa=set(np.argsort(-point[ia],kind='stable')[:10]);sb=set(np.argsort(-point[ib],kind='stable')[:10])
        js.append(len(sa&sb)/len(sa|sb))
        ss.append(spearmanr(point[ia],point[ib]).statistic)
    assert np.allclose(observed[name][0],[np.median(js),np.mean(js),np.median(ss),np.mean(ss)],atol=1e-12)

test=pd.read_csv(ROOT/'data/processed/test_grouped.csv')
indexes=[np.flatnonzero(test.match_id.to_numpy()==g) for g in sorted(test.match_id.unique())]
assert len(indexes)==12
rng=np.random.default_rng(BOOTSTRAP_SEED)
counts=[]
for rep in range(B):
    idx=np.concatenate([indexes[j] for j in rng.integers(0,12,12)])
    counts.append(np.bincount(idx,minlength=40)/len(idx))
counts=np.array(counts)
np.save(HERE/'paired_bootstrap_set_weights.npy',counts)
boot={k:np.empty((B,4)) for k in groups}
for a in range(0,B,50):
    imp=np.einsum('unf,bn->buf',absolute,counts[a:a+50])
    result=evaluate(imp)
    for k in groups:boot[k][a:a+len(imp)]=result[k]
    if (a+50)%1000==0:print('bootstrap',a+50,'/',B,'seconds',round(time.time()-start,1),flush=True)

np.savez_compressed(HERE/'bootstrap_statistics.npz',**boot)
rows=[];draws=[];members=[]
for k,idx in groups.items():
    for j,metric in enumerate(METRICS):
        lo,hi=np.quantile(boot[k][:,j],[.025,.975])
        rows.append(dict(group=k,n_models=len(idx),n_pairs=len(tri[k][0]),metric=metric,
            estimate=float(observed[k][0,j]),CI95_low=float(lo),CI95_high=float(hi)))
    for b in range(B):draws.append(dict(group=k,replicate=b+1,**dict(zip(METRICS,map(float,boot[k][b])))))
    for order,j in enumerate(idx):members.append(dict(group=k,order=order+1,archive_index=int(j),development_AUC_order=int(j)+1,
        n_members=int(sizes[j]),members='+'.join(NAMES[m] for m in np.flatnonzero(W[j]>0))))
pd.DataFrame(rows).to_csv(HERE/'four_groups_estimates_CI95.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(draws).to_csv(HERE/'four_groups_bootstrap_draws.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(members).to_csv(HERE/'fixed_group_members.csv',index=False,encoding='utf-8-sig')
labels={'single12':'12 single learners','random40':'40 random ensembles',
    'matched40':'40 member-count-matched ensembles','topAUC40':'40 high-AUC ensembles'}
lines=['# Four-group SHAP ranking agreement','',
    '| Group | Jaccard median | Jaccard mean | Spearman median | Spearman mean |',
    '|---|---:|---:|---:|---:|']
for k in groups:
    rr=[r for r in rows if r['group']==k]
    cells=[f"{r['estimate']:.3f} ({r['CI95_low']:.3f} to {r['CI95_high']:.3f})" for r in rr]
    lines.append('| '+labels[k]+' | '+' | '.join(cells)+' |')
lines+=['','Parentheses contain two-sided 95% whole-match bootstrap intervals.',
    'Jaccard compares top-ten sets; Spearman compares all 34 indicator importances.',
    'Each random group is one fixed sample of 40 distinct subsets. The matched group also preserves member-count distribution.',
    'All groups use the same 5,000 resampled match sets. Intervals condition on fitted models, explanations and fixed comparison groups.','']
(HERE/'agreement_summary.md').write_text('\n'.join(lines),encoding='utf-8')

audit=dict(status='PASS',all_four_groups_independently_checked=True,original_single_and_top40_reproduced=True,
    point_importance_matches_archived_arrays=True,random_groups_fixed_before_result_evaluation=True,
    size_histogram=hist,groups={k:dict(n_models=len(v),mean_n_members=float(sizes[v].mean()),indices=v.tolist()) for k,v in groups.items()},
    scope=protocol['scope'],source_hashes=sources,elapsed_seconds=time.time()-start)
(HERE/'verification.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
print('\n'.join(lines),flush=True)

if __name__=='__main__':pass
