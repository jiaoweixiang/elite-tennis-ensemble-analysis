"""Opponent-ranking adjustment for Supplementary Table S8.

log2 opponent ATP rank is added as a retained contextual covariate. Fixed-member
adjustment keeps the original fold-specific and final ensemble members;
reselected adjustment repeats all 4,095 subset comparisons inside the original
5 x 5 match-grouped workflow. Outputs are written to outputs/revision2026/opponent.
"""
import os
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS']:
    os.environ[k] = '1'
import sys
if __name__=='__main__' and any(a in ('-h','--help') for a in sys.argv[1:]):
    print(__doc__); raise SystemExit(0)
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT/'outputs/revision2026/opponent'
HERE.mkdir(parents=True,exist_ok=True)
REFERENCE = ROOT/'data/reference/revision2026/opponent'
import hashlib, itertools, json, time, warnings
import joblib, numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.calibration import _SigmoidCalibration
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, roc_curve, balanced_accuracy_score, matthews_corrcoef
import revision_models as rm
from opponent_components import adjusted_pipeline

START = time.time()
NAMES = rm.NAMES
CONTEXT = 'Opponent ATP rank (log2)'
FOCAL = 'Backhand Returns'
tr = pd.read_csv(ROOT/'data/processed/train_grouped.csv')
te = pd.read_csv(ROOT/'data/processed/test_grouped.csv')
y = tr['Set Result'].to_numpy(int); yt = te['Set Result'].to_numpy(int)
groups = tr.cv_group.to_numpy(str)
metadata = pd.read_csv(REFERENCE/'opponent_metadata_239_sets.csv')
tech_features = list(tr.columns[1:35]); features = tech_features + [CONTEXT]
train_meta = metadata[metadata.split.eq('development')]
test_meta = metadata[metadata.split.eq('temporal_2025')]
assert list(train_meta.match_id) == list(tr.match_id) and list(test_meta.match_id) == list(te.match_id)
X = np.column_stack([tr[tech_features].to_numpy(float), train_meta.log2_opponent_rank.to_numpy(float)])
T = np.column_stack([te[tech_features].to_numpy(float), test_meta.log2_opponent_rank.to_numpy(float)])
assert np.isfinite(X).all() and np.isfinite(T).all()
ALL = np.vstack([X,T]); IDX = np.arange(len(y))
ARCHIVE = json.loads((ROOT/'data/reference/primary_results.json').read_text(encoding='utf-8'))['full']
COMBOS = [c for k in range(1,13) for c in itertools.combinations(range(12), k)]
W = np.zeros((12,len(COMBOS)))
for j,c in enumerate(COMBOS): W[list(c),j] = 1/len(c)
CACHE = HERE/'cache'; CACHE.mkdir(exist_ok=True)

def log(message):
    line = f'[{time.time()-START:.1f}s] {message}'
    print(line, flush=True)
    with (HERE/'run.log').open('a',encoding='utf-8') as f:f.write(line+'\n')

def dump(name, data):
    (HERE/name).write_text(json.dumps(data, ensure_ascii=False, indent=2),encoding='utf-8')

def splits(idx, k):
    idx=np.asarray(idx)
    out=[]
    for a,b in StratifiedGroupKFold(k,shuffle=True,random_state=42).split(X[idx],y[idx],groups[idx]):
        a,b=idx[a],idx[b]
        assert not set(groups[a]) & set(groups[b])
        assert len(np.unique(y[a])) == 2 and len(np.unique(y[b])) == 2
        out.append((a,b))
    return out

def block(idx, ablate=False):
    """Fit all learners on this partition only, including fold-local calibration."""
    idx=np.asarray(idx)
    A=np.delete(ALL, tech_features.index(FOCAL), axis=1) if ablate else ALL
    key=hashlib.sha256(A.tobytes()+idx.tobytes()+y[idx].tobytes()+Path(__file__).read_bytes()+
        (Path(__file__).with_name('opponent_components.py')).read_bytes()).hexdigest()[:24]
    cp=CACHE/(key+'.joblib')
    if cp.exists():return joblib.load(cp)
    local={int(v):i for i,v in enumerate(idx)}
    raw_oof=np.full((len(idx),12),np.nan); predictions=[]; models=[]; cals=[]
    for j,name in enumerate(NAMES):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for a,b in splits(idx,3):
                model=adjusted_pipeline(name).fit(A[a],y[a])
                raw_oof[[local[int(i)] for i in b],j]=model.decision_function(A[b])
                assert model.named_steps['screen'].keep_[-1] == A.shape[1]-1
            cal=_SigmoidCalibration().fit(raw_oof[:,j],y[idx])
            model=adjusted_pipeline(name).fit(A[idx],y[idx])
            assert model.named_steps['screen'].keep_[-1] == A.shape[1]-1
            predictions.append(cal.predict(model.decision_function(A)))
            models.append(model); cals.append(cal)
    result=dict(predictions=np.column_stack(predictions),models=models,calibrators=cals,raw_oof=raw_oof)
    assert np.isfinite(result['predictions']).all()
    joblib.dump(result,cp)
    return result

def oof(idx):
    idx=np.asarray(idx); local={int(v):i for i,v in enumerate(idx)}
    result=np.full((len(idx),12),np.nan)
    for a,b in splits(idx,5):
        result[[local[int(i)] for i in b]]=block(a)['predictions'][b]
    assert np.isfinite(result).all()
    return result

def choose(matrix, labels):
    scores=matrix@W; r=rankdata(scores,axis=0,method='average')
    n1=int(labels.sum());n0=len(labels)-n1
    auc=(r[labels==1].sum(axis=0)-n1*(n1+1)/2)/(n1*n0)
    j=int(np.flatnonzero(np.isclose(auc,auc.max(),atol=1e-12,rtol=0))[0])
    fpr,tpr,thr=roc_curve(labels,scores[:,j],drop_intermediate=False)
    return dict(index=j,columns=list(COMBOS[j]),members=[NAMES[k] for k in COMBOS[j]],
        inner_auc=float(auc[j]),threshold=float(thr[int(np.argmax(tpr-fpr))])),auc

def fixed_choice(matrix, labels, archived):
    columns=[NAMES.index(n) for n in archived['members']]
    score=matrix[:,columns].mean(axis=1)
    fpr,tpr,threshold=roc_curve(labels,score,drop_intermediate=False)
    return dict(members=archived['members'],columns=columns,
        threshold=float(threshold[int(np.argmax(tpr-fpr))]))

def stats(labels,p,pred,split):
    # Table S8: AUC, Brier and accuracy in both cohorts; specificity, balanced accuracy and MCC in 2025.
    out=dict(AUC=float(roc_auc_score(labels,p)),Brier=float(np.mean((labels-p)**2)),accuracy=float(np.mean(labels==pred)))
    if split=='temporal_2025':
        out.update(specificity=float(np.mean(pred[labels==0]==0)),balanced_accuracy=float(balanced_accuracy_score(labels,pred)),
                   MCC=float(matthews_corrcoef(labels,pred)))
    return out

def reference(split):
    p=ROOT/'data/reference'/('outer_predictions.csv' if split=='development' else 'test_predictions.csv')
    data=pd.read_csv(p); study=tr if split=='development' else te
    assert list(data.match_id)==list(study.match_id)
    if split=='development':return data,None
    ab=pd.read_csv(ROOT/'data/reference/backhand_ablation_test_predictions.csv')
    assert list(ab.match_id)==list(study.match_id)
    assert np.max(np.abs(ab['full'].to_numpy()-data.auc_selector_score.to_numpy()))<1e-12
    return data,ab

def main():
    protocol=json.loads((REFERENCE/'analysis_protocol.json').read_text(encoding='utf-8'))
    assert protocol['created_before_any_adjusted_model_fit']
    log('Begin fixed protocol: log2 opponent rank; all71matches; same239sets')
    outer={s:np.full(len(y),np.nan) for s in ['fixed_members','reselected']}
    predicted={s:np.zeros(len(y),int) for s in outer}
    choices=[]
    for fold,(a,b) in enumerate(splits(IDX,5)):
        inner=oof(a)
        selected,_=choose(inner,y[a])
        old=ARCHIVE['outer_folds'][fold]['selection']
        fixed=fixed_choice(inner,y[a],old)
        fitted=block(a)
        for scenario,ch in [('fixed_members',fixed),('reselected',selected)]:
            outer[scenario][b]=fitted['predictions'][b][:,ch['columns']].mean(axis=1)
            predicted[scenario][b]=(outer[scenario][b]>=ch['threshold']).astype(int)
            choices.append(dict(fold=fold+1,scenario=scenario,**ch))
        log(f'Outer fold{fold+1}/5 complete; reselected={selected["members"]}')
    final_oof=oof(IDX);final_choice,allauc=choose(final_oof,y)
    old=ARCHIVE['final_selection']
    fixed=fixed_choice(final_oof,y,old)
    final=block(IDX); final_minus=block(IDX,ablate=True)
    joblib.dump(dict(**final,features=features),HERE/'final_adjusted_models.joblib')
    np.savez_compressed(HERE/'adjusted_development_inner_oof.npz',values=final_oof)
    pd.DataFrame(dict(members=['+'.join(NAMES[i] for i in c) for c in COMBOS],inner_auc=allauc)).to_csv(HERE/'all4095_adjusted_selection_auc.csv',index=False)
    dump('choices.json',dict(outer=choices,final_fixed=fixed,final_reselected=final_choice))
    log(f'Final membership chosen on development data: {final_choice["members"]}')
    predictions={}; rows=[]
    for split,labels,study in [('development',y,tr),('temporal_2025',yt,te)]:
        ref,ab=reference(split)
        original=dict(score=ref.auc_selector_score.to_numpy(),prediction=ref.auc_selector_pred.to_numpy())
        if ab is not None:original['without_focal']=ab['without_backhand'].to_numpy()
        predictions[(split,'original')]=original
        for scenario,ch in [('fixed_members',fixed),('reselected',final_choice)]:
            if split=='development':
                predictions[(split,scenario)]=dict(score=outer[scenario],prediction=predicted[scenario])
            else:
                p=final['predictions'][len(y):][:,ch['columns']].mean(axis=1)
                pminus=final_minus['predictions'][len(y):][:,ch['columns']].mean(axis=1)
                predictions[(split,scenario)]=dict(score=p,prediction=(p>=ch['threshold']).astype(int),without_focal=pminus)
        for scenario in ['original','fixed_members','reselected']:
            pp=predictions[(split,scenario)]
            pd.DataFrame(dict(match_id=study.match_id,label=labels,**pp)).to_csv(HERE/f'{split}_{scenario}_predictions.csv',index=False)
            s=stats(labels,pp['score'],pp['prediction'],split)
            s.update(scenario=scenario,split=split,n_sets=len(labels),n_matches=study.match_id.nunique())
            if 'without_focal' in pp:
                # 2025 Backhand Returns deletion-refit dAUC, members fixed, opponent rank retained.
                s['focal_ablation_delta_AUC']=float(s['AUC']-roc_auc_score(labels,pp['without_focal']))
            rows.append(s)
        log(f'{split} metrics: '+str([r for r in rows if r['split']==split]))
    columns=['scenario','split','n_sets','n_matches','AUC','Brier','accuracy','specificity','balanced_accuracy','MCC','focal_ablation_delta_AUC']
    pd.DataFrame(rows).reindex(columns=columns).to_csv(HERE/'performance_summary.csv',index=False)
    # Paired whole-match uncertainty. Each row uses the same resampled matches.
    boot=[]
    for split,labels,study in [('development',y,tr),('temporal_2025',yt,te)]:
        rng=np.random.default_rng(20261112)
        match_idxs=[np.flatnonzero(study.match_id.to_numpy()==g) for g in sorted(study.match_id.unique())]
        for rep in range(2000):
            idx=np.concatenate([match_idxs[j] for j in rng.integers(0,len(match_idxs),len(match_idxs))])
            if len(np.unique(labels[idx]))<2:continue
            orig_auc=roc_auc_score(labels[idx],predictions[(split,'original')]['score'][idx])
            for scenario in ['fixed_members','reselected']:
                pp=predictions[(split,scenario)]
                auc=roc_auc_score(labels[idx],pp['score'][idx])
                row=dict(split=split,scenario=scenario,replicate=rep+1,paired_AUC_change=auc-orig_auc)
                if 'without_focal' in pp:row['focal_ablation_delta_AUC']=auc-roc_auc_score(labels[idx],pp['without_focal'][idx])
                boot.append(row)
    bdf=pd.DataFrame(boot);bdf.to_csv(HERE/'paired_match_bootstrap.csv',index=False)
    intervals=[]
    for (split,scenario),d in bdf.groupby(['split','scenario']):
        for metric in ['paired_AUC_change','focal_ablation_delta_AUC']:
            if d[metric].isna().all():continue
            lo,hi=np.quantile(d[metric],[.025,.975])
            intervals.append(dict(split=split,scenario=scenario,metric=metric,low=float(lo),high=float(hi),valid_draws=len(d)))
    pd.DataFrame(intervals).to_csv(HERE/'paired_uncertainty_summary.csv',index=False)
    log('Fitting and paired performance bootstrap complete; start matched-budget SHAP')
    explain(final,fixed,final_choice)
    # Check all source files still match the protocol's initial hashes.
    source_ok=all(hashlib.sha256((ROOT/'data/processed'/p.split(':',1)[-1]).read_bytes()).hexdigest()==h for p,h in protocol['study_source_hashes'].items())
    assert source_ok
    dump('verification.json',dict(status='PASS',all71matches=True,all239sets=True,
        original_study_files_unchanged=source_ok,context_never_screened_out=True,
        grouped_nested_selection=True,bootstrap_unit='whole match',elapsed_seconds=time.time()-START))
    log('DONE')

def explain(bundle,fixed,reselected):
    import shap
    selected=sorted(set(fixed['columns'])|set(reselected['columns']))
    models=bundle['models'];cals=bundle['calibrators']
    kept=np.unique(np.concatenate([models[j].named_steps['screen'].keep_ for j in selected]))
    mean=X.mean(axis=0)
    def predict(A):
        restored=np.tile(mean,(len(A),1));restored[:,kept]=A
        allp={j:cals[j].predict(models[j].decision_function(restored)) for j in selected}
        return np.column_stack([np.mean([allp[j] for j in ch['columns']],axis=0) for ch in [fixed,reselected]])
    expected=predict(T[:,kept]); background=shap.sample(X[:,kept],50,random_state=42)
    values=[]
    for seed in [42,142,242]:
        p=HERE/f'adjusted_SHAP_seed{seed}.npz'
        if p.exists():v=np.load(p)['values']
        else:
            np.random.seed(seed)
            exp=shap.KernelExplainer(predict,background)
            v=np.asarray(exp.shap_values(T[:,kept],nsamples=2048,l1_reg=0,silent=True))
            if v.shape==(2,len(T),len(kept)):v=v.transpose(1,2,0)
            assert v.shape==(len(T),len(kept),2),v.shape
            assert np.max(np.abs(v.sum(axis=1)+exp.expected_value-expected))<1e-6
            np.savez_compressed(p,values=v,features=np.array(features)[kept],base=np.asarray(exp.expected_value))
        values.append(v);log(f'KernelSHAP seed{seed} complete')
    feature_names=np.array(features)[kept]
    technical=feature_names!=CONTEXT
    original_values=[]; original_names=None
    for seed in [42,142,242]:
        z=np.load(ROOT/f'data/reference/full_auc_SHAP_seed{seed}.npz')
        original_values.append(z['values']);original_names=z['features'].astype(str)
    phi={'original':(np.stack(original_values),original_names)}
    for j,scenario in enumerate(['fixed_members','reselected']):phi[scenario]=(np.stack(values)[:,:,:,j],feature_names)
    ranking=[];summaries=[];rankings={}
    for scenario,(v,names) in phi.items():
        absolute=np.abs(v).mean(axis=0); imp=absolute.mean(axis=0)
        tech=np.flatnonzero(names!=CONTEXT);order=tech[np.argsort(-imp[tech],kind='stable')]
        order_all=np.argsort(-imp,kind='stable')
        rank={str(names[j]):int(i+1) for i,j in enumerate(order)};rankings[scenario]=rank
        for j in order_all:
            ranking.append(dict(scenario=scenario,feature=names[j],mean_abs_shap=float(imp[j]),
                rank_all=int(np.flatnonzero(order_all==j)[0]+1),rank_technical=rank.get(str(names[j]))))
        bh=int(np.flatnonzero(names==FOCAL)[0]);rng=np.random.default_rng(20261112)
        indexes=[np.flatnonzero(te.match_id.to_numpy()==g) for g in sorted(te.match_id.unique())]
        ranks=[]
        for rep in range(2000):
            idx=np.concatenate([indexes[j] for j in rng.integers(0,len(indexes),len(indexes))])
            orderb=tech[np.argsort(-absolute[idx].mean(axis=0)[tech],kind='stable')]
            ranks.append(int(np.flatnonzero(orderb==bh)[0]+1))
        summaries.append(dict(scenario=scenario,focal_rank_technical=rank[FOCAL],
            focal_rank_all=int(np.flatnonzero(order_all==bh)[0]+1),focal_mean_abs_shap=float(imp[bh]),
            focal_rank_bootstrap_low=float(np.quantile(ranks,.025)),focal_rank_bootstrap_high=float(np.quantile(ranks,.975))))
    pd.DataFrame(ranking).to_csv(HERE/'feature_rankings_all_scenarios.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(summaries).to_csv(HERE/'focal_attribution_summary.csv',index=False)
    compare=[]
    original_imp={r['feature']:r['mean_abs_shap'] for r in ranking if r['scenario']=='original'}
    for scenario in ['fixed_members','reselected']:
        new_imp={r['feature']:r['mean_abs_shap'] for r in ranking if r['scenario']==scenario}
        common=sorted(set(rankings['original'])&set(rankings[scenario]))
        for feature in common:
            compare.append(dict(scenario=scenario,feature=feature,original_rank=rankings['original'][feature],
                adjusted_rank=rankings[scenario][feature],original_mean_abs_shap=original_imp[feature],
                adjusted_mean_abs_shap=new_imp[feature]))
    pd.DataFrame(compare).to_csv(HERE/'paired_technical_rank_comparison.csv',index=False,encoding='utf-8-sig')
    dump('SHAP_protocol.json',dict(seeds=[42,142,242],nsamples=2048,background_n=50,background_seed=42,
        l1_reg=0,technical_rank_excludes_context=True,context_retained_in_model=True,
        note='Model-based probability attributions with marginal background; adjustment is not causal conditional SHAP.'))
    log('Attribution summaries '+str(summaries))

if __name__=='__main__':main()
