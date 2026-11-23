"""Recalculate supplementary results from deposited observations and arrays.

Numerical reproduction only (no model refitting). Outputs in outputs/revision2026:
  additional_classification_metrics.csv  Table S3 note (2025 specificity, balanced accuracy, MCC)
  partial_spearman.csv                   Table S1 note
  figure4b_ablation.csv                  Figure 4b; Tables S5, S8, S9 (original-model dAUC)
  four_group_agreement.csv, paired_agreement_differences.csv   Table S6
  tuned_rank_agreement.csv               Table S7 hyperparameter-tuning agreement rows
  opponent_adjustment.csv                Table S8
  court_depth_omission.csv               Table S9
  return_representations_mean_mask.csv   Table S10
  external_recording_agreement.csv, observer_agreement.csv   recording checks; Table S11 note
"""
from pathlib import Path
import csv,json,itertools
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score,accuracy_score,f1_score,balanced_accuracy_score,matthews_corrcoef

ROOT=Path(__file__).resolve().parents[1]
REF=ROOT/'data/reference'
REV=REF/'revision2026'
OUT=ROOT/'outputs/revision2026'
NAMES=['LR','Ridge','Lasso','RF','GB','SVR','AdaB','DT','KNN','MLP','GNB','LDA']
SEEDS=[42,142,242]

def write(name,rows):
    pd.DataFrame(rows).to_csv(OUT/name,index=False)

def same(actual,expected,tol=1e-10):
    assert np.allclose(actual,expected,atol=tol,rtol=0),(actual,expected)

def kappa(a,b):
    a=np.asarray(a);b=np.asarray(b);n=len(a)
    categories=np.union1d(a,b)
    observed=np.mean(a==b)
    expected=sum(np.mean(a==c)*np.mean(b==c) for c in categories)
    return float((observed-expected)/(1-expected)) if expected<1 else None

def icc(a,b):
    x=np.column_stack([a,b]).astype(float);n,k=x.shape
    grand=x.mean();rows=x.mean(axis=1);cols=x.mean(axis=0)
    msr=k*np.sum((rows-grand)**2)/(n-1)
    msc=n*np.sum((cols-grand)**2)/(k-1)
    mse=np.sum((x-rows[:,None]-cols[None,:]+grand)**2)/((n-1)*(k-1))
    den=msr+(k-1)*mse+k*(msc-mse)/n
    return float((msr-mse)/den) if abs(den)>1e-15 else None

def agreement(importance):
    order=np.argsort(-importance,axis=1,kind='stable')
    sets=[set(r[:10]) for r in order]
    pairs=list(itertools.combinations(range(len(sets)),2))
    j=np.array([len(sets[a]&sets[b])/len(sets[a]|sets[b]) for a,b in pairs])
    ranks=rankdata(importance,axis=1,method='average')
    corr=np.corrcoef(ranks)
    s=np.array([corr[a,b] for a,b in pairs])
    return np.array([np.median(j),np.mean(j),np.median(s),np.mean(s)])

def classification():
    p=pd.read_csv(REF/'test_predictions.csv');y=p.label.to_numpy(int)
    pred=p.auc_selector_pred.to_numpy(int)
    tn=np.sum((y==0)&(pred==0));fp=np.sum((y==0)&(pred==1))
    stats={'auc':roc_auc_score(y,p.auc_selector_score),'accuracy':accuracy_score(y,pred),
           'specificity':tn/(tn+fp),'balanced_accuracy':balanced_accuracy_score(y,pred),
           'MCC':matthews_corrcoef(y,pred),'F1':f1_score(y,pred)}
    rows=[];expected=pd.read_csv(REV/'classification_metrics_with_intervals.csv')
    for metric in ['auc','accuracy','specificity','balanced_accuracy','MCC']:
        saved=expected[(expected['sample']=='temporal_2025')&(expected.metric==metric)].iloc[0]
        same(stats[metric],saved.estimate)
        rows.append(dict(metric=metric,estimate=stats[metric],CI95_low=saved.CI2_5,CI95_high=saved.CI97_5))
    write('additional_classification_metrics.csv',rows)
    return stats

def partial_correlation():
    t=pd.read_csv(ROOT/'data/processed/train_grouped.csv')
    b=rankdata(t['Backhand Returns']);total=rankdata(t['Total Returns']);y=rankdata(t['Set Result'])
    design=np.column_stack([np.ones(len(t)),total])
    rb=b-design@np.linalg.lstsq(design,b,rcond=None)[0]
    ry=y-design@np.linalg.lstsq(design,y,rcond=None)[0]
    rho=float(np.corrcoef(rb,ry)[0,1]);assert round(rho,3)==.003
    write('partial_spearman.csv',[dict(indicator='Backhand Returns',adjusted_for='Total Returns',rho=rho)])
    return rho

def primary_ablation():
    p=pd.read_csv(REV/'primary_ablation/test_ablation_predictions.csv')
    target=pd.read_csv(REV/'primary_ablation/fig4b_aligned_95CI_source.csv')
    y=p.label.to_numpy(int);full=p.full.to_numpy(float)
    columns={c.strip():c for c in p.columns}
    names=target.feature.tolist();q=p[[columns[n] for n in names]].to_numpy(float)
    match=p.match_id.to_numpy(str);groups=[np.flatnonzero(match==g) for g in sorted(set(match))]
    rng=np.random.default_rng(20261112)
    for _ in range(2000):rng.integers(0,59,59)  # advance the generator to the state used for the reported intervals
    draws=[]
    for _ in range(2000):
        ix=np.concatenate([groups[j] for j in rng.integers(0,len(groups),len(groups))])
        if len(np.unique(y[ix]))<2:continue
        auc=roc_auc_score(y[ix],full[ix])
        draws.append([auc-roc_auc_score(y[ix],q[ix,j]) for j in range(len(names))])
    draws=np.array(draws);rows=[]
    for j,name in enumerate(names):
        delta=roc_auc_score(y,full)-roc_auc_score(y,q[:,j]);lo,hi=np.quantile(draws[:,j],[.025,.975])
        saved=target.iloc[j];same([delta,lo,hi],[saved.delta_auc,saved.CI2_5,saved.CI97_5])
        rows.append(dict(indicator=name,delta_AUC=delta,CI95_low=lo,CI95_high=hi,method='deletion_refit'))
    write('figure4b_ablation.csv',rows)
    return len(draws)

def primary_agreement():
    archive=np.load(REV/'all4095_explanation_arrays.npz')
    groups=np.load(REV/'agreement/fixed_group_indices.npz')
    boot=np.load(REV/'agreement/bootstrap_statistics.npz')
    values=np.array([[np.load(REF/'member_shap'/f'shap_{m}_seed{s}.npz')['values'] for s in SEEDS] for m in NAMES])
    observed={};rows=[]
    metrics=['Jaccard_median','Jaccard_mean','Spearman_median','Spearman_mean']
    reference=pd.read_csv(REV/'agreement/four_groups_estimates_CI95.csv')
    for group in groups.files:
        idx=groups[group]
        importance=np.abs(np.einsum('km,msnf->ksnf',archive['weights'][idx],values)).mean(axis=(1,2))
        same(importance,archive['importance'][idx])
        point=agreement(importance);observed[group]=point
        for j,metric in enumerate(metrics):
            lo,hi=np.quantile(boot[group][:,j],[.025,.975])
            saved=reference[(reference.group==group)&(reference.metric==metric)].iloc[0]
            same([point[j],lo,hi],[saved.estimate,saved.CI95_low,saved.CI95_high])
            rows.append(dict(group=group,metric=metric,estimate=point[j],CI95_low=lo,CI95_high=hi))
    write('four_group_agreement.csv',rows)
    diffs=[]
    for a,b in [('random40','single12'),('topAUC40','random40'),('topAUC40','matched40')]:
        for j,metric in enumerate(metrics):
            lo,hi=np.quantile(boot[a][:,j]-boot[b][:,j],[.025,.975])
            diffs.append(dict(comparison=a+' minus '+b,metric=metric,estimate=observed[a][j]-observed[b][j],CI95_low=lo,CI95_high=hi))
    write('paired_agreement_differences.csv',diffs)
    return {k:v.tolist() for k,v in observed.items()}

def sensitivity_groups():
    """Table S7 tuning rows: median top-ten Jaccard and full-ranking Spearman, random and high-AUC groups."""
    values=np.array([[np.load(REF/'tuned_member_shap'/f'shap_{m}_seed{s}.npz')['values'] for s in SEEDS] for m in NAMES])
    names=[f.strip() for f in np.load(REF/'tuned_member_shap'/'shap_LR_seed42.npz')['features'].astype(str)]
    rows=[]
    for group,label in [('random40','Random-40'),('topAUC40','High-AUC')]:
        z=np.load(REV/'sensitivity_random'/f'tuned_{group}.npz')
        importance=np.abs(np.einsum('km,msnf->ksnf',z['weights'],values)).mean(axis=(1,2))
        same(importance,z['importance'])
        stat=agreement(importance)
        rows.append(dict(variant='tuned',group=label,Jaccard_median=stat[0],Spearman_median=stat[2]))
        if group=='topAUC40':
            order=np.argsort(-importance[0],kind='stable');rank=int(np.flatnonzero(order==names.index('Backhand Returns'))[0]+1)
            assert rank==8
    write('tuned_rank_agreement.csv',rows)
    return rows

def rank_from_arrays(folder,prefix,focal,column=None):
    arrays=[np.load(folder/f'{prefix}{seed}.npz') for seed in SEEDS]
    values=np.stack([z['values'] for z in arrays])
    if column is not None:values=values[:,:,: ,column]
    names=arrays[0]['features'].astype(str)
    imp=np.abs(values).mean(axis=(0,1));order=np.argsort(-imp,kind='stable')
    j=int(np.flatnonzero(names==focal)[0])
    return int(np.flatnonzero(order==j)[0]+1),float(imp[j]),names,imp

def opponent():
    folder=REV/'opponent';rows=[]
    for j,scenario in enumerate(['fixed_members','reselected']):
        for cohort in ['development','temporal_2025']:
            p=pd.read_csv(folder/f'{cohort}_{scenario}_predictions.csv')
            rows.append(dict(scenario=scenario,cohort=cohort,AUC=roc_auc_score(p.label,p.score),Brier=float(np.mean((p.label-p.score)**2)),accuracy=accuracy_score(p.label,p.prediction)))
        arrays=[np.load(folder/f'adjusted_SHAP_seed{s}.npz') for s in SEEDS]
        names=arrays[0]['features'].astype(str)
        importance=np.abs(np.stack([z['values'][: ,:,j] for z in arrays])).mean(axis=(0,1))
        technical=np.flatnonzero(names!='Opponent ATP rank (log2)');order=technical[np.argsort(-importance[technical],kind='stable')]
        bh=int(np.flatnonzero(names=='Backhand Returns')[0]);rank=int(np.flatnonzero(order==bh)[0]+1)
        assert rank==6
        rows.append(dict(scenario=scenario,cohort='SHAP',backhand_rank=rank))
    write('opponent_adjustment.csv',rows)
    return rows

def representations():
    rows=[]
    scenarios=['court_drop_only','return_share_without_total','return_rates_without_total']
    for scenario in scenarios:
        folder=REV/'representations'/scenario
        result=json.loads((folder/'results.json').read_text(encoding='utf-8'))
        p=pd.read_csv(folder/'test_predictions.csv');auc=roc_auc_score(p.label,p.score)
        rank,importance,_,_=rank_from_arrays(folder,'SHAP_seed',result['focal_variable'])
        same([auc,rank,importance],[result['temporal_auc'],result['focal_shap_rank'],result['focal_mean_abs_shap']])
        if scenario=='court_drop_only':
            delta=auc-roc_auc_score(p.label,p.without_focal);method='deletion_refit';ci=result['temporal_focal_ablation_CI95']
        else:
            mask=pd.read_csv(folder/'mean_mask_predictions.csv');same(mask.full,p.score)
            delta=roc_auc_score(mask.label,mask.full)-roc_auc_score(mask.label,mask.mean_masked)
            same(delta,result['temporal_mean_mask_delta_auc'])
            match=mask.match_id.to_numpy();indices=[np.flatnonzero(match==g) for g in sorted(set(match))]
            rng=np.random.default_rng(20261111);draws=[]
            for _ in range(2000):
                ix=np.concatenate([indices[j] for j in rng.integers(0,len(indices),len(indices))])
                if len(np.unique(mask.label.to_numpy()[ix]))==2:
                    draws.append(roc_auc_score(mask.label.to_numpy()[ix],mask.full.to_numpy()[ix])-roc_auc_score(mask.label.to_numpy()[ix],mask.mean_masked.to_numpy()[ix]))
            ci=np.quantile(draws,[.025,.975]);same(ci,result['temporal_mean_mask_delta_CI95'])
            method='training_mean_mask; fitted model fixed'
        rows.append(dict(representation=scenario,AUC=auc,SHAP_rank=rank,mean_absolute_SHAP=importance,delta_AUC=delta,CI95_low=float(ci[0]),CI95_high=float(ci[1]),perturbation_method=method))
    write('court_depth_omission.csv',[r for r in rows if r['representation']=='court_drop_only'])
    write('return_representations_mean_mask.csv',[r for r in rows if r['representation']!='court_drop_only'])
    return rows

def reliability():
    folder=REV/'reliability';p=pd.read_csv(folder/'external_match_pairs.csv');rows=[]
    assert len(p)==36
    for name in ['Aces','Double Faults','Break Points','Service Breaks']:
        a=p['study_'+name].to_numpy();b=p['reference_'+name].to_numpy()
        rows.append(dict(indicator=name,matches=len(p),kappa=kappa(a,b),ICC_A1=icc(a,b)))
    write('external_recording_agreement.csv',rows)
    observer=pd.read_csv(folder/'observer_records.csv')
    count=observer[observer.count_indicator]
    pooled=kappa(count.observer1,count.observer2)
    assert len(count)==120;same(pooled,.9288361749444033)
    valid=[]
    for name,group in observer.groupby('indicator',sort=False):
        if group.device_indicator.any() or name.startswith('Set outcome'):continue
        value=icc(group.observer1,group.observer2)
        if value is not None:valid.append(value)
    same([min(valid),max(valid)],[.96,1.0])
    manual=observer[~observer.device_indicator & ~observer.indicator.str.startswith('Set outcome')]
    assert manual.indicator.nunique()==26 and len(manual)==156 and len(valid)==25
    assert int(np.sum(manual.observer1==manual.observer2))==143
    write('observer_agreement.csv',[dict(observer1_original_equal=int(np.sum(manual.observer1==manual.original)),observer2_original_equal=int(np.sum(manual.observer2==manual.original)),count_pairs=len(count),equal_pairs=int(np.sum(count.observer1==count.observer2)),pooled_descriptive_kappa=pooled,manual_indicators=26,manual_pairs=len(manual),manual_equal_pairs=int(np.sum(manual.observer1==manual.observer2)),ICC_estimable=len(valid),ICC_min=min(valid),ICC_max=max(valid))])
    return pooled

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    results={}
    for name,function in [('classification',classification),('partial_correlation',partial_correlation),
            ('primary_ablation',primary_ablation),('primary_agreement',primary_agreement),
            ('sensitivity_groups',sensitivity_groups),('opponent',opponent),
            ('representations',representations),('reliability',reliability)]:
        print('Supplementary reproduction:',name,flush=True)
        results[name]=function()
    results['original_workflow_S4_diagnostics_included']=True
    (OUT/'reproduction_checks.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print('Supplementary numerical reproduction passed; see outputs/revision2026.',flush=True)

if __name__=='__main__':main()
