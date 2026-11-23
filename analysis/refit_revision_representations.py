"""Nested sensitivity refits for Supplementary Tables S9 and S10.

Scenarios (5 outer / 5 inner / 3 calibration match-grouped folds, development-only
AUC selection among all 4,095 subsets, as in the primary workflow):
  court_drop_only             S9: Backcourt Deep Balls omitted; Backhand Returns
                              2025 dAUC by deletion and refitting with members fixed
  return_share_without_total  S10: BH/(BH+FH) replaces BH and FH; Total Returns omitted
  return_rates_without_total  S10: BH/T and FH/T replace BH and FH; Total Returns omitted
For the S10 representations the 2025 dAUC is the training-mean mask: the focal
predictor is replaced by its development-set mean while all fitted learners,
screening, calibration and ensemble members stay fixed. Outputs and caches are
written to outputs/revision2026/representations.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key]='1'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/revision2026/representations'
OUT.mkdir(parents=True,exist_ok=True)
import argparse,hashlib,itertools,json,time,warnings
import numpy as np
import pandas as pd
import joblib
from scipy.stats import rankdata
from sklearn.calibration import _SigmoidCalibration
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score,roc_curve
import revision_models as rm

TRAIN_PATH=ROOT/'data/processed/train_grouped.csv'
TEST_PATH=ROOT/'data/processed/test_grouped.csv'
tr=pd.read_csv(TRAIN_PATH);te=pd.read_csv(TEST_PATH)
y=tr['Set Result'].to_numpy(int);yt=te['Set Result'].to_numpy(int)
groups=tr.cv_group.to_numpy(str)
original_features=list(tr.columns[1:35])
COMBOS=[c for k in range(1,13) for c in itertools.combinations(range(12),k)]
W=np.zeros((12,len(COMBOS)))
for j,c in enumerate(COMBOS): W[list(c),j]=1/len(c)
SCENARIOS=['court_drop_only','return_share_without_total','return_rates_without_total']


def transform(df,scenario):
    data=df[original_features].copy()
    if scenario=='court_drop_only': return data.drop(columns=['Backcourt Deep Balls'])
    bh=data['Backhand Returns'].to_numpy(float);fh=data['Forehand Returns'].to_numpy(float)
    total=data['Total Returns'].to_numpy(float);classified=bh+fh
    assert np.all(classified>0) and np.all(total>0)
    if scenario=='return_rates_without_total':
        data['Backhand Returns']=bh/total
        data['Forehand Returns']=fh/total
        return data.rename(columns={'Backhand Returns':'Backhand returns per recorded total','Forehand Returns':'Forehand returns per recorded total'}).drop(columns=['Total Returns'])
    if scenario=='return_share_without_total':
        data['Backhand share among classified returns']=bh/classified
        return data.drop(columns=['Total Returns','Backhand Returns','Forehand Returns'])
    raise ValueError(scenario)


def split(idx,X,k):
    idx=np.asarray(idx)
    out=[]
    for a,b in StratifiedGroupKFold(k,shuffle=True,random_state=42).split(X[idx],y[idx],groups[idx]):
        assert not set(groups[idx[a]]) & set(groups[idx[b]])
        assert len(np.unique(y[idx[a]]))==2
        out.append((idx[a],idx[b]))
    return out


def choose(matrix,labels):
    scores=matrix@W
    r=rankdata(scores,axis=0,method='average')
    positive=np.sum(labels==1);negative=np.sum(labels==0)
    auc=(r[labels==1].sum(axis=0)-positive*(positive+1)/2)/(positive*negative)
    j=int(np.flatnonzero(np.isclose(auc,auc.max(),atol=1e-12,rtol=0))[0])
    fpr,tpr,thresholds=roc_curve(labels,scores[:,j],drop_intermediate=False)
    threshold=float(thresholds[int(np.argmax(tpr-fpr))])
    return {'index':j,'columns':list(COMBOS[j]),'members':[rm.NAMES[i] for i in COMBOS[j]],
        'inner_auc':float(auc[j]),'threshold':threshold},auc


def main(scenario):
    import sklearn
    started=time.time()
    deletion=scenario=='court_drop_only'
    directory=OUT/'nested_sensitivity'/scenario;directory.mkdir(parents=True,exist_ok=True)
    cache=directory/'cache';cache.mkdir(exist_ok=True)
    data=transform(tr,scenario);data_test=transform(te,scenario)
    X=data.to_numpy(float);T=data_test.to_numpy(float);features=list(data.columns)
    def log(message):
        line=f'[{time.time()-started:.1f}s] {scenario}: {message}'
        print(line,flush=True)
        with (directory/'run.log').open('a',encoding='utf-8') as f:f.write(line+'\n')
    def block(A,idx,target,names=rm.NAMES,save=False):
        idx=np.asarray(idx)
        key=hashlib.sha256(A.tobytes()+idx.tobytes()+target.tobytes()+'+'.join(names).encode()).hexdigest()[:24]
        cp=cache/(key+'.npz')
        if cp.exists() and not save:return np.load(cp)['predictions'],None,None
        splits=split(idx,A,3)
        local={int(v):i for i,v in enumerate(idx)}
        pred=[];models=[];cals=[]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for name in names:
                oof=np.full(len(idx),np.nan)
                for a,b in splits:
                    m=rm.pipeline(name).fit(A[a],y[a])
                    oof[[local[int(i)] for i in b]]=m.decision_function(A[b])
                cal=_SigmoidCalibration().fit(oof,y[idx])
                m=rm.pipeline(name).fit(A[idx],y[idx])
                pred.append(cal.predict(m.decision_function(target)))
                models.append(m);cals.append(cal)
        pred=np.column_stack(pred);assert np.isfinite(pred).all()
        np.savez_compressed(cp,predictions=pred)
        return pred,models,cals
    def oof(idx):
        idx=np.asarray(idx);local={int(v):i for i,v in enumerate(idx)}
        result=np.full((len(idx),12),np.nan)
        for a,b in split(idx,X,5):
            p,_,_=block(X,a,X[b]);result[[local[int(i)] for i in b]]=p
        assert np.isfinite(result).all()
        return result
    focal=('Backhand share among classified returns' if scenario=='return_share_without_total' else
           'Backhand returns per recorded total' if scenario=='return_rates_without_total' else 'Backhand Returns')
    focal_idx=features.index(focal)
    outer=np.full(len(y),np.nan);outer_pred=np.zeros(len(y),int);choices=[]
    IDX=np.arange(len(y))
    for fold,(a,b) in enumerate(split(IDX,X,5),1):
        ch,_=choose(oof(a),y[a])
        p,_,_=block(X,a,X[b],ch['members'])
        outer[b]=p.mean(axis=1);outer_pred[b]=outer[b]>=ch['threshold']
        choices.append({'fold':fold,**ch})
        log(f'outer {fold}/5 finished; members={ch["members"]}')
    final_oof=oof(IDX);ch,allauc=choose(final_oof,y)
    log('Final choice '+str(ch))
    p,models,cals=block(X,IDX,T,ch['members'],save=True)
    pt=p.mean(axis=1)
    test_table={'match_id':te.match_id,'label':yt,'score':pt,'pred':(pt>=ch['threshold']).astype(int)}
    if deletion:
        # S9: delete the focal predictor and refit, holding the selected members fixed.
        Xminus=np.delete(X,focal_idx,axis=1);Tminus=np.delete(T,focal_idx,axis=1)
        pm,_,_=block(Xminus,IDX,Tminus,ch['members'])
        ptminus=pm.mean(axis=1);test_table['without_focal']=ptminus
    else:
        # S10: training-mean mask; fitted learners, screening, calibration and members unchanged.
        masked=T.copy();masked[:,focal_idx]=X[:,focal_idx].mean()
        ptmean=np.column_stack([cal.predict(m.decision_function(masked)) for m,cal in zip(models,cals)]).mean(axis=1)
        pd.DataFrame({'match_id':te.match_id,'label':yt,'full':pt,'mean_masked':ptmean}).to_csv(directory/'mean_mask_predictions.csv',index=False)
    joblib.dump({'models':models,'calibrators':cals,'members':ch['members'],'features':features},directory/'final_models.joblib')
    np.savez_compressed(directory/'final_oof.npz',probabilities=final_oof)
    pd.DataFrame({'members':['+'.join(rm.NAMES[i] for i in c) for c in COMBOS],'inner_auc':allauc}).to_csv(directory/'all4095_selection_auc.csv',index=False)
    pd.DataFrame({'match_id':tr.match_id,'label':y,'score':outer,'pred':outer_pred}).to_csv(directory/'outer_predictions.csv',index=False)
    pd.DataFrame(test_table).to_csv(directory/'test_predictions.csv',index=False)
    result={'scenario':scenario,'table':'S9' if deletion else 'S10','n_features':len(features),'focal_variable':focal,
        'outer_auc':float(roc_auc_score(y,outer)),'temporal_auc':float(roc_auc_score(yt,pt)),
        'final_choice':ch,'outer_choices':choices,'sklearn_version':sklearn.__version__,
        'source_train_sha256':hashlib.sha256(TRAIN_PATH.read_bytes()).hexdigest(),
        'source_test_sha256':hashlib.sha256(TEST_PATH.read_bytes()).hexdigest()}
    log('AUC outer/test '+str((result['outer_auc'],result['temporal_auc'])))
    import shap
    kept=np.unique(np.concatenate([m.named_steps['screen'].keep_ for m in models]))
    mean=X.mean(axis=0)
    def predict(A):
        restored=np.tile(mean,(len(A),1));restored[:,kept]=A
        return np.column_stack([cal.predict(m.decision_function(restored)) for m,cal in zip(models,cals)]).mean(axis=1)
    assert np.max(np.abs(predict(T[:,kept])-pt))<1e-10
    background=shap.sample(X[:,kept],50,random_state=42)
    value_list=[]
    for seed in [42,142,242]:
        sp=directory/f'SHAP_seed{seed}.npz'
        if sp.exists():v=np.load(sp)['values']
        else:
            np.random.seed(seed)
            explainer=shap.KernelExplainer(predict,background)
            v=np.asarray(explainer.shap_values(T[:,kept],nsamples=2048,l1_reg=0,silent=True))
            assert np.max(np.abs(v.sum(axis=1)+explainer.expected_value-pt))<1e-6
            np.savez_compressed(sp,values=v,features=np.array(features)[kept],base=float(explainer.expected_value))
        value_list.append(v);log('SHAP seed '+str(seed)+' finished')
    absolute=np.abs(np.stack(value_list)).mean(axis=0)
    importance=absolute.mean(axis=0);order=np.argsort(-importance,kind='stable')
    names=np.array(features)[kept]
    ranking=[{'rank':i+1,'feature':names[j],'mean_abs_shap':float(importance[j])} for i,j in enumerate(order)]
    pd.DataFrame(ranking).to_csv(directory/'SHAP_ranks.csv',index=False,encoding='utf-8-sig')
    focus=int(np.flatnonzero(names==focal)[0]);rank=int(np.flatnonzero(order==focus)[0]+1)
    # 2,000 whole-match resamples of the 2025 cohort (one generator for ranks and dAUC).
    rng=np.random.default_rng(20261111);indexes=[np.flatnonzero(te.match_id.to_numpy()==g) for g in np.unique(te.match_id)]
    boot_rank=[];boot_delta=[]
    for rep in range(2000):
        idx=np.concatenate([indexes[i] for i in rng.integers(0,len(indexes),len(indexes))])
        o=np.argsort(-absolute[idx].mean(axis=0),kind='stable');boot_rank.append(int(np.flatnonzero(o==focus)[0]+1))
        if len(np.unique(yt[idx]))==2:
            altered=ptminus[idx] if deletion else ptmean[idx]
            boot_delta.append(float(roc_auc_score(yt[idx],pt[idx])-roc_auc_score(yt[idx],altered)))
    result.update({'focal_shap_rank':rank,'focal_mean_abs_shap':float(importance[focus]),
        'focal_rank_bootstrap_CI95':list(map(float,np.quantile(boot_rank,[.025,.975]))),
        'SHAP_budget':2048,'SHAP_seeds':[42,142,242],'SHAP_background':50,
        'bootstrap_seed':20261111,'bootstrap_draws':2000})
    if deletion:
        result.update({'temporal_focal_ablation_delta_auc':float(roc_auc_score(yt,pt)-roc_auc_score(yt,ptminus)),
            'temporal_focal_ablation_CI95':list(map(float,np.quantile(boot_delta,[.025,.975]))),
            'ablation_definition':'Delete Backhand Returns and refit screening, learners and calibration, holding the selected members fixed.'})
    else:
        result.update({'temporal_mean_mask_delta_auc':float(roc_auc_score(yt,pt)-roc_auc_score(yt,ptmean)),
            'temporal_mean_mask_delta_CI95':list(map(float,np.quantile(boot_delta,[.025,.975]))),
            'mean_mask_definition':'Replace the focal transformed predictor by its development-set mean; all fitted learners, screening, calibration and ensemble members remain fixed.'})
    (directory/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    log('DONE')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter);parser.add_argument('scenario',choices=SCENARIOS)
    main(parser.parse_args().scenario)
