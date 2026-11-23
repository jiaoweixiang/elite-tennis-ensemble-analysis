"""Rank all 34 deletion-refit comparisons by their 2025 test AUC decrease."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[1]
REF=ROOT/'data/reference'
def main():
    output=ROOT/'outputs/revision2026';output.mkdir(parents=True,exist_ok=True)
    table=pd.read_csv(REF/'feature_ablation.csv')
    assert len(table)==34 and table.feature.str.strip().nunique()==34
    # Twelve decimals resolve floating-point round-off only, not measured differences.
    drops=table.test_full_auc-table.test_without_auc
    assert np.allclose(drops,table.test_auc_drop,atol=1e-12,rtol=0)
    rounded=np.round(drops.to_numpy(float),12)
    table['test_ablation_rank']=pd.Series(rounded).rank(method='min',ascending=False).astype(int)
    table['tied_indicators']=pd.Series(rounded).map(pd.Series(rounded).value_counts()).to_numpy(int)
    p=pd.read_csv(REF/'revision2026/primary_ablation/test_ablation_predictions.csv')
    full=roc_auc_score(p.label,p.full)
    for col in p.columns[3:]:
        expected=table.loc[table.feature.str.strip()==col.strip()].iloc[0]
        actual=full-roc_auc_score(p.label,p[col])
        assert abs(actual-expected.test_auc_drop)<1e-12
    focal=table.loc[table.feature.str.strip()=='Backhand Returns'].iloc[0]
    assert int(focal.test_ablation_rank)==5 and int(focal.tied_indicators)==5
    assert np.sum(rounded>round(float(focal.test_auc_drop),12))==4
    cols=['feature','test_full_auc','test_without_auc','test_auc_drop','test_ablation_rank','tied_indicators']
    table[cols].sort_values(['test_ablation_rank','feature'],kind='stable').to_csv(output/'ablation_test_rank_all34.csv',index=False)
    result={'indicators':34,'focal':'Backhand Returns','delta_AUC':float(focal.test_auc_drop),'rank':5,'ranking':'Descending 2025 test deletion-refit AUC decrease, standard competition ranks for ties','indicators_at_focal_rank':5,'indicators_with_strictly_larger_effect':4,'prediction_columns_independently_checked':len(p.columns)-3}
    (output/'ablation_test_rank_all34.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
