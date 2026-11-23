"""Table S10 return-block diagnostics: maximum VIF and column rank.

For the two S10 representations (backhand share with Total Returns omitted;
backhand and forehand rates with Total Returns omitted) the full-development
feature screen is learned, and the maximum variance inflation factor and the
column rank are computed across the retained return-related predictors.
"""
from pathlib import Path
import sys, json
P = Path(__file__).resolve().parents[1]
W = P / 'outputs'
W.mkdir(exist_ok=True)
sys.path.insert(0, str(P / 'analysis'))
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from models import FoldScreen

tr = pd.read_csv(P / 'data/processed/train_grouped.csv')
features = list(tr.columns[1:35])
y = tr['Set Result'].to_numpy()
LABELS = {'return_share_without_total': 'Share; T omitted', 'return_rates_without_total': 'Rates; T omitted'}


def transform(df, scenario):
    d = df[features].copy()
    bh, fh, t = d['Backhand Returns'], d['Forehand Returns'], d['Total Returns']
    if scenario == 'return_share_without_total':
        d['BH share'] = bh / (bh + fh)
        return d.drop(columns=['Backhand Returns', 'Forehand Returns', 'Total Returns'])
    if scenario == 'return_rates_without_total':
        d['Backhand Returns'] = bh / t
        d['Forehand Returns'] = fh / t
        return d.drop(columns='Total Returns')
    raise ValueError(scenario)


def vif(x):
    z = (x - x.mean(axis=0)) / x.std(axis=0)
    scores = []
    for j in range(z.shape[1]):
        v = z[:, j]
        q = np.delete(z, j, axis=1)
        res = v - q @ np.linalg.lstsq(q, v, rcond=None)[0]
        ratio = sum(res ** 2) / sum(v ** 2)
        scores.append(float(1 / ratio) if ratio > 1e-12 else float('inf'))
    return max(scores)


out = {}
for scenario, label in LABELS.items():
    d = transform(tr, scenario)
    screen = FoldScreen().fit(d.to_numpy(float), y)
    kept = list(d.columns[screen.keep_])
    block = [c for c in kept if 'Return' in c or c == 'BH share']
    x = d[block].to_numpy(float)
    rank = int(np.linalg.matrix_rank(x - x.mean(axis=0)))
    out[scenario] = dict(label=label, retained_return_predictors=block, max_vif=vif(x),
                         column_rank=rank, columns=len(block), full_column_rank=rank == len(block))
    print(label, 'max VIF', round(out[scenario]['max_vif'], 3), 'rank', rank, '/', len(block))
assert round(out['return_share_without_total']['max_vif'], 2) == 4.58
assert round(out['return_rates_without_total']['max_vif'], 2) == 4.69
assert all(v['full_column_rank'] for v in out.values())
(W / 'return_representation_diagnostics.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
