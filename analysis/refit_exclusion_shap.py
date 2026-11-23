"""KernelSHAP for the twelve-indicator exclusion (Supplementary Table S7).

Run after ``python analysis/refit_primary.py --variant twelve``. Each of the 12
refitted, calibrated learners is explained on the 40 sets of the 2025 cohort
(50 training background observations, 2,048 samples, seeds 42/142/242). The 40
ensembles with the highest development OOF AUC are composed linearly from these
member explanations, and the Backhand Returns rank is reported for each. The
first ensemble is the selected model; the count of ensembles with Backhand
Returns in the top six is the "top-six frequency among top 40" entry of Table S7.
"""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS']:
    os.environ[key] = '1'
import argparse
import json
import sys
import time
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / 'analysis'))
import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from reproduce import NAMES, WEIGHTS, REF

SEEDS = [42, 142, 242]
FOCAL = 'Backhand Returns'
WORK = PACKAGE / 'outputs/refit_twelve'
OUT = WORK / 'member_shap'


def explain_member(name):
    import shap
    OUT.mkdir(exist_ok=True)
    bundle = joblib.load(WORK / 'final_models.joblib')
    train = pd.read_csv(PACKAGE / 'data/processed/train_grouped.csv')
    test = pd.read_csv(PACKAGE / 'data/processed/test_grouped.csv')
    features = bundle['features']
    x = train[features].to_numpy(float)
    xt = test[features].to_numpy(float)
    model = bundle['models'][name]
    calibrator = bundle['calibrators'][name]
    kept = model.named_steps['screen'].keep_
    mean_train = x.mean(axis=0)

    def predict(z):
        restored = np.tile(mean_train, (len(z), 1))
        restored[:, kept] = z
        return calibrator.predict(model.decision_function(restored))

    expected = np.load(WORK / 'member_predictions.npz')['test_members'][:, NAMES.index(name)]
    assert np.max(np.abs(predict(xt[:, kept]) - expected)) < 1e-9
    background = shap.sample(x[:, kept], 50, random_state=42)
    for seed in SEEDS:
        cache = OUT / f'shap_{name}_seed{seed}.npz'
        if not cache.exists():
            np.random.seed(seed)
            explainer = shap.KernelExplainer(predict, background)
            values = np.asarray(explainer.shap_values(xt[:, kept], nsamples=2048, l1_reg=0, silent=True))
            assert np.max(np.abs(values.sum(axis=1) + explainer.expected_value - expected)) < 1e-5
            full = np.zeros((len(xt), len(features)))
            full[:, kept] = values
            np.savez_compressed(cache, values=full, features=np.asarray(features), base=float(explainer.expected_value))
        print(f'{name} seed {seed} complete', flush=True)
    return name


def summarize(check_reference=False):
    bundle = joblib.load(WORK / 'final_models.joblib')
    features = np.asarray(bundle['features'])
    focal = int(np.flatnonzero(features == FOCAL)[0])
    train = pd.read_csv(PACKAGE / 'data/processed/train_grouped.csv')
    y = train.iloc[:, 0].to_numpy(int)
    scores = np.load(WORK / 'member_predictions.npz')['train_platt_oof'] @ WEIGHTS
    r = rankdata(scores, axis=0, method='average')
    auc = (r[y == 1].sum(axis=0) - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (1 - y).sum())
    order = np.argsort(-np.round(auc, 12), kind='stable')
    assert int(order[0]) == bundle['selection']['index']
    weights = WEIGHTS[:, order[:40]].T
    values = np.array([[np.load(OUT / f'shap_{m}_seed{s}.npz')['values'] for s in SEEDS] for m in NAMES])
    importance = np.abs(np.einsum('km,msnf->ksnf', weights, values)).mean(axis=1).mean(axis=1)
    ranks = np.empty(importance.shape, dtype=int)
    np.put_along_axis(ranks, np.argsort(-importance, axis=1, kind='stable'),
                      np.tile(np.arange(1, len(features) + 1), (40, 1)), axis=1)
    table = pd.DataFrame({'group': 'topAUC40', 'position': np.arange(1, 41), 'backhand_rank': ranks[:, focal]})
    table.to_csv(WORK / 'topAUC40_backhand_ranks.csv', index=False)
    summary = {'selected_members': bundle['selection']['members'],
               'selected_backhand_shap_rank': int(ranks[0, focal]),
               'backhand_top6_count_top40': int(np.sum(ranks[:, focal] <= 6)),
               'n_ensembles': 40}
    (WORK / 'exclusion_shap_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    if check_reference:
        reference = pd.read_csv(REF / 'revision2026/twelve_agreement/topAUC40_backhand_ranks.csv')
        assert table.equals(reference), 'Top-40 Backhand ranks differ from the deposited record'
        print('Top-40 Backhand ranks match the deposited record', flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--check-reference', action='store_true')
    args = parser.parse_args()
    from concurrent.futures import ProcessPoolExecutor
    start = time.time()
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        list(pool.map(explain_member, NAMES))
    summarize(args.check_reference)
    print('Explanations complete; elapsed seconds', round(time.time() - start, 1), flush=True)
