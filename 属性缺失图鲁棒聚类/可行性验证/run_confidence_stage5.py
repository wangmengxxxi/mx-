"""Label-free FP confidence calibration, with hidden Cora attributes used only for audit.

The score is fit on deliberately hidden *observed* nodes. True attributes of
the originally missing nodes are consulted only after scores are frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from run_minimal_diagnostic import load_cora, fp


ROOT = Path(__file__).resolve().parent


def signals(observed, filled, norm_adj):
    adjacency = norm_adj.copy()
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    adjacency.data[:] = 1
    adjacency = adjacency.tocsr()
    counts1 = np.asarray(adjacency @ observed.astype(np.float32)).ravel()
    counts2 = np.asarray(adjacency @ counts1).ravel()
    degree = np.asarray(adjacency.sum(1)).ravel()
    norm = np.linalg.norm(filled, axis=1)
    return np.column_stack((np.log1p(counts1), np.log1p(counts2),
                            np.log1p(degree), norm)).astype(np.float64)


def error(features, filled):
    return np.sqrt(np.mean((features - filled) ** 2, axis=1))


def correlation(score, actual):
    value = spearmanr(score, actual).statistic
    return float(value) if np.isfinite(value) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rates', default='0.4,0.8')
    ap.add_argument('--masks', type=int, default=5)
    ap.add_argument('--mask-seed', type=int, default=20261023)
    ap.add_argument('--mask-mode', choices=['random', 'degree_biased'], default='random')
    ap.add_argument('--holdout-fraction', type=float, default=.2)
    ap.add_argument('--steps', type=int, default=40)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=False)
    (out / 'config.json').write_text(json.dumps(vars(args), indent=2), encoding='utf-8')
    features, _, norm_adj, degree = load_cora()
    summary = []
    for rate in [float(v) for v in args.rates.split(',')]:
        for mask in range(args.masks):
            rng = np.random.default_rng(np.random.SeedSequence([args.mask_seed, round(rate * 10000), mask]))
            observed = np.ones(len(features), dtype=bool)
            probabilities = (np.sqrt(degree + 1) if args.mask_mode == 'degree_biased' else None)
            if probabilities is not None:
                probabilities = probabilities / probabilities.sum()
            observed[rng.choice(len(features), round(rate * len(features)), replace=False,
                                p=probabilities)] = False
            original_digest = hashlib.sha256(observed.tobytes()).hexdigest()
            known = np.flatnonzero(observed)
            fold_rng = np.random.default_rng(np.random.SeedSequence([args.mask_seed, round(rate * 10000), mask, 9187]))
            held = fold_rng.permutation(known)
            count = max(30, round(args.holdout_fraction * len(known)))
            folds = [held[:count], held[count:2*count]]
            matrices, errors = [], []
            for fold in folds:
                train_observed = observed.copy()
                train_observed[fold] = False
                fill = fp(features, train_observed, norm_adj, args.steps)
                matrices.append(signals(train_observed, fill, norm_adj)[fold])
                errors.append(error(features[fold], fill[fold]))
            model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            model.fit(matrices[0], np.log(errors[0] + 1e-6))
            test_score = -model.predict(matrices[1])
            fill_original = fp(features, observed, norm_adj, args.steps)
            missing = ~observed
            all_score = -model.predict(signals(observed, fill_original, norm_adj))
            actual_missing_error = error(features[missing], fill_original[missing])
            # Equal mean and distribution across informed/shuffled ablations.
            rank = np.argsort(np.argsort(all_score[missing], kind='stable'), kind='stable')
            weights = np.ones(len(features), dtype=np.float32)
            weights[missing] = .2 + .8 * (rank + .5) / int(missing.sum())
            shuffled = weights.copy()
            shuffle_rng = np.random.default_rng(np.random.SeedSequence([args.mask_seed, round(rate * 10000), mask, 8128]))
            shuffled[missing] = shuffle_rng.permutation(shuffled[missing])
            name = f'r{rate:g}_m{mask}'
            np.savez_compressed(out / f'{name}.npz', observed=observed,
                                confidence=weights, shuffled=shuffled,
                                score=all_score, fold_cal=folds[0], fold_test=folds[1])
            row = dict(rate=rate, mask=mask, mask_sha256=original_digest,
                       calibration_count=len(folds[0]), test_count=len(folds[1]),
                       test_spearman=correlation(test_score, errors[1]),
                       missing_spearman=correlation(all_score[missing], actual_missing_error),
                       test_rmse_mean=float(errors[1].mean()),
                       missing_rmse_mean=float(actual_missing_error.mean()),
                       missing_count=int(missing.sum()))
            summary.append(row)
            print(name, row, flush=True)
    (out / 'results.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
