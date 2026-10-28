"""Offline error calibration by frozen confidence quintile; never trains on hidden truth."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from run_minimal_diagnostic import load_cora, fp
from run_confidence_probe import signals


ROOT = Path(__file__).resolve().parent
features, _, adjacency, _ = load_cora()
records = []
signal_records = []
for rate in (.4, .8):
    for mask in range(5):
        path = ROOT / 'confidence_stage4_holdout' / f'r{rate:g}_m{mask}.npz'
        with np.load(path) as archive:
            observed = archive['observed']
            score = archive['score']
        reconstructed = fp(features, observed, adjacency, 40)
        error = np.sqrt(np.mean((features - reconstructed) ** 2, axis=1))
        missing = np.flatnonzero(~observed)
        node_signals = signals(observed, reconstructed, adjacency)
        signal_records.append(dict(rate=rate, mask=mask,
            learned=float(spearmanr(score[missing], error[missing]).statistic),
            one_hop_coverage=float(spearmanr(node_signals[missing, 0], error[missing]).statistic),
            two_hop_coverage=float(spearmanr(node_signals[missing, 1], error[missing]).statistic),
            fp_norm=float(spearmanr(node_signals[missing, 3], error[missing]).statistic)))
        sorted_ids = missing[np.argsort(score[missing])]
        for rank, group in enumerate(np.array_split(sorted_ids, 5), 1):
            records.append(dict(rate=rate, mask=mask, confidence_quintile=rank,
                                count=len(group), mean_rmse=float(error[group].mean()),
                                median_rmse=float(np.median(error[group]))))
with (ROOT / 'confidence_stage4_holdout' / 'quintile_errors.csv').open('w', newline='', encoding='utf-8-sig') as f:
    writer = csv.DictWriter(f, fieldnames=records[0].keys())
    writer.writeheader()
    writer.writerows(records)
with (ROOT / 'confidence_stage4_holdout' / 'signal_correlations.csv').open('w', newline='', encoding='utf-8-sig') as f:
    writer = csv.DictWriter(f, fieldnames=signal_records[0].keys())
    writer.writeheader()
    writer.writerows(signal_records)
for rate in (.4, .8):
    means = [np.mean([r['mean_rmse'] for r in records if r['rate'] == rate and r['confidence_quintile'] == q]) for q in range(1, 6)]
    print(f'{rate:.0%} missing: low-to-high confidence RMSE ' + ', '.join(f'{v:.4f}' for v in means))
