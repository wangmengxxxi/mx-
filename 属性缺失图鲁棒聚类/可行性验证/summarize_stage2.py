"""Summarize calibration runs and verify their saved evaluation protocol."""
import csv
import hashlib
import importlib.util
import json
from collections import defaultdict, deque
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from run_sdcn_baseline import evaluate
from run_minimal_diagnostic import load_cora

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'sdcn_calibration_100_400'


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def main():
    rows = json.loads((OUT / 'results.json').read_text(encoding='utf-8'))
    _, labels, adjacency, _ = load_cora()
    groups = defaultdict(list)
    checks = []
    mask_hashes = defaultdict(set)
    reachability = {}
    fields = ['acc', 'nmi', 'ari', 'missing_acc', 'observed_acc', 'cold_acc',
              'occupied_clusters', 'largest_cluster_fraction',
              'last_prediction_change', 'elapsed_sec', 'cold_count']
    for row in rows:
        groups[(row['rate'], row['method'])].append(row)
        folder = OUT / f"r{row['rate']:g}_m{row['mask']}_{row['method']}_s{row['seed']}"
        with np.load(folder / 'predictions.npz') as data:
            observed, cold = data['observed'], data['cold']
            digest = hashlib.sha256(observed.tobytes()).hexdigest()
            assert digest == row['mask_sha256']
            assert int((~observed).sum()) == row['missing_count']
            expected_cold = ~observed & (adjacency[:, observed].sum(1).A1 < 1e-8)
            assert np.array_equal(cold, expected_cold)
            if digest not in reachability:
                distance = np.full(len(labels), -1, dtype=np.int64)
                distance[observed] = 0
                queue = deque(np.flatnonzero(observed))
                while queue:
                    node = queue.popleft()
                    for neighbor in adjacency.indices[adjacency.indptr[node]:adjacency.indptr[node + 1]]:
                        if distance[neighbor] == -1:
                            distance[neighbor] = distance[node] + 1
                            queue.append(neighbor)
                reachability[digest] = dict(rate=row['rate'], mask=row['mask'],
                    one_hop_cold=int(cold.sum()),
                    unreachable_missing=int(((distance < 0) & ~observed).sum()),
                    reachable_after_multiple_hops=int(((distance > 1) & ~observed).sum()),
                    max_hops=int(distance.max()))
            for branch, prefix in [('pred', ''), ('q', 'q_'), ('ae', 'ae_')]:
                measured = evaluate(labels, data[branch], observed, cold)
                for key, value in measured.items():
                    stored = row[prefix + key]
                    assert value == stored if value is None else np.isclose(value, stored)
        mask_hashes[(row['rate'], row['mask'])].add(digest)
        losses = read_csv(folder / 'loss.csv')
        assert len(losses) == 500
        assert all(np.isfinite(float(item['total'])) for item in losses)
        checks.append(dict(run=folder.name, predictions_recomputed=True,
                           mask_verified=True, finite_losses=True))
    assert len(rows) == 10
    assert all(len(hashes) == 1 for hashes in mask_hashes.values())
    summary = []
    table = ['| 缺失率 | 输入 | ACC | NMI | ARI | 缺失节点 ACC | 冷启动节点 ACC | 有效簇（逐种子） |',
             '|---:|---|---:|---:|---:|---:|---:|---|']
    for (rate, method), runs in sorted(groups.items()):
        item = dict(rate=rate, method=method, runs=len(runs), masks=len({r['mask'] for r in runs}))
        for field in fields:
            values = [r[field] for r in runs if r[field] is not None]
            item[field + '_mean'] = float(np.mean(values)) if values else None
            item[field + '_std'] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        summary.append(item)
        def cell(field):
            mean, std = item[field + '_mean'], item[field + '_std']
            return '—' if mean is None else f'{mean:.4f} ± {std:.4f}'
        table.append(f'| {rate:.0%} | {method} | ' + ' | '.join(cell(f) for f in
                     ['acc', 'nmi', 'ari', 'missing_acc', 'cold_acc']) +
                     ' | ' + ', '.join(str(r['occupied_clusters']) for r in runs) + ' |')
    with (OUT / 'summary.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (OUT / 'summary_table.md').write_text('\n'.join(table) + '\n', encoding='utf-8')
    (OUT / 'verification.json').write_text(json.dumps(dict(
        checks=checks, shared_masks_verified=True,
        reachability=list(reachability.values()),
        training_seconds_sum=sum(r['elapsed_sec'] for r in rows),
        standard_deviation='sample std across 2 model seeds, 1 mask; not a confidence interval'),
        indent=2), encoding='utf-8')

    colors = {0: '#2864a6', .4: '#bb6b15', .8: '#298255'}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for row in rows:
        folder = OUT / f"r{row['rate']:g}_m{row['mask']}_{row['method']}_s{row['seed']}"
        history = read_csv(folder / 'metrics.csv')
        label = f"{row['rate']:.0%} {row['method']} s{row['seed']}"
        style = '--' if row['method'] == 'zero' else '-'
        for ax, key in zip(axes[0], ['acc', 'nmi', 'occupied_clusters']):
            ax.plot([int(h['epoch']) for h in history], [float(h[key]) for h in history],
                    color=colors[row['rate']], linestyle=style,
                    alpha=1 if row['seed'] == 0 else .45, label=label)
        if row['rate'] == .8 and row['seed'] == 0:
            joint = [h for h in read_csv(folder / 'loss.csv') if h['phase'] == 'joint']
            for ax, key in zip(axes[1], ['mse', 'kl_q', 'kl_pred']):
                ax.plot([int(h['epoch']) for h in joint], [float(h[key]) for h in joint],
                        label=row['method'], linestyle=style)
    for ax, name in zip(axes.flat, ['ACC', 'NMI', 'Occupied clusters',
                                   'MSE (80%, seed 0)', 'KL q (80%, seed 0)', 'KL pred (80%, seed 0)']):
        ax.set_title(name)
        ax.set_xlabel('Joint training epoch')
        ax.grid(alpha=.2)
    axes[0, 2].legend(fontsize=7, ncol=2)
    axes[1, 0].legend()
    fig.suptitle('SDCN adaptation: fixed final epoch; 1 mask, 2 model seeds')
    fig.savefig(OUT / 'training_diagnostics.png', dpi=160)
    plt.close(fig)

    manifest = {}
    for name in ['torch_geometric', 'torch_scatter', 'torch_sparse', 'munkres', 'yaml', 'pypdf', 'pdfplumber']:
        manifest[name] = importlib.util.find_spec(name) is not None
    (ROOT / 'references' / 'amgc_dependency_check.json').write_text(
        json.dumps(manifest, indent=2), encoding='utf-8')
    print('\n'.join(table))
    print('Verified 10 runs; training seconds:', sum(r['elapsed_sec'] for r in rows))
    print('Dependencies:', manifest)


if __name__ == '__main__':
    main()
