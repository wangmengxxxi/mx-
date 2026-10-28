"""Instrumented AMGC CPU run, retaining the archived model and losses.

Explicit seeds and non-mutating Hungarian evaluation replace the original
unseeded, potentially prediction-mutating evaluator. No label-based selection.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import math
import random
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import psutil
import torch
import yaml
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from torch_geometric.datasets import Planetoid

from run_minimal_diagnostic import load_cora
from run_sdcn_baseline import evaluate, save_csv

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'amgc_work'))
from models.em import EM
from trains.train_em import compute_centers, compute_cluster_loss_no_contrastive
from utils.loss_funcs import ce_loss
from utils.misc import MaskEdge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--schedule-epochs', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--split-seed', type=int, default=72)
    ap.add_argument('--observed-ratio', type=float, default=.4)
    ap.add_argument('--edge-mask-p', type=float, default=.5)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--eval-every', type=int, default=1)
    ap.add_argument('--output', required=True)
    cli = ap.parse_args()
    if not 0 < cli.observed_ratio <= 1 or cli.epochs < 1:
        raise ValueError('Invalid observed ratio or epochs')
    out = ROOT / cli.output
    out.mkdir(parents=True, exist_ok=False)
    versions = {n: importlib.metadata.version(n) for n in
                ['torch', 'numpy', 'scipy', 'scikit-learn', 'torch-geometric', 'xxhash', 'munkres']}
    cfg = yaml.safe_load((ROOT / 'amgc_work/configs/EM/configs.yml').read_text())['cora']
    cfg.update(in_features=1433, num_class=7, weight_decay=5e-4)
    cfg['lr'] = float(cfg['lr'])
    args = SimpleNamespace(**cfg)
    source_hashes = {str(p.relative_to(ROOT / 'amgc_work')): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in (ROOT / 'amgc_work').rglob('*.py')}
    (out / 'config.json').write_text(json.dumps(dict(vars(cli), model=cfg, versions=versions,
        executable=sys.executable, source_hashes=source_hashes,
        evaluation='raw final predictions, fixed K=7, global Hungarian, no label selection'), indent=2))
    torch.set_num_threads(cli.threads)
    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)
    random.seed(cli.seed)
    torch.use_deterministic_algorithms(True)
    start = time.perf_counter()
    graph = Planetoid(str(ROOT / 'amgc_data'), 'Cora')[0]
    features, labels, norm_adj, _ = load_cora()
    assert np.array_equal(features, graph.x.numpy()), 'Cora feature order mismatch'
    assert np.array_equal(labels, graph.y.numpy()), 'Cora label order mismatch'
    expected = norm_adj.copy()
    expected.setdiag(0)
    expected.eliminate_zeros()
    expected.data[:] = 1
    from scipy.sparse import coo_matrix
    edges = graph.edge_index.numpy()
    actual = coo_matrix((np.ones(edges.shape[1]), edges), shape=expected.shape).tocsr()
    assert (expected != actual).nnz == 0, 'Cora edge mismatch'
    ordered = shuffle(np.arange(len(labels)), random_state=cli.split_seed)
    observed = np.zeros(len(labels), dtype=bool)
    observed[ordered[:int(cli.observed_ratio * len(labels))]] = True
    missing_ids = torch.tensor(ordered[int(cli.observed_ratio * len(labels)):])
    observed_ids = torch.tensor(ordered[:int(cli.observed_ratio * len(labels))])
    cold = ~observed & (norm_adj[:, observed].sum(1).A1 < 1e-8)
    graph.x[missing_ids] = 0
    np.savez_compressed(out / 'mask.npz', observed=observed, cold=cold)
    (out / 'data_check.json').write_text(json.dumps(dict(features_equal=True, labels_equal=True,
        edges_equal=True, nodes=graph.num_nodes, edges=graph.num_edges,
        observed_count=int(observed.sum()), missing_count=int((~observed).sum()),
        cold_count=int(cold.sum()), mask_sha256=hashlib.sha256(observed.tobytes()).hexdigest()), indent=2))
    model = EM(args).cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    mask = MaskEdge(p=cli.edge_mask_p)
    history = []
    metric_history = []

    def cluster():
        model.eval()
        with torch.no_grad():
            feat = torch.nn.functional.normalize(model.embed(graph), dim=1)
        assert torch.isfinite(feat).all()
        pred = KMeans(n_clusters=7, n_init=20, random_state=cli.seed).fit_predict(feat.numpy())
        return pred, compute_centers(args, feat, pred), feat

    pseudo, centers, feat = cluster()
    initial = evaluate(labels, pseudo, observed, cold)
    metric_history.append(dict(epoch=0, prediction_change=None, **initial))
    print('initial', initial, flush=True)
    previous = pseudo.copy()
    np.savez_compressed(out / 'initial_predictions.npz', pred=pseudo)
    try:
        for epoch in range(cli.epochs):
            tick = time.perf_counter()
            if (epoch + 1) % args.cluster_updating == 0:
                pseudo, centers, _ = cluster()
            lr = args.lr * .5 * (1 + math.cos(math.pi * epoch / cli.schedule_epochs))
            for group in optimizer.param_groups:
                group['lr'] = lr
            model.train()
            optimizer.zero_grad()
            # Enable before forward so a failure includes its originating operation.
            with torch.autograd.set_detect_anomaly(True):
                z, pos, neg = model(graph, mask, pseudo, centers, observed_ids, missing_ids)
                current_centers = compute_centers(args, z, pseudo)
                lc = compute_cluster_loss_no_contrastive(centers, current_centers, pseudo,
                                                        args.temperature, args.num_class)
                le = ce_loss(pos, neg)
                loss = lc + le
                assert torch.isfinite(loss).all(), 'Nonfinite loss'
                loss.backward()
            grad_norm = sum(float(p.grad.square().sum()) for p in model.parameters() if p.grad is not None) ** .5
            assert math.isfinite(grad_norm) and grad_norm > 0
            optimizer.step()
            row = dict(epoch=epoch + 1, loss=loss.item(), cluster_loss=lc.item(), edge_loss=le.item(),
                       lr=lr, grad_norm=grad_norm, train_sec=time.perf_counter()-tick,
                       rss_mb=psutil.Process().memory_info().rss/2**20)
            history.append(row)
            save_csv(out / 'loss.csv', history)
            if (epoch + 1) % cli.eval_every == 0 or epoch + 1 == cli.epochs:
                pred, _, feat = cluster()
                scores = evaluate(labels, pred, observed, cold)
                # Permutation-invariant change: align current IDs to previous IDs.
                from scipy.optimize import linear_sum_assignment
                cm = np.zeros((7, 7), dtype=np.int64)
                np.add.at(cm, (previous, pred), 1)
                rr, cc = linear_sum_assignment(-cm)
                mapping = np.zeros(7, dtype=int)
                mapping[cc] = rr
                change = float((mapping[pred] != previous).mean())
                previous = pred.copy()
                metric_history.append(dict(epoch=epoch+1, prediction_change=change, **scores))
                save_csv(out / 'metrics.csv', metric_history)
                print(f'epoch {epoch+1}: loss={loss.item():.6f} ACC={scores["acc"]:.4f} '
                      f'NMI={scores["nmi"]:.4f} clusters={scores["occupied_clusters"]} '
                      f'train_sec={row["train_sec"]:.2f}', flush=True)
        pred, _, feat = cluster()
        final = evaluate(labels, pred, observed, cold)
        final.update(elapsed_sec=time.perf_counter()-start, initial=initial,
                     mean_train_sec=float(np.mean([r['train_sec'] for r in history])),
                     peak_rss_mb=psutil.Process().memory_info().peak_wset/2**20,
                     last_prediction_change=metric_history[-1]['prediction_change'])
        np.savez_compressed(out / 'predictions.npz', pred=pred, embedding=feat.numpy(),
                            observed=observed, cold=cold)
        torch.save(model.state_dict(), out / 'model.pt')
        (out / 'result.json').write_text(json.dumps(final, indent=2))
        print('FINAL', final, flush=True)
    except Exception:
        (out / 'failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
