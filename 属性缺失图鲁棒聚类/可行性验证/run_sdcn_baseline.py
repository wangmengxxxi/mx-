"""CPU calibration of an SDCN adaptation, not an official paper reproduction.

Labels are used only by diagnostic evaluation; the final epoch is reported.
Masks are independent of model seeds and shared by all imputers.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment

from run_minimal_diagnostic import load_cora, neighbor_mean
from run_sdcn_smoke import GNNLayer, fill_features, sparse_tensor, target_distribution

ROOT = Path(__file__).resolve().parent


class AE(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.enc = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims, dims[1:]))
        reverse = list(reversed(dims))
        self.dec = nn.ModuleList(nn.Linear(a, b) for a, b in zip(reverse, reverse[1:]))

    def forward(self, x):
        states = []
        for i, layer in enumerate(self.enc):
            x = layer(x)
            if i < len(self.enc) - 1:
                x = F.relu(x)
            states.append(x)
        out = x
        for i, layer in enumerate(self.dec):
            out = layer(out)
            if i < len(self.dec) - 1:
                out = F.relu(out)
        return out, states


class SDCN(nn.Module):
    def __init__(self, dims, clusters):
        super().__init__()
        self.ae = AE(dims)
        g_dims = dims + [clusters]
        self.gnn = nn.ModuleList(GNNLayer(a, b) for a, b in zip(g_dims, g_dims[1:]))
        self.centers = nn.Parameter(torch.empty(clusters, dims[-1]))
        nn.init.xavier_normal_(self.centers)

    def forward(self, x, adj):
        reconstruction, states = self.ae(x)
        h = self.gnn[0](x, adj)
        for i in range(1, len(self.gnn)):
            h = self.gnn[i](0.5 * h + 0.5 * states[i - 1], adj,
                            active=i < len(self.gnn) - 1)
        pred = F.softmax(h, dim=1)
        z = states[-1]
        q = 1 / (1 + ((z[:, None] - self.centers) ** 2).sum(2))
        q = q / q.sum(1, keepdim=True)
        return reconstruction, q, pred, z


def evaluate(y, pred, observed, cold):
    k = int(y.max()) + 1
    counts = np.bincount(pred, minlength=k)
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (y, pred), 1)
    rows, cols = linear_sum_assignment(-cm)
    mapping = np.empty(k, dtype=np.int64)
    mapping[cols] = rows
    correct = mapping[pred] == y
    # One global label permutation is used for all subgroup accuracies.
    return dict(acc=float(correct.mean()),
                nmi=float(normalized_mutual_info_score(y, pred)),
                ari=float(adjusted_rand_score(y, pred)),
                missing_acc=float(correct[~observed].mean()) if (~observed).any() else None,
                observed_acc=float(correct[observed].mean()),
                cold_acc=float(correct[cold].mean()) if cold.any() else None,
                occupied_clusters=int((counts > 0).sum()),
                largest_cluster_fraction=float(counts.max() / len(y)))


def save_csv(path, rows):
    if rows:
        with path.open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def train(x_np, labels, adj, observed, cold, seed, args, folder):
    t0 = time.perf_counter()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = SDCN([x_np.shape[1]] + args.hidden_dims + [args.latent], args.clusters)
    x = torch.from_numpy(x_np).float()
    history = []
    optimizer = torch.optim.Adam(model.ae.parameters(), lr=args.pretrain_lr)
    for epoch in range(1, args.pretrain_epochs + 1):
        for indices in torch.randperm(len(x)).split(args.batch_size):
            reconstruction, _ = model.ae(x[indices])
            loss = F.mse_loss(reconstruction, x[indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            reconstruction, _ = model.ae(x)
            mse = float(F.mse_loss(reconstruction, x))
        history.append(dict(phase='pretrain', epoch=epoch, mse=mse,
                            kl_q=None, kl_pred=None, total=mse))
        if epoch % 20 == 0:
            print(f'{folder.name} pretrain {epoch}: mse={mse:.6f}', flush=True)

    with torch.no_grad():
        _, states = model.ae(x)
    km = KMeans(n_clusters=args.clusters, n_init=20, random_state=seed).fit(states[-1].numpy())
    model.centers.data.copy_(torch.from_numpy(km.cluster_centers_))
    ae_metrics = evaluate(labels, km.labels_, observed, cold)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    metrics_history = []
    previous = None
    for epoch in range(args.train_epochs + 1):
        if epoch % args.eval_every == 0 or epoch == args.train_epochs:
            with torch.no_grad():
                _, q, pred, _ = model(x, adj)
            hard = pred.argmax(1).numpy()
            item = dict(epoch=epoch, **evaluate(labels, hard, observed, cold))
            item['prediction_change'] = float(np.mean(hard != previous)) if previous is not None else None
            previous = hard.copy()
            metrics_history.append(item)
            print(f'{folder.name} joint {epoch}: ACC={item["acc"]:.4f} '
                  f'NMI={item["nmi"]:.4f} clusters={item["occupied_clusters"]}', flush=True)
        if epoch == args.train_epochs:
            break
        reconstruction, q, pred, _ = model(x, adj)
        p = target_distribution(q.detach())
        mse = F.mse_loss(reconstruction, x)
        kl_q = F.kl_div(q.clamp_min(1e-8).log(), p, reduction='batchmean')
        kl_pred = F.kl_div(pred.clamp_min(1e-8).log(), p, reduction='batchmean')
        loss = mse + 0.1 * kl_q + 0.01 * kl_pred
        if not torch.isfinite(loss):
            raise RuntimeError(f'Non-finite loss at epoch {epoch}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(dict(phase='joint', epoch=epoch + 1, mse=float(mse.detach()),
                            kl_q=float(kl_q.detach()), kl_pred=float(kl_pred.detach()),
                            total=float(loss.detach())))
    with torch.no_grad():
        _, q, pred, z = model(x, adj)
    q_hard, pred_hard = q.argmax(1).numpy(), pred.argmax(1).numpy()
    final = evaluate(labels, pred_hard, observed, cold)
    final.update({f'q_{key}': value for key, value in evaluate(labels, q_hard, observed, cold).items()})
    final.update({f'ae_{key}': value for key, value in ae_metrics.items()})
    final.update(elapsed_sec=time.perf_counter() - t0,
                 pretrain_mse_first=history[0]['mse'],
                 pretrain_mse_last=history[args.pretrain_epochs - 1]['mse'],
                 final_mse=history[-1]['mse'],
                 last_prediction_change=metrics_history[-1]['prediction_change'])
    save_csv(folder / 'loss.csv', history)
    save_csv(folder / 'metrics.csv', metrics_history)
    np.savez_compressed(folder / 'predictions.npz', pred=pred_hard, q=q_hard,
                        ae=km.labels_, observed=observed, cold=cold)
    torch.save(model.state_dict(), folder / 'model.pt')
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rates', default='0,0.4,0.8')
    ap.add_argument('--methods', default='zero,fp')
    ap.add_argument('--seeds', default='0')
    ap.add_argument('--masks', type=int, default=1)
    ap.add_argument('--mask-seed', type=int, default=20261022)
    ap.add_argument('--hidden', default='128,64', help='500,500,2000 for original SDCN widths')
    ap.add_argument('--latent', type=int, default=10)
    ap.add_argument('--clusters', type=int, default=7, help='Known K, not learned from labels')
    ap.add_argument('--pretrain-epochs', type=int, default=50)
    ap.add_argument('--train-epochs', type=int, default=200)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--pretrain-lr', type=float, default=1e-3)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--steps', type=int, default=40)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--eval-every', type=int, default=20)
    ap.add_argument('--output', required=True, help='New output folder; existing folder is never overwritten')
    args = ap.parse_args()
    args.hidden_dims = [int(v) for v in args.hidden.split(',')]
    rates = [float(v) for v in args.rates.split(',')]
    methods = args.methods.split(',')
    seeds = [int(v) for v in args.seeds.split(',')]
    if any(not 0 <= r < 1 for r in rates) or args.masks < 1:
        raise ValueError('Use rates in [0,1) and at least one mask')
    if min(args.pretrain_epochs, args.train_epochs, args.batch_size, args.eval_every) < 1:
        raise ValueError('Epochs, batch size and eval interval must be positive')
    if not set(methods) <= {'zero', 'mean', 'neighbor_mean', 'fp'}:
        raise ValueError('Complete is handled once at rate=0; select supported imputers')
    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    (out / 'config.json').write_text(json.dumps(dict(vars(args), torch=torch.__version__,
            numpy=np.__version__, protocol='SDCN adaptation; raw features; final epoch; no label selection'),
            indent=2), encoding='utf-8')
    features, labels, norm_adj, _ = load_cora()
    adj = sparse_tensor(norm_adj)
    rows = []
    for rate in rates:
        for mask in range(1 if rate == 0 else args.masks):
            rng = np.random.default_rng(np.random.SeedSequence([args.mask_seed, round(rate * 10000), mask]))
            observed = np.ones(len(labels), dtype=bool)
            observed[rng.choice(len(labels), round(rate * len(labels)), replace=False)] = False
            cold = ~observed & (norm_adj[:, observed].sum(1).A1 < 1e-8)
            digest = hashlib.sha256(observed.tobytes()).hexdigest()
            for method in (['complete'] if rate == 0 else methods):
                t_fill = time.perf_counter()
                filled = (neighbor_mean(features, observed, norm_adj) if method == 'neighbor_mean'
                          else fill_features(features, observed, method, norm_adj, args.steps))
                fill_time = time.perf_counter() - t_fill
                for seed in seeds:
                    folder = out / f'r{rate:g}_m{mask}_{method}_s{seed}'
                    folder.mkdir()
                    result = train(filled, labels, adj, observed, cold, seed, args, folder)
                    result.update(rate=rate, mask=mask, method=method, seed=seed,
                                  missing_count=int((~observed).sum()), cold_count=int(cold.sum()),
                                  mask_sha256=digest, fill_sec=fill_time)
                    (folder / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
                    rows.append(result)
                    save_csv(out / 'results.csv', rows)
    (out / 'results.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print('saved', str(out), flush=True)


if __name__ == '__main__':
    main()
