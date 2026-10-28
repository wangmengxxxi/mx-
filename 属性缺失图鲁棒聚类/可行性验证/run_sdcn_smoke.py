"""Small CPU SDCN-style smoke test on the downloaded Planetoid Cora graph.

This is an engineering check for the proposed direction. It keeps the SDCN
autoencoder/GCN/DEC interaction but uses smaller layers and short training so
it can run in the current CPU-only environment.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment

from run_minimal_diagnostic import fp, load_cora

ROOT = Path(__file__).resolve().parent


def acc_score(y_true, y_pred):
    cm = np.zeros((int(y_true.max()) + 1, int(y_pred.max()) + 1), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    r, c = linear_sum_assignment(-cm)
    return float(cm[r, c].sum() / len(y_true))


def sparse_tensor(matrix):
    coo = matrix.tocoo()
    indices = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
    values = torch.tensor(coo.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, size=coo.shape).coalesce()


class AutoEncoder(nn.Module):
    def __init__(self, n_input, hidden=(128, 64), n_z=32):
        super().__init__()
        h1, h2 = hidden
        self.enc1 = nn.Linear(n_input, h1)
        self.enc2 = nn.Linear(h1, h2)
        self.z = nn.Linear(h2, n_z)
        self.dec2 = nn.Linear(n_z, h2)
        self.dec1 = nn.Linear(h2, h1)
        self.out = nn.Linear(h1, n_input)

    def forward(self, x):
        h1 = F.relu(self.enc1(x))
        h2 = F.relu(self.enc2(h1))
        z = self.z(h2)
        d2 = F.relu(self.dec2(z))
        d1 = F.relu(self.dec1(d2))
        return self.out(d1), h1, h2, z


class GNNLayer(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_in, n_out))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj, active=True):
        h = torch.mm(x, self.weight)
        h = torch.sparse.mm(adj, h)
        return F.relu(h) if active else h


class SDCNSmoke(nn.Module):
    def __init__(self, n_input, n_clusters, n_z=32):
        super().__init__()
        self.ae = AutoEncoder(n_input, n_z=n_z)
        self.g1 = GNNLayer(n_input, 128)
        self.g2 = GNNLayer(128, 64)
        self.g3 = GNNLayer(64, n_z)
        self.g4 = GNNLayer(n_z, n_clusters)
        self.cluster_layer = nn.Parameter(torch.empty(n_clusters, n_z))
        nn.init.xavier_normal_(self.cluster_layer)
        self.v = 1.0

    def forward(self, x, adj):
        x_bar, h1, h2, z = self.ae(x)
        sigma = 0.5
        h = self.g1(x, adj)
        h = self.g2((1 - sigma) * h + sigma * h1, adj)
        h = self.g3((1 - sigma) * h + sigma * h2, adj)
        h = self.g4((1 - sigma) * h + sigma * z, adj, active=False)
        pred = F.softmax(h, dim=1)
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.cluster_layer) ** 2, dim=2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, dim=1)).t()
        return x_bar, q, pred, z


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def fill_features(features, observed, method, norm_adj, steps):
    if method == "complete":
        return features.copy()
    if method == "zero":
        out = np.zeros_like(features)
        out[observed] = features[observed]
        return out
    if method == "mean":
        out = np.full_like(features, features[observed].mean(axis=0))
        out[observed] = features[observed]
        return out
    if method == "fp":
        return fp(features, observed, norm_adj, steps)
    raise ValueError(method)


def run_one(x_np, labels, adj, seed, pretrain_epochs, train_epochs):
    torch.manual_seed(seed)
    x = torch.from_numpy(x_np).float()
    model = SDCNSmoke(x.shape[1], int(labels.max() + 1)).float()
    pre_opt = torch.optim.Adam(model.ae.parameters(), lr=1e-3)
    pre_losses = []
    for _ in range(pretrain_epochs):
        x_bar, _, _, _ = model.ae(x)
        loss = F.mse_loss(x_bar, x)
        pre_opt.zero_grad(); loss.backward(); pre_opt.step()
        pre_losses.append(float(loss.detach()))

    with torch.no_grad():
        _, _, _, z = model.ae(x)
    centers = KMeans(n_clusters=int(labels.max() + 1), n_init=10, random_state=seed).fit(
        z.numpy()).cluster_centers_
    model.cluster_layer.data.copy_(torch.from_numpy(centers).float())

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_losses = []
    for _ in range(train_epochs):
        x_bar, q, pred, _ = model(x, adj)
        p = target_distribution(q.detach())
        kl_q = F.kl_div(q.clamp_min(1e-8).log(), p, reduction="batchmean")
        kl_pred = F.kl_div(pred.clamp_min(1e-8).log(), p, reduction="batchmean")
        loss = F.mse_loss(x_bar, x) + 0.1 * kl_q + 0.01 * kl_pred
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        train_losses.append(float(loss.detach()))

    with torch.no_grad():
        _, q, pred, _ = model(x, adj)
    q_labels = q.argmax(1).numpy()
    pred_labels = pred.argmax(1).numpy()
    return {
        "acc_q": acc_score(labels, q_labels),
        "nmi_q": float(normalized_mutual_info_score(labels, q_labels)),
        "ari_q": float(adjusted_rand_score(labels, q_labels)),
        "acc_pred": acc_score(labels, pred_labels),
        "nmi_pred": float(normalized_mutual_info_score(labels, pred_labels)),
        "ari_pred": float(adjusted_rand_score(labels, pred_labels)),
        "pretrain_loss_first": pre_losses[0],
        "pretrain_loss_last": pre_losses[-1],
        "train_loss_first": train_losses[0],
        "train_loss_last": train_losses[-1],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", default="0,0.4,0.8")
    ap.add_argument("--methods", default="complete,zero,fp")
    ap.add_argument("--pretrain-epochs", type=int, default=8)
    ap.add_argument("--train-epochs", type=int, default=20)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--masks", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20261022)
    args = ap.parse_args()
    t0 = time.time()
    features, labels, norm_adj, _ = load_cora()
    torch.set_num_threads(4)
    adj = sparse_tensor(norm_adj)
    rng = np.random.default_rng(args.seed)
    rates = [float(v) for v in args.rates.split(",")]
    methods = [v for v in args.methods.split(",") if v]
    rows = []
    for rate in rates:
        for mask in range(args.masks):
            observed = np.ones(len(labels), dtype=bool)
            if rate > 0:
                hidden = rng.choice(len(labels), size=int(round(rate * len(labels))), replace=False)
                observed[hidden] = False
            for method in methods:
                filled = fill_features(features, observed, method, norm_adj, args.steps)
                row = run_one(filled, labels, adj, args.seed + mask, args.pretrain_epochs, args.train_epochs)
                row.update({"rate": rate, "mask": mask, "method": method,
                            "missing_count": int((~observed).sum())})
                rows.append(row)
                print(json.dumps(row), flush=True)
    out = ROOT / "results_sdcn_smoke.json"
    out.write_text(json.dumps({"records": rows, "elapsed_sec": time.time() - t0}, indent=2), encoding="utf-8")
    with (ROOT / "results_sdcn_smoke.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    print("saved", out)


if __name__ == "__main__":
    main()
