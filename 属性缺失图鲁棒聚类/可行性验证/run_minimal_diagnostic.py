"""Minimal feasibility diagnostic for node-level missing attributes on Cora.

The experiment uses the fixed Planetoid Cora graph and evaluates four imputers
with one shared graph-aware KMeans representation. It is intentionally not a
claim of a final clustering method; its purpose is to decide whether the
missing-attribute failure mode is measurable and worth deeper implementation.
"""
from __future__ import annotations
import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "planetoid"

def read_pickle(name):
    with (DATA / name).open("rb") as f:
        return pickle.load(f, encoding="latin1")

def load_cora():
    # This follows the original Planetoid indexing convention.
    x = read_pickle("ind.cora.x")
    tx = read_pickle("ind.cora.tx")
    allx = read_pickle("ind.cora.allx")
    y = read_pickle("ind.cora.y")
    ty = read_pickle("ind.cora.ty")
    ally_path = DATA / "ind.cora.ally"
    if ally_path.exists():
        ally = read_pickle("ind.cora.ally")
    else:
        # The Planetoid labels are one-hot; reconstructing ally from the
        # downloaded y/ty files is impossible, so fail with a clear message.
        raise FileNotFoundError("ind.cora.ally is required; retry the data download")
    graph = read_pickle("ind.cora.graph")
    test_idx = np.loadtxt(DATA / "ind.cora.test.index", dtype=int)
    features = sp.vstack((allx, tx)).tolil()
    labels = np.vstack((ally, ty))
    order = np.argsort(test_idx)
    features[test_idx[order]] = features[1708 + order]
    labels[test_idx[order]] = labels[1708 + order]
    features = np.asarray(features.tocsr().todense(), dtype=np.float32)
    labels = labels.argmax(axis=1).astype(np.int64)
    n = features.shape[0]
    rows, cols = [], []
    for i, nbrs in graph.items():
        for j in nbrs:
            rows.append(i); cols.append(j)
    adj = sp.coo_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, n))
    adj = adj.maximum(adj.T) + sp.eye(n, dtype=np.float32, format="csr")
    degree = np.asarray(adj.sum(axis=1)).ravel()
    inv_sqrt = np.zeros_like(degree)
    inv_sqrt[degree > 0] = 1.0 / np.sqrt(degree[degree > 0])
    norm_adj = sp.diags(inv_sqrt) @ adj @ sp.diags(inv_sqrt)
    return features, labels, norm_adj.tocsr(), degree - 1.0

def fp(features, observed, norm_adj, steps):
    out = np.zeros_like(features, dtype=np.float32)
    out[observed] = features[observed]
    for _ in range(steps):
        out = norm_adj @ out
        out[observed] = features[observed]
    return np.asarray(out, dtype=np.float32)

def neighbor_mean(features, observed, norm_adj):
    out = fp(features, observed, norm_adj, 1)
    # A node with no observed signal should use the observed global mean.
    support = norm_adj[:, observed].sum(axis=1).A1
    mean = features[observed].mean(axis=0)
    out[~observed & (support < 1e-8)] = mean
    return out

def recovery_accuracy(y_true, y_pred):
    # ACC after optimal cluster-label permutation.
    if len(y_true) == 0:
        return float("nan")
    cm = np.zeros((y_true.max() + 1, y_pred.max() + 1), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    r, c = linear_sum_assignment(-cm)
    return float(cm[r, c].sum() / len(y_true))

def scores(features, labels, norm_adj, method, observed, steps, seeds, projection):
    if method == "complete":
        filled = features.copy()
    elif method == "zero":
        filled = np.zeros_like(features)
        filled[observed] = features[observed]
    elif method == "mean":
        filled = np.full_like(features, features[observed].mean(axis=0))
        filled[observed] = features[observed]
    elif method == "neighbor_mean":
        filled = neighbor_mean(features, observed, norm_adj)
    elif method == "fp":
        filled = fp(features, observed, norm_adj, steps)
    else:
        raise ValueError(method)
    hidden = ~observed
    rmse = float(np.sqrt(np.mean((filled[hidden] - features[hidden]) ** 2))) if hidden.any() else 0.0
    denom = np.linalg.norm(filled[hidden], axis=1) * np.linalg.norm(features[hidden], axis=1)
    cosine = np.divide((filled[hidden] * features[hidden]).sum(axis=1), denom,
                       out=np.zeros_like(denom), where=denom > 0)
    # Use a graph-aware, fixed representation for every imputer.
    # A fixed projection keeps the CPU diagnostic quick and identical across
    # methods; it is not intended to replace the later SDCN representation.
    graph_features = np.hstack((filled, np.asarray(norm_adj @ filled, dtype=np.float32))) @ projection
    results = []
    for seed in seeds:
        pred = KMeans(n_clusters=int(labels.max() + 1), n_init=3, max_iter=100,
                      random_state=seed).fit_predict(graph_features)
        results.append({
            "acc": recovery_accuracy(labels, pred),
            "nmi": float(normalized_mutual_info_score(labels, pred)),
            "ari": float(adjusted_rand_score(labels, pred)),
            "missing_acc": recovery_accuracy(labels[hidden], pred[hidden]) if hidden.any() else None,
            "observed_acc": recovery_accuracy(labels[observed], pred[observed]),
        })
    avg = {}
    for k in results[0]:
        values = [r[k] for r in results if r[k] is not None]
        avg[k] = float(np.mean(values)) if values else None
    avg.update({"rmse": rmse, "cosine": float(cosine.mean()) if hidden.any() else 1.0,
                "missing_count": int(hidden.sum())})
    return avg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", default="0,0.2,0.4,0.6,0.8")
    ap.add_argument("--masks", type=int, default=3)
    ap.add_argument("--steps", type=int, default=5)
    args = ap.parse_args()
    t0 = time.time()
    features, labels, norm_adj, degree = load_cora()
    rng = np.random.default_rng(20261022)
    rates = [float(x) for x in args.rates.split(",")]
    methods = ["complete", "zero", "mean", "neighbor_mean", "fp"]
    rng_projection = np.random.default_rng(20261023)
    projection = rng_projection.normal(0, 1.0 / np.sqrt(64),
                                       size=(features.shape[1] * 2, 64)).astype(np.float32)
    records = []
    for rate in rates:
        for mask_id in range(args.masks):
            if rate == 0:
                observed = np.ones(len(labels), dtype=bool)
            else:
                hidden_idx = rng.choice(len(labels), size=int(round(rate * len(labels))), replace=False)
                observed = np.ones(len(labels), dtype=bool)
                observed[hidden_idx] = False
            for method in methods:
                row = scores(features, labels, norm_adj, method, observed, args.steps, [0, 1, 2], projection)
                row.update({"rate": rate, "mask": mask_id, "method": method,
                            "cold_start": int((~observed & (norm_adj[:, observed].sum(axis=1).A1 < 1e-8)).sum()),
                            "mean_degree_missing": float(degree[~observed].mean()) if (~observed).any() else 0.0})
                records.append(row)
                print(json.dumps(row), flush=True)
    out = ROOT / "results_minimal.json"
    out.write_text(json.dumps({"records": records, "elapsed_sec": time.time() - t0}, indent=2), encoding="utf-8")
    import csv
    with (ROOT / "results_minimal.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader(); writer.writerows(records)
    print("saved", out)

if __name__ == "__main__":
    main()
