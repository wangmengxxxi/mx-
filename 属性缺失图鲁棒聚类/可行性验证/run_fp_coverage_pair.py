"""Paired FP coverage diagnostic on the existing Cora mask protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from run_minimal_diagnostic import load_cora


ROOT = Path(__file__).resolve().parent


def fills(features, observed, norm_adj, steps):
    value = np.zeros_like(features, dtype=np.float32)
    value[observed] = features[observed]
    mass = observed.astype(np.float32)[:, None]
    for _ in range(steps):
        value = np.asarray(norm_adj @ value, dtype=np.float32)
        value[observed] = features[observed]
        mass = np.asarray(norm_adj @ mass, dtype=np.float32)
        mass[observed] = 1.0
    normalized = np.divide(value, mass, out=np.zeros_like(value), where=mass > 1e-8)
    normalized[observed] = features[observed]
    return value, normalized, mass.ravel()


def stats(features, filled, selection):
    if not selection.any():
        return {"n": 0, "rmse": None, "mae": None, "cosine": None}
    actual = features[selection]
    predicted = filled[selection]
    denom = np.linalg.norm(actual, axis=1) * np.linalg.norm(predicted, axis=1)
    cosine = np.divide((actual * predicted).sum(1), denom,
                       out=np.zeros(len(actual)), where=denom > 0)
    return {"n": int(selection.sum()),
            "rmse": float(np.sqrt(np.mean((predicted - actual) ** 2))),
            "mae": float(np.mean(np.abs(predicted - actual))),
            "cosine": float(cosine.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", default="0.4,0.8")
    ap.add_argument("--masks", type=int, default=5)
    ap.add_argument("--mask-seed", type=int, default=20261023)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--output", default="fp_coverage_pair")
    args = ap.parse_args()
    out = ROOT / args.output
    out.mkdir(exist_ok=False)
    features, _, norm_adj, _ = load_cora()
    raw_adj = norm_adj.copy()
    raw_adj.data[:] = 1
    raw_adj.setdiag(0)
    raw_adj.eliminate_zeros()
    rows = []
    for rate in [float(v) for v in args.rates.split(",")]:
        for mask_id in range(args.masks):
            rng = np.random.default_rng(np.random.SeedSequence(
                [args.mask_seed, round(rate * 10000), mask_id]))
            observed = np.ones(len(features), dtype=bool)
            observed[rng.choice(len(features), round(rate * len(features)), replace=False)] = False
            hidden = ~observed
            known_count = np.asarray(raw_adj @ observed.astype(np.float32)).ravel()
            degree = np.asarray(raw_adj.sum(axis=1)).ravel()
            coverage = np.divide(known_count, degree, out=np.zeros_like(known_count),
                                 where=degree > 0)
            fp, mass_normalized, mass = fills(features, observed, norm_adj, args.steps)
            selections = {
                "all_missing": hidden,
                "zero_observed_neighbors": hidden & (known_count == 0),
                "one_observed_neighbor": hidden & (known_count == 1),
                "two_plus_observed_neighbors": hidden & (known_count >= 2),
                "low_coverage_nonzero": hidden & (known_count > 0) & (coverage <= 0.25),
                "high_coverage": hidden & (coverage > 0.5),
            }
            digest = hashlib.sha256(observed.tobytes()).hexdigest()
            for group, selected in selections.items():
                for method, filled in [("fp", fp), ("mass_normalized", mass_normalized)]:
                    rows.append({"rate": rate, "mask": mask_id, "mask_sha256": digest,
                                 "group": group, "method": method,
                                 "mean_mass": float(mass[selected].mean()) if selected.any() else None,
                                 **stats(features, filled, selected)})
            print(f"rate={rate} mask={mask_id} hidden={hidden.sum()} "
                  f"zero_neighbors={(hidden & (known_count == 0)).sum()}", flush=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    (out / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (out / "results.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
