"""Observed-only calibration for a two-hop fallback on FP cold nodes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from run_fp_coverage_pair import fills, stats
from run_minimal_diagnostic import load_cora


ROOT = Path(__file__).resolve().parent
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def two_hop_average(features, observed, norm_adj):
    source = np.zeros_like(features, dtype=np.float32)
    source[observed] = features[observed]
    weights = observed.astype(np.float32)[:, None]
    for _ in range(2):
        source = np.asarray(norm_adj @ source, dtype=np.float32)
        weights = np.asarray(norm_adj @ weights, dtype=np.float32)
    return np.divide(source, weights, out=np.zeros_like(source), where=weights > 1e-8)


def cold_mask(observed, norm_adj):
    raw = norm_adj.copy()
    raw.data[:] = 1
    raw.setdiag(0)
    raw.eliminate_zeros()
    count = np.asarray(raw @ observed.astype(np.float32)).ravel()
    return ~observed & (count == 0)


def choose_alpha(features, fp, backup, selection):
    if selection.sum() < 20:
        return 0.0
    return min(ALPHAS, key=lambda a: np.mean(
        ((1 - a) * fp[selection] + a * backup[selection] - features[selection]) ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", default="0.4,0.8")
    ap.add_argument("--masks", type=int, default=5)
    ap.add_argument("--mask-seed", type=int, default=20261023)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--calibration-fraction", type=float, default=0.2)
    ap.add_argument("--output", default="fp_cold_fallback_pair")
    args = ap.parse_args()
    if not 0 < args.calibration_fraction < 0.5:
        raise ValueError("calibration-fraction must be between 0 and 0.5")
    out = ROOT / args.output
    out.mkdir(exist_ok=False)
    features, _, norm_adj, _ = load_cora()
    rows = []
    for rate in [float(v) for v in args.rates.split(",")]:
        for mask_id in range(args.masks):
            rng = np.random.default_rng(np.random.SeedSequence(
                [args.mask_seed, round(rate * 10000), mask_id]))
            observed = np.ones(len(features), dtype=bool)
            observed[rng.choice(len(features), round(rate * len(features)), replace=False)] = False
            cold = cold_mask(observed, norm_adj)
            held_out = rng.choice(np.flatnonzero(observed),
                                  round(args.calibration_fraction * observed.sum()), replace=False)
            calibration_observed = observed.copy()
            calibration_observed[held_out] = False
            calibration = np.zeros(len(features), dtype=bool)
            calibration[held_out] = True
            calibration_cold = calibration & cold_mask(calibration_observed, norm_adj)
            cal_fp, _, _ = fills(features, calibration_observed, norm_adj, args.steps)
            cal_two_hop = two_hop_average(features, calibration_observed, norm_adj)
            cal_mean = np.broadcast_to(features[calibration_observed].mean(0), features.shape)
            alpha_two_hop = choose_alpha(features, cal_fp, cal_two_hop, calibration_cold)
            alpha_mean = choose_alpha(features, cal_fp, cal_mean, calibration_cold)
            fp, _, _ = fills(features, observed, norm_adj, args.steps)
            two_hop = two_hop_average(features, observed, norm_adj)
            mean = np.broadcast_to(features[observed].mean(0), features.shape)
            methods = {"fp": fp.copy()}
            for name, backup, alpha in [
                ("two_hop_fallback", two_hop, alpha_two_hop),
                ("mean_fallback", mean, alpha_mean),
            ]:
                filled = fp.copy()
                filled[cold] = (1 - alpha) * fp[cold] + alpha * backup[cold]
                methods[name] = filled
            selections = {"all_missing": ~observed,
                          "zero_observed_neighbors": cold,
                          "other_missing": ~observed & ~cold}
            digest = hashlib.sha256(observed.tobytes()).hexdigest()
            for group, selection in selections.items():
                for method, filled in methods.items():
                    rows.append({"rate": rate, "mask": mask_id, "mask_sha256": digest,
                                 "group": group, "method": method,
                                 "alpha": {"fp": 0, "two_hop_fallback": alpha_two_hop,
                                           "mean_fallback": alpha_mean}[method],
                                 "calibration_cold_n": int(calibration_cold.sum()),
                                 **stats(features, filled, selection)})
            print(f"rate={rate} mask={mask_id} cold={cold.sum()} "
                  f"cal_cold={calibration_cold.sum()} "
                  f"alphas=({alpha_two_hop},{alpha_mean})", flush=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    (out / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (out / "results.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
