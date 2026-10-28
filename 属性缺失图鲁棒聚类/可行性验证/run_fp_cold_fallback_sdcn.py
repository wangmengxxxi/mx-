"""Paired SDCN check of the calibrated mean fallback against archived FP runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from run_fp_cold_fallback_pair import choose_alpha, cold_mask, stats
from run_fp_coverage_pair import fills
from run_minimal_diagnostic import load_cora
from run_sdcn_baseline import save_csv, train
from run_sdcn_smoke import sparse_tensor


ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / "sdcn_stage4_holdout"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", default="0.4,0.8")
    ap.add_argument("--masks", type=int, default=3)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--output", default="fp_cold_fallback_sdcn")
    args = ap.parse_args()
    out = ROOT / args.output
    out.mkdir(exist_ok=False)
    reference_config = json.loads((REFERENCE / "config.json").read_text(encoding="utf-8"))
    config = argparse.Namespace(**reference_config)
    config.hidden_dims = reference_config["hidden_dims"]
    torch.set_num_threads(config.threads)
    features, labels, norm_adj, _ = load_cora()
    adj = sparse_tensor(norm_adj)
    rows = []
    for rate in [float(v) for v in args.rates.split(",")]:
        for mask_id in range(args.masks):
            rng = np.random.default_rng(np.random.SeedSequence(
                [config.mask_seed, round(rate * 10000), mask_id]))
            observed = np.ones(len(features), dtype=bool)
            observed[rng.choice(len(features), round(rate * len(features)), replace=False)] = False
            digest = hashlib.sha256(observed.tobytes()).hexdigest()
            cold = cold_mask(observed, norm_adj)
            held_out = rng.choice(np.flatnonzero(observed), round(0.2 * observed.sum()),
                                  replace=False)
            cal_observed = observed.copy()
            cal_observed[held_out] = False
            cal_selection = np.zeros(len(features), dtype=bool)
            cal_selection[held_out] = True
            cal_selection &= cold_mask(cal_observed, norm_adj)
            cal_fp, _, _ = fills(features, cal_observed, norm_adj, config.steps)
            cal_mean = np.broadcast_to(features[cal_observed].mean(0), features.shape)
            alpha = choose_alpha(features, cal_fp, cal_mean, cal_selection)
            fp, _, _ = fills(features, observed, norm_adj, config.steps)
            filled = fp.copy()
            filled[cold] = ((1 - alpha) * fp[cold]
                            + alpha * features[observed].mean(axis=0))
            imp_fp = stats(features, fp, ~observed)
            imp_new = stats(features, filled, ~observed)
            for seed in [int(v) for v in args.seeds.split(",")]:
                ref_path = REFERENCE / f"r{rate:g}_m{mask_id}_fp_s{seed}" / "result.json"
                ref = json.loads(ref_path.read_text(encoding="utf-8"))
                if ref["mask_sha256"] != digest:
                    raise ValueError(f"Mask mismatch: {ref_path}")
                folder = out / f"r{rate:g}_m{mask_id}_mean_fallback_s{seed}"
                folder.mkdir()
                result = train(filled, labels, adj, observed, cold, seed, config, folder)
                result.update(rate=rate, mask=mask_id, seed=seed, method="mean_fallback",
                              mask_sha256=digest, cold_count=int(cold.sum()),
                              calibration_cold_count=int(cal_selection.sum()), alpha=alpha,
                              imputation_fp_rmse=imp_fp["rmse"],
                              imputation_new_rmse=imp_new["rmse"],
                              imputation_fp_cosine=imp_fp["cosine"],
                              imputation_new_cosine=imp_new["cosine"],
                              reference_path=str(ref_path),
                              fp_acc=ref["acc"], fp_nmi=ref["nmi"], fp_ari=ref["ari"],
                              fp_cold_acc=ref["cold_acc"],
                              delta_acc=result["acc"] - ref["acc"],
                              delta_nmi=result["nmi"] - ref["nmi"],
                              delta_ari=result["ari"] - ref["ari"])
                (folder / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
                rows.append(result)
                (out / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
                save_csv(out / "results.csv", rows)
                print(f"pair rate={rate} mask={mask_id} seed={seed} alpha={alpha} "
                      f"delta_acc={result['delta_acc']:+.4f}", flush=True)
    (out / "config.json").write_text(json.dumps({"args": vars(args),
        "reference_config": reference_config, "calibration_fraction": 0.2,
        "candidate": "FP plus calibrated observed-mean fallback on zero-observed-neighbor nodes",
        "reference": str(REFERENCE)}, indent=2), encoding="utf-8")
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
