"""Fixed S^2 oracle diagnostic on the archived Cora/SDCN protocol.

Oracle features are a diagnostic intervention, never a proposed imputer.
Existing scripts and reference outputs are read only. Every output is new.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
import sklearn
import torch
from scipy.optimize import linear_sum_assignment

from run_fp_cold_fallback_pair import cold_mask
from run_fp_coverage_pair import fills, stats
from run_minimal_diagnostic import load_cora
from run_sdcn_baseline import evaluate, save_csv, train
from run_sdcn_smoke import fill_features, sparse_tensor

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / "sdcn_stage4_holdout"
METHODS = {"A": "zero", "B": "fp", "C": "oracle_cold", "D": "oracle_random", "E": "complete"}
SOURCES = ["run_oracle_ceiling.py", "run_sdcn_baseline.py", "run_sdcn_smoke.py",
           "run_minimal_diagnostic.py", "run_fp_cold_fallback_pair.py", "run_fp_coverage_pair.py"]


def write_json(path, value):
    # Replace our own aggregate atomically so progress readers never see partial JSON.
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def sha_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha_array(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def make_mask(n, rate, mask_id, mask_seed):
    rng = np.random.default_rng(np.random.SeedSequence([mask_seed, round(rate * 10000), mask_id]))
    observed = np.ones(n, dtype=bool)
    observed[rng.choice(n, round(rate * n), replace=False)] = False
    return observed


def globally_correct(labels, prediction):
    # Exactly the global Hungarian mapping in the reused evaluate().
    k = int(labels.max()) + 1
    counts = np.zeros((k, k), dtype=np.int64)
    np.add.at(counts, (labels, prediction), 1)
    rows, cols = linear_sum_assignment(-counts)
    mapping = np.empty(k, dtype=np.int64)
    mapping[cols] = rows
    return mapping[prediction] == labels


def subgroup_metrics(labels, prediction, observed, cold):
    correct = globally_correct(labels, prediction)
    noncold_missing = ~observed & ~cold
    noncold_all = ~cold
    return {
        "noncold_missing_acc": float(correct[noncold_missing].mean()) if noncold_missing.any() else None,
        "noncold_all_acc": float(correct[noncold_all].mean()) if noncold_all.any() else None,
    }


def smooth(filled, norm_adj, k):
    value = filled.copy()
    for _ in range(k):
        value = np.asarray(norm_adj @ value, dtype=np.float32)
    return value


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rates", default="0.8,0.4")
    ap.add_argument("--masks", type=int, default=3)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--output", default="oracle_ceiling")
    ap.add_argument("--smoke", action="store_true", help="1 mask, 1 seed, AE 2 / joint 3 epochs; no reference rerun")
    args = ap.parse_args()
    rates = [float(v) for v in args.rates.split(",")]
    seeds = [int(v) for v in args.seeds.split(",")]
    if not set(rates) <= {0.4, 0.8} or len(set(rates)) != len(rates):
        raise ValueError("Only the preregistered rates 0.4/0.8 are supported, without duplicates")
    if args.masks not in (1, 2, 3) or not set(seeds) <= {0, 1} or len(set(seeds)) != len(seeds):
        raise ValueError("Only the first three masks and model seeds 0,1 are supported")
    if args.smoke:
        rates, seeds, args.masks = [0.8], [0], 1
    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT / out
    out = out.resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite output: {out}")
    reference_config = json.loads((REFERENCE / "config.json").read_text(encoding="utf-8"))
    if (reference_config["numpy"], reference_config["torch"]) != (np.__version__, torch.__version__):
        raise RuntimeError("NumPy/PyTorch versions differ from the archived reference")
    config = argparse.Namespace(**reference_config)
    if args.smoke:
        config.pretrain_epochs, config.train_epochs, config.eval_every = 2, 3, 1
    torch.set_num_threads(config.threads)
    features, labels, norm_adj, _ = load_cora()
    adj = sparse_tensor(norm_adj)
    source_hashes = {name: sha_file(ROOT / name) for name in SOURCES}
    archive_manifest = json.loads((ROOT / "stage4_manifest.json").read_text(encoding="utf-8"))
    archive_hashes = {r["path"]: r["sha256"] for r in archive_manifest}
    archived_source_checks = {name: {"current": value, "archived": archive_hashes.get(name),
        "equal": value == archive_hashes[name]} for name, value in source_hashes.items() if name in archive_hashes}
    if not all(row["equal"] for row in archived_source_checks.values()):
        raise RuntimeError(f"Reused source changed since stage 4: {archived_source_checks}")
    reference_file_hashes = {str(p.relative_to(ROOT)): sha_file(p) for p in REFERENCE.rglob("*.json")}
    mask_info, mask_checks = [], []
    # Check every planned pair BEFORE any fit. D's random subset is independent
    # of model seed, labels, and training; the same subset is used for seeds 0,1.
    for rate in rates:
        for mask_id in range(args.masks):
            observed = make_mask(len(features), rate, mask_id, config.mask_seed)
            cold = ~observed & (norm_adj[:, observed].sum(1).A1 < 1e-8)
            if not np.array_equal(cold, cold_mask(observed, norm_adj)):
                raise AssertionError("Cold-node definitions differ")
            digest = sha_array(observed)
            candidates = np.flatnonzero(~observed & ~cold)
            entropy = [config.mask_seed, round(rate * 10000), mask_id, 0x4F524143]
            random_rng = np.random.default_rng(np.random.SeedSequence(entropy))
            selected = np.sort(random_rng.choice(candidates, min(int(cold.sum()), len(candidates)), replace=False))
            for seed in seeds:
                for method in ("zero", "fp"):
                    path = REFERENCE / f"r{rate:g}_m{mask_id}_{method}_s{seed}" / "result.json"
                    reference = json.loads(path.read_text(encoding="utf-8"))
                    check = dict(rate=rate, mask=mask_id, seed=seed, method=method,
                                 reference_path=str(path), mask_sha256=digest,
                                 reference_mask_sha256=reference["mask_sha256"],
                                 equal=digest == reference["mask_sha256"],
                                 cold_count=int(cold.sum()), reference_cold_count=reference["cold_count"])
                    if not check["equal"] or check["cold_count"] != check["reference_cold_count"]:
                        raise AssertionError(f"Reference mask/cold mismatch: {check}")
                    mask_checks.append(check)
            mask_info.append(dict(rate=rate, mask=mask_id, mask_sha256=digest,
                missing_count=int((~observed).sum()), cold_count=int(cold.sum()),
                noncold_missing_count=len(candidates), random_seed_entropy=entropy,
                random_selected_nodes=selected.tolist(),
                random_count_shortfall=len(candidates) < int(cold.sum()),
                cold_selected_nodes=np.flatnonzero(cold).tolist()))

    out.mkdir(parents=True, exist_ok=False)
    protocol = dict(
        created_utc=datetime.now(timezone.utc).isoformat(), smoke=args.smoke,
        cli=vars(args), rates=rates, seeds=seeds, masks=args.masks,
        smoothing_k=2, smoothing="X_in = S @ (S @ X_fill); S=D^-1/2(A+I)D^-1/2 from load_cora; no clamping after smoothing",
        branches=METHODS, reference_config=reference_config, training_config=vars(config),
        evaluation="Fixed final epoch; labels only for offline evaluation; one global Hungarian mapping per fitted model",
        random_oracle="Uniform noncold missing subset; seed entropy [mask_seed, round(rate*10000), mask, 0x4F524143]; shared across model seeds",
        imputation_metrics="RMSE and mean per-node cosine of X_fill vs raw truth on missing nodes BEFORE smoothing; zero-norm cosine=0",
        complete_branch="Train once per model seed with no mask; pair to each masked run with the same seed; subgroup evaluation uses its mask",
        preregistered_rules={"primary_rate": 0.8, "primary_metric": "acc", "pass_cold_acc_min": 0.03,
            "pass_positive_pairs_min": "5/6", "failure_total_acc_less_than": 0.02,
            "pass": "C-B mean ACC >= 0.03, at least 5/6 positive, clearly exceeds D-B",
            "partial": "E-B sufficiently large and C-B approximately D-B",
            "failure": "E-B < 0.02 OR B approximately A",
            "qualitative_boundaries": "No numeric thresholds supplied for clearly exceeds/approximately; report paired values, signs and uncertainty without adding thresholds after results",
            "priority": "Check explicit failure criterion first; do not label an uncovered outcome pass/partial by changing thresholds"},
        optional_k0="Only the mandatory one-pair B rerun; old E(k=0), seeds 0,1, reported as archived context",
        environment=dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
            numpy=np.__version__, torch=torch.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__,
            threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads()),
        source_sha256=source_hashes, archived_source_checks=archived_source_checks,
        reference_json_sha256=reference_file_hashes,
        expected_main_runs=len(rates) * args.masks * len(seeds) * 4 + len(seeds))
    write_json(out / "config.json", protocol)
    write_json(out / "mask_checks.json", mask_checks)
    write_json(out / "mask_metadata.json", mask_info)
    started = time.perf_counter()
    rows = []

    def status(phase, **extra):
        write_json(out / "status.json", dict(phase=phase, completed_main_runs=len(rows),
            expected_main_runs=protocol["expected_main_runs"], elapsed_sec=time.perf_counter() - started,
            updated_utc=datetime.now(timezone.utc).isoformat(), **extra))

    def fit_one(filled, observed, cold, seed, rate, mask_id, branch, meta, k=2, parent=None):
        folder = (parent or out) / f"r{rate:g}_m{mask_id}_{METHODS[branch]}_k{k}_s{seed}"
        folder.mkdir(parents=True, exist_ok=False)
        x_in = smooth(filled, norm_adj, k)
        if not np.isfinite(x_in).all():
            raise AssertionError("Nonfinite model input")
        if branch in ("A", "B", "C", "D") and not np.array_equal(filled[observed], features[observed]):
            raise AssertionError("Observed features changed before smoothing")
        replaced = (meta["cold_selected_nodes"] if branch == "C" else
                    meta["random_selected_nodes"] if branch == "D" else [])
        extra = dict(rate=rate, mask=mask_id, seed=seed, branch=branch,
            method=METHODS[branch], smoothing_k=k, mask_sha256=sha_array(observed),
            missing_count=int((~observed).sum()), cold_count=int(cold.sum()),
            noncold_missing_count=int((~observed & ~cold).sum()),
            replaced_count=len(replaced), random_count_shortfall=bool(meta.get("random_count_shortfall", False)) if branch == "D" else False,
            imputation_rmse=stats(features, filled, ~observed)["rmse"],
            imputation_cosine=stats(features, filled, ~observed)["cosine"],
            fill_sha256=sha_array(filled), input_sha256=sha_array(x_in),
            output_path=str(folder), smoke=args.smoke)
        run_config = dict(**extra, training_config=vars(config), source_sha256=source_hashes,
            selected_nodes=replaced, mask_metadata=meta,
            oracle_diagnostic_only=branch in ("C", "D", "E"),
            protocol_path=str(out / "config.json"))
        write_json(folder / "config.json", run_config)
        np.savez_compressed(folder / "mask.npz", observed=observed, cold=cold,
                            replaced_nodes=np.asarray(replaced, dtype=np.int64))
        print(f"START {folder.name}", flush=True)
        result = train(x_in, labels, adj, observed, cold, seed, config, folder)
        prediction = np.load(folder / "predictions.npz")["pred"]
        result.update(subgroup_metrics(labels, prediction, observed, cold))
        result.update(extra)
        write_json(folder / "result.json", result)
        print(f"DONE {folder.name} ACC={result['acc']:.6f} elapsed={result['elapsed_sec']:.1f}s", flush=True)
        return result

    status("validated")
    # A genuinely new k=0 B run uses exactly the archived training function.
    # This runs first so drift is detected before the 50 main fits.
    if not args.smoke:
        rate, mask_id, seed = 0.4, 0, 0
        observed = make_mask(len(features), rate, mask_id, config.mask_seed)
        cold = ~observed & (norm_adj[:, observed].sum(1).A1 < 1e-8)
        filled = fill_features(features, observed, "fp", norm_adj, config.steps)
        ref_path = REFERENCE / "r0.4_m0_fp_s0" / "result.json"
        ref = json.loads(ref_path.read_text(encoding="utf-8"))
        if sha_array(observed) != ref["mask_sha256"]:
            raise AssertionError("k=0 reference mask mismatch")
        status("reference_rerun")
        rerun = fit_one(filled, observed, cold, seed, rate, mask_id, "B", {}, k=0, parent=out / "refcheck")
        comparison = dict(reference_path=str(ref_path), rerun_path=rerun["output_path"],
            mask_equal=rerun["mask_sha256"] == ref["mask_sha256"],
            reference={key: ref[key] for key in ("acc", "nmi", "ari", "cold_acc")},
            rerun={key: rerun[key] for key in ("acc", "nmi", "ari", "cold_acc")},
            delta={key: rerun[key] - ref[key] for key in ("acc", "nmi", "ari", "cold_acc")},
            investigation_limit=0.005, limit_note="Engineering guard, not a scientific decision threshold; investigate if ACC/NMI/ARI differ by over 0.005")
        write_json(out / "refcheck" / "comparison.json", comparison)
        if max(abs(comparison["delta"][key]) for key in ("acc", "nmi", "ari")) > 0.005:
            status("stopped_reference_drift", comparison=comparison)
            raise RuntimeError("Unexpected k=0 drift: inspect refcheck/comparison.json before further training")

    def add_row(result):
        rows.append(result)
        write_json(out / "results.json", rows)
        save_csv(out / "results.csv", rows)
        status("training", latest_run=result["output_path"])

    complete_observed = np.ones(len(features), dtype=bool)
    complete_cold = np.zeros(len(features), dtype=bool)
    for seed in seeds:
        add_row(fit_one(features.copy(), complete_observed, complete_cold, seed, 0.0, 0, "E", {}))
    for meta in mask_info:
        rate, mask_id = meta["rate"], meta["mask"]
        observed = make_mask(len(features), rate, mask_id, config.mask_seed)
        cold = ~observed & (norm_adj[:, observed].sum(1).A1 < 1e-8)
        fp = fill_features(features, observed, "fp", norm_adj, config.steps)
        # Verify the stage-6 FP implementation agrees, without replacing it.
        if not np.array_equal(fp, fills(features, observed, norm_adj, config.steps)[0]):
            raise AssertionError("FP differs from stage-6 propagation")
        for branch in ("A", "B", "C", "D"):
            filled = fill_features(features, observed, "zero", norm_adj, config.steps) if branch == "A" else fp.copy()
            if branch in ("C", "D"):
                selected = np.asarray(meta["cold_selected_nodes"] if branch == "C" else meta["random_selected_nodes"], dtype=np.int64)
                filled[selected] = features[selected]
                expected = fp.copy()
                expected[selected] = features[selected]
                if not np.array_equal(filled, expected) or not np.array_equal(filled[selected], features[selected]):
                    raise AssertionError("Oracle intervention changed unintended rows")
            for seed in seeds:
                add_row(fit_one(filled, observed, cold, seed, rate, mask_id, branch, meta))

    # E subgroup metrics and exact imputation statistics under each original mask,
    # without duplicate training or claiming six independent E fits.
    complete_views = []
    for meta in mask_info:
        observed = make_mask(len(features), meta["rate"], meta["mask"], config.mask_seed)
        cold = ~observed & (norm_adj[:, observed].sum(1).A1 < 1e-8)
        for seed in seeds:
            row = next(row for row in rows if row["branch"] == "E" and row["seed"] == seed)
            prediction = np.load(Path(row["output_path"]) / "predictions.npz")["pred"]
            metrics = evaluate(labels, prediction, observed, cold)
            complete_views.append(dict(rate=meta["rate"], mask=meta["mask"], seed=seed,
                branch="E", shared_fit_path=row["output_path"], mask_sha256=meta["mask_sha256"],
                missing_count=meta["missing_count"], cold_count=meta["cold_count"],
                replaced_count=meta["missing_count"],
                imputation_rmse=stats(features, features, ~observed)["rmse"],
                imputation_cosine=stats(features, features, ~observed)["cosine"],
                **metrics, **subgroup_metrics(labels, prediction, observed, cold)))
    write_json(out / "complete_mask_evaluations.json", complete_views)
    current_hashes = {name: sha_file(ROOT / name) for name in SOURCES}
    current_refs = {str(p.relative_to(ROOT)): sha_file(p) for p in REFERENCE.rglob("*.json")}
    integrity = dict(source_files_unchanged=source_hashes == current_hashes,
                     reference_json_unchanged=reference_file_hashes == current_refs,
                     source_sha256_after=current_hashes,
                     main_run_count=len(rows), expected_main_runs=protocol["expected_main_runs"],
                     mask_checks_passed=all(item["equal"] for item in mask_checks))
    write_json(out / "integrity.json", integrity)
    if not (integrity["source_files_unchanged"] and integrity["reference_json_unchanged"]):
        raise AssertionError("Source/reference integrity check failed")
    status("complete")
    print(f"ALL COMPLETE: {len(rows)} main fits -> {out}", flush=True)


if __name__ == "__main__":
    main()
