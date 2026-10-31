"""Verify independent full reruns against archived first and second stage outputs."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results_stage2"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    status = read(R / "pipeline_status.json")
    if status["state"] != "complete":
        raise SystemExit("Pipeline incomplete")
    main_p = read(R / "full/cifar100_ir100_s1/P2OT_full_k100/result.json")
    rerun_p = read(R / "rerun/cifar100_ir100_s1/P2OT_full_k100/result.json")
    stage1 = read(ROOT / "results/full/cifar100_ir100_s1/result.json")
    km = read(R / "audit/cifar100_ir100_s1/KM_k100/result.json")
    oracle = read(R / "audit/cifar100_ir100_s1/ORACLE_KM_k100/result.json")
    assert main_p["protocol_sha256"] == rerun_p["protocol_sha256"]
    p2ot = {"prediction_hash_equal": main_p["prediction_sha256"] == rerun_p["prediction_sha256"],
            "BA_abs_diff": abs(main_p["metrics"]["balanced_accuracy"] - rerun_p["metrics"]["balanced_accuracy"]),
            "main_hash": main_p["prediction_sha256"], "rerun_hash": rerun_p["prediction_sha256"]}
    kmcheck = {"prediction_hash_equal": km["prediction_sha256"] == stage1["methods"]["KM"]["pred_array_sha256"],
               "BA_abs_diff": abs(km["metrics"]["balanced_accuracy"] - stage1["methods"]["KM"]["metrics"]["balanced_accuracy"]),
               "inertia_abs_diff": abs(km["fit"]["inertia"] - stage1["methods"]["KM"]["fit"]["inertia"]),
               "main_hash": stage1["methods"]["KM"]["pred_array_sha256"], "rerun_hash": km["prediction_sha256"]}
    oraclecheck = {"prediction_hash_equal": oracle["prediction_sha256"] == stage1["methods"]["ORACLE_KM"]["pred_array_sha256"],
                   "BA_abs_diff": abs(oracle["metrics"]["balanced_accuracy"] - stage1["methods"]["ORACLE_KM"]["metrics"]["balanced_accuracy"])}
    result = {"P2OT_full": p2ot, "KM": kmcheck, "ORACLE_KM": oraclecheck}
    (R / "rerun_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if (not p2ot["prediction_hash_equal"] or not kmcheck["prediction_hash_equal"] or
            not oraclecheck["prediction_hash_equal"] or oraclecheck["BA_abs_diff"] > 1e-6 or
            kmcheck["BA_abs_diff"] > 1e-6 or kmcheck["inertia_abs_diff"] > 1e-4):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
