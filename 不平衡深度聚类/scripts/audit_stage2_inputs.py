"""Check stage-one subset identities without changing stage-one artifacts."""
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def subset(labels, k, ir, seed):
    counts = [int((50000 / k) * (1 / ir) ** (c / (k - 1))) for c in range(k)]
    rng = np.random.RandomState(seed)
    return np.concatenate([
        rng.permutation(np.flatnonzero(labels == c))[:counts[c]]
        for c in range(k)
    ])


def array_sha(array):
    array = np.ascontiguousarray(array)
    return hashlib.sha256(
        str(array.dtype).encode() + str(array.shape).encode() + array.tobytes()
    ).hexdigest()


def main():
    failures = []
    checked = 0
    manifest = json.loads((ROOT / "results/feature_manifest.json").read_text(encoding="utf-8"))
    feature_checks = []
    for dataset in ("cifar10", "cifar100"):
        for split in ("train", "test"):
            key = f"{dataset}_{split}"
            for filename, expected in ((f"{key}.npy", manifest[key]["sha256"]),
                                       (f"{key}_labels.npy", manifest[key]["labels_sha256"])):
                path = ROOT / "data/features" / filename
                actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
                feature_checks.append({"file": filename, "sha256": actual,
                                       "expected": expected, "match": actual == expected})
                if actual != expected:
                    failures.append({"file": filename, "expected": expected, "actual": actual})
    index_checks = []
    for dataset, k in (("cifar10", 10), ("cifar100", 100)):
        labels = np.load(ROOT / f"data/features/{dataset}_train_labels.npy")
        for ir in (1, 10, 100):
            for seed in range(1, 6):
                path = ROOT / f"results/full/{dataset}_ir{ir}_s{seed}/result.json"
                expected = json.loads(path.read_text(encoding="utf-8"))["indices_array_sha256"]
                actual = array_sha(subset(labels, k, ir, seed))
                checked += 1
                index_checks.append({"dataset": dataset, "ir": ir, "seed": seed,
                                     "sha256": actual, "expected": expected, "match": actual == expected})
                if actual != expected:
                    failures.append({"dataset": dataset, "ir": ir, "seed": seed,
                                     "expected": expected, "actual": actual})
    archive = ROOT / "data/cifar/cifar-100-python.tar.gz"
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest() if archive.exists() else None
    archive_expected = "85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7"
    if archive_sha != archive_expected:
        failures.append({"file": str(archive), "expected": archive_expected, "actual": archive_sha})
    record = {"feature_checks": feature_checks, "index_checks": index_checks,
              "cifar100_archive_sha256": archive_sha, "cifar100_archive_expected": archive_expected,
              "checked": checked, "failures": failures}
    output = ROOT / "results_stage2/input_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"features": len(feature_checks), "indices": checked,
                      "failures": failures}, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
