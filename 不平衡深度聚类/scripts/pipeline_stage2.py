"""Sequential, resumable GPU diagnostics; stop on first failure."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone, timedelta

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results_stage2"
STATUS = RESULTS / "pipeline_status.json"
PROGRESS = ROOT / "当前进度.md"
RUNNER = ROOT / "scripts/run_stage2.py"
TZ = timezone(timedelta(hours=8))


def now():
    return datetime.now(TZ).isoformat(timespec="seconds")


def save(obj):
    temp = STATUS.with_suffix(".json.tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(STATUS)
    lines = PROGRESS.read_text(encoding="utf-8").splitlines()
    mark1, mark2 = "<!-- STAGE2_STATUS_BEGIN -->", "<!-- STAGE2_STATUS_END -->"
    if mark1 in lines:
        i, j = lines.index(mark1), lines.index(mark2)
        while j + 1 < len(lines) and not lines[j + 1].strip():
            j += 1
        del lines[i:j+1]
    while len(lines) > 2 and not lines[2].strip():
        del lines[2]
    while len(lines) > 2 and not lines[2].strip():
        del lines[2]
    block = [mark1, f"第二阶段运行状态（{now()}，RTX 5060）：**{obj['state']}**；"
             f"已完成 {obj['completed']}/{obj['total']} 个任务；当前：{obj.get('current','无')}。"
             f"详见 `results_stage2/pipeline_status.json`。", mark2, ""]
    lines[2:2] = block
    PROGRESS.write_text("\n".join(lines) + "\n", encoding="utf-8")


def jobs():
    out = []
    # A/B: three conditions, five paired seeds.
    for dataset, ir in (("cifar100", 100), ("cifar100", 10), ("cifar10", 100)):
        for seed in range(1, 6):
            for method, epochs in (("P2OT_full", 50), ("P2OT_TRUEPRIOR", 50),
                                   ("MiniClustering_feat", 200)):
                out.append((dataset, ir, seed, method, None, epochs, "full"))
            if dataset == "cifar100" and ir == 100:
                out.append((dataset, ir, seed, "KM_ORACLE_INIT", None, 0, "full"))
    # One fresh KM against stage one, including CPU-fit determinism.
    out.append(("cifar100", 100, 1, "KM", 100, 0, "audit"))
    # C: K sensitivity. K=100 full-method results above are reused for IR100.
    for ir in (100, 1):
        for seed in range(1, 6):
            for k in (50, 70, 90, 100, 110, 130):
                for method, epochs in (("KM", 0), ("P2OT_full", 50),
                                       ("MiniClustering_feat", 200)):
                    if ir == 100 and k == 100 and method != "KM":
                        continue
                    out.append(("cifar100", ir, seed, method, k, epochs, "full"))
            out.append(("cifar100", ir, seed, "silhouette", None, 0, "full"))
    # Independent full rerun for P2OT reproducibility.
    out.append(("cifar100", 100, 1, "P2OT_full", 100, 50, "rerun"))
    return out


def main():
    expected = (RESULTS / "protocol_stage2.sha256").read_text().split()[0]
    actual = hashlib.sha256((RESULTS / "protocol_stage2.json").read_bytes()).hexdigest()
    assert actual == expected, "Frozen protocol hash changed"
    planned = jobs()
    status = {"state": "running", "started_at": now(), "updated_at": now(),
              "machine": "RTX 5060 8GB", "protocol_sha256": actual,
              "total": len(planned), "completed": 0, "current": None, "failures": []}
    save(status)
    for dataset, ir, seed, method, k, epochs, group in planned:
        kname = k if k is not None else (10 if dataset == "cifar10" else 100)
        result = RESULTS / group / f"{dataset}_ir{ir}_s{seed}" / f"{method}_k{kname}/result.json"
        name = f"{group}/{dataset}/IR{ir}/seed{seed}/{method}/K{kname}"
        if result.exists():
            entry = json.loads(result.read_text(encoding="utf-8"))
            assert entry["protocol_sha256"] == actual, f"Existing result has wrong protocol: {result}"
            status["completed"] += 1
            status["updated_at"] = now()
            save(status)
            continue
        status["current"] = name
        status["updated_at"] = now()
        save(status)
        log = RESULTS / "logs" / f"{group}_{dataset}_ir{ir}_s{seed}_{method}_k{kname}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        args = [sys.executable, "-I", str(RUNNER), "--dataset", dataset,
                "--ir", str(ir), "--seed", str(seed), "--method", method,
                "--out", f"results_stage2/{group}"]
        if k is not None:
            args += ["--k", str(k)]
        if epochs:
            args += ["--epochs", str(epochs)]
        with log.open("w", encoding="utf-8") as stream:
            stream.write(f"started={now()} command={args!r}\n")
            stream.flush()
            code = subprocess.run(args, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT).returncode
            stream.write(f"finished={now()} exit_code={code}\n")
        if code or not result.exists():
            status["state"] = "failed"
            status["failures"].append({"job": name, "exit_code": code, "log": str(log.relative_to(ROOT))})
            status["updated_at"] = now()
            save(status)
            raise SystemExit(code or 1)
        status["completed"] += 1
        status["updated_at"] = now()
        save(status)
    status["state"] = "complete"
    status["current"] = None
    status["updated_at"] = now()
    save(status)


if __name__ == "__main__":
    main()
