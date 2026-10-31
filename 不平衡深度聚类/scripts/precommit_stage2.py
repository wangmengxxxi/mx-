"""Read-only checks for the stage-two commit; run after report completion."""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / "不平衡深度聚类"
ALLOWED = (
    "不平衡深度聚类/scripts/",
    "不平衡深度聚类/results_stage2/",
    "不平衡深度聚类/第二阶段长尾聚类诊断报告.md",
    "不平衡深度聚类/当前进度.md",
    "不平衡深度聚类/environment_gpu_5060.txt",
)
BANNED_PARTS = {"envs", "data", "third_party"}
BANNED_SUFFIXES = {".npy", ".npz", ".pt", ".pth", ".pdf", ".zip", ".gz"}


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT).decode("utf-8").splitlines()


def main():
    status = json.loads((STAGE / "results_stage2/pipeline_status.json").read_text(encoding="utf-8"))
    assert status["state"] == "complete" and status["completed"] == status["total"]
    assert (STAGE / "第二阶段长尾聚类诊断报告.md").exists()
    audit = json.loads((STAGE / "results_stage2/rerun_audit.json").read_text(encoding="utf-8"))
    assert all(audit[m]["prediction_hash_equal"] for m in ("P2OT_full", "KM", "ORACLE_KM"))
    raw = subprocess.check_output(["git", "diff", "--cached", "--name-only", "-z"], cwd=ROOT)
    paths = [p.decode("utf-8") for p in raw.split(b"\0") if p]
    assert paths, "Nothing staged"
    for path in paths:
        assert any(path == prefix or path.startswith(prefix) for prefix in ALLOWED), path
        relative = Path(path)
        assert not BANNED_PARTS.intersection(relative.parts), path
        assert relative.suffix.lower() not in BANNED_SUFFIXES, path
        file = ROOT / relative
        assert file.exists() and file.stat().st_size <= 20 * 1024 * 1024, path
    print(json.dumps({"staged_files": len(paths), "largest_bytes": max((ROOT / p).stat().st_size for p in paths),
                      "allowed": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
