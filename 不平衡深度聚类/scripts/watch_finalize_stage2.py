"""Wait for background pipeline and generate audited report only after completion."""
import json
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta

ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "results_stage2/pipeline_status.json"
LOG = ROOT / "results_stage2/finalize.log"


def main():
    while True:
        state = json.loads(STATUS.read_text(encoding="utf-8"))["state"]
        if state == "failed":
            LOG.write_text("Pipeline failed; no report generated.\n", encoding="utf-8")
            return 1
        if state == "complete":
            break
        time.sleep(30)
    with LOG.open("w", encoding="utf-8") as stream:
        for script in ("audit_stage2_outputs.py", "analyze_stage2.py"):
            cmd = [sys.executable, "-I", str(ROOT / "scripts" / script)]
            stream.write(f"running {cmd!r}\n")
            stream.flush()
            completed = subprocess.run(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            if completed.returncode:
                stream.write(f"failed {script}: {completed.returncode}\n")
                return completed.returncode
    decision = json.loads((ROOT / "results_stage2/decision.json").read_text(encoding="utf-8"))
    timestamp = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")
    progress = ["# 当前进度", "",
                f"第二阶段已完成：{timestamp}（北京时间），RTX 5060 8GB。流水线 232/232 项，独立 P2OT_full、KM 与 ORACLE_KM 复跑审计通过。",
                "", "- 第一阶段结论：通过；第一阶段文件保持只读。",
                f"- 第二阶段规则 1（强方法与 ORACLE_KM 差距）：{'满足' if decision['rule1'] else '不满足'}。",
                f"- 第二阶段规则 2（真实簇大小先验增益）：{'满足' if decision['rule2'] else '不满足'}。",
                f"- 第二阶段规则 3（K 错设时尾部不对称损失）：{'满足' if decision['rule3'] else '不满足'}。",
                "- 详细结果见 `第二阶段长尾聚类诊断报告.md` 和 `results_stage2/`。第三方官方 MiniClustering 在 CIFAR-100 细类条件无论文配置，未运行。",
                "- GitHub SSH 身份校验未通过（Permission denied (publickey)）；按用户要求仅本地提交，不推送。",
                ""]
    (ROOT / "当前进度.md").write_text("\n".join(progress), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
