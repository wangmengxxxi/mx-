"""Watch stage-five runs and create the summary only after both complete."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil


ROOT = Path(__file__).resolve().parent
STATUS = ROOT / 'stage5_execution_status.json'
JOBS = {
    'reconstruction_weighting': (2392, ROOT / 'sdcn_stage5_reconstruction/results.json'),
    'degree_biased_baseline': (13476, ROOT / 'sdcn_stage5_degree_baseline/results.json'),
}


def write(status, jobs, note=''):
    STATUS.write_text(json.dumps(dict(status=status, jobs=jobs, note=note,
                                      utc=datetime.now(timezone.utc).isoformat()),
                                 ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    while True:
        states = {name: ('finished' if result.exists() else
                         'running' if psutil.pid_exists(pid) else 'exited_without_result')
                  for name, (pid, result) in JOBS.items()}
        if 'exited_without_result' in states.values():
            write('failed', states, 'Inspect logs; no training run was restarted or overwritten.')
            return 1
        if all(state == 'finished' for state in states.values()):
            break
        write('running', states)
        time.sleep(30)
    for script in ('summarize_stage5.py', 'build_stage5_manifest.py'):
        completed = subprocess.run([sys.executable, str(ROOT / script)],
                                   cwd=ROOT, capture_output=True, text=True)
        if completed.returncode:
            write('failed', states, f'{script} failed: {completed.stderr[-3000:]}')
            return completed.returncode
    write('finished', states, 'Full summary, report, and manifest generated.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
