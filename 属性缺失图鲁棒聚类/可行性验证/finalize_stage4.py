"""Safely finalize stage-four evidence when the three current runs have ended.

This watcher never restarts, stops, or overwrites a training run. It checks
terminal artifacts and writes a failure state if a run exits early.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil


ROOT = Path(__file__).resolve().parent
STATUS = ROOT / 'stage4_execution_status.json'
JOBS = {
    'amgc': (11300, ROOT / 'amgc_stage4_full1000/result.json'),
    'sdcn_repeats': (7272, ROOT / 'sdcn_stage4_holdout/results.json'),
    'sdcn_weighted': (6208, ROOT / 'sdcn_stage4_weighted/results.json'),
}


def write(status, jobs, note=''):
    STATUS.write_text(json.dumps(dict(status=status, jobs=jobs, note=note,
                                      utc=datetime.now(timezone.utc).isoformat()),
                                 ensure_ascii=False, indent=2), encoding='utf-8')


def job_state(pid, result):
    if psutil.pid_exists(pid):
        return 'running'
    if result.exists():
        return 'finished'
    return 'exited_without_result'


def main():
    while True:
        states = {name: job_state(pid, result) for name, (pid, result) in JOBS.items()}
        if 'exited_without_result' in states.values():
            write('failed', states, 'Inspect the corresponding stderr and stdout logs; no run was restarted.')
            return 1
        if all(state == 'finished' for state in states.values()):
            break
        write('running', states)
        time.sleep(30)
    for script in ('summarize_stage4.py', 'build_stage4_manifest.py'):
        completed = subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT,
                                   capture_output=True, text=True)
        if completed.returncode:
            write('failed', states, f'{script} failed: {completed.stderr[-2000:]}')
            return completed.returncode
    write('finished', states, 'Full summary, report, and hash manifest generated.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
