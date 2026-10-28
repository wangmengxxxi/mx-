# -*- coding: utf-8 -*-
"""Sequential pipeline with durable logs and fail-fast status; no metric-based retries."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]

def event(stage,**kwargs):
    obj={'stage':stage,'time':time.strftime('%Y-%m-%dT%H:%M:%S'),**kwargs}
    with open(ROOT/'results/pipeline_events.jsonl','a',encoding='utf-8') as f:
        f.write(json.dumps(obj)+'\n')
    (ROOT/'results/pipeline_status.json').write_text(json.dumps(obj,indent=2),encoding='utf-8')
    print(json.dumps(obj),flush=True)

def call(script,args,log):
    event('start',script=script,args=args)
    with open(ROOT/f'results/{log}.stdout.log','w',encoding='utf-8') as stdout,open(ROOT/f'results/{log}.stderr.log','w',encoding='utf-8') as stderr:
        r=subprocess.run([sys.executable,'-I','-u',str(ROOT/'scripts'/script),*args],cwd=ROOT,stdout=stdout,stderr=stderr)
    if r.returncode:
        event('failed',script=script,args=args,exit_code=r.returncode,stderr=f'results/{log}.stderr.log')
        raise RuntimeError(f'{script} failed, see {log}')
    event('complete',script=script,args=args)

def main():
    event('waiting_for_verified_downloads')
    specs=[('cifar-100-python.tar.gz',169001437,'eb9058c3a382ffc7106e4002c42a8d85'),('cifar-10-python.tar.gz',170498071,'c58f30108f718f92721af3b95e74349a')]
    downloader=None
    next_retry=0
    while True:
        ready=True
        for name,total,md5 in specs:
            p=ROOT/'data/cifar'/name
            if not p.exists() or p.stat().st_size!=total:
                ready=False
                break
            if hashlib.md5(p.read_bytes()).hexdigest()!=md5:
                event('archive_md5_failed',file=name)
                raise RuntimeError('Do not consume corrupt data')
        if ready:
            break
        if downloader is not None and downloader.poll() is not None:
            event('download_attempt_ended',exit_code=downloader.returncode)
            downloader=None
            next_retry=time.monotonic()+60
        if downloader is None and time.monotonic()>=next_retry:
            with open(ROOT/'results/download_queue.stdout.log','a',encoding='utf-8') as stdout,open(ROOT/'results/download_queue.stderr.log','a',encoding='utf-8') as stderr:
                downloader=subprocess.Popen([sys.executable,'-I','-u',str(ROOT/'scripts/download_cifar.py')],cwd=ROOT,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
            event('downloading_official_cifar',pid=downloader.pid)
        time.sleep(15)
    manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'scripts').glob('*.py')}
    (ROOT/'results/runtime_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    call('extract_features.py',[],'extraction')
    # Protocol order only, never conditional on performance. Each run is a fresh isolated process.
    for dataset in ['cifar100','cifar10']:
        for ir in [1,10,100]:
            for seed in [1,2,3,4,5]:
                key=f'{dataset}_ir{ir}_s{seed}'
                if (ROOT/'results/full'/key/'result.json').exists():
                    event('skip_completed_run',key=key)
                    continue
                call('run_diagnostic.py',['--dataset',dataset,'--ir',str(ir),'--seed',str(seed)],f'run_{key}')
    call('run_diagnostic.py',['--dataset','cifar100','--ir','100','--seed','1','--base','results/refcheck'],'refcheck')
    call('audit_results.py',[],'audit')
    call('summarize_results.py',[],'summary')
    event('all_complete')

if __name__=='__main__':
    import ctypes
    # Prevent automatic sleep for this process only; release on completion/failure.
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        main()
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
