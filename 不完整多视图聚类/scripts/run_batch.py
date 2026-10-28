"""Background CPU/CUDA experiment driver. Run from root using python -I.

Wait for the first standalone Caltech run, then at most two independent CPU
processes (four threads each). Training is shared by all branches in each run.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import argparse,datetime,hashlib,json,subprocess,sys,time

ROOT=Path(__file__).resolve().parents[1];PY=ROOT/'envs/imvc_cpu/python.exe'
FULL=ROOT/'results/full';LOG=ROOT/'results/batch_events.jsonl'
DEVICE='cpu';WORKERS=2

def event(kind,**kwargs):
    r=dict(event=kind,time=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),**kwargs)
    with LOG.open('a',encoding='utf-8') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    print(json.dumps(r,ensure_ascii=False),flush=True)

def run_one(dataset,rate,seed,rerun=False):
    name=f'{dataset}_r{rate}_s{seed}';folder=(ROOT/'results/refcheck'/name) if rerun else FULL/name
    if (folder/'result.json').exists():
        existing=json.loads((folder/'result.json').read_text(encoding='utf-8'))
        assert existing.get('device','cpu')==DEVICE,'Refusing mixed-device result reuse'
        event('reuse_completed',name=name);return folder
    prefix=folder.parent/name
    folder.parent.mkdir(exist_ok=True)
    cmd=[str(PY),'-I',str(ROOT/'scripts/run_diagnostic.py'),'--root',str(ROOT),'--dataset',dataset,'--rate',str(rate),'--seed',str(seed),'--epochs','500','--threads','4','--out',str(folder)]
    cmd.extend(['--device',DEVICE])
    event('start',name=name,rerun=rerun,command=cmd)
    start=time.monotonic()
    with prefix.with_suffix(prefix.suffix+'.stdout.log').open('w',encoding='utf-8') as stdout,prefix.with_suffix(prefix.suffix+'.stderr.log').open('w',encoding='utf-8') as stderr:
        p=subprocess.run(cmd,cwd=ROOT,stdout=stdout,stderr=stderr)
    if p.returncode or not (folder/'result.json').exists():
        event('failed',name=name,exit_code=p.returncode);raise RuntimeError(name)
    r=json.loads((folder/'result.json').read_text(encoding='utf-8'))
    event('completed',name=name,rerun=rerun,wall_seconds=time.monotonic()-start,training_seconds=r['training_seconds'])
    return folder

def main():
    event('driver_started',device=DEVICE,workers=WORKERS,python=str(PY))
    if DEVICE=='cuda':
        check=json.loads((ROOT/'results/gpu_check/comparison.json').read_text(encoding='utf-8'))
        assert check['passed'],'GPU gate has not passed'
        run_one('Caltech101-20',0.5,1)
    pilot=FULL/'Caltech101-20_r0.5_s1/result.json';t0=time.monotonic()
    while not pilot.exists():
        if time.monotonic()-t0>1800:raise RuntimeError('Pilot did not complete within 30 min after driver start; inspect CPU fallback condition')
        time.sleep(15)
    p=json.loads(pilot.read_text(encoding='utf-8'))
    event('standalone_pilot_checked',training_seconds=p['training_seconds'])
    if DEVICE=='cpu' and p['training_seconds']>1200:
        event('fallback_required',reason='Standalone CPU COMPLETER training exceeded approximately 20 min')
        raise RuntimeError('Do not start full COMPLETER batch. Switch baseline per protocol.')
    scene=run_one('Scene-15',0.5,1)
    scene_time=json.loads((scene/'result.json').read_text(encoding='utf-8'))['training_seconds']
    event('standalone_scene_checked',training_seconds=scene_time)
    if DEVICE=='cpu' and scene_time>1200:
        event('fallback_required',reason='Standalone Scene CPU COMPLETER training exceeded approximately 20 min')
        raise RuntimeError('Do not start remaining COMPLETER batch. Switch baseline per protocol.')
    jobs=[(d,r,s) for r in [0.5,0.7] for d in ['Caltech101-20','Scene-15'] for s in range(1,6) if not (r==0.5 and s==1)]
    errors=[]
    if DEVICE=='cuda':
        # Serial execution also stops admission immediately on any new failure.
        for job in jobs:run_one(*job)
    else:
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            tasks={pool.submit(run_one,*job):job for job in jobs}
            for task in as_completed(tasks):
                try:task.result()
                except Exception as exc:errors.append(dict(job=tasks[task],error=repr(exc)))
    if errors:
        (ROOT/'results/batch_errors.json').write_text(json.dumps(errors,indent=2),encoding='utf-8');raise RuntimeError(errors)
    run_one('Caltech101-20',0.5,1,True)
    event('training_batch_finished',unique_trainings=20,refcheck_trainings=1)
    for runs,output in [(ROOT/'results/full',ROOT/'results/audit.json'),(ROOT/'results/refcheck',ROOT/'results/refcheck/audit.json')]:
        cmd=[str(PY),'-I',str(ROOT/'scripts/audit_results.py'),'--root',str(ROOT),'--runs',str(runs),'--out',str(output),'--recluster']
        with (output.parent/(output.stem+'.stdout.log')).open('w',encoding='utf-8') as stdout,(output.parent/(output.stem+'.stderr.log')).open('w',encoding='utf-8') as stderr:
            subprocess.run(cmd,cwd=ROOT,stdout=stdout,stderr=stderr,check=True)
    event('batch_and_audits_completed')
    if (ROOT/'results/gpu_numeric_check/numeric_summary.json').exists():
        subprocess.run([str(PY),'-I',str(ROOT/'scripts/verify_gpu_roundoff.py'),'verify'],cwd=ROOT,check=True)
    with (ROOT/'results/summary.stdout.log').open('w',encoding='utf-8') as stdout,(ROOT/'results/summary.stderr.log').open('w',encoding='utf-8') as stderr:
        subprocess.run([str(PY),'-I',str(ROOT/'scripts/summarize_results.py'),'--root',str(ROOT)],cwd=ROOT,stdout=stdout,stderr=stderr,check=True)
    event('report_completed',path=str(ROOT/'第一阶段IMVC可行性报告.md'))

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    ap.add_argument('--workers',type=int,default=2)
    args=ap.parse_args()
    if args.workers<1:ap.error('--workers must be positive')
    DEVICE=args.device;WORKERS=1 if DEVICE=='cuda' else args.workers
    PY=ROOT/('envs/imvc_gpu/python.exe' if DEVICE=='cuda' else 'envs/imvc_cpu/python.exe')
    try:main()
    except Exception as e:event('driver_failed',error=repr(e));raise
