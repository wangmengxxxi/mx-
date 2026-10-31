import json, subprocess, sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results_stage3'
PY = sys.executable
TZ = timezone(timedelta(hours=8))
def now(): return datetime.now(TZ).isoformat(timespec='seconds')
def save(status):
    status['updated_at']=now()
    (RESULTS/'pipeline_status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    progress=ROOT/'当前进度.md'
    text=progress.read_text(encoding='utf8')
    start='<!-- STAGE3_STATUS_BEGIN -->'; end='<!-- STAGE3_STATUS_END -->'
    block=f'{start}\n第三阶段经典方法排查（{now()}，RTX 5060 8GB）：**{status["state"]}**，{status["completed"]}/{status["total"]}；当前：{status.get("current") or "无"}。详情见 `results_stage3/pipeline_status.json`。\n{end}'
    if start in text and end in text:
        left=text.index(start); right=text.index(end,left)+len(end); text=text[:left]+block+text[right:]
    else: text=text.rstrip()+'\n\n'+block+'\n'
    progress.write_text(text,encoding='utf8')
def jobs():
    out=[]
    for dataset, ir in [('cifar100',100),('cifar100',10),('cifar10',100),('cifar100',1)]:
        for seed in range(1,6):
            methods=['GMM_diag','vMF_free','OVER_MERGE','FINCH_auto','HDBSCAN_error','HDBSCAN_nearest','Leiden_knn'] if ir==1 else ['GMM_diag','vMF_free','SC_knn','AGG_ward','AGG_avg_cos','OVER_MERGE','FINCH_reqK','FINCH_auto','HDBSCAN_error','HDBSCAN_nearest','Leiden_knn']
            for method in methods: out.append((dataset,ir,seed,method,None))
            for method in ['DeepDPM']: out.append((dataset,ir,seed,method,None))
    for seed in range(1,6):
        for kp in [200,300,500]: out.append(('cifar100',100,seed,'OVER_ORACLE_MERGE',kp))
    return out
def main():
    RESULTS.mkdir(exist_ok=True)
    planned=jobs(); status={'state':'running','started_at':now(),'updated_at':now(),'machine':'RTX 5060 8GB','total':len(planned),'completed':0,'current':None,'skipped':[],'failures':[]}
    save(status)
    for dataset,ir,seed,method,kp in planned:
        name=f'{dataset}/IR{ir}/seed{seed}/{method}'+(f'/Kprime{kp}' if kp else '')
        status['current']=name; save(status)
        if method == 'DeepDPM':
            reason='official shallow clone yielded no worktree; implementation/permission not verified'
            status['skipped'].append({'job':name,'reason':reason}); status['completed']+=1; save(status); continue
        k=10 if dataset=='cifar10' else 100; outk=kp or k
        result=RESULTS/'full'/f'{dataset}_ir{ir}_s{seed}'/f'{method}_k{outk}'/'result.json'
        if result.exists():
            status['completed']+=1; save(status); continue
        args=[PY,'-I',str(ROOT/'scripts/run_stage3.py'),'--dataset',dataset,'--ir',str(ir),'--seed',str(seed),'--method',method]
        if kp: args += ['--kprime',str(kp)]
        log=RESULTS/'logs'/f'{dataset}_ir{ir}_s{seed}_{method}_{kp or "k"}.log'; log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('w',encoding='utf8') as stream:
            stream.write(f'started={now()} command={args!r}\n'); stream.flush(); code=subprocess.run(args,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT).returncode; stream.write(f'finished={now()} exit_code={code}\n')
        if code:
            status['state']='failed'; status['failures'].append({'job':name,'exit_code':code,'log':str(log.relative_to(ROOT))}); save(status); raise SystemExit(code)
        status['completed']+=1; save(status)
    status['state']='complete'; status['current']=None; save(status)
if __name__=='__main__': main()
