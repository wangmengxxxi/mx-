"""Sequential GPU gate runs. Any training/audit failure prevents admission."""
from pathlib import Path
import json,subprocess

ROOT=Path(__file__).resolve().parents[1]
PY=ROOT/'envs/imvc_gpu/python.exe'
BASE=ROOT/'results/gpu_check'

def execute(args,prefix):
    with prefix.with_suffix('.stdout.log').open('w',encoding='utf-8') as out,prefix.with_suffix('.stderr.log').open('w',encoding='utf-8') as err:
        subprocess.run([str(PY),'-I',*map(str,args)],cwd=ROOT,stdout=out,stderr=err,check=True)

if __name__=='__main__':
    BASE.mkdir(exist_ok=True)
    try:
        for idx in [1,2]:
            target=BASE/f'run{idx}'
            execute([ROOT/'scripts/run_diagnostic.py','--root',ROOT,'--dataset','Caltech101-20','--rate','.5','--seed','1',
                     '--epochs','500','--threads','4','--device','cuda','--out',target],BASE/f'run{idx}')
            print('GPU check completed',idx,flush=True)
        execute([ROOT/'scripts/audit_results.py','--root',ROOT,'--runs',BASE,'--out',BASE/'audit.json','--recluster'],BASE/'audit')
        subprocess.run([str(PY),'-I',str(ROOT/'scripts/gpu_transition.py'),'compare'],cwd=ROOT,check=True)
    except Exception as exc:
        (BASE/'failure.json').write_text(json.dumps(dict(passed=False,error=repr(exc)),indent=2),encoding='utf-8')
        if not (BASE/'GPU核对说明.md').exists():
            (BASE/'GPU核对说明.md').write_text('# GPU 核对说明\n\n未通过：执行报错，按要求停止，不进入全量。\n\n错误：`'+repr(exc)+'`\n\n详见各次 stdout/stderr、failure.json。\n',encoding='utf-8')
        raise
