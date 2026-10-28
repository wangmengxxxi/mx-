"""Serial reproduction of every numerical-check failure, using original GPU code."""
from pathlib import Path
import json,subprocess
ROOT=Path(__file__).resolve().parents[1]
PY=ROOT/'envs/imvc_gpu/python.exe'
BASE=ROOT/'results/gpu_numeric_check'

if __name__=='__main__':
    BASE.mkdir(exist_ok=True)
    errors_path=ROOT/'results/batch_errors.json'
    if not errors_path.exists():errors_path=ROOT/'results/operational_gpu_roundoff/batch_errors.json'
    errors=json.loads(errors_path.read_text(encoding='utf-8'))
    summary=[]
    for item in errors:
        dataset,rate,seed=item['job'];assert rate==.7
        name=f'{dataset}_r{rate}_s{seed}';out=BASE/name
        with (BASE/(name+'.stdout.log')).open('w',encoding='utf-8') as stdout,(BASE/(name+'.stderr.log')).open('w',encoding='utf-8') as stderr:
            p=subprocess.run([str(PY),'-I',str(ROOT/'scripts/inspect_gpu_roundoff.py'),'--root',str(ROOT),'--out',str(out),
                              '--dataset',dataset,'--seed',str(seed)],cwd=ROOT,stdout=stdout,stderr=stderr)
        assert p.returncode!=0 and (out/'numeric_difference.json').exists() and (out/'model_at_failure.pt').exists(),name
        info=json.loads((out/'numeric_difference.json').read_text(encoding='utf-8'))
        summary.append(dict(group=name,**info));print(json.dumps(summary[-1]),flush=True)
    (BASE/'numeric_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
