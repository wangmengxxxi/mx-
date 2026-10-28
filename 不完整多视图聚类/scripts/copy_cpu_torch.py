"""Copy fixed installed CPU package files into an independent same-Python conda env.

Source is read-only. Files are real copies, no links or PYTHONPATH reuse.
All other packages are separately installed in destination.
"""
from pathlib import Path
import shutil,hashlib,json
root=Path(__file__).resolve().parents[1]
source=Path(r'F:\Anaconda\envs\pytorch\Lib\site-packages')
target=root/'envs/imvc_cpu/Lib/site-packages'
records=[]
for pattern in ['torch','torch-*.dist-info','functorch','torchgen']:
    for src in source.glob(pattern):
        dst=target/src.name
        if dst.exists():raise RuntimeError(f'Refusing overwrite {dst}')
        shutil.copytree(src,dst,ignore=shutil.ignore_patterns('__pycache__'))
        for p in dst.rglob('*'):
            if p.is_file():
                rel=p.relative_to(target);old=source/rel
                h=hashlib.sha256(p.read_bytes()).hexdigest()
                assert h==hashlib.sha256(old.read_bytes()).hexdigest()
                records.append(dict(file=str(rel),bytes=p.stat().st_size,sha256=h))
(root/'results/torch_copy_manifest.json').write_text(json.dumps(dict(source=str(source),target=str(target),reason='Avoid 619.3 MB duplicate CPU-wheel download; identical Python 3.9.21 ABI; independent byte-identical copies',files=records),indent=2),encoding='utf-8')
print('Copied and verified',len(records),'CPU torch files')
