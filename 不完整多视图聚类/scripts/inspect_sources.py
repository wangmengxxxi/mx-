"""Read-only source/data inventory. Never imports downloaded Python modules."""
import argparse, ast, hashlib, json, platform, sys
from pathlib import Path
import numpy as np
import scipy.io as sio

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args()
    root=a.root.resolve();repo=root/'third_party/2021-CVPR-Completer'
    inventory={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in repo.rglob('*') if p.is_file()}
    for p in repo.glob('*.py'): ast.parse(p.read_text(encoding='utf-8'))
    datasets={}
    for name,inds,n,k,dims in [('Caltech101-20',[3,4],2386,20,[1984,512]),('Scene-15',[0,1],4485,15,[20,59])]:
        source=repo/'data'/f'{name}.mat';d=sio.loadmat(source);views=list(d['X'][0]);y=d['Y'].ravel()
        info=dict(samples=len(y),classes=len(np.unique(y)),stored_views=len(views),all_shapes=[list(x.shape) for x in views],used_view_indices=inds,used_dimensions=[views[i].shape[1] for i in inds],sha256=sha(source))
        assert len(y)==n and len(np.unique(y))==k and info['used_dimensions']==dims,info
        for i in inds:
            assert views[i].shape[0]==n and np.isfinite(views[i]).all()
        datasets[name]=info
        target=root/'data'/source.name
        if not target.exists(): target.write_bytes(source.read_bytes())
        assert sha(target)==sha(source)
    out=dict(repository='https://github.com/XLearning-SCU/2021-CVPR-Completer',commit='624dd9fd10431802214fab1d7b34c44f5a0a92df',license='MIT',retrieval='GitHub raw files pinned to commit; clone failed TLS',datasets=datasets,files=inventory,python=sys.version,platform=platform.platform())
    (root/'results/source_inventory.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(datasets,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
