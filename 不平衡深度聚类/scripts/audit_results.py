# -*- coding: utf-8 -*-
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_diagnostic import setup,sha,array_sha,dump,subset,evaluate,METRICS,Head

def main():
    setup()
    checks=[]
    fm=json.loads((ROOT/'results/feature_manifest.json').read_text())
    assert len(fm)==4
    for key,info in fm.items():
        p=ROOT/info['path']
        a=np.load(p,mmap_mode='r')
        assert sha(p)==info['sha256'] and list(a.shape)==info['shape'] and a.dtype==np.float32
        assert np.isfinite(a).all() and np.linalg.norm(a,axis=1).min()>0
        assert sha(p.with_name(p.stem+'_labels.npy'))==info['labels_sha256']
        checks.append({'feature':key,'hash_shape_finite':True})
    third=json.loads((ROOT/'results/third_party_manifest.json').read_text())
    for path,h in third.items():
        assert sha(ROOT/path)==h,path
    code=json.loads((ROOT/'results/runtime_manifest.json').read_text())
    for path,h in code.items():
        assert sha(ROOT/path)==h,path
    files=sorted((ROOT/'results/full').glob('*/result.json'))
    assert len(files)==30
    for p in files:
        result=json.loads(p.read_text(encoding='utf-8'))
        dataset,ir,seed,k=(result[key] for key in ['dataset','ir','seed','k'])
        ys=np.load(ROOT/f'data/features/{dataset}_train_labels.npy')
        ids,counts=subset(ys,k,ir,seed)
        assert np.array_equal(ids,np.load(p.parent/'indices.npy'))
        assert array_sha(ids)==result['indices_array_sha256']
        assert np.array_equal(counts,result['counts'])
        x=normalize(np.load(ROOT/f'data/features/{dataset}_train.npy',mmap_mode='r')[ids])
        x=np.ascontiguousarray(x)
        assert array_sha(x)==result['input_array_sha256']
        y=ys[ids]
        for method,entry in result['methods'].items():
            archive=p.parent/f'{method}_arrays.npz'
            assert sha(archive)==entry['archive_sha256']
            a=np.load(archive)
            pred=a['pred']
            assert array_sha(pred)==entry['pred_array_sha256']
            metrics=evaluate(y,pred,k)
            assert metrics==entry['metrics'],(p,method,'metric mismatch')
            if method in ['KM','ORACLE_KM']:
                centers=a['centers']
                if method=='ORACLE_KM':
                    expected=np.stack([x[y==c].mean(0) for c in range(k)])
                    assert np.array_equal(centers,expected)
                rebuilt=(2*x@centers.T-(centers**2).sum(1)).argmax(1)
            elif method in ['BAL','P2OT_feat']:
                head=Head(x.shape[1],k).cuda()
                saved=torch.load(p.parent/f'{method}_head.pt',map_location='cpu',weights_only=True)
                assert saved['epoch']==100
                head.load_state_dict(saved['state_dict'])
                assert np.array_equal(a['weights'],saved['state_dict']['linear.weight'].numpy())
                with torch.inference_mode():
                    rebuilt=torch.cat([head(torch.from_numpy(b).cuda()).argmax(1).cpu() for b in np.array_split(x,max(1,(len(x)+4095)//4096))]).numpy()
                del head
            else:
                rebuilt=(x@a['coef'].T+a['intercept']).argmax(1)
            assert np.array_equal(pred,rebuilt),(p,method,'prediction reconstruction mismatch',int((pred!=rebuilt).sum()))
            checks.append({'run':p.parent.name,'method':method,'subset_metrics_saved_parameters':True})
    original=ROOT/'results/full/cifar100_ir100_s1'
    repeat=ROOT/'results/refcheck/cifar100_ir100_s1'
    ro=json.loads((original/'result.json').read_text(encoding='utf-8'))
    rr=json.loads((repeat/'result.json').read_text(encoding='utf-8'))
    comparison={'same_indices':np.array_equal(np.load(original/'indices.npy'),np.load(repeat/'indices.npy')),
        'same_input_hash':ro['input_array_sha256']==rr['input_array_sha256'],'methods':{}}
    for method in ro['methods']:
        aa=np.load(original/f'{method}_arrays.npz')
        bb=np.load(repeat/f'{method}_arrays.npz')
        arrays={}
        for key in aa.files:
            arrays[key]={'equal':np.array_equal(aa[key],bb[key]),'max_abs_diff':float(np.max(np.abs(aa[key].astype(np.float64)-bb[key].astype(np.float64))))}
            assert np.allclose(aa[key],bb[key],atol=1e-6,rtol=1e-5)
        diffs={m:rr['methods'][method]['metrics'][m]-ro['methods'][method]['metrics'][m] for m in METRICS}
        assert max(abs(v) for v in diffs.values())<=1e-6
        comparison['methods'][method]={'metric_differences':diffs,'arrays':arrays}
    assert comparison['same_indices'] and comparison['same_input_hash']
    dump(ROOT/'results/refcheck/comparison.json',comparison)
    dump(ROOT/'results/audit.json',{'passed':True,'formal_runs':len(files),'checks':checks,'protocol_sha256':sha(ROOT/'results/protocol.json'),
        'third_party_original_and_compatibility_hashes':True,'runtime_code_hashes':True,'refcheck_passed':True})
    print('All feature, source, code, 30-run reconstruction and independent rerun checks passed',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=4):
        main()
