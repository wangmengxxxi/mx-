"""Inspect batch-shape roundoff and verify a validation-only repair changes no weights."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
import torch
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/completer'))
from model import Completer
from datasets import load_data

def inspect():
    records=[]
    for folder in sorted((ROOT/'results/gpu_numeric_check').iterdir()):
        if not (folder/'model_at_failure.pt').exists():continue
        config=json.loads((folder/'run/config.json').read_text(encoding='utf-8'))['config']
        mask=np.load(folder/'run/mask.npy');X,_=load_data(config)
        model=Completer(config)
        weights=torch.load(folder/'model_at_failure.pt',map_location='cpu',weights_only=True)
        for name,state in weights.items():getattr(model,name).load_state_dict(state);getattr(model,name).eval()
        model.to_device(torch.device('cuda'))
        with torch.no_grad():
            for v,ae in enumerate([model.autoencoder1,model.autoencoder2]):
                observed=mask[:,v]==1
                x=torch.from_numpy((X[v]*mask[:,v,None]).astype(np.float32)).cuda()
                subset=ae.encoder(x[observed]).cpu().numpy()
                full=ae.encoder(torch.from_numpy(X[v].astype(np.float32)).cuda()).cpu().numpy()[observed]
                delta=np.abs(subset-full)
                records.append(dict(group=folder.name,view=v,shape=list(subset.shape),max_abs_difference=float(delta.max()),
                                    original_failed_elements=int((delta>1e-7+1e-5*np.abs(full)).sum()),
                                    finite=bool(np.isfinite(subset).all() and np.isfinite(full).all()),
                                    rtol_1e5_atol_1e6_passed=bool(np.allclose(subset,full,rtol=1e-5,atol=1e-6))))
        del model,weights
    (ROOT/'results/gpu_numeric_check/all_view_checks.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    print(json.dumps(records,indent=2))
    assert records and all(r['finite'] and r['rtol_1e5_atol_1e6_passed'] for r in records)

def verify():
    records=[]
    for folder in sorted((ROOT/'results/gpu_numeric_check').iterdir()):
        if not (folder/'model_at_failure.pt').exists():continue
        repaired=ROOT/'results/full'/folder.name
        before=torch.load(folder/'model_at_failure.pt',map_location='cpu',weights_only=True)
        after=torch.load(repaired/'model.pt',map_location='cpu',weights_only=True)
        equal=all(torch.equal(value,after[name][key]) for name,state in before.items() for key,value in state.items())
        arrays=np.load(repaired/'diagnostic_arrays.npz');original=np.load(folder/'failed_comparison.npz')
        mask=np.load(folder/'run/mask.npy')
        assert np.array_equal(mask,arrays['mask'])
        matched=[]
        for v in range(2):
            observed=mask[:,v]==1;a=arrays[f'B_v{v}'][observed];b=arrays[f'oracle_v{v}'][observed]
            if a.shape==original['subset_encoded'].shape:
                matched.append(bool(np.array_equal(a,original['subset_encoded']) and np.array_equal(b,original['full_batch_encoded'])))
        record=dict(group=folder.name,all_weights_and_buffers_bitwise_equal=equal,mask_bitwise_equal=True,
                    failed_comparison_arrays_bitwise_equal=any(matched))
        assert equal and any(matched),record
        records.append(record)
    initial=json.loads((ROOT/'results/gpu_numeric_check/initial_completed_provenance.json').read_text(encoding='utf-8'))
    for item in initial['runs']:
        p=ROOT/'results/full'/item['group']/'result.json'
        assert hashlib.sha256(p.read_bytes()).hexdigest()==item['result_sha256'],item['group']
    result=dict(all_passed=bool(records),checks=records,initial_completed_results_hash_unchanged=len(initial['runs']),protocol_sha256=hashlib.sha256((ROOT/'results/protocol.json').read_bytes()).hexdigest())
    (ROOT/'results/gpu_numeric_check/repair_comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['inspect','verify']);args=ap.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True,warn_only=True)
    with threadpool_limits(limits=4):dict(inspect=inspect,verify=verify)[args.action]()
