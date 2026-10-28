# -*- coding: utf-8 -*-
"""Reconstruct all interventions and independently recompute offline metrics."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score,adjusted_rand_score
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_diagnostic import digest,KM,KM_SEED,SPEC
from get_mask import get_mask
from model import Completer
from configure import get_default_config
from datasets import load_data
from evaluation import clustering_metric

def audit_one(p,recluster=False,root=None):
    run_dir=p.parent
    r=json.loads(p.read_text(encoding='utf-8'));z=np.load(p.parent/'diagnostic_arrays.npz',allow_pickle=False)
    mask=z['mask'];missing=z['missing'];n=len(mask);k=SPEC[r['dataset']][2]
    np.random.seed(r['data_seed']);rebuilt=get_mask(2,n,r['rate']).astype(np.int64)
    assert np.array_equal(mask,rebuilt)
    assert digest(mask)==r['mask_sha256']
    assert np.array_equal(missing,np.flatnonzero(mask.sum(1)==1))
    assert np.array_equal(z['missing_view'],np.argmin(mask[missing],axis=1))
    B=[z[f'B_v{v}'] for v in range(2)];oracle=[z[f'oracle_v{v}'] for v in range(2)]
    if root is not None:
        spec=SPEC[r['dataset']][0];config=get_default_config(spec)
        config.update(dataset=spec,data_root=str(root/'data'))
        model=Completer(config)
        weights=torch.load(run_dir/'model.pt',map_location='cpu',weights_only=True)
        for name,state in weights.items():getattr(model,name).load_state_dict(state);getattr(model,name).eval()
        X,Y=load_data(config)
        assert np.array_equal(Y[0],z['labels'])
        with torch.no_grad():
            for v,ae in enumerate([model.autoencoder1,model.autoencoder2]):
                truth_z=ae.encoder(torch.from_numpy(X[v].astype(np.float32))).numpy()
                assert np.allclose(truth_z,oracle[v],rtol=1e-5,atol=2e-7)
                observed=mask[:,v]==1
                assert np.allclose(truth_z[observed],B[v][observed],rtol=1e-5,atol=2e-7)
                missing_idx=mask[:,v]==0
                other_ae=[model.autoencoder2,model.autoencoder1][v]
                predictor=[model.txt2img,model.img2txt][v]
                predicted=predictor(other_ae.encoder(torch.from_numpy(X[1-v][missing_idx].astype(np.float32))))[0].numpy()
                assert np.allclose(predicted,B[v][missing_idx],rtol=1e-5,atol=2e-7)
    pred=np.stack([B[v][i] for i,v in zip(missing,z['missing_view'])]);truth=np.stack([oracle[v][i] for i,v in zip(missing,z['missing_view'])])
    err=1-np.sum(pred*truth,axis=1)/np.maximum(np.linalg.norm(pred,axis=1)*np.linalg.norm(truth,axis=1),1e-12)
    assert np.array_equal(err,z['error'])
    centers=z['prototypes'].astype(np.float64)
    obs=np.stack([B[1-v][i] for i,v in zip(missing,z['missing_view'])])
    def q(x):
        d=((x.astype(np.float64)[:,None,:]-centers[None,:,:])**2).sum(2);w=1/(1+d);return w/w.sum(1,keepdims=True)
    p1,p2=q(pred),q(obs);mid=(p1+p2)/2
    s1=-np.maximum(0.5*np.sum(p1*np.log(p1/mid)+p2*np.log(p2/mid),axis=1),0)
    assert np.allclose(s1,z['s1'],atol=1e-15,rtol=1e-12) and np.isfinite(s1).all()
    bp=z['prediction_B'];agrees=[]
    for part in z['stability_partitions']:
        c=np.zeros((k,k),dtype=int);np.add.at(c,(part,bp),1)
        rows,cols=linear_sum_assignment(-c);mapping=np.empty(k,dtype=int);mapping[rows]=cols
        agrees.append(mapping[part]==bp)
    s2=np.mean(agrees,axis=0)[missing];assert np.array_equal(s2,z['s2'])
    rng=np.random.default_rng(np.random.SeedSequence([20261027,SPEC[r['dataset']][4],round(r['rate']*1000),r['data_seed'],5198418]))
    rand=np.sort(rng.choice(missing,size=len(missing)//2,replace=False));assert np.array_equal(rand,z['selection_C_rand'])
    for name,s in [('s1',s1),('s2',s2)]:
        assert np.array_equal(np.sort(missing[np.lexsort((missing,s))[:len(missing)//2]]),z['selection_C_low_'+name])
        rho,pval=spearmanr(s,err)
        saved=r['correlations'][name]['spearman']
        if saved is None:
            assert not np.isfinite(rho) and (len(np.unique(s))==1 or len(np.unique(err))==1)
        else:assert abs(rho-saved)<1e-12
    metrics_recomputed={}
    _,y=np.unique(z['labels'],return_inverse=True)
    for name,expected_hashes in r['input_sha256'].items():
        zs=[b.copy() for b in B]
        if name=='A':
            for v in range(2):zs[v][mask[:,v]==0]=0
        elif name=='E':
            for v in range(2):zs[v][mask[:,v]==0]=oracle[v][mask[:,v]==0]
        elif name.startswith('C_'):
            idx=z['selection_'+name]
            assert idx.tolist()==r['selections'][name] and len(idx)==len(missing)//2
            for v in range(2):
                take=idx[mask[idx,v]==0];zs[v][take]=oracle[v][take]
        assert [digest(x) for x in zs]==expected_hashes
        for v in range(2):assert np.array_equal(zs[v][mask[:,v]==1],B[v][mask[:,v]==1])
        p=z['prediction_'+name]
        if recluster:
            rebuilt_p=KMeans(n_clusters=k,random_state=KM_SEED,**KM).fit_predict(np.concatenate(zs,axis=1))
            assert np.array_equal(rebuilt_p,p),(str(p),name)
        table=np.zeros((k,k),dtype=np.int64);np.add.at(table,(y,p),1);rows,cols=linear_sum_assignment(-table)
        vals=dict(ACC=float(table[rows,cols].sum()/len(y)),NMI=float(normalized_mutual_info_score(y,p)),ARI=float(adjusted_rand_score(y,p)))
        for metric,val in vals.items():assert abs(val-r['branches'][name][metric])<1e-12
        official,_=clustering_metric(y,p,k)
        for metric,key in [('ACC','accuracy'),('NMI','NMI'),('ARI','ARI')]:
            assert abs(float(np.round(vals[metric],4))-float(official[key]))<1e-12
        metrics_recomputed[name]=vals
    assert hashlib.sha256((run_dir/'model.pt').read_bytes()).hexdigest()==r['checkpoint_sha256']
    return dict(run=str(run_dir),mask_match=True,selections_match=True,branch_hashes_match=True,metrics_match=True,official_metrics_rounding_match=True,scores_match=True,checkpoint_hash_match=True,kmeans_refit_match=recluster)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--runs',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--recluster',action='store_true');a=ap.parse_args()
    results=[]
    for p in sorted(a.runs.rglob('result.json')):
        result=audit_one(p,a.recluster,a.root);results.append(result);print('AUDIT OK',str(p.parent),flush=True)
    inventory=json.loads((a.root/'results/source_inventory.json').read_text(encoding='utf-8'))
    for path,meta in inventory['files'].items():assert hashlib.sha256((a.root/path).read_bytes()).hexdigest()==meta['sha256'],path
    for name,info in inventory['datasets'].items():assert hashlib.sha256((a.root/'data'/f'{name}.mat').read_bytes()).hexdigest()==info['sha256']
    protocol_hash=hashlib.sha256((a.root/'results/protocol.json').read_bytes()).hexdigest()
    assert protocol_hash=='7ddf3260ccebdf199cea02efdb9f562e8afc66c6a725f05ff4bcf0d6553cb34a'
    runtime=json.loads((a.root/'results/runtime_manifest.json').read_text(encoding='utf-8'))
    for path,h in runtime.items():assert hashlib.sha256((a.root/path).read_bytes()).hexdigest()==h,path
    out=dict(all_passed=True,runs=len(results),checks=results,third_party_unchanged=True,data_hashes_match=True,training_runtime_unchanged=True,protocol_sha256=protocol_hash)
    a.out.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print('ALL AUDITS PASSED',len(results),flush=True)
if __name__=='__main__':
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    with threadpool_limits(limits=4):main()
