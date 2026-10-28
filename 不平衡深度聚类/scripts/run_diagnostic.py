# -*- coding: utf-8 -*-
"""Frozen feature diagnostics. Unsupervised fit functions do not receive labels."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['OMP_NUM_THREADS']='4'
os.environ['MKL_NUM_THREADS']='4'
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time
import warnings
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import jensenshannon
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
METRICS=['ACC','NMI','ARI','balanced_accuracy','macro_F1','tail_recall']

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):
            h.update(b)
    return h.hexdigest()

def array_sha(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()

def dump(p,v):
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    tmp.replace(p)

def setup():
    assert sha(ROOT/'results/protocol.json')==(ROOT/'results/protocol.sha256').read_text().split()[0]
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False

def subset(labels,k,ir,seed):
    counts=np.array([int((50000/k)*(1/ir)**(c/(k-1))) for c in range(k)],dtype=np.int64)
    rng=np.random.RandomState(seed)
    ids=np.concatenate([rng.permutation(np.flatnonzero(labels==c))[:counts[c]] for c in range(k)])
    assert np.array_equal(np.bincount(labels[ids],minlength=k),counts)
    return ids,counts

def evaluate(labels,pred,k):
    table=np.zeros((k,k),dtype=np.int64)
    np.add.at(table,(pred,labels),1)
    rows,cols=linear_sum_assignment(-table)
    mapping=np.zeros(k,dtype=np.int64)
    mapping[rows]=cols
    mapped=mapping[pred]
    cm=confusion_matrix(labels,mapped,labels=np.arange(k))
    counts=cm.sum(1)
    predicted_counts=cm.sum(0)
    recall=np.diag(cm)/counts
    f1=2*np.diag(cm)/np.maximum(counts+predicted_counts,1)
    tail=np.lexsort((np.arange(k),counts))[:k//3]
    p=counts.astype(np.float64)/len(labels)
    q=predicted_counts.astype(np.float64)/len(labels)
    ps=(counts+1e-12)/(len(labels)+k*1e-12)
    qs=(predicted_counts+1e-12)/(len(labels)+k*1e-12)
    positive=q>0
    return {'ACC':float((mapped==labels).mean()),'NMI':float(normalized_mutual_info_score(labels,pred,average_method='arithmetic')),
        'ARI':float(adjusted_rand_score(labels,pred)),'balanced_accuracy':float(recall.mean()),'macro_F1':float(f1.mean()),
        'tail_recall':float(recall[tail].mean()),'mapping':mapping.tolist(),'recall_per_class':recall.tolist(),
        'tail_classes':tail.tolist(),'true_counts':counts.tolist(),'raw_cluster_counts':np.bincount(pred,minlength=k).tolist(),
        'aligned_cluster_counts':predicted_counts.tolist(),'empty_clusters':int((predicted_counts==0).sum()),
        'KL_true_to_pred':float(np.sum(ps*np.log(ps/qs))), 'JS_true_to_pred':float(jensenshannon(p,q)**2),
        'JS_sorted':float(jensenshannon(np.sort(p),np.sort(q))**2),'JS_pred_to_uniform':float(jensenshannon(q,np.ones(k)/k)**2),
        'normalized_entropy':float(-np.sum(q[positive]*np.log(q[positive]))/np.log(k))}

def fit_km(x,k,seed):
    estimator=KMeans(n_clusters=k,init='k-means++',n_init=10,max_iter=300,tol=1e-4,algorithm='lloyd',random_state=seed)
    pred=estimator.fit_predict(x)
    return pred.astype(np.int64),{'centers':estimator.cluster_centers_},{'n_iter':int(estimator.n_iter_),'inertia':float(estimator.inertia_)}

class Head(nn.Module):
    def __init__(self,d,k):
        super().__init__()
        self.linear=nn.Linear(d,k,bias=False)
    def forward(self,x):
        return 10*F.linear(x,F.normalize(self.linear.weight,dim=1))

@torch.no_grad()
def uniform_assignment(logits):
    # logits=10*cosine, divide by 10 to apply epsilon=.05 to cosine.
    logq=(logits/10/0.05).T.double()
    k,n=logq.shape
    logq-=torch.logsumexp(logq.flatten(),dim=0)
    for _ in range(3):
        logq-=torch.logsumexp(logq,dim=1,keepdim=True)+math.log(k)
        logq-=torch.logsumexp(logq,dim=0,keepdim=True)+math.log(n)
    return (logq+math.log(n)).exp().T.float()

def load_ppot():
    p=next((ROOT/'third_party').glob('PPOT-*'))/'utils/sinkhorn_knopp.py'
    spec=importlib.util.spec_from_file_location('reviewed_ppot_solver',p)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SemiCurrSinkhornKnopp(epsilon=.1,gamma=1,stoperr=1e-6,numItermax=1000,semi_use=True,rho=.1)

def fit_head(x,k,seed,method,out):
    # No labels, no class counts, no oracle centers or KM initialization.
    torch.manual_seed(seed)
    head=Head(x.shape[1],k).cuda()
    generator=torch.Generator().manual_seed(seed)
    xt=torch.from_numpy(x).cuda()
    optimizer=torch.optim.Adam(head.parameters(),lr=.0005,betas=(.9,.999),weight_decay=0)
    solver=load_ppot() if method=='P2OT_feat' else None
    batch=1024
    total_steps=100*math.ceil(len(x)/batch)
    step=0
    history=[]
    started=time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(100):
        lr=.0005*.5*(1+math.cos(math.pi*epoch/100))
        for group in optimizer.param_groups:
            group['lr']=lr
        perm=torch.randperm(len(x),generator=generator).cuda()
        total_loss=0.
        for idx in perm.split(batch):
            logits=head(xt[idx])
            if solver is None:
                target=uniform_assignment(logits.detach())
            else:
                solver.rho=.1+.9*math.exp(-5*(1-step/total_steps)**2)
                target=solver(logits.detach())
            assert torch.isfinite(target).all(),(method,epoch,step,'nonfinite target')
            loss=-(target*F.log_softmax(logits,dim=1)).sum(1).mean()
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss+=float(loss.detach())*len(idx)
            step+=1
        history.append({'epoch':epoch+1,'loss':total_loss/len(x),'lr':lr,'rho':None if solver is None else solver.rho})
        if (epoch+1)%20==0:
            print(json.dumps({'method':method,**history[-1],'elapsed_s':time.perf_counter()-started}),flush=True)
            dump(out/f'{method}_training.json',history)
    with torch.inference_mode():
        pred=torch.cat([head(b).argmax(1) for b in xt.split(4096)]).cpu().numpy().astype(np.int64)
    weights=head.linear.weight.detach().cpu().numpy()
    torch.save({'state_dict':{a:b.cpu() for a,b in head.state_dict().items()},'epoch':100},out/f'{method}_head.pt')
    peak=torch.cuda.max_memory_allocated()
    del head,xt,optimizer,solver
    torch.cuda.empty_cache()
    return pred,{'weights':weights},{'epochs':100,'steps':step,'last_loss':history[-1]['loss'],'peak_allocated_bytes':peak}

def fit_lp(x,y,k,seed):
    estimator=LogisticRegression(C=1,solver='lbfgs',max_iter=1000,tol=1e-4,multi_class='multinomial',class_weight=None,fit_intercept=True,random_state=seed)
    pred=estimator.fit(x,y).predict(x).astype(np.int64)
    return pred,{'coef':estimator.coef_,'intercept':estimator.intercept_},{'n_iter':estimator.n_iter_.tolist()}

def run(dataset,ir,seed,base):
    setup()
    k=10 if dataset=='cifar10' else 100
    features=ROOT/f'data/features/{dataset}_train.npy'
    labels_path=ROOT/f'data/features/{dataset}_train_labels.npy'
    fm=json.loads((ROOT/'results/feature_manifest.json').read_text())
    assert sha(features)==fm[f'{dataset}_train']['sha256']
    assert sha(labels_path)==fm[f'{dataset}_train']['labels_sha256']
    yall=np.load(labels_path)
    ids,counts=subset(yall,k,ir,seed)
    x=normalize(np.load(features,mmap_mode='r')[ids].astype(np.float32),norm='l2')
    x=np.ascontiguousarray(x)
    y=yall[ids]
    out=ROOT/base/f'{dataset}_ir{ir}_s{seed}'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():
        raise RuntimeError('Do not overwrite completed run')
    np.save(out/'indices.npy',ids)
    result={'dataset':dataset,'ir':ir,'seed':seed,'n':len(ids),'k':k,'counts':counts.tolist(),
        'indices_array_sha256':array_sha(ids),'input_array_sha256':array_sha(x),'feature_file_sha256':sha(features),
        'protocol_sha256':sha(ROOT/'results/protocol.json'),'methods':{},'torch':torch.__version__,'device':torch.cuda.get_device_name(0)}
    started=time.perf_counter()
    for method in ['KM','BAL','P2OT_feat','ORACLE_KM','LP']:
        print(json.dumps({'start':method,'dataset':dataset,'ir':ir,'seed':seed,'n':len(ids)}),flush=True)
        t=time.perf_counter()
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter('always')
            if method=='KM':
                pred,arrays,meta=fit_km(x,k,seed)
            elif method in ['BAL','P2OT_feat']:
                pred,arrays,meta=fit_head(x,k,seed,method,out)
            elif method=='ORACLE_KM':
                centers=np.stack([x[y==c].mean(0) for c in range(k)])
                # Distance x^2 is constant over centroids; do not normalize centers.
                pred=(2*x@centers.T-(centers**2).sum(1)).argmax(1).astype(np.int64)
                arrays={'centers':centers}
                meta={'iterations':0}
            else:
                pred,arrays,meta=fit_lp(x,y,k,seed)
        arrays['pred']=pred
        np.savez_compressed(out/f'{method}_arrays.npz',**arrays)
        measured=evaluate(y,pred,k)
        result['methods'][method]={'metrics':measured,'fit':meta,'elapsed_s':time.perf_counter()-t,
            'warnings':[str(w.message) for w in records],'pred_array_sha256':array_sha(pred),
            'archive_sha256':sha(out/f'{method}_arrays.npz')}
        dump(out/'partial_results.json',result)
        print(json.dumps({'complete':method,'elapsed_s':time.perf_counter()-t}),flush=True)
    result['elapsed_s']=time.perf_counter()-started
    result['code_sha256']=sha(Path(__file__))
    dump(out/'result.json',result)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',choices=['cifar10','cifar100'],required=True)
    parser.add_argument('--ir',type=int,choices=[1,10,100],required=True)
    parser.add_argument('--seed',type=int,choices=[1,2,3,4,5],required=True)
    parser.add_argument('--base',default='results/full')
    args=parser.parse_args()
    with threadpool_limits(limits=4):
        run(args.dataset,args.ir,args.seed,args.base)
