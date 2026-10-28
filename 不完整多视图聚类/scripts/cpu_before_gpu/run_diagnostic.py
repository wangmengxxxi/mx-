"""CPU-only, fixed-final-epoch COMPLETER diagnostic. No label-driven training.

Reviewed baseline modules are loaded only from scripts/completer, with python -I.
The wrapper changes evaluation instrumentation, never the training objective.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import argparse, hashlib, itertools, json, logging, random, sys, time, types
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from threadpoolctl import threadpool_limits, threadpool_info

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'completer'))
from model import Completer
from configure import get_default_config
from datasets import load_data
from get_mask import get_mask

SPEC={'Caltech101-20':('Caltech101-20',2386,20,[1984,512],0),
      'Scene-15':('Scene_15',4485,15,[20,59],1)}
KM=dict(n_init=10,init='k-means++',max_iter=300,tol=1e-4,algorithm='lloyd')
KM_SEED=20261027

def digest(x):
    x=np.ascontiguousarray(x)
    return hashlib.sha256(str(x.dtype).encode()+str(x.shape).encode()+x.tobytes()).hexdigest()

def dump(p,x):
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def extract(m,mask,x1,x2,dim):
    modules=[m.autoencoder1,m.autoencoder2,m.img2txt,m.txt2img]
    for obj in modules:obj.eval()
    with torch.no_grad():
        zs=[]
        for v,(x,ae,predictor,other_ae,other_x) in enumerate([
            (x1,m.autoencoder1,m.txt2img,m.autoencoder2,x2),
            (x2,m.autoencoder2,m.img2txt,m.autoencoder1,x1)]):
            observed=mask[:,v]==1;missing=~observed
            z=torch.zeros((len(mask),dim),dtype=torch.float32)
            z[observed]=ae.encoder(x[observed])
            if missing.any():z[missing]=predictor(other_ae.encoder(other_x[missing]))[0]
            zs.append(z.numpy())
    for obj in modules:obj.train()
    return zs

def instrumentation_evaluation(self,config,logger,mask,x1,x2,Y_list,device):
    # Preserve official mode switches and global NumPy RNG consumption at 100-epoch cadence.
    # Original evaluation's only training-side effect is KMeans(random_state=None).
    zs=extract(self,mask,x1,x2,config['Autoencoder']['arch1'][-1])
    fusion=np.concatenate(zs,axis=1)
    KMeans(n_clusters=config['n_clusters'],**KM).fit(fusion)
    self.diagnostic_final_z=zs
    logger.info('Unlabelled evaluation mode switch, no intermediate label metrics')
    return {'kmeans':{'accuracy':None,'NMI':None,'ARI':None}}

def cluster(z,k,seed=KM_SEED):
    fusion=np.concatenate(z,axis=1)
    assert np.isfinite(fusion).all()
    return KMeans(n_clusters=k,random_state=seed,**KM).fit_predict(fusion)

def metrics(y,p,k):
    _,y=np.unique(y,return_inverse=True)
    c=np.zeros((k,k),dtype=np.int64);np.add.at(c,(y,p),1)
    r,col=linear_sum_assignment(-c)
    return dict(ACC=float(c[r,col].sum()/len(y)),NMI=float(normalized_mutual_info_score(y,p)),ARI=float(adjusted_rand_score(y,p)))

def run(a):
    started=time.perf_counter();root=a.root.resolve();out=a.out.resolve()
    if (out/'result.json').exists():raise RuntimeError(f'Refusing to overwrite completed run: {out}')
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(a.threads);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    spec,n,k,dims,di=SPEC[a.dataset];config=get_default_config(spec)
    config.update(dataset=spec,data_root=str(root/'data'),n_clusters=k,print_num=100)
    config['training']['missing_rate']=a.rate;config['training']['epoch']=a.epochs
    # Smoke uses a short single final evaluation. Full evaluation cadence is official.
    if a.epochs<100:config['print_num']=a.epochs
    X,Y=load_data(config)
    assert [x.shape for x in X]==[(n,dims[0]),(n,dims[1])]
    # Class count validation is metadata checking, labels are never fed into training/scoring.
    assert len(np.unique(Y[0]))==k
    np.random.seed(a.seed);mask=get_mask(2,n,a.rate).astype(np.int64)
    assert np.isin(mask,[0,1]).all() and (mask.sum(axis=1)>=1).all()
    x=[torch.from_numpy((X[v]*mask[:,v,None]).astype(np.float32)) for v in range(2)]
    seed=config['training']['seed']
    np.random.seed(seed);random.seed(seed+1);torch.manual_seed(seed+2)
    m=Completer(config);m.to_device(torch.device('cpu'))
    m.evaluation=types.MethodType(instrumentation_evaluation,m)
    optimizer=torch.optim.Adam(itertools.chain(m.autoencoder1.parameters(),m.autoencoder2.parameters(),m.img2txt.parameters(),m.txt2img.parameters()),lr=config['training']['lr'])
    logger=logging.getLogger('completer');logger.setLevel(logging.INFO)
    if not logger.handlers:logger.addHandler(logging.StreamHandler(sys.stdout))
    dump(out/'config.json',dict(config=config,arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},mask_sha256=digest(mask),versions=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__),threadpools=threadpool_info()))
    np.save(out/'mask.npy',mask,allow_pickle=False)
    print(f'START {a.dataset} rate={a.rate} mask_seed={a.seed} model_seed={seed} complete={(mask.sum(1)==2).sum()} threads={a.threads}',flush=True)
    train_start=time.perf_counter()
    m.train(config,logger,*x,None,torch.from_numpy(mask).long(),optimizer,torch.device('cpu'))
    train_seconds=time.perf_counter()-train_start
    B=extract(m,torch.from_numpy(mask),*x,config['Autoencoder']['arch1'][-1])
    for actual,expected in zip(B,m.diagnostic_final_z):assert np.array_equal(actual,expected)
    # Training is now over. True withheld features are accessed only below this line.
    for mod in [m.autoencoder1,m.autoencoder2,m.img2txt,m.txt2img]:mod.eval()
    with torch.no_grad():
        oracle=[ae.encoder(torch.from_numpy(raw.astype(np.float32))).numpy() for ae,raw in zip([m.autoencoder1,m.autoencoder2],X)]
    for v in range(2):assert np.allclose(B[v][mask[:,v]==1],oracle[v][mask[:,v]==1],rtol=1e-5,atol=1e-7)
    missing=np.flatnonzero(mask.sum(1)==1);mv=np.argmin(mask[missing],axis=1)
    observed=1-mv;pred=np.stack([B[v][i] for i,v in zip(missing,mv)])
    truth=np.stack([oracle[v][i] for i,v in zip(missing,mv)])
    obs=np.stack([B[v][i] for i,v in zip(missing,observed)])
    error=1-np.sum(pred*truth,axis=1)/np.maximum(np.linalg.norm(pred,axis=1)*np.linalg.norm(truth,axis=1),1e-12)
    pooled=np.concatenate([B[v][mask[:,v]==1] for v in range(2)])
    prototypes=KMeans(n_clusters=k,random_state=KM_SEED,**KM).fit(pooled).cluster_centers_
    def q(z):
        z=z.astype(np.float64);centers=prototypes.astype(np.float64)
        d=np.sum((z[:,None,:]-centers[None,:,:])**2,axis=2);w=1/(1+d)
        return w/w.sum(axis=1,keepdims=True)
    qp,qo=q(pred),q(obs);mid=(qp+qo)/2
    divergence=0.5*np.sum(qp*np.log(qp/mid)+qo*np.log(qo/mid),axis=1)
    s1=-np.maximum(divergence,0)
    assert np.isfinite(s1).all()
    bp=cluster(B,k);stability=[];stability_partitions=[]
    for kmseed in range(100,110):
        p=cluster(B,k,kmseed);c=np.zeros((k,k),dtype=np.int64);np.add.at(c,(p,bp),1)
        rows,cols=linear_sum_assignment(-c);mapping=np.empty(k,dtype=int);mapping[rows]=cols
        aligned=mapping[p];stability.append(aligned==bp);stability_partitions.append(p)
    s2=np.mean(stability,axis=0)[missing]
    half=len(missing)//2
    rng=np.random.default_rng(np.random.SeedSequence([20261027,di,round(a.rate*1000),a.seed,5198418]))
    rand=np.sort(rng.choice(missing,size=half,replace=False))
    selections={'C_rand':rand}
    for name,s in [('s1',s1),('s2',s2)]:selections['C_low_'+name]=np.sort(missing[np.lexsort((missing,s))[:half]])
    branches={'B':B,'A':[z.copy() for z in B],'E':[z.copy() for z in B]}
    for v in range(2):
        branches['A'][v][mask[:,v]==0]=0
        branches['E'][v][mask[:,v]==0]=oracle[v][mask[:,v]==0]
    for name,idx in selections.items():
        zs=[z.copy() for z in B]
        for v in range(2):
            use=idx[mask[idx,v]==0];zs[v][use]=oracle[v][use]
        branches[name]=zs
    arrays=dict(mask=mask,missing=missing,missing_view=mv,error=error,s1=s1,s2=s2,prototypes=prototypes,stability_partitions=np.stack(stability_partitions),labels=Y[0])
    for v in range(2):arrays[f'B_v{v}']=B[v];arrays[f'oracle_v{v}']=oracle[v]
    for name,idx in selections.items():arrays['selection_'+name]=idx
    results={};hashes={}
    for name,z in branches.items():
        p=bp if name=='B' else cluster(z,k)
        arrays['prediction_'+name]=p
        results[name]=metrics(Y[0],p,k)
        counts=np.bincount(p,minlength=k)
        results[name].update(occupied_clusters=int((counts>0).sum()),max_cluster_fraction=float(counts.max()/n))
        hashes[name]=[digest(zv) for zv in z]
        for v in range(2):assert np.array_equal(z[v][mask[:,v]==1],B[v][mask[:,v]==1])
    correlations={}
    for name,s in [('s1',s1),('s2',s2)]:
        rho,pval=spearmanr(s,error)
        correlations[name]=dict(spearman=float(rho) if np.isfinite(rho) else None,p_value=float(pval) if np.isfinite(pval) else None,unique_scores=int(len(np.unique(s))),mean_error_selected=float(error[np.lexsort((missing,s))[:half]].mean()),mean_error_random=float(error[np.isin(missing,rand)].mean()))
    np.savez_compressed(out/'diagnostic_arrays.npz',**arrays)
    torch.save({name:getattr(m,name).state_dict() for name in ['autoencoder1','autoencoder2','img2txt','txt2img']},out/'model.pt')
    result=dict(dataset=a.dataset,rate=a.rate,data_seed=a.seed,model_seed=seed,epochs=a.epochs,training_seconds=train_seconds,total_seconds=time.perf_counter()-started,missing_count=len(missing),replacement_count=half,actual_missing_rate=len(missing)/n,mask_sha256=digest(mask),branches=results,correlations=correlations,input_sha256=hashes,checkpoint_sha256=hashlib.sha256((out/'model.pt').read_bytes()).hexdigest(),selections={name:idx.tolist() for name,idx in selections.items()})
    dump(out/'result.json',result)
    print('COMPLETE '+json.dumps({key:result[key] for key in ['dataset','rate','data_seed','training_seconds','total_seconds','branches','correlations']}),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--dataset',choices=list(SPEC),required=True);ap.add_argument('--rate',type=float,required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--epochs',type=int,default=500);ap.add_argument('--threads',type=int,default=2);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    with threadpool_limits(limits=a.threads):run(a)
