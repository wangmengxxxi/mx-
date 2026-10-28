# -*- coding: utf-8 -*-
"""Synthetic label/metric fixtures and full-length head execution, excluded from diagnosis."""
import sys
from pathlib import Path
import numpy as np
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_diagnostic import setup,evaluate,subset,fit_head,dump

def main():
    setup()
    y=np.repeat(np.arange(3),[8,4,2])
    pred=np.array([1]*8+[0]*6)
    metrics=evaluate(y,pred,3)
    assert metrics['ACC']==12/14
    assert metrics['balanced_accuracy']==2/3
    assert metrics['tail_recall']==0
    assert metrics['aligned_cluster_counts']==[8,6,0]
    assert metrics['empty_clusters']==1 and np.isfinite(metrics['KL_true_to_pred'])
    ys=np.repeat(np.arange(100),500)
    a,c=subset(ys,100,1,1)
    b,d=subset(ys,100,10,1)
    e,f=subset(ys,100,100,1)
    assert len(a)==50000 and c[0]==c[-1]==500 and f[0]==500 and f[-1]==5
    assert set(e)<=set(b)<=set(a)
    x=normalize(np.random.RandomState(20261027).normal(size=(1024,768)).astype(np.float32))
    out=ROOT/'results/synthetic_checks'
    out.mkdir(exist_ok=True)
    checked={}
    for method in ['BAL','P2OT_feat']:
        pred,arrays,meta=fit_head(x,10,1,method,out)
        assert pred.shape==(1024,) and arrays['weights'].shape==(10,768) and np.isfinite(arrays['weights']).all()
        checked[method]=meta
    dump(out/'checks.json',{'passed':True,'all_data_synthetic':True,'global_hungarian_shared_by_tail':True,
        'nested_subsets_and_counts':True,'full_100_epoch_head_checks':checked})
    print('Synthetic checks passed; no real-data metrics computed',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=4):
        main()
