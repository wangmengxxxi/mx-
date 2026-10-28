"""Reproduce a failed numeric assertion without changing its tolerance or training.

Run only after the batch has exited, to retain one GPU training process at a time.
The original assertion is kept; arrays and final weights are saved on failure.
"""
from pathlib import Path
import argparse,importlib.util,json,sys,traceback
sys.path.insert(0,str(Path(__file__).resolve().parent))
original=Path(__file__).resolve().parent/'gpu_before_roundoff_fix/run_diagnostic.py'
if original.exists():
    sys.path.insert(0,str(Path(__file__).resolve().parent/'completer'))
    spec=importlib.util.spec_from_file_location('gpu_initial_diagnostic',original)
    rd=importlib.util.module_from_spec(spec);spec.loader.exec_module(rd)
else:
    import run_diagnostic as rd
from threadpoolctl import threadpool_limits

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--dataset',choices=list(rd.SPEC),required=True);ap.add_argument('--seed',type=int,required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    original_close=rd.np.allclose;original_extract=rd.extract;latest={}
    def capture_model(model,*other,**kwargs):
        latest['model']=model
        return original_extract(model,*other,**kwargs)
    def inspect_close(a,b,*other,**kwargs):
        ok=original_close(a,b,*other,**kwargs)
        if not ok:
            diff=rd.np.abs(a-b);rtol=kwargs.get('rtol',1e-5);atol=kwargs.get('atol',1e-8)
            limits=atol+rtol*rd.np.abs(b);worst=rd.np.unravel_index(diff.argmax(),diff.shape)
            info=dict(shape=list(a.shape),dtype=str(a.dtype),original_rtol=rtol,original_atol=atol,
                      max_abs_difference=float(diff.max()),max_difference_to_allowed_ratio=float((diff/limits).max()),
                      failed_elements=int((diff>limits).sum()),total_elements=int(diff.size),
                      value_at_max_difference=[float(a[worst]),float(b[worst])],
                      all_finite=bool(rd.np.isfinite(a).all() and rd.np.isfinite(b).all()),
                      allclose_at_rtol_1e5_atol_1e6=bool(original_close(a,b,rtol=1e-5,atol=1e-6)))
            rd.np.savez_compressed(args.out/'failed_comparison.npz',subset_encoded=a,full_batch_encoded=b)
            (args.out/'numeric_difference.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
            print(json.dumps(info,indent=2),flush=True)
        return ok
    rd.extract=capture_model;rd.np.allclose=inspect_close
    run_args=argparse.Namespace(root=args.root,out=args.out/'run',dataset=args.dataset,rate=.7,seed=args.seed,epochs=500,threads=4,device='cuda')
    try:
        with threadpool_limits(limits=4):rd.run(run_args)
    except AssertionError:
        model=latest['model']
        rd.torch.save({name:getattr(model,name).state_dict() for name in ['autoencoder1','autoencoder2','img2txt','txt2img']},args.out/'model_at_failure.pt')
        (args.out/'assertion.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
