# -*- coding: utf-8 -*-
"""Synthetic compute checks only, no dataset labels or feasibility metrics."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['XFORMERS_DISABLED']='1'
import sys
import time
import json
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'third_party/dinov2_py39'))
from dinov2.models.vision_transformer import vit_base
from run_diagnostic import setup, uniform_assignment, load_ppot, sha, dump

def main():
    setup()
    model=vit_base(img_size=518,patch_size=14,init_values=1,block_chunks=0)
    model.load_state_dict(torch.load(ROOT/'data/dinov2_vitb14_pretrain.pth',weights_only=True,map_location='cpu'),strict=True)
    model.requires_grad_(False).eval().half().cuda()
    timings=[]
    for b in [32,16]:
        torch.cuda.reset_peak_memory_stats()
        x=torch.zeros(b,3,224,224,device='cuda',dtype=torch.float16)
        try:
            with torch.inference_mode():
                model(x)
                torch.cuda.synchronize()
                t=time.perf_counter()
                for _ in range(3):
                    z=model(x)
                torch.cuda.synchronize()
                elapsed=time.perf_counter()-t
            assert torch.isfinite(z).all()
            timings.append({'batch':b,'images_per_second':3*b/elapsed,'peak_allocated':torch.cuda.max_memory_allocated()})
            print(timings[-1],flush=True)
        except torch.cuda.OutOfMemoryError:
            timings.append({'batch':b,'OOM':True})
        del x
        torch.cuda.empty_cache()
    del model
    torch.cuda.empty_cache()
    logits=torch.zeros(1024,100,device='cuda')
    q=uniform_assignment(logits)
    assert torch.allclose(q.sum(1),torch.ones(1024,device='cuda'),atol=1e-5)
    assert torch.allclose(q.sum(0),torch.full((100,),10.24,device='cuda'),atol=1e-4)
    solver=load_ppot()
    solver.rho=.5
    t=time.perf_counter()
    qt=solver(logits)
    assert torch.isfinite(qt).all()
    info={'synthetic_feature_timings':timings,'uniform_assignment_marginal_check':True,'ppot_finite_check':True,
        'ppot_mean_row_mass_at_rho_half':float(qt.sum(1).mean()),'ppot_synthetic_s':time.perf_counter()-t,
        'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(0),
        'protocol_sha256':sha(ROOT/'results/protocol.json')}
    dump(ROOT/'results/runtime_check.json',info)

if __name__=='__main__':
    main()
