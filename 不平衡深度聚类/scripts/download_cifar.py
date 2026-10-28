# -*- coding: utf-8 -*-
"""Range downloads from official Toronto redirect target; final MD5 mandatory."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import time
import re

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'data/cifar'
SPECS=[('cifar-100-python.tar.gz',169001437,'eb9058c3a382ffc7106e4002c42a8d85'),
       ('cifar-10-python.tar.gz',170498071,'c58f30108f718f92721af3b95e74349a')]

def main():
    chunk=2*1024*1024
    jobs=[]
    for name,total,md5 in SPECS:
        dest=BASE/name
        if dest.exists() and dest.stat().st_size==total:
            assert hashlib.md5(dest.read_bytes()).hexdigest()==md5
            continue
        parts=BASE/(name+'.parts')
        parts.mkdir(exist_ok=True)
        for start in range(0,total,chunk):
            end=min(total-1,start+chunk-1)
            jobs.append((name,start,end,parts/f'{start:012d}.bin'))
    def get(job):
        name,start,end,p=job
        if p.exists() and p.stat().st_size==end-start+1:
            return
        for attempt in range(8):
            have=p.stat().st_size if p.exists() else 0
            request_start=start+have
            tail=p.with_suffix('.tail')
            header=p.with_suffix('.headers')
            args=['curl.exe','-L','--fail','--silent','--show-error','--connect-timeout','30','--max-time','600','--tlsv1.2','--tls-max','1.2',
                  '--range',f'{request_start}-{end}','https://cave.cs.toronto.edu/kriz/'+name,'-D',str(header),'-o',str(tail)]
            res=subprocess.run(args,capture_output=True)
            headers=header.read_text(errors='replace') if header.exists() else ''
            valid_range=re.search(r'(?im)^content-range:\s*bytes\s+'+str(request_start)+'-'+str(end)+r'/\d+',headers)
            if valid_range and tail.exists() and 0<tail.stat().st_size<=end-request_start+1:
                with open(p,'ab') as target:
                    target.write(tail.read_bytes())
            if p.exists() and p.stat().st_size==end-start+1:
                return
            print(json.dumps({'retry':attempt,'file':name,'start':start,'stderr':res.stderr.decode(errors='replace')}),flush=True)
            time.sleep(min(5*(attempt+1),30))
        raise RuntimeError(str(job))
    t=time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(get,job) for job in jobs]
        for n,f in enumerate(concurrent.futures.as_completed(futures),1):
            f.result()
            print(json.dumps({'chunks_complete':n,'chunks_total':len(jobs),'elapsed_s':time.perf_counter()-t}),flush=True)
    for name,total,md5 in SPECS:
        dest=BASE/name
        if dest.exists() and dest.stat().st_size==total:
            assert hashlib.md5(dest.read_bytes()).hexdigest()==md5
            continue
        parts=BASE/(name+'.parts')
        with open(dest.with_suffix('.assembling'),'wb') as out:
            for start in range(0,total,chunk):
                out.write((parts/f'{start:012d}.bin').read_bytes())
        temp=dest.with_suffix('.assembling')
        assert temp.stat().st_size==total
        assert hashlib.md5(temp.read_bytes()).hexdigest()==md5
        temp.replace(dest)
        print('VERIFIED',name,md5,flush=True)

if __name__=='__main__':
    main()
