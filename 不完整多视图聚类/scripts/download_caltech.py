"""Range-resumable official data retrieval with Git blob verification."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib,time,urllib.request,json
ROOT=Path(__file__).resolve().parents[1]
SHA='624dd9fd10431802214fab1d7b34c44f5a0a92df'
URL=f'https://raw.githubusercontent.com/XLearning-SCU/2021-CVPR-Completer/{SHA}/data/Caltech101-20.mat'
TOTAL=36482487;STEP=512*1024
parts=ROOT/'results/download_parts';parts.mkdir(exist_ok=True)
def get(start):
    end=min(TOTAL-1,start+STEP-1);p=parts/f'{start:09d}.part'
    if p.exists() and p.stat().st_size==end-start+1:return p
    for attempt in range(12):
        try:
            req=urllib.request.Request(URL+f'?part={start}',headers={'Range':f'bytes={start}-{end}'})
            with urllib.request.urlopen(req,timeout=30) as r:
                assert r.status==206 and r.headers['Content-Range']==f'bytes {start}-{end}/{TOTAL}',dict(r.headers)
                b=r.read()
            assert len(b)==end-start+1
            p.write_bytes(b);print(f'part {start}-{end} complete',flush=True);return p
        except Exception as exc:
            print(f'retry {start} {attempt} {exc!r}',flush=True);time.sleep(min(attempt+1,5))
    raise RuntimeError(f'Part failed {start}')
with ThreadPoolExecutor(max_workers=8) as pool:pieces=list(pool.map(get,range(0,TOTAL,STEP)))
b=b''.join(p.read_bytes() for p in pieces)
assert len(b)==TOTAL
blob=hashlib.sha1(f'blob {len(b)}\0'.encode()+b).hexdigest()
assert blob=='f61831f54343b90b1adc8da218a98ee855fe8974',blob
p=ROOT/'third_party/2021-CVPR-Completer/data/Caltech101-20.mat';p.write_bytes(b)
(ROOT/'results/caltech_download.json').write_text(json.dumps(dict(url=URL,commit=SHA,git_blob_sha1=blob,sha256=hashlib.sha256(b).hexdigest(),bytes=len(b)),indent=2),encoding='utf-8')
print('VERIFIED',len(b),blob,flush=True)
