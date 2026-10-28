# -*- coding: utf-8 -*-
"""Use an isolated local proxy instance from existing user config; no config/credentials logged."""
import concurrent.futures
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import time
import urllib.request
import urllib.parse
import base64

ROOT=Path(__file__).resolve().parents[1]

def main():
    original=Path(os.environ['APPDATA'])/'io.github.clash-verge-rev.clash-verge-rev'
    runtime=ROOT/'results/proxy_runtime'
    runtime.mkdir(exist_ok=True)
    for name in ['Country.mmdb','ASN.mmdb','geoip.dat','geosite.dat']:
        p=original/name
        if p.exists() and not (runtime/name).exists():
            shutil.copyfile(p,runtime/name)
    text=(original/'clash-verge.yaml').read_text(encoding='utf-8')
    for key,value in [('mixed-port','7891'),('port','0'),('socks-port','0'),('redir-port','0'),('tproxy-port','0'),('mode','global'),('allow-lan','false')]:
        text=re.sub(r'(?m)^'+key+r':[^\n]*$',key+': '+value,text)
    for key in ['tun','dns']:
        text=re.sub(r'(?m)^'+key+r':[^\n]*\n(?:[ \t]+[^\n]*\n|\n)*',key+':\n  enable: false\n',text)
    secret=secrets.token_hex(24)
    config=base64.b64encode(text.encode()).decode()
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def api(path,method='GET',body=None,timeout=10):
        req=urllib.request.Request('http://127.0.0.1:9091'+path,data=None if body is None else json.dumps(body).encode(),method=method,
            headers={'Authorization':'Bearer '+secret,'Content-Type':'application/json'})
        with opener.open(req,timeout=timeout) as r:
            b=r.read()
        return json.loads(b) if b else None
    out=open(runtime/'core.log','w',encoding='utf-8')
    core=subprocess.Popen(['F:/clashVerge/verge-mihomo.exe','-d',str(runtime),'-f',str(original/'clash-verge.yaml'),'-ext-ctl','127.0.0.1:9091','-secret',secret],
        stdout=out,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        for _ in range(20):
            try:
                proxies=api('/proxies')['proxies']
                break
            except Exception:
                time.sleep(.5)
        else:
            raise RuntimeError('isolated proxy did not start')
        api('/configs','PATCH',{'mixed-port':7891,'port':0,'socks-port':0,'redir-port':0,'tproxy-port':0,'mode':'global','allow-lan':False,'tun':{'enable':False}})
        candidates=[name for name,entry in proxies.items() if entry['type'] not in ['Selector','URLTest','Fallback','LoadBalance','Direct','Reject','Compatible','Pass']]
        print(json.dumps({'proxy_candidates':len(candidates),'types':sorted(set(e['type'] for e in proxies.values()))}),flush=True)
        def probe(name):
            try:
                result=api('/proxies/'+urllib.parse.quote(name,safe='')+'/delay?'+urllib.parse.urlencode({'url':'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz','timeout':6000}),timeout=8)
                return name,result.get('delay',0)
            except Exception:
                return name,0
        selected=None
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for name,delay in pool.map(probe,candidates):
                print(json.dumps({'probe_delay_ms':delay,'available':delay>0}),flush=True)
                if delay>0 and selected is None:
                    selected=name
        if selected is None:
            raise RuntimeError('No configured proxy node can reach official Toronto URL')
        api('/proxies/GLOBAL','PUT',{'name':selected})
        print('Isolated proxy node selected; downloading official archives only',flush=True)
        specs=[('cifar-100-python.tar.gz',169001437,'eb9058c3a382ffc7106e4002c42a8d85'),('cifar-10-python.tar.gz',170498071,'c58f30108f718f92721af3b95e74349a')]
        import hashlib
        for name,total,md5 in specs:
            p=ROOT/'data/cifar'/name
            args=['curl.exe','-L','-C','-','--proxy','http://127.0.0.1:7891','--retry','3','--retry-delay','5','--retry-all-errors','--fail','--silent','--show-error',
                '--connect-timeout','30','--max-time','1200','https://www.cs.toronto.edu/~kriz/'+name,'-o',str(p)]
            subprocess.run(args,check=True)
            assert p.stat().st_size==total and hashlib.md5(p.read_bytes()).hexdigest()==md5
            print('VERIFIED',name,flush=True)
    finally:
        core.terminate()
        core.wait(timeout=15)
        out.close()

if __name__=='__main__':
    main()
