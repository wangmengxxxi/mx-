"""Download original public data and implementation references; never install packages."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent
JOBS = {
    'data/cora.tgz': 'https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz',
    'references/fp_README.md': 'https://raw.githubusercontent.com/twitter-research/feature-propagation/main/README.md',
    'references/fp_tree.json': 'https://api.github.com/repos/twitter-research/feature-propagation/git/trees/main?recursive=1',
    'references/sdcn_README.md': 'https://raw.githubusercontent.com/bdy9527/SDCN/master/README.md',
    'references/amgc_openreview.html': 'https://openreview.net/forum?id=hcohfNBI1c',
    'references/fp.pdf': 'https://arxiv.org/pdf/2111.12128',
    'references/sdcn.pdf': 'https://arxiv.org/pdf/2002.01633',
    'references/amgc.pdf': 'https://openreview.net/pdf?id=hcohfNBI1c',
}

def fetch(item):
    name, url = item
    p = ROOT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        if name.endswith('.pdf') and not r.content.startswith(b'%PDF'):
            raise ValueError('Response is not a PDF')
        p.write_bytes(r.content)
        return dict(path=name, url=url, final_url=r.url, bytes=len(r.content),
                    sha256=hashlib.sha256(r.content).hexdigest(), status='ok')
    except Exception as e:
        return dict(path=name, url=url, status='failed', error=str(e))

if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(fetch, JOBS.items()))
    (ROOT / 'download_manifest.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    for row in results:
        print(json.dumps(row, ensure_ascii=True), flush=True)
