import concurrent.futures
import json
from download_resources import ROOT, fetch

jobs = {
    'references/fp_README.md': 'https://raw.githubusercontent.com/twitter-research/feature-propagation/main/README.md',
    'references/fp_official.py': 'https://raw.githubusercontent.com/twitter-research/feature-propagation/af2733589eab4023fca67f7e71a3b46ddbbea8cd/src/feature_propagation.py',
    'references/sdcn.pdf': 'https://arxiv.org/pdf/2002.01633',
    'references/sdcn_official.py': 'https://raw.githubusercontent.com/bdy9527/SDCN/master/sdcn.py',
    'references/sdcn_GNN.py': 'https://raw.githubusercontent.com/bdy9527/SDCN/master/GNN.py',
    'references/sdcn_pretrain.py': 'https://raw.githubusercontent.com/bdy9527/SDCN/master/data/pretrain.py',
    'references/amgc_search.json': 'https://api.github.com/search/repositories?q=AMGC+clustering',
}
for suffix in ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']:
    jobs['data/planetoid/ind.cora.'+suffix] = 'https://raw.githubusercontent.com/kimiyoung/planetoid/master/data/ind.cora.'+suffix
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
    results=list(pool.map(fetch, jobs.items()))
(ROOT/'supplement_manifest.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
for row in results:
    print(json.dumps(row),flush=True)
