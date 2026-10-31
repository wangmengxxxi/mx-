"""Fetch fixed, read-only official source snapshots without large model/data archives."""
import concurrent.futures, hashlib, json, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPOS = [('BGU-CS-VIL/DeepDPM', '95c9376ac1b0cb48ea017c52d3abbf2a901940b3', 'deepdpm'),
         ('ssarfraz/FINCH-Clustering', 'ab3b014ae30c0dd2eaa82ff37378e076ca21198a', 'finch')]

def main():
    manifest = []
    tasks = []
    for repo, commit, name in REPOS:
        tree = json.loads((ROOT / f'third_party/{name}_tree.json').read_text())['tree']
        for entry in tree:
            p = entry['path']
            if entry['type'] == 'blob' and (p.endswith('.py') or p in ('LICENSE','LICENSE.txt','README.md','requirements.txt','finch/requirements.txt')):
                tasks.append((repo, commit, name, entry))
    def fetch(task):
        repo, commit, name, entry = task
        path = entry['path']
        target = ROOT / 'third_party' / f'{name}-source-{commit}' / path
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        data = target.read_bytes() if target.exists() else opener.open(f'https://raw.githubusercontent.com/{repo}/{commit}/{path}', timeout=30).read()
        blob = hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
        assert blob == entry['sha'], path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return {'repo':repo,'commit':commit,'path':path,'git_blob':blob,'sha256':hashlib.sha256(data).hexdigest()}
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for item in pool.map(fetch, tasks): manifest.append(item)
    out = ROOT / 'results_stage3/source_manifest.json'
    out.write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf8')
    print(f'Verified {len(manifest)} source files against fixed Git tree blobs')

if __name__ == '__main__': main()
