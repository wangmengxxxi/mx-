"""Audit completed output without altering results from earlier stages."""
import hashlib, importlib.metadata, importlib.util, json, math, platform
from datetime import datetime
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'results_stage3'
spec = importlib.util.spec_from_file_location('pipeline_stage3', ROOT/'scripts/pipeline_stage3.py')
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)
def read(p): return json.loads(p.read_text(encoding='utf8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def finite(obj):
    if isinstance(obj, float): assert math.isfinite(obj)
    elif isinstance(obj, dict):
        for v in obj.values(): finite(v)
    elif isinstance(obj, list):
        for v in obj: finite(v)

def main():
    protocol_hash = (R/'protocol_stage3.sha256').read_text().split()[0].lower()
    assert sha(R/'protocol_stage3.json') == protocol_hash
    manifest = read(ROOT/'results/feature_manifest.json')
    features = []
    for name, item in manifest.items():
        for key, suffix in [('sha256',''),('labels_sha256','_labels')]:
            p = ROOT/f'data/features/{name}{suffix}.npy'
            actual = sha(p)
            assert actual == item[key], str(p)
            features.append({'path':str(p.relative_to(ROOT)), 'sha256':actual})
    records=[]
    for ds,ir,seed,method,kp in pipeline.jobs():
        if method == 'DeepDPM': continue
        k=10 if ds=='cifar10' else 100
        path=R/f'full/{ds}_ir{ir}_s{seed}/{method}_k{kp or k}/result.json'
        item=read(path); base=read(ROOT/f'results/full/{ds}_ir{ir}_s{seed}/result.json')
        assert item['protocol_sha256']==protocol_hash, str(path)
        assert item['indices_array_sha256']==base['indices_array_sha256'], str(path)
        assert item['n']==base['n'], str(path)
        assert item['elapsed_s']>0 and item['peak_memory_bytes']>0, str(path)
        finite(item)
        for key in ['ACC','balanced_accuracy','tail_recall','stage1_tail_recall','many_to_one_BA','many_to_one_tail_recall','tail_coverage','normalized_entropy']:
            assert 0 <= item['metrics'][key] <= 1+1e-10, (path,key)
        m=item['metrics']
        if 'recall_per_class' in m:
            assert abs(np.mean(m['recall_per_class'])-m['balanced_accuracy'])<1e-12
            assert sum(m['cluster_counts'])==item['n']
        records.append({'path':str(path.relative_to(ROOT)), 'sha256':sha(path),
                        'indices_array_sha256':item['indices_array_sha256'],
                        'recorded_code_sha256':item.get('code_sha256'),
                        'recorded_prediction_sha256':item.get('prediction_sha256')})
    assert len(records)==215
    versions={name:importlib.metadata.version(name) for name in ['torch','numpy','scipy','scikit-learn','igraph','leidenalg']}
    audit={'passed':True,'audited_at':datetime.now().astimezone().isoformat(),
           'protocol_sha256':protocol_hash,'features':features,'result_count':len(records),
           'result_files':records,'python':platform.python_version(),'platform':platform.platform(),
           'versions':versions,'scripts':{p.name:sha(p) for p in (ROOT/'scripts').glob('*stage3*.py')},
           'limitations':['Early full results did not record code/prediction hashes; no retrospective hashes invented.',
                         'Older HDBSCAN_nearest fit.raw_noise was overwritten through array aliasing; use paired HDBSCAN_error raw_noise.',
                         'Protocol frozen_at is a manually entered timestamp, not independent evidence of freeze time.',
                         'Protocol omitted some implementation details; no hyperparameter search was performed.']}
    (R/'integrity_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(f'PASS: {len(features)} feature files; {len(records)} full results; matching protocol/index hashes; finite metrics')

if __name__=='__main__': main()
