# -*- coding: utf-8 -*-
"""Exercise report generation with explicitly synthetic fixtures outside formal results."""
import importlib.util
import json
from pathlib import Path
import sys
import contextlib

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_diagnostic import dump,METRICS

def main():
    dest=ROOT/'results/synthetic_checks/summary_fixture'
    (dest/'results/full').mkdir(parents=True,exist_ok=True)
    (dest/'README.txt').write_text('ALL DATA HERE ARE SYNTHETIC SOFTWARE CHECK FIXTURES, NOT EXPERIMENTAL RESULTS.',encoding='utf-8')
    dump(dest/'results/audit.json',{'passed':True,'formal_runs':30})
    (dest/'results/protocol.json').write_bytes((ROOT/'results/protocol.json').read_bytes())
    dump(dest/'results/source_inventory.json',[])
    dump(dest/'results/feature_manifest.json',{})
    methods=['KM','BAL','P2OT_feat','ORACLE_KM','LP']
    for ds,k in [('cifar100',100),('cifar10',10)]:
        for ir in [1,10,100]:
            counts=[int((50000/k)*(1/ir)**(c/(k-1))) for c in range(k)]
            for seed in range(1,6):
                r={'dataset':ds,'ir':ir,'seed':seed,'counts':counts,'n':sum(counts),'methods':{}}
                for method in methods:
                    value=({'KM':.6,'BAL':.7,'P2OT_feat':.8,'ORACLE_KM':.85,'LP':.9}[method] if ir==100 else .9)
                    m={name:value for name in METRICS}
                    m.update({name:0 for name in ['KL_true_to_pred','JS_true_to_pred','JS_sorted','JS_pred_to_uniform','empty_clusters']})
                    m.update(normalized_entropy=1,true_counts=counts,raw_cluster_counts=counts,aligned_cluster_counts=counts,
                        recall_per_class=[value]*k,mapping=list(range(k)))
                    r['methods'][method]={'metrics':m}
                p=dest/f'results/full/{ds}_ir{ir}_s{seed}'
                p.mkdir(exist_ok=True)
                dump(p/'result.json',r)
    spec=importlib.util.spec_from_file_location('summary_under_test',ROOT/'scripts/summarize_results.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT=dest
    with open(dest/'generator.stdout.log','w',encoding='utf-8') as log,contextlib.redirect_stdout(log):
        module.main()
    summary=json.loads((dest/'results/summary.json').read_text(encoding='utf-8'))
    assert summary['verdict']=='通过' and abs(summary['long_tail_loss']['mean']-.2)<1e-12
    assert abs(summary['room']['mean']-.15)<1e-12 and summary['room']['positive']==5
    assert (dest/'第一阶段长尾聚类可行性报告.md').exists()
    dump(ROOT/'results/synthetic_checks/summary_check.json',{'passed':True,'all_inputs_synthetic':True,'primary_best_branch_and_threshold_logic':True,'report_generated_in_synthetic_fixture_only':True})
    print('Synthetic report generation and paired primary decision checks passed; not real results',flush=True)

if __name__=='__main__':
    main()
