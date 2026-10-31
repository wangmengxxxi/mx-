import json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'results_stage3'
METHODS = ['GMM_diag', 'OVER_MERGE', 'HDBSCAN_error', 'HDBSCAN_nearest']

def main():
    status = json.loads((R / 'pipeline_status.json').read_text(encoding='utf8'))
    if status['state'] != 'complete':
        raise RuntimeError('Full pipeline is not complete')
    audit = {'passed': True, 'condition': 'cifar100 IR100 seed1', 'methods': {}}
    for method in METHODS:
        args = [sys.executable, '-I', str(ROOT / 'scripts/run_stage3.py'), '--dataset', 'cifar100',
                '--ir', '100', '--seed', '1', '--method', method, '--out', 'results_stage3/rerun']
        log = R / 'logs' / f'rerun_cifar100_ir100_s1_{method}.log'
        with log.open('w', encoding='utf8') as stream:
            code = subprocess.run(args, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT).returncode
        if code:
            audit['passed'] = False
            audit['methods'][method] = {'passed': False, 'error': f'exit {code}', 'log': str(log)}
            break
        rel = f'cifar100_ir100_s1/{method}_k100/result.json'
        original = json.loads((R / 'full' / rel).read_text(encoding='utf8'))
        rerun = json.loads((R / 'rerun' / rel).read_text(encoding='utf8'))
        metrics_equal = original['metrics'] == rerun['metrics']
        fit_original = original['fit'].copy()
        fit_rerun = rerun['fit'].copy()
        if method == 'HDBSCAN_nearest':
            fit_original.pop('raw_noise', None)
            fit_rerun.pop('raw_noise', None)
        fit_equal = fit_original == fit_rerun
        hashes_equal = original['indices_array_sha256'] == rerun['indices_array_sha256']
        prediction_equal = original.get('prediction_sha256') == rerun.get('prediction_sha256') if original.get('prediction_sha256') else None
        passed = metrics_equal and fit_equal and hashes_equal and prediction_equal is not False
        audit['methods'][method] = {'passed': passed, 'metrics_equal': metrics_equal,
                                    'fit_equal': fit_equal, 'indices_equal': hashes_equal,
                                    'prediction_equal': prediction_equal,
                                    'prediction_hash_original': original.get('prediction_sha256'),
                                    'prediction_hash_rerun': rerun.get('prediction_sha256'),
                                    'BA_difference': rerun['metrics']['balanced_accuracy'] - original['metrics']['balanced_accuracy']}
        audit['passed'] &= passed
        if not passed:
            break
    (R / 'rerun_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    if not audit['passed']:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
