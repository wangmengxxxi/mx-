"""Hash the final stage-four plan, code, logs, and experimental evidence."""
from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parent
names = ['stage4_plan.json', 'run_amgc_stage3.py', 'run_sdcn_baseline.py',
         'run_sdcn_weighted.py', 'run_confidence_probe.py',
         'summarize_confidence_bins.py', 'summarize_stage4.py',
         'finalize_stage4.py',
         'stage4_summary.json', '第四阶段验证报告.md',
         'amgc_stage4_full1000.stdout.log', 'amgc_stage4_full1000.stderr.log',
         'sdcn_stage4_holdout.stdout.log', 'sdcn_stage4_holdout.stderr.log',
         'sdcn_stage4_weighted.stdout.log', 'sdcn_stage4_weighted.stderr.log']
folders = ['amgc_stage4_full1000', 'sdcn_stage4_holdout',
           'sdcn_stage4_weighted', 'confidence_stage4_holdout']
paths = [root / name for name in names]
for folder in folders:
    paths.extend((root / folder).rglob('*'))
items = []
for path in sorted(p for p in paths if p.is_file()):
    data = path.read_bytes()
    items.append(dict(path=str(path.relative_to(root)).replace('\\', '/'),
                      bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
(root / 'stage4_manifest.json').write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding='utf-8')
print('Stage 4 files:', len(items))
