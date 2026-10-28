"""Hash stage-five code, protocol, logs, and results after finalization."""
from pathlib import Path
import hashlib
import json


root = Path(__file__).resolve().parent
names = ['stage5_plan.json', 'run_sdcn_stage5_weighted.py',
         'run_sdcn_stage5_baseline.py', 'run_confidence_stage5.py',
         'summarize_stage5.py', 'finalize_stage5.py',
         'stage5_summary.json', '第五阶段验证报告.md',
         'sdcn_stage5_reconstruction.stdout.log',
         'sdcn_stage5_reconstruction.stderr.log',
         'sdcn_stage5_degree_baseline.stdout.log',
         'sdcn_stage5_degree_baseline.stderr.log']
folders = ['sdcn_stage5_reconstruction', 'sdcn_stage5_degree_baseline',
           'confidence_stage5_degree']
paths = [root / name for name in names]
for folder in folders:
    paths.extend((root / folder).rglob('*'))
items = []
for path in sorted(p for p in paths if p.is_file()):
    data = path.read_bytes()
    items.append(dict(path=str(path.relative_to(root)).replace('\\', '/'),
                      bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
(root / 'stage5_manifest.json').write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding='utf-8')
print('Stage 5 files:', len(items))
