"""Hash stage 3 evidence while excluding the inherited dependency environment."""
from pathlib import Path
import hashlib, json
root = Path(__file__).resolve().parent
excluded = {'envs', '__pycache__'}
items=[]
for p in sorted(root.rglob('*')):
    if not p.is_file() or any(part in excluded for part in p.relative_to(root).parts):
        continue
    if p.name == Path(__file__).name or p.name == 'stage3_manifest.json':
        continue
    data=p.read_bytes()
    items.append({'path':str(p.relative_to(root)).replace('\\','/'),'bytes':len(data),
                  'sha256':hashlib.sha256(data).hexdigest()})
(root/'stage3_manifest.json').write_text(json.dumps(items,ensure_ascii=False,indent=2),encoding='utf-8')
print('files',len(items))
