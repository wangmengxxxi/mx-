"""Create an editable source copy and a new PyG cache from archived raw Cora."""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
source = ROOT / 'references/amgc_source/AMGC-master'
target = ROOT / 'amgc_work'
target.mkdir(exist_ok=False)
for name in ['models', 'trains', 'utils', 'configs']:
    shutil.copytree(source / name, target / name)
for name in ['main.py', 'README.md', 'LICENSE']:
    shutil.copy2(source / name, target / name)
raw = ROOT / 'amgc_data/Cora/raw'
raw.mkdir(parents=True, exist_ok=False)
for file in (ROOT / 'data/planetoid').glob('ind.cora.*'):
    shutil.copy2(file, raw / file.name)
print(target, raw)
