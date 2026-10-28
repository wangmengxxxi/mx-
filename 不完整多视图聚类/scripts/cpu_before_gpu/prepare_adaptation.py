"""Copy reviewed baseline source, preserving originals and recording exact diffs."""
from pathlib import Path
import argparse, difflib, hashlib, json

ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args()
root=args.root.resolve();source=root/'third_party/2021-CVPR-Completer';target=root/'scripts/completer'
target.mkdir(exist_ok=True)
patch=[];records=[]
for name in ['model.py','loss.py','get_mask.py','configure.py','util.py','evaluation.py','datasets.py']:
    old=(source/name).read_text(encoding='utf-8');new=old
    if name=='get_mask.py': new=new.replace('np.int)', 'int)')
    if name=='datasets.py':
        new=new.replace("main_dir = sys.path[0]", "main_dir = config['data_root']  # explicit reviewed data root; never third_party cwd")
        new=new.replace("os.path.join(main_dir, 'data', ", "os.path.join(main_dir, ")
    with (target/name).open('w',encoding='utf-8',newline='\n') as f: f.write(new)
    patch.extend(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile=f'third_party/2021-CVPR-Completer/{name}',tofile=f'scripts/completer/{name}'))
    records.append(dict(file=name,source_sha256=hashlib.sha256((source/name).read_bytes()).hexdigest(),copy_sha256=hashlib.sha256((target/name).read_bytes()).hexdigest(),changed=old!=new))
(root/'scripts/completer_compatibility.patch').write_text(''.join(patch),encoding='utf-8')
(root/'results/adaptation_manifest.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
print('Copied reviewed modules. Only np.int compatibility and explicit data path were changed.')
