# -*- coding: utf-8 -*-
"""Single pass, resumable at split/batch boundaries; never overwrites completed features."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['XFORMERS_DISABLED'] = '1'
import sys
import json
import hashlib
import time
import tarfile
from pathlib import Path
import numpy as np
import torch
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'third_party/dinov2_py39'))
from dinov2.models.vision_transformer import vit_base

def sha(p, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def dump(p, v):
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(v, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    tmp.replace(p)

def event(**v):
    v['time'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    with open(ROOT / 'results/extraction_events.jsonl', 'a', encoding='utf-8') as f:
        f.write(json.dumps(v)+'\n')
    print(json.dumps(v), flush=True)

def main():
    assert sha(ROOT/'results/protocol.json') == (ROOT/'results/protocol.sha256').read_text().split()[0]
    torch.set_num_threads(4)
    torch.manual_seed(20261027)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sources = []
    for cls in [datasets.CIFAR100, datasets.CIFAR10]:
        archive = ROOT/'data/cifar'/cls.filename
        assert sha(archive,'md5') == cls.tgz_md5, ('archive MD5 failure',archive)
        entry={'name':cls.__name__, 'url':cls.url, 'path':str(archive.relative_to(ROOT)), 'sha256':sha(archive),'md5':cls.tgz_md5}
        cache_path=ROOT/'results/cache_reuse.json'
        if cache_path.exists():
            cache=json.loads(cache_path.read_text(encoding='utf-8'))
            if cls is datasets.CIFAR10:
                entry['acquisition']='reused existing official torchvision archive, verified against built-in MD5; read-only source'
                entry['cache_record']=cache['cifar10']
        sources.append(entry)
        if not (ROOT/'data/cifar'/cls.base_folder).exists():
            with tarfile.open(archive) as tar:
                base = (ROOT/'data/cifar').resolve()
                for m in tar.getmembers():
                    target = (base/m.name).resolve()
                    assert base in target.parents and not m.issym() and not m.islnk()
                tar.extractall(base)
    sources.append({'name':'DINOv2 ViT-B/14','url':json.loads((ROOT/'results/protocol.json').read_text(encoding='utf-8'))['features']['weights_url'],
        'path':'data/dinov2_vitb14_pretrain.pth','sha256':sha(ROOT/'data/dinov2_vitb14_pretrain.pth')})
    for name in ['dinov2','ppot']:
        commit=json.loads((ROOT/f'results/{name}_commit.json').read_text(encoding='utf-8'))['sha']
        repo='facebookresearch/dinov2' if name=='dinov2' else 'rhfeiyang/PPOT'
        sources.append({'name':repo,'url':f'https://codeload.github.com/{repo}/zip/{commit}', 'commit':commit,
            'sha256':sha(ROOT/f'third_party/{name}.zip')})
    dump(ROOT/'results/source_inventory.json',sources)
    # Same constructor arguments as official hub dinov2_vitb14; local weights only.
    model = vit_base(img_size=518,patch_size=14,init_values=1.0,block_chunks=0)
    state=torch.load(ROOT/'data/dinov2_vitb14_pretrain.pth',map_location='cpu',weights_only=True)
    model.load_state_dict(state,strict=True)
    del state
    model.requires_grad_(False).eval().half().cuda()
    transform=transforms.Compose([transforms.Resize(224,interpolation=transforms.InterpolationMode.BICUBIC),transforms.CenterCrop(224),
        transforms.ToTensor(),transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])
    batch=32
    for candidate in [32,16,8,4,2,1]:
        try:
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                z=model(torch.zeros(candidate,3,224,224,device='cuda',dtype=torch.float16))
            assert z.shape==(candidate,768) and torch.isfinite(z).all()
            torch.cuda.synchronize()
            batch=candidate
            event(stage='memory_probe',batch=batch,peak_bytes=torch.cuda.max_memory_allocated())
            del z
            break
        except torch.cuda.OutOfMemoryError:
            event(stage='memory_probe_oom',batch=candidate)
            torch.cuda.empty_cache()
    manifest_path=ROOT/'results/feature_manifest.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for cls,name in [(datasets.CIFAR100,'cifar100'),(datasets.CIFAR10,'cifar10')]:
        for train,split in [(True,'train'),(False,'test')]:
            key=f'{name}_{split}'
            path=ROOT/f'data/features/{key}.npy'
            labels_path=ROOT/f'data/features/{key}_labels.npy'
            if key in manifest:
                assert path.exists() and sha(path)==manifest[key]['sha256']
                event(stage='skip_completed',split=key)
                continue
            ds=cls(str(ROOT/'data/cifar'),train=train,transform=transform,download=False)
            labels=np.asarray(ds.targets,dtype=np.int64)
            np.save(labels_path,labels)
            temp=path.with_suffix('.partial.npy')
            progress=path.with_suffix('.progress.json')
            offset=json.loads(progress.read_text())['completed'] if progress.exists() else 0
            feats=np.lib.format.open_memmap(temp,mode='r+' if temp.exists() else 'w+',dtype=np.float32,shape=(len(ds),768))
            # Subset retains original order. Workers receive no model and perform image preprocessing only.
            remaining=torch.utils.data.Subset(ds,range(offset,len(ds)))
            loader=torch.utils.data.DataLoader(remaining,batch_size=batch,shuffle=False,num_workers=2,pin_memory=True,persistent_workers=True)
            started=time.perf_counter()
            last_event=0
            event(stage='start_split',split=key,completed=offset,total=len(ds),batch=batch)
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                for images,_ in loader:
                    z=model(images.cuda(non_blocking=True).half()).float().cpu().numpy()
                    assert np.isfinite(z).all()
                    feats[offset:offset+len(z)]=z
                    offset+=len(z)
                    if time.perf_counter()-last_event >=30 or offset==len(ds):
                        feats.flush()
                        dump(progress,{'completed':offset,'batch':batch})
                        event(stage='progress',split=key,completed=offset,total=len(ds),elapsed_s=time.perf_counter()-started)
                        last_event=time.perf_counter()
            del loader,feats
            temp.replace(path)
            manifest[key]={'path':str(path.relative_to(ROOT)),'shape':[len(ds),768],'dtype':'float32','inference_dtype':'float16',
                'sha256':sha(path),'labels_sha256':sha(labels_path),'batch':batch,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                'elapsed_this_process_s':time.perf_counter()-started,'torch':torch.__version__,'device':torch.cuda.get_device_name(0)}
            dump(manifest_path,manifest)
            event(stage='complete_split',split=key,**manifest[key])
    event(stage='all_features_complete')

if __name__=='__main__':
    main()
