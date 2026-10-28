# -*- coding: utf-8 -*-
"""Run once, before feature extraction or any metric evaluation."""
import datetime
import difflib
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def dump(p, v):
    p.write_text(json.dumps(v, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def main():
    out = ROOT / 'results/protocol.json'
    if out.exists():
        raise RuntimeError('Protocol already frozen; do not overwrite')
    dino = next((ROOT / 'third_party').glob('dinov2-*'))
    ppot = next((ROOT / 'third_party').glob('PPOT-*'))
    compat = ROOT / 'third_party/dinov2_py39'
    shutil.copytree(dino / 'dinov2', compat / 'dinov2')
    patches = []
    for name in ['attention.py', 'block.py']:
        p = compat / 'dinov2/layers' / name
        old = p.read_text(encoding='utf-8')
        new = 'from __future__ import annotations\n' + old
        p.write_text(new, encoding='utf-8')
        patches.extend(difflib.unified_diff(old.splitlines(True), new.splitlines(True), fromfile=name, tofile=name))
    (ROOT / 'results/dinov2_py39.patch').write_text(''.join(patches), encoding='utf-8')
    protocol = {
        'frozen_at': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        'purpose': 'diagnosis only; no new method; no label-based hyperparameter or checkpoint selection',
        'datasets': {'cifar10': 10, 'cifar100': 100}, 'irs': [1, 10, 100], 'seeds': [1, 2, 3, 4, 5],
        'features': {'backbone': 'official DINOv2 ViT-B/14 without registers', 'dim': 768,
            'weights_url': 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth',
            'dino_commit': dino.name.split('-', 1)[1], 'frozen': True, 'inference': 'model.half + input.half; eval; inference_mode',
            'output': 'final LayerNorm CLS token, saved float32 (fp16 values)',
            'preprocess': {'resize': 224, 'center_crop': 224, 'interpolation': 'PIL bicubic', 'mean': [0.485,0.456,0.406], 'std': [0.229,0.224,0.225]},
            'preprocess_note': 'user explicitly requests resize224; official eval defaults resize256/crop224, overridden resize_size=224. Square CIFAR images retain full image.',
            'extraction_order': ['cifar100_train', 'cifar100_test', 'cifar10_train', 'cifar10_test'],
            'batch_candidates_memory_only': [32,16,8,4,2,1], 'workers': 2, 'extract_once_reuse': True},
        'subset': {'formula': 'int((50000/K) * (1/IR)**(c/(K-1))), c=0..K-1',
            'class_order': 'torchvision label id ascending, fixed across all seeds and IRs',
            'rng': 'numpy RandomState(seed), independently permute each class pool once in class order; take prefix; identical permutation across IRs',
            'tail': 'floor(K/3) classes with smallest counts; ties by class id ascending',
            'setting': 'transductive: fit and evaluate on same selected train subset; test features never used'},
        'methods': {
            'KM': {'init': 'k-means++', 'n_init': 10, 'max_iter': 300, 'tol': 0.0001, 'algorithm': 'lloyd', 'random_state': 'seed', 'features': 'L2 normalized float32'},
            'BAL': {'head': 'single cosine linear head, bias=False, 768->K, random torch Linear initialization, normalized weights each forward',
                'epochs': 100, 'batch': 1024, 'last_batch': 'included', 'optimizer': 'Adam', 'lr': 0.0005, 'betas': [0.9,0.999], 'weight_decay': 0,
                'scheduler': 'cosine lr=base*0.5*(1+cos(pi*epoch/100))', 'logit_scale': 10,
                'sinkhorn': 'uniform marginals, log-domain, 3 row/column normalization iterations, epsilon=0.05 on cosine similarities',
                'loss': 'mean soft-target CE on logits=10*cosine, stop-gradient assignment; same cached feature as input and target; no augmentation or backbone tuning',
                'checkpoint': 'final epoch 100 only', 'features': 'same L2 features as KM', 'labels_received': False},
            'P2OT_feat': {'included': True, 'ppot_commit': ppot.name.split('-',1)[1],
                'solver': 'unmodified official utils/sinkhorn_knopp.py SemiCurrSinkhornKnopp loaded by absolute importlib path',
                'epsilon': 0.1, 'gamma': 1.0, 'stoperr': 1e-6, 'numItermax': 1000, 'semi_use': True,
                'rho': '0.1+0.9*exp(-5*(1-step/total_steps)**2), step=0..total_steps-1, official sigmoid schedule',
                'wrapper': 'same head, initialization, permutation, optimizer, epochs, final checkpoint as BAL; official solver replaces uniform Sinkhorn; soft CE retains partial target mass',
                'adaptation_limits': 'one head; no logits memory bank, no image multiview, no backbone tuning; module adaptation, not paper reproduction',
                'on_nonfinite': 'stop and preserve failure; no tuning or deleting runs'},
            'ORACLE_KM': {'centers': 'mean of L2 normalized features by true class within current subset, NOT renormalized',
                'assignment': 'nearest centroid squared Euclidean, zero Lloyd updates', 'caveat': 'operational supervised reference, not mathematical upper bound'},
            'LP': {'implementation': 'sklearn LogisticRegression', 'C': 1.0, 'solver': 'lbfgs', 'max_iter': 1000, 'tol': 0.0001,
                'multi_class': 'multinomial', 'class_weight': None, 'fit_intercept': True, 'random_state': 'seed', 'features': 'same L2 normalized features',
                'evaluation': 'resubstitution on same selected subset, supervised representation reference only; not generalization'}},
        'evaluation': {'metrics': ['ACC','NMI','ARI','balanced_accuracy','macro_F1','tail_recall'],
            'mapping': 'one global Hungarian mapping maximizing total correct sample count per run; shared by ACC and all subgroup metrics; NMI/ARI permutation invariant; LP also same global mapping',
            'NMI_average_method': 'arithmetic', 'zero_division': 0,
            'cluster_sizes': 'all K cluster ids including empty; Hungarian aligned counts + raw counts; per-class true counts',
            'distribution': 'KL(p_true||q_aligned) with additive count smoothing 1e-12; natural log; also JS(p_true,q_aligned), JS(sorted p,sorted q), JS(q,uniform), normalized entropy'},
        'decision': {'primary': ['cifar100',100,'balanced_accuracy'],
            'best': 'per-seed max(KM,BAL) in balanced_accuracy, offline diagnostic envelope, never deployment selection; excludes P2OT_feat',
            'long_tail_loss': 'mean_s(best(IR1,s)-best(IR100,s)) >= 0.08 AND strictly positive in >=4/5 seeds',
            'room': 'mean_s(ORACLE_KM(IR100,s)-best(IR100,s)) >=0.05 AND strictly positive in >=4/5 seeds',
            'classification': {'both': '通过', 'one': '部分通过', 'neither': '不通过'},
            'uniform_prior_harm': 'descriptive only: BAL-KM BA/tail mean<0, or BAL has smaller JS(q,uniform); all per-seed differences reported',
            'paired': 'same dataset, IR, seed; cross-IR uses same seed, nested class prefixes; zero not positive',
            'statistics': 'mean and population std ddof=0, no post-hoc thresholds or significance test; auxiliary CIFAR10 and IR10 do not alter primary'},
        'runtime': {'python': '3.9.21', 'torch': '2.8.0+cu126', 'threads': 4,
            'determinism': 'torch deterministic True; cudnn deterministic True benchmark False; TF32 disabled; CUBLAS_WORKSPACE_CONFIG=:4096:8',
            'device': 'single GTX1650Ti 4GB, sequential GPU jobs'},
        'audit': {'full_rerun': 'cifar100 IR100 seed1 all five methods in separate process using cached features',
            'recompute': 'all subset indices, predictions->metrics, counts and hashes; saved KM centers and heads->predictions; refit KM during full rerun',
            'rerun_tolerances': {'metrics_abs': 1e-6, 'head_float_atol': 1e-6, 'head_float_rtol': 1e-5},
            'feature_audit': 'finite shape/dtype and SHA256 all four splits; no second full extraction'},
        'label_firewall': 'labels used only for requested LT construction, offline evaluation and explicitly supervised ORACLE_KM/LP. KM/BAL/P2OT_feat training accept features,K,seed only.'}
    dump(out, protocol)
    (ROOT / 'results/protocol.sha256').write_text(sha(out)+'  protocol.json\n', encoding='utf-8')
    files = [p for base in [dino, ppot, compat] for p in base.rglob('*') if p.is_file()]
    dump(ROOT / 'results/third_party_manifest.json', {str(p.relative_to(ROOT)): sha(p) for p in files})
    (ROOT / 'results/安全审阅与来源说明.md').write_text('''# 安全审阅与来源说明

旧方向目录仅只读参考，没有修改。lt_gpu 从 imvc_gpu 克隆后仅在新环境安装 torchvision/Pillow。
代理 127.0.0.1:7890 未监听，官方 HTTPS 直连可用。官方提交 API 和压缩快照保存于 results/、third_party/，无第三方镜像。

DINOv2：已通读实际导入闭包 __init__.py、models/__init__.py、models/vision_transformer.py、layers 下全部 Python 文件；评估 transforms.py、hub/backbones.py/utils.py 作为配置来源也已阅读。没有运行 hub 入口或训练入口；直接构造官方 vit_base 并以 torch.load(weights_only=True) 加载本地权重，strict=True。原始文件不改动。官方新代码的 float|None 注解不兼容 Python3.9，在 third_party/dinov2_py39 两文件开头增加 future annotations；diff 已保存，不改变前向数学。

P²OT：已通读 utils/sinkhorn_knopp.py 全文、losses/losses.py、losses/ramps.py、models/classifer.py、utils/config.py，并阅读 train.py 到主训练与选择部分、README/LICENSE。实际执行仅以绝对路径 importlib 加载 sinkhorn_knopp.py；该文件仅导入 torch/numpy，无网络、删除或外部命令执行。forward 只接收 logits，不依赖 backbone，故加入 P2OT_feat。使用原始求解器；包装器说明见冻结协议。其余训练入口不运行，未宣称整仓库所有代码已审阅。

CIFAR tar 为 torchvision 官方 URL 的多伦多大学原始档；先对照 torchvision MD5，再安全检查归档成员路径后解压，由 torchvision 加载。CIFAR Python pickle 是官方数据格式，仅加载已核验档案。模型以 weights_only=True 加载。所有运行使用新环境 python -I，从项目根/明确绝对路径执行。-I 是导入隔离，不是系统安全沙箱。

哈希记录：third_party_manifest.json、source_inventory.json、feature_manifest.json、runtime_manifest.json，协议哈希独立记录且运行时核验。
''', encoding='utf-8')
    print('FROZEN', sha(out), flush=True)

if __name__ == '__main__':
    main()
