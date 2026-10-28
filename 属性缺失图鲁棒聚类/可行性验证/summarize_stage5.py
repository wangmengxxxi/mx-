"""Summarize the preregistered stage-five paired experiments."""
from __future__ import annotations

import hashlib
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from run_minimal_diagnostic import load_cora


ROOT = Path(__file__).resolve().parent
METRICS = ('acc', 'nmi', 'ari')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def key(row):
    return (float(row['rate']), int(row['mask']), int(row['seed']))


def index(rows, mode=None):
    selected = [r for r in rows if mode is None or r.get('weight_mode') == mode]
    indexed = {key(r): r for r in selected}
    assert len(indexed) == len(selected), 'Duplicate rate/mask/seed combination'
    return indexed


def compare(left, right, metric, rate):
    keys = sorted(k for k in left if k[0] == rate)
    assert set(keys) == {k for k in right if k[0] == rate}, 'Unpaired runs'
    diffs = []
    by_mask = defaultdict(list)
    for k in keys:
        assert left[k]['mask_sha256'] == right[k]['mask_sha256'], f'Mask mismatch: {k}'
        d = float(left[k][metric]) - float(right[k][metric])
        assert math.isfinite(d), f'Nonfinite result: {k}'
        diffs.append(d)
        by_mask[k[1]].append(d)
    return dict(n=len(diffs), mean=float(np.mean(diffs)),
                positive=int(np.sum(np.asarray(diffs) > 0)),
                mask_means={str(k): float(np.mean(v)) for k, v in by_mask.items()})


def main():
    plan = read(ROOT / 'stage5_plan.json')
    random_reference = [r for r in read(ROOT / 'sdcn_stage4_holdout/results.json')
                        if r['method'] == 'fp' and r['rate'] in (0.4, 0.8)]
    weighted = read(ROOT / 'sdcn_stage5_reconstruction/results.json')
    degree = read(ROOT / 'sdcn_stage5_degree_baseline/results.json')
    confidence = read(ROOT / 'confidence_stage5_degree/results.json')
    assert len(random_reference) == 30 and len(weighted) == 90
    assert len(degree) == 36 and len(confidence) == 6
    baseline = index(random_reference)
    modes = {name: index(weighted, name) for name in ('fixed', 'informed', 'shuffled')}
    assert all(set(v) == set(baseline) for v in modes.values())
    ae_only_differences = []
    for k in baseline:
        for name, values in modes.items():
            row = values[k]
            assert row['mask_sha256'] == baseline[k]['mask_sha256']
            assert row['pretrain_mse_last'] == baseline[k]['pretrain_mse_last'], f'Pretrain mismatch: {name}, {k}'
            bpath = ROOT / 'sdcn_stage4_holdout' / f'r{k[0]:g}_m{k[1]}_fp_s{k[2]}' / 'metrics.csv'
            wpath = ROOT / 'sdcn_stage5_reconstruction' / f'r{k[0]:g}_m{k[1]}_fp_{name}_s{k[2]}' / 'metrics.csv'
            with bpath.open(encoding='utf-8-sig', newline='') as f:
                b0 = next(csv.DictReader(f))
            with wpath.open(encoding='utf-8-sig', newline='') as f:
                w0 = next(csv.DictReader(f))
            for metric in METRICS:
                assert b0[metric] == w0[metric], f'Joint epoch-zero mismatch: {name}, {k}, {metric}'
            if row['ae_acc'] != baseline[k]['ae_acc'] or row['ae_nmi'] != baseline[k]['ae_nmi']:
                ae_only_differences.append(dict(rate=k[0], mask=k[1], seed=k[2], mode=name,
                                                ae_acc_delta=row['ae_acc']-baseline[k]['ae_acc'],
                                                ae_nmi_delta=row['ae_nmi']-baseline[k]['ae_nmi']))
    config = read(ROOT / 'sdcn_stage5_reconstruction/config.json')
    assert config['weight_location'] == 'reconstruction' and config['mask_mode'] == 'random'
    dconfig = read(ROOT / 'sdcn_stage5_degree_baseline/config.json')
    assert dconfig['mask_mode'] == 'degree_biased'
    assert config['pretrain_epochs'] == dconfig['pretrain_epochs'] == 100
    assert config['train_epochs'] == dconfig['train_epochs'] == 400
    assert config['lr'] == dconfig['lr'] == .001
    random_comparisons = {}
    for rate in (.4, .8):
        for metric in METRICS:
            for name in modes:
                random_comparisons[f'{rate:g}:{name}-uniform:{metric}'] = compare(modes[name], baseline, metric, rate)
            for name in ('fixed', 'shuffled'):
                random_comparisons[f'{rate:g}:informed-{name}:{metric}'] = compare(modes['informed'], modes[name], metric, rate)

    d_zero = index([r for r in degree if r['method'] == 'zero'])
    d_fp = index([r for r in degree if r['method'] == 'fp'])
    assert len(d_zero) == len(d_fp) == 18
    degree_comparisons = {f'{rate:g}:fp-zero:{metric}': compare(d_fp, d_zero, metric, rate)
                          for rate in (.4, .8) for metric in METRICS}
    features, _, norm_adj, node_degree = load_cora()
    mask_diagnostics = []
    for rate in (.4, .8):
        for mask in range(3):
            rng = np.random.default_rng(np.random.SeedSequence([20261024, round(rate * 10000), mask]))
            probabilities = np.sqrt(node_degree + 1)
            probabilities /= probabilities.sum()
            observed = np.ones(len(features), dtype=bool)
            observed[rng.choice(len(features), round(rate * len(features)), replace=False,
                                p=probabilities)] = False
            expected_hash = hashlib.sha256(observed.tobytes()).hexdigest()
            assert d_fp[(rate, mask, 0)]['mask_sha256'] == expected_hash
            cold = ~observed & (norm_adj[:, observed].sum(1).A1 < 1e-8)
            mask_diagnostics.append(dict(rate=rate, mask=mask,
                                         mean_degree_missing=float(node_degree[~observed].mean()),
                                         mean_degree_observed=float(node_degree[observed].mean()),
                                         cold_count=int(cold.sum()),
                                         mask_sha256=expected_hash))
    output = dict(plan=plan, counts=dict(random_baseline=len(random_reference),
                                        reconstruction_weighted=len(weighted),
                                        degree_baseline=len(degree), degree_confidence=len(confidence)),
                  random_comparisons=random_comparisons,
                  degree_comparisons=degree_comparisons,
                  degree_mask_diagnostics=mask_diagnostics,
                  degree_confidence=confidence,
                  ae_diagnostic_differences=ae_only_differences)
    (ROOT / 'stage5_summary.json').write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')

    lines = ['# 第五阶段验证报告', '',
             '固定最终轮；节点标签只用于离线评价。随机缺失使用第四阶段的等权 FP 运行作配对参照。', '',
             '## 1. 置信度用于联合训练的属性重建损失', '',
             'Cora 随机整节点属性缺失，每个缺失率 5 掩码 × 3 种子。FP 输入、预训练、KL 项不变；比较固定降权、校准置信度和打乱置信度。', '',
             '| 缺失率 | 比较 | 指标 | 平均差 | 正差配对 |',
             '|---:|---|---|---:|---:|']
    for rate in (.4, .8):
        for metric in METRICS:
            for pair in ('fixed-uniform', 'informed-uniform', 'shuffled-uniform',
                         'informed-fixed', 'informed-shuffled'):
                row = random_comparisons[f'{rate:g}:{pair}:{metric}']
                lines.append(f'| {rate:.0%} | {pair} | {metric.upper()} | {row["mean"]:+.4f} | {row["positive"]}/{row["n"]} |')
    lines += ['', '## 2. 高连接度节点偏向缺失', '',
              '按 sqrt(度数+1) 加权无放回抽样缺失节点；每个缺失率 3 掩码 × 3 种子。', '',
              '| 缺失率 | 指标 | FP − Zero 平均差 | 正差配对 | 缺失节点平均度 | 观测节点平均度 |',
              '|---:|---|---:|---:|---:|---:|']
    for rate in (.4, .8):
        masks = [x for x in mask_diagnostics if x['rate'] == rate]
        md = np.mean([x['mean_degree_missing'] for x in masks])
        od = np.mean([x['mean_degree_observed'] for x in masks])
        for metric in METRICS:
            row = degree_comparisons[f'{rate:g}:fp-zero:{metric}']
            lines.append(f'| {rate:.0%} | {metric.upper()} | {row["mean"]:+.4f} | {row["positive"]}/{row["n"]} | {md:.2f} | {od:.2f} |')
    lines += ['', '此设置的置信度校准在独立观测节点测试折上的 Spearman：']
    for rate in (.4, .8):
        c = [x for x in confidence if x['rate'] == rate]
        lines.append(f'- {rate:.0%}：平均 {np.mean([x["test_spearman"] for x in c]):+.4f}，负相关 {sum(x["test_spearman"] < 0 for x in c)}/{len(c)}。')
    lines += ['', '## 结论与判断边界', '',
              '置信度用于联合训练的属性重建损失，没有稳定优于等权、固定降权或随机打乱；结合第四阶段对 KL 项加权的负结果，当前置信度加权机制未通过创新效果验证。',
              'FP 相对置零的收益在高连接度节点偏向缺失时仍成立：40% 和 80% 缺失下，ACC/NMI/ARI 均为 9/9 组配对正差。这支持属性补全方向在两种缺失设置下的工程可行性。',
              '两步仍只覆盖 Cora；高连接度缺失设置的 FP 收益不能代替置信度机制的消融，也不能证明跨数据集泛化。',
              f'配对核验：所有运行的预训练末 MSE 和联合训练第 0 轮 ACC/NMI/ARI 与等权参照一致；独立 AE KMeans 诊断有 {len(ae_only_differences)} 组轻微差异，详见汇总 JSON。',
              '详见 `stage5_plan.json`、`stage5_summary.json` 与各实验目录的原始结果。', '']
    (ROOT / '第五阶段验证报告.md').write_text('\n'.join(lines), encoding='utf-8')
    print('saved stage5_summary.json and 第五阶段验证报告.md')


if __name__ == '__main__':
    main()
