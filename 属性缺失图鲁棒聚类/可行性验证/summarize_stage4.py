"""Audit and summarize the three stage-four feasibility gates after all runs finish."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def mean(values):
    return float(np.mean(list(values)))


def check_finite(items, fields):
    for row in items:
        for field in fields:
            if not np.isfinite(float(row[field])):
                raise ValueError(f'Nonfinite {field}: {row}')


def main():
    baseline = rows(ROOT / 'sdcn_stage4_holdout/results.csv')
    weighted = rows(ROOT / 'sdcn_stage4_weighted/results.csv')
    confidence = json.loads((ROOT / 'confidence_stage4_holdout/results.json').read_text())
    quintiles = rows(ROOT / 'confidence_stage4_holdout/quintile_errors.csv')
    signal_correlations = rows(ROOT / 'confidence_stage4_holdout/signal_correlations.csv')
    amgc_dir = ROOT / 'amgc_stage4_full1000'
    amgc = json.loads((amgc_dir / 'result.json').read_text())
    data_check = json.loads((amgc_dir / 'data_check.json').read_text())
    if not all(data_check[key] for key in ('features_equal', 'labels_equal', 'edges_equal')):
        raise ValueError('AMGC Cora data alignment failed')
    amgc_loss = rows(amgc_dir / 'loss.csv')
    amgc_metrics = rows(amgc_dir / 'metrics.csv')
    if len(baseline) != 63 or len(weighted) != 54 or len(confidence) != 10 or len(quintiles) != 50 or len(signal_correlations) != 10:
        raise ValueError(f'Incomplete records: baseline={len(baseline)}, weighted={len(weighted)}, confidence={len(confidence)}')
    if len(amgc_loss) != 1000 or int(amgc_loss[-1]['epoch']) != 1000 or int(amgc_metrics[-1]['epoch']) != 1000:
        raise ValueError('AMGC did not complete 1000 epochs')
    check_finite(amgc_loss, ['loss', 'grad_norm', 'train_sec'])
    check_finite(amgc_metrics, ['acc', 'nmi', 'ari'])
    check_finite(amgc_metrics[1:], ['prediction_change'])

    base_map = {}
    for r in baseline:
        key = (float(r['rate']), int(r['mask']), int(r['seed']), r['method'])
        if key in base_map:
            raise ValueError(f'Duplicate baseline {key}')
        base_map[key] = r
        check_finite([r], ['acc', 'nmi', 'ari'])
    weighted_map = {}
    for r in weighted:
        key = (float(r['rate']), int(r['mask']), int(r['seed']), r['weight_mode'])
        if key in weighted_map:
            raise ValueError(f'Duplicate weighted {key}')
        weighted_map[key] = r
        reference = base_map[(key[0], key[1], key[2], 'fp')]
        if r['mask_sha256'] != reference['mask_sha256']:
            raise ValueError(f'Weighted mask mismatch {key}')
        stem = f'r{key[0]:g}_m{key[1]}_fp'
        initial_base = rows(ROOT / 'sdcn_stage4_holdout' /
                            f'{stem}_s{key[2]}' / 'metrics.csv')[0]
        initial_weighted = rows(ROOT / 'sdcn_stage4_weighted' /
                                f'{stem}_{key[3]}_s{key[2]}' / 'metrics.csv')[0]
        if any(abs(float(initial_base[field]) - float(initial_weighted[field])) > 1e-10
               for field in ('acc', 'nmi', 'ari')):
            raise ValueError(f'Weighted initialization mismatch {key}')
        check_finite([r], ['acc', 'nmi', 'ari'])
    for c in confidence:
        for seed in range(3):
            key = (c['rate'], c['mask'], seed, 'fp')
            if base_map[key]['mask_sha256'] != c['mask_sha256']:
                raise ValueError(f'Confidence mask mismatch {key}')

    comparisons = []
    for rate in (.4, .8):
        for metric in ('acc', 'nmi', 'ari'):
            fp = [float(base_map[(rate, m, s, 'fp')][metric]) for m in range(5) for s in range(3)]
            zero = [float(base_map[(rate, m, s, 'zero')][metric]) for m in range(5) for s in range(3)]
            diff = np.array(fp) - zero
            comparisons.append(dict(rate=rate, comparison='fp-minus-zero', metric=metric,
                                    n=len(diff), first_mean=mean(fp), second_mean=mean(zero),
                                    mean_delta=float(diff.mean()), positive_pairs=int((diff > 0).sum()),
                                    mask_mean_deltas=[float(np.mean(diff[m*3:(m+1)*3])) for m in range(5)]))
            for mode in ('fixed', 'informed', 'shuffled'):
                revised = [float(weighted_map[(rate, m, s, mode)][metric]) for m in range(3) for s in range(3)]
                uniform = [float(base_map[(rate, m, s, 'fp')][metric]) for m in range(3) for s in range(3)]
                d = np.array(revised) - uniform
                comparisons.append(dict(rate=rate, comparison=f'{mode}-minus-uniform-fp', metric=metric,
                                        n=len(d), first_mean=mean(revised), second_mean=mean(uniform),
                                        mean_delta=float(d.mean()), positive_pairs=int((d > 0).sum()),
                                        mask_mean_deltas=[float(np.mean(d[m*3:(m+1)*3])) for m in range(3)]))
            informed = [float(weighted_map[(rate, m, s, 'informed')][metric]) for m in range(3) for s in range(3)]
            shuffled = [float(weighted_map[(rate, m, s, 'shuffled')][metric]) for m in range(3) for s in range(3)]
            d = np.array(informed) - shuffled
            comparisons.append(dict(rate=rate, comparison='informed-minus-shuffled', metric=metric,
                                    n=len(d), first_mean=mean(informed), second_mean=mean(shuffled),
                                    mean_delta=float(d.mean()), positive_pairs=int((d > 0).sum()),
                                    mask_mean_deltas=[float(np.mean(d[m*3:(m+1)*3])) for m in range(3)]))

    baseline_diagnostics = []
    for rate, method in ((0., 'complete'), (.4, 'zero'), (.4, 'fp'), (.8, 'zero'), (.8, 'fp')):
        group = [r for r in baseline if float(r['rate']) == rate and r['method'] == method]
        baseline_diagnostics.append(dict(rate=rate, method=method, runs=len(group),
            acc_mean=mean(float(r['acc']) for r in group),
            nmi_mean=mean(float(r['nmi']) for r in group),
            ari_mean=mean(float(r['ari']) for r in group),
            min_occupied=min(int(r['occupied_clusters']) for r in group),
            severe_collapse=sum(int(r['occupied_clusters']) <= 2 or
                                float(r['largest_cluster_fraction']) >= .8 for r in group),
            mean_final_prediction_change=mean(float(r['last_prediction_change']) for r in group)))

    by_rate = defaultdict(list)
    for c in confidence:
        by_rate[c['rate']].append(c)
    confidence_summary = {str(rate): dict(test_spearman_mean=mean(x['test_spearman'] for x in group),
                                          test_negative_count=sum(x['test_spearman'] < 0 for x in group),
                                          missing_spearman_mean=mean(x['missing_spearman'] for x in group),
                                          missing_negative_count=sum(x['missing_spearman'] < 0 for x in group))
                          for rate, group in by_rate.items()}
    late = [r for r in amgc_metrics if int(r['epoch']) >= 900]
    amgc_summary = dict(initial=amgc['initial'], final={k: amgc[k] for k in
        ['acc', 'nmi', 'ari', 'missing_acc', 'cold_acc', 'occupied_clusters', 'largest_cluster_fraction']},
        epoch_count=len(amgc_loss), elapsed_sec=amgc['elapsed_sec'],
        mean_epoch_sec=amgc['mean_train_sec'], peak_rss_mb=amgc['peak_rss_mb'],
        late_acc_range=[min(float(x['acc']) for x in late), max(float(x['acc']) for x in late)],
        late_nmi_range=[min(float(x['nmi']) for x in late), max(float(x['nmi']) for x in late)],
        single_cluster_checkpoints=sum(int(r['occupied_clusters']) < 2 for r in amgc_metrics),
        late_prediction_change_mean=mean(float(x['prediction_change']) for x in late))
    quintile_summary = {str(rate): [mean(float(q['mean_rmse']) for q in quintiles
                                   if float(q['rate']) == rate and int(q['confidence_quintile']) == index)
                                  for index in range(1, 6)] for rate in (.4, .8)}
    quintile_pair_checks = {}
    for rate in (.4, .8):
        distributions = [[float(q['mean_rmse']) for q in quintiles
                          if float(q['rate']) == rate and int(q['mask']) == mask]
                         for mask in range(5)]
        quintile_pair_checks[str(rate)] = dict(top_lower_than_bottom=sum(v[-1] < v[0] for v in distributions),
                                               strict_monotonic=sum(all(a > b for a, b in zip(v, v[1:]))
                                                                    for v in distributions))
    signal_summary = {str(rate): {key: mean(float(r[key]) for r in signal_correlations if float(r['rate']) == rate)
                                  for key in ('learned', 'one_hop_coverage', 'two_hop_coverage', 'fp_norm')}
                      for rate in (.4, .8)}
    audit = dict(amgc=amgc_summary, confidence=confidence_summary,
                 confidence_quintile_rmse=quintile_summary,
                 confidence_quintile_checks=quintile_pair_checks,
                 signal_correlations=signal_summary, comparisons=comparisons,
                 baseline_diagnostics=baseline_diagnostics,
                 counts=dict(sdcn_baseline=len(baseline), sdcn_weighted=len(weighted), confidence=len(confidence)))
    (ROOT / 'stage4_summary.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# 第四阶段可行性验证报告', '',
             '固定协议下的完整 AMGC 周期、独立掩码重复与置信度消融。方法对照只使用最终轮 ACC/NMI/ARI；训练不使用节点标签。', '',
             '## 1. AMGC 固定掩码 1000 轮', '',
             f"完成 {len(amgc_loss)} 轮，耗时 {amgc['elapsed_sec']/3600:.2f} 小时，平均每轮 {amgc['mean_train_sec']:.2f} 秒；最终 ACC={amgc['acc']:.4f}、NMI={amgc['nmi']:.4f}、ARI={amgc['ari']:.4f}，占用 {amgc['occupied_clusters']} 簇。", '',
             f"最后 100 轮 ACC 范围 {amgc_summary['late_acc_range'][0]:.4f}–{amgc_summary['late_acc_range'][1]:.4f}，NMI 范围 {amgc_summary['late_nmi_range'][0]:.4f}–{amgc_summary['late_nmi_range'][1]:.4f}；每 5 轮检查的平均预测变化率 {amgc_summary['late_prediction_change_mean']:.4f}；单簇检查点 {amgc_summary['single_cluster_checkpoints']} 个。单掩码单种子不能证明泛化或严格复现论文。", '',
             '## 2. SDCN 独立掩码与种子', '',
             '新 mask seed=20261023；40% 与 80% 缺失各 5 个掩码 × 3 个模型种子，共 60 个缺失输入运行，加 3 个完整属性运行。固定最终轮，FP 和 Zero 使用相同掩码。', '',
             '| 缺失率 | 输入 | 运行数 | ACC 均值 | NMI 均值 | ARI 均值 | 严重塌缩数 | 最少占用簇 |',
             '|---:|---|---:|---:|---:|---:|---:|---:|']
    for d in baseline_diagnostics:
        lines.append(f"| {d['rate']:.0%} | {d['method']} | {d['runs']} | {d['acc_mean']:.4f} | {d['nmi_mean']:.4f} | {d['ari_mean']:.4f} | {d['severe_collapse']} | {d['min_occupied']} |")
    lines += ['', '“严重塌缩”仅作诊断：占用簇≤2 或最大簇占比≥0.8；不是以标签挑选模型的规则。', '',
             '| 缺失率 | 对比 | 指标 | 平均差值 | 正差配对数 | 逐掩码均值差 |',
             '|---:|---|---|---:|---:|---|']
    for c in comparisons:
        if c['comparison'] != 'fp-minus-zero':
            continue
        lines.append(f"| {c['rate']:.0%} | FP − Zero | {c['metric'].upper()} | {c['mean_delta']:+.4f} | {c['positive_pairs']}/{c['n']} | " + ', '.join(f'{v:+.4f}' for v in c['mask_mean_deltas']) + ' |')
    lines += ['', '## 3. 无标签置信度与加权消融', '',
              '每个掩码从已观测节点取两个互不重叠的 20% 子集，分别作校准与测试。校准时人为隐藏该子集的属性，用结构覆盖和 FP 输出信号预测补全误差。原本缺失节点的真值仅用于评分后的离线核验。Spearman 为“置信度对误差”，负值表示预期方向。', '',
              '| 缺失率 | 测试折平均 Spearman | 测试折负相关数 | 原缺失节点平均 Spearman |',
              '|---:|---:|---:|---:|']
    for rate in ('0.4', '0.8'):
        c = confidence_summary[rate]
        lines.append(f"| {float(rate):.0%} | {c['test_spearman_mean']:.4f} | {c['test_negative_count']}/5 | {c['missing_spearman_mean']:.4f} |")
    lines += ['', '原缺失节点的置信度五分位，从低到高的平均补全 RMSE：', '',
              '| 缺失率 | Q1 最低 | Q2 | Q3 | Q4 | Q5 最高 |',
              '|---:|---:|---:|---:|---:|---:|']
    for rate in ('0.4', '0.8'):
        lines.append(f"| {float(rate):.0%} | " + ' | '.join(f'{x:.4f}' for x in quintile_summary[rate]) + ' |')
    lines += ['', '各缺失率的 5 个掩码中，最高置信五分位误差低于最低五分位的个数分别为 ' +
              '、'.join(f"{int(float(rate)*100)}%：{quintile_pair_checks[rate]['top_lower_than_bottom']}/5"
                       for rate in ('0.4', '0.8')) + '。', '']
    lines += ['与单一信号的离线相关性比较（同样为置信度或信号对误差的 Spearman；原缺失真值仅用于计算相关性）：', '',
              '| 缺失率 | 组合置信度 | 一跳覆盖 | 两跳覆盖 | FP 向量范数 |',
              '|---:|---:|---:|---:|---:|']
    for rate in ('0.4', '0.8'):
        s = signal_summary[rate]
        lines.append(f"| {float(rate):.0%} | {s['learned']:.4f} | {s['one_hop_coverage']:.4f} | {s['two_hop_coverage']:.4f} | {s['fp_norm']:.4f} |")
    lines += ['', '固定 FP 输入，在前 3 个掩码 × 3 个种子上比较等权、缺失节点固定权重 0.6、置信度权重和缺失节点内随机打乱的同分布权重。只改两个逐节点 KL 项，权重总和归一化。', '',
              '| 缺失率 | 对比 | 指标 | 平均差值 | 正差配对数 |',
              '|---:|---|---|---:|---:|']
    for c in comparisons:
        if c['comparison'] == 'fp-minus-zero':
            continue
        lines.append(f"| {c['rate']:.0%} | {c['comparison']} | {c['metric'].upper()} | {c['mean_delta']:+.4f} | {c['positive_pairs']}/{c['n']} |")
    lines += ['', '## 判断边界', '',
              '这组实验用于判断方向是否能实现和置信度机制是否值得继续开发。只有 Cora、随机整节点属性缺失与 SDCN 诊断适配；不等于 AMGC 论文严格复现，也不足以声称新方法普遍优于已有方法。置信度效果须同时优于固定降权和随机打乱，才能把收益归因于置信度信息。', '',
              '复核：`stage4_plan.json`、`stage4_summary.json`、三个实验目录的 `config.json`/逐轮 CSV/最终 `result.json`，以及对应 stdout/stderr 日志。']
    (ROOT / '第四阶段验证报告.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('Stage 4 audit passed; report and summary written.')


if __name__ == '__main__':
    main()
