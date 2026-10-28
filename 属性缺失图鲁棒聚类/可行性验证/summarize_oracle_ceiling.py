# -*- coding: utf-8 -*-
"""Audit and summarize the fixed, completed stage-7 oracle experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from run_fp_coverage_pair import stats
from run_minimal_diagnostic import load_cora
from run_oracle_ceiling import (METHODS, REFERENCE, ROOT, globally_correct,
                               make_mask, sha_array, sha_file, smooth, write_json)
from run_sdcn_baseline import evaluate, save_csv
from run_sdcn_smoke import fill_features


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values):
    return float(np.mean(values))


def summary(values):
    values = np.asarray(values, dtype=float)
    return dict(mean=float(values.mean()), std=float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                min=float(values.min()), max=float(values.max()),
                positive=int((values > 0).sum()), negative=int((values < 0).sum()),
                zero=int((values == 0).sum()), n=len(values))


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join(["---"] * len(headers)) + " |"] +
                     ["| " + " | ".join(str(v) for v in row) + " |" for row in rows])


def f(value, digits=4, signed=False):
    if value is None:
        return "—"
    return format(value, f"{'+' if signed else ''}.{digits}f")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="oracle_ceiling")
    ap.add_argument("--report", default="第七阶段上限分解报告.md")
    args = ap.parse_args()
    out = ROOT / args.input
    protocol = read_json(out / "config.json")
    status = read_json(out / "status.json")
    if protocol["smoke"] or status["phase"] != "complete":
        raise RuntimeError("Only a completed full experiment can produce the stage-7 report")
    rows = read_json(out / "results.json")
    views = read_json(out / "complete_mask_evaluations.json")
    masks = read_json(out / "mask_metadata.json")
    checks = read_json(out / "mask_checks.json")
    integrity = read_json(out / "integrity.json")
    refcheck = read_json(out / "refcheck" / "comparison.json")
    assert len(rows) == protocol["expected_main_runs"] == 50
    assert protocol["rates"] == [0.8, 0.4] and protocol["seeds"] == [0, 1] and protocol["masks"] == 3
    assert len(views) == 12 and len(masks) == 6 and len(checks) == 24
    assert all(check["equal"] for check in checks)
    assert integrity["source_files_unchanged"] and integrity["reference_json_unchanged"]
    assert refcheck["mask_equal"]
    assert all(sha_file(ROOT / name) == digest for name, digest in protocol["source_sha256"].items())
    assert all(sha_file(ROOT / name) == digest for name, digest in protocol["reference_json_sha256"].items())
    features, labels, norm_adj, _ = load_cora()
    index = {(r["rate"], r["mask"], r["seed"], r["branch"]): r for r in rows if r["branch"] != "E"}
    index.update({(r["rate"], r["mask"], r["seed"], "E"): r for r in views})
    assert len(index) == 60
    mask_index = {(r["rate"], r["mask"]): r for r in masks}
    audit_records = []

    # Reconstruct each input independently from saved selections, and recompute
    # final metrics from saved predictions. No training or model selection here.
    for rate in (0.4, 0.8):
        for mask_id in range(3):
            meta = mask_index[rate, mask_id]
            observed = make_mask(len(features), rate, mask_id, protocol["training_config"]["mask_seed"])
            cold = ~observed & (norm_adj[:, observed].sum(1).A1 < 1e-8)
            assert sha_array(observed) == meta["mask_sha256"]
            assert np.array_equal(np.flatnonzero(cold), meta["cold_selected_nodes"])
            random_nodes = np.asarray(meta["random_selected_nodes"], dtype=np.int64)
            assert len(np.unique(random_nodes)) == len(random_nodes)
            assert np.all((~observed & ~cold)[random_nodes])
            assert len(random_nodes) == min(int(cold.sum()), int((~observed & ~cold).sum()))
            expected_random = np.sort(np.random.default_rng(np.random.SeedSequence(meta["random_seed_entropy"])).choice(
                np.flatnonzero(~observed & ~cold), len(random_nodes), replace=False))
            assert np.array_equal(random_nodes, expected_random)
            fp = fill_features(features, observed, "fp", norm_adj, protocol["training_config"]["steps"])
            for branch in METHODS:
                filled = (fill_features(features, observed, "zero", norm_adj, 40) if branch == "A" else
                          features.copy() if branch == "E" else fp.copy())
                if branch in ("C", "D"):
                    selected = np.flatnonzero(cold) if branch == "C" else random_nodes
                    filled[selected] = features[selected]
                input_sha = sha_array(smooth(filled, norm_adj, 2))
                quality = stats(features, filled, ~observed)
                for seed in (0, 1):
                    row = index[rate, mask_id, seed, branch]
                    fit = next(r for r in rows if r["branch"] == "E" and r["seed"] == seed) if branch == "E" else row
                    folder = Path(fit["output_path"])
                    fit_config = read_json(folder / "config.json")
                    assert fit_config["training_config"] == protocol["training_config"]
                    assert read_json(folder / "result.json") == fit
                    assert fit["input_sha256"] == input_sha
                    assert fit["fill_sha256"] == sha_array(filled)
                    assert fit["smoothing_k"] == 2
                    assert row["mask_sha256"] == meta["mask_sha256"]
                    assert abs(row["imputation_rmse"] - quality["rmse"]) < 1e-10
                    assert abs(row["imputation_cosine"] - quality["cosine"]) < 1e-10
                    if branch != "E":
                        saved_mask = np.load(folder / "mask.npz")
                        assert np.array_equal(saved_mask["observed"], observed)
                        assert np.array_equal(saved_mask["cold"], cold)
                        expected = meta["cold_selected_nodes"] if branch == "C" else meta["random_selected_nodes"] if branch == "D" else []
                        assert saved_mask["replaced_nodes"].tolist() == expected
                        assert fit_config["selected_nodes"] == expected and row["replaced_count"] == len(expected)
                    pred = np.load(folder / "predictions.npz")["pred"]
                    recomputed = evaluate(labels, pred, observed, cold)
                    correct = globally_correct(labels, pred)
                    for key in ("acc", "nmi", "ari", "missing_acc", "observed_acc", "cold_acc"):
                        assert abs(row[key] - recomputed[key]) < 1e-12
                    assert abs(row["noncold_missing_acc"] - float(correct[~observed & ~cold].mean())) < 1e-12
                    assert abs(row["noncold_all_acc"] - float(correct[~cold].mean())) < 1e-12
                    audit_records.append(dict(rate=rate, mask=mask_id, seed=seed, branch=branch,
                        input_sha256=input_sha, mask_sha256=meta["mask_sha256"],
                        input_reconstruction_equal=True, predictions_metrics_equal=True))
    write_json(out / "audit.json", dict(passed=True, unique_fits=50, mask_branch_evaluations=60,
        reference_json_unchanged=True, sources_unchanged=True, records=audit_records))

    means, pairs, paired_summary, subgroups = [], [], [], []
    for rate in (0.4, 0.8):
        for branch in METHODS:
            selected = [index[rate, mask_id, seed, branch] for mask_id in range(3) for seed in (0, 1)]
            means.append(dict(rate=rate, branch=branch, method=METHODS[branch],
                fitted_models=2 if branch == "E" else 6, paired_evaluations=6,
                **{key: mean([row[key] for row in selected]) for key in
                   ("acc", "nmi", "ari", "cold_acc", "noncold_missing_acc", "noncold_all_acc",
                    "imputation_rmse", "imputation_cosine", "replaced_count")},
                acc_std_across_evaluations=summary([row["acc"] for row in selected])["std"]))
        for mask_id in range(3):
            for seed in (0, 1):
                by = {branch: index[rate, mask_id, seed, branch] for branch in METHODS}
                item = dict(rate=rate, mask=mask_id, seed=seed, mask_sha256=mask_index[rate, mask_id]["mask_sha256"])
                for name, a, b in (("E-B", "E", "B"), ("C-B", "C", "B"), ("D-B", "D", "B"),
                                   ("B-A", "B", "A"), ("C-D", "C", "D")):
                    for metric in ("acc", "nmi", "ari", "cold_acc", "noncold_missing_acc"):
                        item[f"{name}_{metric}"] = by[a][metric] - by[b][metric]
                item["B_cold_minus_noncold_missing"] = by["B"]["cold_acc"] - by["B"]["noncold_missing_acc"]
                item["B_cold_minus_noncold_all"] = by["B"]["cold_acc"] - by["B"]["noncold_all_acc"]
                pairs.append(item)
        selected_pairs = [row for row in pairs if row["rate"] == rate]
        for name in ("E-B", "C-B", "D-B", "B-A", "C-D"):
            for metric in ("acc", "nmi", "ari", "cold_acc", "noncold_missing_acc"):
                paired_summary.append(dict(rate=rate, contrast=name, metric=metric,
                    **summary([row[f"{name}_{metric}"] for row in selected_pairs])))
        bymean = {row["branch"]: row for row in means if row["rate"] == rate}
        subgroups.append(dict(rate=rate, B_cold_acc=bymean["B"]["cold_acc"],
            B_noncold_missing_acc=bymean["B"]["noncold_missing_acc"],
            B_noncold_all_acc=bymean["B"]["noncold_all_acc"],
            B_cold_minus_noncold_missing=bymean["B"]["cold_acc"] - bymean["B"]["noncold_missing_acc"],
            B_cold_minus_noncold_all=bymean["B"]["cold_acc"] - bymean["B"]["noncold_all_acc"],
            C_cold_acc=bymean["C"]["cold_acc"],
            C_minus_B_cold_acc=bymean["C"]["cold_acc"] - bymean["B"]["cold_acc"],
            C_minus_B_cold_positive=sum(row["C-B_cold_acc"] > 0 for row in selected_pairs)))

    old_complete = [read_json(REFERENCE / f"r0_m0_complete_s{seed}" / "result.json") for seed in (0, 1)]
    old_E = {key: mean([row[key] for row in old_complete]) for key in ("acc", "nmi", "ari")}
    new_E = {key: mean([row[key] for row in rows if row["branch"] == "E"]) for key in ("acc", "nmi", "ari")}
    archive_context = dict(old_E_k0=old_E, new_E_k2=new_E,
        paired_difference={key: new_E[key] - old_E[key] for key in old_E},
        note="Same model seeds 0,1; old archived k=0 vs newly trained k=2; descriptive smoothing context only")
    collapses = [dict(path=row["output_path"], acc=row["acc"], occupied_clusters=row["occupied_clusters"],
                    largest_cluster_fraction=row["largest_cluster_fraction"]) for row in rows
                 if row["occupied_clusters"] <= 2 or row["largest_cluster_fraction"] >= 0.8]
    equal_count_pairs = [row for row in pairs if row["rate"] == 0.8
                        and not mask_index[row["rate"], row["mask"]]["random_count_shortfall"]]
    equal_count_context = summary([row["C-D_acc"] for row in equal_count_pairs])
    result = dict(branch_means=means, paired_summary=paired_summary, subgroups=subgroups,
        pairs=pairs, archive_context=archive_context, collapse_definition="occupied_clusters <= 2 OR largest_cluster_fraction >= 0.8",
        severe_collapses=collapses, equal_count_80_C_minus_D_descriptive_only=equal_count_context,
        smoke="oracle_ceiling_smoke: 5 fits, AE 2 / joint 3; excluded from all scientific tables")
    write_json(out / "summary.json", result)
    save_csv(out / "branch_means.csv", means)
    save_csv(out / "paired_differences.csv", pairs)
    save_csv(out / "paired_summary.csv", paired_summary)

    lookup = {(r["rate"], r["contrast"], r["metric"]): r for r in paired_summary}
    def diff(rate, contrast, metric="acc"):
        return lookup[rate, contrast, metric]

    branch_table = []
    for row in means:
        branch_table.append([f"{row['rate']:.0%}", row["branch"] + " " + row["method"],
            f(row["acc"]), f(row["nmi"]), f(row["ari"]), f(row["imputation_rmse"], 6), f(row["imputation_cosine"]),
            row["fitted_models"]])
    pair_table = []
    for rate in (0.4, 0.8):
        for contrast in ("E-B", "C-B", "D-B", "B-A", "C-D"):
            acc, nmi, ari = [diff(rate, contrast, key) for key in ("acc", "nmi", "ari")]
            pair_table.append([f"{rate:.0%}", contrast, f(acc["mean"], signed=True),
                f"{acc['positive']}/6", f"{f(nmi['mean'], signed=True)} / {nmi['positive']}/6",
                f"{f(ari['mean'], signed=True)} / {ari['positive']}/6"])
    subgroup_table = [[f"{r['rate']:.0%}", f(r["B_cold_acc"]), f(r["B_noncold_missing_acc"]),
        f(r["B_cold_minus_noncold_missing"], signed=True), f(r["B_noncold_all_acc"]),
        f(r["B_cold_minus_noncold_all"], signed=True),
        f(r["C_cold_acc"]), f(r["C_minus_B_cold_acc"], signed=True), f"{r['C_minus_B_cold_positive']}/6"] for r in subgroups]
    detail_table = [[f"m{r['mask']}/s{r['seed']}"] + [f(r[key], signed=True) for key in
        ("E-B_acc", "C-B_acc", "D-B_acc", "B-A_acc", "C-D_acc", "C-B_cold_acc")]
        for r in pairs if r["rate"] == 0.8]
    mask_table = [[f"{r['rate']:.0%}/m{r['mask']}", r["missing_count"], r["cold_count"],
        r["noncold_missing_count"], len(r["random_selected_nodes"]),
        "是" if r["random_count_shortfall"] else "否", f"`{r['mask_sha256']}`"] for r in masks]

    main_EB, main_CB, main_DB, main_BA, main_CD = [diff(0.8, name) for name in ("E-B", "C-B", "D-B", "B-A", "C-D")]
    if main_EB["mean"] < 0.02:
        verdict = ("**失败。**80% 缺失下 E−B 的平均 ACC 差为 " + f(main_EB["mean"], signed=True) +
            "，低于事先固定的 0.02，触发明确失败条件。在此流程中，将缺失属性恢复为原始真值未显示足够的聚类收益，"
            "本阶段不支持把提高原始属性恢复质量作为主要突破口。")
    elif main_CB["positive"] < 5 and main_CD["mean"] <= 0 and main_CB["mean"] > 0 and main_DB["mean"] > 0:
        verdict = ("**部分通过：存在整体补全空间，但冷节点特殊瓶颈未获支持。**E−B 的 ACC 平均增益为 "
            + f(main_EB["mean"], signed=True) + "，NMI/ARI 平均差为 "
            + f(diff(0.8, "E-B", "nmi")["mean"], signed=True) + "/"
            + f(diff(0.8, "E-B", "ari")["mean"], signed=True)
            + "；B−A 为 " + f(main_BA["mean"], signed=True) + "。未触发预设失败情形。"
            + "C−B 的均值" + ("达到" if main_CB["mean"] >= 0.03 else "未达到")
            + " +0.03 的门槛，但只有 " + str(main_CB["positive"])
            + "/6 正差，未达到 5/6，且 C−D 平均为 " + f(main_CD["mean"], signed=True)
            + "、仅 " + str(main_CD["positive"]) + "/6 正差，不能判为通过。"
            + "C、D 的平均增益同量级，D 更高，C−D 的逐组范围为 "
            + f(main_CD["min"], signed=True) + " 至 " + f(main_CD["max"], signed=True)
            + "。“≈”仅作描述性近似，不声称统计等效。方向应调整为覆盖全部缺失节点的面向聚类的补全；"
            + "oracle 增益没有证明任何新方法有效。")
    else:
        verdict = "【固定规则尚需人工结合 C−D 与 B−A 作描述性判读；不得修改阈值。】"
    baseline_hash = protocol["source_sha256"]["run_sdcn_baseline.py"]
    report = f"""# 第七阶段上限分解：固定平滑下的 oracle 诊断

日期：2026-10-27。目标是在相同图平滑与 SDCN 下游流程下，测量真实属性替换的诊断增益，并判断冷节点是否是不成比例的瓶颈。Oracle 使用缺失属性真值，**只用于诊断，不是可部署方法**。

## 1. 固定协议

- Planetoid Cora，完整图结构，随机整节点属性遮盖；mask seed `20261023`。40%/80% 缺失，各取前 3 个掩码 × 模型种子 0、1，与上一阶段相同。
- 所有分支统一使用 `X_in = S² X_fill`，`S = D^(-1/2)(A+I)D^(-1/2)`。`k=2` 事先固定，平滑后不钳制观测值；SDCN 图结构相同。
- A：置零；B：40 步 FP；C：FP 后只把冷节点替换为真值；D：FP 后随机选择与冷节点等数量的非冷缺失节点替换为真值，数量不足时全部替换；E：完整真值属性。
- 冷节点指“缺失且没有一跳观测邻居”。D 用 `SeedSequence([20261023, round(rate*10000), mask, 0x4F524143])` 选点，两个模型种子共用，节点编号与不足标记均存档。
- 原样复用 `load_cora`、`fill_features`、`train`、`evaluate`。hidden 128,64、latent 10、AE 100 轮、联合 400 轮、两阶段 lr=1e-3、batch 256、threads 2；只报告固定最终轮，不用标签选择模型或参数。
- E 只训练种子 0、1 两次，按同种子配对，按各掩码重算子群指标。主训练共 50 次，另做 1 次 B(k=0) 复跑。5 次 2/3 轮冒烟不进入下表。

## 2. 分支均值与补全质量

{table(['缺失率', '分支', 'ACC', 'NMI', 'ARI', '缺失节点 RMSE', '缺失节点余弦', '独立训练数'], branch_table)}

补全指标只在**平滑前 `X_fill` 的缺失节点**上对原始真值计算，零向量余弦记为 0。E 按原始掩码统计；各配对组等权。Oracle 的补全误差下降不等于方法有效。

## 3. 配对差值与正差计数

总上限 = E−B；冷节点上限 = C−B；随机对照 = D−B；统一平滑后补全价值 = B−A。另列 C−D，直接检查冷节点干预是否超过随机干预。

{table(['缺失率', '对比', 'ACC 平均差', 'ACC 正差', 'NMI 平均差 / 正差', 'ARI 平均差 / 正差'], pair_table)}

全部逐组差值见 `paired_differences.csv`。严重簇塌缩沿用“占用簇数 ≤2 或最大簇占比 ≥0.8”，触发 {len(collapses)}/50 次；未删除任何负差或塌缩运行。

## 4. 冷节点与随机对照的含义

{table(['缺失率', 'B 冷节点 ACC', 'B 非冷缺失 ACC', '冷−非冷缺失', 'B 全部非冷 ACC', '冷−全部非冷', 'C 冷节点 ACC', '冷节点 C−B', '冷节点正差'], subgroup_table)}

“全部非冷”包含观测节点，“非冷缺失”提供同缺失状态对照。子群沿用各模型的全图 Hungarian 映射，不单独匹配。C 的变化经 S² 传播到其他节点，C−B 不是可加的冷节点误差贡献。

{table(['缺失率/掩码', '缺失数', '冷节点/C 替换数', '非冷缺失数', 'D 替换数', 'D 数量不足', 'mask SHA-256'], mask_table)}

80% 的 m0 中 D 替换全部 1038 个非冷缺失节点，少于 C 的 1128 个，且没有剩余抽样自由度；其余掩码严格等数量。作为数量不足的描述性核查，80% 等数量的 m1/m2 四组 C−D 均值为 {f(equal_count_context['mean'], signed=True)}，{equal_count_context['positive']}/4 正差，不替代预设六组判读。选择列表见 `mask_metadata.json` 和逐组配置。

## 5. 掩码、代码与 FP 复跑核对

- 6 掩码 × 2 种子 × 旧 A/B，共 24 项掩码哈希、冷节点数核对全部一致，见 `mask_checks.json`。冷节点定义与旧实现一致，FP 与上一阶段传播输出逐元素一致。
- 复用的第四阶段来源文件与旧存档一致，训练前后来源脚本和旧参照 JSON 未变化，见 `integrity.json`。`run_sdcn_baseline.py` SHA-256：`{baseline_hash}`。
- 单独新跑 `r0.4_m0_fp_k0_s0`：旧/新 ACC 为 {f(refcheck['reference']['acc'], 6)} / {f(refcheck['rerun']['acc'], 6)}（差 {f(refcheck['delta']['acc'], 6, True)}），NMI 为 {f(refcheck['reference']['nmi'], 6)} / {f(refcheck['rerun']['nmi'], 6)}（差 {f(refcheck['delta']['nmi'], 6, True)}），ARI 为 {f(refcheck['reference']['ari'], 6)} / {f(refcheck['rerun']['ari'], 6)}（差 {f(refcheck['delta']['ari'], 6, True)}）。冷节点 ACC 差 {f(refcheck['delta']['cold_acc'], 6, True)}。这些是约千分之一量级的 CPU 数值路径差异；不能将它们解释为新补全方法的收益。
- 项目 `envs/amgc_cpu/Scripts/python.exe`：NumPy 1.26.4、PyTorch 2.8.0+cpu；环境详见配置。日志为 `oracle_ceiling.stdout.log`、`oracle_ceiling.stderr.log`。
- 全量重建输入、核验 oracle 节点和输入哈希，并从预测重算指标：50 次独立训练、60 个含 E 的掩码分支评价全部通过，见 `audit.json`。

## 6. 按事先规则判读

按 80% 缺失、ACC 判读，同时报告 NMI/ARI。通过：C−B ≥ +0.03、至少 5/6 正差且明显超过 D−B；部分通过：E−B 足够大、C−B≈D−B；失败：E−B < 0.02 或 B≈A。原协议未量化“明显”“足够大”“≈”，本报告不补设阈值。

80% 主结果：E−B={f(main_EB['mean'], signed=True)}，C−B={f(main_CB['mean'], signed=True)}（{main_CB['positive']}/6 正），D−B={f(main_DB['mean'], signed=True)}，C−D={f(main_CD['mean'], signed=True)}（{main_CD['positive']}/6 正），B−A={f(main_BA['mean'], signed=True)}。

{verdict}

40% 辅助实验的 E−B 仅为 {f(diff(0.4, 'E-B')['mean'], signed=True)}，C−B 为 {f(diff(0.4, 'C-B')['mean'], signed=True)}。补全空间随缺失率变化，不能把 80% 的部分通过推广为所有缺失率有效。

## 7. 平滑参照与判断边界

同种子 0、1，旧 E(k=0) 的 ACC/NMI/ARI 为 {f(old_E['acc'])}/{f(old_E['nmi'])}/{f(old_E['ari'])}，新 E(k=2) 为 {f(new_E['acc'])}/{f(new_E['nmi'])}/{f(new_E['ari'])}，描述性差值为 {f(new_E['acc']-old_E['acc'], signed=True)}/{f(new_E['nmi']-old_E['nmi'], signed=True)}/{f(new_E['ari']-old_E['ari'], signed=True)}。未铺开额外 B(k=0)；旧完整属性 0.423 来自 3 种子，不能与这里的两种子均值混用。

边界：只有 Cora、随机整节点缺失、3 掩码 × 2 种子、SDCN 适配版、固定 k=2。六组共享掩码/种子，E 仅两次训练，不能视为六个完全独立样本；每掩码仅一个随机 oracle 子集。

S² 统一了额外平滑，但 FP 仍包含 40 步传播，不能把全部收益严格归因为属性恢复。E−B 是此下游的**操作性上限诊断**，不是所有补全方法的数学上界。未开展第二数据集、AMGC、新方法或超参搜索。

## 8. 复查入口

- 执行：`run_oracle_ceiling.py`；复算与报告：`summarize_oracle_ceiling.py`。
- 配置与判读规则：`oracle_ceiling/config.json`；结果：`results.json`、`results.csv` 及 50 个运行子目录。
- 逐组差值：`paired_differences.csv`；统计：`paired_summary.csv`、`branch_means.csv`、`summary.json`。
- 共享 E 的掩码子群评价：`complete_mask_evaluations.json`；FP 重跑：`refcheck/comparison.json`。
- 冒烟独立保存在 `oracle_ceiling_smoke/`，不会覆盖全量或历史结果。
"""
    report_path = ROOT / args.report
    report_path.write_text(report, encoding="utf-8")
    print(json.dumps(dict(main_EB=main_EB, main_CB=main_CB, main_DB=main_DB,
        main_CD=main_CD, main_BA=main_BA, subgroups=subgroups,
        severe_collapses=collapses, report=str(report_path)), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
