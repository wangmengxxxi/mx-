"""Analyze completed stage-two runs; never create a report from partial output."""
import importlib.util
import json
from pathlib import Path
import sys
from datetime import datetime, timezone, timedelta

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results_stage2"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_stage2 import evaluate, sha
from audit_stage2_inputs import subset, array_sha


def dump(name, obj):
    (RESULTS / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def stage1(dataset, ir, seed, method):
    group = ROOT / f"results/full/{dataset}_ir{ir}_s{seed}"
    recorded = json.loads((group / "result.json").read_text(encoding="utf-8"))
    yall = np.load(ROOT / f"data/features/{dataset}_train_labels.npy")
    k = 10 if dataset == "cifar10" else 100
    ids = subset(yall, k, ir, seed)
    assert array_sha(ids) == recorded["indices_array_sha256"]
    measured = recorded["methods"][method]["metrics"].copy()
    recalls = np.asarray(measured["recall_per_class"])
    counts = np.asarray(measured["true_counts"])
    ranked = np.lexsort((np.arange(k), -counts))
    h = int(round(k * 0.3))
    t = int(round(k * 0.3))
    measured["head_recall"] = float(recalls[ranked[:h]].mean())
    measured["medium_recall"] = float(recalls[ranked[h:k-t]].mean())
    measured["tail_recall"] = float(recalls[ranked[k-t:]].mean())
    measured["stage1_tail_recall"] = float(recalls[np.lexsort((np.arange(k), counts))[:k//3]].mean())
    return measured, recorded


def stage2(dataset, ir, seed, method, k=100, group="full"):
    path = RESULTS / group / f"{dataset}_ir{ir}_s{seed}/{method}_k{k}/result.json"
    return json.loads(path.read_text(encoding="utf-8"))


def stats(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "std": float(values.std(ddof=0)),
            "values": values.tolist()}


def main():
    status = json.loads((RESULTS / "pipeline_status.json").read_text(encoding="utf-8"))
    if status["state"] != "complete" or status["completed"] != status["total"]:
        raise SystemExit("Pipeline incomplete; report not generated")
    protocol_sha = sha(RESULTS / "protocol_stage2.json")
    assert protocol_sha == status["protocol_sha256"]
    metrics = ["ACC", "NMI", "balanced_accuracy", "macro_F1", "head_recall",
               "medium_recall", "tail_recall", "stage1_tail_recall"]
    branch_rows = []
    paired_rows = []
    primary = {}
    for dataset, ir in (("cifar100",100),("cifar100",10),("cifar10",100)):
        k = 100 if dataset == "cifar100" else 10
        by_method = {}
        methods = ["KM","ORACLE_KM","P2OT_full","P2OT_TRUEPRIOR","MiniClustering_feat"]
        if dataset == "cifar100" and ir == 100:
            methods += ["KM_ORACLE_INIT","P2OT_feat"]
        for method in methods:
            rows = []
            for seed in range(1,6):
                if method in ("KM","ORACLE_KM","P2OT_feat"):
                    measured, _ = stage1(dataset, ir, seed, method)
                else:
                    measured = stage2(dataset, ir, seed, method, k)["metrics"]
                rows.append(measured)
            by_method[method] = rows
            branch_rows.append({"dataset":dataset,"ir":ir,"method":method,
                                **{metric:stats([r[metric] for r in rows]) for metric in metrics}})
        contrasts = [("P2OT_TRUEPRIOR","P2OT_full"),("P2OT_full","KM"),
                     ("MiniClustering_feat","KM"),("ORACLE_KM","P2OT_full")]
        if dataset == "cifar100" and ir == 100:
            contrasts += [("KM_ORACLE_INIT","ORACLE_KM"),("P2OT_full","P2OT_feat"),
                          ("ORACLE_KM","MiniClustering_feat")]
        for left,right in contrasts:
            for metric in metrics:
                diffs = np.array([a[metric]-b[metric] for a,b in zip(by_method[left],by_method[right])])
                paired_rows.append({"dataset":dataset,"ir":ir,"contrast":f"{left}-{right}",
                                    "metric":metric,**stats(diffs),"positive":int((diffs>0).sum())})
        if dataset == "cifar100" and ir == 100:
            primary = by_method
    k_rows = []
    k_data = {}
    for ir in (100,1):
        k_data[ir] = {}
        for method in ("KM","P2OT_full","MiniClustering_feat"):
            k_data[ir][method] = {}
            for k in (50,70,90,100,110,130):
                rows = []
                for seed in range(1,6):
                    if k == 100 and ir == 100:
                        if method == "KM":
                            measured = stage2("cifar100",ir,seed,method,100)["metrics"]
                        else:
                            measured = stage2("cifar100",ir,seed,method,100)["metrics"]
                    else:
                        measured = stage2("cifar100",ir,seed,method,k)["metrics"]
                    rows.append(measured)
                k_data[ir][method][k] = rows
                k_rows.append({"ir":ir,"method":method,"k":k,
                               **{metric:stats([r[metric] for r in rows]) for metric in
                                  ("balanced_accuracy","NMI","head_recall","medium_recall",
                                   "tail_recall","stage1_tail_recall","many_to_one_BA",
                                   "many_to_one_tail_recall","covered_classes","tail_coverage")}})
    silhouette = []
    for ir in (100,1):
        for seed in range(1,6):
            row = stage2("cifar100",ir,seed,"silhouette",100)
            silhouette.append({"ir":ir,"seed":seed,"estimated_k":row["k"],
                               "metrics":row["metrics"],"scores":row["fit"]["scores"]})
    gap = []
    prior = []
    for seed in range(5):
        best = max(primary[m][seed]["balanced_accuracy"] for m in ("P2OT_full","MiniClustering_feat"))
        gap.append(primary["ORACLE_KM"][seed]["balanced_accuracy"]-best)
        prior.append(primary["P2OT_TRUEPRIOR"][seed]["balanced_accuracy"]-
                     primary["P2OT_full"][seed]["balanced_accuracy"])
    k_rule = {}
    for ir in (100,1):
        k_rule[ir] = {}
        for method in ("KM","P2OT_full"):
            drops = []
            for at70,at100 in zip(k_data[ir][method][70],k_data[ir][method][100]):
                drops.append({"tail":at100["tail_recall"]-at70["tail_recall"],
                              "head":at100["head_recall"]-at70["head_recall"]})
            td = np.array([v["tail"] for v in drops])
            hd = np.array([v["head"] for v in drops])
            count = int(((td>=0.05)&(td>2*hd)).sum())
            k_rule[ir][method] = {"drops":drops,"mean_tail_drop":float(td.mean()),
                                  "mean_head_drop":float(hd.mean()),"per_seed_count":count,
                                  "satisfies":bool(td.mean()>=0.05 and td.mean()>2*hd.mean() and count>=4)}
    rule1 = bool(np.mean(gap)>=0.10 and np.sum(np.array(gap)>0)>=4)
    rule2 = bool(np.mean(prior)>=0.05 and np.sum(np.array(prior)>0)>=4)
    rule3 = bool(any(k_rule[100][m]["satisfies"] for m in ("KM","P2OT_full")) and
                 not any(k_rule[1][m]["satisfies"] for m in ("KM","P2OT_full")))
    decision = {"rule1":rule1,"rule2":rule2,"rule3":rule3,
                "oracle_gap":stats(gap),"oracle_gap_positive":int((np.array(gap)>0).sum()),
                "prior_gain":stats(prior),"prior_gain_positive":int((np.array(prior)>0).sum()),
                "k_rule":k_rule}
    dump("branch_summary.json",branch_rows)
    dump("paired_summary.json",paired_rows)
    dump("k_sensitivity_summary.json",k_rows)
    dump("silhouette_summary.json",silhouette)
    dump("decision.json",decision)
    make_plot(k_rows)
    make_report(branch_rows, paired_rows, k_rows, silhouette, decision)


def make_plot(rows):
    import os
    os.environ["MPLCONFIGDIR"] = str(ROOT / "data/matplotlib_cache")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for i, ir in enumerate((100,1)):
        for method in ("KM","P2OT_full","MiniClustering_feat"):
            selected = sorted((r for r in rows if r["ir"]==ir and r["method"]==method), key=lambda r:r["k"])
            x = [r["k"] for r in selected]
            axes[i,0].plot(x,[r["balanced_accuracy"]["mean"] for r in selected],marker="o",label=method)
            line = axes[i,1].plot(x,[r["tail_recall"]["mean"] for r in selected],marker="o",label=f"{method} tail")[0]
            axes[i,1].plot(x,[r["head_recall"]["mean"] for r in selected],linestyle=":",
                           color=line.get_color(),label=f"{method} head")
        axes[i,0].set_ylabel(f"IR={ir}: BA")
        axes[i,1].set_ylabel(f"IR={ir}: tail recall")
        for ax in axes[i]:
            ax.axvline(100,color="gray",linestyle="--",linewidth=1)
            ax.grid(alpha=0.3)
            ax.set_xlabel("K")
    axes[0,0].legend(fontsize=8)
    axes[0,1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    out = RESULTS / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "k_sensitivity.png",dpi=160)
    plt.close(fig)


def fmt(stat):
    return f"{stat['mean']:.4f} ± {stat['std']:.4f}"


def make_report(branches, paired, krows, silhouette, decision):
    audit = json.loads((RESULTS / "input_audit.json").read_text(encoding="utf-8"))
    rerun = json.loads((RESULTS / "rerun_audit.json").read_text(encoding="utf-8"))
    peak = 0
    for path in (RESULTS / "full").rglob("result.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        peak = max(peak, record.get("fit", {}).get("peak_allocated_bytes", 0))
    assert not audit["failures"]
    lines = ["# 第二阶段长尾聚类诊断报告", "",
             f"生成日期：{datetime.now(timezone(timedelta(hours=8))).date()}（北京时间）。本次只做诊断，不开发新方法。指标 0–1，标准差 ddof=0。",
             "", "## 1. 冻结协议与判读", "",
             f"协议 SHA-256 `{sha(RESULTS/'protocol_stage2.json')}`。流水线完成 232/232 项后生成本报告。",
             "", "| 规则 | 均值差 | 正差 | 判读 |", "| --- | ---: | ---: | --- |",
             f"| 1 ORACLE_KM − 最强特征方法 ≥0.10 | {decision['oracle_gap']['mean']:+.4f} | {decision['oracle_gap_positive']}/5 | {'满足' if decision['rule1'] else '不满足'} |",
             f"| 2 TRUEPRIOR − P2OT_full ≥0.05 | {decision['prior_gain']['mean']:+.4f} | {decision['prior_gain_positive']}/5 | {'满足' if decision['rule2'] else '不满足'} |",
             f"| 3 K=70 尾部不对称下降，IR=1 不同样发生 | 见下表 | — | {'满足' if decision['rule3'] else '不满足'} |",
             "", "规则 1/2 还要求至少 4/5 严格正差；规则 3 的均值和逐种子条件按冻结协议执行。",
             "规则 1 五个配对差：" + ", ".join(f"{v:+.4f}" for v in decision["oracle_gap"]["values"]) + "。",
             "规则 2 五个配对差：" + ", ".join(f"{v:+.4f}" for v in decision["prior_gain"]["values"]) + "。",
             "", "## 2. 机器、数据与复现边界", "",
             "本机 RTX 5060 8GB，驱动 610.88，CUDA 13.3（驱动显示），Python 3.12.16，torch 2.11.0+cu128（运行时 CUDA 12.8）。第一阶段为 GTX 1650 Ti。用户要求沿用本机已有 PyTorch，因此与原计划的 Python 3.9.21/torch 2.8.0 有版本偏差；基础旧环境未改。GPU sm_120 的确定性矩阵乘法与卷积均通过，TF32 和 cuDNN benchmark 关闭。",
             f"训练分支记录的最大 torch CUDA allocated 峰值 {peak/1024**2:.1f} MiB（不含 CUDA 上下文和非 PyTorch 分配）；环境与依赖见 `environment_gpu_5060.txt`。",
             f"8/8 个特征文件 SHA-256 与第一阶段清单一致；30/30 个子集索引哈希一致；CIFAR-100 官方归档哈希 `{audit['cifar100_archive_sha256']}` 一致。数据与第三方源码未提交。",
             "", "## 3. 论文配置与本次差异", "",
             "| 项目 | 原论文/官方代码 | 本次 |", "| --- | --- | --- |",
             "| P²OT 骨干 | DINO ViT-B/16，微调最后一个 block（附录 F.1） | 冻结 DINOv2 ViT-B/14 的 768 维 CLS；无法微调 block |",
             "| P²OT 头与训练 | CIFAR100 两个余弦头；50 epoch，Adam，batch 512，lr 5e-4 余弦衰减；ε=0.1、λ/γ=1、ρ 从 0.1 sigmoid 升温，5120 logits memory bank | 保留两头、OT 求解器、日程、batch 与 bank；相同冻结特征作两视图 |",
             "| P²OT 增强与选择 | Resize256/RandomCrop224、水平翻转、ColorJitter；按训练损失/ρ 选头和 epoch | 单次中心裁剪特征，无图像随机增强；仍按训练损失/ρ 选择 |",
             "| TRUEPRIOR 诊断 | P²OT 原法每簇目标质量 ρ/K | 使用真实类别大小降序列表；在第 1 轮后按无标签预测簇占用量排序配给簇编号，之后固定，不提供哪簇对应哪一真实类 |",
             "| MiniClustering 骨干 | ResNet-18，BYOL 同长尾数据预训练 800 epoch，随后 encoder 与双线性头联合训练 200 epoch | 冻结 DINOv2 特征，只训双线性头 200 epoch；无 BYOL、图像三视图及 encoder 更新 |",
             "| MiniClustering 参数 | batch 512、Adam lr 1e-5、阈值 0.99；M、δ、α、β 随数据条件变化（附录 C） | M=2K、δ=0.3、α=β=1 预先固定；CIFAR100 细类没有原文对应配置 |",
             "| MiniClustering_official | 论文表 1/2 是 CIFAR-10/20/STL-10 的 IR=5/10；CIFAR-20 是 CIFAR-100 超类 | 跳过 CIFAR-100 细类官方版：固定代码无该入口、存在 TabError，且需额外 800 epoch BYOL 预训练；不能声称原论文配置 |",
             "| 许可证 | PPOT 固定 commit 含 CC BY-NC 4.0 LICENSE；MiniClustering 固定 commit 无 LICENSE | 仅本地诊断；不再分发第三方源码；运行只导入已审查的 PPOT 求解器 |",
             "", "P²OT 论文正文写学习率衰减到 5e-6；固定代码的默认余弦调度实际 eta_min=5e-7，本次遵循代码默认值。P²OT 论文 ICLR 2024 表 1：CIFAR100-LT R=100 ACC 38.2±0.8%、NMI 69.6±0.3%、F1 32.0±0.9%。MiniClustering 论文 ICLR 2026 表 2：CIFAR-20 IR=10 ACC 44.3%、CAA 41.5%、NMI 47.2%、ARI 30.4%；附录表 6：CIFAR-10 IR=100 ACC 46.9%、CAA 44.6%、NMI 54.0%、ARI 34.6%。没有 CIFAR-100 细类论文数字。这些不同骨干和类别定义的成绩不作为直接复现阈值。",
             "", "## 4. A/B 分支（均值 ± 标准差）", "",
             "| 数据 | IR | 方法 | ACC | BA | NMI | macro-F1 | 头 | 中 | 尾 | 阶段一尾 |",
             "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in branches:
        lines.append("| {dataset} | {ir} | {method} | {ACC} | {BA} | {NMI} | {F1} | {head} | {medium} | {tail} | {stage1} |".format(
            dataset=r["dataset"],ir=r["ir"],method=r["method"],ACC=fmt(r["ACC"]),BA=fmt(r["balanced_accuracy"]),
            NMI=fmt(r["NMI"]),F1=fmt(r["macro_F1"]),head=fmt(r["head_recall"]),
            medium=fmt(r["medium_recall"]),tail=fmt(r["tail_recall"]),stage1=fmt(r["stage1_tail_recall"])))
    lines += ["", "主条件配对差（逐种子原始值见 `paired_summary.json`）：", "",
              "| 对比 | 指标 | 均差 ± 标准差 | 正差 |", "| --- | --- | ---: | ---: |"]
    for r in paired:
        if r["dataset"]=="cifar100" and r["ir"]==100:
            lines.append(f"| {r['contrast']} | {r['metric']} | {fmt(r)} | {r['positive']}/5 |")
    lookup = {(r["dataset"], r["ir"], r["method"]): r for r in branches}
    def main_ba(method):
        return lookup[("cifar100", 100, method)]["balanced_accuracy"]["mean"]
    oracle_entropy = []
    init_entropy = []
    true_entropy = []
    for seed in range(1, 6):
        oracle, _ = stage1("cifar100", 100, seed, "ORACLE_KM")
        init = stage2("cifar100", 100, seed, "KM_ORACLE_INIT", 100)["metrics"]
        for source, dest in ((oracle["aligned_cluster_counts"], oracle_entropy),
                             (init["cluster_counts"], init_entropy),
                             (oracle["true_counts"], true_entropy)):
            p = np.asarray(source, dtype=float)
            p = p[p > 0] / p.sum()
            dest.append(float(-np.sum(p*np.log(p))/np.log(100)))
    lines += ["", f"KM_ORACLE_INIT 收敛后 BA 相对 ORACLE_KM 为 {main_ba('KM_ORACLE_INIT')-main_ba('ORACLE_KM'):+.4f}；P2OT_full 相对第一阶段简化 P2OT_feat 为 {main_ba('P2OT_full')-main_ba('P2OT_feat'):+.4f}。MiniClustering_official 未运行，因此不能给出与特征版的数值差。",
              f"与论文已报告数字作描述性比较：本次 CIFAR-100 IR=100 P2OT_full 的 ACC 均值相对论文表 1 的 0.382 为 {lookup[('cifar100',100,'P2OT_full')]['ACC']['mean']-0.382:+.4f}，NMI 相对 0.696 为 {lookup[('cifar100',100,'P2OT_full')]['NMI']['mean']-0.696:+.4f}；骨干、增强、微调和聚类评价设置均不同，不能解释为同协议复现误差。CIFAR-10 IR=100 的 MiniClustering_feat ACC 均值相对论文附录表 6 的 0.469 为 {lookup[('cifar10',100,'MiniClustering_feat')]['ACC']['mean']-0.469:+.4f}，同样仅作描述。",
              f"归一化簇大小熵：真实 {np.mean(true_entropy):.4f}，ORACLE_KM 零次更新 {np.mean(oracle_entropy):.4f}，KM_ORACLE_INIT 收敛后 {np.mean(init_entropy):.4f}（1 表示完全均匀）。此比较结合 BA 差判断 Lloyd 目标是否改变长尾分配。",
              f"独立完整 P2OT_full 复跑预测哈希{'一致' if rerun['P2OT_full']['prediction_hash_equal'] else '不一致'}；KM CPU 复跑预测哈希{'一致' if rerun['KM']['prediction_hash_equal'] else '不一致'}，BA 绝对差 {rerun['KM']['BA_abs_diff']:.8f}、inertia 绝对差 {rerun['KM']['inertia_abs_diff']:.8f}；ORACLE_KM 单独复算预测哈希{'一致' if rerun['ORACLE_KM']['prediction_hash_equal'] else '不一致'}，BA 绝对差 {rerun['ORACLE_KM']['BA_abs_diff']:.8f}。", ""]
    lines += ["", "## 5. K 敏感性", "",
              "矩形 Hungarian 的未匹配类召回为 0；多对一评价将每簇映到多数类，用于区分拆分与合并。",
              "", "| IR | 方法 | K | BA | 头召回 | 论文尾召回 | 阶段一尾召回 | 多对一 BA | 多对一尾召回 | 覆盖类数 | 尾覆盖 | NMI |",
              "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in krows:
        lines.append(f"| {r['ir']} | {r['method']} | {r['k']} | {fmt(r['balanced_accuracy'])} | {fmt(r['head_recall'])} | {fmt(r['tail_recall'])} | {fmt(r['stage1_tail_recall'])} | {fmt(r['many_to_one_BA'])} | {fmt(r['many_to_one_tail_recall'])} | {fmt(r['covered_classes'])} | {fmt(r['tail_coverage'])} | {fmt(r['NMI'])} |")
    lines += ["", "![K 敏感性曲线](results_stage2/figures/k_sensitivity.png)", "",
              "K=70 相对 K=100 的配对下降：", "",
              "| IR | 方法 | 尾部均降 | 头部均降 | 满足逐种子条件 | 判读 |",
              "| ---: | --- | ---: | ---: | ---: | --- |"]
    for ir in (100,1):
        for method in ("KM","P2OT_full"):
            r = decision["k_rule"][ir][method]
            lines.append(f"| {ir} | {method} | {r['mean_tail_drop']:+.4f} | {r['mean_head_drop']:+.4f} | {r['per_seed_count']}/5 | {'满足' if r['satisfies'] else '不满足'} |")
    lines += ["", "逐种子 K=100 减 K=70 召回差（尾/头）：", ""]
    for ir in (100,1):
        for method in ("KM","P2OT_full"):
            drops = decision["k_rule"][ir][method]["drops"]
            lines.append(f"- IR={ir} {method}: " + ", ".join(f"{v['tail']:+.4f}/{v['head']:+.4f}" for v in drops))
    lines += ["", "## 6. 无标签 K 估计", "",
              "在 L2 特征上用欧氏距离，固定 2000 点和 K=20..200（步长 10）最大 silhouette 选择。",
              "", "| IR | seed1 | seed2 | seed3 | seed4 | seed5 | K 均值 | 聚类 BA | 聚类 NMI |",
              "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for ir in (100,1):
        selected = [r for r in silhouette if r["ir"]==ir]
        ks = [r["estimated_k"] for r in selected]
        ba = stats([r["metrics"]["balanced_accuracy"] for r in selected])
        nmi = stats([r["metrics"]["NMI"] for r in selected])
        lines.append(f"| {ir} | " + " | ".join(str(v) for v in ks) + f" | {np.mean(ks):.1f} | {fmt(ba)} | {fmt(nmi)} |")
    k100 = np.mean([r["estimated_k"] for r in silhouette if r["ir"]==100])
    k1 = np.mean([r["estimated_k"] for r in silhouette if r["ir"]==1])
    lines += ["", f"IR=100 的估计 K 均值相对 IR=1 为 {k100-k1:+.1f}；此处只作描述性比较，没有预设显著性门槛。"]
    lines += ["", "## 7. 判断边界", "",
              "单一 DINOv2 骨干、以冻结特征为主、无法复现图像随机增强、仅 CIFAR、5 个配对种子。ORACLE_KM 使用真实类中心且零次 Lloyd 更新，只是受监督诊断参照，不是数学上界；KM_ORACLE_INIT 使用真实中心作初始化。TRUEPRIOR 使用真实类大小，未告知簇和类的对应关系，不能用于部署。IR=1 的五个种子共用完整数据集，不是五个独立样本。MiniClustering_feat 无原论文 CIFAR100 细类配置，其成绩只表明这一冻结特征适配的行为。",
              "", "完整逐种子结果、训练损失、配对差、silhouette 曲线数据和运行日志位于 `results_stage2/`。所有负结果与偏差均保留。",
              "", "## 参考来源", "",
              "- Zhang, Ren, He. [P²OT: Progressive Partial Optimal Transport for Deep Imbalanced Clustering](https://openreview.net/forum?id=hD3sGVqPsr), ICLR 2024，表 1 与附录 F.1。",
              "- Li et al. [Mini-cluster Guided Long-tailed Deep Clustering](https://openreview.net/forum?id=3JlljaiQwR), ICLR 2026，表 1/2、附录表 6 与附录 C。",
              "- 第一阶段冻结协议和结果：`results/protocol.json`、`results/full/`；第三方固定 commit 与审查见 `results_stage2/third_party_review.json`。", ""]
    (ROOT / "第二阶段长尾聚类诊断报告.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
