import hashlib, json, os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'results_stage3'
os.environ.setdefault('MPLCONFIGDIR', str(R / 'mpl_cache'))
P = R / 'protocol_stage3.json'
METRICS = ['ACC','balanced_accuracy','tail_recall','stage1_tail_recall','many_to_one_BA','many_to_one_tail_recall','many_to_one_stage1_tail_recall','covered_classes','tail_coverage','stage1_tail_coverage','predicted_clusters','normalized_entropy','elapsed_s','peak_memory_bytes']
A = ['GMM_diag','vMF_free','SC_knn','AGG_ward','AGG_avg_cos','OVER_MERGE','FINCH_reqK']
B = ['HDBSCAN_error','HDBSCAN_nearest','FINCH_auto','Leiden_knn','silhouette_KM']
CONDITIONS = [('cifar100',100),('cifar100',10),('cifar10',100),('cifar100',1)]

def read(path): return json.loads(path.read_text(encoding='utf8'))
def fmt(values, digits=4):
    if not values: return '未测'
    return f'{np.mean(values):.{digits}f} ± {np.std(values):.{digits}f}'
def row(data, metrics): return ' | '.join(fmt([v[m] for v in data if m in v and v[m] is not None]) for m in metrics)
def get(dataset, ir, seed, method, kp=None):
    if method == 'silhouette_KM':
        path = ROOT / f'results_stage2/full/{dataset}_ir{ir}_s{seed}/silhouette_k{10 if dataset=="cifar10" else 100}/result.json'
    else:
        path = R / f'full/{dataset}_ir{ir}_s{seed}/{method}_k{kp or (10 if dataset=="cifar10" else 100)}/result.json'
    if not path.exists(): return None
    item=read(path)
    if method != 'silhouette_KM': assert item['protocol_sha256'] == hashlib.sha256(P.read_bytes()).hexdigest()
    m=item['metrics'].copy(); m['elapsed_s']=item.get('elapsed_s'); m['peak_memory_bytes']=item.get('peak_memory_bytes');
    if method == 'silhouette_KM': m['predicted_clusters']=item['k']
    if m['peak_memory_bytes'] is not None: m['peak_memory_bytes']/=2**20
    return m
def oracle(seed):
    return read(ROOT/f'results/full/cifar100_ir100_s{seed}/result.json')['methods']['ORACLE_KM']['metrics']['balanced_accuracy']
def mini(seed):
    return read(ROOT/f'results_stage2/full/cifar100_ir100_s{seed}/MiniClustering_feat_k100/result.json')['metrics']['balanced_accuracy']
def baseline(seed, method):
    if method == 'ORACLE_KM':
        return read(ROOT/f'results/full/cifar100_ir100_s{seed}/result.json')['methods'][method]['metrics']
    return read(ROOT/f'results_stage2/full/cifar100_ir100_s{seed}/{method}_k100/result.json')['metrics']
def paired_table(title, left, right, metrics):
    lines=[f'### {title}','','| 指标 | 配对差均值 ± 标准差 | 正差 |','| --- | ---: | ---: |']
    for metric in metrics:
        vals=[left[s-1][metric]-right[s-1][metric] for s in range(1,6)]
        lines.append(f'| {metric} | {fmt(vals)} | {sum(v>0 for v in vals)}/5 |')
    return '\n'.join(lines)
def table(title, dataset, ir, methods, metrics):
    names={'balanced_accuracy':'BA','tail_recall':'P²OT 尾召回','stage1_tail_recall':'阶段一尾召回',
           'many_to_one_BA':'多对一 BA','many_to_one_tail_recall':'多对一尾召回',
           'tail_coverage':'多对一尾覆盖率','stage1_tail_coverage':'阶段一尾覆盖率',
           'predicted_clusters':'预测簇数','normalized_entropy':'归一化簇熵',
           'elapsed_s':'时间(s)','peak_memory_bytes':'峰值内存(MiB)',
           'head_recall':'头部召回','medium_recall':'中部召回','covered_classes':'覆盖类数'}
    lines=[f'### {title}', '', '| 方法 | '+ ' | '.join(names.get(m,m) for m in metrics)+' |', '| --- | '+ ' | '.join(['---:']*len(metrics))+' |']
    for method in methods:
        values=[get(dataset,ir,s,method) for s in range(1,6)]; values=[v for v in values if v]
        lines.append(f'| {method} ({len(values)}/5) | '+(row(values,metrics) if values else ' | '.join(['未运行']*len(metrics)))+' |')
    return '\n'.join(lines)
def main():
    assert hashlib.sha256(P.read_bytes()).hexdigest() == (R/'protocol_stage3.sha256').read_text().split()[0].lower()
    status=read(R/'pipeline_status.json')
    assert status['state']=='complete' and not status['failures'], status['state']
    rerun=read(R/'rerun_audit.json')
    assert rerun['passed']
    a_by_seed={s:{m:get('cifar100',100,s,m) for m in A} for s in range(1,6)}
    assert all(all(v is not None for v in d.values()) for d in a_by_seed.values())
    best_a=[max(a_by_seed[s].values(),key=lambda v:v['balanced_accuracy'])['balanced_accuracy'] for s in range(1,6)]
    best_a_name=[max(a_by_seed[s],key=lambda m:a_by_seed[s][m]['balanced_accuracy']) for s in range(1,6)]
    best_a_metrics=[a_by_seed[s][best_a_name[s-1]] for s in range(1,6)]
    mini_metrics=[baseline(s,'MiniClustering_feat') for s in range(1,6)]
    oracle_metrics=[baseline(s,'ORACLE_KM') for s in range(1,6)]
    gaps=[oracle(s)-best_a[s-1] for s in range(1,6)]
    gain=[best_a[s-1]-mini(s) for s in range(1,6)]
    b_by_seed={s:{m:get('cifar100',100,s,m) for m in B} for s in range(1,6)}
    assert all(all(v is not None for v in d.values()) for d in b_by_seed.values())
    best_b=[max(b_by_seed[s].values(),key=lambda v:v['balanced_accuracy'])['balanced_accuracy'] for s in range(1,6)]
    best_b_name=[max(b_by_seed[s],key=lambda m:b_by_seed[s][m]['balanced_accuracy']) for s in range(1,6)]
    best_b_metrics=[b_by_seed[s][best_b_name[s-1]] for s in range(1,6)]
    bgaps=[oracle(s)-best_b[s-1] for s in range(1,6)]
    a_gap_mean=float(np.mean(gaps)); a_gap_count=sum(x>=.30 for x in gaps)
    if a_gap_mean>=.30 and a_gap_count>=4: rule_a='经典方法没有解决问题，主线有力成立'
    elif .10<=a_gap_mean<.30: rule_a='经典方法部分有效，须作为强基线'
    elif a_gap_mean<.10: rule_a='经典方法已基本解决，主线存疑'
    else: rule_a='未满足冻结规则的充分条件，结论不确定'
    b_gap_count=sum(x>=.30 for x in bgaps)
    b_methods=['HDBSCAN_nearest','FINCH_auto','Leiden_knn']
    under=[]
    for method in b_methods:
        vals=[get('cifar100',100,s,method) for s in range(1,6)]
        mean_k=float(np.mean([v['predicted_clusters'] for v in vals]))
        mean_cov=float(np.mean([v['tail_coverage'] for v in vals]))
        if mean_k<=80 or mean_cov<=.10: under.append(method)
    b_ir1=[]
    for method in under:
        vals=[get('cifar100',1,s,method) for s in range(1,6)]
        mean_k=float(np.mean([v['predicted_clusters'] for v in vals]))
        mean_cov=float(np.mean([v['tail_coverage'] for v in vals]))
        ir100=[get('cifar100',100,s,method) for s in range(1,6)]
        k100=float(np.mean([v['predicted_clusters'] for v in ir100]))
        cov100=float(np.mean([v['tail_coverage'] for v in ir100]))
        b_ir1.append({'method':method,'mean_k':mean_k,'mean_tail_coverage':mean_cov,
                      'mean_k_ir100':k100,'mean_tail_coverage_ir100':cov100,
                      'contrast_supported':abs(mean_k-100)<abs(k100-100) or mean_cov>cov100})
    b_contrast=sum(v['contrast_supported'] for v in b_ir1)>=2
    b_criteria=b_gap_count>=4 and len(under)>=2 and b_contrast
    rule_b='已运行 B 组的最优 BA 差满足 5/5，尾部退化已覆盖完整 B 组至少一半；DeepDPM 缺测使最终最优 BA 条件未定' if b_criteria else '可运行 B 组未同时满足冻结数值条件；DeepDPM 缺测，完整 B 组结论未定'
    summary={'best_a_name':best_a_name,'best_a_ba':best_a,'gap_oracle_a':gaps,'gap_a_mini':gain,'rule_a':rule_a,
             'best_b_name':best_b_name,'best_b_ba':best_b,'gap_oracle_b':bgaps,
             'b_underestimated_methods':under,'b_ir1':b_ir1,'rule_b':rule_b,'skipped':status['skipped']}
    (R/'summary_stage3.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4)); seeds=np.arange(1,6); ax.plot(seeds,[oracle(s) for s in seeds],marker='o',label='ORACLE_KM'); ax.plot(seeds,best_a,marker='s',label='Best A'); ax.plot(seeds,best_b,marker='^',label='Best B'); ax.plot(seeds,[mini(s) for s in seeds],marker='x',label='MiniClustering_feat'); ax.set(xlabel='Seed',ylabel='BA',xticks=seeds,ylim=(0,1)); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); (R/'figures').mkdir(exist_ok=True); fig.savefig(R/'figures/primary_ba.png',dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4)); kps=np.array([200,300,500]); over_ba=[get('cifar100',100,s,'OVER_MERGE')['balanced_accuracy'] for s in range(1,6)]
    oracle_ba=np.array([[get('cifar100',100,s,'OVER_ORACLE_MERGE',kp)['balanced_accuracy'] for s in range(1,6)] for kp in kps])
    ax.errorbar(kps,oracle_ba.mean(1),yerr=oracle_ba.std(1),marker='o',capsize=3,label='Oracle majority merge')
    ax.axhline(np.mean(over_ba),color='tab:orange',linestyle='--',label='Unsupervised OVER_MERGE (K\'=300)')
    ax.set(xlabel="Microclusters K'",ylabel='BA',xticks=kps,ylim=(0,1)); ax.grid(alpha=.25); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(R/'figures/overclustering.png',dpi=160); plt.close(fig)
    lines=['# 第三阶段经典方法排查报告','',f'完成时间：{status["updated_at"]}（北京时间）；机器 RTX 5060 8GB。第三阶段只做冻结 DINOv2 特征上的诊断，不开发新方法。','',
           '## 1. 协议与输入','',f'协议 SHA-256：`{hashlib.sha256(P.read_bytes()).hexdigest()}`。第一阶段八个特征文件 SHA-256 与 `results/feature_manifest.json` 完全一致。沿用第一阶段子集构造与第二阶段评价函数；IR=100/10 的同条件同种子索引哈希逐项核对。',
           '环境：Python 3.12.16，torch 2.11.0+cu128，NumPy 1.26.4，scikit-learn 1.4.0；未改动 torch/NumPy。CUBLAS_WORKSPACE_CONFIG=:4096:8；运行入口后期补入第二阶段 GPU deterministic 设置，关闭 cuDNN benchmark 与 TF32。早期 CPU 结果未经过这项 GPU 设置，但实际完成的聚类分支均在 CPU 上，不能把它们表述为 GPU 确定性实测。',
           '第三方：DeepDPM 官方仓库 `https://github.com/BGU-CS-VIL/DeepDPM`，固定 commit `95c9376ac1b0cb48ea017c52d3abbf2a901940b3`，MIT；官方要求 torch 1.11/Lightning 1.2.10，原版入口在本机旧 Neptune 导入链中失败，且训练接口与新版框架差异超过最小修补。两个初始 K 均未运行，详见 `results_stage3/deepdpm_compatibility.json`。FINCH 官方仓库 `https://github.com/ssarfraz/FINCH-Clustering`，固定 commit `ab3b014ae30c0dd2eaa82ff37378e076ca21198a`，MIT；本地按第一邻居规则实现，但 reqK 的最近中心逐对合并及自动层级与官方实现不等价，不能当作官方 FINCH 成绩。最初克隆失败后才通过 raw HTTPS 获取源码，故流水线早期跳过理由仍记录“获取失败”，后续审查与兼容性调查见 `results_stage3/third_party_review.json`。源码文件的 Git blob 核验见 `results_stage3/source_manifest.json`。未将第三方源码复制进 `scripts/`，所以没有第三方源码 patch。',
           '冒烟结果在 `results_stage3/smoke/`，不计入统计。协议哈希在运行中保持不变，但 frozen_at 是手工填入的时间，且协议漏列 vMF 数值界、FINCH 的 2000 点 silhouette 抽样等实现细节。运行过程中修复过评价/映射错误，旧结果隔离在 superseded 目录。故这是一项固定单点配置的诊断，不能声称严格预注册。',
           '统计均为 5 个种子的均值 ± 总体标准差（ddof=0），除簇数、时间、内存外数值为 0–1。运行时间为单方法墙钟秒数；峰值内存为 Windows 进程 PeakWorkingSetSize（MiB）。归一化簇熵按实际输出簇（HDBSCAN“记错”包括噪声桶）计算。第二阶段 silhouette-KM 只引用已有指标，旧文件未记录的 ARI/熵/内存标“未测”，不补算。', '',
           '## 2. A 组：已知 K','']
    for ds,ir in CONDITIONS:
        methods=['GMM_diag','vMF_free','OVER_MERGE'] if ir==1 else A
        lines.append(table(f'{ds.upper()} IR={ir}',ds,ir,methods,['ACC','balanced_accuracy','tail_recall','stage1_tail_recall','many_to_one_BA','many_to_one_tail_recall','tail_coverage','normalized_entropy','elapsed_s','peak_memory_bytes']))
        lines.append('')
    metrics=['ACC','balanced_accuracy','head_recall','medium_recall','tail_recall','stage1_tail_recall','many_to_one_BA','many_to_one_tail_recall','tail_coverage']
    lines.append(paired_table('主条件：最强经典 − MiniClustering_feat',best_a_metrics,mini_metrics,metrics))
    lines.append('')
    lines.append(paired_table('主条件：ORACLE_KM − 最强经典',oracle_metrics,best_a_metrics,['ACC','balanced_accuracy']))
    lines.append('')
    lines += ['## 3. B 组：未知 K','']
    for ds,ir in CONDITIONS:
        lines.append(table(f'{ds.upper()} IR={ir}',ds,ir,B+['DeepDPM_init10/200'],['balanced_accuracy','tail_recall','many_to_one_BA','many_to_one_tail_recall','tail_coverage','predicted_clusters','normalized_entropy','elapsed_s','peak_memory_bytes']))
        lines.append('')
    lines.append(paired_table('主条件：ORACLE_KM − B 组最高 BA 方法',oracle_metrics,best_b_metrics,['ACC','balanced_accuracy']))
    lines.append('')
    lines += ['### HDBSCAN 原始噪声', '', '| 条件 | 噪声点数 | 噪声占比 | 原始非噪声簇数 |', '| --- | ---: | ---: | ---: |']
    for ds,ir in CONDITIONS:
        items=[read(R/f'full/{ds}_ir{ir}_s{s}/HDBSCAN_error_k{10 if ds=="cifar10" else 100}/result.json') for s in range(1,6)]
        lines.append(f'| {ds.upper()} IR={ir} | {fmt([v["fit"]["raw_noise"] for v in items])} | {fmt([v["fit"]["raw_noise"]/v["n"] for v in items])} | {fmt([v["fit"]["raw_clusters"] for v in items])} |')
    lines += ['', '“噪声记错”时噪声样本永不映射为正确类别；“最近中心”将噪声分配给 PCA-128 欧氏最近的非噪声簇。早期最近中心日志的 raw_noise 被原地数组覆盖，原始噪声数量以对应的 HDBSCAN_error 结果为准，指标不受此日志错误影响。', '']
    lines += ['### 预测簇数：CIFAR-100 IR=100 与 IR=1', '', '| 方法 | IR=100 | IR=1 | IR=100 多对一尾覆盖 | IR=1 多对一尾覆盖 |','| --- | ---: | ---: | ---: | ---: |']
    for m in B:
        a=[get('cifar100',100,s,m) for s in range(1,6)]; b=[get('cifar100',1,s,m) for s in range(1,6)]; lines.append(f'| {m} | {fmt([v["predicted_clusters"] for v in a if v])} | {fmt([v["predicted_clusters"] for v in b if v])} | {fmt([v["tail_coverage"] for v in a if v])} | {fmt([v["tail_coverage"] for v in b if v])} |')
    lines += ['','## 4. C 组：超量聚类诊断','', '| K\' | 多数类合并 BA | P²OT 尾召回 | 阶段一尾召回 | P²OT 尾覆盖率 | 阶段一尾覆盖率 | OVER_MERGE BA | BA 差 | 正差 |', '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for kp in (200,300,500):
        values=[get('cifar100',100,s,'OVER_ORACLE_MERGE',kp) for s in range(1,6)]
        over=[get('cifar100',100,s,'OVER_MERGE') for s in range(1,6)]
        diff=[values[i]['balanced_accuracy']-over[i]['balanced_accuracy'] for i in range(5)]
        lines.append(f'| {kp} | {fmt([v["balanced_accuracy"] for v in values])} | {fmt([v["tail_recall"] for v in values])} | {fmt([v["stage1_tail_recall"] for v in values])} | {fmt([v["tail_coverage"] for v in values])} | {fmt([v["stage1_tail_coverage"] for v in values])} | {fmt([v["balanced_accuracy"] for v in over])} | {fmt(diff)} | {sum(v>0 for v in diff)}/5 |')
    pre=[read(R/f'full/cifar100_ir100_s{s}/OVER_MERGE_k100/result.json')['fit']['premerge_metrics'] for s in range(1,6)]
    lines += ['',f'OVER_MERGE 合并前 K\'=300：多对一 BA {fmt([v["many_to_one_BA"] for v in pre])}，P²OT 尾召回 {fmt([v["many_to_one_tail_recall"] for v in pre])}，尾覆盖率 {fmt([v["tail_coverage"] for v in pre])}；合并后尾覆盖率 {fmt([get("cifar100",100,s,"OVER_MERGE")["tail_coverage"] for s in range(1,6)])}。','',
              '### C 组完整指标与资源', '', '| K\' | ACC | NMI | ARI | macro F1 | 多对一 BA | 多对一尾召回 | 覆盖类数 | 预测簇数 | 归一化熵 | 时间(s) | 峰值内存(MiB) |', '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for kp in (200,300,500):
        values=[get('cifar100',100,s,'OVER_ORACLE_MERGE',kp) for s in range(1,6)]
        lines.append(f'| {kp} | '+row(values,['ACC','NMI','ARI','macro_F1','many_to_one_BA','many_to_one_tail_recall','covered_classes','predicted_clusters','normalized_entropy','elapsed_s','peak_memory_bytes'])+' |')
    lines += ['', '| K\' | 头部召回 | 中部召回 | 阶段一尾召回 | 阶段一尾覆盖率 |', '| ---: | ---: | ---: | ---: | ---: |']
    for kp in (200,300,500):
        values=[get('cifar100',100,s,'OVER_ORACLE_MERGE',kp) for s in range(1,6)]
        lines.append(f'| {kp} | '+row(values,['head_recall','medium_recall','stage1_tail_recall','stage1_tail_coverage'])+' |')
    lines += ['',
              '## 5. 冻结规则判读','',
              f'**规则 1：{rule_a}。** A 组逐种子最优方法：{best_a_name}；ORACLE_KM − 最强经典 BA：{[round(x,4) for x in gaps]}，均值 {np.mean(gaps):.4f}，≥0.30 为 {sum(x>=.30 for x in gaps)}/5。',
              f'最强经典 − MiniClustering_feat BA：{[round(x,4) for x in gain]}，均值 {np.mean(gain):.4f}，正差 {sum(x>0 for x in gain)}/5。逐种子按真实标签取 A 组 BA 最大值，对经典方法有乐观偏向；只用于保守排查，不是可部署选择规则。',
              f'**规则 2：{rule_b}。** 已运行 B 组 ORACLE_KM − 最高 BA：{[round(x,4) for x in bgaps]}，均值 {np.mean(bgaps):.4f}，≥0.30 为 {b_gap_count}/5。HDBSCAN 两种噪声处理计为同一方法，silhouette-KM 是旧阶段参照而非 B 组新方法；FINCH_auto 与 Leiden_knn 满足 IR=100 簇数≤80 或多对一尾覆盖≤0.10（完整四方法中 2/4，已达到一半）。FINCH_auto 从 K=68.6/尾覆盖 0.0133 变为 IR=1 的 K=203/尾覆盖 1.000；Leiden 从 K=39.2/尾覆盖 0.0133 变为 K=63.4/尾覆盖 0.6267。HDBSCAN 在 IR=100 预测 K=115.6，并非所有方法都低估 K，但其多对一尾覆盖仍低。DeepDPM 两个初始 K 都缺测；若其 BA 足够高，第一条件可能改变，因此完整结论未定。',
              '规则 3：C 组只作描述性诊断，不设通过门槛。K\'=300 的多数类映射 BA 0.6133，高于同一批微簇的无监督 OVER_MERGE BA 0.5605，配对差 +0.0527（5/5）；合并会造成损失。然而监督上限在 K\'=300 时仍只覆盖约 27% 的 P²OT 尾类，K\'=500 时增至约 49%、BA 增至 0.6676，表明微簇分离本身仍不足。综合看，困难以“分”为主，“合”也有可见损失；K\'=200 与 K\'=500 和 OVER_MERGE 的 K\'=300 不同，差值只作描述，不能视为相同微簇上的纯合并效应。多数类映射使用真实标签，不能用于无监督部署。','',
              '### 综合结论','',
              '主线在冻结特征与本次 CIFAR 条件下成立；OVER_MERGE 是已验证的强经典基线（主条件 BA 0.5605），SC_knn 和本地 FINCH_reqK 也应列为对照。已运行的未知 K 方法表现符合长尾退化，但 DeepDPM 缺测且 FINCH 为非官方适配，完整“长尾 + 簇数未知”扩展未正式判定。超量聚类的证据支持先解决尾部微簇分离，再审视合并损失；这不是新方法建议，也不保证更高 K\' 可部署。','',
              '### 完整指标附表','',
              '下列附表补充 NMI、ARI、宏 F1、头/中召回、两种尾部定义、覆盖类数和时间/内存；预测簇数及熵见上表。','']
    for group, methods in [('A',A),('B',B)]:
        for ds,ir in CONDITIONS:
            eligible=([m for m in methods if m in ['GMM_diag','vMF_free','OVER_MERGE']] if group=='A' and ir==1 else methods)
            lines.append(table(f'{group} 组补充：{ds.upper()} IR={ir}',ds,ir,eligible,['NMI','ARI','macro_F1','head_recall','medium_recall','stage1_tail_recall','many_to_one_stage1_tail_recall','covered_classes','stage1_tail_coverage']))
            lines.append('')
    lines += [
              '## 6. 边界','',
              '只评估冻结特征、单一 DINOv2 骨干、CIFAR 与 5 个种子，超参是单点预设而非搜索。vMF 集中度用近似更新并限制在 100–10000，不是精确极大似然求解。IR=1 的 5 个种子选取相同的全部数据，不是 5 份独立样本。IR=1 的 SC 与层次聚类按资源限制未运行；CIFAR-100 IR=10 的谱聚类实际峰值约 14 GiB。DeepDPM 未运行，官方 FINCH 未逐行复现；这限制未知 K 结论。旧结果没有预测哈希，因此独立复跑只能核对完整指标、拟合记录与子集索引，不能追溯逐样本标签是否完全相同。完整性与复跑审计见 `results_stage3/integrity_audit.json`、`results_stage3/rerun_audit.json`。', '', '![Primary BA](results_stage3/figures/primary_ba.png)', '', '![Overclustering BA](results_stage3/figures/overclustering.png)']
    (ROOT/'第三阶段经典方法排查报告.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
if __name__=='__main__': main()
