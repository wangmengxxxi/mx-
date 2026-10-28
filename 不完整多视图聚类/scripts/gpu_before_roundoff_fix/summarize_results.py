# -*- coding: utf-8 -*-
"""Deterministic tables and report from fixed-final-epoch stored runs only."""
from pathlib import Path
import argparse,csv,hashlib,json,statistics,sys
from collections import defaultdict
import numpy as np

METRICS=['ACC','NMI','ARI']
BRANCHES=['A','B','E','C_low_s1','C_low_s2','C_rand']
DIFFS=[('B','A'),('E','B'),('C_low_s1','B'),('C_low_s2','B'),('C_rand','B'),('C_low_s1','C_rand'),('C_low_s2','C_rand')]
PAPER={'Caltech101-20':dict(ACC=.6844,NMI=.6739,ARI=.7544),'Scene-15':dict(ACC=.395,NMI=.4235,ARI=.2351)}

def write_csv(path,rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--partial',action='store_true');a=ap.parse_args();root=a.root.resolve();results=root/'results'
    runs=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((results/'full').glob('*/result.json'))]
    expected={(d,r,s) for d in PAPER for r in [.5,.7] for s in range(1,6)}
    actual={(r['dataset'],r['rate'],r['data_seed']) for r in runs}
    assert len(actual)==len(runs) and actual<=expected
    if not a.partial:assert actual==expected,(len(runs),expected-actual)
    assert all(r['epochs']==500 for r in runs)
    devices={r.get('device','cpu') for r in runs}
    assert len(devices)<=1,'Refusing mixed CPU/GPU formal statistics'
    device=next(iter(devices),'cuda' if (results/'gpu_check/comparison.json').exists() else 'cpu')
    gpu=device=='cuda'
    device_label='GPU' if gpu else 'CPU'
    gpu_check=json.loads((results/'gpu_check/comparison.json').read_text(encoding='utf-8')) if gpu else None
    if gpu:assert gpu_check['passed']
    groups=defaultdict(list)
    for r in runs:groups[r['dataset'],r['rate']].append(r)
    branches=[];paired=[];pairs=[];correlations=[];corr_summary=[];repro=[];decisions=[]
    for (d,rate),rs in sorted(groups.items()):
        rs.sort(key=lambda r:r['data_seed'])
        for b in BRANCHES:
            row=dict(dataset=d,rate=rate,branch=b,n=len(rs))
            for m in METRICS:
                vals=[r['branches'][b][m] for r in rs];row[m+'_mean']=statistics.mean(vals);row[m+'_std_population']=statistics.pstdev(vals)
            branches.append(row)
        for x,y in DIFFS:
            name=x+'-'+y;row=dict(dataset=d,rate=rate,comparison=name,n=len(rs))
            for m in METRICS:
                vals=[r['branches'][x][m]-r['branches'][y][m] for r in rs]
                row[m+'_mean_diff']=statistics.mean(vals);row[m+'_positive']=sum(v>0 for v in vals);row[m+'_zero']=sum(v==0 for v in vals);row[m+'_min']=min(vals);row[m+'_max']=max(vals)
            pairs.append(row)
            for r in rs:
                p=dict(dataset=d,rate=rate,data_seed=r['data_seed'],comparison=name)
                p.update({m:r['branches'][x][m]-r['branches'][y][m] for m in METRICS});paired.append(p)
        for s in ['s1','s2']:
            for r in rs:correlations.append(dict(dataset=d,rate=rate,data_seed=r['data_seed'],score=s,**r['correlations'][s]))
            rr=[r['correlations'][s]['spearman'] for r in rs]
            defined=[x for x in rr if x is not None]
            row=dict(dataset=d,rate=rate,score=s,n=len(rs),undefined_count=len(rr)-len(defined),mean_spearman=statistics.mean(rr) if len(defined)==len(rr) else None,min_spearman=min(defined) if defined else None,max_spearman=max(defined) if defined else None,coefficients=rr)
            corr_summary.append(row)
        def getpair(c):return next(x for x in pairs if x['dataset']==d and x['rate']==rate and x['comparison']==c)
        ceiling=getpair('E-B');completion=getpair('B-A');valid_scores=[]
        for s in ['s1','s2']:
            cs=next(x for x in corr_summary if x['dataset']==d and x['rate']==rate and x['score']==s)
            diff=getpair('C_low_'+s+'-C_rand')
            if cs['mean_spearman'] is not None and cs['mean_spearman']<=-.3 and diff['ACC_mean_diff']>=.01 and diff['ACC_positive']>=4 and len(rs)==5:valid_scores.append(s)
        has_ceiling=len(rs)==5 and ceiling['ACC_mean_diff']>=.03 and ceiling['ACC_positive']>=4
        has_completion=len(rs)==5 and completion['ACC_mean_diff']>0 and completion['ACC_positive']>=4
        verdict='通过' if has_ceiling and valid_scores else '部分通过' if has_ceiling else '不通过' if len(rs)==5 and ceiling['ACC_mean_diff']<.01 else '未落入预设三类，证据不足'
        decisions.append(dict(dataset=d,rate=rate,n=len(rs),completion=has_completion,ceiling=has_ceiling,reliable_scores=valid_scores,verdict=verdict))
        if rate==.5:
            row=dict(dataset=d,n=len(rs))
            for m in METRICS:
                mean=statistics.mean(r['branches']['B'][m] for r in rs);row[m+'_paper']=PAPER[d][m];row[m+'_reproduced']=mean;row[m+'_diff']=mean-PAPER[d][m]
            row['ACC_acceptable_approx_3pp']=len(rs)==5 and abs(row['ACC_diff'])<=.03;repro.append(row)
    masks=[dict(dataset=r['dataset'],rate=r['rate'],data_seed=r['data_seed'],model_seed=r['model_seed'],actual_missing_rate=r['actual_missing_rate'],missing_count=r['missing_count'],replacement_count=r['replacement_count'],mask_sha256=r['mask_sha256'],training_seconds=r['training_seconds']) for r in runs]
    for row,r in zip(masks,runs):
        for key in ['device','torch_version','cuda_version','gpu_name','peak_memory_allocated_bytes','peak_memory_reserved_bytes','warning_count']:
            row[key]=r.get(key)
    for filename,rows in [('branch_means.csv',branches),('paired_differences.csv',paired),('paired_summary.csv',pairs),('reliability_correlations.csv',correlations),('reliability_summary.csv',corr_summary),('reproduction.csv',repro),('mask_metadata.csv',masks)]:write_csv(results/filename,rows)
    primary=[x for x in decisions if x['rate']==.5]
    if actual!=expected:overall='实验进行中，暂不判读'
    elif any(x['verdict']=='通过' for x in primary):overall='通过'
    elif any(x['ceiling'] for x in primary):overall='部分通过'
    elif all(x['verdict']=='不通过' for x in primary):overall='不通过'
    else:overall='未落入预设三类，证据不足'
    implication={'通过':'工作一有继续研究的诊断依据，但尚未验证任何可部署的新方法。','部分通过':'补全仍有操作性上限空间，当前可靠性信号未能满足向聚类收益转化的全部条件；需要重新考虑信号设计或切入角度。','不通过':'在本次 COMPLETER、固定最终轮和模拟随机缺失协议下，oracle 额外空间很小，当前证据不支持直接把选择性补全作为已获验证的工作一；这不等于否定全部 IMVC baseline 或缺失设定。','未落入预设三类，证据不足':'原规则没有覆盖本次中间情形，保留证据不足，不事后修改阈值或强行贴入三类。'}.get(overall,'全量运行尚未结束，不提前作可行性结论。')
    summary=dict(runs=len(runs),expected_runs=20,device=device,overall=overall,decisions=decisions,branches=branches,pairs=pairs,correlations=corr_summary,reproduction=repro)
    (results/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    inventory=json.loads((results/'source_inventory.json').read_text(encoding='utf-8'))
    def f(x):return f'{x:.4f}'
    def signed(x):return f'{x:+.4f}'
    branch_table=table(['数据集','缺失率','分支','ACC 均值±标准差','NMI 均值±标准差','ARI 均值±标准差','训练数'],[[r['dataset'],r['rate'],r['branch'],*[f(r[m+'_mean'])+'±'+f(r[m+'_std_population']) for m in METRICS],r['n']] for r in branches])
    paired_table=table(['数据集','缺失率','对比','ACC 平均差 / 正差','NMI 平均差 / 正差','ARI 平均差 / 正差'],[[r['dataset'],r['rate'],r['comparison'],*[signed(r[m+'_mean_diff'])+f" / {r[m+'_positive']}/{r['n']}" for m in METRICS]] for r in pairs])
    corr_table=table(['数据集','缺失率','得分','Spearman 均值','5 次系数（按种子 1–5）'],[[r['dataset'],r['rate'],r['score'],f(r['mean_spearman']) if r['mean_spearman'] is not None else '未定义',', '.join(f(x) if x is not None else '未定义' for x in r['coefficients'])] for r in corr_summary])
    repro_table=table(['数据集','ACC 论文 / 本次 / 差','NMI 论文 / 本次 / 差','ARI 论文 / 本次 / 差','ACC 约 3pp 内'],[[r['dataset'],*[f(r[m+'_paper'])+' / '+f(r[m+'_reproduced'])+' / '+signed(r[m+'_diff']) for m in METRICS],'是' if r['ACC_acceptable_approx_3pp'] else '否'] for r in repro])
    decision_table=table(['数据集','缺失率','补全有意义','上限空间','可靠性通过得分','按规则判读'],[[r['dataset'],r['rate'],'是' if r['completion'] else '否','是' if r['ceiling'] else '否',','.join(r['reliable_scores']) or '无',r['verdict']] for r in decisions])
    audit=json.loads((results/'audit.json').read_text(encoding='utf-8')) if (results/'audit.json').exists() else None
    ref=[]
    for p in (results/'refcheck').glob('*/result.json'):
        rr=json.loads(p.read_text(encoding='utf-8'));old=next(r for r in runs if (r['dataset'],r['rate'],r['data_seed'])==(rr['dataset'],rr['rate'],rr['data_seed']))
        assert rr.get('device','cpu')==old.get('device','cpu')
        delta={b:{m:rr['branches'][b][m]-old['branches'][b][m] for m in METRICS} for b in BRANCHES}
        arrays1=np.load(p.parent/'diagnostic_arrays.npz');arrays0=np.load(results/'full'/p.parent.name/'diagnostic_arrays.npz')
        equal={key:bool(np.array_equal(arrays1[key],arrays0[key])) for key in arrays0.files}
        ref.append(dict(run=p.parent.name,branch_metric_deltas=delta,array_equal=equal,mask_hash_equal=old['mask_sha256']==rr['mask_sha256'],input_hash_equal=old['input_sha256']==rr['input_sha256']))
    (results/'refcheck').mkdir(exist_ok=True)
    (results/'refcheck/comparison.json').write_text(json.dumps(ref,ensure_ascii=False,indent=2),encoding='utf-8')
    reftext='尚未完成独立完整复跑。'
    if ref:
        item=ref[0];maxdelta=max(abs(v) for b in item['branch_metric_deltas'].values() for v in b.values())
        reftext=f"完整复跑 `{item['run']}`：掩码哈希相同={item['mask_hash_equal']}；所有分支输入哈希相同={item['input_hash_equal']}；{sum(item['array_equal'].values())}/{len(item['array_equal'])} 个存档数组逐元素一致；六分支 ACC/NMI/ARI 最大绝对差 {maxdelta:.12g}。详见 `results/refcheck/comparison.json`。"
    times=[r['training_seconds'] for r in runs]
    runtime=f"训练耗时范围 {min(times)/60:.2f}–{max(times)/60:.2f} 分钟，{len(runs)} 次训练累计{device_label}进程墙钟时间 {sum(times)/3600:.2f} 小时。" if times else ''
    execution=('本次正式统计全部在单张 GTX 1650 Ti 4GB 上按单进程重新训练，每进程 CPU 线程数 4；仅神经网络训练与表示提取使用 CUDA，KMeans、可靠性得分及评价使用 CPU。' if gpu else '本次正式训练使用 CPU，最多两进程，每进程 4 线程。')
    envtext=('独立 conda 环境 `envs/imvc_gpu/`：Python 3.9.21、Torch 2.8.0+cu126（官方 cu126 源）、CUDA runtime 12.6，GTX 1650 Ti 4GB、驱动 621.01；其余包版本与 CPU 环境固定一致：NumPy 1.26.4、SciPy 1.13.1、scikit-learn 1.4.0、munkres 1.1.4 等。清单 `environment_gpu.txt`，依赖核对与 CUDA 验证见 `results/gpu_environment_verification.json`。原 `envs/imvc_cpu` 和 F:/Anaconda 下的环境均未改动。' if gpu else '独立环境 envs/imvc_cpu，Python 3.9.21、Torch 2.8.0+cpu，详见 environment.txt。')
    gpusection=''
    if gpu:
        gpusection=f"""### GPU 切换核对与 CPU 归档

CPU 已完成的 6 组和中断的 Scene 0.5 种子 2、3 全部移动至 `results/archive_cpu_partial/`，原进度报告复制存档；CPU 部分结果只作设备对照，不进入正式统计。原 smoke、benchmark、operational_restarts、下载和来源记录保持原位。

Caltech101-20、0.5、seed 1 的两次独立 500 轮 GPU 核对：训练耗时 {gpu_check['gpu_training_seconds'][0]:.3f}/{gpu_check['gpu_training_seconds'][1]:.3f} 秒，对比 CPU {gpu_check['cpu_training_seconds']:.3f} 秒，加速 {gpu_check['training_speedups'][0]:.3f}/{gpu_check['training_speedups'][1]:.3f} 倍；两次六分支 ACC 最大差 {gpu_check['max_run_ACC_difference']:.8f}，通过既定 ≥1.5 倍及 ≤0.01 门槛。完整 ACC/NMI/ARI 及 CPU 差异见 `results/gpu_check/GPU核对说明.md`。检查模型不复用于正式全量。原 CPU 环境 3 轮冒烟及重新聚类审计通过，见 `results/smoke_cpu_after_gpu/`。

CUDA 设置 cuDNN deterministic=True、benchmark=False、CUBLAS_WORKSPACE_CONFIG=:4096:8、torch.use_deterministic_algorithms(True,warn_only=True)。逐次 stderr 和 warnings.jsonl（有警告时）保留运行警告；result.json 记录设备、Torch/CUDA、显卡、allocated/reserved 显存峰值和警告计数。显存峰值只计 PyTorch 分配，不含显示/驱动。GPU 包 CUDA 12.6 与 nvidia-smi 的驱动支持上限 13.5 是不同概念。

GPU 设备改动差异为 `scripts/gpu_device_adaptation.patch`；CPU 脚本原版在 `scripts/cpu_before_gpu/`，当前哈希在 `results/runtime_manifest.json`，旧 runtime 清单在 CPU 归档。协议文件字节和阈值保持不变。
"""
    paperhash=hashlib.sha256((root/'文献/COMPLETER_CVPR2021.pdf').read_bytes()).hexdigest()
    findings=[]
    for d in PAPER:
        if (d,.5) not in groups:continue
        cp=next(x for x in pairs if x['dataset']==d and x['rate']==.5 and x['comparison']=='E-B')
        ba=next(x for x in pairs if x['dataset']==d and x['rate']==.5 and x['comparison']=='B-A')
        findings.append(f"{d}：E−B={signed(cp['ACC_mean_diff'])}（{cp['ACC_positive']}/{cp['n']} 正），B−A={signed(ba['ACC_mean_diff'])}（{ba['ACC_positive']}/{ba['n']} 正）。")
        for s in ['s1','s2']:
            c=next(x for x in corr_summary if x['dataset']==d and x['rate']==.5 and x['score']==s)
            delta=next(x for x in pairs if x['dataset']==d and x['rate']==.5 and x['comparison']=='C_low_'+s+'-C_rand')
            findings.append(f"{s} 的 Spearman 均值为 {f(c['mean_spearman']) if c['mean_spearman'] is not None else '未定义'}，C_low−C_rand={signed(delta['ACC_mean_diff'])}（{delta['ACC_positive']}/{delta['n']} 正）。")
    text=f'''# 第一阶段 IMVC 可行性报告：固定模型下的 oracle 与可靠性诊断

日期：2026-10-27。目标是检验“可靠性感知的选择性补全”是否有继续研究的实证空间；当前只诊断，不提出新方法。**总体判读：{overall}。** 完成正式训练 {len(runs)}/20 次；每次训练的所有分支共用同一个模型和掩码。数值使用 0–1 单位，+0.03 为 +3 个百分点；标准差为五次运行的总体标准差（ddof=0）。

## 1. Baseline 选择与复现核对

使用 Lin 等 CVPR 2021 的官方 COMPLETER：[仓库](https://github.com/XLearning-SCU/2021-CVPR-Completer)，commit `624dd9fd10431802214fab1d7b34c44f5a0a92df`，MIT 许可证。论文：[CVF 原文](https://openaccess.thecvf.com/content/CVPR2021/papers/Lin_COMPLETER_Incomplete_Multi-View_Clustering_via_Contrastive_Prediction_CVPR_2021_paper.pdf)，表 1 Incomplete（η=0.5）。已提取并渲染核对表 1 和 §4.1。论文 PDF SHA-256：`{paperhash}`。

Git clone 因 TLS 中断失败，官方 raw/API 可用，故用固定提交的原始文件快照，校验 Git blob 后使用；没有把暂时的 Git 传输失败误判为实现不可获取。此前独立 Caltech 和 Scene CPU 运行均满足约 20 分钟门槛，因此未获取、未运行 HSACC。后按用户要求改用 GPU 重跑全阶段。{execution}{runtime}

{repro_table}

论文五次为随机初始化/数据划分；官方非零缺失率入口实际上把模型初始化固定为数据集配置 seed，而重复 1–5 只改变掩码。为按 baseline 自身协议复现，本次严格遵循后者：Caltech 的 np/random/torch seed 为 4/5/6，Scene 为 8/9/10，掩码 seed=1–5。不能把五次称为五个不同模型初始化。0.5 复现运行直接用于后续分支诊断，没有重复训练第二套模型。

ACC 差超过约 3pp 时如实标为复现不达标，仍按要求继续诊断。可能因素包括 Torch 1.2→2.8、sklearn 0.23→1.4、运行设备与数值库版本、原论文未公开的逐次掩码/随机状态以及本次固定的最终 KMeans seed；这些是可能原因，未做因果归因。未按结果调整模型、轮次、KMeans seed 或超参。

## 2. 数据核对与来源

| 数据集 | 样本数 | 类别数 | mat 内视图数 | 实际采用视图（0 起始） | 实际输入维度 | 预处理 |
| --- | --- | --- | --- | --- | --- | --- |
| Caltech101-20 | 2386 | 20 | 6 | 3、4 | 1984、512 | 官方每个完整视图全局 min–max，再 float32 |
| Scene-15 | 4485 | 15 | 3 | 0、1 | 20、59 | 官方原值直接转 float32 |

论文 §4.1 的样本数、类别数、实际使用两个视图与本次一致。论文将 Caltech 描述为 HOG/GIST，Scene 描述为 PHOG/GIST，但没有报告输入维度、降维细节或文件哈希；维度只能与官方 `configure.py`、`datasets.py` 和实际 mat 交叉核对，不能宣称已从论文独立验证所有特征语义。仓库文件包含更多候选视图，不等于本实验使用多于两视图。没有自行换视图、PCA 或抽样。Caltech 官方归一化在模拟掩码前计算，保持其协议，未声称完全避开缺失特征分布统计。

数据均来自官方同一 commit 的 `data/<name>.mat`，下载 URL 和 Git blob 见 `results/git_blob_checks.json`；项目 `data/` 与第三方原始文件逐字节一致：

- Caltech101-20：SHA-256 `{inventory['datasets']['Caltech101-20']['sha256']}`。
- Scene-15：SHA-256 `{inventory['datasets']['Scene-15']['sha256']}`。

## 3. 环境、安全与代码改动

CPU 为 AMD Ryzen 7 4800H，8 物理核心/16 逻辑处理器。{envtext}

{gpusection}

运行前通读第三方入口、数据加载、模型、损失、聚类评价和工具模块，未发现网络上传、删除、外部命令执行。代码与数据均视为不可信，核查后只从项目根用 `python -I` 和显式路径运行；不在 third_party 内执行。完整审阅说明见 `results/安全审阅与来源说明.md`。`-I` 是导入隔离，不是系统沙箱。

原始文件未修改。复制到 `scripts/completer/` 后，仅将掩码中 `np.int` 改为 `int`、数据加载改为显式根路径，逐行 diff 为 `scripts/completer_compatibility.patch`。`model.py`、`loss.py`、`configure.py` 和批处理保持原样。诊断入口 `scripts/run_diagnostic.py` 单独包装最终表示、KMeans 和离线评价：训练中的评价保留官方 100/200/300/400/500 轮模式切换及不固定 seed 的 KMeans 对 NumPy RNG 的消耗，但不计算/输出中间标签指标。训练只用完整样本，无标签、无 oracle；最终各分支统一固定 KMeans seed。

3 轮冒烟保存在 `results/smoke/`。首次冒烟的 float32 Jensen–Shannon 距离出现舍入导致的 NaN，失败记录保留，未用于判读；正式实验前已改为 float64 直接计算相同散度并断言有限。修正后的冒烟全量分支重建和重新聚类通过。线程数短轮检查仅确定 CPU 执行配置，不比较模型性能选参数；短轮结果不进入统计。

## 4. 事先固定的协议

- 主缺失率 0.5，辅助 0.7；两个数据集 × 5 个官方掩码种子，共 20 次训练。缺失率定义 η=(n−完整样本数)/n；每个不完整样本只缺一个视图。原始 `get_mask` 保证每行至少观测一个视图，容许每元素观测比例误差 <0.005，因此实际不完整样本比例可与 η 相差接近 0.01。各运行保存实际比例和掩码，不擅自改成精确数量采样。
- 训练 500 轮，Adam lr=1e−4，batch=256，alpha=9，lambda1=lambda2=0.1，100 轮后启用 dual prediction；编码器为输入–1024–1024–1024–128，双向预测器按官方 128–128–256–128 及镜像解码。只取固定最终第 500 轮，不按标签选轮次。
- 原生 128 维 latent 的两个视图直接拼接为 256 维，不另作归一化。各分支 KMeans：k 为公开类别数，k-means++，n_init=10，max_iter=300，tol=1e−4，lloyd，random_state=20261027。这一固定 seed 替代官方最终 random_state=None，是用户要求的配对控制；训练中的原 RNG 消耗另行保留。
- A：只将缺失视图 latent 置零；B：官方跨视图预测补全；E：只把缺失位置换成同一个训练好编码器作用于真实视图的输出，观测位置保持 B 原值。E 没有重新训练完整数据模型。
- C_low_s1、C_low_s2：分别按事先得分取所有缺失样本中最低 floor(M/2) 个，只有一个缺失视图，故以原始样本编号确定替换位置；得分并列按样本编号升序打破。C_rand：同数量的固定随机子集，两种得分共用，样本索引完整保存。随机种子由 `[20261027, dataset_index, round(rate*1000), data_seed, 5198418]` 的 SeedSequence 派生。
- s1（高=可靠）：在所有已观测视图 latent 的合并池上，用同类 KMeans、seed=20261027 建立共享 K 个无标签原型。q_k(z)∝(1+||z−c_k||²)^−1，跨 k 归一化；s1=−JS(q(预测缺失 latent),q(已有视图 latent))。没有用 oracle 或类别标签来拟合原型/选样本。
- s2（高=可靠）：对 B 拼接表示用 KMeans seed=100…109 重复 10 次，其余设置相同；用簇列联表的 Hungarian 匹配将每次编号对齐到固定 B 分区，逐样本计算与 B 一致的比例。匹配只用聚类分区，不用真实标签。
- 补全误差是预测 latent 与 oracle latent 的余弦距离，只在训练后用于诊断。Spearman 在每次运行的缺失样本上计算，报告全部五个系数及算术均值，不混池不同模型。标签仅用于最终离线 Hungarian ACC、NMI、ARI；不进入训练、打分、选点、KMeans seed 选择或阈值选择。

协议冻结文件 `results/protocol.json`，SHA-256 `7ddf3260ccebdf199cea02efdb9f562e8afc66c6a725f05ff4bcf0d6553cb34a`。以下判读不因结果改动：B−A>0 且 ≥4/5 正差为补全有意义；主缺失率下 E−B≥+0.03 且 ≥4/5 正差为有上限；同一种得分 Spearman 均值≤−0.3、C_low−C_rand≥+0.01 且 ≥4/5 正差为可靠性有效。上限和可靠性须在同一数据集成立才判通过；有上限但可靠性不达标为部分通过；E−B<+0.01 为该数据集不通过。原阈值对 +0.01–+0.03 或正差不足的中间情形没有唯一分类，本报告保留“证据不足”，不补设门槛。

## 5. 分支均值

{branch_table}

六个分支每组共享一次训练，表中训练数不是每分支独立重训数。C_rand 两种得分共用，不重复计为两个随机对照。没有排除负差运行。完整精度见 `results/branch_means.csv`。

## 6. 配对差与正差计数

{paired_table}

每项差严格按同数据集、同缺失率、同掩码/种子、同训练模型配对。统计以未舍入值计算，正差严格 >0；零差不算正。全部逐组差值见 `results/paired_differences.csv`，范围/零差见 `results/paired_summary.csv`。只有 5 次配对，没有另报未经预设的显著性结论。

## 7. 可靠性与补全误差

{corr_table}

高得分应对应低误差，故期望负相关。s2 有离散并列值，Spearman 使用平均秩；每次唯一得分数、相关 p 值、所选子集/随机子集平均余弦误差见 `results/reliability_correlations.csv`。p 值仅描述，不参与预设判读。相关达到门槛但未取得 C_low−C_rand 收益，仍不能认定可靠性信号有效；替换误差更低也不代表聚类收益更大。

## 8. 按规则判读

{decision_table}

{' '.join(findings)}

**主判读：{overall}。** {implication} 0.7 的结果仅作辅助描述，不用于改变 0.5 的判断。如有复现差距，本结论只对本次固定最终轮、{device_label}适配实现成立，不包装为已精确复现论文后的普遍结论。

## 9. 哈希、复跑和复算核验

{('20 次正式训练的掩码重建、干预样本选择、六分支输入哈希、得分重算、离线指标重算、固定种子重新 KMeans 全部通过。逐模型以保存权重重编码真实数据并重算补全，对比 oracle/B 表示通过容差检查；第三方原始文件及数据哈希保持一致。' if audit and audit.get('runs')==20 and audit.get('all_passed') else '正式运行及最终全量审计尚未全部完成，不能把待执行核验写为通过。')}

{reftext}

每次运行存档：`mask.npy`、`config.json`、`model.pt`、`diagnostic_arrays.npz`、`result.json`。数组包含全部 B/oracle 表示、两得分、余弦误差、10 次稳定性聚类、所选索引和六分支预测。掩码/各分支输入 SHA-256 使用 dtype+shape+连续字节计算；文件 SHA-256 单独记录，不混淆两种哈希。逐组掩码哈希、实际缺失比例、替换数量与耗时汇总为 `results/mask_metadata.csv`。所有正式组保留；队列切换时中断的两个未完成启动单独存于 `results/operational_restarts/`，没有形成最终结果，不是按指标排除的运行。

## 10. 判断边界与复查入口

只有单一 COMPLETER、两个数据集、每数据集/缺失率五个随机掩码、配置固定的模型初始化、模拟随机缺一个视图。随机 oracle 对照每组只有一个固定子集，两种得分共享对照，不能当作独立证据。两个数据集或 0.7 的结果不能替代 0.5 主条件。

E−B 是该编码器与下游聚类的操作性 oracle 诊断，不是任意补全方法的数学上界；oracle 使用真实被遮盖视图，仅用于离线分析，不是可部署方案。C_low 比 C_rand 好最多说明信号有可利用的诊断价值，没有验证一种实际的选择性补全新方法；A/B/E 改动会影响全局 KMeans，不能把差值当成可加的逐样本误差贡献。

未开展新方法、联邦设定、噪声对应、多 baseline 或超参搜索。旧方向目录保持只读。

- 固定协议：`results/protocol.json`；代码来源/数据：`results/source_inventory.json`、`results/git_blob_checks.json`。
- 主入口：`scripts/run_diagnostic.py`；后台队列：`scripts/run_batch.py`，事件 `results/batch_events.jsonl`，每次 stdout/stderr 位于 `results/full/`。
- 复算：`scripts/audit_results.py`，全量审计 `results/audit.json`；完整复跑核对 `results/refcheck/comparison.json`。
- 汇总与本报告：`scripts/summarize_results.py`，输出 CSV 和 `results/summary.json`；只纳入 `results/full/*/result.json` 中 500 轮结果，不读取 smoke/benchmark。
'''
    (root/'第一阶段IMVC可行性报告.md').write_text(text,encoding='utf-8')
    print(json.dumps(dict(runs=len(runs),overall=overall,decisions=decisions),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
