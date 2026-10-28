# -*- coding: utf-8 -*-
"""Generate final tables/report only after complete audit; no edits to decision rules."""
import csv
import json
import sys
import datetime
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_diagnostic import sha,dump,METRICS

METHODS=['KM','BAL','P2OT_feat','ORACLE_KM','LP']

def csvwrite(name,rows):
    with open(ROOT/'results'/name,'w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',*['| '+' | '.join(map(str,r))+' |' for r in rows]])+'\n'

def main():
    date=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).date().isoformat()
    audit=json.loads((ROOT/'results/audit.json').read_text(encoding='utf-8'))
    assert audit['passed'] and audit['formal_runs']==30
    runs={}
    for p in sorted((ROOT/'results/full').glob('*/result.json')):
        r=json.loads(p.read_text(encoding='utf-8'))
        runs[(r['dataset'],r['ir'],r['seed'])]=r
    assert len(runs)==30
    means=[]
    sizes=[]
    distribution=[]
    subsetrows=[]
    for dataset,k in [('cifar100',100),('cifar10',10)]:
        for ir in [1,10,100]:
            counts=runs[(dataset,ir,1)]['counts']
            for c,n in enumerate(counts):
                subsetrows.append({'dataset':dataset,'ir':ir,'class_id':c,'count':n})
            for method in METHODS:
                entries=[runs[(dataset,ir,s)]['methods'][method] for s in range(1,6)]
                row={'dataset':dataset,'ir':ir,'method':method,'n_seeds':5}
                for m in METRICS:
                    values=[e['metrics'][m] for e in entries]
                    row[m+'_mean']=float(np.mean(values))
                    row[m+'_std']=float(np.std(values,ddof=0))
                means.append(row)
                drow={'dataset':dataset,'ir':ir,'method':method}
                for m in ['KL_true_to_pred','JS_true_to_pred','JS_sorted','JS_pred_to_uniform','normalized_entropy','empty_clusters']:
                    drow[m+'_mean']=float(np.mean([e['metrics'][m] for e in entries]))
                drow['size_min_mean']=float(np.mean([min(e['metrics']['raw_cluster_counts']) for e in entries]))
                drow['size_max_mean']=float(np.mean([max(e['metrics']['raw_cluster_counts']) for e in entries]))
                distribution.append(drow)
                for s,e in zip(range(1,6),entries):
                    v=e['metrics']
                    for c in range(k):
                        sizes.append({'dataset':dataset,'ir':ir,'seed':s,'method':method,'true_class_id':c,
                            'true_count':v['true_counts'][c],'aligned_cluster_count':v['aligned_cluster_counts'][c],
                            'recall':v['recall_per_class'][c], 'raw_cluster_id':c,'raw_cluster_count':v['raw_cluster_counts'][c],
                            'raw_cluster_maps_to_class':v['mapping'][c]})
    paired=[]
    def metric(ds,ir,s,method,m):
        if method=='BEST':
            km=runs[(ds,ir,s)]['methods']['KM']['metrics']['balanced_accuracy']
            bal=runs[(ds,ir,s)]['methods']['BAL']['metrics']['balanced_accuracy']
            method='KM' if km>=bal else 'BAL'
        return runs[(ds,ir,s)]['methods'][method]['metrics'][m]
    def pair(ds,ir,label,aa,bb,ir_b=None):
        for s in range(1,6):
            row={'dataset':ds,'ir':ir,'comparison':label,'seed':s}
            for m in METRICS:
                row[m]=metric(ds,ir,s,aa,m)-metric(ds,ir if ir_b is None else ir_b,s,bb,m)
            paired.append(row)
    for ds in ['cifar100','cifar10']:
        for ir in [1,10,100]:
            for aa,bb in [('BAL','KM'),('P2OT_feat','KM'),('P2OT_feat','BEST'),('ORACLE_KM','BEST'),('LP','BEST')]:
                pair(ds,ir,aa+'-'+bb,aa,bb)
        pair(ds,1,'BEST_IR1-BEST_IR10','BEST','BEST',10)
        pair(ds,1,'BEST_IR1-BEST_IR100','BEST','BEST',100)
    pairsummary=[]
    keys=list(dict.fromkeys((r['dataset'],r['ir'],r['comparison']) for r in paired))
    for ds,ir,label in keys:
        rows=[r for r in paired if (r['dataset'],r['ir'],r['comparison'])==(ds,ir,label)]
        v={'dataset':ds,'ir':ir,'comparison':label}
        for m in METRICS:
            values=np.array([r[m] for r in rows])
            v[m+'_mean']=float(values.mean())
            v[m+'_positive']=int((values>0).sum())
            v[m+'_zero']=int((values==0).sum())
        pairsummary.append(v)
    loss=[metric('cifar100',1,s,'BEST','balanced_accuracy')-metric('cifar100',100,s,'BEST','balanced_accuracy') for s in range(1,6)]
    room=[metric('cifar100',100,s,'ORACLE_KM','balanced_accuracy')-metric('cifar100',100,s,'BEST','balanced_accuracy') for s in range(1,6)]
    loss_ok=bool(np.mean(loss)>=.08 and sum(v>0 for v in loss)>=4)
    room_ok=bool(np.mean(room)>=.05 and sum(v>0 for v in room)>=4)
    verdict='通过' if loss_ok and room_ok else '部分通过' if loss_ok or room_ok else '不通过'
    summary={'verdict':verdict,'long_tail_loss':{'values':loss,'mean':float(np.mean(loss)),'positive':sum(v>0 for v in loss),'passed':loss_ok},
        'room':{'values':room,'mean':float(np.mean(room)),'positive':sum(v>0 for v in room),'passed':room_ok},'formal_runs':30,'methods_per_run':5,
        'protocol_sha256':sha(ROOT/'results/protocol.json'),'audit_passed':True}
    dump(ROOT/'results/summary.json',summary)
    for name,rows in [('branch_means.csv',means),('paired_differences.csv',paired),('paired_summary.csv',pairsummary),
            ('cluster_size_distributions.csv',sizes),('distribution_summary.csv',distribution),('class_counts.csv',subsetrows)]:
        csvwrite(name,rows)
    lines=['# 第一阶段长尾聚类可行性报告：冻结 DINOv2 特征的长尾损失与分配空间\n',
        f'报告生成日期：{date}（北京时间）；协议冻结于2026-10-27。当前只做诊断，不开发新方法。**总体判读：{verdict}。** 两个数据集 × 三个 IR × 五个种子，30/30 组，每组五种方法，共 150 个方法结果全部保留。指标单位为 0–1；+0.01 即 +1 个百分点，标准差为 ddof=0。\n',
        '## 1. 事先冻结的协议与主判读\n',
        f'协议文件 [results/protocol.json](results/protocol.json)，SHA-256 `{summary["protocol_sha256"]}`。冻结在特征提取和任何真实数据指标评价前；阈值未根据结果修改。\n',
        '主条件仅 CIFAR-100、IR=100、平衡准确率（BA）。每个种子在 IR=1 和 IR=100 分别取 max(KM,BAL)，只作离线诊断包络，不能按真实标签部署选择方法；P2OT_feat 不进入该 max。跨 IR 采用同种子、同类排列的嵌套子集，其他方法差按相同数据集、IR、种子配对。其他指标使用同一个由 BA 选定的 BEST 分支，未各自挑最好分支。\n',
        table(['规则','五个配对差（种子1–5）','均值 / 正差','是否满足'],[
            ['长尾损失 ≥0.08 且 ≥4/5 正差',', '.join(f'{v:+.6f}' for v in loss),f'{np.mean(loss):+.6f} / {sum(v>0 for v in loss)}/5',loss_ok],
            ['ORACLE_KM−BEST ≥0.05 且 ≥4/5 正差',', '.join(f'{v:+.6f}' for v in room),f'{np.mean(room):+.6f} / {sum(v>0 for v in room)}/5',room_ok]]),
        f'两项都满足为通过，只满足一项为部分通过，都不满足为不通过。本次为 **{verdict}**。这里的“显著”沿用预设的实用差值与正差门槛，不表示完成了统计显著性检验。CIFAR-10 和 IR=10 不改变结论。\n',
        '## 2. 数据、模型与特征来源\n',
        'CIFAR-10/100 均使用 torchvision 的官方多伦多大学 Python 档案，50000 训练样本、10000 测试样本；先核验 torchvision 内置 MD5，再核验归档路径后解压。官网当前重定向至 cave.cs.toronto.edu。网络尝试、失败及代理路径保留日志。DINOv2 为 Meta 官方 LVD-142M ViT-B/14，无 registers，未使用第三方训练的替代权重。\n']
    sources=json.loads((ROOT/'results/source_inventory.json').read_text(encoding='utf-8'))
    lines.append(table(['来源','URL','SHA-256'],[[v['name'],v['url'],f'`{v["sha256"]}`'] for v in sources]))
    lines.extend(['使用官方构造参数 img_size=518、patch_size=14、init_values=1、block_chunks=0，权重 strict=True/weights_only=True 加载；eval、冻结全部参数，model/input fp16 推理，保存最终 LayerNorm 后的 768 维 CLS 为 float32 数组（值来自 fp16）。没有拼接 patch token、没有 PCA、没有重新预训练。\n',
        '预处理为 PIL bicubic Resize(224)、CenterCrop(224)、ToTensor，均值 (0.485,0.456,0.406)，标准差 (0.229,0.224,0.225)。官方评估工厂默认 Resize256/Crop224，本次按用户明确指定覆盖为 Resize224；CIFAR 原图为正方形，裁剪不额外丢边。此差别在协议中事先写明。四个 split 各完成一次提取，此后全部实验与复跑共用同一缓存；测试特征只保存和校验，未参与聚类、LP、调参或主判读。\n'])
    fm=json.loads((ROOT/'results/feature_manifest.json').read_text(encoding='utf-8'))
    lines.append(table(['缓存','shape / batch','SHA-256'],[[k,f'{v["shape"]} / {v["batch"]}',f'`{v["sha256"]}`'] for k,v in fm.items()]))
    lines.extend(['## 3. 长尾构造与每类样本数\n',
        '类别顺序固定为 torchvision 标签编号 0…K−1。n_c=floor((50000/K)·IR^(−c/(K−1)))；NumPy RandomState(seed)，按类别顺序对各类索引分别排列、取前 n_c，种子为 1–5。同一个种子不同 IR 复用相同类内排列；IR=1 的五次含有相同的完整训练集，仅顺序和聚类随机状态不同，不能视为五份独立数据。尾部取样本数最少的 floor(K/3) 类，并列按类别编号升序；IR=1 全部并列，尾部仅是固定标签子组。\n'])
    for ds,k in [('cifar100',100),('cifar10',10)]:
        for ir in [1,10,100]:
            r=runs[(ds,ir,1)]
            lines.append(f'**{ds} / IR={ir}**：N={r["n"]}，每类数量（类别编号递增）：`'+', '.join(map(str,r['counts']))+'`。\n')
    lines.extend(['完整数量见 [class_counts.csv](results/class_counts.csv)；逐组 indices.npy 和索引/输入数组哈希均存档。聚类与评价都在所构造的同一训练子集上。\n',
        '## 4. 方法、固定超参及标签隔离\n',
        '- 所有方法读取相同缓存并先 L2 归一化，无标签特征变换。KM：sklearn KMeans，k=K，k-means++，n_init=10，max_iter=300，tol=1e−4，lloyd，random_state=seed。\n'
        '- BAL：单个余弦线性头 768→K、无偏置，torch Linear 随机初始化，每次前向归一化权重，logits=10·cos；对 cos/0.05 施加 log-domain 均匀 Sinkhorn（3 次行列归一化），stop-gradient 软目标交叉熵。Adam lr=5e−4、betas=(0.9,0.999)、wd=0，cosine 日程，batch=1024，最后小批保留，固定最终100轮。不训练骨干、不新增图像增强；同一缓存特征产生目标与预测，是 SwAV 风格分配自训练基线，不是完整 SwAV 复现。\n'
        '- P2OT_feat：官方 SemiCurrSinkhornKnopp 能独立接收 logits，故加入。原始求解器不改动，epsilon=0.1、gamma=1、stoperr=1e−6、最多1000次，semi_use=True；rho=0.1+0.9·exp(−5(1−step/total_steps)²)。头、初始化、打乱、优化器、100轮和最终检查点同 BAL，保留部分输运软目标的行质量，不逐行归一化目标。本次单头、无 logits memory bank、无双视图、无骨干微调；是模块适配，不是 P²OT 论文成绩复现。\n'
        '- ORACLE_KM：在当前子集以标签计算各类 L2 特征均值，不再单位化类中心；以平方欧氏最近中心分配，Lloyd 迭代0次。只用于离线诊断，依赖真实标签。\n'
        '- LP：sklearn 多项逻辑回归，L2 特征，C=1、lbfgs、max_iter=1000、tol=1e−4、截距开启、class_weight=None。在同一子集监督拟合并重代入评价，是表示可分性的监督参照，不是泛化精度，也不是无监督可达成绩。\n',
        '标签仅用于按要求构造长尾、最终离线评价，以及明确标注的监督 ORACLE_KM/LP；KM/BAL/P2OT_feat 拟合函数只接收特征、K、seed，不接收真实类别或每类数量。没有用标签挑超参、轮次或检查点。\n',
        '每组按最大样本正确数求一次全局 Hungarian 映射，ACC、各类召回、BA、macro-F1 与尾部召回共用该映射，子组不重新匹配。NMI 使用 arithmetic 平均，ARI/NMI 本身对编号不敏感。LP 也使用相同评价入口。\n',
        '## 5. 分支均值（均值±总体标准差）\n'])
    lines.append(table(['数据集','IR','方法',*METRICS],[[r['dataset'],r['ir'],r['method'],*[f'{r[m+"_mean"]:.4f}±{r[m+"_std"]:.4f}' for m in METRICS]] for r in means]))
    lines.extend(['完整精度见 [branch_means.csv](results/branch_means.csv)。未剔除负差、空簇或低分运行。\n','## 6. 配对差与正差计数\n'])
    lines.append(table(['数据集','IR','差值',*[m+' 均值 / 正差' for m in METRICS]],[[r['dataset'],r['ir'],r['comparison'],*[f'{r[m+"_mean"]:+.4f} / {r[m+"_positive"]}/5' for m in METRICS]] for r in pairsummary]))
    lines.extend(['逐种子未舍入差值见 [paired_differences.csv](results/paired_differences.csv)，零差数量见 [paired_summary.csv](results/paired_summary.csv)。正差严格 >0，零不算正；没有额外的事后显著性检验。\n',
        '## 7. 簇大小分布与均匀先验机制\n',
        '预测簇保留全部 K 个编号（含空簇），通过同一全局 Hungarian 映射与真实类别对齐。KL 为 KL(p_true∥q_aligned)，各计数加1e−12防零，单位 nat；另报无平滑 JS、与均匀分布的 JS、归一化熵，避免只用对编号敏感的 KL 判断。JS 越小代表更接近，归一化熵越接近1代表越均匀。\n'])
    lines.append(table(['数据集','IR','方法','最小/最大簇均值','空簇均值','KL true∥pred','JS true/pred','JS pred/uniform','归一化熵'],[
        [r['dataset'],r['ir'],r['method'],f'{r["size_min_mean"]:.1f}/{r["size_max_mean"]:.1f}',f'{r["empty_clusters_mean"]:.1f}',
         f'{r["KL_true_to_pred_mean"]:.4f}',f'{r["JS_true_to_pred_mean"]:.4f}',f'{r["JS_pred_to_uniform_mean"]:.4f}',f'{r["normalized_entropy_mean"]:.4f}'] for r in distribution]))
    for ds in ['cifar100','cifar10']:
        r=next(r for r in pairsummary if (r['dataset'],r['ir'],r['comparison'])==(ds,100,'BAL-KM'))
        dk=next(r for r in distribution if (r['dataset'],r['ir'],r['method'])==(ds,100,'KM'))
        db=next(r for r in distribution if (r['dataset'],r['ir'],r['method'])==(ds,100,'BAL'))
        lines.append(f'{ds}、IR100：BAL−KM BA={r["balanced_accuracy_mean"]:+.4f}（{r["balanced_accuracy_positive"]}/5 正），尾部召回={r["tail_recall_mean"]:+.4f}（{r["tail_recall_positive"]}/5 正）；JS(pred,uniform) KM={dk["JS_pred_to_uniform_mean"]:.4f}、BAL={db["JS_pred_to_uniform_mean"]:.4f}。这是均匀先验机制的描述性证据，不额外修改主判读。\n')
    lines.append('每个方法 × 数据集 × IR × 种子的全部原始簇计数、全局映射、对齐计数、各类召回在 [cluster_size_distributions.csv](results/cluster_size_distributions.csv) 和逐组 result.json 中；CSV 的 raw_cluster_id 与 true_class_id 是两套索引，映射列明确连接，不能按同一编号混读。主条件逐类的五次平均对齐簇大小如下，类别0…99顺序固定：\n')
    for method in METHODS:
        arr=np.array([runs[('cifar100',100,s)]['methods'][method]['metrics']['aligned_cluster_counts'] for s in range(1,6)]).mean(0)
        lines.append(f'**{method}**：`'+', '.join(f'{v:.1f}' for v in arr)+'`。\n')
    lines.extend(['## 8. 复跑、哈希与环境审计\n',
        '独立 lt_gpu 环境从只读 imvc_gpu 克隆，Python3.9.21、Torch2.8.0+cu126、torchvision0.23.0+cu126、NumPy1.26.4、sklearn1.4.0、Pillow11.3.0；详见 environment_gpu.txt。单张 GTX1650Ti 4GB、驱动621.01，GPU工作顺序执行；KMeans/LP与评价在CPU，线程数4。开启 deterministic、关闭 cuDNN benchmark/TF32，CUBLAS_WORKSPACE_CONFIG=:4096:8。\n',
        '第三方代码放在 third_party/，实际执行的 DINOv2 导入闭包与 P²OT 求解模块已通读。原始源码不修改；仅 DINOv2 的两文件在兼容副本添加 future annotations 以适配 Python3.9，数学前向未改，diff 在 results/dinov2_py39.patch。没有执行仓库训练/下载入口。全部使用 python -I；它只提供导入隔离，不是系统沙箱。安全和来源说明见 [安全审阅与来源说明.md](results/安全审阅与来源说明.md)。\n',
        '全量审计：四个特征缓存的 shape/dtype/有限性/哈希、原始及兼容源码哈希、运行脚本哈希、30组索引重建、150个方法的指标/簇分布重算及从保存中心/头/逻辑回归参数重建预测均通过。独立进程完整复跑 CIFAR100 IR100 seed1 的五种方法，复用相同缓存；索引与输入哈希相同，数值比较按预先容差通过。逐数组是否逐位一致与最大差保存在 [comparison.json](results/refcheck/comparison.json)，未把容差一致等同于逐位一致。\n',
        '协议、来源、特征、第三方文件和运行脚本的清单分别位于 results/protocol.sha256、source_inventory.json、feature_manifest.json、third_party_manifest.json、runtime_manifest.json。数组内容哈希使用 dtype+shape+连续字节，文件哈希为文件字节，两者分别记录。正式日志和所有警告均保留；审计见 [audit.json](results/audit.json)。\n',
        '## 9. 判断边界\n',
        f'**按冻结规则：{verdict}。** 本结论只覆盖单一 DINOv2 ViT-B/14、冻结特征、所选头及超参、转导式设定、CIFAR两个数据集与五个配对种子。强预训练语料与 CIFAR 的潜在重叠未独立审计，不能将本结果当成完全未知域的泛化证据。尾类在 CIFAR100 IR100 最少仅5张，其重代入监督结果尤其不能解释为尾部泛化。\n',
        'ORACLE_KM 是使用真实类中心的操作性分配参照，不是任意聚类器的数学上界；若某个无监督结果超过它也不矛盾。LP 是带标签的重代入参照。即使规则通过，也只说明长尾下降和监督参照差距同时存在，不能单凭 oracle 差值证明一个可部署无监督方法一定能找回该差距；P2OT_feat 的配对收益是本次实际测试的无监督方法证据。若 P2OT_feat 不改善，必须保留这一负结果，不以其失败反推所有不平衡方法无效。BAL 同缓存自训练没有图像多视图约束，所见现象不能全部归因于均匀先验。\n',
        '本次没有开发新方法、搜索超参或更换骨干，未因结果调整8/5个百分点阈值。既往两个方向目录保持只读。\n',
        '## 10. 复查入口\n',
        '- 队列：scripts/run_pipeline.py，事件与状态 results/pipeline_events.jsonl、pipeline_status.json。\n'
        '- 特征：scripts/extract_features.py；正式单组：scripts/run_diagnostic.py --dataset cifar100 --ir 100 --seed 1。\n'
        '- 独立复跑使用 --base results/refcheck；完成目录拒绝覆盖。\n'
        '- 审计：scripts/audit_results.py；汇总与本文生成：scripts/summarize_results.py。均使用 envs/lt_gpu/python.exe -I，从项目根运行。\n'])
    (ROOT/'第一阶段长尾聚类可行性报告.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':
    main()
