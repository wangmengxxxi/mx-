# AMGC 论文与官方代码复现核查

日期：2026-10-22。状态：**已获取有效全文、官方仓库并完成静态核查，尚未安装其完整运行依赖，也未执行 AMGC 训练。**

## 1. 来源与版本

- 论文：Wenxuan Tu 等，*Attribute-Missing Graph Clustering Network*，AAAI 2024，15392–15401。[DOI](https://doi.org/10.1609/aaai.v38i14.29464)；[官方 PDF](https://ojs.aaai.org/index.php/AAAI/article/download/29464/30760)；[本地 PDF](references/amgc.pdf)，有效 10 页；[提取文本](references/amgc.txt)。
- [官方仓库](https://github.com/WxTu/AMGC)；[本地 README](references/amgc_source/AMGC-master/README.md)。
- 查询得到的 master 提交：`e532af69e32f5adaedb5c29c5cd931f0e72333b7`，查询响应保存在 [amgc_commit.json](references/amgc_commit.json)。实际下载为 master ZIP 归档，未证明归档严格对应此提交；按 [stage2_manifest.json](stage2_manifest.json) 中的 ZIP 和逐文件 SHA256 标识当前快照。正式复现应另取明确 commit 的归档。
- 仓库原始代码保持原样。以下是静态核查，潜在运行边界尚未触发，不能称为已经复现出的故障。

## 2. 方法究竟补全什么

论文 15394–15395 页，公式 1–4 及 Algorithm 1 描述聚类引导的表示补全：缺失原始属性可先填零或做 Laplacian smoothing，编码后在同一伪簇内选择邻近节点，对缺失节点的**隐表示**进行修正。该表示用于边重建与簇中心相关约束；EMA 分支提供表示与伪标签更新。

代码 [models/em.py](references/amgc_source/AMGC-master/models/em.py) 对原始缺失属性置零，再在 GCN 后为缺失节点隐表示加入同伪簇近邻均值。近邻候选没有限制为属性观测节点。这不是一个输出完整原始属性矩阵、以原始属性重建误差训练的填补器。

因此，原方向规划中若把 AMGC 当成“原始属性填补器”，后续必须修正这一理解。比较时要区分属性空间和隐空间；不能不加说明地把 AMGC 的隐表示直接当成原始属性补全值计算 RMSE。

论文公式 7 为双非对比簇中心相关损失，公式 8–9 为边重建，公式 10 为 EMA 更新，公式 11 总损失为两部分之和。Table 4 支持其模块在论文设置中有作用，但并没有直接证明“伪标签错误反馈必然存在”，也没有证明我们计划的置信度门控会有效。这些应作为待检验假设。

## 3. 论文、代码与本地诊断的协议差异

| 项目 | 论文 / 当前官方代码 | 复现处理 |
|---|---|---|
| 缺失率 | 正文主实验 40% 观测，即 60% 缺失；代码 `train_fts_ratio=0.4` 也是观测率 | 不要误当作 40% 缺失；先按 60% 缺失复现 |
| 掩码 | `utils/misc.py` 用 sklearn shuffle，split_seed=72，观测数 floor(N×ratio) | 本地诊断用 round(N×missing_rate)，正式比较应直接共享保存的布尔 mask |
| 边遮蔽 | 论文及 yaml 的 p=0.7；`trains/train_em.py` 写死 `MaskEdge(p=0.5)` | 记录实际生效参数；先代码配置冒烟，再单列论文配置 |
| 伪标签 | Algorithm 1 每轮更新；代码训练伪标签每 20 轮更新，每轮另做评估 KMeans | 不能把评估频率当作训练伪标签频率 |
| 训练预算 | 正文至少 100 epochs 至收敛；代码 Cora 配置 1000 epochs | 冒烟只检查运行；正式训练预算需预先固定并记录收敛诊断 |
| 预训练 | 正文提到可选仅边重建预训练；当前 main 未见该入口 | 不假定当前代码已经执行该预训练 |
| 随机性 | 正文重复 10 次报告均值/标准差；未明确每次是否重新抽 mask | 不能自行解释成 10 个独立 mask；复现实验分别记录 mask seed 与模型 seed |
| 最终模型 | 代码每轮评估，训练结束保存最后状态再加载，文件名含 best | 文件名不代表按标签选择最佳 epoch |

Cora 代码配置还包括 GCN 1433→512→256、ELU、dropout=0.4、无 BN、边解码维度 64、近邻 K=2、EMA=0.999、lr=0.001 余弦调度、weight_decay=0.0005。配置文件在 [configs/EM/configs.yml](references/amgc_source/AMGC-master/configs/EM/configs.yml)，训练入口在 [main.py](references/amgc_source/AMGC-master/main.py) 和 [train_em.py](references/amgc_source/AMGC-master/trains/train_em.py)。

论文 Table 2 的 Cora 结果是在 **60% 缺失** 下：AMGC ACC 66.65±2.04、NMI 47.99±1.67、ARI 43.40±1.90；SDCN 对应 34.61±2.32、12.56±4.95、6.38±4.74（百分数）。本地第二阶段为 40%/80% 缺失、小模型与两个种子，不能据此直接排名，也不能把论文数值当成本地复现结果。

## 4. 两个必须显式处理的复现问题

### 随机种子入口

`main.py` 定义 seed 参数，`utils/misc.py` 定义 `set_random_seed`，但当前主流程没有调用该函数；KMeans 也未显式设置 random_state。仅传 `--seed` 不足以证明模型随机性被固定。后续应在独立工作副本连接入口、显式固定 KMeans 随机性，并记录改动；原始源码保留作为对照。

### 评估会修改传入的预测

[utils/evaluation.py](references/amgc_source/AMGC-master/utils/evaluation.py) 的 `cluster_acc` 在预测簇数不足时就地改写 `y_pred` 来补类；`eva_cluster` 随后用这个数组算 NMI/ARI。初始伪标签若通过该路径也可能被改变。

这一副作用已从源码确认，影响大小尚未运行量化。正式统一评估应使用 SciPy Hungarian 的矩形/补零混淆矩阵匹配，保存未修改预测，再计算 ACC/NMI/ARI。既要标注与原仓库评估的差异，也不能为追求论文数字保留会改写预测的统一评估。标签只用于外部评价。

## 5. 运行条件与待实测风险

当前 pytorch 环境检测：torch_geometric、torch_sparse、torch_scatter、munkres 未安装，yaml 可用，见 [依赖检查](references/amgc_dependency_check.json)。仓库没有提供完整 requirements。本轮没有向现有环境安装任何包，也没有修改 CUDA。

后续在专用环境或工作区隔离环境准备依赖；根据实际导入与算子决定所需 PyG 扩展及版本，不预先断言所有扩展都必须安装。仓库自带 Cora raw/processed，但新版本 PyG 的缓存兼容性需实测；优先核对 raw 文件并重建处理缓存。本地 Planetoid 与 AMGC 仓库的 8 个 Cora raw 文件逐个 SHA256 一致，检查保存在来源清单中；原始文件相同也不自动证明处理后的节点顺序与归一化相同。

模型按缺失节点执行 Python 循环与距离计算，CPU 开销可能明显高于本地 SDCN。singleton 簇下 squeeze/topk/近邻形状，以及零缺失时空列表 stack，都是需要覆盖的边界；目前只是静态风险。完整属性率设为 1 之前应处理零缺失路径，不能直接宣称原代码已支持所有缺失率。

## 6. 下次实际运行顺序与通过条件

1. 创建隔离环境，确认 CPU torch、PyG 与实际算子导入，保存环境版本；不改变现有 pytorch 环境。
2. 建立可修改工作副本，保留原始快照；固定代码版本、随机种子、节点顺序、缺失布尔 mask 与统一评估，逐项记录补丁。先读源码中的实际参数，不仅看 yaml。
3. Cora 观测率 0.4、20–50 轮冒烟，输出实际观测/缺失数、每轮时间、峰值内存、loss、预测占用簇数和未修改预测。先验证一个 seed；不使用标签选择最佳轮。
4. 若出现依赖、shape、NaN 或极慢问题，保留日志先修复并披露；不扩大实验。若能跑通且资源可接受，再按预先固定预算完整训练，并分别核对 p=0.5 和 p=0.7 的设置。
5. 用共享 mask 与一致评价进行 SDCN/FP/AMGC 对照，确认方法稳定后再扩 5 masks × 3 seeds 和多数据集。

冒烟通过只代表运行可行；完整训练、重复结果和协议解释通过以后，才可以说建立了 AMGC 对照基线。置信度与历史门控模块要在这些前置验证之后决定，不能用当前静态核查代替创新证据。
