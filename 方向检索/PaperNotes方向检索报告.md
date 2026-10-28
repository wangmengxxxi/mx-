# PaperNotes 硕士选题方向检索报告

调研日期：2026-10-27（Asia/Shanghai）。对象：计算机硕士，尚未开题；以两个有联系的方法章节为目标，主要实验应能落在 RTX 5060 8GB 上，必要时短租云卡。本报告只做文献检索、原文抽查与仓库可用性审计，未运行实验，未下载或运行论文代码。

**建议保留三个有条件的候选：A「噪声对应下的可靠多视图聚类」、B「模态质量变化下的可靠融合与适度再平衡」、C「缺失模式异构下的个性化多模态联邦学习」。**若重视实验室聚类积累和低显存，先补核 A 的原始结果；若重视已核实的问题退化证据，先看 B；若希望两个章节形成较完整的交叉研究链条，C 最自然，但通信轮数、数据预处理和强本地适配基线的风险更高。这三条均尚不能直接判定为“已经找到可提升的题目”。

## 1. 站点概况与检索方法

### 1.1 站点范围与可信度

已读取[首页](https://papernotes.org/)、[关于本站](https://papernotes.org/about/)和 [robots.txt](https://papernotes.org/robots.txt)。网站是独立的 AI 辅助论文阅读笔记站，以「会议年份 / 类别 / 标题」组织页面，并非会议官方论文库。关于页明确表示：不保证所有笔记经过逐句人工核验，公式、数字、引用、会议归属和解读都可能有错。笔记的星级、“首次”“SOTA”及具体分数在本报告中均不作为已核实证据。

下载的[完整搜索索引](https://papernotes.org/search/search_index.json)为 **9,205,988 字节、26,609 条记录**。首页展示 26,590，与本次索引快照差 19 条；未将展示数当成精确总量。26,609 个去锚点论文形态路径不重复，但这只是索引路径去重，不代表已核查每条均为不同正式论文。索引仅包含题目、短关键词摘要 `text`、标签，并非笔记全文。

首页/索引覆盖 **7 个会议系列、13 个会议年份栏目，年份跨度 2024–2026**。计数是站内标签，不能替代真实录用身份：

|会议系列|站内年份与索引条数|
|---|---|
|AAAI|2026：1,382|
|ACL|2025：1,855；2026：1,409|
|CVPR|2025：1,820；2026：4,065|
|ECCV|2024：869；2026：2,907|
|ICCV|2025：1,315|
|ICLR|2026：5,316|
|ICML|2025：1,063；2026：1,845|
|NeurIPS|2025：2,530；2026：233|

因此不能据此覆盖范围断言某方向“2025 年无工作”或“2026 年已全部覆盖”。例如没有 ICLR2025 栏目，NeurIPS2026 仅 233 条；传统聚类、联邦经典文献以及非上述会议的工作明显不足。类别也可能错放，检索没有限定类别目录。

### 1.2 执行流程与阅读范围

先直连 HTTPS，成功；本机 `127.0.0.1:7890` 未监听，未启用代理。robots 允许读取 `/` 且要求 `Crawl-delay: 1`；笔记请求串行并间隔至少 1 秒。只读网页，无注册、表单提交或评论；网页文字均作为待核查资料，不作为操作指令。

索引只下载一次，随后本地做 **31 组主题、62 条中英文检索**，按 2025–2026 标签分层筛选，再按任务设定、硬件和潜在失败模式选择 **51 篇笔记细读**（P01–P51，低于约150页上限）。其中包含用于排除误命中及审计强基线的论文；P48–P51只用于目标检测/RL粗筛，不将其当作已深入验证的推荐方向。每篇均保存问题、方法、实验、局限段落和来源链接，完整记录在 [papers.csv](F:/研究生毕设/方向检索/papers.csv)。

最终候选的抽查范围：A 抽查 P02/P05/P08，P05/P08 的题目、作者、会议在 CVF 核实，P02 的 OpenReview 受限，只有作者 README 的身份声明；B 抽查 G2D/MNL/CPSC，并补查 MIDAS/AUG/BALM；C 抽查 PEPSY/FED-PRIME/pFedMMA，并补查 FedMVP。arXiv 的作者会议声明与 CVF 正式录用页分级记录，未混为同一强度证据。共核查 14 个仓库地址，其中13个返回存在、G2D旧地址返回404；涉及13篇论文。仅查看元数据、README 与根目录结构，没有抓取源代码、依赖文件正文、ZIP、模型或数据。

以下命中数顺序均为“全量 / 2025–2026”。中英文为不同规则，**不能相加为论文数量；同一篇可被多个主题重复命中**。宽检索命中不等于直接竞争数。

|方向|英文关键词（全量/2025–26）|中文关键词（全量/2025–26）|
|---|---|---|
|聚类|deep clustering (8/8)|深度聚类 (6/6)|
|聚类|long-tailed / imbalanced clustering (3/3)|长尾/不平衡聚类 (6/5)|
|聚类|unknown number of clusters (2/2)|簇数未知/类别数估计 (4/4)|
|聚类|category discovery / GCD (36/33)|类别发现/广义类别发现 (33/31)|
|聚类|multi-view clustering (25/24)|多视图/多模态聚类 (22/22)|
|聚类|clustering + foundation / CLIP / DINO (7/6)|基础模型/CLIP/DINO + 聚类 (18/17)|
|聚类|text clustering (3/3)|文本聚类 (3/3)|
|多模态|missing modality (34/32)|缺失模态/模态缺失 (50/48)|
|多模态|modality imbalance (6/6)|模态不平衡/模态失衡 (18/18)|
|多模态|noisy correspondence (12/12)|噪声对应/错配/未对齐 (13/13)|
|多模态|multimodal fusion robustness (148/142)|多模态融合鲁棒性 (122/119)|
|多模态|multimodal sentiment / emotion (27/27)|多模态情感/情绪 (23/23)|
|图学习|graph clustering (5/5)|图聚类 (26/26)|
|图学习|graph structure learning (0/0)|图结构学习 (10/8)|
|图学习|heterophily / heterophilic (8/8)|异配图/异质配性 (2/2)|
|图学习|graph OOD (9/9)|图分布外/分布偏移 (28/26)|
|图学习|graph prompt (14/13)|图提示 (6/5)|
|联邦学习|clustered / personalized FL (28/28)|聚类/个性化联邦 (35/35)|
|联邦学习|FL with missing modality (3/3)|缺失模态联邦 (2/2)|
|联邦学习|federated clustering (5/5)|联邦聚类 (5/5)|
|联邦学习|federath noisy labels / federated noisy labels (1/1)|联邦噪声标签 (3/3)|
|时间序列|time series clustering (1/1)|时间序列聚类 (3/3)|
|时间序列|anomaly detection + time series (22/22)|时间序列异常检测 (28/28)|
|时间序列|imputation + time series (8/8)|时间序列插补/缺失 (18/17)|
|时间序列|TS foundation models (31/31)|时序基础模型 (76/74)|
|交叉|clustering x federated (7/7)|聚类×联邦 (11/11)|
|交叉|clustering x multimodal (29/28)|聚类×多模态/多视图 (39/39)|
|交叉|federated x multimodal (10/9)|联邦×多模态 (11/10)|
|交叉|graph x federated (7/7)|图×联邦 (25/23)|
|目标检测|lightweight / open-vocabulary detection (103/99)|轻量/开放词汇检测 (113/108)|
|强化学习|offline RL / small environments (55/55)|离线强化学习/小环境 (78/78)|

完整正则、两语查询及命中数见 [search_log.md](F:/研究生毕设/方向检索/search_log.md)。例如英文 `graph structure learning` 为0而中文为10，并不表示该领域不存在；中文“图×联邦”可能命中“图像”，已人工排除。用户给出的 `federath noisy labels` 同时扩展为 `federated/noisy labels` 检索，未因拼写漏搜。

### 1.3 防止重复进入“损失很小”的陷阱

对于越大越好的指标，区分两种量：

- **问题损失** `G_damage = 同一方法在干净/完整条件的分数 − 同一方法在实际受扰条件的分数`。两者须匹配训练预算、测试集和指标；重训模型与固定模型的推理退化也应分开。
- **方法增益** `G_gain = 某方法在受扰条件的分数 − 同协议强基线在受扰条件的分数`。受扰损失很大，并不能证明最新强方法之后还有很大增益。

真实视图、干净配对或真实类别数可以作诊断性 oracle，但有额外信息，其分数不是现实方法必然可达的上限。IID 与 non-IID 的分数差同时混入分布变化，不能归因于缺失。第一阶段优先做原表审计，之后的最小实验仅是给学生未来的验证计划，本次没有执行。

## 2. 各实验室方向的研究热点概览

**聚类。**2025–2026 的线索集中在基础模型语义引导、GCD/未知类数、视图关系对齐与不可靠对应，以及长尾分布。P01 MiniClustering 是当前长尾选题的直接竞争补充；P45 BIA、P46 SpectralGCD、P47 非参数细粒度聚类说明“谱正则、CLIP 概念筛选、DPMM估K”均已有近邻，不能只加一个模块就算创新。P02 CorreGen 将配对作为隐变量，比普通缺失补全更值得查剩余空间。P07 CRAFT 提醒，相同名义缺失率不等于相同观测结构或信息预算；这不推翻学生已排除的低损失 benchmark。

**多模态。**热点已从静态融合扩展到学习速度/能力不平衡、负学习蒸馏、SAM优化、训练增强及测试质量变化。P09 G2D、P10 MIDAS、P11 MNL、P12 AUG、P13 RollingQ、P14 MASAM、P15 BALM、P16 CPSC、P17 EBMC 构成至少9篇相关细读工作，通用“增强弱模态”非常拥挤。候选 B 只保留“弱但有信息”与“弱且受污染”的区分。缺失情感分析仍可作既有备选，不重复把 LNLN/P-RMF 主线作为新推荐。

**图学习。**异配、OOD、图提示、结构泛化和联邦子图异构均有线索。P33 IGNN 与 P34 图学习常见信念审计提示：必须加入调好的经典 GNN、MLP 和适当的多跳基线。P35 RedOUT 的测试时图OOD、P36 GraphUniverse 的跨图归纳评测较轻量，但本次尚未核实可用增益。P28 FedSDR 的结构污染检测与重构接近学生之前的“修复”路径，先检查真实损伤，不能直接回到修图选题。

**联邦学习。**除普通个性化，主要交叉线索是缺失多模态、结构异构子图、含噪标签和未知混合分量。P18 PEPSY、P19 FED-PRIME 直接针对异构缺失；P23 联邦图级聚类、P25 FedAux、P27 FedSST 是图×联邦近邻；P31 FedRG、P32 FedGR 涉及噪声客户端/标签。P30 FedGEM 的“联邦估计数据混合分量”不同于 Clustered FL 的“把客户端按模型/任务分组”，不能混列为同一协议基线。P29 是另一个同缩写 FedSDR，实际做联邦 LLM，已排除。

**时间序列。**基础模型迁移和异常检测线索明显多于时序聚类。P37 EMTC 是本次命中的直接时序聚类代表；P39 TSPulse 提供轻量预训练模型线索，P40 在冻结预测器后校准异常告警，可能适合8GB，但尚未核一手效果。P38 反思 TSFM 可能退化为简单统计分数；P41 CAROTS、P42 KAN-AD 提供对照。需核验 point adjustment、是否利用测试标签选阈值、窗口泄漏及每序列归一化，不能把不同评测协议的 F1 横向相减。

**目标检测（粗筛）。**存在轻量适配线索：P48 ABRA 做类别残差及跨域传递，P49 做敏感性剪枝的测试时适配。但 ABRA 仍涉及域专家训练，后者仍依赖源检测器及框架；训练参数少不代表峰值显存小。本次未发现有一手显存/统一小数据协议证据、可以直接承诺8GB复现的检测主线。冻结检测器后的小模块或小模型适配可留作观察，不以大规模开放词汇检测训练为首选。P48 的站内 CVPR2025 标签与 arXiv 2603 编号存在年份线索不一致，未据此确认会议。

**强化学习（粗筛）。**P50 ASPC 在小MLP离线RL上动态调整策略约束，P51 SAM处理离线数据污染，模型本身可能适合8GB。瓶颈更多在 MuJoCo/D4RL 等环境版本、多随机种子、数据质量与强基线重调，未核实总训练成本。P51 为 Student Abstract 线索，不能当成完整强基线证据。LLM/RLHF误命中已排除，不推荐转入大模型强化学习。

上述论文的完整题目、站内会议标签及原文/仓库链接见第7节和 CSV；只有第3节明确标注的内容具有原文核查支持。

## 3. 推荐候选方向

评分均为1–5分，越高越有利：提升空间看原文支持和饱和风险；拥挤程度以适中为佳（过少意味着资料/协议风险，过多意味着创新压力）；入门成本分高表示更易入门；算力分高表示更适合8GB；工作量分高表示能自然形成两个方法章节；契合度考虑实验室方向。分数是调研判断，不是论文质量或录用概率，也不按总分机械排序。

### A. 噪声对应下的可靠多视图聚类

**问题定义。**在没有类别标签时，同一行的多个视图不一定属于同一实例；还可能出现无法对应的样本及同类不同实例。目标是在不依赖干净配对监督的前提下学习语义簇。主设定应是“视图存在但对应不可靠”，与“把缺失视图补全”分开。可先用预计算多视图特征和MLP，选择 Scene15、LandUse21、Caltech 等公开小数据，原始特征来源、是否含标签训练的编码器必须一致。

**为什么可能有空间。**站内 P02 声称配对错误会明显伤害聚类，并报告潜在对应恢复的效果；这些数字尚未从 OpenReview 原表核实，因此不能作为已经证实的提升空间。已核的 P05 是**监督分类、训练配对干净、测试配对受扰**，证明该任务设定确有正式近邻，但不能把分类结果移植为无监督聚类结论。已核 P08 原文 Table 3 的噪声是均值0、标准差0.4的高斯特征污染，**不是跨视图行索引置换**：DHA 的 GLC 从噪声率0.1时 ACC83.9降至1.0时72.2，差11.7个百分点；LandUse仅27.3→24.7（2.6点），ProteinFold仅31.8→29.5（2.3点）。没有噪声率0的干净格，不能称其为 clean–noise 差。它支持“需要逐数据集核损伤”，并未替代 A 核心错配问题的证据。

**代表论文（5篇，注明任务边界）。**

|论文 / 站内标签|问题与核心机制；本次定位|笔记|原文|代码|
|---|---|---|---|---|
|P02 Uncover Underlying Correspondence for Robust Multi-view Clustering / ICLR2026|跨视图实例错配、同类假负例与无有效对应的污染样本共存; CorreGen以潜在软对应为隐变量做EM；GMM估边际、OT求对应、虚拟样本吸收无法对齐质量; A核心；OpenReview受阻，原文数值待核验|[笔记](https://papernotes.org/ICLR2026/self_supervised/uncover_underlying_correspondence_for_robust_multi-view_clustering/)|[原文](https://openreview.net/forum?id=a4S1nQay3b)|[仓库](https://github.com/XLearning-SCU/2026-ICLR-CorreGen)|
|P03 Relationship Alignment for View-aware Multi-view Clustering / ICLR2026|完整干净多视图中视图差异导致强制一致性损伤语义; RAV对齐跨视图关系矩阵，并以Wasserstein距离调节簇级对齐强度; A近邻；不是已验证的噪声对应方法；仓库只有ZIP|[笔记](https://papernotes.org/ICLR2026/self_supervised/relationship_alignment_for_view-aware_multi-view_clustering/)|[原文](https://openreview.net/forum?id=uRA9cT4MK6)|[仓库](https://github.com/chenzhe207/RAV)|
|P04 Calibrated Information Bottleneck for Trusted Multi-modal Clustering / ICLR2026|完整单标签多模态聚类中的互信息估计偏差、伪标签噪声与过自信; CLIB以模态校准头筛选伪标签，训练主聚类头并用一致性约束校准; A近邻；主页链接不是已确认的代码仓库|[笔记](https://papernotes.org/ICLR2026/multimodal_vlm/calibrated_information_bottleneck_for_trusted_multi-modal_clustering/)|[原文](https://openreview.net/forum?id=iedlZOdI0d)|个人主页，非已确认代码|
|P05 Bootstrapping Multi-view Learning for Test-time Noisy Correspondence / CVPR2026|有标签多视图分类训练配对干净、部署时视图错配; BML在线制造错配并记录掩码，以揭示监督学习可靠性加权融合; A关键近邻；监督分类不能直接充当无监督聚类基线|[笔记](https://papernotes.org/CVPR2026/others/bootstrapping_multi-view_learning_for_test-time_noisy_correspondence/)|[原文](https://openaccess.thecvf.com/content/CVPR2026/html/He_Bootstrapping_Multi-view_Learning_for_Test-time_Noisy_Correspondence_CVPR_2026_paper.html)|[仓库](https://github.com/XLearning-SCU/2026-CVPR-BML)|
|P08 Global-Graph Guided and Local-Graph Weighted Contrastive Learning for Unified Clustering on Incomplete and Noise Multi-View Data / CVPR2026|不完整视图减少可配对样本，特征高斯噪声使对比学习监督失真; 全局图扩展正负对、局部高阶图赋对级权重；原文简称GLC，代码仓库名GLGC; A关键近邻；高斯污染不等于实例索引错配|[笔记](https://papernotes.org/CVPR2026/multimodal_vlm/global-graph_guided_and_local-graph_weighted_contrastive_learning_for_unified_cl/)|[原文](https://openaccess.thecvf.com/content/CVPR2026/html/He_Global-Graph_Guided_and_Local-Graph_Weighted_Contrastive_Learning_for_Unified_Clustering_CVPR_2026_paper.html)<br>[补充原文](https://arxiv.org/abs/2512.21516)|[仓库](https://github.com/hhq-sr/GLGC)|

**两个可能的工作点，均待验证。**

1. 未知、类别相关错配率下的可靠对应与少数语义保留：研究匹配不确定性怎样避免将真正罕见类当异常丢弃，并明确哪些证据来自语义、哪些只来自实例相似。必须比较 CorreGen 本身的虚拟样本、GMM和OT，而非把“置信度+OT”再次包装。
2. 错配机制或可用视图集合改变时的稳定融合：在固定表征和有限匹配预算下，研究如何迁移关系证据并控制大批匹配开销。第二章需解决与第一章不同的部署变化，不仅再加一个图损失；稀疏近邻/匹配已有大量相关工作，正式开题前还要扩站外查重。

**主要风险。**本次细读中直接处理噪声对应MVC的2025–2026工作只有 P02，其他为干净关系对齐、可信聚类或不同噪声任务的近邻；中英宽检索并集21条不能叫21篇直接竞争，也不能据此称“蓝海”。随机置换只是一种容易构造的合成噪声，若缺少真实错配集可能被质疑。CLIP/DINO特征过强时，单视图K-means就可能掩盖配对损害。OT批内矩阵为二次开销；CorreGen笔记提到的大batch不应直接视为轻量。RAV只有ZIP、多个仓库未见许可证，入门并非完全无摩擦。

|标准|分数|理由|
|---|---:|---|
|提升空间|3|核心错配聚类损失尚缺原表核实；相邻污染协议有退化，也有仅2点的饱和例子|
|拥挤程度|3|直接工作数量暂少、关系对齐与匹配近邻较多；资料不足和漏检风险同时存在|
|入门成本|3|CorreGen有README、配置和数据入口；OpenReview受限、torch未知、RAV仅ZIP|
|算力|5|预计算特征+小MLP适合作为8GB主实验；不包含端到端大编码器预训练承诺|
|工作量|4|可靠对应与部署变化可组成两章，但必须各自有独立问题和增益|
|实验室契合度|5|聚类×多模态，最贴近现有积累|

**第一阶段最小“提升空间”核查与对照。**先取得并读 CorreGen 原表/补充，按同一编码器分别提取“干净对应、实例置换、无对应污染”三条曲线，确认退化来自对应而非额外删样本或污染。未来最小验证选2个小数据，固定表征，比较**最佳单视图K-means、拼接K-means、忽略配对的聚合、CorreGen**；干净真实对应只作诊断对照，不可用标签调参数。ACC同时看NMI/ARI，若涉及少数类再看类平均召回，避免总ACC掩盖吞类。若强基线在合理错配率下相对干净只损失约2点，或忽略配对已打平，就停止以“修复对应”为主线。此门槛为选题建议，尚未执行，A 目前是**证据待补齐的候选**。

### B. 模态质量变化下的可靠融合与适度再平衡

**问题定义。**弱模态可能只是学习慢，也可能本身被污染；盲目增强两者会产生相反效果。研究小模型在训练/部署质量不一致时，怎样利用有用弱模态，同时抑制不可靠模态。应先固定一个任务族，例如图文分类或音视频情绪分类，不把情感回归、图文检索和多标签分类的指标混为一套。与普通缺失情感分析不同，主问题是**已观测模态的质量和作用变化**。

**已核实的空间与饱和风险。**CPSC 原文 [Table 2](https://arxiv.org/html/2605.03820v1) 的 MVSA-Single，指标为准确率（%）：clean80.07、Gaussian@10 63.32、SP@10 61.27，clean至受扰分别差**16.75、18.80个百分点**。这里10是噪声强度ε，不是10%缺失率。这是同一方法在不同数据条件下的退化证据，受扰程度的定义须沿用原文，不能理解成任意自然噪声。但同一表 Gaussian@10 的 EAU 为61.78，CPSC增益仅**1.54点**；SP@10 的 ECML 为61.04，CPSC仅多**0.23点**。因此“问题损失明显”已得到原文支持，“最新方法之后可恢复3–5点”尚未得到支持。CPSC Table 1 的 CREMA-D 为87.83，对LFM86.02仅高1.81点，也不支持随意许诺大幅提升。

**代表论文（5篇）。**

|论文 / 站内标签|问题与核心机制；本次定位|笔记|原文|代码|
|---|---|---|---|---|
|P09 G2D: Boosting Multimodal Learning with Gradient-Guided Distillation / ICCV2025|联合训练中强模态压制弱模态，分类与回归表征不足; G2D结合独立单模态教师的特征/logit蒸馏与按弱到强顺序的训练调制; B关键；原笔记仓库404，arXiv给出有效新地址|[笔记](https://papernotes.org/ICCV2025/multimodal_vlm/g2d_boosting_multimodal_learning_with_gradient-guided_distillation/)|[原文](https://arxiv.org/abs/2506.21514)|[仓库](https://github.com/rAIson-Lab/G2D) (corrected; old 404)|
|P10 MIDAS: Misalignment-based Data Augmentation Strategy for Imbalanced Multimodal Learning / NeurIPS2025|有标签多模态分类过度依赖主导模态; MIDAS制造语义冲突的跨模态组合，以单模态置信构造软标签并按弱模态和困难度加权; B直接竞争；笔记及摘要页未给出代码链接|[笔记](https://papernotes.org/NeurIPS2025/multimodal_vlm/midas_misalignment-based_data_augmentation_strategy_for_imbalanced_multimodal_le/)|[原文](https://arxiv.org/abs/2509.25831)|未提供|
|P11 Multimodal Negative Learning / NeurIPS2025|双模态后融合的弱模态margin低且对污染不稳健; MNL以置信与margin选择逐样本优势模态，仅蒸馏非目标类信息; B关键；对强动态融合的边际收益可能很小|[笔记](https://papernotes.org/NeurIPS2025/multimodal_vlm/multimodal_negative_learning/)|[原文](https://arxiv.org/abs/2510.20877)|[仓库](https://github.com/BaoquanGong/Multimodal-Negative-Learning)|
|P12 Rethinking Multimodal Learning from the Perspective of Mitigating Classification Ability Disproportion / NeurIPS2025|多模态模型的分类能力差距造成欠利用; AUG持续boosting弱模态分类器，按训练状态自适应分配分类器容量; B直接竞争及强基线；README为torch1.11.0|[笔记](https://papernotes.org/NeurIPS2025/multimodal_vlm/rethinking_multimodal_learning_from_the_perspective_of_mitig/)|[原文](https://arxiv.org/abs/2502.20120)|[仓库](https://github.com/njustkmg/NeurIPS25-AUG)|
|P16 Multimodal Learning on Low-Quality Data with Conformal Predictive Self-Calibration / CVPR2026|模态不平衡和输入噪声污染下样本及模态可靠性不同; CPSC将表征分量可靠性筛选、梯度校准和动态共形预测器更新组合成训练循环; B关键；已核原文表1/2，README现代torch但cu128未验证|[笔记](https://papernotes.org/CVPR2026/multimodal_vlm/multimodal_learning_on_low-quality_data_with_conformal_predictive_self-calibrati/)|[原文](https://arxiv.org/abs/2605.03820)|[仓库](https://github.com/XunCHN/CPSC)|

AUG（P12）、RollingQ（P13）、MASAM（P14）、BALM（P15）、EBMC（P17）还构成必须跟踪的竞争地图：能力差距、注意力协作、优化几何、缺失概率不均和情感增强均有人覆盖。MASAM与MIDAS的完整链接见第7节，不因不在本表就忽略。

**两个可能的工作点，均待验证。**

1. 区分“可学习但暂弱”和“质量差且应降权”：以逐样本质量、学习动态及跨模态冲突的可辨识性为问题，控制弱模态训练干预。必须证明不同于 MNL 的优势模态选择、AUG 的持续增强和 CPSC 的可靠分量筛选，不能只是换一个权重公式。
2. 从训练内污染推广到测试质量变化：研究轻量融合在污染类型/强度变化时的校准与选择性使用，报告精度、拒绝率/覆盖率、质量估计误差和最差质量组表现。若共形校准被使用，必须明确其交换性/分布假设；不能宣称任意shift下自动有覆盖保证。

**主要风险。**直接相关9篇，纯梯度平衡非常拥挤。CPSC已经做自校准，不能把“加共形”当新意。原文校准假设不能自然覆盖分布变化；共形p值也不是“本次报警为假的概率”。冻结编码器能省显存，但不能证明解决了端到端编码器梯度竞争；若章节声称改善优化，至少需要小CNN端到端对照。G2D的多教师、视频帧、SAM多次反传可能增加内存和时间；本次没有显存实测。

|标准|分数|理由|
|---|---:|---|
|提升空间|3|受扰损失十余点已核实，但强方法间常不足2点；剩余可恢复空间缺少证据，保守给中分|
|拥挤程度|2|至少9篇细读近邻，普通再平衡创新压力大|
|入门成本|3|多份代码与公开小数据可查；版本分裂、特征处理与质量协议仍须统一|
|算力|4|小CNN或冻结特征适合8GB；多教师视频全量设置不能原样承诺|
|工作量|4|训练干预与部署质量变化可形成两章，需避免都归结为同一门控权重|
|实验室契合度|4|多模态契合强，聚类/联邦交叉不是主线|

**第一阶段最小核查与对照。**优先核 CPSC 的 MVSA-Single Gaussian/SP定义、训练与测试污染位置、干净格的模型训练方式、随机种子和数据切分；同时读 MNL、AUG 对应数据的原表，不能跨论文直接减分。未来最小验证先用一个固定图文特征集和一个小音视频CNN设置，对比**最佳单模态、等权后融合、动态置信融合、MNL或AUG、CPSC**；加入“完全丢弃坏模态”诊断对照，检查复杂算法是否只实现模态选择。质量估计/阈值只能用训练/验证集。只有在相同预算下，合理质量偏移仍造成明显损伤，且现有可靠融合未消除时才继续；跨至少2个数据集、多个强度稳定恢复约3点可作为后续目标，不是本次结果。若只能在极端人工噪声胜出或仅改善弱基线，降级为备选。

### C. 缺失模式异构下的分组个性化多模态联邦学习

**问题定义。**不同客户端可用的模态/传感器组合不同，类别分布也不同；相同缺失模式未必适合共享，相异模式也未必必须分离。以小型生理信号模型为主，研究可靠共享边界与个性化适配。它是联邦×多模态的专项方向，**不是重新推荐普通Clustered FL**；也不以生成补全缺失模态为核心。

**已核实的证据。**PEPSY 原文 [Table 1](https://arxiv.org/html/2510.22880v1) 的 PTBXL、`p_m=0.2`，指标为服务器测试集准确率（%）：IID、`p_s=0.2` 时 FedMAC78.56±0.47、PEPSY78.81±0.72，仅**0.25个百分点**；IID、`p_s=1` 时74.70→75.41，仅**0.71点**。non-IID、`p_s=1` 时48.38±0.59→66.75±5.32，差**18.37点**，但这是已有PEPSY获得的收益，不是新方法还可获得的空间，且方差不小。`p_m`是受影响样本内缺失模态比例，`p_s`是有缺失样本比例，不能混成一个缺失率。

FED-PRIME 原文 [Table 1](https://arxiv.org/html/2602.07081v1)：同一个 MissBoth 训练模型，在 Food101 的完整模态测试93.64±0.58、与训练缺失模式一致的测试84.44±2.65，差**9.20点**，说明推理可用信息变化仍影响强方法。它不代表无额外信息时能恢复9.20点；该实验只取最常见8类，不是全101类。表内 FedAvg-P 分数低也不能替代“容量、mask处理和本地微调预算匹配的FedAvg”对照。

**算力证据要分模型。**PEPSY Table 6 报训练显存 **2.15–2.86GB**，支持小模型主线在8GB上的可行性；论文是在A6000 48GB运行，这不表示算法需要48GB。其32客户端、每轮选10、本地3 epoch；正文概述写1000通信轮，但附录Table 4明确区分**PTBXL 1000轮、EDF 500轮**。应按数据集报告预算，README的500轮示例不能概括所有主实验。时间成本并不小。FED-PRIME Table 3 的原模型为 **16.80/23.68GB**（Food/MMIMDb），不适合8GB原样复现，适合作为短租云扩展。冻结主干权重但训练输入提示，仍须反传经过主干，不能把它等同于缓存最终特征。

**代表论文（5篇）。**

|论文 / 站内标签|问题与核心机制；本次定位|笔记|原文|代码|
|---|---|---|---|---|
|P18 Learning Reconfigurable Representations for Multimodal Federated Learning with Missing Data / NeurIPS2025|客户端模态集合和样本缺失模式同时异构，局部表征难以聚合; PEPSY用可检索嵌入控制表示缺失画像，并对相似画像进行非参数聚合与表征重配置; C主基线；原文报告约2.15–2.86GB训练显存，但完整协议1000轮|[笔记](https://papernotes.org/NeurIPS2025/optimization/learning_reconfigurable_representations_for_multimodal_federated_learning_with_m/)|[原文](https://arxiv.org/abs/2510.22880)|[仓库](https://github.com/nmduonggg/PEPSY)|
|P19 Federated Prompt-Tuning with Heterogeneous and Incomplete Multimodal Client Data / ICCV2025|图文联邦中客户端间和客户端内缺失模式异构，提示语义无法直接平均; FED-PRIME区分两组prompt，并用客户端间对齐聚合处理提示排列与语义差异; C关键近邻；原文16.80/23.68GB，不能直接宣称8GB可复现|[笔记](https://papernotes.org/ICCV2025/optimization/federated_prompt-tuning_with_heterogeneous_and_incomplete_multimodal_client_data/)|[原文](https://arxiv.org/abs/2602.07081)<br>[补充原文](https://openaccess.thecvf.com/content/ICCV2025/html/Phung_Federated_Prompt-Tuning_with_Heterogeneous_and_Incomplete_Multimodal_Client_Data_ICCV_2025_paper.html)|[仓库](https://github.com/hangpt01/FedPrime)|
|P20 FedMVP: Federated Multimodal Visual Prompt Tuning for Vision-Language Models / ICCV2025|联邦CLIP提示学习对本地已见类过拟合，未见类及域泛化不足; FedMVP以图像patch与类别属性文本生成视觉提示，只训练轻量PromptFormer等适配模块; C关键强对照；不是实际多模态传感器缺失任务|[笔记](https://papernotes.org/ICCV2025/multimodal_vlm/fedmvp_federated_multimodal_visual_prompt_tuning_for_vision-language_models/)|[原文](https://arxiv.org/abs/2504.20860)|[仓库](https://github.com/mainaksingha01/FedMVP)|
|P21 pFedMMA: Personalized Federated Fine-Tuning with Multi-Modal Adapter for Vision-Language Models / ICLR2026|CLIP联邦个性化与未见类、域泛化存在权衡; pFedMMA本地保留适配器上下投影，仅聚合跨模态共享投影; C强近邻；arXiv题目作者已核，ICLR归属未从OpenReview独立核实|[笔记](https://papernotes.org/ICLR2026/multimodal_vlm/pfedmma_personalized_federated_fine-tuning_with_multi-modal_adapter_for_vision-l/)|[原文](https://openreview.net/forum?id=aX3E6LirK5)<br>[补充原文](https://arxiv.org/abs/2507.05394)|[仓库](https://github.com/sajjad-ucsb/pFedMMA)|
|P22 FedAFD: Multimodal Federated Learning via Adversarial Fusion and Distillation / CVPR2026|图像、文本、图文检索客户端模型和任务不一致，局部及全局目标冲突; FedAFD使用公共多模态数据做双层对抗对齐、细粒度融合和集成蒸馏; C近邻；依赖公共数据，不与无公共数据设定混比|[笔记](https://papernotes.org/CVPR2026/llm_safety/fedafd_multimodal_federated_learning_via_adversarial_fusion_and_distillation/)|[原文](https://arxiv.org/abs/2603.04890)|[仓库](https://github.com/Chao2433/FedAFD)|

**两个可能的工作点，均待验证。**

1. 缺失模式相同而任务分布不同的客户端之间，如何识别负迁移、决定共享与本地保留，并控制分组抖动。PEPSY本身已利用相似缺失画像聚合，FED-PRIME已做提示对齐，故“按缺失模式聚类”本身不能作为创新。
2. 客户端进出、新缺失组合或缺失模式变化下的增量个性化：研究共享知识的可迁移部分与小规模本地适配，兼顾最差客户端和校准。需明确评测的变化在训练前是否可见，不能把已知模式重采样包装成真正新模式泛化。

**主要风险。**PEPSY代码README缺数据预处理和样例，PTBXL/EDF需补规范；PTBXL的12条ECG导联更像同一生理模态的多通道，把其直接称作“多模态”要在论文定义里说清楚。按主体切分，避免同一受试者片段跨训练测试；模态缺失与类别分布的相关性要独立控制。公开数据上的联邦模拟不等于真实隐私保证，客户端统计摘要可能泄露信息。最重要的是：**PEPSY、纯本地和FedAvg+本地适配可能已经足够**，本次原文并未证明超越它们的剩余空间。

|标准|分数|理由|
|---|---:|---|
|提升空间|3|困难设置已有PEPSY取得大收益，测试缺失仍有损伤；超越PEPSY与本地适配的剩余空间未证实|
|拥挤程度|3|中英联邦×多模态并集13条，直接缺失线索较少；个性化/提示邻域仍竞争明显|
|入门成本|3|有PEPSY等代码和公开信号数据；预处理缺口、旧依赖和协议差异增加成本|
|算力|4|PEPSY原文低于3GB支持主线；通信时间重，FED-PRIME需云卡或协议重设|
|工作量|5|可靠共享与动态模式适配形成连贯且有区别的两章|
|实验室契合度|5|联邦×多模态，个性化分组可连接聚类，但不强行堆图模块|

**第一阶段最小核查与对照。**先把PEPSY原表按数据集、IID/non-IID、`p_m/p_s`和指标完整抄对，核实其缺失mask生成、FedMAC实现与本地适配预算；查明完整信息/无缺失对照是否真的存在，缺失就写“损失上限未知”，不能拿IID代替。未来最小验证仅选一个小信号数据集、同一小编码器、约10–20客户端做协议试验，再决定扩展；该缩小设置是新协议，不冒充原文复现。强对照至少有**纯本地、FedAvg+同预算本地微调、FedProx/适当个性化方法、FedMAC、PEPSY**，并加入仅按缺失模式静态分组。按总体与最差客户端指标、完整/缺失条件和计算预算联合判断。若PEPSY与FedAvg+本地适配已打平，或缺失条件下最新方法仅比完整少约2点，停止做“缺失修复”；只有稳定负迁移或未见模式损伤仍明显时，才推进两章。

### 3.4 原文身份与仓库审计

身份抽查详细区分如下（题目见代表论文表或第7节）：

|论文|核对的作者|会议证据|
|---|---|---|
|P02 CorreGen|Haochen Zhou, Guofeng Ding, Mouxing Yang, Peng Hu, Yijie Lin, Xi Peng|仅作者README称ICLR2026 Oral；OpenReview验证/权限受限，未完成官方核实|
|P05 BML|Changhao He, Di Xue, Shuxian Li, Yanji Hao, Xi Peng, Peng Hu|CVF正式CVPR2026论文页，页码1627–1638|
|P08 GLC|Hongqing He, Jie Xu, Wenyuan Yang, Yonghua Zhu, Guoqiu Wen, Xiaofeng Zhu|CVF正式CVPR2026论文页，页码24813–24822；arXiv同题|
|P09 G2D|Mohammed Rakib, Arunkumar Bagavathi|arXiv题目作者已核；comments声明ICCV2025，未另核官方程序|
|P11 MNL|Baoquan Gong, Xiyuan Gao, Pengfei Zhu, Qinghua Hu, Bing Cao|arXiv题目作者已核；comments声明NeurIPS2025|
|P16 CPSC|Xun Jiang, Yufan Gu, Disen Hu, Yuqing Hou, Yazhou Yao, Fumin Shen, Heng Tao Shen, Xing Xu|arXiv题目作者已核；comments声明CVPR2026|
|P18 PEPSY|Duong M. Nguyen, Trong Nghia Hoang, Thanh Trung Huynh, Quoc Viet Hung Nguyen, Phi Le Nguyen|arXiv题目作者已核；comments声明NeurIPS2025|
|P19 FED-PRIME|Thu Hang Phung, Duong M. Nguyen, Thanh Trung Huynh, Quoc Viet Hung Nguyen, Trong Nghia Hoang, Phi Le Nguyen|CVF正式ICCV2025；arXiv上传2026不是会议2026|
|P21 pFedMMA|Sajjad Ghiasvand, Mahnoosh Alizadeh, Ramtin Pedarsani|arXiv题目作者已核；未确认站内ICLR2026归属|

MIDAS、AUG、BALM还核对了arXiv题目、作者和作者会议声明，FedMVP核对arXiv与CVF。原文核查记录见 [primary_audit.json](F:/研究生毕设/方向检索/verification/primary_audit.json) 和 [curated_checks.json](F:/研究生毕设/方向检索/verification/curated_checks.json)。

GitHub“最近推送”使用API的 `pushed_at`，是仓库推送时间，**不是经逐commit核对的代码修改时间，也不证明算法维护活跃**。日期为UTC；“未见”指API未识别许可证且根目录未见许可证文件，并非断言无任何授权或可随意复用。

|论文 / 仓库|状态及最近push（UTC）|许可证|README的torch/环境|
|---|---|---|---|
|P02 [XLearning-SCU/2026-ICLR-CorreGen](https://github.com/XLearning-SCU/2026-ICLR-CorreGen)|200; 2026-10-08T14:34:24Z|API未识别且根目录未见许可证；授权待核|未写具体版本|
|P03 [chenzhe207/RAV](https://github.com/chenzhe207/RAV)|200; 2026-05-19T05:33:57Z|API未识别且根目录未见许可证；授权待核|无README，未知|
|P05 [XLearning-SCU/2026-CVPR-BML](https://github.com/XLearning-SCU/2026-CVPR-BML)|200; 2026-07-01T09:16:34Z|MIT|未写具体版本|
|P09 [rakib062/G2D](https://github.com/rakib062/G2D)|404; -|未审计|旧地址404；已从原文找到新地址|
|P11 [BaoquanGong/Multimodal-Negative-Learning](https://github.com/BaoquanGong/Multimodal-Negative-Learning)|200; 2025-12-10T04:52:27Z|API未识别且根目录未见许可证；授权待核|未写具体版本|
|P12 [njustkmg/NeurIPS25-AUG](https://github.com/njustkmg/NeurIPS25-AUG)|200; 2025-10-21T13:04:25Z|API未识别且根目录未见许可证；授权待核|1.11.0；Python 3.8（README）|
|P18 [nmduonggg/PEPSY](https://github.com/nmduonggg/PEPSY)|200; 2026-07-04T15:47:01Z|API未识别且根目录未见许可证；授权待核|未写具体版本；Python 3.9（README）|
|P19 [hangpt01/FedPrime](https://github.com/hangpt01/FedPrime)|200; 2025-07-22T03:11:21Z|API未识别且根目录未见许可证；授权待核|未写具体版本；Python 3.8（README）|
|P20 [mainaksingha01/FedMVP](https://github.com/mainaksingha01/FedMVP)|200; 2025-10-08T23:42:23Z|MIT|1.12.1；torchvision 0.13.1、cudatoolkit 10.2（README）|
|P21 [sajjad-ucsb/pFedMMA](https://github.com/sajjad-ucsb/pFedMMA)|200; 2026-02-04T04:16:33Z|API未识别且根目录未见许可证；授权待核|未写具体版本|
|P08 [hhq-sr/GLGC](https://github.com/hhq-sr/GLGC)|200; 2026-03-09T10:20:57Z|API未识别且根目录未见许可证；授权待核|未写具体版本|
|P09 [rAIson-Lab/G2D](https://github.com/rAIson-Lab/G2D)|200; 2026-03-01T22:38:06Z|MIT|2.0.1；Python 3.11、CUDA 11.7（README）|
|P16 [XunCHN/CPSC](https://github.com/XunCHN/CPSC)|200; 2026-05-14T05:10:08Z|API未识别且根目录未见许可证；授权待核|2.9.0；README另写tested CUDA 12.1，未证明cu128|
|P07 [dk23lhl/CRAFT](https://github.com/dk23lhl/CRAFT)|200; 2026-09-29T17:39:42Z|API未识别且根目录未见许可证；授权待核|2.1.0（README）|

按用户给出的5060门槛，需要 **torch ≥2.7 且使用cu128构建**。本次没有安装或运行任何环境；老README的torch1.x/2.0/2.1需兼容迁移。CPSC虽写torch2.9，同时写tested CUDA12.1，不能据此判定已适配5060：系统CUDA工具包、驱动和PyTorch wheel构建不是一回事。README未给版本的一律记未知；没有读取requirements正文来猜版本。**所有仓库的cu128运行兼容性均未验证。**

## 4. 与当前评估方向的比较

对学生已经评估的长尾聚类和Clustered FL，只补竞争与决策差异，不重复完整调研。

|方向|本次新增证据/竞争|8GB主实验|两个章节的自然性|最先排除的失败模式|建议位置|
|---|---|---|---|---|---|
|已有：长尾深度聚类|P01 MiniClustering（站内ICLR2026）；P²OT/SP²OT为用户已有基线，本次不重核|冻结特征/小头适合；全量预训练另算|频率估计与鲁棒自训练可衔接，但需避开已有重加权|强自监督表征+强聚类已缓解失衡，尾类指标未提升|继续原评估；加入MiniClustering，别以普通重加权为创新|
|已有：Clustered FL|P30 FedGEM是数据混合分量估计，不等同于客户端分组；本次不重做IFCA/CFL综述|小CNN串行模拟可行，通信轮数耗时|分组与个性化可形成两章，也容易同质|FedAvg+同预算本地微调已经打平|保留强本地适配门槛|
|A 噪声对应聚类|CorreGen已EM+GMM+OT+虚拟样本；RAV/CLIB为近邻|最友好，注意OT矩阵|较自然，第二章需独立部署变化|强单视图或忽略配对已足够；核心原表仍未核|聚类偏好下优先补证据|
|B 质量变化与可靠融合|CPSC原文证实十余点损伤，但部分格强方法增益仅0.23–1.54点；9篇近邻|小CNN/冻结特征可行|训练干预与部署变化可以衔接|单模态选择或CPSC已消除可用增益|证据较清楚，但创新压力最大|
|C 异构缺失多模态FL|PEPSY低显存；IID增益仅0.25点，困难non-IID收益已被现有方法取得|小信号模型可行；FED-PRIME原样超8GB|共享边界与动态模式适配最连贯|PEPSY或FedAvg+本地适配已足够；信息预算不公平|希望交叉方向与完整工作量时优先|

三条新候选并无证据表明一定比长尾聚类/CFL更容易毕业。A的信息缺口较大，B较拥挤，C工程与评测复杂；应依据第一阶段核查结果收敛到一条，而不是把三条分别开成独立实验项目。

## 5. 不推荐或暂不优先的方向

1. **属性缺失图聚类、COMPLETER之后的选择性补全。**接受学生已有的上限审计结论，不以站内新笔记重新鼓励同类修复。P07强调协议差异，正确用法是核查信息预算，不是更换缺失协议直到出现大收益。
2. **普通模态梯度平衡、加一个SAM/蒸馏/共形模块。**P09–P17已覆盖相当多组合。只有能提出与既有可靠性选择不同的可检验问题，才进入B。
3. **泛GCD、普通未知K或“CLIP+聚类”。**P45–P47说明谱几何、概念选择和非参数混合均有直接竞争。GCD通常有旧类标签，不能当纯无监督聚类。教师模型、概念词典、K先验的额外监督必须匹配。只作为学生当前备选的竞争补充。
4. **图结构修复/联邦去噪作为首选。**P28有正式近邻，但本次未核实同方法干净–含噪损失，更未核实对强MLP、IGNN/FED-PUB一类基线的剩余空间。图×联邦的P23双知识分离很契合，但代码未提供，任务划分和强对照仍欠证据，因此暂不挤入前三。
5. **泛异配GNN或仅比GCN的图OOD方法。**P33/P34要求重新审视基线；P35部分数据可能近饱和，未核一手结果，不能以概念新颖代替增益证据。若再查，先聚焦特定结构偏移而非“解决所有图问题”。
6. **时序基础模型套壳、插补作为泛主线。**P38提示简单统计强基线，P40虽轻量但非平稳校准也已有工作。时序聚类并集4条，人工检查仅P37是明确直接工作，不能据少命中称无人研究；需要补AAAI/期刊和站外综述。必须先核点调整、阈值、窗口和主体切分，再谈提升。
7. **文本LLM聚类作为立即主选。**P43/P44有低GPU成本线索，但摘要/成对约束引入额外LLM语义监督，API成本、强嵌入基线下的剩余空间和噪声约束处理尚未核实。只有3条英文命中说明本站覆盖窄，不能证明领域空白。
8. **大模型预训练、大规模开放词汇检测训练、联邦LLM指令微调。**与主要8GB预算不合；短租云适合补充验证，不应承担整个硕士训练管线。P48/P49是轻量检测线索，缺一手显存证据，暂不承诺可落地。
9. **离线RL污染+SAM直接作为题目。**P51是Student Abstract线索且SAM应用已有；P50有较完整方法近邻。小MLP显存友好不代表实验简单，环境兼容、seed和强基线成本尚未核实，可后续观察。

## 6. 局限与使用方式

- 51篇笔记不是随机样本或系统综述全集，检索字段不是全文，关键词距离窗口可能漏掉跨句同义表达。每个方向都做了中英查询，但没有把网站之外的会议、期刊全部补齐；2026覆盖尤其不均匀。
- 除明确写“原文核实”的结果，CSV中的实验数值和局限来自站内笔记，**没有逐项核实**；“局限”也可能是笔记作者分析，并非原论文作者正式承认。每行以字段区分来源，不能把CSV当正式论文引用库。
- 核实的数值只是原文报告值，本次没有复现实验；原表仍可能受选择协议、随机性、数据泄漏或实现影响。不同论文的绝对分数不能直接比较。对照门槛与两个工作点都是本报告的建议，未证明创新性或预期增益。
- OpenReview的P02/P03访问受限，未绕验证或登录；P02身份只与作者README核对。A最关键的原文损失证据仍待补齐。B/C的部分会议身份仅由arXiv作者comments支持，pFedMMA的ICLR2026归属未确认。CVF核实项与作者自述项分别标注。
- 仓库审计只看README、根目录及API元数据，没有读训练实现、依赖文件正文或提交历史。公开仓库存在不等于完整可复现，近期推送也不等于适配新版torch。未见许可证的项目需进一步澄清授权；本次没有据此执行任何源码复用。
- 所有显存判断除原文PEPSY/FED-PRIME表格外均为架构层面的估计，未测本机显存。没有验证torch≥2.7/cu128、Windows/Linux差异、5060驱动或自定义CUDA算子。
- 候选A/B/C的两章划分是问题组织，不是已经验证的两项贡献。开题前仍需原文阅读、强基线可用性核查及学生后续最小验证，尤其防止“问题损失大、现有强方法已解决”这一第二种陷阱。

## 7. 细读论文目录与本地证据

下表会议年份是**站内标签**，核实状态以第3.4节与CSV为准；外链未抽查者不保证可访问。完整问题设定、核心方法、笔记实验/局限、原文和代码链接、核查状态、许可证、torch与显存判断保存在CSV中。CSV为UTF-8 BOM编码，51行、31字段；`datasets_and_results_from_note`与`limitations_from_note`完整保留笔记来源标记，`verified_findings`单列已抽查的一手发现，`primary_evidence_urls`和本地证据路径用于回查。`site_venue/site_year`不得当成已核录用信息；README未写版本和未审计的仓库均明确记未知。

|ID / 站内会议年|题目与笔记|原文|代码线索|本次用途|
|---|---|---|---|---|
|P01 / ICLR2026|[Mini-cluster Guided Long-tailed Deep Clustering](https://papernotes.org/ICLR2026/self_supervised/mini-cluster_guided_long-tailed_deep_clustering/)|[原文](https://openreview.net/forum?id=3JlljaiQwR)|[仓库](https://github.com/LZX-001/MiniClustering)|只补充直接竞争，不重新推荐已有方向|
|P02 / ICLR2026|[Uncover Underlying Correspondence for Robust Multi-view Clustering](https://papernotes.org/ICLR2026/self_supervised/uncover_underlying_correspondence_for_robust_multi-view_clustering/)|[原文](https://openreview.net/forum?id=a4S1nQay3b)|[仓库](https://github.com/XLearning-SCU/2026-ICLR-CorreGen)|A核心；OpenReview受阻，原文数值待核验|
|P03 / ICLR2026|[Relationship Alignment for View-aware Multi-view Clustering](https://papernotes.org/ICLR2026/self_supervised/relationship_alignment_for_view-aware_multi-view_clustering/)|[原文](https://openreview.net/forum?id=uRA9cT4MK6)|[仓库](https://github.com/chenzhe207/RAV)|A近邻；不是已验证的噪声对应方法；仓库只有ZIP|
|P04 / ICLR2026|[Calibrated Information Bottleneck for Trusted Multi-modal Clustering](https://papernotes.org/ICLR2026/multimodal_vlm/calibrated_information_bottleneck_for_trusted_multi-modal_clustering/)|[原文](https://openreview.net/forum?id=iedlZOdI0d)|个人主页，非已确认代码|A近邻；主页链接不是已确认的代码仓库|
|P05 / CVPR2026|[Bootstrapping Multi-view Learning for Test-time Noisy Correspondence](https://papernotes.org/CVPR2026/others/bootstrapping_multi-view_learning_for_test-time_noisy_correspondence/)|[原文](https://openaccess.thecvf.com/content/CVPR2026/html/He_Bootstrapping_Multi-view_Learning_for_Test-time_Noisy_Correspondence_CVPR_2026_paper.html)|[仓库](https://github.com/XLearning-SCU/2026-CVPR-BML)|A关键近邻；监督分类不能直接充当无监督聚类基线|
|P06 / CVPR2026|[Rethinking Cross-Modal Anchor Alignment for Mitigating Error Accumulation](https://papernotes.org/CVPR2026/multimodal_vlm/rethinking_cross-modal_anchor_alignment_for_mitigating_error_accumulation/)|[原文](https://openaccess.thecvf.com/content/CVPR2026/html/Liu_Rethinking_Cross-Modal_Anchor_Alignment_for_Mitigating_Error_Accumulation_CVPR_2026_paper.html)|未提供|检索任务近邻，成本与协议不同|
|P07 / NeurIPS2026|[Beyond Missing Rates: Rethinking Incomplete Multi-View Clustering with Protocol Divergence](https://papernotes.org/NeurIPS2026/self_supervised/beyond_missing_rates_rethinking_incomplete_multi-view_clustering_with_protocol_d/)|[原文](https://arxiv.org/abs/2606.04857)|[仓库](https://github.com/dk23lhl/CRAFT)|防止缺失损失很小陷阱；不重新推荐补全|
|P08 / CVPR2026|[Global-Graph Guided and Local-Graph Weighted Contrastive Learning for Unified Clustering on Incomplete and Noise Multi-View Data](https://papernotes.org/CVPR2026/multimodal_vlm/global-graph_guided_and_local-graph_weighted_contrastive_learning_for_unified_cl/)|[原文](https://openaccess.thecvf.com/content/CVPR2026/html/He_Global-Graph_Guided_and_Local-Graph_Weighted_Contrastive_Learning_for_Unified_Clustering_CVPR_2026_paper.html)<br>[补充原文](https://arxiv.org/abs/2512.21516)|[仓库](https://github.com/hhq-sr/GLGC)|A关键近邻；高斯污染不等于实例索引错配|
|P09 / ICCV2025|[G2D: Boosting Multimodal Learning with Gradient-Guided Distillation](https://papernotes.org/ICCV2025/multimodal_vlm/g2d_boosting_multimodal_learning_with_gradient-guided_distillation/)|[原文](https://arxiv.org/abs/2506.21514)|[仓库](https://github.com/rAIson-Lab/G2D) (corrected; old 404)|B关键；原笔记仓库404，arXiv给出有效新地址|
|P10 / NeurIPS2025|[MIDAS: Misalignment-based Data Augmentation Strategy for Imbalanced Multimodal Learning](https://papernotes.org/NeurIPS2025/multimodal_vlm/midas_misalignment-based_data_augmentation_strategy_for_imbalanced_multimodal_le/)|[原文](https://arxiv.org/abs/2509.25831)|未提供|B直接竞争；笔记及摘要页未给出代码链接|
|P11 / NeurIPS2025|[Multimodal Negative Learning](https://papernotes.org/NeurIPS2025/multimodal_vlm/multimodal_negative_learning/)|[原文](https://arxiv.org/abs/2510.20877)|[仓库](https://github.com/BaoquanGong/Multimodal-Negative-Learning)|B关键；对强动态融合的边际收益可能很小|
|P12 / NeurIPS2025|[Rethinking Multimodal Learning from the Perspective of Mitigating Classification Ability Disproportion](https://papernotes.org/NeurIPS2025/multimodal_vlm/rethinking_multimodal_learning_from_the_perspective_of_mitig/)|[原文](https://arxiv.org/abs/2502.20120)|[仓库](https://github.com/njustkmg/NeurIPS25-AUG)|B直接竞争及强基线；README为torch1.11.0|
|P13 / ICML2025|[RollingQ: Reviving the Cooperation Dynamics in Multimodal Transformer](https://papernotes.org/ICML2025/multimodal_vlm/rollingq_reviving_the_cooperation_dynamics_in_multimodal_transformer/)|[原文](https://arxiv.org/abs/2506.11465)|[仓库](https://github.com/GeWu-Lab/RollingQ_ICML2025)|B直接竞争；原始多帧Transformer不自动适合8GB|
|P14 / ICLR2026|[MASAM: Multimodal Adaptive Sharpness-Aware Minimization for Heterogeneous Data Fusion](https://papernotes.org/ICLR2026/optimization/masam_multimodal_adaptive_sharpness-aware_minimization_for_heterogeneous_data_fu/)|[原文](https://openreview.net/forum?id=AUKeDukcUi)|[仓库](https://github.com/Orange2107/MASAM-Multimodal-Adaptive-SAM)|B直接竞争；多次梯度与SAM增加训练开销|
|P15 / CVPR2026|[BALM: A Model-Agnostic Framework for Balanced Multimodal Learning under Imbalanced Missing Rates](https://papernotes.org/CVPR2026/multimodal_vlm/balm_a_model-agnostic_framework_for_balanced_multimodal_learning_under_imbalance/)|[原文](https://arxiv.org/abs/2603.19718)|[仓库](https://github.com/np4s/BALM_CVPR2026.git)|B直接竞争；不是补全缺失特征|
|P16 / CVPR2026|[Multimodal Learning on Low-Quality Data with Conformal Predictive Self-Calibration](https://papernotes.org/CVPR2026/multimodal_vlm/multimodal_learning_on_low-quality_data_with_conformal_predictive_self-calibrati/)|[原文](https://arxiv.org/abs/2605.03820)|[仓库](https://github.com/XunCHN/CPSC)|B关键；已核原文表1/2，README现代torch但cu128未验证|
|P17 / CVPR2026|[Enhance-then-Balance Modality Collaboration for Robust Multimodal Sentiment Analysis](https://papernotes.org/CVPR2026/multimodal_vlm/enhance-then-balance_modality_collaboration_for_robust_multimodal_sentiment_anal/)|[原文](https://openaccess.thecvf.com/content/CVPR2026/html/He_Enhance-then-Balance_Modality_Collaboration_for_Robust_Multimodal_Sentiment_Analysis_CVPR_2026_paper.html)|[仓库](https://github.com/kangverse/EBMC)|B近邻及已有MSA备选的直接竞争补充|
|P18 / NeurIPS2025|[Learning Reconfigurable Representations for Multimodal Federated Learning with Missing Data](https://papernotes.org/NeurIPS2025/optimization/learning_reconfigurable_representations_for_multimodal_federated_learning_with_m/)|[原文](https://arxiv.org/abs/2510.22880)|[仓库](https://github.com/nmduonggg/PEPSY)|C主基线；原文报告约2.15–2.86GB训练显存，但完整协议1000轮|
|P19 / ICCV2025|[Federated Prompt-Tuning with Heterogeneous and Incomplete Multimodal Client Data](https://papernotes.org/ICCV2025/optimization/federated_prompt-tuning_with_heterogeneous_and_incomplete_multimodal_client_data/)|[原文](https://arxiv.org/abs/2602.07081)<br>[补充原文](https://openaccess.thecvf.com/content/ICCV2025/html/Phung_Federated_Prompt-Tuning_with_Heterogeneous_and_Incomplete_Multimodal_Client_Data_ICCV_2025_paper.html)|[仓库](https://github.com/hangpt01/FedPrime)|C关键近邻；原文16.80/23.68GB，不能直接宣称8GB可复现|
|P20 / ICCV2025|[FedMVP: Federated Multimodal Visual Prompt Tuning for Vision-Language Models](https://papernotes.org/ICCV2025/multimodal_vlm/fedmvp_federated_multimodal_visual_prompt_tuning_for_vision-language_models/)|[原文](https://arxiv.org/abs/2504.20860)|[仓库](https://github.com/mainaksingha01/FedMVP)|C关键强对照；不是实际多模态传感器缺失任务|
|P21 / ICLR2026|[pFedMMA: Personalized Federated Fine-Tuning with Multi-Modal Adapter for Vision-Language Models](https://papernotes.org/ICLR2026/multimodal_vlm/pfedmma_personalized_federated_fine-tuning_with_multi-modal_adapter_for_vision-l/)|[原文](https://openreview.net/forum?id=aX3E6LirK5)<br>[补充原文](https://arxiv.org/abs/2507.05394)|[仓库](https://github.com/sajjad-ucsb/pFedMMA)|C强近邻；arXiv题目作者已核，ICLR归属未从OpenReview独立核实|
|P22 / CVPR2026|[FedAFD: Multimodal Federated Learning via Adversarial Fusion and Distillation](https://papernotes.org/CVPR2026/llm_safety/fedafd_multimodal_federated_learning_via_adversarial_fusion_and_distillation/)|[原文](https://arxiv.org/abs/2603.04890)|[仓库](https://github.com/Chao2433/FedAFD)|C近邻；依赖公共数据，不与无公共数据设定混比|
|P23 / ICLR2026|[Federated Graph-Level Clustering Network with Dual Knowledge Separation](https://papernotes.org/ICLR2026/graph_learning/federated_graph-level_clustering_network_with_dual_knowledge_separation/)|[原文](https://openreview.net/forum?id=FwKFjBX0PK)|未提供|交叉方向观察名单；笔记未提供代码|
|P24 / ICLR2026|[Low-pass Personalized Subgraph Federated Recommendation](https://papernotes.org/ICLR2026/recommender/low-pass_personalized_subgraph_federated_recommendation/)|[原文](https://openreview.net/forum?id=SSd3GENRAU)|未提供|粗筛；额外推荐协议与谱分解成本|
|P25 / NeurIPS2025|[Personalized Subgraph Federated Learning with Differentiable Auxiliary Projections](https://papernotes.org/NeurIPS2025/optimization/personalized_subgraph_federated_learning_with_differentiable_auxiliary_projectio/)|[原文](https://arxiv.org/abs/2505.23864)|[仓库](https://github.com/JhuoW/FedAux)|图联邦代表；对强FED-PUB边际提升需核验|
|P26 / NeurIPS2025|[Towards Effective Federated Graph Foundation Model via Mitigating Knowledge Entanglement](https://papernotes.org/NeurIPS2025/model_compression/towards_effective_federated_graph_foundation_model_via_mitigating_knowledge_enta/)|[原文](https://arxiv.org/abs/2505.12684)|未提供|粗筛；预训练与跨任务验证使工程量增加|
|P27 / CVPR2026|[FedSST: Rethinking Fair Federated Graph Learning under Structural Shift](https://papernotes.org/CVPR2026/others/fedsst_rethinking_fair_federated_graph_learning_under_structural_shift/)|[原文](https://openaccess.thecvf.com/content/CVPR2026/html/Zhao_FedSST_Rethinking_Fair_Federated_Graph_Learning_under_Structural_Shift_CVPR_2026_paper.html)|未提供|观察名单；笔记主表与消融数值不一致，未用作量化证据|
|P28 / CVPR2026|[FedSDR: Federated Graph Learning with Structural Noise Detection and Reconstruction](https://papernotes.org/CVPR2026/others/fedsdr_federated_graph_learning_with_structural_noise_detection_and_reconstructi/)|[原文](https://openaccess.thecvf.com/content/CVPR2026/html/Liu_FedSDR_Federated_Graph_Learning_with_Structural_Noise_Detection_and_Reconstruction_CVPR_2026_paper.html)|[仓库](https://github.com/Subtleazure/FedSDR)|不优先；属于结构修复，必须先证实噪声损失及强MLP基线|
|P29 / ICML2026|[FedSDR: Federated Self-Distillation with Rectification](https://papernotes.org/ICML2026/model_compression/fedsdr_federated_self-distillation_with_rectification/)|[原文](https://arxiv.org/abs/2605.18028)|未提供|与P28同名但任务不同；排除大模型训练路线|
|P30 / ICLR2026|[A Federated Generalized Expectation-Maximization Algorithm for Mixture Models with an Unknown Number of Components](https://papernotes.org/ICLR2026/others/a_federated_generalized_expectation-maximization_algorithm_for_mixture_models_wi/)|[原文](https://arxiv.org/abs/2601.21160)|未提供|与客户端模型分组的Clustered FL不同；高斯假设是主要风险|
|P31 / CVPR2026|[FedRG: Unleashing the Representation Geometry for Federated Learning with Noisy Clients](https://papernotes.org/CVPR2026/federated_learning/fedrg_unleashing_the_representation_geometry_for_federated_learning_with_noisy_c/)|[原文](https://openaccess.thecvf.com/content/CVPR2026/html/Wen_FedRG_Unleashing_the_Representation_Geometry_for_Federated_Learning_with_Noisy_CVPR_2026_paper.html)|[仓库](https://github.com/Tianjoker/FedRG)|后续候补；不是客户端聚类FL；自监督预训练和几何拟合有成本|
|P32 / ICML2026|[Learning Locally, Revising Globally: Global Reviser for Federated Learning with Noisy Labels](https://papernotes.org/ICML2026/optimization/learning_locally_revising_globally_global_reviser_for_federated_learning_with_no/)|[原文](https://arxiv.org/abs/2412.00452)|[仓库](https://github.com/cs-yuxintian/FedGR-ICML26)|候补；站内标ICML2026，尚未核实正式年份|
|P33 / NeurIPS2025|[Making Classic GNNs Strong Baselines Across Varying Homophily: A Smoothness-Generalization Perspective](https://papernotes.org/NeurIPS2025/graph_learning/making_classic_gnns_strong_baselines_across_varying_homophily_a_smoothness-gener/)|[原文](https://arxiv.org/abs/2412.09805)|[仓库](https://github.com/galogm/IGNN)|基线警戒；不能只超过弱GCN|
|P34 / ICLR2026|[Oversmoothing, Oversquashing, Heterophily, Long-Range, and More: Demystifying Common Beliefs in Graph Machine Learning](https://papernotes.org/ICLR2026/others/oversmoothing_oversquashing_heterophily_long-range_and_more_demystifying_common_/)|[原文](https://arxiv.org/abs/2505.15547)|[仓库](https://github.com/AdrianArnaiz/demystifyingGraphML)|方法论阅读；反例否定普遍性不等于普遍无效|
|P35 / NeurIPS2025|[Redundancy-Aware Test-Time Graph Out-of-Distribution Detection](https://papernotes.org/NeurIPS2025/llm_safety/redundancy-aware_test-time_graph_out-of-distribution_detection/)|[原文](https://arxiv.org/abs/2510.14562)|[仓库](https://github.com/name-is-what/RedOUT)|轻量备选；部分数据对接近饱和，社会网络收益弱|
|P36 / ICLR2026|[GraphUniverse: Synthetic Graph Generation for Evaluating Inductive Generalization](https://papernotes.org/ICLR2026/graph_learning/graphuniverse_synthetic_graph_generation_for_evaluating_inductive_generalization/)|[原文](https://arxiv.org/abs/2509.21097)|[仓库](https://github.com/LouisVanLangendonck/GraphUniverse)|评测近邻；合成社区不代表真实部署|
|P37 / AAAI2026|[Mask the Redundancy: Evolving Masking Representation Learning for Multivariate Time-Series Clustering](https://papernotes.org/AAAI2026/time_series/mask_the_redundancy_evolving_masking_representation_learning_for_multivariate_ti/)|[原文](https://arxiv.org/abs/2511.17008)|[仓库](https://github.com/yueliangy/EMTC)|轻量观察名单；需核对论文缩写与UEA切分，尚未核实一手结果|
|P38 / ICLR2026|[When Foundation Models Are One-Liners: Limitations and Future Directions for Time Series Anomaly Detection](https://papernotes.org/ICLR2026/time_series/when_foundation_models_are_one-liners_limitations_and_future_directions_for_time/)|[原文](https://openreview.net/forum?id=H27kvyG4qf)|未提供|强基线警戒；OpenReview未独立核实，不把相关性直接当显著性结论|
|P39 / ICLR2026|[TSPulse: Tiny Pre-Trained Models with Disentangled Representations for Rapid Time Series](https://papernotes.org/ICLR2026/time_series/tspulse_tiny_pre-trained_models_with_disentangled_representations_for_rapid_time/)|[原文](https://arxiv.org/abs/2505.13033)|模型卡，非代码仓库|轻量备选；不推荐自行预训练，论文数字未核实|
|P40 / ICLR2026|[Adaptive Conformal Anomaly Detection with Time Series Foundation Models for Signal Monitoring](https://papernotes.org/ICLR2026/anomaly_detection/adaptive_conformal_anomaly_detection_with_time_series_foundation_models_for_sign/)|未提供|[仓库](https://github.com/ibm-granite/granite-tsfm/tree/main/notebooks/hfdemo/adaptive_conformal_tsad)|轻量备选；p值不是单次报警为假的后验概率，依赖初始校准正常段|
|P41 / ICML2025|[Causality-Aware Contrastive Learning for Robust Multivariate Time-Series Anomaly Detection](https://papernotes.org/ICML2025/object_detection/causality-aware_contrastive_learning_for_robust_multivariate_time-series_anomaly/)|[原文](https://arxiv.org/abs/2506.03964)|[仓库](https://github.com/kimanki/CAROTS)|候补；需核对点调整和阈值选择协议|
|P42 / ICML2025|[KAN-AD: Time Series Anomaly Detection with Kolmogorov-Arnold Networks](https://papernotes.org/ICML2025/object_detection/kan-ad_time_series_anomaly_detection_with_kolmogorov-arnold_networks/)|[原文](https://arxiv.org/abs/2411.00278)|未提供|轻量候补；站内实验表缺少具体基准名，不支撑量化结论|
|P43 / ICLR2026|[Summaries as Centroids for Interpretable and Scalable Text Clustering](https://papernotes.org/ICLR2026/information_retrieval/summaries_as_centroids_for_interpretable_and_scalable_text_clustering/)|[原文](https://arxiv.org/abs/2502.09667)|未提供|算力友好候补；代码及强嵌入下剩余空间未核实|
|P44 / AAAI2026|[Optimized Algorithms for Text Clustering with LLM-Generated Constraints](https://papernotes.org/AAAI2026/aigc_detection/optimized_algorithms_for_text_clustering_with_llm-generated_constraints/)|[原文](https://arxiv.org/abs/2601.11118)|[仓库](https://github.com/weihong-wu/LSCK-HC)|算力友好候补；API成本与额外语义监督要记账|
|P45 / ICLR2026|[Bures-Isotropy Alignment: Manifold Learning of Generalized Category Discovery](https://papernotes.org/ICLR2026/self_supervised/bures-isotropy_alignment_manifold_learning_of_generalized_category_discovery/)|[原文](https://openreview.net/forum?id=nfVKTJ1MJ3)|[仓库](https://github.com/lytang63/BIA)|只补充直接竞争；不是空白的谱正则方向|
|P46 / ICLR2026|[SpectralGCD: Spectral Concept Selection and Cross-modal Representation Learning for Generalized Category Discovery](https://papernotes.org/ICLR2026/multimodal_vlm/spectralgcd_spectral_concept_selection_and_cross-modal_representation_learning_f/)|[原文](https://arxiv.org/abs/2602.17395)|[仓库](https://github.com/miccunifi/SpectralGCD)|只补充直接竞争；强ViT-H教师和词典信息预算必须匹配|
|P47 / CVPR2026|[Nonparametric Deep Fine-grained Clustering with Low-Rank Guided Vision-Language Model](https://papernotes.org/CVPR2026/self_supervised/nonparametric_deep_fine-grained_clustering_with_low-rank_guided_vision-language_/)|[原文](https://openaccess.thecvf.com/content/CVPR2026/html/Ye_Nonparametric_Deep_Fine-grained_Clustering_with_Low-Rank_Guided_Vision-Language_Model_CVPR_2026_paper.html)|[仓库](https://github.com/HenryWells02/VLMFine-Clustering)|直接竞争补充；不是简单加DPMM即可创新|
|P48 / CVPR2025|[ABRA: Teleporting Fine-Tuned Knowledge Across Domains for Open-Vocabulary Object Detection](https://papernotes.org/CVPR2025/object_detection/abra_teleporting_fine-tuned_knowledge_across_domains_for_open-vocabulary_object_/)|[原文](https://arxiv.org/abs/2603.12409)|未提供|有轻量适配线索；仍需训练域专家，不能按参数少推断8GB适配|
|P49 / CVPR2025|[Efficient Test-Time Adaptive Object Detection via Sensitivity-Guided Pruning](https://papernotes.org/CVPR2025/object_detection/efficient_test-time_adaptive_object_detection_via_sensitivity-guided_pruning/)|[原文](https://arxiv.org/abs/2506.02462)|未提供|仅粗筛；Faster R-CNN源模型与框架兼容成本高|
|P50 / ICLR2026|[Adaptive Scaling of Policy Constraints for Offline Reinforcement Learning](https://papernotes.org/ICLR2026/reinforcement_learning/adaptive_scaling_of_policy_constraints_for_offline_reinforcement_learning/)|[原文](https://openreview.net/forum?id=liOHottW7G)|未提供|小MLP环境可能适合8GB；多seed和仿真环境成本仍高|
|P51 / AAAI2026|[Enhancing Robustness of Offline RL Under Data Corruption via SAM](https://papernotes.org/AAAI2026/reinforcement_learning/enhancing_robustness_of_offline_reinforcement_learning_under_data_corruption_via/)|[原文](https://arxiv.org/abs/2511.17568)|未提供|Student Abstract不等于充分的大论文基线；只测少数环境|

可复核文件：

- [search_index.json](F:/研究生毕设/方向检索/search_index.json)：索引原文件。SHA256：`bd24c6a64805cfb81bb90a8dafee1b8eaa4fab01579a46a456826085a27cd4cb`。
- [search_log.md](F:/研究生毕设/方向检索/search_log.md)、[search_counts.json](F:/研究生毕设/方向检索/search_counts.json)、[search_results.json](F:/研究生毕设/方向检索/search_results.json)：查询规则、计数及命中集合。
- [read_notes.json](F:/研究生毕设/方向检索/read_notes.json)、[paper_annotations.json](F:/研究生毕设/方向检索/paper_annotations.json)：51篇笔记分节文本与调研摘要；原始页面在 `F:/研究生毕设/方向检索/notes/`，按P编号命名。
- [requests.jsonl](F:/研究生毕设/方向检索/requests.jsonl)：后续50篇笔记请求记录；P01在最初读取时保存为sample_note后纳入，总计51个不同笔记页。
- [github_audit.json](F:/研究生毕设/方向检索/verification/github_audit.json)：14条仓库审计记录，包含失效旧链接；README本地快照及原文HTML、表格在同一verification目录。
- [verified_findings.json](F:/研究生毕设/方向检索/verification/verified_findings.json)：关键核实结果与一手URL，包含PEPSY正文/附录通信轮数差异。
- [P16_tables.json](F:/研究生毕设/方向检索/verification/P16_tables.json)、[P18_tables.json](F:/研究生毕设/方向检索/verification/P18_tables.json)、[P19_tables.json](F:/研究生毕设/方向检索/verification/P19_tables.json)、[P08_tables.json](F:/研究生毕设/方向检索/verification/P08_tables.json)：原文HTML提取的完整表格，保留caption以核对条件。

本地处理仅用于下载索引/网页、解析文本及生成报告清单，没有建立或运行论文实现，没有改动指定目录以外的文件。
