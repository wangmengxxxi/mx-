# 检索日志与可复核方法

访问日期：2026-10-27（Asia/Shanghai）。直连HTTPS成功；检查127.0.0.1:7890未监听，未使用代理。robots.txt允许/并声明Crawl-delay: 1；笔记请求串行、至少间隔1秒。只读、不登录、不提交表单、不评论。

索引只下载一次：https://papernotes.org/search/search_index.json 。共26,609条索引记录；首页标称26,590，两者差19。按去锚点论文形态路径去重仍为26,609，但不代表人工确认每条均为不同正式论文。检索字段是title+text短关键词摘要+tags，不是全文。大小9,205,988字节；SHA256：bd24c6a64805cfb81bb90a8dafee1b8eaa4fab01579a46a456826085a27cd4cb。

以下62条检索（31组中英文）使用不区分大小写正则，年份来自站内路径标签；全量含2024，近期为2025–2026。中英与跨主题存在重叠，不可相加当论文数；所有命中保存于search_results.json。宽词并非完整同义词词典，英文0命中不代表领域空白。

|方向|语言|查询词|全量|2025–2026|
|---|---|---|---:|---:|
|聚类|EN|deep clustering|8|8|
|聚类|ZH|深度聚类|6|6|
|聚类|EN|long-tailed / imbalanced clustering|3|3|
|聚类|ZH|长尾/不平衡聚类|6|5|
|聚类|EN|unknown number of clusters|2|2|
|聚类|ZH|簇数未知/类别数估计|4|4|
|聚类|EN|category discovery / GCD|36|33|
|聚类|ZH|类别发现/广义类别发现|33|31|
|聚类|EN|multi-view clustering|25|24|
|聚类|ZH|多视图/多模态聚类|22|22|
|聚类|EN|clustering + foundation / CLIP / DINO|7|6|
|聚类|ZH|基础模型/CLIP/DINO + 聚类|18|17|
|聚类|EN|text clustering|3|3|
|聚类|ZH|文本聚类|3|3|
|多模态|EN|missing modality|34|32|
|多模态|ZH|缺失模态/模态缺失|50|48|
|多模态|EN|modality imbalance|6|6|
|多模态|ZH|模态不平衡/模态失衡|18|18|
|多模态|EN|noisy correspondence|12|12|
|多模态|ZH|噪声对应/错配/未对齐|13|13|
|多模态|EN|multimodal fusion robustness|148|142|
|多模态|ZH|多模态融合鲁棒性|122|119|
|多模态|EN|multimodal sentiment / emotion|27|27|
|多模态|ZH|多模态情感/情绪|23|23|
|图学习|EN|graph clustering|5|5|
|图学习|ZH|图聚类|26|26|
|图学习|EN|graph structure learning|0|0|
|图学习|ZH|图结构学习|10|8|
|图学习|EN|heterophily / heterophilic|8|8|
|图学习|ZH|异配图/异质配性|2|2|
|图学习|EN|graph OOD|9|9|
|图学习|ZH|图分布外/分布偏移|28|26|
|图学习|EN|graph prompt|14|13|
|图学习|ZH|图提示|6|5|
|联邦学习|EN|clustered / personalized FL|28|28|
|联邦学习|ZH|聚类/个性化联邦|35|35|
|联邦学习|EN|FL with missing modality|3|3|
|联邦学习|ZH|缺失模态联邦|2|2|
|联邦学习|EN|federated clustering|5|5|
|联邦学习|ZH|联邦聚类|5|5|
|联邦学习|EN|federath noisy labels / federated noisy labels|1|1|
|联邦学习|ZH|联邦噪声标签|3|3|
|时间序列|EN|time series clustering|1|1|
|时间序列|ZH|时间序列聚类|3|3|
|时间序列|EN|anomaly detection + time series|22|22|
|时间序列|ZH|时间序列异常检测|28|28|
|时间序列|EN|imputation + time series|8|8|
|时间序列|ZH|时间序列插补/缺失|18|17|
|时间序列|EN|TS foundation models|31|31|
|时间序列|ZH|时序基础模型|76|74|
|交叉|EN|clustering x federated|7|7|
|交叉|ZH|聚类×联邦|11|11|
|交叉|EN|clustering x multimodal|29|28|
|交叉|ZH|聚类×多模态/多视图|39|39|
|交叉|EN|federated x multimodal|10|9|
|交叉|ZH|联邦×多模态|11|10|
|交叉|EN|graph x federated|7|7|
|交叉|ZH|图×联邦|25|23|
|目标检测|EN|lightweight / open-vocabulary detection|103|99|
|目标检测|ZH|轻量/开放词汇检测|113|108|
|强化学习|EN|offline RL / small environments|55|55|
|强化学习|ZH|离线强化学习/小环境|78|78|

## 实际检索正则

- deep clustering: `deep.{0,15}cluster|cluster.{0,15}deep`
- 深度聚类: `深度聚类`
- long-tailed / imbalanced clustering: `(?:long.tail|imbalan).{0,50}cluster|cluster.{0,50}(?:long.tail|imbalan)`
- 长尾/不平衡聚类: `(?:长尾|不平衡).{0,15}聚类|聚类.{0,15}(?:长尾|不平衡)`
- unknown number of clusters: `unknown.{0,25}(?:cluster|categor)|(?:cluster|categor).{0,25}(?:unknown|number estimation)|estimating.{0,20}(?:cluster|categor)`
- 簇数未知/类别数估计: `簇数|类别数|聚类数|未知类别`
- category discovery / GCD: `category discovery|class discovery|\bGCD\b`
- 类别发现/广义类别发现: `类别发现|类发现`
- multi-view clustering: `multi.view.{0,15}cluster|multi.modal.{0,15}cluster`
- 多视图/多模态聚类: `多视图聚类|多模态聚类`
- clustering + foundation / CLIP / DINO: `^(?=.*cluster)(?=.*(?:foundation|\bCLIP\b|\bDINO\b))`
- 基础模型/CLIP/DINO + 聚类: `^(?=.*聚类)(?=.*(?:基础模型|预训练|CLIP|DINO))`
- text clustering: `text.{0,20}cluster|short.text.{0,20}cluster`
- 文本聚类: `文本聚类|短文本聚类`
- missing modality: `missing.{0,20}modal|incomplete.{0,20}modal|modal.{0,20}missing`
- 缺失模态/模态缺失: `缺失模态|模态缺失|不完整模态|不完整多模态`
- modality imbalance: `modal.{0,25}(?:imbalan|dominance)|imbalan.{0,25}modal`
- 模态不平衡/模态失衡: `模态不平衡|模态失衡|模态偏置|模态偏差|模态主导`
- noisy correspondence: `noisy correspondence|noisy.{0,15}(?:alignment|pair)|(?:unpaired|unaligned|misaligned).{0,25}(?:modal|view)|partially.view.aligned`
- 噪声对应/错配/未对齐: `噪声对应|错误对应|错配|不对齐|未对齐|部分对齐`
- multimodal fusion robustness: `^(?=.*(?:multimodal|multi.modal|fusion))(?=.*(?:robust|noise|noisy|corrupt))`
- 多模态融合鲁棒性: `^(?=.*(?:多模态|融合))(?=.*(?:鲁棒|噪声|噪音|污染))`
- multimodal sentiment / emotion: `(?:multimodal|multi.modal).{0,30}(?:sentiment|emotion)`
- 多模态情感/情绪: `多模态情感|多模态情绪`
- graph clustering: `graph.{0,20}cluster`
- 图聚类: `图聚类|图节点聚类`
- graph structure learning: `graph structure learning|structure refinement|graph rewiring`
- 图结构学习: `图结构学习|图结构优化|图重连`
- heterophily / heterophilic: `heterophil`
- 异配图/异质配性: `异配|异质配`
- graph OOD: `^(?=.*graph)(?=.*(?:out.of.distribution|\bOOD\b|distribution shift))`
- 图分布外/分布偏移: `^(?=.*图)(?=.*(?:分布外|分布偏移))`
- graph prompt: `graph.{0,20}prompt|prompt.{0,20}graph`
- 图提示: `图提示|图 Prompt|图prompt`
- clustered / personalized FL: `^(?=.*federat)(?=.*(?:cluster|personaliz))`
- 聚类/个性化联邦: `^(?=.*联邦)(?=.*(?:聚类|个性化))`
- FL with missing modality: `^(?=.*federat)(?=.*(?:missing|incomplete))(?=.*modal)`
- 缺失模态联邦: `^(?=.*联邦)(?=.*(?:缺失|不完整))(?=.*模态)`
- federated clustering: `federated.{0,25}cluster|clustered federated`
- 联邦聚类: `联邦聚类|聚类联邦`
- federath noisy labels / federated noisy labels: `^(?=.*(?:federath|federat))(?=.*(?:noisy|noise).{0,10}label)`
- 联邦噪声标签: `^(?=.*联邦)(?=.*(?:噪声标签|标签噪声|噪声标注))`
- time series clustering: `time.series.{0,25}cluster|cluster.{0,25}time.series`
- 时间序列聚类: `时间序列聚类|时序聚类`
- anomaly detection + time series: `^(?=.*(?:time.series|temporal))(?=.*anomal)`
- 时间序列异常检测: `^(?=.*(?:时间序列|时序))(?=.*异常)`
- imputation + time series: `^(?=.*(?:time.series|temporal))(?=.*(?:imput|missing))`
- 时间序列插补/缺失: `^(?=.*(?:时间序列|时序))(?=.*(?:插补|填补|缺失|补全))`
- TS foundation models: `^(?=.*(?:time.series|\bTS\b))(?=.*(?:foundation|pre.train))`
- 时序基础模型: `^(?=.*(?:时间序列|时序))(?=.*(?:基础模型|预训练))`
- clustering x federated: `^(?=.*cluster)(?=.*federat)`
- 聚类×联邦: `^(?=.*聚类)(?=.*联邦)`
- clustering x multimodal: `^(?=.*cluster)(?=.*(?:multi.modal|multimodal|multi.view))`
- 聚类×多模态/多视图: `^(?=.*聚类)(?=.*(?:多模态|多视图))`
- federated x multimodal: `^(?=.*federat)(?=.*(?:multi.modal|multimodal|modalit))`
- 联邦×多模态: `^(?=.*联邦)(?=.*多模态)`
- graph x federated: `^(?=.*federat)(?=.*graph)`
- 图×联邦: `^(?=.*联邦)(?=.*图)`
- lightweight / open-vocabulary detection: `^(?=.*(?:detect|\bYOLO\b))(?=.*(?:lightweight|efficient|open.vocabulary|parameter.efficient|few.shot))`
- 轻量/开放词汇检测: `^(?=.*(?:检测|YOLO))(?=.*(?:轻量|高效|开放词汇|少样本))`
- offline RL / small environments: `offline reinforcement|offline.{0,10}\bRL\b|(?:reinforcement|\bRL\b).{0,20}(?:efficient|small)`
- 离线强化学习/小环境: `离线强化学习|离线RL|离线 RL|小环境`

## 筛选漏斗、人工处理与阅读量

从26,609条索引做62条查询，按2025–2026、任务相关性、8GB可行性及现有选题风险筛出51篇不同笔记（P01–P51）。没有用宽命中数推断全部直接竞争数。按问题/方法/实验/局限段细读；P48–P51用于检测与RL粗筛，P29等用于误命中排除。

人工排除/纠正：中文“图×联邦”的图像误命中；offline RL中的LLM/RLHF；P28与P29同名FedSDR任务不同；federated clustering的数据聚类与Clustered FL客户端分组不同；P05监督分类与P02无监督聚类不同；P08高斯特征污染与实例错配不同；P04个人主页不充作代码仓库；P09失效旧仓库从原文修正。用户的federath拼写同时扩展federated。时序聚类中英并集4条，明确直接相关仅P37；噪声对应宽查询并集21条，并非21篇聚类直接竞争。

总计请求/保存51个不同笔记页：P01最初下载sample_note后复制入notes；其余50篇有requests.jsonl记录。没有逐页爬全站，低于约150页上限。首页、关于、robots和索引为辅助页，不计为论文笔记。未再额外下载笔记页。

一手核查：候选A抽查P02/P05/P08（P02官方OpenReview受限）；B抽查P09/P11/P16并补P10/P12/P15；C抽查P18/P19/P21并补P20。另查P07协议审计。arXiv作者会议声明与CVF正式会议页分开记录。GitHub审计14个地址，13个存在，1个旧址404；只读取README、根目录结构和元数据，没有下载源代码/ZIP/模型/数据或运行论文代码。P05的PDF下载未完成，未使用其表格。原文数字只是原文报告，未复现。

## 51篇阅读清单

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
