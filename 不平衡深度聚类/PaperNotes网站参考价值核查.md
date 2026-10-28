# PaperNotes 网站参考价值核查

访问日期：2026-10-27。结论：**有用，适合作为长尾深度聚类的文献发现与查新入口；不足以直接作为实验有效性、创新性或可行性的证据。** 不因此改动当前冻结协议、方法矩阵或判读阈值。

## 已实际查看的内容

读取了[首页](https://papernotes.org/)、[关于本站](https://papernotes.org/about/)、完整搜索索引（26609 条索引记录，包含论文页及其他记录），并阅读三篇相关笔记。该记录数不是人工核验过的论文篇数，也不是本站相关方向的完整覆盖量。

“关于”页明确说明：这是独立论文笔记站，使用 AI 辅助工具，不保证全部已发布笔记经过逐句人工核验；公式、数字、引用、会议归属和解读仍可能有错，需核对原论文与作者代码。因此应把站内摘要、五星评分、“首次”“普遍有效”等评价与一手证据分开。

## 对当前方向最有用的线索

| 条目 | 对当前选题的具体用途 | 核查边界 |
| --- | --- | --- |
| [Mini-cluster Guided Long-tailed Deep Clustering](https://papernotes.org/ICLR2026/self_supervised/mini-cluster_guided_long-tailed_deep_clustering/) | 直接研究长尾深度聚类，利用过度聚类/mini-cluster 占据关系做无监督重加权。以后若把“过度聚类估计头尾、再重加权”当创新点，必须先与它比较。笔记还提示观察“头类被拆分、尾类被合并”的机制。 | [作者仓库](https://github.com/LZX-001/MiniClustering) README 的题名、作者、ICLR2026 BibTeX 和 [OpenReview 编号](https://openreview.net/forum?id=3JlljaiQwR) 与笔记对应。抽查官方 MiniClustering.py，明确将 encoder_k 和 projector_k 参数放入 Adam，并设 requires_grad=True；不能把“下游模块”直接等同于“冻结特征下仅改变分配”。尚未完成原论文全文、附录与全部实验表格的独立核验，笔记中的具体成绩不作为本次证据。 |
| [Samples Are Not Equal: A Sample Selection Approach for Deep Clustering](https://papernotes.org/ICLR2026/self_supervised/samples_are_not_equal_a_sample_selection_approach_for_deep_clustering/) | 提供密度偏置、原型初始化、伪标签稳定性/样本选择的相关工作线索，有助于后续区分类别频率不平衡与类别内部密度差异。 | [作者仓库](https://github.com/notoaudrey/Samples-Are-Not-Equal) README 自述为该题名的 ICLR2026 官方实现；[OpenReview](https://openreview.net/forum?id=SpsmpVo349)。不是专门证明本次 DINOv2 冻结特征、CIFAR100 IR100 仍有收益的证据。没有运行该代码。 |
| [CAE: Hierarchical Semantic Alignment for Image Clustering](https://papernotes.org/AAAI2026/others/hierarchical_semantic_alignment_for_image_clustering/) | 展示“training-free 图像聚类”文献线索，可供了解语义辅助与 OT。 | 笔记描述使用 CLIP、WordNet 名词和 Flickr 描述，意味着引入外部语义信息并融合表示，不等价于本次“同一个 DINOv2 特征完全不变”。其原文链接为 [arXiv:2512.00904](https://arxiv.org/abs/2512.00904)；本次未独立核实其全部数字与理论表述。 |

另外，标题中的 **Imbalanced View Contribution** 指视图贡献不均衡，不能仅凭“不平衡”一词把它当成类别长尾聚类工作；长尾识别、长尾半监督分类也不能直接当作完全无监督聚类的同协议 baseline。

## 对本次诊断的处理

1. 当前实验继续回答“强冻结特征是否出现长尾损失，以及固定特征下的分配参照空间”，保持 results/protocol.json 字节和 8/5 个百分点阈值不变。
2. MiniClustering 的存在说明这个研究方向已有直接竞争工作，**不能把长尾聚类本身、mini-cluster 重加权本身当成新的贡献**；也不能仅凭它的旧骨干/其他数据协议成绩判断本次强特征诊断必然通过。
3. 站内结果只作线索，不混入本次分支均值和配对差，不据此调 BAL、挑种子或增加新方法。后续只有在本次诊断确有空间之后，再另行冻结更广泛的 baseline 协议。
4. 选题“可行”至少包含问题空间与创新空间两层。本次 KM/BAL/oracle/LP 测的是前一层；该网站对发现后一层已有工作有帮助，但两层都需要一手证据。

HTML、搜索索引、作者 README、官方代码抽查文件和提交信息已保存在 results/、third_party/。MiniClustering 代码仅下载阅读，未导入或运行；当前实验仅执行已审阅的 DINOv2 和 P²OT 模块。该网页不是数据集、预训练权重或第三方实验代码的替代来源。
