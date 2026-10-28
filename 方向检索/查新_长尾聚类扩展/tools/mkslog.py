# -*- coding: utf-8 -*-
"""Assemble search_log.md from generated OpenAlex query table + manual records."""
import io, os

BASE = r"F:\研究生毕设\方向检索\查新_长尾聚类扩展"
tbl = io.open(os.path.join(BASE, "raw", "openalex_queries_table.md"), encoding="utf-8").read().strip()

head = """# 检索日志（search_log.md）

- **检索执行日期**：2026-10-27（Asia/Shanghai）
- **网络方式**：直连。按任务要求先检查 `127.0.0.1:7890`，**该端口未监听**（另查 7891/10808/10809/8080 亦未监听；`netsh winhttp show proxy` 为 Direct access），故全程未使用代理。
- **请求节流**：OpenAlex 每条检索式之间 sleep 0.5–1 s，并在 URL 中带 `mailto`；PDF 下载每文件间隔 1.5 s；GitHub API 只取元数据不 clone。
- **合规声明**：未登录任何站点，未绕过任何访问限制（遇到 403/429/付费墙一律记录为"不可达"并换源，不做规避）；未下载或运行任何论文代码。

---

## A. OpenAlex 结构化检索（主力）

检索端点：

```
GET https://api.openalex.org/works
    ?filter=title_and_abstract.search:<布尔式>,from_publication_date:<YYYY-MM-DD>
    &per-page=<N>&mailto=research@example.org[&sort=cited_by_count:desc]
```

- "命中总数" = OpenAlex 返回的 `meta.count`（该布尔式在**题名+摘要**上的全库匹配数，不是我们读过的篇数）。
- "已取回" = 实际写入 `raw/openalex/*.json` 的条数（受 `per-page` 限制）。
- 原始结果：`raw/openalex/BATCH_c1.txt`、`BATCH_c2.txt`、`BATCH_c1b_c2b.txt`、`BATCH_extra.txt`；每条的 JSON 在同目录。

### A.1 查询与命中数（共 53 条）

"""

mid = """

### A.2 关键词敏感性观察（重要）

同一意图换术语，命中数可以差一个数量级，**这直接影响"空白"判断的可信度**：

| 意图 | 严格写法 | 命中 | 放宽写法 | 命中 | 倍数 |
|---|---|---|---|---|---|
| 长尾 + 多视图 | `"long-tailed" AND "multi-view clustering"` | 2 | `"imbalanced" AND "multi-view clustering"` | 38 | **19×** |
| 长尾 + 聚类 | `"long-tailed" AND ("deep clustering" OR "unsupervised clustering")` | 13 | `(imbalanced OR "class imbalance") AND "deep clustering"` | 43 | 3.3× |
| 联邦聚类 + 不平衡 | `"federated clustering" AND ("class imbalance" OR "imbalanced classes" OR "long-tailed")` | 0 | `federated AND clustering AND (imbalanced OR "long-tailed" OR "class imbalance")` | 193 | **∞** |

**教训**：`"federated clustering" AND (...imbalanced...)` 命中 0 并不代表"联邦不平衡聚类没人做"——FedHC、Fed-$k^*$-HC 都做这件事，只是它们的题摘里用的是 "imbalanced clusters" / "cluster sizes are naturally imbalanced"，被严格短语式漏掉了。**本报告因此对所有"命中 0/极少"的检索式都补做了放宽式交叉验证**，并在正文中注明依据的是哪一条。

---

## B. OpenReview API v2 检索

端点：`GET https://api2.openreview.net/notes/search?term=<query>&limit=<N>&source=forum`

| # | 查询式 | 返回条数 | 有价值的命中 |
|---|---|---|---|
| B1 | `Partial Optimal Transport Deep Imbalanced Clustering` | 10 | P$^2$OT（ICLR 2024 poster, id=hD3sGVqPsr）；SP$^2$OT（仅 CoRR 2024，两条重复记录） |
| B2 | `federated clustering` | 30 | Jigsaw Game (TMLR 2024)；CCFC；Federated Spectral Clustering (NeurIPS 2023)；AFMCC；DFCA；**FedUGCN (ICLR 2027 desk rejected)**；Differentially Private Federated Clustering with Random Rebalancing；Personalized Federated Graph-Level Clustering (AAAI 2026)；Federated Multi-View Clustering via Tensor Factorization (IJCAI 2024) |
| B3 | `federated long-tailed` | 20 | Fed-GraB (NeurIPS 2023)；FedLoGe (ICLR 2024 poster)；Global Balanced Experts (ICCV 2023)；FedLF (ACML 2024)；FedPGRL (ICLR 2027 投稿) 等——**全部有监督** |
| B4 | `Mini-cluster Guided Long-tailed Deep Clustering` | 5 | MiniClustering（ICLR 2026 Poster，TLDR "can effectively handle the long-tailed data"） |

> 注：B2/B3 的返回里混有大量 CoRR/Crossref 镜像记录（同一论文多条），已按标题去重。

---

## C. 官方论文集定点核查

| 动作 | 端点/方法 | 结果 |
|---|---|---|
| ICLR 2024 论文集索引 | `GET https://proceedings.iclr.cc/paper_files/paper/2024` 后按标题搜 hash | 找到 P$^2$OT hash=`3d03791377a31cb3e7357014ba58eb80`，PDF 下载成功（1,557,848 B，EOF 校验通过） |
| ICLR 2026 论文集 | 同上（paper/2026） | 找到 MiniClustering hash=`813ed01a11220fcdeed10e0d7da2de25`，作者、摘要核实；**PDF 下载被限速**（约 9–12 KB/s），4 轮续传后 2,663,681 B 仍缺 `%%EOF`，表格数字未核实 |
| CVF CVPR 2024 | `GET https://openaccess.thecvf.com/CVPR2024?day=all` + 直接猜测 URL | DeepDPM (CVPR2022) PDF 完整下载；Federated GCD (CVPR2024) **Content-Length 9,119,078 B，限速下只取到 ~1.1 MB，未完整** |
| AAAI OJS | `https://ojs.aaai.org/index.php/AAAI/article/download/34429/36584` | AFCL (AAAI 2025) PDF 完整下载（1,903,785 B） |
| P$^2$OT OpenReview PDF | `https://openreview.net/pdf?id=hD3sGVqPsr` | **403 Forbidden**（加 Referer/UA 均无效），改用 proceedings.iclr.cc |
| Springer LNCS（FedHC） | `https://link.springer.com/chapter/10.1007/978-981-97-8487-5_35` | 302 跳转至 `idp.springer.com`（登录墙），**未获取** |
| ScienceDirect（Inf Sci 2025） | `https://www.sciencedirect.com/science/article/abs/pii/...` | **403**；改用 alphaxiv 镜像读 arXiv 摘要 |

---

## D. 引文追踪（cited by）

脚本：`tools/citedby.py`（OpenAlex `filter=cites:<ID>`）

| # | 根论文 | OpenAlex 施引总数 | 取回 | 关键词过滤后 | 结论 |
|---|---|---|---|---|---|
| D1 | DeepDPM (CVPR 2022, W4312387594) | 84 | 84（全量） | 57 | **未发现"无监督+长尾+未知K"三条件同时满足的工作**；发现 `DGD: Density Gradient-Guided Diffusion for Long-Tailed Clustering`（CVPRF 2026，线索）、`Beyond Predefined Clusters`（TKDE 2026 综述）、`Dying Clusters Is All You Need`（ICDMW 2024） |
| D2 | P$^2$OT (ICLR 2024) | **0**（明显错误） | — | — | **追踪失败**。OpenAlex 对 ICLR 论文的 `referenced_works` 抓取不完整 |
| D3 | MiniClustering (ICLR 2026) | 未执行 | — | — | 2026 年新论文，OpenAlex 尚无施引记录 |

**D2 是本报告最大的漏检风险**：无法用 OpenAlex 找到"引用了 P²OT 并进一步做长尾/未知 K"的工作。Semantic Scholar API 全程 429，Google Scholar 不可达，无法补偿。

---

## E. 通用搜索引擎 / 其他线索源

| # | 查询 | 引擎 | 用途 | 结果 |
|---|---|---|---|---|
| E1 | `P2OT Progressive Partial Optimal Transport Deep Imbalanced Clustering ICLR 2024` | web_search | 定位官方页面 | 得到 proceedings.iclr.cc 与 ar5iv 链接，确认 ICLR 2024 |
| E2 | `Mini-cluster Guided Long-tailed Deep Clustering ICLR 2026` | web_search | 定位官方页面 | 得到 proceedings.iclr.cc 官方摘要页（**据此把该论文从"仅 arXiv"升级为"官方论文集已核实"**） |
| E3 | `SP2OT Semantic-Regularized Progressive Partial Optimal Transport Imbalanced Clustering` | web_search | 确认 venue | 只有 arXiv/ar5iv，确认"仅预印本" |
| E4 | `federated clustering class imbalance heterogeneous class distribution` | web_search | 探索 | 噪声大（大量 ScienceDirect 链接），未产生新的高价值命中 |
| E5 | `long-tailed deep clustering unknown number of clusters` | web_search | 探索 | 噪声大，未产生新命中 |

> **papernotes.org 的使用**：检索中确实出现了 `zhaoyang97/Paper-Notes-en` 的 MiniClustering 笔记（AI 辅助生成）。**本报告未引用其任何内容**，只把它当作"存在这篇 ICLR 2026 论文"的线索，随后回到 `proceedings.iclr.cc` 官方页面核实。这符合任务对 papernotes 类站点的要求。

---

## F. 代码仓库核查（只读元数据，未 clone、未运行）

端点：`GET https://api.github.com/repos/<owner>/<repo>`

| 仓库 | license (SPDX) | stars | pushed_at | 备注 |
|---|---|---|---|---|
| `rhfeiyang/PPOT`（P$^2$OT） | `NOASSERTION` | 20 | 2026-06-24 | 无标准开源许可证 |
| `LZX-001/MiniClustering` | `none`（无 LICENSE 文件） | 1 | 2026-02-26 | 仅 1 star，代码较新 |
| `Yunfan-Zhang/AFCL`（AAAI 2025） | `none` | 2 | 2025-03-24 | — |
| `TPCD/FedGCD`（CVPR 2024） | `none` | 1 | 2024-03-14 | — |

另读取了 MiniClustering 仓库 README（仅文档）：训练入口 `main_byol.py`（BYOL 预训练）与 `MiniClustering.py`（聚类阶段），依赖 `torchrun` 单卡启动，致谢 `Hzzone/torch_clustering`。**未读取 requirements.txt，torch 版本未核实。**

---

## G. 不可达/受限数据源汇总

| 数据源 | 现象 | 处理 |
|---|---|---|
| arxiv.org / export.arxiv.org / ar5iv / r.jina.ai | 连接超时（DNS 可解析到 151.101.x.x） | 改用 alphaxiv（摘要级）+ OpenAlex 摘要重建 |
| Google Scholar | 超时 | 用 OpenAlex `cites:` 替代（覆盖不全） |
| api.semanticscholar.org | 全程 HTTP 429 | 放弃 |
| dblp.org | Anubis 反爬（"Making sure you're not a bot"） | 放弃，venue 核实改用官方论文集 |
| openreview.net/pdf | 403 | 改用 proceedings.iclr.cc / CVF / AAAI OJS |
| link.springer.com | 302 → idp.springer.com | 放弃（FedHC 内容未核实） |
| sciencedirect.com | 403 | 放弃（Inf Sci 2025 改用 alphaxiv 摘要） |
| papers.nips.cc / proceedings.neurips.cc | 间歇超时 | NeurIPS 论文改用 OpenReview 元数据（等级 B，未读 PDF） |
| github.com（HTML） | 间歇返回空 | 改用 api.github.com + raw.githubusercontent.com |

---

## H. 复现方式

```powershell
$py = "<bundled python>"
& $py tools\\batch.py   tools\\queries_c1.txt  raw\\openalex\\BATCH_c1.txt
& $py tools\\batch.py   tools\\queries_c2.txt  raw\\openalex\\BATCH_c2.txt
& $py tools\\batch.py   tools\\queries_c1b.txt raw\\openalex\\BATCH_c1b_c2b.txt
& $py tools\\batch.py   tools\\queries_x.txt   raw\\openalex\\BATCH_extra.txt
& $py tools\\citedby.py 10.1109/cvpr52688.2022.00963 "cluster;imbalanc;long-tail;number of" 4
& $py tools\\absfetch.py 10.48550/arxiv.2506.12515 10.1016/j.ins.2025.122957 ...
& $py tools\\pdftxt.py            # 抽取 raw/pdfs/*.pdf 到 raw/text/
& $py tools\\mkcsv.py             # 生成 papers.csv
```

检索式清单在 `tools/queries_c1.txt`、`queries_c2.txt`、`queries_c1b.txt`、`queries_x.txt`，可直接复用。
"""

io.open(os.path.join(BASE, "search_log.md"), "w", encoding="utf-8").write(head + tbl + mid)
print("search_log.md written")
