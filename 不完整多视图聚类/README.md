# 第一阶段 IMVC 可行性诊断

> 2026-10-27：GPU 重跑已完成，正式 20/20 组及 1 次独立复跑均通过 --recluster 审计，后台队列已退出。按冻结规则总体判读为“不通过”；CPU 部分结果已归档，不参与正式统计。当前报告为最终版。

工作目录：`F:/研究生毕设/不完整多视图聚类`。只用 COMPLETER，GPU 单进程，固定最终 500 轮。第三方原始文件只读，运行的是 `scripts/completer/` 中的审阅副本。协议冻结于 `results/protocol.json`。

## 查看进度

`results/batch_events.jsonl` 是后台队列事件；每次运行的 stdout/stderr 在 `results/full/`。只有生成 `result.json` 才算完成；没有文件表示运行中或失败，不代表缺失数值为零。

## 复查结果

在项目根目录，用此环境执行，所有参数为显式路径：

```powershell
& 'F:\研究生毕设\不完整多视图聚类\envs\imvc_gpu\python.exe' -I 'F:\研究生毕设\不完整多视图聚类\scripts\audit_results.py' --root 'F:\研究生毕设\不完整多视图聚类' --runs 'F:\研究生毕设\不完整多视图聚类\results\full' --out 'F:\研究生毕设\不完整多视图聚类\results\audit.json' --recluster

& 'F:\研究生毕设\不完整多视图聚类\envs\imvc_gpu\python.exe' -I 'F:\研究生毕设\不完整多视图聚类\scripts\summarize_results.py' --root 'F:\研究生毕设\不完整多视图聚类'
```

第一条从保存权重、表示、预测与选点重算并核对；第二条不重新训练，生成所有均值、逐组配对差、相关性和报告，正式模式要求完整 20 项。正式结果不足时可附加 `--partial` 查看进度报告；此模式明确写“暂不判读”，不能当作最终结论。

## 单组复跑

选择一个新的 `--out` 目录以保留已有结果：

```powershell
& 'F:\研究生毕设\不完整多视图聚类\envs\imvc_gpu\python.exe' -I 'F:\研究生毕设\不完整多视图聚类\scripts\run_diagnostic.py' --root 'F:\研究生毕设\不完整多视图聚类' --dataset Caltech101-20 --rate 0.5 --seed 1 --epochs 500 --threads 4 --device cuda --out 'F:\研究生毕设\不完整多视图聚类\results\manual_refcheck\Caltech101-20_r0.5_s1'
```

不要从 `third_party/` 内运行，不要调用原始 `run.py`，不要用 base Python。后台队列恢复前应先确认没有另一个 `run_batch.py`/同组训练进程，避免并发覆盖同一目录。GPU 清单见 `environment_gpu.txt`，CPU 历史复建说明见 `环境复建说明.md`。`--device` 默认 cpu；GPU 必须显式给 cuda。

GPU 核对通过后，后台队列入口为 `scripts/run_batch.py --device cuda --workers 1`，必须用 `envs/imvc_gpu/python.exe -I`。队列会依次执行全部 20 次正式训练、1 次独立复跑、`audit_results.py --recluster` 和最终汇总。

## 目录

- `third_party/2021-CVPR-Completer/`：固定官方提交的所需文件快照，MIT。不是完整 git checkout，未下载 DCP 子模块。
- `data/`：官方文件逐字节副本。
- `scripts/`：审阅副本、兼容 patch、运行、核验和汇总脚本。
- `results/smoke/`、`results/benchmark/`：仅工程检查，不参与统计。
- `results/full/`：20 个正式共享模型诊断组；`results/refcheck/`：额外一次完整复跑。
- `results/operational_restarts/`：队列切换时中断的两次未完成启动，未形成最终结果，不是按性能排除的运行。
- `文献/`：CVF 原始论文。

最终交付为 `第一阶段IMVC可行性报告.md`；全部关键数值同时导出 CSV 和 JSON，报告可由存档重建。

GPU 完成记录：`results/gpu_transition_state.json`；最终文件哈希：`results/final_artifact_sha256.json`；环境复建见 [环境GPU复建说明](环境GPU复建说明.md)。数值校验适配、3 组失败复现及逐位修正核对见报告第 3 节和 `results/gpu_numeric_check/`，失败原件保留于 `results/operational_gpu_roundoff/`。旧暂停记录为历史记录，当前状态以本 README 及 GPU 完成记录为准。
