# 第二阶段第三方代码使用与适配记录

- 固定来源：PPOT `33bd8872a87203ddb753cb94a05d639a414c6c06`；MiniClustering `e17dddcee21294e6d274bcdde8086ada7c1a08ff`。下载包和解压源码仅在被忽略的 `third_party/`。
- 实际执行的第三方代码只有 PPOT 的 `utils/sinkhorn_knopp.py` 中 `SemiCurrSinkhornKnopp`，由 `run_stage2.py` 通过绝对路径加载。该文件已通读；无网络、删除、上传、文件写入或子进程操作。所有入口用 `python -I` 启动。
- **第三方文件没有被复制或修改**，因此没有可应用到第三方源码的代码 patch。P2OT_full 复用原求解器；P2OT_TRUEPRIOR 的边缘分布替换在本项目独立的 `PriorSolver` 类中，两个分支的其余训练框架相同。头、memory bank、冻结特征双视图等适配均在本阶段新脚本中，源码及差异见 `run_stage2.py` 和冻结协议。
- MiniClustering 官方代码未执行。`MiniClustering_feat` 是按论文损失和结构在冻结特征上的独立诊断适配，使用同一特征作两个视图，不代表官方 CIFAR-100 细类结果；论文与代码没有该条件的设置。固定代码 `MiniClustering.py` 第 190 行有 TabError，数据集入口的 CIFAR100 转为 CIFAR-20 超类。若未来要修补并运行，应在独立后续工作中保存实际 patch 和新协议；本阶段不在结果之后补调参数。
- PPOT 固定仓库含 `LICENSE`，为 CC BY-NC 4.0；MiniClustering 固定仓库无 LICENSE。第三方源码不提交，不再分发，也不作为自研方法代码。
- 2026-10-30 11:15 的实现审计发现 memory bank 初始 5120 条错误地取了按真实类顺序拼接的前缀，违背冻结协议的种子打乱初始化。已暂停流水线，原 P2OT_full/TRUEPRIOR 结果与日志保存在 `results_stage2/superseded_pre_bank_fix/`；修正为种子排列的 5120 条后重新冒烟并重跑受影响结果。旧结果不进入任何汇总或判读，协议和阈值未改。
- KM CPU 核对时，阶段一将 sklearn 的 int32 预测显式转为 int64 再计算数组哈希。本阶段脚本已做相同转换；CIFAR100 IR100 seed1 的预测 SHA-256、BA 和 inertia 均与第一阶段完全一致。之前仅 dtype 不同的试跑 JSON 保留在 `superseded_pre_bank_fix/`，不进入正式结果。
- 同条件 ORACLE_KM 也已单独复算；预测 SHA-256 与 BA 和第一阶段完全一致。两项 CPU 复核保存在 `results_stage2/audit/`。
