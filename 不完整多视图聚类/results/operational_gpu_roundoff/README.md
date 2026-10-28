# GPU 数值校验失败存档

Caltech101-20 0.7 seed 1、Scene-15 0.7 seeds 2/3 均完成 500 轮，但在观测子批次和完整批次编码的一致性断言失败，没有 result.json。

原目录、日志及初始队列错误记录均移动保存，没有删除、没有按指标排除。原入口在 scripts/gpu_before_roundoff_fix/；复现及逐位修正核对在 results/gpu_numeric_check/。只调整 CUDA 编码数值复算容差，协议和判读阈值不变。
