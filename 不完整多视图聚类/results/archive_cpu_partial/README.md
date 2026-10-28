# CPU 部分结果归档

2026-10-27：按用户决定改用 GPU，CPU 队列已手动中止。以下文件移动归档，未删除；原始报告复制归档。
这些结果不参与新的正式统计，只作 CPU/GPU 设备对照。没有 result.json 的半成品没有最终检查点，不能接续训练。

| 组 | 状态 |
| --- | --- |
| Caltech101-20_r0.5_s1 | 完成 500 轮 |
| Caltech101-20_r0.5_s2 | 完成 500 轮 |
| Caltech101-20_r0.5_s3 | 完成 500 轮 |
| Caltech101-20_r0.5_s4 | 完成 500 轮 |
| Caltech101-20_r0.5_s5 | 完成 500 轮 |
| Scene-15_r0.5_s1 | 完成 500 轮 |
| Scene-15_r0.5_s2 | 中断，无 result.json |
| Scene-15_r0.5_s3 | 中断，无 result.json |

其余 0.7 组尚未启动。smoke、benchmark、operational_restarts、下载/来源记录保持原位。
协议未改；原 CPU runtime/adaptation 清单与暂停记录保存在本目录。脚本原版另存 scripts/cpu_before_gpu/。
