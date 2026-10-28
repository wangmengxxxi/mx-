# GPU 环境复建与设备切换

本环境为 `F:/研究生毕设/不完整多视图聚类/envs/imvc_gpu`。原 `envs/imvc_cpu`、`F:/Anaconda` 下的环境和旧方向均不改动。

使用 CPU 原环境的 conda 显式包清单新建前缀，获得完全相同的 Python 3.9.21 和基础运行库，再从官方源安装 CUDA Torch，最后固定其他包版本：

```powershell
& 'F:\Anaconda\Scripts\conda.exe' create --prefix 'F:\研究生毕设\不完整多视图聚类\envs\imvc_gpu' --file 'F:\研究生毕设\不完整多视图聚类\environment_explicit.txt' --yes --no-default-packages
& 'F:\研究生毕设\不完整多视图聚类\envs\imvc_gpu\python.exe' -I -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
& 'F:\研究生毕设\不完整多视图聚类\envs\imvc_gpu\python.exe' -I -m pip install --requirement 'F:\研究生毕设\不完整多视图聚类\scripts\gpu_requirements.txt' --index-url https://pypi.org/simple
& 'F:\研究生毕设\不完整多视图聚类\envs\imvc_gpu\python.exe' -I 'F:\研究生毕设\不完整多视图聚类\scripts\verify_gpu_environment.py'
```

`verify_gpu_environment.py` 核对 Python 版本、Torch 2.8.0+cu126、CUDA 可用、GTX 1650 Ti、其余每一项固定包版本及 `pip check`。结果写 `results/gpu_environment_verification.json`，实际安装日志在 `results/gpu_environment_install.log`，最终完整包清单为 `environment_gpu.txt`。

驱动 621.01 的 `nvidia-smi` 显示 CUDA 13.5，表示驱动支持的上限；Torch 使用自带 CUDA 12.6 runtime。无需改变驱动或另装 CUDA toolkit。

设备变更前的 6 组 CPU 结果和 2 组中断任务移动至 `results/archive_cpu_partial`，不进入 GPU 正式统计。GPU 核对为两次独立的 Caltech101-20、0.5、seed 1、500 轮，输出 `results/gpu_check/run1`、`run2`。两次均须无错误，相对归档 CPU 训练加速均不低于 1.5，任一分支两次 ACC 差不超过 0.01，才允许全量。CPU/GPU 指标差异如实记录，不作为调参理由。

`run_diagnostic.py` 的 `--device` 默认 cpu。GPU 需要显式指定 `--device cuda`。批量入口 `run_batch.py --device cuda --workers 1` 自动使用 imvc_gpu，并强制 GPU workers=1。KMeans、可靠性打分及评价仍在 CPU 执行。协议文件保持原字节；设备变更 patch、原 CPU 脚本及哈希清单分别在 `scripts/gpu_device_adaptation.patch`、`scripts/cpu_before_gpu`、`results/adaptation_manifest.json` 和 `results/runtime_manifest.json`。
