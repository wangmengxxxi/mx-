# -*- coding: utf-8 -*-
"""Record stopped experiment state without importing torch or restarting work."""
from pathlib import Path
import datetime,hashlib,json,re

root=Path(__file__).resolve().parents[1];res=root/'results'
completed=[];partial=[]
for p in sorted((res/'full').glob('*/result.json')):
    r=json.loads(p.read_text(encoding='utf-8'))
    completed.append(dict(run=p.parent.name,dataset=r['dataset'],rate=r['rate'],data_seed=r['data_seed'],epochs=r['epochs'],B_ACC=r['branches']['B']['ACC'],E_minus_B=r['branches']['E']['ACC']-r['branches']['B']['ACC'],result_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),checkpoint_sha256=r['checkpoint_sha256']))
for folder in sorted((res/'full').iterdir()):
    if not folder.is_dir() or (folder/'result.json').exists():continue
    log=folder.parent/(folder.name+'.stdout.log')
    text=log.read_bytes().decode('utf-8',errors='replace') if log.exists() else ''
    epochs=re.findall(r'Epoch : (\d+)/500',text)
    partial.append(dict(run=folder.name,last_logged_epoch=int(epochs[-1]) if epochs else 0,checkpoint_exists=(folder/'model.pt').exists(),note='Interrupted at user request; last logged epoch is only a lower bound, not a resumable checkpoint'))
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
state=dict(status='paused_by_user',time_shanghai=now,reason='User plans to switch to GPU PyTorch to accelerate execution',background_training_stopped=True,completed_count=len(completed),planned_count=20,completed=completed,interrupted=partial,remaining_count=20-len(completed),full_refcheck_completed=False,full_20_run_audit_completed=False,final_report_completed=False,protocol_sha256=hashlib.sha256((res/'protocol.json').read_bytes()).hexdigest(),environment=str(root/'envs/imvc_cpu/python.exe'),resume_note='Do not restart automatically. Keep CPU results; after GPU setup, explicitly reconcile device/runtime changes and output directories before resuming. Existing runner and queue hardcode CPU.')
(res/'pause_state.json').write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
with (res/'batch_events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(event='paused_by_user',time=now,completed=len(completed),interrupted=[r['run'] for r in partial]),ensure_ascii=False)+'\n')
rows='\n'.join('| '+r['run']+' | '+f"{r['B_ACC']:.6f}"+' | '+f"{r['E_minus_B']:+.6f}"+' |' for r in completed)
parts='\n'.join(f"- `{r['run']}`：最后已记录第 {r['last_logged_epoch']} 轮；保存最终模型={r['checkpoint_exists']}。" for r in partial)
md=f'''# IMVC 任务暂停记录

暂停时间（北京时间）：{now}。原因：用户准备更换 GPU 版 PyTorch，以加快执行。**队列及当前训练进程已停止，不自动恢复。没有安装或尝试 GPU。**

## 已完成与未完成

正式 500 轮运行完成 **{len(completed)}/20** 组，全部保存在 `results/full/`，每组包含同一模型的 A/B/E/C_low_s1/C_low_s2/C_rand 六个分支、掩码、模型、表示、选点、预测和哈希。

| 已完成 CPU 组 | B ACC | E−B ACC |
| --- | --- | --- |
{rows}

中断的运行（不计为正式完成）：

{parts}

训练脚本只在最终轮结束后保存模型；中断组没有可续接的中间权重，恢复时须从头训练。最后日志轮次不是实际停止轮次的精确值。

尚欠：14 组正式运行（包括上述两个中断组）、一次 Caltech r0.5/s1 的独立完整复跑、全部 20 组最终审计和最终判读。0.7 辅助实验尚未开始。`第一阶段IMVC可行性报告.md` 是六组进度稿，不是全量最终报告。

## 已固定的研究协议

- 唯一 baseline：官方 COMPLETER，commit `624dd9fd10431802214fab1d7b34c44f5a0a92df`，MIT。两数据集：Caltech101-20（2386×[1984,512]，20 类）、Scene-15（4485×[20,59]，15 类）。源代码/数据 Git blob 与 SHA-256 已核对。
- 原协议 `results/protocol.json`，SHA-256 `{state['protocol_sha256']}`，判读阈值不更改。
- 缺失率 0.5 主、0.7 辅；官方掩码 seed 1–5。官方固定模型 seed：Caltech=4、Scene=8；500 轮，100 轮间隔的无标签模式切换，标签只做最终离线评价。
- 分支共享模型/掩码；最终 KMeans seed=20261027、n_init=10、lloyd；s1 为负 JS 软分配差异，s2 为 10 次无标签对齐聚类一致率。最低 50% 与等数量固定随机 oracle 对照，选点全部存档。

## 当前五种子主结果（仅 Caltech 完整）

Caltech r0.5：B−A=+0.113328（5/5 正），E−B=+0.000838（4/5 正）。按预设 E−B<+0.01，该数据集“不通过”；这不代表完整双数据集任务已经结束。s1/s2 Spearman 均值 +0.137734/−0.195278，均未达 −0.3；C_low_s1−C_rand=−0.000084，C_low_s2−C_rand=−0.001257。

Caltech 复现 B ACC 均值 0.725063，论文 0.6844，偏高 4.0663 个百分点，超出约 3pp 可接受范围；偏差必须保留，不能因为切换设备而删除已有结果。Scene 目前只有一个完整种子，不能五次判读。当前两个数据集独立 CPU 单次训练约 7.7/10.7 分钟，没有触发 HSACC 备用条件。

## 环境与恢复要点

- 当前独立 CPU 环境：`{state['environment']}`；Torch 运行时 2.8.0+cpu，NumPy 1.26.4、SciPy 1.13.1、sklearn 1.4.0；完整清单见 `environment.txt`。已有 pytorch 源环境、旧 amgc_cpu 和旧方向文件没有被安装/修改。
- 代码实际在 `scripts/completer/`，原始 `third_party/` 保持只读；改动 patch 为 `scripts/completer_compatibility.patch`。运行入口 `scripts/run_diagnostic.py`，队列 `scripts/run_batch.py`，核验 `scripts/audit_results.py`，报告汇总 `scripts/summarize_results.py`。
- **当前入口与队列写死 CPU，并隐藏 CUDA 设备。只更换环境不会自动启用 GPU。** GPU 环境准备好后，需要另行适配设备及张量路径，保留现有 CPU 代码、协议和结果；不要直接重启本 CPU 队列误以为已使用 GPU。
- 原硬件信息是 GTX 1650 4GB、旧驱动（CUDA 11.1）；本次没有探测或验证新 GPU 安装。GPU 恢复前应核对实际 torch/CUDA/驱动版本及单组显存、耗时，并先冒烟。
- CPU 与 GPU 会改变数值路径。建议 GPU 正式结果单独输出并保持同一设备/版本的一整套主表，6 组 CPU 保留为参照；不要无说明地把 CPU/GPU 的不同路径混成原 CPU 复现均值。如何续跑由用户后续指令确定，本记录不启动新实验。

## 已做的核验

修正后的冒烟核验通过；两个数据集首个完整 CPU 组均已通过掩码重建、分支重建、重新 KMeans、保存权重重编码、误差/得分/指标核对。其中 Scene 另与官方评价函数四位小数输出核对一致。证据：`results/smoke/audit.json`、`results/pilot_audit.json`、`results/scene_pilot_audit.json`。全量 20 项最终审计尚未执行。

机器可读断点：`results/pause_state.json`；停止前进程：`results/pause_process_snapshot.json`；事件日志：`results/batch_events.jsonl`。六个已完成 result 文件的 SHA-256 也保存在 pause_state 中。
'''
(root/'任务暂停记录_20261027.md').write_text(md,encoding='utf-8')
report=root/'第一阶段IMVC可行性报告.md'
if report.exists():
    text=report.read_text(encoding='utf-8').replace('实验进行中，暂不判读','用户已暂停，暂不判读')
    banner='> 状态：用户于 2026-10-27 暂停，正式运行完成 6/20。本文是进度稿，不是最终报告。恢复说明见 `任务暂停记录_20261027.md`。\n\n'
    if not text.startswith('> 状态：'):report.write_text(banner+text,encoding='utf-8')
print('Pause recorded:',len(completed),'completed;',len(partial),'interrupted')
