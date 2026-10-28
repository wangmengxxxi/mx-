"""One-off, auditable CPU archival, runtime freeze and GPU admission checks."""
from pathlib import Path
import argparse,datetime,difflib,hashlib,json,shutil

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'results'
FROZEN='7ddf3260ccebdf199cea02efdb9f562e8afc66c6a725f05ff4bcf0d6553cb34a'
FILES=['run_diagnostic.py','run_batch.py','audit_results.py','summarize_results.py','prepare_adaptation.py']

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')

def archive():
    target=RESULTS/'archive_cpu_partial'
    if target.exists():raise RuntimeError('CPU archive already exists; refusing overwrite')
    assert sha(RESULTS/'protocol.json')==FROZEN
    groups=[]
    for p in sorted((RESULTS/'full').iterdir()):
        if p.is_dir():
            result=p/'result.json'
            groups.append(dict(group=p.name,completed=result.exists(),result_sha256=sha(result) if result.exists() else None))
    target.mkdir()
    shutil.copy2(ROOT/'第一阶段IMVC可行性报告.md',target/'第一阶段IMVC可行性报告.md')
    for name in ['runtime_manifest.json','adaptation_manifest.json','pause_state.json','pause_process_snapshot.json']:
        if (RESULTS/name).exists():shutil.copy2(RESULTS/name,target/name)
    names=['full','refcheck','batch_events.jsonl','batch.stdout.log','batch.stderr.log','batch_process.json','progress_summary.log',
           'summary.json','branch_means.csv','paired_differences.csv','paired_summary.csv','reliability_correlations.csv',
           'reliability_summary.csv','mask_metadata.csv','reproduction.csv','audit.json','audit.stdout.log','audit.stderr.log',
           'summary.stdout.log','summary.stderr.log','batch_errors.json']
    moved=[]
    for name in names:
        p=RESULTS/name
        if p.exists():shutil.move(str(p),str(target/name));moved.append(name)
    for group in groups:
        if group['completed']:assert sha(target/'full'/group['group']/'result.json')==group['result_sha256']
    dump(target/'archive_manifest.json',dict(groups=groups,moved=moved,protocol_sha256=FROZEN,reason='User requested full homogeneous GPU rerun'))
    rows=['# CPU 部分结果归档','', '2026-10-27：按用户决定改用 GPU，CPU 队列已手动中止。以下文件移动归档，未删除；原始报告复制归档。',
          '这些结果不参与新的正式统计，只作 CPU/GPU 设备对照。没有 result.json 的半成品没有最终检查点，不能接续训练。','',
          '| 组 | 状态 |','| --- | --- |']
    rows.extend('| '+g['group']+' | '+('完成 500 轮' if g['completed'] else '中断，无 result.json')+' |' for g in groups)
    rows.extend(['','其余 0.7 组尚未启动。smoke、benchmark、operational_restarts、下载/来源记录保持原位。',
                 '协议未改；原 CPU runtime/adaptation 清单与暂停记录保存在本目录。脚本原版另存 scripts/cpu_before_gpu/。'])
    (target/'README.md').write_text('\n'.join(rows)+'\n',encoding='utf-8')
    print(json.dumps(groups,ensure_ascii=False,indent=2))

def freeze():
    assert sha(RESULTS/'protocol.json')==FROZEN
    old=json.loads((ROOT/'scripts/cpu_before_gpu/runtime_manifest.json').read_text(encoding='utf-8'))
    # Only wrappers change. Reviewed training, loss, architecture, masking and loader stay fixed.
    for path,h in old.items():
        if path!='scripts\\run_diagnostic.py':assert sha(ROOT/path)==h,path
    patch=[]
    records=json.loads((ROOT/'scripts/cpu_before_gpu/adaptation_manifest.json').read_text(encoding='utf-8'))
    for name in FILES:
        before=ROOT/'scripts/cpu_before_gpu'/name;after=ROOT/'scripts'/name
        patch.extend(difflib.unified_diff(before.read_text(encoding='utf-8').splitlines(True),after.read_text(encoding='utf-8').splitlines(True),fromfile='scripts/cpu_before_gpu/'+name,tofile='scripts/'+name))
        records.append(dict(file='scripts/'+name,source_sha256=sha(before),copy_sha256=sha(after),changed=sha(before)!=sha(after),purpose='Device support, GPU verification and accurate device reporting; protocol unchanged'))
    paths=list(old)+['scripts\\'+name for name in FILES if 'scripts\\'+name not in old]
    for name in ['gpu_transition.py','run_gpu_checks.py','verify_gpu_environment.py','inspect_gpu_roundoff.py','inspect_failed_gpu_runs.py','verify_gpu_roundoff.py']:
        if (ROOT/'scripts'/name).exists():
            paths.append('scripts\\'+name)
            records.append(dict(file='scripts/'+name,source_sha256=None,copy_sha256=sha(ROOT/'scripts'/name),changed=True,purpose='New local GPU transition/validation utility, no third-party modification'))
            patch.extend(difflib.unified_diff([], (ROOT/'scripts'/name).read_text(encoding='utf-8').splitlines(True),fromfile='/dev/null',tofile='scripts/'+name))
    (ROOT/'scripts/gpu_device_adaptation.patch').write_text(''.join(patch),encoding='utf-8')
    dump(RESULTS/'adaptation_manifest.json',records)
    dump(RESULTS/'runtime_manifest.json',{p:sha(ROOT/p) for p in paths})
    print('Runtime frozen; protocol SHA-256',FROZEN)

def compare():
    base=RESULTS/'gpu_check'
    cpu=json.loads((RESULTS/'archive_cpu_partial/full/Caltech101-20_r0.5_s1/result.json').read_text(encoding='utf-8'))
    gpu=[json.loads((base/f'run{i}/result.json').read_text(encoding='utf-8')) for i in [1,2]]
    smoke=json.loads((RESULTS/'smoke_cpu_after_gpu/audit.json').read_text(encoding='utf-8'))
    assert smoke['all_passed'] and smoke['runs']==1
    audit=json.loads((base/'audit.json').read_text(encoding='utf-8'))
    assert audit['all_passed'] and audit['runs']==2
    speedups=[cpu['training_seconds']/r['training_seconds'] for r in gpu]
    metrics=['ACC','NMI','ARI'];branches=list(cpu['branches'])
    deltas={b:{m:gpu[1]['branches'][b][m]-gpu[0]['branches'][b][m] for m in metrics} for b in branches}
    max_acc=max(abs(d['ACC']) for d in deltas.values())
    mask_match=all(r['mask_sha256']==cpu['mask_sha256'] for r in gpu)
    runtime_ok=all(r['device']=='cuda' and r['torch_version']=='2.8.0+cu126' and 'GTX 1650 Ti' in r['gpu_name'] for r in gpu)
    passed=min(speedups)>=1.5 and max_acc<=.01 and mask_match and runtime_ok
    data=dict(passed=passed,protocol_sha256=sha(RESULTS/'protocol.json'),cpu_training_seconds=cpu['training_seconds'],
              gpu_training_seconds=[r['training_seconds'] for r in gpu],training_speedups=speedups,max_run_ACC_difference=max_acc,
              branch_run2_minus_run1=deltas,mask_match=mask_match,runtime_ok=runtime_ok,cpu_smoke_audit_passed=True,
              peak_memory_allocated_bytes=[r['peak_memory_allocated_bytes'] for r in gpu],warning_counts=[r['warning_count'] for r in gpu],
              checked_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat())
    assert data['protocol_sha256']==FROZEN
    dump(base/'comparison.json',data)
    lines=['# GPU 核对说明','', 'Caltech101-20，缺失率 0.5，掩码种子 1，固定最终 500 轮。两次从头训练，不复用 GPU 检查模型进入正式统计。',
           '协议 SHA-256：`'+FROZEN+'`。模型/损失/超参/训练轮数/掩码/分支/阈值不变；只迁移设备。', '',
           '| 项目 | CPU 归档 | GPU run1 | GPU run2 |','| --- | --- | --- | --- |',
           f"| 训练秒数（含既定中间无标签聚类） | {cpu['training_seconds']:.3f} | {gpu[0]['training_seconds']:.3f} | {gpu[1]['training_seconds']:.3f} |",
           f'| 相对 CPU 加速 | 1 | {speedups[0]:.3f}× | {speedups[1]:.3f}× |',
           '| 设备分配峰值 MiB | — | '+' | '.join(f"{r['peak_memory_allocated_bytes']/2**20:.1f}" for r in gpu)+' |','',
           '| 分支 | 指标 | CPU | GPU run1 | GPU run2 | run1−CPU | run2−run1 |','| --- | --- | --- | --- | --- | --- | --- |']
    for b in branches:
        for m in metrics:
            c=cpu['branches'][b][m];x=gpu[0]['branches'][b][m];y=gpu[1]['branches'][b][m]
            lines.append(f'| {b} | {m} | {c:.8f} | {x:.8f} | {y:.8f} | {x-c:+.8f} | {y-x:+.8f} |')
    lines.extend(['',f'两次 GPU 任一分支 ACC 最大绝对差：{max_acc:.8f}；掩码一致：{mask_match}。',
                  'CUDA：cuDNN deterministic=True、benchmark=False；CUBLAS_WORKSPACE_CONFIG=:4096:8；确定性算法 True、warn_only=True。',
                  f"警告数 run1/run2：{data['warning_counts']}；警告同时写入各次 stderr 和 warnings.jsonl（若有）。",
                  '显存峰值为 torch.cuda.max_memory_allocated，不包括驱动、显示及其他进程；另存 reserved 峰值。',
                  'CPU 设备路径使用原 imvc_cpu 完成 3 轮冒烟及 --recluster 审计。GPU 核对也独立审计，所有指标仍由 CPU NumPy/sklearn 计算。',
                  'CPU/GPU 差异仅描述，不据此修改协议；速度以同组训练计时比较。',
                  '', '**核对结论：'+('通过，可开始单进程 GPU 全量重跑。' if passed else '未通过，停止，不启动全量。')+'**'])
    (base/'GPU核对说明.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(data,ensure_ascii=False,indent=2))
    if not passed:raise RuntimeError('GPU admission conditions failed; do not start batch')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['archive','freeze','compare']);args=ap.parse_args()
    dict(archive=archive,freeze=freeze,compare=compare)[args.action]()
