"""Verify the requested isolated GPU runtime against pinned CPU requirements."""
from pathlib import Path
import importlib.metadata,json,platform,subprocess,sys
import torch

root=Path(__file__).resolve().parents[1]
assert Path(sys.prefix).resolve()==(root/'envs/imvc_gpu').resolve()
assert platform.python_version()=='3.9.21'
assert torch.__version__=='2.8.0+cu126',torch.__version__
assert torch.cuda.is_available(),'CUDA unavailable'
name=torch.cuda.get_device_name(0)
assert 'GTX 1650 Ti' in name,name
versions={}
for line in (root/'scripts/gpu_requirements.txt').read_text(encoding='utf-8').splitlines():
    package,expected=line.split('==');actual=importlib.metadata.version(package)
    assert actual==expected,(package,actual,expected)
    versions[package]=actual
subprocess.run([sys.executable,'-I','-m','pip','check'],check=True)
data=dict(python=platform.python_version(),python_executable=sys.executable,torch=torch.__version__,
          cuda=torch.version.cuda,cuda_available=True,gpu_name=name,total_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
          compute_capability=torch.cuda.get_device_capability(0),torch_cuda_arch_list=torch.cuda.get_arch_list(),
          cudnn_version=torch.backends.cudnn.version(),other_packages=versions,
          nvidia_smi=subprocess.run(['nvidia-smi'],check=True,capture_output=True,text=True).stdout)
(root/'results/gpu_environment_verification.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(data,ensure_ascii=False,indent=2))
