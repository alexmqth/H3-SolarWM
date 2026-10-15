"""CPU-only reproducible import/ABI/toolkit gate for EXP-026/v4."""
from __future__ import annotations
import contextlib
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
OVERLAY=HERE/'overlay'
if Path(sys.path[0]).resolve()==OVERLAY.resolve():
    pass
if str(OVERLAY.resolve()) not in [str(Path(p).resolve()) for p in sys.path if p]:
    raise RuntimeError('overlay is not on PYTHONPATH')
import torch
start_initialized=torch.cuda.is_initialized()
import_error=None
native_path=None
with contextlib.redirect_stderr(io.StringIO()) as import_stderr:
    try:
        import flash_attn,flash_attn_2_cuda
        native_path=str(Path(flash_attn_2_cuda.__file__).resolve())
        if not native_path.startswith(str(OVERLAY.resolve())):
            raise RuntimeError('native extension did not load from overlay')
    except BaseException as error:
        import_error=f'{type(error).__name__}: {error}'
from diffusers.models.attention_dispatch import AttentionBackendName,_check_attention_backend_requirements
backend_error=None
with contextlib.redirect_stderr(io.StringIO()) as backend_stderr:
    try:_check_attention_backend_requirements(AttentionBackendName.FLASH)
    except BaseException as error:backend_error=f'{type(error).__name__}: {error}'
wheel=json.loads((HERE/'WHEEL_RECEIPT.json').read_text())
old=ROOT/'.venvs/solarwm-h3/lib/python3.10/site-packages/flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so'
new=OVERLAY/'flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so'
def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*2**20),b''):digest.update(block)
    return digest.hexdigest()
old_sha,new_sha=sha(old),sha(new)
os.environ['CUDA_HOME']='/usr/local/cuda-13.0'
from torch.utils.cpp_extension import _check_cuda_version,TorchVersion
try:_check_cuda_version('g++',TorchVersion('13.3.0'))
except BaseException as error:compiler_error=f'{type(error).__name__}: {error}'
else:compiler_error=None
c10=ROOT/'.venvs/solarwm-h3/lib/python3.10/site-packages/torch/lib/libc10.so'
required_symbol='_ZN3c105ErrorC2ENS_14SourceLocationENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE'
provided_symbol='_ZN3c105ErrorC2ENS_14SourceLocationESs'
provided=subprocess.check_output(['nm','-D',str(c10)],text=True)
required=subprocess.check_output(['nm','-D',str(new)],text=True)
result={
 'status':'BLOCKED_ABI_AND_CUDA_TOOLKIT_MISMATCH',
 'cuda_initialized_before':start_initialized,'cuda_initialized_after':torch.cuda.is_initialized(),
 'torch_version':torch.__version__,'torch_cuda':torch.version.cuda,
 'torch_cxx11_abi':bool(torch._C._GLIBCXX_USE_CXX11_ABI),
 'python_version':sys.version.split()[0],
 'flash_attn_distribution':importlib.metadata.version('flash-attn'),
 'overlay_path':str(OVERLAY.resolve()),'native_extension_import_path':native_path,
 'native_import_error':import_error,
 'diffusers_flash_requirement_error':backend_error,
 'diffusers_import_warning':backend_stderr.getvalue().strip(),
 'official_wheel':wheel,
 'installed_overlay_binary_sha256':new_sha,
 'shared_env_binary_sha256':old_sha,
 'same_failing_binary':old_sha==new_sha,
 'required_new_abi_symbol_present_in_extension':required_symbol in required,
 'required_new_abi_symbol_exported_by_torch_libc10':required_symbol in provided,
 'old_abi_symbol_exported_by_torch_libc10':provided_symbol in provided,
 'existing_nvcc_path':'/usr/local/cuda-13.0/bin/nvcc',
 'existing_nvcc_version':'13.0.88',
 'torch_cpp_extension_cuda_check_error':compiler_error,
 'source_compile_attempted':False,
 'gpu_initialized':False,'gpu_attempt_started':False,
 'decision':'do not compile with CUDA13 against torch cu12.4; do not start GPU without CPU import/backend PASS',
}
assert not result['cuda_initialized_after']
assert result['native_import_error'] and result['diffusers_flash_requirement_error']
assert result['same_failing_binary'] and result['required_new_abi_symbol_present_in_extension']
assert not result['required_new_abi_symbol_exported_by_torch_libc10']
assert result['old_abi_symbol_exported_by_torch_libc10'] and result['torch_cpp_extension_cuda_check_error']
(HERE/'ENV_PREFLIGHT.json').write_text(json.dumps(result,indent=2)+'\n')
print(result['status'])
