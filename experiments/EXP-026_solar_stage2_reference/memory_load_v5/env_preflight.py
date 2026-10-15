"""CPU-only, fail-closed gate for the isolated EXP-026/v5 FlashAttention build."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OVERLAY = HERE / "overlay"
OUTPUT = HERE / "ENV_PREFLIGHT.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    receipt: dict[str, object] = {
        "status": "FAIL_CPU_GATE",
        "python_executable": sys.executable,
        "overlay_path": str(OVERLAY),
        "cuda_initialized_before": None,
        "cuda_initialized_after": None,
    }
    torch = None
    try:
        if not OVERLAY.is_dir() or str(OVERLAY) != sys.path[0]:
            # PYTHONPATH is supplied by the caller; do not silently fall back
            # to the old, ABI-incompatible venv installation.
            if str(OVERLAY) not in sys.path:
                raise RuntimeError("isolated overlay absent from sys.path")
        import torch as torch_module
        torch = torch_module
        receipt["cuda_initialized_before"] = torch.cuda.is_initialized()
        if receipt["cuda_initialized_before"]:
            raise RuntimeError("CUDA initialized before CPU preflight")
        receipt["torch_version"] = torch.__version__
        receipt["torch_cuda"] = torch.version.cuda
        receipt["torch_cxx11_abi"] = bool(torch._C._GLIBCXX_USE_CXX11_ABI)
        if (torch.__version__, torch.version.cuda, receipt["torch_cxx11_abi"]) != (
            "2.6.0+cu124", "12.4", False
        ):
            raise RuntimeError("torch version/CUDA/CXX11 ABI contract differs")

        import flash_attn
        import flash_attn_2_cuda
        from diffusers.models.attention_dispatch import (
            AttentionBackendName,
            _check_attention_backend_requirements,
        )

        _check_attention_backend_requirements(AttentionBackendName.FLASH)
        package_path = Path(flash_attn.__file__).resolve()
        extension_path = Path(flash_attn_2_cuda.__file__).resolve()
        if not package_path.is_relative_to(OVERLAY.resolve()):
            raise RuntimeError(f"flash_attn came from non-overlay path: {package_path}")
        if not extension_path.is_relative_to(OVERLAY.resolve()):
            raise RuntimeError(f"native extension came from non-overlay path: {extension_path}")
        receipt.update({
            "flash_attn_distribution": importlib.metadata.version("flash-attn"),
            "flash_attn_package_path": str(package_path),
            "native_extension_import_path": str(extension_path),
            "native_extension_sha256": sha256(extension_path),
            "diffusers_flash_requirement": "PASS",
            "native_extension_dependencies": subprocess.run(
                ["ldd", str(extension_path)], capture_output=True, text=True, check=True,
            ).stdout,
        })
        if receipt["flash_attn_distribution"] != "2.8.3":
            raise RuntimeError("flash-attn version differs")
        receipt["cuda_initialized_after"] = torch.cuda.is_initialized()
        if receipt["cuda_initialized_after"]:
            raise RuntimeError("CPU preflight initialized CUDA")
        receipt["status"] = "PASS_FLASH_BACKEND_CPU_NO_CUDA"
    except BaseException as exc:
        receipt["error"] = repr(exc)
        receipt["traceback"] = traceback.format_exc()
        if torch is not None:
            receipt["cuda_initialized_after"] = torch.cuda.is_initialized()
    OUTPUT.write_text(json.dumps(receipt, indent=2) + "\n")
    print(receipt["status"], flush=True)
    return 0 if receipt["status"] == "PASS_FLASH_BACKEND_CPU_NO_CUDA" else 1


if __name__ == "__main__":
    raise SystemExit(main())
