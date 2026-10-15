"""Generate an exact-context patch without modifying the official checkout."""
from __future__ import annotations

import difflib
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOLAR = ROOT / 'SolarWM'
FILES = [
    'src/solarwm/backends/minimax_h3/optional.py',
    'src/solarwm/backends/minimax_h3/runtime.py',
]


def replace_once(source: str, old: str, new: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f'exact patch anchor occurs {count} times: {old[:75]!r}')
    return source.replace(old, new, 1)


def optional_patch(s: str) -> str:
    s = replace_once(s,
        'def load_transformer(model_cfg: Mapping[str, Any], *, device: Any) -> H3RuntimeModules:',
        'def load_transformer(\n    model_cfg: Mapping[str, Any], *, device: Any, defer_cuda_for_fsdp: bool = False\n) -> H3RuntimeModules:')
    s = replace_once(s,
        '    if model_cfg.get("transformer_device_map") is None:\n        transformer = transformer.to(device)\n',
        '    if defer_cuda_for_fsdp:\n'
        '        if model_cfg.get("transformer_device_map") is not None:\n'
        '            raise BackendContractError("CPU-sharded load requires transformer_device_map=null")\n'
        '        if any(parameter.device.type != "cpu" for parameter in transformer.parameters()):\n'
        '            raise BackendContractError("CPU-sharded load found a non-CPU base parameter")\n'
        '    elif model_cfg.get("transformer_device_map") is None:\n'
        '        transformer = transformer.to(device)\n')
    return s



def runtime_patch(s: str) -> str:
    marker = 'def run_inference(config: Mapping[str, Any]) -> int:\n'
    helper = '''def _prepare_h3_inference_model(
    config: Mapping[str, Any], *, device: Any, topology: Topology, stage: str,
    event_hook=None,
):
    """Keep the opt-in CPU -> cross-SP FSDP load order explicit and testable."""
    from .fsdp import wrap_h3_fsdp
    from .lora import inject_h3_lora
    import torch.distributed as dist

    model_cfg = config["model"]
    cpu_sharded = model_cfg.get("cpu_sharded_load", False) is True
    if cpu_sharded and (
        stage != "stage2" or topology.raw_world_size != 4
        or topology.sp_size != 4 or topology.dp_world_size != 1
        or int(config["validation"]["sample_count"]) != 1
    ):
        raise BackendContractError("CPU-sharded fixed Stage2 requires SP4, DP1 and one case")
    if cpu_sharded and (not dist.is_initialized() or dist.get_world_size() != 4):
        raise BackendContractError("CPU-sharded inference requires an initialized four-rank group")
    modules = load_transformer(model_cfg, device=device, defer_cuda_for_fsdp=cpu_sharded)
    if event_hook is not None:
        event_hook("base_loaded")
    if stage == "stage1":
        modules.transformer.enable_anyflow(float(config["train"]["anyflow_gate"]))
        modules.fp32_fsdp_units = (
            *modules.fp32_fsdp_units,
            modules.transformer.delta_embedding.linear_1,
            modules.transformer.delta_embedding.linear_2,
        )
    modules.transformer.eval().requires_grad_(False)
    base_model = _base_model_load_receipt(model_cfg, modules.transformer)
    model, lora = inject_h3_lora(
        modules.transformer, model_cfg["adapter"], base_identity=base_model,
    )
    if event_hook is not None:
        event_hook("lora_injected")
    if cpu_sharded:
        # FSDP ignores replicated LoRA tensors; move only these to local CUDA.
        # The frozen base stays on CPU until each FSDP unit is flattened/sharded.
        for parameter in lora.parameters:
            parameter.data = parameter.data.to(device=device)
        ignored_ids = {id(item) for item in lora.parameters}
        if any(parameter.device.type != "cpu" for parameter in modules.transformer.parameters()
               if id(parameter) not in ignored_ids):
            raise BackendContractError("frozen base moved to CUDA before FSDP")
    model = wrap_h3_fsdp(
        model,
        local_rank=topology.local_rank,
        transformer_block_cls=modules.transformer_block_cls,
        fp32_units=modules.fp32_fsdp_units,
        ignored_parameters=lora.parameters,
        activation_checkpointing=False,
        frozen_base_shard_size=4 if cpu_sharded else None,
    )
    if event_hook is not None:
        event_hook("base_sharded")
    return model, lora, base_model


'''
    s = replace_once(s, marker, helper + marker)
    old = '''    modules = load_transformer(config["model"], device=device)
    if stage == "stage1":
        modules.transformer.enable_anyflow(float(config["train"]["anyflow_gate"]))
        modules.fp32_fsdp_units = (
            *modules.fp32_fsdp_units,
            modules.transformer.delta_embedding.linear_1,
            modules.transformer.delta_embedding.linear_2,
        )
    modules.transformer.eval().requires_grad_(False)
    base_model = _base_model_load_receipt(config["model"], modules.transformer)
    model, lora = inject_h3_lora(
        modules.transformer,
        config["model"]["adapter"],
        base_identity=base_model,
    )
    model = wrap_h3_fsdp(
        model,
        local_rank=topology.local_rank,
        transformer_block_cls=modules.transformer_block_cls,
        fp32_units=modules.fp32_fsdp_units,
        ignored_parameters=lora.parameters,
        activation_checkpointing=False,
    )
'''
    new = '''    model, lora, base_model = _prepare_h3_inference_model(
        config, device=device, topology=topology, stage=stage,
    )
'''
    s = replace_once(s, old, new)
    return s


TRANSFORMS = [optional_patch, runtime_patch]
patch_parts = []
receipt = {}
for name, transform in zip(FILES, TRANSFORMS):
    original = (SOLAR / name).read_text()
    modified = transform(original)
    receipt[name] = {
        'base_sha256': hashlib.sha256(original.encode()).hexdigest(),
        'modified_sha256': hashlib.sha256(modified.encode()).hexdigest(),
    }
    patch_parts.extend(difflib.unified_diff(
        original.splitlines(keepends=True), modified.splitlines(keepends=True),
        fromfile=f'a/{name}', tofile=f'b/{name}',
    ))
(HERE / 'cpu_sharded_load.patch').write_text(''.join(patch_parts))
(HERE / 'SOURCE_RECEIPT.json').write_text(json.dumps(receipt, indent=2) + '\n')
print('PATCH_GENERATED', len(patch_parts), 'lines')
