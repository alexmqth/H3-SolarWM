"""Apply the independent patch in a temp copy and test production boundaries on CPU."""
from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import tempfile
import types
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOLAR = ROOT / 'SolarWM'

with tempfile.TemporaryDirectory(prefix='exp026_v2_') as temp:
    copy = Path(temp)
    shutil.copytree(SOLAR / 'src/solarwm', copy / 'src/solarwm')
    apply = subprocess.run(
        ['patch', '-p1', '-i', str(HERE / 'cpu_sharded_load.patch')],
        cwd=copy, text=True, capture_output=True, check=True,
    )
    files = [
        copy / 'src/solarwm/backends/minimax_h3/optional.py',
        copy / 'src/solarwm/backends/minimax_h3/runtime.py',
    ]
    subprocess.run([sys.executable, '-m', 'py_compile', *(str(f) for f in files)], check=True)
    sys.path.insert(0, str(copy / 'src'))

    import torch
    import yaml
    from solarwm.backends.minimax_h3 import optional, runtime, fsdp, lora, weights
    from solarwm.backends.minimax_h3.config import validate_h3_config
    from solarwm.errors import BackendContractError, ConfigurationError

    original = yaml.safe_load((ROOT / 'submission/experiments/EXP-026_solar_stage2_reference/candidate_single_case.yaml').read_text())
    try:
        validate_h3_config(original)
    except ConfigurationError as error:
        assert 'SP4' in str(error)
    else:
        raise AssertionError('v1 SP8 fixed Stage2 config unexpectedly passed backend validation')
    candidate = yaml.safe_load((HERE / 'candidate_one_case.yaml').read_text())
    contract = validate_h3_config(candidate)
    assert contract.sequence_parallel_size == 4
    assert candidate['validation']['sample_count'] == 1
    assert candidate['model']['cpu_sharded_load'] is True

    class TinyH3(torch.nn.Module):
        cuda_moves = 0
        last_kwargs = None

        def __init__(self):
            super().__init__()
            self.proj_in = torch.nn.Linear(1, 1, dtype=torch.float32)
            self.audio_proj_in = torch.nn.Linear(1, 1, dtype=torch.float32)
            self.time_embedder = torch.nn.Module()
            self.time_embedder.linear_1 = torch.nn.Linear(1, 1, dtype=torch.float32)
            self.time_embedder.linear_2 = torch.nn.Linear(1, 1, dtype=torch.float32)
            self.proj_out = torch.nn.Linear(1, 1, dtype=torch.float32)
            self.audio_proj_out = torch.nn.Linear(1, 1, dtype=torch.float32)
            self.block = torch.nn.Linear(1, 1, dtype=torch.bfloat16)

        @classmethod
        def strict_from_pretrained(cls, path, **kwargs):
            cls.last_kwargs = (path, kwargs)
            return cls()

        def set_attention_backend(self, value):
            self.attention_backend = value

        def to(self, *args, **kwargs):
            target = args[0] if args else kwargs.get('device')
            if str(target).startswith('cuda'):
                type(self).cuda_moves += 1
                raise AssertionError('whole-model .to(cuda) was attempted')
            return super().to(*args, **kwargs)

    fake_model_module = types.ModuleType('solarwm.backends.minimax_h3.model')
    fake_model_module.MiniMaxH3TransformerBlock = TinyH3
    fake_model_module.SolarMiniMaxH3Transformer3DModel = TinyH3
    with mock.patch.dict(sys.modules, {fake_model_module.__name__: fake_model_module}), \
         mock.patch.object(optional, 'require_h3_runtime', return_value=(torch, None, None)):
        modules = optional.load_transformer(
            candidate['model'], device=torch.device('cuda:0'), defer_cuda_for_fsdp=True,
        )
        assert TinyH3.cuda_moves == 0
        assert all(p.device.type == 'cpu' for p in modules.transformer.parameters())
        assert modules.transformer.attention_backend == 'flash'
        assert TinyH3.last_kwargs[1]['device_map'] is None
        assert TinyH3.last_kwargs[1]['torch_dtype'] == torch.bfloat16
        assert len(modules.fp32_fsdp_units) == 6
        try:
            optional.load_transformer(candidate['model'], device=torch.device('cuda:0'))
        except AssertionError as error:
            assert 'whole-model' in str(error)
        else:
            raise AssertionError('fault injection did not catch legacy whole-model CUDA move')
        bad = dict(candidate['model'], transformer_device_map='auto')
        try:
            optional.load_transformer(bad, device=torch.device('cpu'), defer_cuda_for_fsdp=True)
        except BackendContractError:
            pass
        else:
            raise AssertionError('unsafe device_map passed deferred-load guard')

    events = []
    tiny = TinyH3()
    holder = {}
    def fake_load(model_cfg, *, device, defer_cuda_for_fsdp=False):
        events.append('load')
        assert defer_cuda_for_fsdp and device.type == 'cpu'
        return optional.H3RuntimeModules(transformer=tiny, transformer_block_cls=TinyH3,
                                         fp32_fsdp_units=(tiny.proj_in, tiny.audio_proj_in,
                                             tiny.time_embedder.linear_1, tiny.time_embedder.linear_2,
                                             tiny.proj_out, tiny.audio_proj_out))
    def fake_inject(model, adapter_cfg, *, base_identity):
        events.append('inject')
        assert all(p.device.type == 'cpu' for p in model.parameters())
        model.lora_A = torch.nn.Parameter(torch.tensor([0.0], dtype=torch.bfloat16))
        adapter = types.SimpleNamespace(
            parameters=(model.lora_A,),
            parameter_by_key=OrderedDict([('tiny.lora_A.weight', model.lora_A)]),
        )
        holder['lora'] = adapter
        return model, adapter
    def fake_wrap(model, **kwargs):
        events.append('wrap')
        assert kwargs['frozen_base_shard_size'] == 4
        assert kwargs['activation_checkpointing'] is False
        assert tuple(kwargs['ignored_parameters']) == holder['lora'].parameters
        assert all(p.device.type == 'cpu' for p in model.parameters())
        return model
    topology = types.SimpleNamespace(raw_world_size=4, sp_size=4, dp_world_size=1,
                                     local_rank=0)
    phase_events = []
    with mock.patch.object(runtime, 'load_transformer', side_effect=fake_load), \
         mock.patch.object(lora, 'inject_h3_lora', side_effect=fake_inject), \
         mock.patch.object(fsdp, 'wrap_h3_fsdp', side_effect=fake_wrap), \
         mock.patch.object(torch.distributed, 'is_initialized', return_value=True), \
         mock.patch.object(torch.distributed, 'get_world_size', return_value=4):
        prepared_model, adapter, receipt = runtime._prepare_h3_inference_model(
            candidate, device=torch.device('cpu'), topology=topology, stage='stage2',
            event_hook=phase_events.append,
        )
    assert events == ['load', 'inject', 'wrap']
    assert phase_events == ['base_loaded', 'lora_injected', 'base_sharded']
    assert prepared_model is tiny and receipt['strict_state_load']['missing_keys'] == 0
    assert holder['lora'] is adapter

    with tempfile.TemporaryDirectory(prefix='exp026_weights_') as weight_dir:
        checkpoint = Path(weight_dir)
        (checkpoint / 'COMPLETE.json').write_text('{}')
        (checkpoint / 'checkpoint-manifest.json').write_text(json.dumps({
            'step': 3900,
            'contract': {
                'family': 'minimax_h3', 'stage': 'stage2',
                'camera_translation_transform': 'logd4',
                'parameterization': 'peft-lora-r384-alpha384',
                'extras': {
                    'encoder_profile': {'pixel_frames': 158, 'height': 768, 'width': 1344},
                    'chunk_latents': 5, 'window_chunks': 6,
                    'target_latents': 47, 'rollout_latents': 50,
                    'student_rope_mode': 'sliding_local',
                },
            },
        }))
        torch.save({'schema': 'solarwm.minimax-h3-ema.v1',
                    'shadow': {'tiny.lora_A.weight': torch.tensor([0.75], dtype=torch.float32)}},
                   checkpoint / 'ema.pt')
        label = weights.load_initial_weights(
            {'path': str(checkpoint), 'stage': 'stage2', 'weight_source': 'ema'}, adapter,
        )
        assert label == 'stage2:ema:step=3900'
        assert adapter.parameters[0].item() == torch.tensor(0.75, dtype=torch.bfloat16).item()

    result = {
        'status': 'CPU_BOUNDARY_PASS_GPU_NOT_TESTED',
        'patch_apply': apply.stdout.strip(),
        'h3_backend_v1_sp8_rejected': True,
        'h3_backend_v3_sp4_one_case_pass': True,
        'deferred_load_no_whole_cuda_move': True,
        'legacy_cuda_move_fault_caught': True,
        'load_inject_wrap_order': events,
        'cross_sp_frozen_base_shard_size': 4,
        'tiny_ema_copy_value': adapter.parameters[0].item(),
        'real_fsdp_collective_tested': False,
        'cuda_initialized': torch.cuda.is_initialized(),
        'peak_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
    }
    assert not result['cuda_initialized']
    (HERE / 'CPU_TEST.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
