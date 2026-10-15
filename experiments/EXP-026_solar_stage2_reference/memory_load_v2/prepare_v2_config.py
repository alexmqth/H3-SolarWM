"""Freeze the opt-in two-case config while preserving official Stage2 SP4 semantics."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOLAR = ROOT / 'SolarWM'
OFFICIAL = SOLAR / 'configs/examples/minimax_h3/infer-stage2-158f-sp4.yaml'
DATA = ROOT / 'data/SolarWM-Data/releases-v1'
MODEL = ROOT / 'models/SolarWM/SolarWM-h3-33B-base'
STAGE2 = ROOT / 'models/SolarWM/SolarWM-h3-33B-sgf-stage2-158f-fix'
SUPPORT = DATA / 'latent-wds/minimax-h3-158f-768p-nomind-v1/support'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

original = yaml.safe_load(OFFICIAL.read_text())
candidate = yaml.safe_load(OFFICIAL.read_text())
changes = {
    'model.checkpoint_path': str(MODEL),
    'model.cpu_sharded_load': True,
    'data.index_root': str(DATA),
    'data.transport.root': str(DATA),
    'data.silence_latents_path': str(SUPPORT / 'h3_silence_153_158_170.safetensors'),
    'data.encoder_contract_path': str(SUPPORT / 'encoder_contract.json'),
    'checkpoint.resume_from': str(STAGE2),
    'validation.sample_count': 2,
    'runtime.output_dir': str(ROOT / 'outputs/EXP-026_solar_stage2_reference_v2'),
}
diff=[]
for dotted, value in changes.items():
    lhs, rhs = original, candidate
    parts = dotted.split('.')
    for part in parts[:-1]:
        lhs, rhs = lhs[part], rhs[part]
    previous = lhs.get(parts[-1])
    rhs[parts[-1]] = value
    if previous != value:
        diff.append({'path': dotted, 'official': previous, 'candidate': value})
assert candidate['distributed']['sequence_parallel_size'] == 4
assert candidate['distributed']['world_size'] == 8
assert candidate['validation']['balance_by_dataset'] is True
out = HERE / 'candidate_two_case.yaml'
out.write_text(yaml.safe_dump(candidate, sort_keys=False, allow_unicode=True))
sys.path.insert(0, str(SOLAR / 'src'))
from solarwm.config.loader import load_config
from solarwm.backends.minimax_h3.config import validate_h3_config
assert load_config(out).values == candidate
contract = validate_h3_config(candidate)
assert contract.sequence_parallel_size == 4
(HERE / 'CONFIG_DIFF.json').write_text(json.dumps({
    'official_path': str(OFFICIAL.relative_to(ROOT)),
    'official_sha256': sha(OFFICIAL),
    'candidate_path': str(out.relative_to(ROOT)),
    'candidate_sha256': sha(out),
    'h3_backend_contract_pass': True,
    'changes': diff,
    'warning': 'two authentic balanced samples remain inaccessible; config is not a reader or GPU pass',
}, indent=2) + '\n')
print('V2_CONFIG_AND_H3_BACKEND_PASS_INPUT_BLOCKED')
