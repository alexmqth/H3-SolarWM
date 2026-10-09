import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from causal.stage1_protocol import validate_stage1_protocol


def fixture(anyflow):
    config = dict(adapter_scope='all_qkvo_ffn', action_adapter='initial_action.pt',
        anchor_mode='rgb', chunk_frames=5, history_chunks=5, action_prefix_mode='causal',
        action_feedback=True, flow_shift=2.22, precision_profile='h3_fp32')
    visual = dict(objective='TF-AnyFlow v1.5' if anyflow else 'TF-FM control',
        optimizer_step=16, config=config, precision={'profile':'h3_fp32'})
    inference = dict(config, anchor_mode='dynamic_last_frame_rgb_dual',
                     modes=['cached'], causal_adapter_scope='all')
    return [visual, copy.deepcopy(visual), copy.deepcopy(visual),
            copy.deepcopy(visual) if anyflow else None, inference]


@pytest.mark.parametrize('anyflow', [False, True])
def test_matching_bank_and_objective_accepted(anyflow):
    validate_stage1_protocol(*fixture(anyflow))


@pytest.mark.parametrize('anyflow', [False, True])
@pytest.mark.parametrize('mutation', ['no_bank', 'no_action', 'wrong_time', 'other_step',
    'other_training', 'other_objective', 'wrong_precision', 'other_anchor', 'no_feedback'])
def test_cannot_silently_omit_bank_or_mix_protocol(anyflow, mutation):
    args = fixture(anyflow)
    if mutation == 'no_bank': args[1] = None
    if mutation == 'no_action': args[2] = None
    if mutation == 'wrong_time': args[3] = None if anyflow else copy.deepcopy(args[0])
    if mutation == 'other_step': args[1]['optimizer_step'] = 4
    if mutation == 'other_training': args[2]['config']['flow_shift'] = 12.
    if mutation == 'other_objective': args[1]['objective'] = 'unknown'
    if mutation == 'wrong_precision': args[1]['precision']['profile'] = 'legacy'
    if mutation == 'other_anchor': args[4]['anchor_mode'] = 'fixed'
    if mutation == 'no_feedback': args[4]['action_feedback'] = False
    with pytest.raises(ValueError):
        validate_stage1_protocol(*args)


def test_old_tail_scope_keeps_its_separate_validation_path():
    validate_stage1_protocol({}, None, None, None, {})
    args = fixture(False)
    args[0]['config']['adapter_scope'] = 'tail_qkv'
    with pytest.raises(ValueError):
        validate_stage1_protocol(*args)
