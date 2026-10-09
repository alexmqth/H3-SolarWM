"""Require one coherent full-scope checkpoint for either FM or AnyFlow."""


def validate_stage1_protocol(visual, bank, action, time, inference):
    config = visual.get('config', {})
    full = config.get('adapter_scope') == 'all_qkvo_ffn'
    if full != (bank is not None):
        raise ValueError('Checkpoint requires its matching full-scope Stage1 bank')
    if not full:
        return
    objective = visual.get('objective')
    if objective not in ('TF-FM control', 'TF-AnyFlow v1.5'):
        raise ValueError('Unsupported full-scope training objective')
    if (objective == 'TF-AnyFlow v1.5') != (time is not None):
        raise ValueError('Target-time module must match the training objective')
    if bool(config.get('action_adapter')) != (action is not None):
        raise ValueError('Full-scope inference requires the matching action checkpoint')
    for label, metadata in [('bank', bank), ('action', action), ('time', time)]:
        if metadata is None:
            continue
        for key in ('objective', 'optimizer_step', 'config', 'precision'):
            if metadata.get(key) != visual.get(key):
                raise ValueError(f'Stage1 {label}/{key} checkpoint mismatch')
    aliases = {'fixed': 'fixed', 'latent': 'dynamic_last_frame_dual',
               'rgb': 'dynamic_last_frame_rgb_dual'}
    expected = dict(anchor_mode=aliases[config['anchor_mode']],
        chunk_frames=config['chunk_frames'], history_chunks=config['history_chunks'],
        action_prefix_mode=config['action_prefix_mode'], action_feedback=config['action_feedback'],
        flow_shift=config['flow_shift'], precision_profile=config['precision_profile'],
        modes=['cached'], causal_adapter_scope='all')
    for key, value in expected.items():
        if inference.get(key) != value:
            raise ValueError(f'Stage1 {key} mismatch: trained={value}, inference={inference.get(key)}')
