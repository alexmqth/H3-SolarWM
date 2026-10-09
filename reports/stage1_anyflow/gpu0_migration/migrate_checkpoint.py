"""Move only this experiment's processes after a complete step08 checkpoint."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
SOURCE = ROOT / 'outputs/2026-10-08-04/stage1_anyflow39_frozen_time'
OLD_EXTENSION = ROOT / 'outputs/2026-10-08-04/stage1_anyflow39_frozen_extend64'
sys.path.insert(0, str(SOURCE / 'runtime/code/causal'))
from training_state import load_training_state


def live(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


def verify_process(pid, expected):
    command = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
    if expected not in command or not live(pid):
        raise RuntimeError(f'PID {pid} does not match the experiment process')


def main():
    if (OUT / 'migration_receipt.json').exists():
        raise SystemExit('Migration already recorded')
    checkpoint = SOURCE / 'train_anyflow/step_08'
    while not (checkpoint / 'trainer_state.pt').exists():
        queue = json.loads((SOURCE / 'run.json').read_text())
        if queue['status'] != 'running' or not live(queue['active']['pid']):
            raise RuntimeError('Training stopped before the migration checkpoint')
        time.sleep(5)
    queue = json.loads((SOURCE / 'run.json').read_text())
    extension = json.loads((OLD_EXTENSION / 'run.json').read_text())
    if queue['active']['label'] != 'frozen_time_anyflow16' or extension['active'] is not None:
        raise RuntimeError('Unexpected active process; refusing migration')
    trainer_pid, controller_pid, extension_pid = queue['active']['pid'], queue['pid'], extension['pid']
    verify_process(trainer_pid, str(SOURCE / 'runtime/code/causal/train_stage1_anyflow.py'))
    verify_process(controller_pid, 'stage1_anyflow39_frozen_time/run_frozen_time.py')
    verify_process(extension_pid, 'stage1_anyflow39_frozen_extend64/run_extension.py')
    source_training = json.loads((SOURCE / 'train_anyflow/training.json').read_text())
    config = dict(source_training['config'])
    config.update(out_dir=str(OUT / 'train_anyflow'), resume_from=str(checkpoint))
    saved = load_training_state(checkpoint, config)
    if saved['optimizer_step'] != 8 or saved['updates'] != source_training['updates'][:8]:
        raise RuntimeError('Checkpoint integrity/history mismatch')
    # Stop the exact owned trainer first so it cannot start another update;
    # stop the waiting controllers before terminating the stopped child.
    os.kill(trainer_pid, signal.SIGSTOP)
    for pid in (extension_pid, controller_pid, trainer_pid):
        os.kill(pid, signal.SIGTERM)
    try:
        os.kill(trainer_pid, signal.SIGCONT)
    except ProcessLookupError:
        pass
    stopped = [trainer_pid, controller_pid, extension_pid]
    deadline = time.monotonic() + 30
    while any(live(pid) for pid in stopped):
        if time.monotonic() > deadline:
            raise RuntimeError('Owned processes have not exited; no GPU0 job launched')
        time.sleep(.25)
    receipt = dict(at=datetime.now().astimezone().isoformat(), gpu_before=2, gpu_after=0,
        stopped_pids=stopped, checkpoint_path=str(checkpoint), checkpoint_step=8,
        state_sha256=hashlib.sha256((checkpoint / 'trainer_state.pt').read_bytes()).hexdigest(),
        saved_optimizer_and_rng=True, source_update_history_matches=True,
        vram_reserve_gib_before=20, vram_reserve_gib_after=20,
        target_total_updates=16, new_run=str(OUT))
    for directory in (SOURCE, OLD_EXTENSION):
        file = directory / 'run.json'
        file.replace(directory / 'run.before_gpu0_migration.json')
        state = json.loads((directory / 'run.before_gpu0_migration.json').read_text())
        state.update(status='migrated_to_gpu0', active=None, migration=receipt)
        file.write_text(json.dumps(state, indent=2) + '\n')
    file = SOURCE / 'train_anyflow/training.json'
    file.replace(file.with_name('training.before_gpu0_migration.json'))
    state = json.loads(file.with_name('training.before_gpu0_migration.json').read_text())
    state.update(status='interrupted_for_gpu0_migration', migration=receipt)
    file.write_text(json.dumps(state, indent=2) + '\n')
    (OUT / 'migration_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
