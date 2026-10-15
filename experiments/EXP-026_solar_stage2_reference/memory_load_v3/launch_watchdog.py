"""Fail-closed one-attempt launcher and independent four-L40 watchdog."""
from __future__ import annotations
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo
import psutil

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
AUTH=ROOT/'submission/experiments/EXP-026_solar_stage2_reference/AUTHORIZATION_v3.json'
GPUS=(0,1,6,7)
HKT=ZoneInfo('Asia/Hong_Kong')

def meminfo(name: str) -> int:
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith(name+':'):
            return int(line.split()[1])*1024
    raise RuntimeError(f'missing /proc/meminfo {name}')

def gpu_rows():
    out=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total',
        '--format=csv,noheader,nounits'],text=True)
    rows={}
    for line in out.strip().splitlines():
        fields=[x.strip() for x in line.split(',')]
        index=int(fields[0]); rows[index]={
            'index':index,'uuid':fields[1],
            'used_mib':int(fields[2]),'total_mib':int(fields[3]),
        }
    return rows

def compute_apps():
    out=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
        '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout
    apps=[]
    for line in out.strip().splitlines():
        if not line.strip():continue
        fields=[x.strip() for x in line.split(',')]
        apps.append({'uuid':fields[0],'pid':int(fields[1]),
                     'used_mib':int(fields[2]) if fields[2].isdigit() else None})
    return apps

def group_rss(root_pid:int):
    try:
        parent=psutil.Process(root_pid)
        processes=[parent,*parent.children(recursive=True)]
    except psutil.NoSuchProcess:
        return 0,[]
    rows=[]
    for proc in processes:
        try:
            rows.append({'pid':proc.pid,'rss':proc.memory_info().rss,'name':proc.name()})
        except (psutil.NoSuchProcess,psutil.AccessDenied):
            pass
    return sum(row['rss'] for row in rows),rows

def stop_group(pgid:int, sig):
    try:os.killpg(pgid,sig)
    except ProcessLookupError:pass

def main():
    auth=json.loads(AUTH.read_text())
    if auth.get('status')!='authorized' or auth.get('plan_version')!=3 or not auth.get('user_approved'):
        raise RuntimeError('v3 user/Judge authorization missing')
    if auth.get('max_gpus')!=4 or auth.get('wall_seconds')!=600 or auth.get('attempts')!=1:
        raise RuntimeError('v3 resource authorization differs')
    config=HERE/'candidate_one_case.yaml'
    patched=HERE/'patched_src/src'
    required=[config,HERE/'CPU_TEST.json',HERE/'CPU_GROUP_TEST.json',HERE/'PATCHED_SOURCE.json',
              HERE/'cpu_sharded_load.patch',patched/'solarwm/backends/minimax_h3/runtime.py']
    if any(not item.is_file() for item in required):
        raise RuntimeError('CPU gate or patched source missing')
    if json.loads((HERE/'CPU_TEST.json').read_text())['status']!='CPU_BOUNDARY_PASS_GPU_NOT_TESTED':
        raise RuntimeError('CPU boundary gate failed')
    if json.loads((HERE/'CPU_GROUP_TEST.json').read_text())['status']!='GLOO_GROUP_PASS_NO_FSDP_MODEL':
        raise RuntimeError('CPU shard group gate failed')
    source_receipt=json.loads((HERE/'SOURCE_RECEIPT.json').read_text())
    for name,receipt in source_receipt.items():
        if hashlib.sha256((ROOT/'SolarWM'/name).read_bytes()).hexdigest()!=receipt['base_sha256']:
            raise RuntimeError('official source changed since patch: '+name)
        if hashlib.sha256((HERE/'patched_src'/name).read_bytes()).hexdigest()!=receipt['modified_sha256']:
            raise RuntimeError('patched source digest differs: '+name)
    now=dt.datetime.now(HKT)
    if now.time()>=dt.time(8,48) or now.time()<dt.time(0,0):
        raise RuntimeError('outside approved four-GPU launch time')
    initial_mem=meminfo('MemAvailable')
    initial_swap=meminfo('SwapFree')
    if initial_mem<400*2**30:
        raise RuntimeError(f'MemAvailable below 400GiB: {initial_mem/2**30:.2f}')
    before=gpu_rows()
    target=[before[index] for index in GPUS]
    if any(row['used_mib']>=1024 for row in target):
        raise RuntimeError('preferred GPU occupied >=1GiB')
    busy={row['uuid'] for row in compute_apps()}
    if any(row['uuid'] in busy for row in target):
        raise RuntimeError('preferred GPU has another compute process')
    attempt=HERE/'gpu_attempt_001'
    attempt.mkdir(exist_ok=True)
    lock=attempt/'ATTEMPT_LOCK.json'
    lock_data={'plan_version':3,'locked_unix':time.time(),'launcher_pid':os.getpid()}
    fd=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
    with os.fdopen(fd,'w') as stream:json.dump(lock_data,stream,indent=2)
    env=os.environ.copy()
    env.update({
        'CUDA_VISIBLE_DEVICES':'0,1,6,7',
        'PYTHONPATH':str(patched),
        'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
        'RAYON_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false',
        'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','DIFFUSERS_OFFLINE':'1',
        'HF_HUB_DISABLE_TELEMETRY':'1',
    })
    command=[str(ROOT/'.venvs/solarwm-h3/bin/torchrun'),'--standalone','--nproc-per-node=4',
             str(HERE/'load_only.py'),'--config',str(config),'--log-dir',str(attempt)]
    stdout=(attempt/'torchrun.stdout.log').open('w',buffering=1)
    stderr=(attempt/'torchrun.stderr.log').open('w',buffering=1)
    start=time.time()
    process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr,
                             start_new_session=True)
    launch={
        'status':'LAUNCHED','command':command,'cwd':str(ROOT),'pid':process.pid,
        'process_group_id':process.pid,'start_unix':start,
        'start_hkt':dt.datetime.fromtimestamp(start,HKT).isoformat(),
        'physical_gpus':target,'initial_mem_available_bytes':initial_mem,
        'initial_swap_free_bytes':initial_swap,
        'cuda_visible_devices':env['CUDA_VISIBLE_DEVICES'],
        'patch_sha256':hashlib.sha256((HERE/'cpu_sharded_load.patch').read_bytes()).hexdigest(),
        'checkpoint':str(ROOT/'models/SolarWM/SolarWM-h3-33B-sgf-stage2-158f-fix'),
        'ema_sha256_prior_verified':'92e8f4b07aa92b0453a2f63954db1421caa3b8cac83ece706535db1a91493a05',
    }
    (attempt/'GPU_LAUNCH.json').write_text(json.dumps(launch,indent=2)+'\n')
    print('GPU_LAUNCHED',process.pid,','.join(str(x) for x in GPUS),flush=True)
    reason='process_exit'
    peak_rss=0
    min_mem=initial_mem
    min_swap=initial_swap
    max_gpu_mib={str(index):before[index]['used_mib'] for index in GPUS}
    try:
        with (attempt/'RESOURCE_SAMPLES.jsonl').open('w',buffering=1) as samples:
            while process.poll() is None:
                elapsed=time.time()-start
                current_mem=meminfo('MemAvailable')
                current_swap=meminfo('SwapFree')
                rss,procs=group_rss(process.pid)
                gpu=gpu_rows()
                peak_rss=max(peak_rss,rss)
                min_mem=min(min_mem,current_mem)
                min_swap=min(min_swap,current_swap)
                for index in GPUS:max_gpu_mib[str(index)]=max(max_gpu_mib[str(index)],gpu[index]['used_mib'])
                record={'elapsed_sec':elapsed,'mem_available_bytes':current_mem,
                        'swap_free_bytes':current_swap,'task_rss_bytes':rss,
                        'processes':procs,'target_gpu_used_mib':{str(i):gpu[i]['used_mib'] for i in GPUS}}
                samples.write(json.dumps(record)+'\n')
                if current_mem<128*2**30:reason='host_mem_below_128GiB'
                elif rss>400*2**30:reason='task_rss_above_400GiB'
                elif initial_swap-current_swap>1*2**30:reason='swap_consumption_above_1GiB'
                elif any(gpu[i]['used_mib']>40*1024 for i in GPUS):reason='gpu_used_above_40GiB'
                elif elapsed>=580:reason='watchdog_580_seconds'
                elif dt.datetime.now(HKT).time()>=dt.time(8,59):reason='hard_finish_guard'
                if reason!='process_exit':
                    stop_group(process.pid,signal.SIGTERM)
                    break
                time.sleep(2)
        if reason!='process_exit':
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                stop_group(process.pid,signal.SIGKILL)
        returncode=process.wait(timeout=10)
    except BaseException as exc:
        reason='watchdog_exception:'+repr(exc)
        stop_group(process.pid,signal.SIGTERM)
        try:process.wait(timeout=10)
        except subprocess.TimeoutExpired:stop_group(process.pid,signal.SIGKILL)
        returncode=process.poll()
    finally:
        stdout.close();stderr.close()
        end=time.time()
        ledger={
            'reason':reason,'torchrun_exit_code':returncode if 'returncode' in locals() else process.poll(),
            'start_unix':start,'end_unix':end,'elapsed_sec':end-start,
            'gpu_count':4,'charged_gpu_seconds':4*(end-start),
            'peak_task_rss_bytes_sampled':peak_rss,
            'min_host_mem_available_bytes_sampled':min_mem,
            'min_swap_free_bytes_sampled':min_swap,
            'max_physical_gpu_used_mib_sampled':max_gpu_mib,
            'forward_count':0,'backward_count':0,'codec_count':0,'updates':0,
        }
        (attempt/'GPU_LEDGER.json').write_text(json.dumps(ledger,indent=2)+'\n')
        print('GPU_ATTEMPT_FINISHED',reason,ledger['torchrun_exit_code'],f'{ledger["elapsed_sec"]:.1f}s',flush=True)
    return 0 if reason=='process_exit' and returncode==0 else 1

if __name__=='__main__':
    raise SystemExit(main())
