"""One bounded official FlashAttention source build in the isolated CUDA 12.4 tree."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import psutil
import resource
import shutil
import signal
import subprocess
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
TOOL=ROOT/'.toolchains/exp026_v5'
CUDA=TOOL/'cuda-12.4'
SRC=TOOL/'src/flash_attn-2.8.3'


def descendants_rss(pid):
    try:
        parent=psutil.Process(pid)
        group=[parent,*parent.children(recursive=True)]
    except psutil.NoSuchProcess:return 0,[]
    rows=[]
    for proc in group:
        try:rows.append({'pid':proc.pid,'rss':proc.memory_info().rss,'name':proc.name()})
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    return sum(x['rss'] for x in rows),rows


def stop_group(pid,sig):
    try:os.killpg(pid,sig)
    except ProcessLookupError:pass


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--attempt',type=int,choices=(1,2),required=True)
    args=ap.parse_args()
    if not (CUDA/'bin/nvcc').is_file() or not (SRC/'setup.py').is_file():
        raise RuntimeError('CUDA 12.4 compiler or source missing')
    if args.attempt==2 and not (HERE/'BUILD_01_LEDGER.json').is_file():
        raise RuntimeError('second build requires first attempt evidence')
    prior_elapsed=(json.loads((HERE/'BUILD_01_LEDGER.json').read_text())['elapsed_sec']
                   if args.attempt==2 else 0.0)
    max_this_build=min(3300.0,3600.0-prior_elapsed-30.0)
    if max_this_build<=0:
        raise RuntimeError('cumulative 60-minute compile budget exhausted')
    lock=HERE/f'BUILD_{args.attempt:02d}_LOCK.json'
    fd=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
    with os.fdopen(fd,'w') as stream:json.dump({'created_unix':time.time(),'pid':os.getpid()},stream)
    env=os.environ.copy()
    env.update({
        'CUDA_HOME':str(CUDA),'CUDA_PATH':str(CUDA),
        'PATH':str(CUDA/'bin')+os.pathsep+str(ROOT/'.venvs/solarwm-h3/bin')+os.pathsep+env.get('PATH',''),
        'LD_LIBRARY_PATH':str(CUDA/'lib64')+os.pathsep+str(ROOT/'.venvs/solarwm-h3/lib/python3.10/site-packages/torch/lib')+os.pathsep+env.get('LD_LIBRARY_PATH',''),
        'LIBRARY_PATH':str(CUDA/'lib64')+os.pathsep+env.get('LIBRARY_PATH',''),
        'MAX_JOBS':'4','NVCC_THREADS':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
        'FLASH_ATTN_CUDA_ARCHS':'80','TORCH_CUDA_ARCH_LIST':'8.0',
        'FLASH_ATTENTION_FORCE_BUILD':'TRUE','PIP_NO_INDEX':'1',
        'FLASH_ATTENTION_FORCE_CXX11_ABI':'FALSE',
        'PIP_DISABLE_PIP_VERSION_CHECK':'1','PIP_NO_CACHE_DIR':'1',
        'CXX':'g++','CC':'gcc',
    })
    cmd=[str(ROOT/'.venvs/solarwm-h3/bin/python'),'-m','pip','wheel',
         '--no-build-isolation','--no-deps','--no-index','--no-cache-dir',
         '--wheel-dir',str(TOOL/'build_wheels'),'.']
    logfile=HERE/f'BUILD_{args.attempt:02d}.log'
    started=time.time()
    with logfile.open('w',buffering=1) as out:
        out.write('COMMAND '+json.dumps(cmd)+'\n')
        out.write('ENV '+json.dumps({key:env[key] for key in (
            'CUDA_HOME','MAX_JOBS','NVCC_THREADS','FLASH_ATTN_CUDA_ARCHS',
            'TORCH_CUDA_ARCH_LIST','FLASH_ATTENTION_FORCE_BUILD','CXX')})+'\n')
        process=subprocess.Popen(cmd,cwd=SRC,env=env,stdout=out,stderr=subprocess.STDOUT,
                                 start_new_session=True)
        print('BUILD_STARTED',args.attempt,process.pid,str(logfile),flush=True)
        reason='process_exit';peak_rss=0;max_disk=0
        with (HERE/f'BUILD_{args.attempt:02d}_RESOURCE.jsonl').open('w',buffering=1) as metrics:
            while process.poll() is None:
                elapsed=time.time()-started
                rss,procs=descendants_rss(process.pid)
                peak_rss=max(peak_rss,rss)
                if int(elapsed)%30<5:
                    disk=sum(f.stat().st_size for f in TOOL.rglob('*') if f.is_file())
                    max_disk=max(max_disk,disk)
                else:disk=max_disk
                metrics.write(json.dumps({'elapsed_sec':elapsed,'rss_bytes':rss,
                                          'peak_rss_bytes':peak_rss,'new_toolchain_bytes':disk,
                                          'processes':procs})+'\n')
                if rss>48*2**30:reason='RSS_OVER_48GiB'
                elif disk>20*2**30:reason='DISK_OVER_20GiB'
                elif shutil.disk_usage(ROOT).free<200*2**30:reason='DISK_FREE_BELOW_200GiB'
                elif elapsed>max_this_build:reason='CUMULATIVE_BUILD_TIME_LIMIT'
                if reason!='process_exit':
                    stop_group(process.pid,signal.SIGTERM)
                    break
                time.sleep(5)
        if reason!='process_exit':
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:stop_group(process.pid,signal.SIGKILL)
        code=process.wait(timeout=15)
    finished=time.time()
    ledger={'attempt':args.attempt,'pid':process.pid,'start_unix':started,'end_unix':finished,
            'elapsed_sec':finished-started,'exit_code':code,'reason':reason,
            'prior_compile_elapsed_sec':prior_elapsed,'max_this_build_seconds':max_this_build,
            'peak_aggregate_rss_bytes_sampled':peak_rss,
            'max_new_toolchain_bytes_sampled':max_disk,
            'compile_jobs':4,'nvcc_threads':1,'gpu_model_forward_count':0}
    (HERE/f'BUILD_{args.attempt:02d}_LEDGER.json').write_text(json.dumps(ledger,indent=2)+'\n')
    print('BUILD_FINISHED',args.attempt,code,reason,f'{finished-started:.1f}s',flush=True)
    return 0 if code==0 and reason=='process_exit' else 1

if __name__=='__main__':raise SystemExit(main())
