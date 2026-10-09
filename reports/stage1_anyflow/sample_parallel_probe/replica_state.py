"""Actual end-of-run model/Adam/RNG equality between all four replicas."""
import hashlib
import torch
import torch.distributed as dist


def audit_replicas(parameters, optimizer, generator, *, device, smoke):
    def tensor_hash(values):
        h=hashlib.sha256()
        for p in values:
            h.update(str((tuple(p.shape),p.dtype)).encode())
            h.update(p.detach().float().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    moments=[]
    for p in parameters:
        for key in ('step','exp_avg','exp_avg_sq'):
            moments.append(optimizer.state[p][key])
    row=dict(rank=dist.get_rank(),parameter_sha256=tensor_hash(parameters),
        optimizer_sha256=tensor_hash(moments),logical_rng_sha256=tensor_hash([generator.get_state()]),
        cpu_rng_sha256=tensor_hash([torch.get_rng_state()]),
        cuda_rng_sha256=None if smoke else tensor_hash([torch.cuda.get_rng_state(device)]),
        allocated_peak_MiB=0. if smoke else torch.cuda.max_memory_allocated(device)/2**20)
    rows=[None]*4
    dist.all_gather_object(rows,row)
    keys=('parameter_sha256','optimizer_sha256','logical_rng_sha256','cpu_rng_sha256','cuda_rng_sha256')
    equality={key:len({r[key] for r in rows})==1 for key in keys}
    if not all(equality.values()):
        raise RuntimeError(f'Replica model/Adam/RNG divergence: {equality}')
    return dict(equality=equality,rows=rows)
