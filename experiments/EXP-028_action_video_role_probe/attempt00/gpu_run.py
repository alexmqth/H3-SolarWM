"""One bounded EXP-028 pilot. Launch only through admit.py after CPU PASS."""
from __future__ import annotations

import json
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EXP23 = ROOT / "submission/experiments/EXP-023_v3_fm_small_data"
EXP25 = ROOT / "submission/experiments/EXP-025_v3_fm_full_coverage"
EXP19 = ROOT / "submission/experiments/EXP-019_v3_mixed_action_validation"
for path in (HERE, EXP23, ROOT / "submission/code", EXP19):
    sys.path.insert(0, str(path))

import torch
import torch.nn.functional as F
from common import sha, write
from role_probe import RoleAccumulator, ROLE_NAMES, diagnostic_forward, original_layout, v3_layout

CLIPS = ("43866101158a538d22c52ac5b2fa606b_src1245_rgb56",
         "b784d995827f53977a8067e5b9dc4b5e_src1635_rgb56")
SIGMAS = ((.25, 113), (.75, 127))
LIMIT = 2400


def rss_gib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


class Ledger:
    def __init__(self):
        self.start = time.monotonic()
        self.counts = dict(dit=0, backward=0, commit=0, identity=0, update=0, video=0)
        self.events = []
        self.status = "RUNNING"
        self.save()

    def save(self, **extra):
        write(HERE / "GPU_LEDGER.json", dict(task="EXP-028/v1", status=self.status,
            elapsed_seconds=time.monotonic()-self.start, counts=self.counts,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
            rss_gib=rss_gib(), events=self.events, **extra))

    def reserve(self, kind, label):
        if time.monotonic()-self.start >= LIMIT-30:
            raise TimeoutError("GPU process time cap")
        if (kind in ("commit", "identity", "main") and self.counts["dit"] >= 14):
            raise RuntimeError("DiT call cap")
        if kind == "backward" and self.counts["backward"] >= 8:
            raise RuntimeError("call cap")
        if kind in ("commit", "identity", "main"):
            self.counts["dit"] += 1
            if kind != "main": self.counts[kind] += 1
        else:
            self.counts[kind] += 1
        self.events.append(dict(kind=kind, label=label, seconds=time.monotonic()-self.start))
        self.save()

    def gate(self):
        torch.cuda.synchronize()
        if torch.cuda.max_memory_allocated()/2**30 >= 40 or rss_gib() >= 96:
            raise MemoryError("GPU or process memory cap")
        if time.monotonic()-self.start >= LIMIT-20:
            raise TimeoutError("GPU process time cap")
        self.save()


def qkv_metrics(acc):
    values = {}
    for role in ROLE_NAMES:
        g = acc.values[role].view(56, 3, 128, 8)
        values[role] = {name: float(g[:, i].norm()) for i, name in enumerate("qkv")}
    def cosine(a, b):
        na, nb = float(a.norm()), float(b.norm())
        return None if min(na, nb) <= 1e-12 else float(torch.sum(a*b)/(na*nb))
    return dict(role_qkv_norms=values,
                action_video_cos={name: cosine(
                    sum(acc.values[r] for r in ("action_history", "action_current")).view(56,3,128,8)[:, i],
                    sum(acc.values[r] for r in ("video_history", "video_current")).view(56,3,128,8)[:, i])
                    for i, name in enumerate("qkv")})


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES", "").isdigit()
    auth = json.loads((HERE / "AUTHORIZATION.json").read_text())
    assert auth["task_id"] == "EXP-028" and auth["max_top_level_dit_attempts"] == 14
    cpu = json.loads((HERE / "CPU_PREFLIGHT.json").read_text())
    assert cpu["status"] == "PASS_CPU"
    for path, expected in cpu["source_sha256"].items():
        if sha(ROOT / path) != expected:
            raise RuntimeError(f"source/input SHA changed: {path}")
    import run as prior
    import fm_run as old
    from causal.pretrained_lora import install_adapters
    from causal.h3_cached import H3ChunkCache
    import causal
    causal.__path__.append(str(ROOT / "H3-World/code/causal"))
    from causal.local_transition import transition_forward
    from interval_train import interval_train
    prior.setup()
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    ledger = Ledger()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("process alarm")))
    signal.alarm(LIMIT-5)
    results = []
    try:
        pipe, model, precision = prior.model()
        del pipe
        adapters, indices = install_adapters(model, rank=8, block_indices=(0,49), device="cuda:0")
        assert indices == [0,49]
        state = torch.load(EXP25 / "checkpoint_000.pt", map_location="cpu", weights_only=True)
        assert state["step"] == 0 and len(state["adapter"]) == 50
        with torch.no_grad():
            for a, layer in zip(adapters, indices):
                assert tuple(state["adapter"][layer]["A"].shape) == (8,5376)
                assert tuple(a.lora_B.shape) == (21504,8)
                a.lora_A.copy_(state["adapter"][layer]["A"])
                a.lora_B.zero_()
                a.forward = diagnostic_forward.__get__(a, type(a))
                a.role_accumulator = None
        del state
        for clip_index, clip in enumerate(CLIPS):
            data = old.load_clip(clip)
            clean = data["z"][:,:,12:17]
            history = data["z"][:,:,:12]
            packed, prompt = data["vis"][17]
            cache = H3ChunkCache(5,"cpu")
            ledger.reserve("commit", clip)
            with torch.no_grad(), old.verified_attention_scope():
                interval_train(model, history, start=0, index=0, cache=cache,
                    full_packed=data["vis"][12][0], prompt=data["vis"][12][1],
                    anchor=data["anchor"], audio=data["audio"], sigma=0.0, commit=True)
            assert cache.commits == 50
            ledger.gate()
            for sigma, seed in SIGMAS:
                generator = torch.Generator(device="cpu").manual_seed(seed)
                noise = torch.randn(tuple(data["z"].shape), generator=generator, dtype=torch.float32).to("cuda:0")
                current = (1-sigma)*clean + sigma*noise[:,:,12:17]
                target = noise[:,:,12:17]-clean
                for graph in ("original_visible", "v3"):
                    label = f"{clip}/{sigma}/{graph}"
                    labels = original_layout(packed,390) if graph == "original_visible" else v3_layout(packed,390)
                    references = None
                    if clip_index == 0:
                        ledger.reserve("identity", label)
                        for a in adapters: a.enabled = False
                        with torch.no_grad():
                            if graph == "original_visible":
                                references = transition_forward(model,current,history=history,
                                    history_noise=noise[:,:,:12],full_packed=packed,prompt=prompt,
                                    anchor=data["anchor"],audio=data["audio"],sigma=sigma,index=1,
                                    chunk_frames=12,checkpoint=False)
                            else:
                                with old.verified_attention_scope():
                                    references = interval_train(model,current,start=12,index=1,
                                        cache=cache,full_packed=packed,prompt=prompt,
                                        anchor=data["anchor"],audio=data["audio"],sigma=sigma)
                        for a in adapters: a.enabled = True
                        ledger.gate()
                    for a in adapters:
                        a.lora_A.grad = a.lora_B.grad = None
                        a.role_accumulator = RoleAccumulator(labels, 21504, 8, a.scale)
                    ledger.reserve("main",label)
                    if graph == "original_visible":
                        prediction = transition_forward(model,current,history=history,
                            history_noise=noise[:,:,:12],full_packed=packed,prompt=prompt,
                            anchor=data["anchor"],audio=data["audio"],sigma=sigma,index=1,
                            chunk_frames=12,checkpoint=True,checkpoint_offload=True)
                    else:
                        with old.verified_attention_scope():
                            prediction = interval_train(model,current,start=12,index=1,
                                cache=cache,full_packed=packed,prompt=prompt,
                                anchor=data["anchor"],audio=data["audio"],sigma=sigma,
                                checkpoint=True,offload=True)
                            loss = F.mse_loss(prediction.float(),target.float())
                            if references is not None:
                                diff = prediction.float()-references.float()
                                rel = float(diff.square().mean().sqrt() / references.float().square().mean().sqrt().clamp_min(1e-12))
                                mx = float(diff.abs().max())
                                assert rel <= 1e-6 and mx <= 1e-6*max(1,float(references.float().abs().max()))
                            ledger.reserve("backward",label)
                            loss.backward()
                    if graph == "original_visible":
                        loss = F.mse_loss(prediction.float(),target.float())
                        if references is not None:
                            diff = prediction.float()-references.float()
                            rel = float(diff.square().mean().sqrt() / references.float().square().mean().sqrt().clamp_min(1e-12))
                            mx = float(diff.abs().max())
                            assert rel <= 1e-6 and mx <= 1e-6*max(1,float(references.float().abs().max()))
                        ledger.reserve("backward",label)
                        loss.backward()
                    layer_rows = {}
                    for a, layer in zip(adapters,indices):
                        check = a.role_accumulator.check(a.lora_B.grad)
                        if not check["pass_role_sum"]:
                            raise RuntimeError(f"role sum failed layer {layer}: {check}")
                        layer_rows[str(layer)] = dict(**check, **qkv_metrics(a.role_accumulator),
                            grad_a_structural_zero=bool(torch.count_nonzero(a.lora_A.grad) == 0))
                        a.role_accumulator = None
                    results.append(dict(clip=clip,sigma=sigma,seed=seed,graph=graph,
                        loss=float(loss.detach()),identity_relative_rms=rel if references is not None else None,
                        identity_max_abs=mx if references is not None else None,layers=layer_rows,
                        history="same_sigma_recompute" if graph=="original_visible" else "GT_C1_sigma0_detached_KV"))
                    write(HERE / "ROLE_RESULTS.json", results)
                    del prediction, loss, references
                    ledger.gate()
            del cache, data
        ledger.status = "COMPLETE_PENDING_JUDGE"
        ledger.save(precision=precision)
    except BaseException as exc:
        ledger.status = "FAILED_PENDING_JUDGE"
        ledger.save(error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
