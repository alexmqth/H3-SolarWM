"""Serial student/Original-teacher/critic roles on one frozen H3 backbone.

Independent LoRA parameters are selected by flags, never copied over live
student parameters. This preserves an outstanding checkpointed FMBS graph
while a critic optimizer updates its own bank. This module does not choose
attention, conditioning, targets, loss, rollout, or the training schedule.
"""
from contextlib import contextmanager, nullcontext

import torch

from .stage1_lora import Stage1LinearLoRA, Stage1LoRABank


def _residual_parameters(module):
    # A QKV residual can wrap a large frozen base. Only its added parameters
    # are role-owned; the nested base must remain in the shared-frozen audit.
    return [p for name,p in module.named_parameters() if not name.startswith('base.')]


class SharedH3Roles:
    """Role isolation for an installed student Stage1 bank; single-thread use.

    Teacher/critic disable added student banks and optional student-only
    residuals. They temporarily remove the AnyFlow time conditioner so they
    predict instantaneous velocities using Original's time embedder. Original
    released H3 action LoRA inside the frozen base remains active.

    A caller must explicitly select bidirectional Original attention for
    teacher/critic; this manager intentionally does not equate a causal cache
    with that reference. Student, teacher and critic cannot share history KV
    computed with different effective weights/attention.

    Backward must run inside the same role as forward, including checkpoint
    recomputation. Attach guard_backward to the returned endpoint or loss to
    reject an incorrectly scoped backward before any graph is replayed.
    """
    def __init__(self, model, student_bank, *, critic_names=None, critic_rank=8,
                 critic_alpha=8., device='cpu', student_only_modules=()):
        if not isinstance(student_bank, Stage1LoRABank):
            raise TypeError('An installed Stage1LoRABank is required')
        if len(student_bank.names) != len(student_bank.modules):
            raise ValueError('Malformed student bank')
        if any(model.get_submodule(n) is not m for n,m in zip(student_bank.names,student_bank.modules)):
            raise ValueError('Student bank must be the currently installed projections')
        if any(not m.enabled for m in student_bank.modules):
            raise ValueError('Student bank must initially be enabled')
        names=tuple(student_bank.names if critic_names is None else critic_names)
        if not names or len(names)!=len(set(names)) or any(n not in student_bank.names for n in names):
            raise ValueError('Critic targets must be a nonempty subset of the student projections')
        if critic_rank < 1 or critic_alpha <= 0 or not torch.isfinite(torch.tensor(float(critic_alpha))):
            raise ValueError('Invalid critic rank/alpha')
        extras=tuple(student_only_modules)
        if len({id(m) for m in extras}) != len(extras):
            raise ValueError('Duplicate student-only module')
        if any(not hasattr(m,'enabled') for m in extras):
            raise ValueError('Student-only modules must expose an enabled flag')
        if any(m in student_bank.modules for m in extras):
            raise ValueError('Student bank modules need not be listed twice')
        self.model=model;self.student_bank=student_bank;self.active=None
        self.student_only_modules=extras
        self._student_extra_flags=tuple(m.enabled for m in extras)
        self._anyflow=getattr(model,'anyflow_conditioner',None)
        # Stage1LinearLoRA freezes its base recursively. Preserve all existing
        # trainability flags when nesting an independent critic wrapper.
        trainability=[(p,p.requires_grad) for p in model.parameters()]
        modules=[];dev=torch.device(device)
        devices=[dev.index if dev.index is not None else torch.cuda.current_device()] if dev.type=='cuda' else []
        try:
            with torch.random.fork_rng(devices=devices):
                for name in names:
                    parent,_,slot=name.rpartition('.')
                    module=Stage1LinearLoRA(model.get_submodule(name),rank=critic_rank,alpha=critic_alpha,
                        device=dev,split_qkv=name.endswith('attn.qkv_proj'))
                    module.enabled=False
                    setattr(model.get_submodule(parent),slot,module);modules.append(module)
        except BaseException:
            for name,module in zip(names,modules):
                parent,_,slot=name.rpartition('.');setattr(model.get_submodule(parent),slot,module.base)
            raise
        finally:
            for p,flag in trainability:p.requires_grad_(flag)
        self.critic_bank=Stage1LoRABank(names,modules,int(critic_rank),float(critic_alpha))
        owned={id(p) for p in student_bank.parameters()+self.critic_bank.parameters()}
        for extra in extras:
            owned.update(id(p) for p in _residual_parameters(extra))
        if self._anyflow is not None:owned.update(id(p) for p in self._anyflow.parameters())
        self._shared=[p for p in model.parameters() if id(p) not in owned]
        if any(p.requires_grad for p in self._shared):
            # Undo installation on invalid initial state, not half-installed.
            for name,module in zip(names,modules):
                parent,_,slot=name.rpartition('.');setattr(model.get_submodule(parent),slot,module.base)
            raise ValueError('Shared backbone parameters must be frozen before role installation')
        if {id(p) for p in self.student_bank.parameters()} & {id(p) for p in self.critic_bank.parameters()}:
            raise RuntimeError('Student and critic parameters alias')

    @contextmanager
    def context(self, role):
        if role not in ('student','teacher','critic'):
            raise ValueError(f'Unknown H3 role: {role}')
        modules=[*self.student_bank.modules,*self.critic_bank.modules,*self.student_only_modules]
        flags=[m.enabled for m in modules];previous=self.active
        previous_time=getattr(self.model,'anyflow_conditioner',None)
        if previous_time is not None and previous_time is not self._anyflow:
            raise RuntimeError('AnyFlow module was replaced outside role management')
        frozen_versions=[(p,p._version) for p in self._shared]
        protected=[]
        if role!='student':
            protected=[(p,p._version) for p in self.student_bank.parameters()]
            if self._anyflow is not None:protected += [(p,p._version) for p in self._anyflow.parameters()]
            for extra in self.student_only_modules:protected += [(p,p._version) for p in _residual_parameters(extra)]
        self.active=role
        try:
            for m in self.student_bank.modules:m.enabled=role=='student'
            for m in self.critic_bank.modules:m.enabled=role=='critic'
            for m,flag in zip(self.student_only_modules,self._student_extra_flags):
                m.enabled=flag if role=='student' else False
            if hasattr(self.model,'anyflow_conditioner'):delattr(self.model,'anyflow_conditioner')
            if role=='student' and self._anyflow is not None:self.model.anyflow_conditioner=self._anyflow
            with torch.no_grad() if role=='teacher' else nullcontext():
                yield self
            if any(p._version!=v for p,v in frozen_versions):
                raise RuntimeError('Role evaluation changed frozen backbone parameters')
            if any(p._version!=v for p,v in protected):
                raise RuntimeError('Teacher/critic role changed student parameters')
        finally:
            for m,flag in zip(modules,flags):m.enabled=flag
            if hasattr(self.model,'anyflow_conditioner'):delattr(self.model,'anyflow_conditioner')
            if previous_time is not None:self.model.anyflow_conditioner=previous_time
            self.active=previous

    def guard_backward(self, tensor, role):
        if role not in ('student','critic') or self.active!=role:
            raise ValueError('Attach backward guard while the gradient role is active')
        if not tensor.requires_grad:
            raise ValueError('Backward guard requires a tensor with gradients')
        def check(grad):
            if self.active!=role:
                raise RuntimeError(f'{role} backward requires the {role} role, including checkpoint replay')
            return grad
        tensor.register_hook(check)
        return tensor
