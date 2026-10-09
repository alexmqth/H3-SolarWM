"""DMD sign, detach semantics, official parity and live H3 FMBS integration."""
import importlib.util
from pathlib import Path
import sys

import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from causal.dmd import score_sample,critic_flow_loss,distribution_matching_loss


@pytest.mark.parametrize('masked',[False,True])
def test_matches_solarwm_gradient_and_mean_surrogate(masked):
    source=ROOT.parent/'SolarWM/src/solarwm/training/sgf.py'
    if not source.exists():pytest.skip('SolarWM reference checkout is required for source parity')
    spec=importlib.util.spec_from_file_location('solarwm_sgf_reference',source)
    official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
    torch.manual_seed(332)
    output=torch.randn(2,3,4,requires_grad=True)
    noise=torch.randn_like(output,requires_grad=True)
    sigma=torch.tensor([.2,.8]);noisy=score_sample(output,noise,sigma)
    real=torch.randn_like(output,requires_grad=True);fake=torch.randn_like(output,requires_grad=True)
    mask=None
    if masked:
        mask=torch.ones(2,1,4,dtype=torch.bool);mask[0,:,0]=False
        with torch.no_grad():real[0,:,0]=float('nan');fake[0,:,0]=float('nan')
    loss,direction,stats=distribution_matching_loss(output,noisy,sigma,
        real_velocity=real,fake_velocity=fake,mask=mask)
    s=sigma[:,None,None]
    expected=official.compute_sgf_kl_gradient(fake_x0=noisy-s*fake,real_x0=noisy-s*real,
        student_output=output.detach(),mask=mask)
    expected_loss=official.sgf_student_loss(output,expected,mask=mask)
    torch.testing.assert_close(direction,expected,rtol=0,atol=0)
    torch.testing.assert_close(loss,expected_loss,rtol=0,atol=0)
    actual=torch.autograd.grad(loss,output,retain_graph=True)[0]
    reference=torch.autograd.grad(expected_loss,output)[0]
    torch.testing.assert_close(actual,reference,rtol=0,atol=0)
    assert not direction.requires_grad and not noisy.requires_grad
    assert real.grad is None and fake.grad is None and noise.grad is None


def test_oracle_gaussian_score_moves_student_mean_toward_teacher():
    # Exact conditional means for equal-variance Gaussian real/fake models.
    mu=torch.tensor([[1.4,-.9]],requires_grad=True);teacher_mean=torch.tensor([[-.3,.6]])
    sigma=.65;alpha=1-sigma;variance=.4
    noisy=score_sample(mu,torch.tensor([[.2,-.1]]),sigma)
    gain=alpha*variance/(alpha*alpha*variance+sigma*sigma)
    fake_x0=mu.detach()+gain*(noisy-alpha*mu.detach())
    real_x0=teacher_mean+gain*(noisy-alpha*teacher_mean)
    loss,direction,_=distribution_matching_loss(mu,noisy,sigma,
        real_velocity=(noisy-real_x0)/sigma,fake_velocity=(noisy-fake_x0)/sigma,normalize=False)
    loss.backward()
    expected=(1-alpha*gain)*(mu.detach()-teacher_mean)
    torch.testing.assert_close(direction,expected)
    before=(mu.detach()-teacher_mean).square().sum()
    after=(mu.detach()-.1*mu.grad-teacher_mean).square().sum()
    wrong_sign=(mu.detach()+.1*mu.grad-teacher_mean).square().sum()
    assert after<before<wrong_sign


def test_critic_target_is_detached_and_zero_when_fitted():
    endpoint=torch.randn(1,3,4,requires_grad=True);noise=torch.randn_like(endpoint,requires_grad=True)
    prediction=(noise-endpoint).detach().requires_grad_()
    exact=critic_flow_loss(prediction,endpoint,noise);assert exact==0
    loss=critic_flow_loss(prediction+.1,endpoint,noise);loss.backward()
    assert prediction.grad.abs().sum()>0 and endpoint.grad is None and noise.grad is None


def test_invalid_protocol_and_active_nan_rejected():
    x=torch.ones(2,3,requires_grad=True);v=torch.zeros_like(x)
    for sigma in (0.,-1.,1.1,float('nan'),True,torch.ones(3)):
        with pytest.raises(ValueError):score_sample(x,v,sigma)
    with pytest.raises(ValueError,match='live student'):distribution_matching_loss(x.detach(),v,.5,real_velocity=v,fake_velocity=v)
    with pytest.raises(ValueError,match='share shape'):distribution_matching_loss(x,v[:1],.5,real_velocity=v,fake_velocity=v)
    with pytest.raises(FloatingPointError):distribution_matching_loss(x,v,.5,real_velocity=v+float('nan'),fake_velocity=v)


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_h3_full_fmbs_receives_original_vs_updated_critic_direction(dtype):
    from test_shared_h3_roles import setup,generated_history,original_forward
    from causal.fmbs import simulate_h3_chunk
    model,bank,time,roles,b,common,_=setup(dtype)
    cache,history,grid=generated_history(model,roles,b,common)
    params=bank.parameters()+list(time.parameters())
    critic=roles.critic_bank.parameters();optimizer=torch.optim.AdamW(critic,lr=.001)
    kv=[(x,x.clone(),x._version) for entries in cache.layers.values() for e in entries for x in (e.key,e.value,e.rope)]
    with roles.context('student'):
        live=roles.guard_backward(simulate_h3_chunk(model,b['noise'][:,:,10:],sigmas=grid,
            interval_index=2,chunk_index=2,cache=cache,conditions=common),'student')
        noise=b['noise'][:,:,10:];sigma=.65;noisy=score_sample(live,noise,sigma)
        with roles.context('critic'):
            prediction=original_forward(model,noisy,history,b,sigma,checkpoint=True)
            loss=roles.guard_backward(critic_flow_loss(prediction,live,noise),'critic')
            loss.backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
        assert all(p.grad is None for p in params)
        with roles.context('teacher'):
            real=original_forward(model,noisy,history,b,sigma)
        with roles.context('critic'),torch.no_grad():
            fake=original_forward(model,noisy,history,b,sigma)
        loss,direction,_=distribution_matching_loss(live,noisy,sigma,real_velocity=real,fake_velocity=fake)
        reference=torch.autograd.grad(live,params,grad_outputs=direction/live.numel(),retain_graph=True,allow_unused=True)
        roles.guard_backward(loss,'student').backward()
    assert float(direction.norm())>0
    for p,ref in zip(params,reference):
        if ref is None:assert p.grad is None
        else:torch.testing.assert_close(p.grad,ref,rtol=1e-4,atol=2e-7)
    assert all(p.grad is None for p in critic+roles._shared)
    assert all(torch.equal(x,old) and x._version==version for x,old,version in kv)
