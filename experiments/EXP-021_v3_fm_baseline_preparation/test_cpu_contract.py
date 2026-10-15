"""No 33B calls: tests CPU FM math, visibility, gradient boundary and resume."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from fm_contract import (detached_clean_commit, fm_pair, fm_update,
                         load_training_state, save_training_state, visible_actions)


class Contracts(unittest.TestCase):
    def test_fm_sign_and_sigma(self):
        clean, noise = torch.tensor([2., -1.]), torch.tensor([-2., 3.])
        x0, velocity = fm_pair(clean, noise, 0.)
        x1, _ = fm_pair(clean, noise, 1.)
        mid, _ = fm_pair(clean, noise, .25)
        torch.testing.assert_close(x0, clean)
        torch.testing.assert_close(x1, noise)
        torch.testing.assert_close(mid, clean + .25 * velocity)
        torch.testing.assert_close(velocity, torch.tensor([-4., 4.]))

    def test_future_action_isolation(self):
        a = ["A"] * 12 + ["D"] * 5
        b = ["A"] * 12 + ["W"] * 5
        self.assertEqual(visible_actions(a, 12), visible_actions(b, 12))
        self.assertNotEqual(visible_actions(a, 17), visible_actions(b, 17))

    def test_clean_commit_has_no_history_grad(self):
        history = torch.ones(2, requires_grad=True)
        cache = detached_clean_commit(lambda x: x * 3, history)
        self.assertFalse(cache.requires_grad)
        current = torch.ones(2, requires_grad=True)
        (current * cache).sum().backward()
        self.assertIsNone(history.grad)
        torch.testing.assert_close(current.grad, cache)

    def test_adapter_only_update_and_resume(self):
        base = torch.nn.Linear(2, 2, bias=False)
        adapter = torch.nn.Linear(2, 2, bias=False)
        for p in base.parameters():
            p.requires_grad_(False)
        adapter.weight.data.zero_()
        before = base.weight.detach().clone()
        opt = torch.optim.AdamW(adapter.parameters(), lr=.1)
        gen = torch.Generator().manual_seed(13)
        clean = torch.tensor([[1., 2.]])
        noise = torch.tensor([[3., -1.]])
        history = detached_clean_commit(lambda x: x * 2, clean)
        actions = visible_actions(["A", "D"], 1)
        def forward(x, sigma, cache, prefix):
            self.assertEqual(prefix, ("A",))
            self.assertFalse(cache.requires_grad)
            return base(x) + adapter(x)
        fm_update(forward, clean, noise, .5, opt, history, actions)
        torch.testing.assert_close(base.weight, before)
        self.assertTrue(any(p.grad is not None for p in adapter.parameters()))
        meta = {"task": "EXP-021", "objective": "ordinary_fm", "partition": [12, 5]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "step1.pt"
            save_training_state(path, adapter, opt, 1, gen, meta)
            expected = torch.rand(3, generator=gen)
            adapter2 = torch.nn.Linear(2, 2, bias=False)
            opt2 = torch.optim.AdamW(adapter2.parameters(), lr=.1)
            gen2 = torch.Generator().manual_seed(999)
            self.assertEqual(load_training_state(path, adapter2, opt2, gen2, meta), 1)
            torch.testing.assert_close(adapter.weight, adapter2.weight)
            self.assertEqual(len(opt.state_dict()["state"]), len(opt2.state_dict()["state"]))
            torch.testing.assert_close(torch.rand(3, generator=gen2), expected)
            with self.assertRaises(ValueError):
                load_training_state(path, adapter2, opt2, gen2, {"objective": "anyflow"})


if __name__ == "__main__":
    unittest.main()
