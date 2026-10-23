"""Independent formula (11)/(12) checks on tiny nonlinear networks, not experiment results."""

from __future__ import annotations

import copy
from types import SimpleNamespace
import unittest

import torch
from torch.nn import functional as F

from MDLM.candidate import CandidateModel


class CandidateTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Tanh(), torch.nn.Linear(3, 2)).double().eval()
        self.factors = {"0": torch.randn(2, 4, dtype=torch.float64), "2": torch.randn(2, 3, dtype=torch.float64)}
        self.inputs = torch.randn(2, 5, 4, dtype=torch.float64)

    def test_zero_r_is_exact_and_scope_restores_weights_flags_and_hooks(self):
        before = copy.deepcopy(self.model.state_dict())
        expected = self.model(self.inputs).detach()
        with CandidateModel(self.model, self.factors, dtype="float64") as candidate:
            self.assertTrue(torch.equal(self.model(self.inputs), expected))
            self.assertTrue(all(not value.requires_grad for value in self.model.parameters()))
            self.assertTrue(all(not branch.q.requires_grad for branch in candidate.branches.values()))
        self.assertTrue(all(value.requires_grad for value in self.model.parameters()))
        self.assertFalse(self.model[0]._forward_hooks)
        self.assertTrue(all(torch.equal(value, before[name]) for name, value in self.model.state_dict().items()))

    def test_all_positions_match_explicit_candidate_weights_with_bias(self):
        rs = {name: torch.randn(2, 2, dtype=torch.float64) if name == "2" else torch.randn(3, 2, dtype=torch.float64)
              for name in self.factors}
        expected = F.linear(torch.tanh(F.linear(self.inputs, self.model[0].weight + rs["0"] @ self.factors["0"],
                                                self.model[0].bias)),
                            self.model[2].weight + rs["2"] @ self.factors["2"], self.model[2].bias)
        with CandidateModel(self.model, self.factors, dtype="float64") as candidate:
            candidate.set_residuals(rs)
            torch.testing.assert_close(self.model(self.inputs), expected, atol=1e-12, rtol=1e-12)

    def test_multilayer_gradient_matches_independent_effective_weights_and_formula_12(self):
        with CandidateModel(self.model, self.factors, dtype="float64") as candidate:
            candidate.set_residuals({name: torch.randn_like(branch.r) * 0.1 for name, branch in candidate.branches.items()})
            weights = {name: (self.model.get_submodule(name).weight + branch.r.detach() @ branch.q).detach().requires_grad_()
                       for name, branch in candidate.branches.items()}
            direct = F.linear(torch.tanh(F.linear(self.inputs, weights["0"], self.model[0].bias)),
                              weights["2"], self.model[2].bias).square().sum()
            gradients = torch.autograd.grad(direct, tuple(weights.values()))
            loss = self.model(self.inputs).square().sum()
            actual = torch.autograd.grad(loss, candidate.parameters())
            for name, gradient, value in zip(weights, gradients, actual):
                torch.testing.assert_close(value, gradient @ self.factors[name].T, atol=1e-11, rtol=1e-11)
                self.assertGreater(float(value.norm()), 0)

    def test_early_residual_changes_later_input_and_gradients_are_continuous(self):
        captured = []
        handle = self.model[2].register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0].detach().clone()))
        try:
            with CandidateModel(self.model, self.factors, dtype="float64") as candidate:
                self.model(self.inputs)
                with torch.no_grad():
                    candidate.branches["0"].r.fill_(0.2)
                self.model(self.inputs).sum().backward()
                self.assertFalse(torch.equal(captured[0], captured[1]))
                self.assertGreater(float(candidate.branches["0"].r.grad.norm()), 0)
        finally:
            handle.remove()

    def test_repeated_module_invocations_each_receive_the_branch(self):
        class Reused(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Linear(2, 2, bias=False).double()

            def forward(self, x):
                return self.layer(torch.tanh(self.layer(x)))

        model = Reused()
        q = torch.eye(2, dtype=torch.float64)
        x = torch.randn(3, 2, dtype=torch.float64)
        r = torch.ones_like(q) * 0.2
        weight = model.layer.weight.detach() + r @ q
        expected = F.linear(torch.tanh(F.linear(x, weight)), weight)
        with CandidateModel(model, {"layer": q}, dtype="float64") as candidate:
            candidate.set_residuals({"layer": r})
            torch.testing.assert_close(model(x), expected)

    def test_hooks_are_removed_after_forward_exception(self):
        with self.assertRaisesRegex(RuntimeError, "constructed failure"):
            with CandidateModel(self.model, self.factors, dtype="float64"):
                raise RuntimeError("constructed failure")
        self.assertFalse(self.model[0]._forward_hooks)
        self.assertTrue(all(value.requires_grad for value in self.model.parameters()))

    def test_nested_candidate_is_rejected_without_damaging_outer_scope(self):
        with CandidateModel(self.model, self.factors, dtype="float64"):
            with self.assertRaisesRegex(ValueError, "already installed"):
                CandidateModel(self.model, self.factors, dtype="float64")
            self.assertEqual(len(self.model[0]._forward_hooks), 1)
        self.assertFalse(hasattr(self.model[0], "_mdlm_candidate_owner"))

    def test_q_is_copied_and_dtype_change_is_rejected(self):
        with CandidateModel(self.model, self.factors, dtype="float64") as candidate:
            before = candidate.branches["0"].q.clone()
            self.factors["0"].add_(1)
            self.assertTrue(torch.equal(candidate.branches["0"].q, before))
        with self.assertRaisesRegex(ValueError, "saved Q dtype"):
            CandidateModel(self.model, self.factors, dtype="float32")

    def test_bad_shapes_nonfinite_and_shared_weights_fail_before_install(self):
        for q in (torch.ones(2, 5, dtype=torch.float64), torch.full((2, 4), float("nan"), dtype=torch.float64)):
            with self.assertRaises(ValueError):
                CandidateModel(self.model, {"0": q}, dtype="float64")
        self.assertFalse(self.model[0]._forward_hooks)
        model = torch.nn.ModuleDict({"a": torch.nn.Linear(2, 2), "b": torch.nn.Linear(2, 2)})
        model["b"].weight = model["a"].weight
        with self.assertRaisesRegex(ValueError, "shared target"):
            CandidateModel(model, {"a": torch.eye(2), "b": torch.eye(2)})

    def test_target_shared_with_unselected_module_is_rejected(self):
        model = torch.nn.ModuleDict({"a": torch.nn.Linear(2, 2), "b": torch.nn.Linear(2, 2)})
        model["b"].weight = model["a"].weight
        with self.assertRaisesRegex(ValueError, "shared target"):
            CandidateModel(model, {"a": torch.eye(2)})

    def test_gradient_blocking_hooks_on_any_layer_are_rejected(self):
        for hook in (SimpleNamespace(no_grad=True, offload=False),
                     SimpleNamespace(no_grad=False, hooks=[SimpleNamespace(offload=True)])):
            with self.subTest(hook=hook):
                self.model[1]._hf_hook = hook  # 未选中的中间层也可能切断早期 R 的梯度。
                with self.assertRaisesRegex(ValueError, "gradient|offload"), \
                        CandidateModel(self.model, self.factors, dtype="float64"):
                    pass
                self.assertFalse(self.model[0]._forward_hooks)
        del self.model[1]._hf_hook

    def test_close_is_idempotent_and_preserves_each_module_mode(self):
        self.model.train()
        self.model[1].eval()
        modes = [module.training for module in self.model.modules()]
        first = CandidateModel(self.model, self.factors, dtype="float64")
        first.close()
        self.assertEqual([module.training for module in self.model.modules()], modes)
        with CandidateModel(self.model, self.factors, dtype="float64"):
            first.close()
            self.assertTrue(all(not parameter.requires_grad for parameter in self.model.parameters()))


if __name__ == "__main__":
    unittest.main()
