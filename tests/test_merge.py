"""Independent P5 matrix and nonlinear forward checks; constructed CPU fixtures only."""

import copy
import unittest
from unittest.mock import patch

import torch

from MDLM.merge import EquivalenceError, WeightMerge, capture_outputs, merge_with_equivalence
from tests.test_optimize import TinyAdapter
from model.base import KeepModelInput


class MergeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.Tanh(), torch.nn.Linear(3, 3)).double().eval()
        self.q = {name: torch.randn(2, 3, dtype=torch.float64) for name in ("0", "2")}
        self.r = {name: torch.randn(3, 2, dtype=torch.float64) / 10 for name in self.q}
        self.before = copy.deepcopy(self.model.state_dict())
        self.adapter = TinyAdapter()
        self.input = KeepModelInput((0, 1, 2), (1, 1, 1), ())

    def merger(self, **kwargs):
        return WeightMerge(self.model, self.q, self.r, compute_dtype="float64", chunk_size=2, **kwargs)

    def test_formula13_and15_trials_rebuild_baseline_and_leave_biases_unchanged(self):
        merger = self.merger()
        for alpha in (1.0, 0.5, 0.125):
            merger.apply(alpha)
            for name in self.q:
                expected = self.before[name + ".weight"] + alpha * (self.r[name] @ self.q[name])
                torch.testing.assert_close(self.model.get_submodule(name).weight, expected, atol=0, rtol=0)
                self.assertTrue(torch.equal(self.model.get_submodule(name).bias, self.before[name + ".bias"]))
        merger.restore()
        self.assertTrue(all(torch.equal(value, self.before[name]) for name, value in self.model.state_dict().items()))

    def test_all_position_multilayer_factor_and_actual_merge_equivalence(self):
        merger = self.merger()
        result = merge_with_equivalence(merger, self.adapter, self.q, self.r, [(self.input, (0, 2))], .5,
                                        atol=1e-12, rtol=1e-12)
        self.assertTrue(result["passed"])
        self.assertLess(result["logits_max_abs"], 1e-12)
        self.assertTrue(all(not module._forward_hooks for module in self.model.modules()))
        self.assertTrue(all(not hasattr(module, "_mdlm_candidate_owner") for module in self.model.modules()))

    def test_formula14_soft_fit_error_is_diagnostic_and_can_be_nonzero(self):
        merger = self.merger()
        keys = {name: torch.randn(3, 2, dtype=torch.float64) for name in self.q}
        rows = merger.diagnostics(self.q, self.r, keys)
        for name, row in rows.items():
            error = self.r[name] @ (self.q[name] @ keys[name] - torch.eye(2))
            self.assertAlmostEqual(row["soft_fit_error_norm"], float(error.norm()), places=12)
            self.assertGreater(row["soft_fit_error_norm"], 0)

    def test_shared_target_with_unselected_module_is_rejected(self):
        self.model.add_module("alias", torch.nn.Linear(3, 3).double())
        self.model.alias.weight = self.model[0].weight
        with self.assertRaisesRegex(ValueError, "shared target"):
            self.merger()

    def test_shape_dtype_and_finiteness_contracts(self):
        for r in ({"0": torch.zeros(3, 1), "2": self.r["2"]},
                  {**self.r, "0": self.r["0"].float()},
                  {**self.r, "0": self.r["0"] * float("nan")}):
            with self.assertRaises((ValueError, FloatingPointError)):
                WeightMerge(self.model, self.q, r, compute_dtype="float64")
        with self.assertRaisesRegex(ValueError, "preserve saved"):
            WeightMerge(self.model, self.q, self.r, compute_dtype="float32")

    def test_parameter_replacement_and_nontarget_changes_are_detected(self):
        merger = self.merger()
        with torch.no_grad():
            self.model[0].bias.add_(1)
        with self.assertRaisesRegex(RuntimeError, "whitelist"):
            merger.check_integrity()

    def test_target_output_check_runs_even_when_logits_match(self):
        merger = self.merger()
        from MDLM import merge
        original, calls = merge.capture_outputs, 0

        def wrong_outputs(*args):
            nonlocal calls
            logits, outputs = original(*args)
            calls += 1
            if calls == 2:
                outputs["0"][0].add_(1)
            return logits, outputs

        with patch.object(merge, "capture_outputs", side_effect=wrong_outputs):
            with self.assertRaisesRegex(EquivalenceError, "target outputs"):
                merge_with_equivalence(merger, self.adapter, self.q, self.r, [(self.input, (0,))], 1,
                                       atol=1e-12, rtol=1e-12)
        merger.restore()

    def test_capture_hooks_are_removed_when_forward_raises(self):
        with patch.object(self.adapter, "forward_logits", side_effect=RuntimeError("forward failed")):
            with self.assertRaisesRegex(RuntimeError, "forward failed"):
                capture_outputs(self.adapter, self.model, self.input, (0,), self.q)
        self.assertTrue(all(not module._forward_hooks for module in self.model.modules()))

    def test_repeated_target_invocations_match_after_merge(self):
        class Repeated(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Linear(3, 3).double()

            def forward(self, inputs):
                return self.layer(torch.tanh(self.layer(inputs)))

        model = Repeated().eval()
        q, r = {"layer": self.q["0"]}, {"layer": self.r["0"]}
        merger = WeightMerge(model, q, r, compute_dtype="float64")
        result = merge_with_equivalence(merger, self.adapter, q, r, [(self.input, (0, 1, 2))], .75, atol=1e-12, rtol=1e-12)
        self.assertTrue(result["passed"])

    def test_logits_mismatch_is_detected_and_hooks_are_clean(self):
        from MDLM import merge
        original, calls = merge.capture_outputs, 0

        def wrong_logits(*args):
            nonlocal calls
            logits, outputs = original(*args)
            calls += 1
            return (logits + 1 if calls == 2 else logits), outputs

        merger = self.merger()
        with patch.object(merge, "capture_outputs", side_effect=wrong_logits):
            with self.assertRaisesRegex(EquivalenceError, "logits"):
                merge_with_equivalence(merger, self.adapter, self.q, self.r, [(self.input, (0,))], 1, atol=1e-12, rtol=1e-12)
        merger.restore()
        self.assertTrue(all(not module._forward_hooks for module in self.model.modules()))

    def test_half_precision_storage_overflow_is_rejected_and_recoverable(self):
        self.model.half()
        q = {name: value.float() for name, value in self.q.items()}
        r = {name: value.float() * 1e7 for name, value in self.r.items()}
        merger = WeightMerge(self.model, q, r)
        before = copy.deepcopy(self.model.state_dict())
        with self.assertRaises(FloatingPointError):
            merger.apply(1)
        merger.restore()
        self.assertTrue(all(torch.equal(value, before[name]) for name, value in self.model.state_dict().items()))

    def test_merge_precision_must_also_preserve_double_baseline(self):
        with self.assertRaisesRegex(ValueError, "baseline precision"):
            WeightMerge(self.model, {name: q.float() for name, q in self.q.items()},
                        {name: r.float() for name, r in self.r.items()}, compute_dtype="float32")

    def test_target_weight_mutation_during_equivalence_is_detected(self):
        from MDLM import merge
        original, calls = merge.capture_outputs, 0

        def mutate(*args):
            nonlocal calls
            result = original(*args)
            calls += 1
            if calls == 2:
                with torch.no_grad():
                    self.model[0].weight.add_(1e-7)
            return result

        merger = self.merger()
        with patch.object(merge, "capture_outputs", side_effect=mutate):
            with self.assertRaisesRegex(RuntimeError, "integrity"):
                merge_with_equivalence(merger, self.adapter, self.q, self.r, [(self.input, (0,))], 1,
                                       atol=1e-4, rtol=1e-4)
        merger.restore()

    def test_all_layers_require_the_same_nonempty_fact_dimension(self):
        q = {**self.q, "2": torch.ones(3, 3, dtype=torch.float64)}
        r = {**self.r, "2": torch.ones(3, 3, dtype=torch.float64)}
        with self.assertRaisesRegex(ValueError, "fact dimension"):
            WeightMerge(self.model, q, r, compute_dtype="float64")


if __name__ == "__main__":
    unittest.main()
