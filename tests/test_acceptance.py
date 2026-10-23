"""P5 gates, original references and full recovery on a real two-token CPU model."""

from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from MDLM import acceptance
from MDLM.acceptance import AcceptanceError, cumulative_change, development_edits, gate_decision, select_merged_candidate
from MDLM.contracts import AcceptanceConfig, FactSelection
from MDLM.merge import EquivalenceError, WeightMerge
from MDLM.optimize import KeepTrainingExample
from MDLM.states import EditTrainingExample
from model.base import KeepModelInput, MaskedEditInput


class Adapter:
    def forward_logits(self, model, model_input, positions):
        x = torch.nn.functional.one_hot(torch.tensor(model_input.input_ids), 2).double()
        return model(x)[list(positions)]


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.model = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False)).double().eval()
        with torch.no_grad():
            self.model[0].weight.copy_(torch.eye(2) / 2)
        self.q = {"0": torch.eye(2, dtype=torch.float64)}
        self.r = {"0": torch.tensor([[0., 2.], [0., -2.]], dtype=torch.float64)}
        self.adapter = Adapter()
        self.originals = {"0": self.model[0].weight.detach().clone()}
        self.merger = WeightMerge(self.model, self.q, self.r, compute_dtype="float64", chunk_size=1)
        inp = MaskedEditInput((1,), (1,), (0,), 0)
        self.edits = [EditTrainingExample("constructed", inp, (0,), (0,), (1.,))]
        with torch.no_grad():
            logs = torch.log_softmax(self.adapter.forward_logits(self.model, inp, (0,)), -1)
        self.keeps = [KeepTrainingExample(inp, (0,), (1.,), logs)]
        self.settings = AcceptanceConfig("dev.json", (FactSelection("a", "p"),), "keep.json", "original", "sha", "tok",
                    alphas=(3., 1., .25), delta=.01, tau_g=10, tau_u=5, compute_dtype="float64",
                    equivalence_atol=1e-12, equivalence_rtol=1e-12)

    def run_selection(self, settings=None):
        return select_merged_candidate(self.merger, self.adapter, self.q, self.r, self.edits, self.keeps,
                                        self.originals, settings or self.settings)

    def test_first_passing_alpha_is_selected_in_configured_order(self):
        report = self.run_selection()
        self.assertEqual(report["status"], "accepted")
        self.assertEqual(report["selected_alpha"], 1)
        self.assertEqual(len(report["trials"]), 2)
        self.assertFalse(report["trials"][0]["gate"]["checks"]["u"])
        self.assertTrue(report["trials"][0]["baseline_restored"])
        torch.testing.assert_close(self.model[0].weight, self.originals["0"] + self.r["0"])

    def test_all_rejected_recovers_baseline_bit_for_bit(self):
        report = self.run_selection(replace(self.settings, delta=100))
        self.assertEqual(report["status"], "rejected")
        self.assertIsNone(report["selected_alpha"])
        self.assertTrue(report["baseline_restored"])
        self.assertTrue(torch.equal(self.model[0].weight, self.originals["0"]))
        self.assertEqual(len(report["trials"]), 3)

    def test_nonfinite_scaled_trial_is_rejected_then_next_alpha_is_tried(self):
        report = self.run_selection(replace(self.settings, alphas=(1e308, 1.)))
        self.assertEqual(report["selected_alpha"], 1)
        self.assertIn("nonfinite", report["trials"][0]["reasons"][0])
        self.assertTrue(report["trials"][0]["baseline_restored"])

    def test_each_gate_fails_independently_and_boundary_is_inclusive(self):
        settings = replace(self.settings, delta=.1, tau_g=.2, tau_u=.3)
        good = {"f_dev": .9, "g_dev": .2, "u": .3}
        self.assertTrue(gate_decision(good, 1., settings)["passed"])
        for name in good:
            for bad in (100, float("nan"), float("inf")):
                decision = gate_decision({**good, name: bad}, 1., settings)
                self.assertFalse(decision["passed"])
                self.assertFalse(decision["checks"][name])
                self.assertEqual(len(decision["reasons"]), 1)

    def test_u_uses_original_net_change_and_actual_stored_weights(self):
        # 本轮基准已偏离最初权重；不能用本轮 delta 或路径长度替代净差。
        originals = {"0": self.originals["0"] + self.r["0"]}
        self.merger.apply(1)
        self.assertEqual(cumulative_change(self.merger, originals, 1e-12), 0)
        self.merger.apply(.5)
        expected = float((self.model[0].weight.detach() - originals["0"]).norm() / originals["0"].norm())
        self.assertAlmostEqual(cumulative_change(self.merger, originals, 1e-12), expected, places=12)

    def test_u_zero_denominator_and_extreme_finite_norms(self):
        zeros = {"0": torch.zeros_like(self.originals["0"])}
        self.assertAlmostEqual(cumulative_change(self.merger, zeros, .5), float(self.originals["0"].norm()) / .5)
        with torch.no_grad():
            self.model[0].weight.mul_(1e200)
        original = {"0": self.originals["0"] * 5e199}
        self.assertAlmostEqual(cumulative_change(self.merger, original, 1e-12), 1)

    def test_preservation_reference_is_theta0_full_vocabulary(self):
        reference = torch.log_softmax(torch.tensor([[1., -2.]], dtype=torch.float64), -1)
        self.keeps = [KeepTrainingExample(self.keeps[0].model_input, (0,), (1.,), reference)]
        metrics = acceptance.development_metrics(self.merger, self.adapter, self.edits, self.keeps, self.originals, 1e-12)
        candidate_logs = torch.log_softmax(self.adapter.forward_logits(self.model, self.keeps[0].model_input, (0,)), -1)
        expected = float((reference.exp() * (reference - candidate_logs)).sum())
        self.assertAlmostEqual(metrics["g_dev"], expected, places=12)
        self.assertGreater(metrics["g_dev"], 0)

    def test_missing_development_preservation_never_skips_gate(self):
        self.keeps = []
        with self.assertRaisesRegex(AcceptanceError, "requires both"):
            self.run_selection()
        self.assertTrue(torch.equal(self.model[0].weight, self.originals["0"]))

    def test_equivalence_failure_rejects_trial_and_recovers(self):
        with patch.object(acceptance, "merge_with_equivalence", side_effect=EquivalenceError("constructed mismatch")):
            report = self.run_selection()
        self.assertEqual(report["status"], "rejected")
        self.assertTrue(all("constructed mismatch" in row["reasons"][0] for row in report["trials"]))
        self.assertTrue(torch.equal(self.model[0].weight, self.originals["0"]))

    def test_runtime_exception_after_weight_write_recovers_baseline(self):
        original = acceptance.development_metrics
        calls = 0

        def fail(*args):
            nonlocal calls
            calls += 1
            if calls > 1:
                raise RuntimeError("constructed evaluation failure")
            return original(*args)

        with patch.object(acceptance, "development_metrics", side_effect=fail):
            with self.assertRaisesRegex(AcceptanceError, "evaluation failure") as caught:
                self.run_selection()
        self.assertTrue(caught.exception.report["baseline_restored"])
        self.assertTrue(torch.equal(self.model[0].weight, self.originals["0"]))
        self.assertTrue(all(not module._forward_hooks for module in self.model.modules()))

    def test_development_normalization_is_by_fact_probe_then_token(self):
        class PreparedAdapter:
            def prepare_development_input(self, case, probe):
                count = 2 if probe == "long" else 1
                inp = MaskedEditInput((1,) * count, (1,) * count, tuple(range(count)), 0)
                return inp, SimpleNamespace(token_ids=(0,) * count)

        selections = (FactSelection("a", "long"), FactSelection("a", "short"), FactSelection("b", "short"))
        examples = development_edits([{"case_id": "a"}, {"case_id": "b"}], selections, PreparedAdapter())
        self.assertEqual([item.weights for item in examples], [(.125, .125), (.25,), (.5,)])

    def test_configuration_requires_positive_delta_epsilon_and_finite_alpha(self):
        self.assertEqual(AcceptanceConfig.from_record(self.settings.as_record()), self.settings)
        for kwargs in ({"alphas": ()}, {"alphas": (0,)}, {"alphas": (float("inf"),)}, {"delta": 0},
                       {"epsilon": 0}, {"tau_g": -1}, {"edit_protocol": "unfrozen"}):
            with self.assertRaises(ValueError):
                replace(self.settings, **kwargs).check()

    def test_u_finite_ratio_survives_unrepresentable_denominator_norm(self):
        with torch.no_grad():
            self.model[0].weight.fill_(1.5e308)
        originals = {"0": torch.full((2, 2), 1e308, dtype=torch.float64)}
        self.assertAlmostEqual(cumulative_change(self.merger, originals, 1e-12), .5)

    def test_u_scales_before_opposite_sign_weight_subtraction(self):
        with torch.no_grad():
            self.model[0].weight.fill_(1e308)
        originals = {"0": torch.full((2, 2), -1e308, dtype=torch.float64)}
        self.assertAlmostEqual(cumulative_change(self.merger, originals, 1e-12), 2.)

    def test_recovery_error_keeps_original_failure_and_truthful_report(self):
        restore = self.merger.restore
        calls = 0

        def broken_restore():
            nonlocal calls
            calls += 1
            if calls > 1:
                raise RuntimeError("constructed recovery failure")
            restore()

        with patch.object(acceptance, "development_metrics", side_effect=RuntimeError("original evaluation error")), \
                patch.object(self.merger, "restore", side_effect=broken_restore):
            with self.assertRaises(AcceptanceError) as caught:
                self.run_selection()
        self.assertIn("original evaluation error", caught.exception.report["failure"])
        self.assertIn("recovery failure", caught.exception.report["recovery_failure"])
        self.assertFalse(caught.exception.report["baseline_restored"])


if __name__ == "__main__":
    unittest.main()
