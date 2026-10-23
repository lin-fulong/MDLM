"""Formula (3)/(10)/(12) loss/gradient and finite-budget regression checks."""

from __future__ import annotations

import copy
from dataclasses import replace
import unittest
from unittest.mock import patch

import torch

from MDLM.candidate import CandidateModel
from MDLM.contracts import ResidualOptimizationConfig
from MDLM.optimize import (KeepTrainingExample, OptimizationError, accumulate_data_gradients,
                           edit_loss, evaluate_objective, keep_loss, optimize_residuals, regularization,
                           residual_diagnostics)
from MDLM.states import EditTrainingExample
from model.base import KeepModelInput, MaskedEditInput


class TinyAdapter:
    def forward_logits(self, model, model_input, positions):
        x = torch.nn.functional.one_hot(torch.tensor(model_input.input_ids), num_classes=3).double()
        return model(x)[list(positions)]


class OptimizationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        self.model = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.Tanh(), torch.nn.Linear(3, 3)).double().eval()
        self.adapter = TinyAdapter()
        self.q = {name: torch.tensor([[1., 0., 0.], [0., 1., 1.]], dtype=torch.float64) for name in ("0", "2")}
        self.settings = ResidualOptimizationConfig({"0": 0.03, "2": 0.02}, learning_rate=0.03,
                         max_steps=6, checkpoints=(1, 3, 6), residual_dtype="float64")
        inp = MaskedEditInput((0, 1, 2), (1, 1, 1), (1, 2), 0)
        # 不等数量状态/监督位置，防止错误的按状态或微批平均也通过测试。
        self.edits = [EditTrainingExample("a", inp, (1, 2), (2, 0), (0.125, 0.125)),
                      EditTrainingExample("a", inp, (2,), (0,), (0.25,)),
                      EditTrainingExample("b", inp, (1,), (2,), (0.5,))]
        self.keep_input = KeepModelInput((0, 2), (1, 1), ())
        with torch.no_grad():
            reference = torch.log_softmax(self.adapter.forward_logits(self.model, self.keep_input, (0, 1)) +
                                          torch.tensor([0.2, -0.3, 0.1]), -1)
        self.keeps = [KeepTrainingExample(self.keep_input, (0, 1), (0.3, 0.7), reference)]

    def keep_settings(self):
        return replace(self.settings, beta=0.4, keep_train_file="explicit_training_inputs.json",
                       reference_model_path="original_theta_0", reference_model_revision="original_revision",
                       reference_tokenizer_revision="frozen_tokenizer")

    def test_edit_weights_match_independent_nested_formula(self):
        rows = [self.adapter.forward_logits(self.model, item.model_input, item.positions) for item in self.edits]
        ce = [torch.nn.functional.cross_entropy(logits, torch.tensor(item.targets), reduction="none")
              for logits, item in zip(rows, self.edits)]
        expected = ((ce[0].mean() + ce[1].mean()) / 2 + ce[2].mean()) / 2
        actual = sum(edit_loss(self.adapter, self.model, item) for item in self.edits)
        torch.testing.assert_close(actual, expected)

    def test_kl_uses_theta0_to_candidate_direction_and_full_vocabulary(self):
        logp = torch.log_softmax(self.adapter.forward_logits(self.model, self.keep_input, (0, 1)), -1)
        reference = self.keeps[0].reference_log_probs
        expected = ((reference.exp() * (reference - logp)).sum(-1) * torch.tensor([0.3, 0.7])).sum()
        actual = keep_loss(self.adapter, self.model, self.keeps[0])
        torch.testing.assert_close(actual, expected, atol=1e-8, rtol=1e-7)
        reverse = ((logp.exp() * (logp - reference)).sum(-1) * torch.tensor([0.3, 0.7])).sum()
        self.assertGreater(float((actual - reverse).abs()), 1e-5)

    def test_microbatch_gradients_equal_independent_full_objective_with_one_regularizer(self):
        results = []
        for microbatch in (1, 2, 8):
            settings = replace(self.keep_settings(), microbatch_size=microbatch)
            with CandidateModel(self.model, self.q, dtype="float64") as candidate:
                with torch.no_grad():
                    for branch in candidate.branches.values():
                        branch.r.fill_(0.1)
                full = sum(edit_loss(self.adapter, self.model, item) for item in self.edits)
                full = full + settings.beta * sum(keep_loss(self.adapter, self.model, item) for item in self.keeps)
                full = full + regularization(candidate, settings)
                independent = torch.autograd.grad(full, candidate.parameters())
                accumulate_data_gradients(candidate, self.adapter, self.edits, self.keeps, settings)
                for parameter, expected in zip(candidate.parameters(), independent):
                    torch.testing.assert_close(parameter.grad, expected, atol=1e-12, rtol=1e-12)
                results.append([parameter.grad.clone() for parameter in candidate.parameters()])
        for values in results[1:]:
            for actual, expected in zip(values, results[0]):
                torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)

    def test_formula12_with_regularization_matches_finite_difference(self):
        with CandidateModel(self.model, self.q, dtype="float64") as candidate:
            with torch.no_grad():
                candidate.branches["0"].r.fill_(0.1)
            accumulate_data_gradients(candidate, self.adapter, self.edits, [], self.settings)
            parameter = candidate.branches["0"].r
            gradient, value = float(parameter.grad[1, 0]), float(parameter[1, 0])
            totals = []
            for delta in (1e-5, -1e-5):
                with torch.no_grad():
                    parameter[1, 0] = value + delta
                totals.append(evaluate_objective(candidate, self.adapter, self.edits, [], self.settings)["total"])
            self.assertAlmostEqual(gradient, (totals[0] - totals[1]) / 2e-5, places=9)

    def test_optimization_improves_j_and_reloads_selected_checkpoint(self):
        weights = copy.deepcopy(self.model.state_dict())
        with CandidateModel(self.model, self.q, dtype="float64") as candidate:
            result = optimize_residuals(candidate, self.adapter, self.edits, [], self.settings)
            self.assertEqual(result.report["status"], "candidate_ready")
            self.assertLess(result.report["selected_loss"]["total"], result.report["baseline"]["total"])
            self.assertIn(result.report["selected_step"], self.settings.checkpoints)
            self.assertTrue(all(parameter.grad is None for parameter in self.model.parameters()))
            self.assertTrue(all(torch.equal(result.residuals[name], value) for name, value in candidate.residuals().items()))
            self.assertTrue(result.optimizer_state["state"])
        self.assertTrue(all(torch.equal(value, weights[name]) for name, value in self.model.state_dict().items()))

    def test_fixed_seed_is_reproducible(self):
        outputs = []
        for _ in range(2):
            with CandidateModel(self.model, self.q, dtype="float64") as candidate:
                outputs.append(optimize_residuals(candidate, self.adapter, self.edits, self.keeps, self.keep_settings()))
        self.assertEqual(outputs[0].report["selected_loss"], outputs[1].report["selected_loss"])
        self.assertTrue(all(torch.equal(outputs[0].residuals[name], value) for name, value in outputs[1].residuals.items()))

    def test_best_checkpoint_is_retained_when_final_checkpoint_worsens(self):
        snapshots = {}
        losses = iter((10.0, 9.0, 8.0, 11.0, 8.0))  # baseline, three checkpoints, selected reload

        def evaluate(*args):
            result = evaluate_objective(*args)
            return {**result, "total": next(losses)}

        with CandidateModel(self.model, self.q, dtype="float64") as candidate:
            def progress(report):
                snapshots[report["history"][-1]["step"]] = candidate.residuals()

            with patch("MDLM.optimize.evaluate_objective", side_effect=evaluate):
                result = optimize_residuals(candidate, self.adapter, self.edits, [], self.settings, on_progress=progress)
            self.assertEqual(result.report["selected_step"], 3)
            self.assertTrue(all(torch.equal(value, snapshots[3][name]) for name, value in result.residuals.items()))
            self.assertTrue(all(float(state["step"]) == 3 for state in result.optimizer_state["state"].values()))

    def test_no_improvement_records_failure_without_residual_artifact(self):
        settings = replace(self.settings, learning_rate=1e-300, max_steps=1, checkpoints=(1,))
        with CandidateModel(self.model, self.q, dtype="float64") as candidate:
            result = optimize_residuals(candidate, self.adapter, self.edits, [], settings)
        self.assertEqual(result.report["status"], "no_candidate")
        self.assertIsNone(result.residuals)

    def test_nonfinite_gradient_stops_with_history_and_failure_phase(self):
        with CandidateModel(self.model, self.q, dtype="float64") as candidate:
            handle = candidate.parameters()[0].register_hook(lambda gradient: gradient * float("nan"))
            try:
                with self.assertRaises(OptimizationError) as failure:
                    optimize_residuals(candidate, self.adapter, self.edits, [], self.settings)
                self.assertEqual(failure.exception.report["phase"], "backward")
                self.assertEqual(failure.exception.report["history"][0]["step"], 0)
            finally:
                handle.remove()

    def test_beta_zero_does_not_call_keep_loss(self):
        with CandidateModel(self.model, self.q, dtype="float64") as candidate, \
                patch("MDLM.optimize.keep_loss", side_effect=AssertionError("disabled")):
            optimize_residuals(candidate, self.adapter, self.edits, [], self.settings)

    def test_frozen_config_rejects_invalid_choices(self):
        for change in ({"rho_by_layer": {"0": 0}}, {"rho_by_layer": {"0": True}}, {"beta": -1},
                       {"beta": 1}, {"learning_rate": 0}, {"max_steps": 0}, {"microbatch_size": True},
                       {"checkpoints": (1, 1, 6)}, {"checkpoints": (1, 2)}, {"optimizer": "adamw"},
                       {"residual_dtype": "bfloat16"}, {"seed": -1}, {"keep_train_file": "unused"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(self.settings, **change).check()

    def test_regularizer_preserves_finite_values_and_gradient_at_extreme_scales(self):
        for value, rho in ((1e-200, 1e300), (1e200, 1e-300)):
            with self.subTest(value=value), CandidateModel(self.model, self.q, dtype="float64") as candidate:
                settings = replace(self.settings, rho_by_layer={name: rho for name in self.q})
                with torch.no_grad():
                    candidate.branches["0"].r[0, 0] = value
                penalty = regularization(candidate, settings)
                expected = 0.5 * (value * rho**0.5)**2
                self.assertAlmostEqual(float(penalty) / expected, 1.0, places=12)
                penalty.backward()
                self.assertAlmostEqual(float(candidate.branches["0"].r.grad[0, 0]) / (rho * value), 1.0, places=12)

    def test_regularizer_applies_half_before_a_large_positive_sum(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 1, bias=False)).double()
        with CandidateModel(model, {"0": torch.eye(2, dtype=torch.float64)}, dtype="float64") as candidate:
            with torch.no_grad():
                candidate.branches["0"].r.fill_(1e154)
            penalty = regularization(candidate, replace(self.settings, rho_by_layer={"0": 1.0}))
            self.assertAlmostEqual(float(penalty) / 1e308, 1.0, places=12)

    def test_float32_regularizer_keeps_a_representable_small_penalty_and_gradient(self):
        model = torch.nn.Sequential(torch.nn.Linear(1, 1, bias=False))
        with CandidateModel(model, {"0": torch.ones(1, 1)}) as candidate:
            with torch.no_grad():
                candidate.branches["0"].r.fill_(1e-30)
            penalty = regularization(candidate, replace(self.settings, rho_by_layer={"0": 1e30}))
            self.assertAlmostEqual(float(penalty) / 5e-31, 1.0, places=6)
            penalty.backward()
            self.assertAlmostEqual(float(candidate.branches["0"].r.grad), 1.0, places=6)

    def test_diagnostic_norms_do_not_turn_nonzero_r_into_zero_or_infinity(self):
        for value in (1e-200, 1e200):
            with self.subTest(value=value), CandidateModel(self.model, self.q, dtype="float64") as candidate:
                with torch.no_grad():
                    candidate.branches["0"].r[0, 0] = value
                row = residual_diagnostics(candidate)["0"]
                self.assertEqual(row["residual_norm"], value)
                self.assertEqual(row["increment_norm"], value)

    def test_data_gradient_disconnection_is_not_hidden_by_regularization(self):
        class Detached(torch.nn.Module):
            def forward(self, inputs):
                return inputs.detach()

        self.model[1] = Detached()
        with CandidateModel(self.model, self.q, dtype="float64") as candidate:
            with self.assertRaisesRegex(OptimizationError, "data gradient.*0"):
                optimize_residuals(candidate, self.adapter, self.edits, [], self.settings)

    def test_core_rejects_factor_width_different_from_fact_count(self):
        wrong = {name: torch.eye(3, dtype=torch.float64) for name in self.q}
        with CandidateModel(self.model, wrong, dtype="float64") as candidate:
            with self.assertRaisesRegex(ValueError, "fixed fact count N"):
                optimize_residuals(candidate, self.adapter, self.edits, [], self.settings)

    def test_core_rejects_residual_dtype_different_from_frozen_config(self):
        with CandidateModel(self.model, self.q, dtype="float64") as candidate:
            with self.assertRaisesRegex(ValueError, "dtype differs"):
                optimize_residuals(candidate, self.adapter, self.edits, [], replace(self.settings, residual_dtype="float32"))


if __name__ == "__main__":
    unittest.main()
