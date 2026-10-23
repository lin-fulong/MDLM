"""Independent formula (6)--(9) checks using small CPU matrices."""

from __future__ import annotations

import unittest
import json
import weakref
from dataclasses import replace
from unittest.mock import patch

import torch

from MDLM.contracts import WritebackConfig
from MDLM.writeback import WritebackSolveError, relative_residual, solve_writeback_factor


class WritebackMathTests(unittest.TestCase):
    def setUp(self):
        self.settings = WritebackConfig({"layer": 0.3}, storage_dtype="float64", chunk_size=2)
        self.k = torch.tensor([[1.0, 0.5], [-0.3, 2.0], [1.2, -0.4]], dtype=torch.float64)
        vectors = torch.tensor([[1.0, 0.0, 0.2], [-0.2, 1.0, 0.0]], dtype=torch.float64)
        self.c = vectors.T @ vectors

    def solve(self, k=None, c=None, ridge=0.3, settings=None):
        return solve_writeback_factor(self.k if k is None else k, self.c if c is None else c,
                                      ridge, settings or self.settings)

    def test_solve_matches_independent_system_without_inverse_or_input_mutation(self):
        before_k, before_c = self.k.clone(), self.c.clone()
        h = self.c + self.k @ self.k.T + 0.3 * torch.eye(3, dtype=torch.float64)
        expected = torch.linalg.solve(h, self.k).T
        with patch.object(torch.linalg, "inv", side_effect=AssertionError("explicit inverse is forbidden")), \
                patch.object(torch, "inverse", side_effect=AssertionError("explicit inverse is forbidden")):
            q, report = self.solve()
        torch.testing.assert_close(q, expected, atol=1e-12, rtol=1e-12)
        self.assertTrue(torch.equal(self.k, before_k) and torch.equal(self.c, before_c))
        self.assertLess(report["relative_residual"], 1e-12)
        self.assertEqual(report["ridge"], 0.3)
        self.assertFalse(q.requires_grad)

    def test_rq_satisfies_normal_equation_and_unique_quadratic_minimum(self):
        q, _ = self.solve()
        r = torch.tensor([[0.4, -0.7], [1.1, 0.2]], dtype=torch.float64)
        d = r @ q
        h = self.c + self.k @ self.k.T + 0.3 * torch.eye(3, dtype=torch.float64)
        torch.testing.assert_close(d @ h, r @ self.k.T, atol=1e-12, rtol=1e-12)

        def objective(value):
            return ((value @ self.k - r).square().sum() + torch.trace(value @ self.c @ value.T)
                    + 0.3 * value.square().sum()) / 2

        perturbation = torch.tensor([[0.1, 0.3, -0.2], [-0.4, 0.1, 0.2]], dtype=torch.float64)
        difference = objective(d + perturbation) - objective(d)
        torch.testing.assert_close(difference, torch.trace(perturbation @ h @ perturbation.T) / 2)
        self.assertGreater(float(difference), 0)

    def test_soft_fit_does_not_require_qk_identity(self):
        q, _ = self.solve(k=torch.tensor([[1.0], [0.0]], dtype=torch.float64),
                          c=torch.diag(torch.tensor([2.0, 3.0], dtype=torch.float64)), ridge=0.5)
        torch.testing.assert_close(q, torch.tensor([[1 / 3.5, 0.0]], dtype=torch.float64))

    def test_zero_k_and_low_rank_c_are_legal(self):
        q, report = self.solve(k=torch.zeros_like(self.k))
        self.assertTrue(torch.equal(q, torch.zeros_like(q)))
        self.assertEqual(report["relative_residual"], 0.0)
        q, _ = self.solve(c=torch.zeros_like(self.c))
        self.assertTrue(torch.isfinite(q).all())

    def test_fact_permutation_only_permutes_q_rows(self):
        q, _ = self.solve()
        permuted, _ = self.solve(k=self.k[:, [1, 0]])
        torch.testing.assert_close(permuted, q[[1, 0]])

    def test_autograd_inputs_produce_fixed_detached_factors(self):
        q, _ = self.solve(k=self.k.clone().requires_grad_(), c=self.c.clone().requires_grad_())
        self.assertFalse(q.requires_grad)
        self.assertIsNone(q.grad_fn)

    def test_non_spd_h_fails_without_changing_ridge(self):
        with self.assertRaises(WritebackSolveError) as caught:
            self.solve(c=-100 * torch.eye(3, dtype=torch.float64))
        self.assertEqual(caught.exception.diagnostics["phase"], "cholesky")
        self.assertEqual(caught.exception.diagnostics["ridge"], 0.3)
        self.assertGreater(caught.exception.diagnostics["cholesky_info"], 0)

    def test_asymmetric_nonfinite_and_invalid_shapes_are_rejected(self):
        asymmetric = self.c.clone()
        asymmetric[0, 1] += 1
        for k, c in ((self.k, asymmetric), (self.k * float("nan"), self.c),
                     (self.k, self.c * float("inf")), (self.k, torch.eye(2)),
                     (torch.zeros((3, 0)), self.c), (self.k.to(torch.complex128), self.c)):
            with self.subTest(shape=k.shape), self.assertRaises(WritebackSolveError):
                self.solve(k=k, c=c)

    def test_invalid_ridge_and_configuration_are_rejected(self):
        for ridge in (0, -1, float("inf"), float("nan"), True):
            with self.subTest(ridge=ridge), self.assertRaises(WritebackSolveError):
                self.solve(ridge=ridge)
        for values in ({"ridge_by_layer": {}}, {"ridge_by_layer": {"layer": 0}},
                       {"ridge_by_layer": "invalid"}, {"device": None},
                       {"compute_dtype": "bfloat16"}, {"failure_policy": "increase_ridge"},
                       {"residual_epsilon": 0}, {"chunk_size": 0}, {"device": "invalid"}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                replace(self.settings, **values).check()

    def test_failure_diagnostics_are_strict_json(self):
        with self.assertRaises(WritebackSolveError) as caught:
            self.solve(ridge=float("nan"))
        json.dumps(caught.exception.diagnostics, allow_nan=False)

    def test_compute_and_storage_residual_thresholds_are_enforced(self):
        with patch("MDLM.writeback.relative_residual", return_value=1.0):
            with self.assertRaisesRegex(WritebackSolveError, "linear solve residual"):
                self.solve()
        with patch("MDLM.writeback.relative_residual", side_effect=[0.0, 0.0, 1.0]):
            with self.assertRaisesRegex(WritebackSolveError, "stored Q residual"):
                self.solve()

    def test_nearly_symmetric_c_must_also_satisfy_transposed_normal_equation(self):
        k = torch.tensor([[1e-4], [0.0]], dtype=torch.float64)
        c = torch.tensor([[1e-8, 1e-8], [0.0, 1e-8]], dtype=torch.float64)
        # 该 C 在对称容差内，且 HX=K 可通过，但 QH=Kᵀ 的相对误差约为 0.5。
        with self.assertRaisesRegex(WritebackSolveError, "normal equation residual"):
            self.solve(k=k, c=c, ridge=1e-12)

    def test_tiny_rhs_cannot_publish_zero_float32_q_after_norm_underflow(self):
        settings = replace(self.settings, storage_dtype="float32", residual_epsilon=1e-300)
        with self.assertRaisesRegex(WritebackSolveError, "stored Q residual"):
            self.solve(k=torch.full((2, 1), 1e-200, dtype=torch.float64),
                       c=torch.eye(2, dtype=torch.float64), ridge=1.0, settings=settings)

    def test_relative_residual_uses_stable_norm_at_extreme_scales(self):
        for scale in (1e-200, 1.0, 1e154):
            k = torch.full((2, 1), scale, dtype=torch.float64)
            error = relative_residual(k, torch.eye(2, dtype=torch.float64), torch.zeros_like(k),
                                      1.0, epsilon=1e-300, chunk_size=1)
            with self.subTest(scale=scale):
                self.assertAlmostEqual(error, 1.0, places=14)

    def test_system_storage_is_released_before_triangular_solve(self):
        original_factor = torch.linalg.cholesky_ex
        original_solve = torch.cholesky_solve
        reference = None

        def factor(system, **kwargs):
            nonlocal reference
            reference = weakref.ref(system)
            return original_factor(system, **kwargs)

        def solve(*args, **kwargs):
            self.assertIsNone(reference(), "a row view must not retain the dense H allocation")
            return original_solve(*args, **kwargs)

        with patch.object(torch.linalg, "cholesky_ex", new=factor), \
                patch.object(torch, "cholesky_solve", new=solve):
            self.solve()

    def test_compute_and_stored_normal_equation_thresholds_are_enforced(self):
        with patch("MDLM.writeback.relative_residual", side_effect=[0.0, 1.0]):
            with self.assertRaisesRegex(WritebackSolveError, "normal equation residual"):
                self.solve()
        with patch("MDLM.writeback.relative_residual", side_effect=[0.0, 0.0, 0.0, 1.0]):
            with self.assertRaisesRegex(WritebackSolveError, "stored Q normal equation residual"):
                self.solve()

    def test_storage_overflow_and_unrepresentable_ridge_are_rejected(self):
        with self.assertRaisesRegex(WritebackSolveError, "stored Q contains nonfinite"):
            self.solve(k=torch.tensor([[1e-40]], dtype=torch.float64),
                       c=torch.zeros((1, 1), dtype=torch.float64), ridge=1e-80,
                       settings=replace(self.settings, storage_dtype="float32"))
        with self.assertRaisesRegex(WritebackSolveError, "ridge is not representable"):
            self.solve(c=1e30 * torch.eye(3, dtype=torch.float64))

    def test_residual_matches_full_equation_in_float32_and_float64(self):
        for dtype in ("float32", "float64"):
            settings = replace(self.settings, compute_dtype=dtype, storage_dtype="float32",
                               residual_tolerance=1e-5)
            q, report = self.solve(settings=settings)
            h = self.c + self.k @ self.k.T + 0.3 * torch.eye(3, dtype=torch.float64)
            expected = float((h @ q.T.double() - self.k).norm() / self.k.norm())
            actual = relative_residual(self.k, self.c, q.T, 0.3, epsilon=1e-12, chunk_size=1)
            self.assertAlmostEqual(actual, expected, places=14)
            self.assertLess(report["stored_relative_residual"], 1e-5)


if __name__ == "__main__":
    unittest.main()
