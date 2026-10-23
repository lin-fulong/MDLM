"""Formula (5) checks on artificial CPU activations; no language model is loaded."""

from __future__ import annotations

import json
import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from MDLM.keep_statistics import SecondMomentAccumulator


@unittest.skipIf(torch is None, "PyTorch is required for keep-statistics tests")
class SecondMomentTests(unittest.TestCase):
    def test_uncentered_unequal_weights_and_multiple_calls(self):
        values = torch.tensor([[2., 3., 4.], [5., 6., 7.], [2., 8., 3.]], dtype=torch.float64)
        weights = [0.5, 2.0, 3.0]
        accumulator = SecondMomentAccumulator(3)
        accumulator.add(values[:2], weights[:2])
        accumulator.add(values[2:], weights[2:])
        expected = sum(weight * torch.outer(row, row) for row, weight in zip(values, weights))
        torch.testing.assert_close(accumulator.matrix, expected, rtol=1e-14, atol=1e-14)
        report = accumulator.diagnose(accumulator.matrix, chunk_size=1)
        self.assertTrue(report["all_checks_passed"])
        self.assertEqual(accumulator.vector_count, 3)
        self.assertEqual(accumulator.positive_weight_count, 3)
        self.assertEqual(accumulator.weight_sum, 5.5)
        self.assertAlmostEqual(accumulator.expected_trace, float(expected.trace()), places=10)
        self.assertFalse(report["full_spectrum_checked"])

    def test_each_layer_keeps_its_own_input_width(self):
        for width in (2, 5):
            with self.subTest(width=width):
                accumulator = SecondMomentAccumulator(width)
                values = torch.arange(1, 1 + 2 * width, dtype=torch.float64).reshape(2, width)
                accumulator.add(values, [2.0, 0.25])
                self.assertEqual(accumulator.matrix.shape, (width, width))
                self.assertTrue(accumulator.diagnose(accumulator.matrix)["all_checks_passed"])

    def test_zero_weights_and_rank_deficient_matrix_are_legal(self):
        accumulator = SecondMomentAccumulator(5)
        accumulator.add(torch.tensor([[1., 2., 3., 4., 5.], [99., 98., 97., 96., 95.]]), [2.0, 0.0])
        self.assertEqual(accumulator.positive_weight_count, 1)
        self.assertTrue(accumulator.diagnose(accumulator.matrix)["all_checks_passed"])
        zeros = SecondMomentAccumulator(3)
        zeros.add(torch.ones(2, 3), [0.0, 0.0])
        self.assertTrue(zeros.diagnose(zeros.matrix)["all_checks_passed"])

    def test_zero_weight_does_not_overflow_during_float32_conversion(self):
        accumulator = SecondMomentAccumulator(2, accumulation_dtype="float32")
        accumulator.add(torch.tensor([[1e300, 1e300], [2., 3.]], dtype=torch.float64), [0.0, 2.0])
        torch.testing.assert_close(accumulator.matrix, torch.tensor([[8., 12.], [12., 18.]]))
        self.assertEqual(accumulator.vector_count, 2)
        self.assertTrue(accumulator.diagnose(accumulator.matrix)["all_checks_passed"])

    def test_invalid_numeric_inputs_are_rejected_before_accumulation(self):
        accumulator = SecondMomentAccumulator(2)
        invalid = ((torch.ones(1, 2), [-1.0]), (torch.ones(1, 2), [float("nan")]),
                   (torch.ones(1, 2), [float("inf")]),
                   (torch.tensor([[float("nan"), 1.]]), [1.0]),
                   (torch.tensor([[1., float("inf")]]), [1.0]))
        for values, weights in invalid:
            with self.subTest(values=values, weights=weights):
                with self.assertRaises(ValueError):
                    accumulator.add(values, weights)
        self.assertEqual(accumulator.vector_count, 0)
        self.assertEqual(float(accumulator.matrix.sum()), 0.0)

    def test_required_tensor_shapes(self):
        accumulator = SecondMomentAccumulator(2)
        for values, weights in ((torch.ones(3), [1.]), (torch.ones(1, 3), [1.]),
                                (torch.ones(2, 2), [1.]), (torch.ones(1, 2), [[1.]])):
            with self.subTest(shape=values.shape, weights=weights):
                with self.assertRaises(ValueError):
                    accumulator.add(values, weights)
        with self.assertRaises(ValueError):
            accumulator.diagnose(torch.zeros(3, 3))

    def test_trace_penalty_identity(self):
        generator = torch.Generator().manual_seed(25)
        values = torch.randn(6, 4, generator=generator, dtype=torch.float64) + 2.0
        delta = torch.randn(3, 4, generator=generator, dtype=torch.float64)
        weights = [0.0, 0.25, 2.0, 4.0, 0.5, 1.0]
        accumulator = SecondMomentAccumulator(4)
        accumulator.add(values, weights)
        expected = sum(weight * float((delta @ row).square().sum()) for row, weight in zip(values, weights))
        self.assertAlmostEqual(float(torch.trace(delta @ accumulator.matrix @ delta.T)), expected, places=10)

    def test_modified_matrix_is_rejected(self):
        accumulator = SecondMomentAccumulator(5)
        accumulator.add(torch.arange(1., 16.).reshape(3, 5), [0.5, 2.0, 3.0])
        for kind in ("asymmetry", "scale", "negative", "nonfinite"):
            with self.subTest(kind=kind):
                matrix = accumulator.matrix.clone()
                if kind == "asymmetry":
                    matrix[0, 1] += 10.0
                elif kind == "scale":
                    matrix *= 2.0
                elif kind == "negative":
                    matrix *= -1.0
                else:
                    matrix[0, 1] = float("nan")
                self.assertFalse(accumulator.diagnose(matrix, chunk_size=2)["all_checks_passed"])

    def test_float32_accumulation_and_storage_roundoff(self):
        generator = torch.Generator().manual_seed(31)
        values = torch.randn(12, 7, generator=generator, dtype=torch.float64) + 1.0
        weights = [0.2, 0.7, 1.1] * 4
        for dtype in ("float32", "float64"):
            with self.subTest(dtype=dtype):
                accumulator = SecondMomentAccumulator(7, accumulation_dtype=dtype)
                accumulator.add(values, weights)
                report = accumulator.diagnose(accumulator.matrix.float(), chunk_size=3)
                self.assertTrue(report["all_checks_passed"])
                self.assertEqual(report["dtype"], "float32")

    def test_large_finite_second_moment_does_not_overflow_reference_checks(self):
        accumulator = SecondMomentAccumulator(3)
        accumulator.add(torch.full((1, 3), 1e200, dtype=torch.float64), [1e-100])
        self.assertTrue(torch.isfinite(accumulator.matrix).all())
        self.assertAlmostEqual(accumulator.expected_trace / 3e300, 1.0, places=12)
        self.assertTrue(accumulator.diagnose(accumulator.matrix)["all_checks_passed"])

    def test_small_finite_second_moment_is_not_lost_before_weighting(self):
        accumulator = SecondMomentAccumulator(3)
        accumulator.add(torch.full((1, 3), 1e-200, dtype=torch.float64), [1e300])
        self.assertGreater(accumulator.expected_trace, 0.0)
        self.assertAlmostEqual(accumulator.expected_trace / 3e-100, 1.0, places=12)
        self.assertTrue(accumulator.diagnose(accumulator.matrix, atol=0.0)["all_checks_passed"])
        self.assertFalse(accumulator.diagnose(torch.zeros_like(accumulator.matrix), atol=0.0)["all_checks_passed"])

    def test_weighted_vector_prevents_float32_weight_underflow(self):
        accumulator = SecondMomentAccumulator(2, accumulation_dtype="float32")
        accumulator.add(torch.full((1, 2), 1e25, dtype=torch.float64), [1e-50])
        torch.testing.assert_close(accumulator.matrix, torch.ones(2, 2))
        self.assertEqual(accumulator.weight_sum, 1e-50)
        self.assertTrue(accumulator.diagnose(accumulator.matrix)["all_checks_passed"])

    def test_float32_storage_near_zero_projection_uses_matrix_scale(self):
        for orthogonal_count in (1, 8):
            with self.subTest(orthogonal_count=orthogonal_count):
                accumulator = SecondMomentAccumulator(32, projection_count=8)
                generator = torch.Generator().manual_seed(42)
                vector = torch.randn(32, generator=generator, dtype=torch.float64)
                basis = accumulator.projection_vectors[:, :orthogonal_count]
                vector = (vector - basis @ (basis.T @ vector)) * 1000
                accumulator.add(vector[None, :], [1.0])
                report = accumulator.diagnose(accumulator.matrix.float())
                self.assertTrue(report["all_checks_passed"])
                self.assertEqual(report["projection_reference_scale"], accumulator.expected_trace)
                self.assertLess(report["projection_normalized_error"], report["rtol"])

    def test_direct_check_preserves_small_coordinate_scale(self):
        accumulator = SecondMomentAccumulator(3)
        accumulator.add(torch.tensor([[1e8, 1e8, 1e-8], [1e8, -1e8, 1e-8]], dtype=torch.float64), [1., 1.])
        rounded = accumulator.matrix.clone()
        rounded[0, 1] = rounded[1, 0] = 0.01
        self.assertTrue(accumulator.diagnose(rounded, atol=0.0)["all_checks_passed"])
        rounded[2, 2] += 1e-8
        report = accumulator.diagnose(rounded, atol=0.0)
        self.assertFalse(report["direct_outer_product_matches"])
        self.assertFalse(report["all_checks_passed"])

    def test_float32_batch_size_changes_stay_within_frozen_tolerances(self):
        generator = torch.Generator().manual_seed(32)
        values = torch.randn(3000, 32, generator=generator, dtype=torch.float64) * 1000
        weights = torch.linspace(0.1, 2, len(values)).tolist()
        for batch_size in (1, len(values)):
            with self.subTest(batch_size=batch_size):
                accumulator = SecondMomentAccumulator(32, accumulation_dtype="float32")
                for start in range(0, len(values), batch_size):
                    accumulator.add(values[start:start + batch_size], weights[start:start + batch_size])
                self.assertTrue(accumulator.diagnose(accumulator.matrix)["all_checks_passed"])

    def test_single_thread_batch_preserves_exact_outer_product_symmetry(self):
        threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, threads)
        torch.set_num_threads(1)
        generator = torch.Generator().manual_seed(32)
        values = torch.randn(3000, 32, generator=generator, dtype=torch.float64) * 1000
        weights = torch.linspace(0.1, 2, len(values)).tolist()
        accumulator = SecondMomentAccumulator(32, accumulation_dtype="float32")
        accumulator.add(values, weights)
        self.assertTrue(torch.equal(accumulator.matrix, accumulator.matrix.T))
        self.assertTrue(accumulator.diagnose(accumulator.matrix)["all_checks_passed"])

    def test_symmetry_copy_handles_more_than_one_row_block(self):
        generator = torch.Generator().manual_seed(8)
        values = torch.randn(7, 270, generator=generator, dtype=torch.float64)
        accumulator = SecondMomentAccumulator(270)
        accumulator.add(values[:3], [0.5, 0.0, 2.0])
        accumulator.add(values[3:], [0.1, 0.3, 0.4, 0.8])
        expected = sum(weight * torch.outer(row, row) for row, weight in zip(
            values, [0.5, 0.0, 2.0, 0.1, 0.3, 0.4, 0.8]))
        torch.testing.assert_close(accumulator.matrix, expected, rtol=1e-12, atol=1e-12)
        self.assertTrue(torch.equal(accumulator.matrix, accumulator.matrix.T))
        self.assertTrue(accumulator.diagnose(accumulator.matrix)["all_checks_passed"])

    def test_extreme_finite_corruption_produces_serializable_failure_report(self):
        accumulator = SecondMomentAccumulator(1)
        accumulator.add(torch.tensor([[1e204]], dtype=torch.float64), [1e-100])
        report = accumulator.diagnose(-accumulator.matrix)
        self.assertFalse(report["all_checks_passed"])
        json.dumps(report, allow_nan=False)
        asymmetric = SecondMomentAccumulator(2)
        asymmetric.add(torch.tensor([[1e154, 0.]], dtype=torch.float64), [1.0])
        matrix = asymmetric.matrix.clone()
        matrix[0, 1], matrix[1, 0] = 1e308, -1e308
        report = asymmetric.diagnose(matrix)
        self.assertFalse(report["all_checks_passed"])
        json.dumps(report, allow_nan=False)

    def test_configuration_and_diagnostic_tolerances(self):
        for kwargs in ({"width": 0}, {"width": 2, "accumulation_dtype": "float16"},
                       {"width": 2, "projection_count": 0}, {"width": 2, "direct_block_size": 0},
                       {"width": 2, "seed": -1}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    SecondMomentAccumulator(**kwargs)
        accumulator = SecondMomentAccumulator(2)
        for kwargs in ({"atol": -1.}, {"rtol": float("nan")}, {"chunk_size": 0}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    accumulator.diagnose(accumulator.matrix, **kwargs)


if __name__ == "__main__":
    unittest.main()
