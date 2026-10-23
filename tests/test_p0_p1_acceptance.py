"""Acceptance-gate logic on constructed records; no 8B model is loaded."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from MDLM.states import FactTrajectory, build_supervision, save_supervision, save_trajectories
from model.base import DenoiseTrace, MaskedEditInput
from util.accept_p0_p1_single_case import check_p0, check_p1


class SingleCaseAcceptanceTests(unittest.TestCase):
    def test_p0_requires_real_ff_out_capture_and_replay(self):
        row = {
            "case_id": "constructed_1", "probe_id": "edit_1", "purpose": "example_only",
            "real_forward": True, "replay_within_tolerance": True,
            "full_mask_input": True, "mask_count": 2, "target_token_count": 2,
            "target_modules": {"model.transformer.blocks.0.ff_out": {"in_features": 8}},
            "ffn_input_shapes": {"model.transformer.blocks.0.ff_out": [8]},
            "mask_logits_shape": [2, 100],
        }
        result = check_p0([row], case_id="constructed_1", probe_id="edit_1", layer=0)
        self.assertTrue(result["real_forward_replayed"])
        row["ffn_input_shapes"]["model.transformer.blocks.0.ff_out"] = [7]
        with self.assertRaisesRegex(ValueError, "dimension"):
            check_p0([row], case_id="constructed_1", probe_id="edit_1", layer=0)

    def test_p1_checks_saved_trajectory_replay_and_position_weights(self):
        initial = MaskedEditInput((7, 99, 99), (1, 1, 1), (1, 2), 0)
        second = MaskedEditInput((7, 11, 99), (1, 1, 1), (2,), 0)
        terminal = MaskedEditInput((7, 11, 12), (1, 1, 1), (), 0)
        trace = DenoiseTrace(
            (initial, second), ("a" * 64, "b" * 64),
            (((1, 11),), ((2, 12),)), terminal,
        )
        trajectory = FactTrajectory("constructed_1", "edit_1", (1, 2), (11, 12), 1, trace)
        fixed, coverage = build_supervision([trajectory], 99)
        coverage.update({"status": "complete", "replayed_fact_count": 1,
                         "logits_replayed_fact_count": 1})
        collection = {"failure": None, "completed": [{
            "case_id": "constructed_1", "probe_id": "edit_1", "purpose": "example_only",
            "forward_logits_finite": True, "replayed": True, "logits_replayed": True,
        }]}
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            save_trajectories(run_dir / "trajectories.jsonl", [trajectory])
            save_supervision(run_dir / "fixed_states.jsonl", fixed)
            (run_dir / "coverage_report.json").write_text(json.dumps(coverage), encoding="utf-8")
            (run_dir / "collection_report.json").write_text(json.dumps(collection), encoding="utf-8")
            result = check_p1(run_dir, case_id="constructed_1", probe_id="edit_1", purpose="example_only")
            self.assertAlmostEqual(result["weight_sum"], 1.0)
            self.assertEqual(result["compatible_state_count"], 2)

            fixed[0]["states"][1]["supervision"][0]["weight"] = 0.25
            save_supervision(run_dir / "fixed_states.jsonl", fixed)
            with self.assertRaisesRegex(ValueError, "weight"):
                check_p1(run_dir, case_id="constructed_1", probe_id="edit_1", purpose="example_only")

            save_supervision(run_dir / "fixed_states.jsonl", build_supervision([trajectory], 99)[0])
            collection["completed"][0]["logits_replayed"] = False
            (run_dir / "collection_report.json").write_text(json.dumps(collection), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "evidence"):
                check_p1(run_dir, case_id="constructed_1", probe_id="edit_1", purpose="example_only")


if __name__ == "__main__":
    unittest.main()
