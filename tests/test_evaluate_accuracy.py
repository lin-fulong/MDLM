"""Accuracy scoring and saved-edit selection without loading an 8B model."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from MDLM.contracts import file_sha256
from model.base import tensor_sha256
from util import evaluate_accuracy as accuracy


class AccuracyTests(unittest.TestCase):
    def test_score_uses_generated_answer_and_normalized_exact_match(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return messages[0]["content"] + ":"

            def encode(self, text, **kwargs):
                return [ord(char) for char in text]

            def decode(self, ids, **kwargs):
                return "".join(chr(value) for value in ids)

        class Adapter:
            tokenizer = Tokenizer()
            mask_token_id = 99999

            def sample_denoise(self, model, masked, **kwargs):
                ids = list(masked.input_ids)
                ids[masked.mask_positions[0]] = ord("a")
                return SimpleNamespace(final_input=SimpleNamespace(input_ids=ids))

        result = accuracy.score_probe(Adapter(), object(), {
            "case_id": "one", "probe_id": "test", "role": "independent_test",
            "prompt": "Q", "answer": "A",
        })
        self.assertTrue(result["correct"])
        self.assertEqual(result["prediction"], "a")
        self.assertEqual(result["answer_token_count"], 1)

    def test_accepted_uses_saved_weights_and_rejected_keeps_baseline(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model_dir = root / "baseline"
            model_dir.mkdir()
            final_dir = root / "final"
            final_dir.mkdir()
            model = torch.nn.Module()
            model.layer = torch.nn.Linear(1, 1, bias=False)
            with torch.no_grad():
                model.layer.weight.zero_()
            baseline = model.layer.weight.detach().clone()
            saved = {"layer": torch.ones_like(baseline)}
            save_file(saved, str(final_dir / "weights.safetensors"))
            accuracy.write_json(final_dir / "manifest.json", {
                "status": "accepted", "p5_stage_complete": True, "tensor_file": "weights.safetensors",
                "tensor_file_sha256": file_sha256(final_dir / "weights.safetensors"),
                "baseline_weight_sha256": {"layer": tensor_sha256(baseline)},
                "layers": {"layer": {"sha256": tensor_sha256(saved["layer"])}}
            })
            inputs = root / "inputs.json"
            accuracy.write_json(inputs, {"inputs": [
                {"case_id": "one", "probe_id": "i1", "role": "independent_test", "prompt": "Q1", "answer": "A"},
                {"case_id": "two", "probe_id": "i2", "role": "independent_test", "prompt": "Q2", "answer": "A"},
                {"case_id": "keep", "probe_id": "p", "role": "preservation_test", "prompt": "K", "answer": "B"},
            ]})
            accuracy.write_json(root / "effective_config.json", {
                "target_layers": [0], "evaluation": {"inputs_file": str(inputs)},
                "execution": {"device": "cpu"}, "p1": {"seed": 2026}})
            accuracy.write_json(root / "run_report.json", {
                "status": "complete", "mode": "independent", "original_model": str(model_dir),
                "rounds": [
                    {"round_id": 0, "case_id": "one", "fact_ids": ["one"], "baseline_model": str(model_dir),
                     "decision": "accepted", "final_model_dir": str(final_dir)},
                    {"round_id": 1, "case_id": "two", "fact_ids": ["two"], "baseline_model": str(model_dir),
                     "decision": "rejected", "final_model_dir": None},
                ]})

            class Adapter:
                def load_model(self, *args, **kwargs):
                    with torch.no_grad():
                        model.layer.weight.zero_()
                    return model

                def target_module_names(self):
                    return ("layer",)

                def snapshot_target_weights(self, loaded):
                    return loaded.layer.weight.detach().clone()

                def restore_target_weights(self, loaded, snapshot):
                    with torch.no_grad():
                        loaded.layer.weight.copy_(snapshot)

            def fake_score(adapter, loaded, probe, **kwargs):
                changed = bool(loaded.layer.weight.item())
                return {"case_id": probe["case_id"], "probe_id": probe["probe_id"],
                        "role": probe["role"], "correct": changed if probe["role"] == "independent_test" else not changed}

            with patch.object(accuracy.LLaDA8BInstructAdapter, "from_local_tokenizer", return_value=Adapter()), \
                 patch.object(accuracy, "score_probe", side_effect=fake_score):
                result = accuracy.evaluate_run(root)
            self.assertEqual(result["independent"]["before"], {"correct": 0, "total": 2, "accuracy": 0.0})
            self.assertEqual(result["independent"]["after"], {"correct": 1, "total": 2, "accuracy": 0.5})
            self.assertEqual(result["preservation"]["after"], {"correct": 1, "total": 2, "accuracy": 0.5})
            self.assertEqual(model.layer.weight.item(), 0.0)
            self.assertTrue((root / "accuracy_report.json").is_file())


if __name__ == "__main__":
    unittest.main()
