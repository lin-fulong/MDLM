"""An accepted edit changes only its indexed target tensor in the next baseline."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file

from MDLM.contracts import file_sha256
from model.base import tensor_sha256
from util import export_edited_checkpoint as exporter


class ExportCheckpointTests(unittest.TestCase):
    def test_rewrites_target_shard_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline, editing, output = (root / name for name in ("baseline", "editing", "checkpoint"))
            baseline.mkdir()
            editing.mkdir()
            original = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
            changed = torch.tensor([[3.0, 4.0]], dtype=torch.float32)
            other = torch.tensor([5.0], dtype=torch.float32)
            name = "model.transformer.blocks.0.ff_out"
            save_file({name + ".weight": original}, str(baseline / "model-00001.safetensors"))
            save_file({"unrelated.weight": other}, str(baseline / "model-00002.safetensors"))
            (baseline / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
                name + ".weight": "model-00001.safetensors",
                "unrelated.weight": "model-00002.safetensors"}}), encoding="utf-8")
            (baseline / "config.json").write_text("{}", encoding="utf-8")
            weights_path = editing / "final_target_weights.safetensors"
            save_file({name: changed}, str(weights_path))
            identity = {"model_revision": "old", "tokenizer_revision": "tokenizer"}
            (editing / "manifest.json").write_text(json.dumps({
                "status": "accepted", "example_only": False, "p5_stage_complete": True,
                "model_identity": identity, "tensor_file": weights_path.name,
                "tensor_file_sha256": file_sha256(weights_path),
                "layers": {name: {"shape": list(changed.shape), "dtype": str(changed.dtype),
                                  "sha256": tensor_sha256(changed)}},
            }), encoding="utf-8")

            def fake_identity(path):
                return identity if path == baseline else {
                    "model_revision": "new", "tokenizer_revision": "tokenizer"}

            with patch.object(exporter, "local_model_identity", side_effect=fake_identity):
                report = exporter.export_checkpoint(baseline, editing, output)
            self.assertEqual(report["status"], "complete")
            self.assertTrue(torch.equal(load_file(str(baseline / "model-00001.safetensors"))[name + ".weight"], original))
            self.assertTrue(torch.equal(load_file(str(output / "model-00001.safetensors"))[name + ".weight"], changed))
            self.assertTrue(torch.equal(load_file(str(output / "model-00002.safetensors"))["unrelated.weight"], other))
            self.assertEqual(json.loads((output / "export_report.json").read_text())["rewritten_shards"],
                             ["model-00001.safetensors"])


if __name__ == "__main__":
    unittest.main()
