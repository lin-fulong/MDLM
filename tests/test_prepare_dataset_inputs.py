"""Dataset preparation, disjoint partitions, and content-addressed cache checks."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from util import prepare_dataset_inputs as preparation


class DatasetPreparationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dataset = self.root / "selected.json"
        self.model = self.root / "model"
        self.model.mkdir()
        for filename in ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
            (self.model / filename).write_text("{}", encoding="utf-8")
        self.case = {"case_id": "fact", "quality_control": {"status": "approved"}, "prompts": {
            "edit": [{"probe_id": "edit", "prompt": "edit subject?", "answer": "new"}],
            "preservation": [{"probe_id": f"keep_{index}", "prompt": f"preserve {index}?", "answer": "old"}
                             for index in range(4)],
            "free_form_qa": [{"probe_id": "independent", "prompt": "held out?", "answer": "new", "role": "independent_test"}]}}
        self.dataset.write_text(json.dumps([self.case]), encoding="utf-8")
        tokenizer = SimpleNamespace(all_special_ids=[],
                                    apply_chat_template=lambda messages, **kwargs: messages[0]["content"] + " answer:",
                                    encode=lambda text, **kwargs: [ord(character) + 10 for character in text])
        self.adapter = SimpleNamespace(tokenizer=tokenizer, mask_token_id=99999)

    def prepare(self, **kwargs):
        with patch.object(preparation.LLaDA8BInstructAdapter, "from_local_tokenizer", return_value=self.adapter):
            return preparation.prepare_inputs(self.dataset, self.model, (0,), self.root / "cache", **kwargs)

    def test_builds_disjoint_keep_inputs(self):
        result = self.prepare(settings={"statistics_scale": 2.0})
        self.assertFalse(result["cache_hit"])
        files = result["files"]
        parts = json.loads(Path(files["partitions.json"]).read_text())["preservation"]
        pools = [set(parts[name]) for name in ("dev", "test", "statistics_train")]
        self.assertFalse(pools[0] & pools[1] or pools[0] & pools[2] or pools[1] & pools[2])
        keep = json.loads(Path(files["keep_statistics.json"]).read_text())
        weights = [weight for row in keep["inputs"] for weight in row["weights_by_layer"]["model.transformer.blocks.0.ff_out"]]
        self.assertAlmostEqual(sum(weights), 2.0)
        for filename in ("keep_training.json", "development_keep.json"):
            payload = json.loads(Path(files[filename]).read_text())
            self.assertAlmostEqual(sum(w for row in payload["inputs"] for w in row["weights"]), 1.0)
            for row in payload["inputs"]:
                self.assertTrue(all(row["input"]["input_ids"][p] == 99999 for p in row["positions"]))
        probes = json.loads(Path(files["development_probes.json"]).read_text())
        self.assertEqual(probes, [{"case_id": "fact", "probe_id": "edit"}])
        self.assertNotIn("evaluation_inputs.json", files)

    def test_cache_hit_avoids_tokenizing_and_dataset_change_misses(self):
        first = self.prepare()
        with patch.object(preparation.LLaDA8BInstructAdapter, "from_local_tokenizer") as tokenizer:
            cached = preparation.prepare_inputs(self.dataset, self.model, (0,), self.root / "cache")
            tokenizer.assert_not_called()
        self.assertTrue(cached["cache_hit"])
        self.assertEqual(cached["cache_key"], first["cache_key"])
        self.case["prompts"]["edit"][0]["answer"] = "another answer"
        self.dataset.write_text(json.dumps([self.case]), encoding="utf-8")
        second = self.prepare()
        self.assertNotEqual(first["cache_key"], second["cache_key"])

    def test_corrupted_cached_file_is_rebuilt_in_place(self):
        first = self.prepare()
        expected = {name: Path(path).read_text() for name, path in first["files"].items()}
        Path(first["files"]["keep_statistics.json"]).write_text("broken", encoding="utf-8")
        second = self.prepare()
        self.assertFalse(second["cache_hit"])
        self.assertEqual(second["cache_key"], first["cache_key"])
        self.assertEqual({name: Path(path).read_text() for name, path in second["files"].items()}, expected)
        self.assertEqual(len(list((self.root / "cache/datasets").iterdir())), 1)


if __name__ == "__main__":
    unittest.main()
