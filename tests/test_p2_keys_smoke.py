"""P2 smoke orchestration on constructed artifacts; no model is loaded."""

from __future__ import annotations

import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from MDLM.contracts import ReferenceKeyConfig, file_sha256
from util import smoke_p2_keys


class ReferenceKeySmokeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run_root = self.root / "runs" / "example_only_smoke" / "round_0"
        self.config_path = self.root / "keys_config.json"
        self.config_path.write_text(json.dumps(ReferenceKeyConfig().as_record()), encoding="utf-8")
        self.arguments = [
            "smoke_p2_keys", "--layer", "0", "--model-path", str(self.root / "model"),
            "--config", str(self.config_path), "--output-root", str(self.root / "runs"),
            "--run-id", "example_only_smoke",
        ]

    def complete_p0_p1(self, command, **kwargs):
        self.run_root.mkdir(parents=True)
        (self.run_root / "p1").mkdir()
        (self.run_root / "acceptance_report.json").write_text(
            json.dumps({"status": "passed"}), encoding="utf-8"
        )
        return subprocess.CompletedProcess(command, 0)

    def complete_keys(self, p1_dir, settings):
        self.assertEqual(p1_dir, self.run_root / "p1")
        self.assertEqual(settings, ReferenceKeyConfig())
        output_dir = self.run_root / "matrices" / "reference_keys"
        output_dir.mkdir(parents=True)
        (output_dir / "manifest.json").write_text(
            json.dumps({"status": "complete", "example_only_fact_count": 1}), encoding="utf-8"
        )
        return {"status": "complete", "output_dir": str(output_dir)}

    def report(self):
        return json.loads((self.run_root / "p2_keys_acceptance_report.json").read_text(encoding="utf-8"))

    def test_success_runs_p0_p1_before_keys_and_binds_reports(self):
        with patch.object(sys, "argv", self.arguments), patch("sys.stdout", new_callable=io.StringIO), \
                patch.object(smoke_p2_keys.subprocess, "run", side_effect=self.complete_p0_p1) as previous, \
                patch.object(smoke_p2_keys, "extract_run", side_effect=self.complete_keys) as extract:
            smoke_p2_keys.main()
        previous.assert_called_once()
        command = previous.call_args.args[0]
        self.assertEqual(command[:3], [sys.executable, "-m", "util.accept_p0_p1_single_case"])
        self.assertEqual(command[command.index("--layer") + 1], "0")
        self.assertEqual(command[command.index("--run-id") + 1], "example_only_smoke")
        self.assertTrue(previous.call_args.kwargs["check"])
        extract.assert_called_once_with(self.run_root / "p1", ReferenceKeyConfig())
        report = self.report()
        self.assertEqual((report["status"], report["phase"]), ("passed", "complete"))
        self.assertTrue(report["example_only"])
        self.assertFalse(report["formal_dataset_acceptance"])
        self.assertFalse(report["p2_stage_complete"])
        self.assertEqual(report["p0_p1_report_sha256"], file_sha256(self.run_root / "acceptance_report.json"))
        self.assertEqual(report["key_manifest_sha256"], file_sha256(
            self.run_root / "matrices" / "reference_keys" / "manifest.json"
        ))

    def test_p0_p1_subprocess_failure_does_not_extract_keys(self):
        def fail_previous(command, **kwargs):
            self.run_root.mkdir(parents=True)
            raise subprocess.CalledProcessError(1, command)

        with patch.object(sys, "argv", self.arguments), patch("sys.stdout", new_callable=io.StringIO), \
                patch.object(smoke_p2_keys.subprocess, "run", side_effect=fail_previous), \
                patch.object(smoke_p2_keys, "extract_run") as extract:
            with self.assertRaises(subprocess.CalledProcessError):
                smoke_p2_keys.main()
        extract.assert_not_called()
        report = self.report()
        self.assertEqual((report["status"], report["phase"]), ("failed", "p0_p1"))
        self.assertIn("non-zero exit status", report["failure"])

    def test_key_extraction_failure_keeps_failed_acceptance_report(self):
        with patch.object(sys, "argv", self.arguments), patch("sys.stdout", new_callable=io.StringIO), \
                patch.object(smoke_p2_keys.subprocess, "run", side_effect=self.complete_p0_p1), \
                patch.object(smoke_p2_keys, "extract_run", side_effect=ValueError("key replay failed")):
            with self.assertRaisesRegex(ValueError, "key replay failed"):
                smoke_p2_keys.main()
        report = self.report()
        self.assertEqual((report["status"], report["phase"]), ("failed", "reference_keys"))
        self.assertEqual(report["failure"], "key replay failed")
        self.assertNotIn("key_manifest_sha256", report)

    def test_existing_run_is_preserved_without_starting_experiments(self):
        self.run_root.mkdir(parents=True)
        path = self.run_root / "p2_keys_acceptance_report.json"
        original = b'{"status": "passed", "existing": true}\n'
        path.write_bytes(original)
        with patch.object(sys, "argv", self.arguments), \
                patch.object(smoke_p2_keys.subprocess, "run") as previous, \
                patch.object(smoke_p2_keys, "extract_run") as extract:
            with self.assertRaisesRegex(FileExistsError, "will not be overwritten"):
                smoke_p2_keys.main()
        previous.assert_not_called()
        extract.assert_not_called()
        self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
