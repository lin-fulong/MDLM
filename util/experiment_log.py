"""Mirror terminal output and append a readable summary for every experiment."""

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import threading
import traceback


def _name(value):
    """Keep path components short and portable across Linux and Windows."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip(".-")
    return (cleaned or "unknown")[:80]


class _Tee:
    """Write to the original terminal stream and the experiment log."""

    def __init__(self, terminal, log, lock):
        self.terminal = terminal
        self.log = log
        self.lock = lock

    def write(self, text):
        with self.lock:
            self.terminal.write(text)
            self.log.write(text)
            self.log.flush()
        return len(text)

    def flush(self):
        with self.lock:
            self.terminal.flush()
            self.log.flush()

    def isatty(self):
        return self.terminal.isatty()

    @property
    def encoding(self):
        return getattr(self.terminal, "encoding", "utf-8")

    def __getattr__(self, name):
        return getattr(self.terminal, name)


def _summary_lines(kind, result):
    lines = [f"status: {result.get('status', 'failed')}"]
    if result.get("failure"):
        lines.append(f"failure: {result['failure']}")
    if kind == "edit":
        lines.extend(f"{key}: {result[key]}" for key in
                     ("case_count", "fact_count", "accepted_count", "rejected_count") if key in result)
        if result.get("resource_csv_file"):
            lines.append(f"resource_csv_file: {result['resource_csv_file']}")
            lines.append(f"elapsed_seconds: {result['elapsed_seconds']}")
            lines.append(f"total_samples: {result['total_samples']}")
            if result.get("peak_vram_gib") is not None:
                lines.append(f"peak_vram_gib: {result['peak_vram_gib']}")
        for row in result.get("rounds", []):
            lines.append("round: " + json.dumps({key: row.get(key) for key in
                ("round_id", "case_id", "fact_ids", "decision", "selected_alpha", "selected_step", "final_model_dir")},
                ensure_ascii=False))
    else:
        lines.extend(f"{key}: {result[key]}" for key in
                     ("expected_tasks", "completed_tasks", "coverage", "error_count") if key in result)
        for group, metrics in result.get("metrics", {}).get("all", {}).items():
            lines.append("metric: " + json.dumps({"name": group, **metrics}, ensure_ascii=False))
    return lines


class ExperimentLog:
    """Own one log and capture stdout/stderr without hiding terminal output."""

    def __init__(self, output: Path, *, algorithm: str, model, dataset, kind: str,
                 parameters: dict):
        model_name = _name(Path(model).name if model else "unknown")
        dataset_name = _name(Path(dataset).stem if dataset else "unknown")
        directory = output.parent.parent / "logs" / _name(algorithm) / model_name / dataset_name
        directory.mkdir(parents=True, exist_ok=True)
        parameter_name = "_".join(f"{_name(key)}-{_name(value)}" for key, value in parameters.items())
        if len(parameter_name) > 110:
            digest = hashlib.sha256(parameter_name.encode()).hexdigest()[:12]
            parameter_name = parameter_name[:97] + "-" + digest
        self.path = directory / f"{_name(kind)}_{parameter_name}_{_name(output.name)}.log"
        self.metadata = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(), "kind": kind,
            "run_id": output.name, "model": model, "dataset": dataset,
            "parameters": json.dumps(parameters, ensure_ascii=False, sort_keys=True),
            "output": output,
        }
        self.kind = kind
        self.result = None

    def finish(self, result):
        """Provide the final report appended when terminal capture exits."""
        self.result = result

    @contextmanager
    def capture(self):
        """Tee stdout/stderr and preserve a traceback for an escaping error."""
        lock = threading.RLock()
        with self.path.open("x", encoding="utf-8", buffering=1) as stream:
            for key, value in self.metadata.items():
                stream.write(f"{key}: {value}\n")
            stream.write("\n[terminal_output]\n")
            stdout = _Tee(sys.stdout, stream, lock)
            stderr = _Tee(sys.stderr, stream, lock)
            try:
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    yield self
            finally:
                error = sys.exc_info()
                if error[0] is not None:
                    stream.write("\n[uncaught_exception]\n")
                    stream.writelines(traceback.format_exception(*error))
                stream.write("\n[experiment_summary]\n")
                stream.write("\n".join(_summary_lines(self.kind, self.result or {})) + "\n")


def append_command_output(path, text):
    """Append output printed by the CLI after the experiment context closes."""
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write("\n[command_output]\n" + text + "\n")
