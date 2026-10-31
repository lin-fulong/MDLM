"""Record editing duration and PyTorch CUDA peak memory in a compact CSV."""

import csv
import os
from pathlib import Path
import sys
import time

import torch


FIELDS = ("item", "status", "total_samples", "elapsed_seconds", "peak_vram_gib")


class EditResourceUsage:
    """Measure one edit run without changing its model or random state."""

    def __init__(self, output: Path, *, use_cuda: bool):
        self.path = output / "result" / "resource_report.csv"
        self.partial = self.path.with_name("resource_report.partial.csv")
        self.started = time.perf_counter()
        self.devices = ()
        self.rows = []
        self.current = None
        try:
            if use_cuda and torch.cuda.is_available():
                self.devices = tuple(range(torch.cuda.device_count()))
                for device in self.devices:
                    torch.cuda.reset_peak_memory_stats(device)
        except Exception:
            self.devices = ()

    def _sync(self):
        for device in self.devices:
            try:
                torch.cuda.synchronize(device)
            except Exception:
                # Resource measurement must not change the edit outcome.
                pass

    def start_case(self, case_id):
        self._sync()
        self.current = (case_id, time.perf_counter())

    def finish_case(self, status="complete"):
        if self.current is None:
            return
        self._sync()
        case_id, started = self.current
        self.rows.append({"item": case_id, "status": status, "total_samples": "",
                          "elapsed_seconds": f"{time.perf_counter() - started:.3f}",
                          "peak_vram_gib": ""})
        self.current = None
        try:
            self._write(self.partial, "running")
        except OSError as error:
            print(f"[resource_report] 临时快照保存失败：{error}", file=sys.stderr)

    def finish(self, status):
        self._sync()
        if self.current is not None:
            self.finish_case("failed")
        peaks = []
        for device in self.devices:
            try:
                peaks.append(torch.cuda.max_memory_reserved(device))
            except Exception:
                pass
        elapsed = time.perf_counter() - self.started
        peak = f"{max(peaks) / 1024 ** 3:.3f}" if peaks else ""
        try:
            self._write(self.path, status, elapsed=elapsed, peak=peak)
            self.partial.unlink(missing_ok=True)
        except OSError as error:
            print(f"[resource_report] 保存失败：{error}", file=sys.stderr)
        return {"resource_csv_file": str(self.path) if self.path.is_file() else None,
                "elapsed_seconds": round(elapsed, 3), "total_samples": len(self.rows),
                "peak_vram_gib": float(peak) if peak else None}

    def _write(self, path, status, *, elapsed=None, peak=""):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        total = {"item": "total", "status": status, "total_samples": len(self.rows),
                 "elapsed_seconds": f"{elapsed if elapsed is not None else time.perf_counter() - self.started:.3f}",
                 "peak_vram_gib": peak}
        with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows((total, *self.rows))
        os.replace(temporary, path)
