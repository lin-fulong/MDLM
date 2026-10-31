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
        self.use_cuda = use_cuda
        self.devices = ()
        self.error = None
        self.rows = []
        self.current = None

    def _monitor_error(self, action, error):
        message = f"{action}: {type(error).__name__}: {error}"
        if self.error is None:
            self.error = message
            print(f"[resource_report] 显存监控失败：{message}", file=sys.stderr)

    def start_gpu(self):
        """Reset peaks after execution_context has initialized the CUDA device."""
        if not self.use_cuda:
            return
        try:
            if torch.cuda.is_available():
                torch.cuda.current_device()
                devices = tuple(range(torch.cuda.device_count()))
                for device in devices:
                    torch.cuda.reset_peak_memory_stats(device)
                self.devices = devices
        except Exception as error:
            self._monitor_error("初始化峰值统计", error)

    def _sync(self):
        for device in self.devices:
            try:
                torch.cuda.synchronize(device)
            except Exception as error:
                # Resource measurement must not change the edit outcome.
                self._monitor_error("同步 CUDA", error)

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
            except Exception as error:
                self._monitor_error("读取峰值显存", error)
        elapsed = time.perf_counter() - self.started
        peak = f"{max(peaks) / 1024 ** 3:.3f}" if peaks else ""
        try:
            self._write(self.path, status, elapsed=elapsed, peak=peak)
            self.partial.unlink(missing_ok=True)
        except OSError as error:
            print(f"[resource_report] 保存失败：{error}", file=sys.stderr)
        return {"resource_csv_file": str(self.path) if self.path.is_file() else None,
                "elapsed_seconds": round(elapsed, 3), "total_samples": len(self.rows),
                "peak_vram_gib": float(peak) if peak else None,
                "resource_monitor_error": self.error}

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
