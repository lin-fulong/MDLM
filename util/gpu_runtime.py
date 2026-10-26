"""Run stages in one process and reuse a resident baseline model on the GPU."""

from __future__ import annotations

import gc
import importlib
import os
import subprocess
import sys
import traceback
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch

from MDLM.contracts import cached_file_hashes
from model.adapters.llada_8b_instruct import LLaDA8BInstructAdapter


class ResidentModel:
    """One live model; restore only edited layers before every new stage load."""

    def __init__(self, loader, layers, device):
        self.loader, self.device = loader, device
        self.names = tuple(f"model.transformer.blocks.{layer}.ff_out" for layer in layers)
        self.model, self.key = None, None
        self.baseline, self.versions = {}, {}
        self.load_count, self.reuse_count = 0, 0

    def restore(self):
        if self.model is None:
            return
        for name in self.names:
            module = self.model.get_submodule(name)
            if getattr(module, "_mdlm_candidate_owner", None) is not None:
                raise RuntimeError("复用模型前发现未移除的残差分支，停止运行")
            if module.weight._version != self.versions[name]:
                with torch.no_grad():
                    module.weight.copy_(self.baseline[name])
                self.versions[name] = module.weight._version
        self.model.eval()

    def load(self, model_path, **kwargs):
        kwargs["device_map"] = self.device
        root = Path(model_path).resolve()
        signatures = []
        for path in sorted(root.glob("*")):
            if path.is_file() and path.suffix in (".safetensors", ".json", ".py"):
                stat = path.stat()
                signatures.append((path.name, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
        key = (str(root), str(kwargs.get("torch_dtype", "auto")), tuple(signatures))
        if self.model is not None and self.key == key:
            self.restore()
            self.reuse_count += 1
            print(f"[GPU] 复用驻留模型：{self.device}", flush=True)
            return self.model
        self.close()
        model = self.loader(model_path, **kwargs)
        requested = torch.device(self.device)
        if requested.type == "cuda":
            index = requested.index if requested.index is not None else torch.cuda.current_device()
            if any(p.device != torch.device("cuda", index) for p in model.parameters()):
                raise RuntimeError("模型未完整加载到指定 GPU，停止运行以避免 CPU 卸载")
        self.model, self.key = model, key
        # 只保存目标层基线，非连续编辑及阶段复用均从该基线精确恢复。
        self.baseline = {name: model.get_submodule(name).weight.detach().clone() for name in self.names}
        self.versions = {name: model.get_submodule(name).weight._version for name in self.names}
        self.load_count += 1
        print(f"[GPU] 模型已驻留：{self.device}，累计加载 {self.load_count} 次", flush=True)
        return model

    def close(self):
        self.restore()
        self.model, self.key = None, None
        self.baseline.clear()
        self.versions.clear()
        gc.collect()
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()


@contextmanager
def execution_context(settings, layers):
    backend = settings.get("backend", "subprocess")
    if backend not in ("resident", "subprocess"):
        raise ValueError("execution.backend must be resident or subprocess")
    if backend == "subprocess":
        yield None
        return
    device = settings.get("device", "cuda:0")
    selected = torch.device(device)
    if selected.type not in ("cpu", "cuda"):
        raise ValueError("execution.device must be cpu or cuda:<index>")
    if selected.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("当前 Python 无法使用 CUDA，请激活含 CUDA 的 mdlm 环境；或显式使用 --device cpu")
        selected = torch.device("cuda", selected.index if selected.index is not None else 0)
        torch.cuda.set_device(selected)
        print(f"[GPU] {torch.cuda.get_device_name(selected)} | {selected} | 进程内复用模型", flush=True)
    pool = ResidentModel(LLaDA8BInstructAdapter.load_model, layers, str(selected))
    # 保持与原有验收相同的 float32/float64 算术精度。
    matmul_tf32, cudnn_tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with cached_file_hashes(settings.get("hash_cache_path"), force_rehash=settings.get("rehash_files", False),
                                progress=lambda message: print(message, flush=True)), \
                patch.object(LLaDA8BInstructAdapter, "load_model", staticmethod(pool.load)):
            yield pool
        print(f"[模型复用] 实际加载 {pool.load_count} 次，复用 {pool.reuse_count} 次", flush=True)
    finally:
        pool.close()
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32


def execute_stage(arguments, *, stream, cwd, resident=False):
    """Preserve existing CLI/artifact contracts without launching another Python."""
    if not resident:
        return subprocess.run([sys.executable, "-u", "-m", *arguments], cwd=cwd,
                              stdout=stream, stderr=subprocess.STDOUT, check=True)
    previous_argv, previous_cwd = sys.argv, Path.cwd()
    try:
        sys.argv = [arguments[0], *arguments[1:]]
        os.chdir(cwd)
        with redirect_stdout(stream), redirect_stderr(stream):
            try:
                importlib.import_module(arguments[0]).main()
            except SystemExit as error:
                if error.code not in (None, 0):
                    raise subprocess.CalledProcessError(error.code, arguments) from error
            except BaseException:
                traceback.print_exc()
                raise
    finally:
        sys.argv = previous_argv
        os.chdir(previous_cwd)
        gc.collect()
