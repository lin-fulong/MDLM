"""Materialize an accepted P5 edit as a local sharded model for the next round."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from MDLM.contracts import file_sha256, local_model_identity
from model.base import tensor_sha256


def export_checkpoint(baseline: Path, editing: Path, output: Path) -> dict:
    baseline, editing, output = baseline.resolve(), editing.resolve(), output.resolve()
    manifest = json.loads((editing / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "accepted" or manifest["example_only"] or not manifest["p5_stage_complete"]:
        raise ValueError("only an accepted non-example P5 result can become the next baseline")
    if local_model_identity(baseline) != manifest["model_identity"]:
        raise ValueError("checkpoint export baseline differs from the accepted P5 baseline")
    weights_path = editing / manifest["tensor_file"]
    if file_sha256(weights_path) != manifest["tensor_file_sha256"]:
        raise ValueError("P5 final target weights changed before checkpoint export")
    weights = load_file(str(weights_path), device="cpu")
    if set(weights) != set(manifest["layers"]):
        raise ValueError("P5 target layer set differs from the saved weights")
    index = json.loads((baseline / "model.safetensors.index.json").read_text(encoding="utf-8"))
    by_shard: dict[str, dict[str, torch.Tensor]] = {}
    for name, weight in weights.items():
        key = name + ".weight"
        shard = index["weight_map"].get(key)
        info = manifest["layers"][name]
        if shard is None or list(weight.shape) != info["shape"] or str(weight.dtype) != info["dtype"] \
                or tensor_sha256(weight) != info["sha256"] or not bool(torch.isfinite(weight).all()):
            raise ValueError(f"invalid P5 target weight for {name}")
        by_shard.setdefault(shard, {})[key] = weight

    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "export_report.json"
    try:
        for source in baseline.iterdir():
            if not source.is_file() or source.name == "export_report.json":
                continue
            target = output / source.name
            if source.name in by_shard:
                with safe_open(str(source), framework="pt", device="cpu") as stream:
                    tensors = {key: stream.get_tensor(key) for key in stream.keys()}
                    metadata = stream.metadata()
                if not set(by_shard[source.name]).issubset(tensors):
                    raise ValueError(f"target tensor is absent from {source.name}")
                for key, weight in by_shard[source.name].items():
                    if tensors[key].shape != weight.shape or tensors[key].dtype != weight.dtype:
                        raise ValueError(f"target tensor shape or dtype differs for {key}")
                    tensors[key] = weight
                save_file(tensors, str(target), metadata=metadata)
                del tensors
            elif source.suffix == ".safetensors":
                try:
                    os.link(source, target)
                except OSError:
                    shutil.copy2(source, target)
            else:
                shutil.copy2(source, target)
        for shard, replacements in by_shard.items():
            with safe_open(str(output / shard), framework="pt", device="cpu") as stream:
                if any(not torch.equal(stream.get_tensor(key), value) for key, value in replacements.items()):
                    raise ValueError(f"saved checkpoint differs in {shard}")
        identity = local_model_identity(output)
        if identity["tokenizer_revision"] != manifest["model_identity"]["tokenizer_revision"]:
            raise ValueError("exported tokenizer differs from the accepted baseline")
        report = {"status": "complete", "baseline": str(baseline), "editing": str(editing),
                  "checkpoint": str(output), "model_identity": identity,
                  "rewritten_shards": sorted(by_shard)}
    except Exception as error:
        report = {"status": "failed", "failure": str(error), "baseline": str(baseline),
                  "editing": str(editing), "checkpoint": str(output)}
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
