"""Choose a model-specific adapter without importing its runtime eagerly."""

from __future__ import annotations


def get_adapter(model_name: str, **kwargs):
    if model_name == "LLaDA-8B-Instruct":
        from model.adapters.llada_8b_instruct import LLaDA8BInstructAdapter

        return LLaDA8BInstructAdapter(**kwargs)
    raise ValueError(f"no P0 adapter for {model_name!r}")
