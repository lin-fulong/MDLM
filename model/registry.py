"""按名称创建本地模型适配器，具体模型实现按需导入。"""

from __future__ import annotations


def get_adapter(model_name: str = "LLaDA-8B-Instruct", **kwargs):
    """集中选择模型和加载 tokenizer；默认沿用现有 LLaDA，无需新增配置字段。"""
    from model.adapters.llada_8b_instruct import LLaDA8BInstructAdapter

    adapter = {"LLaDA-8B-Instruct": LLaDA8BInstructAdapter}[model_name]
    return adapter.from_local_tokenizer(**kwargs)
