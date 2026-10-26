"""Exercise real local LLaDA RoPE caching with 21K parameters, without 8B weights."""

import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from MDLM.candidate import CandidateModel
from model.adapters.llada_8b_instruct import DEFAULT_MODEL_PATH, LLaDA8BInstructAdapter
from model.base import MaskedEditInput


@unittest.skipUnless((DEFAULT_MODEL_PATH / "modeling_llada.py").is_file(), "local LLaDA implementation unavailable")
class LLaDAResidentBackwardTests(unittest.TestCase):
    def prepare(self, device):
        from transformers import AutoConfig, AutoModelForCausalLM
        config = AutoConfig.from_pretrained(DEFAULT_MODEL_PATH, trust_remote_code=True, local_files_only=True)
        for name, value in dict(d_model=16, n_heads=2, n_kv_heads=2, n_layers=2, mlp_hidden_size=32,
                                vocab_size=32, embedding_size=32, max_sequence_length=16,
                                mask_token_id=31, block_group_size=1).items():
            setattr(config, name, value)
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True,
                                               torch_dtype=torch.float32).to(device).eval()
        adapter = LLaDA8BInstructAdapter(SimpleNamespace(all_special_ids=[31]), mask_token_id=31, target_layers=(0,))
        cache = model.model.transformer.blocks[0].rotary_emb._RotaryEmbedding__cache
        cache.clear()  # 强制在本次前向生成缓存，CPU 也能复现原来的问题。
        model_input = MaskedEditInput((1, 2, 31, 31), (1, 1, 1, 1), (2, 3), 1)
        return model, adapter, cache, model_input

    def check_backward(self, model, adapter, cache, model_input):
        name = adapter.target_module_names()[0]
        width = model.get_submodule(name).in_features
        q = {name: torch.ones(1, width, dtype=torch.float32) / width}
        with CandidateModel(model, q) as candidate:
            self.assertTrue(all(not torch.is_inference(value) for value in cache.values()))
            with torch.no_grad():
                candidate.parameters()[0].fill_(.01)
            logits = adapter.forward_logits(model, model_input, model_input.mask_positions)
            loss = -logits.log_softmax(-1)[:, 3].mean()
            loss.backward()
            gradient = candidate.parameters()[0].grad
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().max()), 0)

    def test_capture_warmup_then_real_llada_backward_on_cpu(self):
        model, adapter, cache, model_input = self.prepare("cpu")
        adapter.forward_with_capture(model, model_input)
        self.assertTrue(cache)
        self.assertTrue(all(not torch.is_inference(value) for value in cache.values()))
        self.check_backward(model, adapter, cache, model_input)

    def test_existing_inference_cache_is_converted_without_value_changes(self):
        model, adapter, cache, model_input = self.prepare("cpu")
        with torch.inference_mode():
            adapter.forward_with_capture(model, model_input)
        self.assertTrue(any(torch.is_inference(value) for value in cache.values()))
        before = {key: value.clone() for key, value in cache.items()}
        cache_identity = id(cache)
        self.check_backward(model, adapter, cache, model_input)
        self.assertEqual(id(cache), cache_identity)
        self.assertTrue(all(torch.equal(value, before[key]) for key, value in cache.items()))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_capture_warmup_then_real_llada_backward_on_cuda(self):
        model, adapter, cache, model_input = self.prepare("cuda:0")
        adapter.forward_with_capture(model, model_input)
        self.assertTrue(all(value.device.type == "cuda" and not torch.is_inference(value) for value in cache.values()))
        self.check_backward(model, adapter, cache, model_input)


if __name__ == "__main__":
    unittest.main()
