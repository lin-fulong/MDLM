"""Optional tiny CPU checks for the adapter; no downloaded model is loaded."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None

from model.adapters.llada_8b_instruct import LLaDA8BInstructAdapter
from model.base import ForwardCapture, MaskedEditInput


@unittest.skipIf(torch is None, "PyTorch is not installed; tiny CPU adapter checks remain pending")
class LLaDAAdapterTests(unittest.TestCase):
    def setUp(self):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(block_group_size=1, mask_token_id=7, max_sequence_length=16)
                self.embedding = torch.nn.Embedding(8, 4)
                self.model = torch.nn.Module()
                self.model.transformer = torch.nn.Module()
                block = torch.nn.Module()
                block.ff_out = torch.nn.Linear(4, 3)
                self.model.transformer.blocks = torch.nn.ModuleList([block])
                self.head = torch.nn.Linear(3, 8)
                with torch.no_grad():
                    self.head.weight.zero_()
                    # 特殊 token 故意占据最高分，确认采样只选择合法普通 token。
                    self.head.bias.copy_(torch.tensor([100., 10., 9., 8., 7., 6., 5., 99.]))

            def get_input_embeddings(self):
                return self.embedding

            def forward(self, input_ids, attention_mask, use_cache, return_dict):
                if use_cache or not return_dict:
                    raise ValueError("tiny model requires explicit forward conditions")
                hidden = self.model.transformer.blocks[0].ff_out(self.embedding(input_ids))
                return SimpleNamespace(logits=self.head(hidden))

        self.model = TinyModel().eval()
        self.adapter = LLaDA8BInstructAdapter(
            SimpleNamespace(all_special_ids=[0, 7]), mask_token_id=7, target_layers=(0,)
        )
        self.model_input = MaskedEditInput((1, 7, 7, 7), (1, 1, 1, 1), (1, 2, 3), 0)

    def test_capture_uses_ffn_input_and_cleans_hooks_on_failure(self):
        result = self.adapter.forward_with_capture(self.model, self.model_input)
        torch.testing.assert_close(result.module_inputs[self.adapter.target_module_names()[0]], self.model.embedding.weight[1])
        self.assertEqual(tuple(result.mask_logits.shape), (3, 8))
        module = self.model.model.transformer.blocks[0].ff_out
        self.assertFalse(module._forward_pre_hooks)
        with patch.object(self.model, "forward", side_effect=RuntimeError("example forward failure")):
            with self.assertRaisesRegex(RuntimeError, "forward failure"):
                self.adapter.forward_with_capture(self.model, self.model_input)
        self.assertFalse(module._forward_pre_hooks)

    def test_weight_snapshot_restores_exact_parameter(self):
        snapshot = self.adapter.snapshot_target_weights(self.model)
        weight = self.model.model.transformer.blocks[0].ff_out.weight
        original = weight.detach().clone()
        with torch.no_grad():
            weight.add_(3)
        self.adapter.restore_target_weights(self.model, snapshot)
        self.assertTrue(torch.equal(weight, original))
        self.assertIs(self.model.model.transformer.blocks[0].ff_out.weight, weight)

    def test_sampler_replays_and_keeps_independent_inputs(self):
        for temperature in (0., 0.8):
            with self.subTest(temperature=temperature):
                kwargs = {"steps": 2, "temperature": temperature, "seed": 17}
                trace = self.adapter.sample_denoise(self.model, self.model_input, **kwargs)
                self.assertEqual(trace, self.adapter.sample_denoise(self.model, self.model_input, **kwargs))
                self.assertEqual(trace.pre_forward_inputs[0], self.model_input)
                self.assertEqual([len(item.mask_positions) for item in trace.pre_forward_inputs], [3, 1])
                self.assertEqual(trace.final_input.mask_positions, ())
                self.assertTrue(all(token not in (0, 7) for step in trace.revealed_tokens for _, token in step))
                self.assertEqual(self.model_input.input_ids, (1, 7, 7, 7))

    def test_logit_digest_keeps_original_precision(self):
        model_input = MaskedEditInput((1, 7), (1, 1), (1,), 0)
        traces = []
        for difference in (0., 1e-10):
            logits = torch.tensor([[0., 10. + difference, 1., 2., 3., 4., 5., 6.]], dtype=torch.float64)
            with patch.object(self.adapter, "forward_with_capture", return_value=ForwardCapture(logits, {})):
                traces.append(self.adapter.sample_denoise(None, model_input, steps=1, temperature=0., seed=1))
        self.assertEqual(traces[0].revealed_tokens, traces[1].revealed_tokens)
        self.assertNotEqual(traces[0].mask_logit_sha256, traces[1].mask_logit_sha256)


if __name__ == "__main__":
    unittest.main()
