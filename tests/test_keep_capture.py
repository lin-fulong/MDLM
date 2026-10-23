"""Formula (5) multi-position capture checks on an artificial tiny CPU model."""

from __future__ import annotations

import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None

from model.adapters.llada_8b_instruct import LLaDA8BInstructAdapter
from model.base import KeepModelInput, MaskedEditInput


@unittest.skipIf(torch is None, "PyTorch is required for tiny preservation-capture tests")
class KeepCaptureTests(unittest.TestCase):
    def setUp(self):
        class TinyContextModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(block_group_size=1, mask_token_id=7, max_sequence_length=16)
                self.embedding = torch.nn.Embedding(8, 4)
                self.model = torch.nn.Module()
                self.model.transformer = torch.nn.Module()
                self.model.transformer.blocks = torch.nn.ModuleList()
                for width_in, width_out in ((4, 3), (3, 2)):
                    block = torch.nn.Module()
                    block.ff_out = torch.nn.Linear(width_in, width_out)
                    self.model.transformer.blocks.append(block)
                self.head = torch.nn.Linear(2, 8)

            def get_input_embeddings(self):
                return self.embedding

            def forward(self, input_ids, attention_mask, use_cache, return_dict):
                if use_cache or not return_dict or not attention_mask.all():
                    raise ValueError("unexpected forward conditions")
                hidden = self.embedding(input_ids)
                # 人工双向上下文使 mask 变化能影响其他保持位置的实际输入。
                hidden = hidden + hidden.mean(dim=1, keepdim=True)
                for block in self.model.transformer.blocks:
                    hidden = block.ff_out(hidden)
                return SimpleNamespace(logits=self.head(hidden))

        torch.manual_seed(42)
        self.model = TinyContextModel().eval()
        self.adapter = LLaDA8BInstructAdapter(
            SimpleNamespace(all_special_ids=[7]), mask_token_id=7, target_layers=(0, 1)
        )
        self.model_input = KeepModelInput((1, 7, 2, 7, 3), (1, 1, 1, 1, 1), (1, 3))

    def assert_hooks_removed(self):
        for block in self.model.model.transformer.blocks:
            self.assertFalse(block.ff_out._forward_pre_hooks)

    def test_captures_actual_inputs_at_mask_and_context_positions_in_given_order(self):
        positions = (4, 1, 0)
        original = {name: value.detach().clone() for name, value in self.model.named_parameters()}
        result = self.adapter.forward_keep_with_capture(self.model, self.model_input, positions)
        embeddings = self.model.embedding.weight[list(self.model_input.input_ids)].detach()
        expected_first = (embeddings + embeddings.mean(dim=0))[list(positions)]
        expected_second = self.model.model.transformer.blocks[0].ff_out(expected_first).detach()
        names = self.adapter.target_module_names()
        torch.testing.assert_close(result.module_inputs[names[0]], expected_first, atol=0, rtol=0)
        torch.testing.assert_close(result.module_inputs[names[1]], expected_second)
        self.assertEqual(tuple(result.mask_logits.shape), (2, 8))
        self.assertEqual(tuple(result.module_inputs[names[1]].shape), (3, 3))
        self.assertTrue(all(not value.requires_grad for value in result.module_inputs.values()))
        for name, value in self.model.named_parameters():
            self.assertTrue(torch.equal(value, original[name]))
        self.assert_hooks_removed()

    def test_mask_context_changes_other_positions_and_all_visible_is_supported(self):
        masked = self.adapter.forward_keep_with_capture(self.model, self.model_input, (0, 4))
        visible_input = KeepModelInput((1, 5, 2, 6, 3), self.model_input.attention_mask, ())
        visible = self.adapter.forward_keep_with_capture(self.model, visible_input, (0, 4))
        self.assertEqual(tuple(visible.mask_logits.shape), (0, 8))
        for name in self.adapter.target_module_names():
            self.assertFalse(torch.equal(masked.module_inputs[name], visible.module_inputs[name]))
        self.assertEqual(self.model_input.input_ids, (1, 7, 2, 7, 3))

    def test_bad_capture_and_mask_positions_are_rejected(self):
        for positions in ((), (0, 0), (-1,), (5,), (True,)):
            with self.subTest(capture_positions=positions):
                with self.assertRaisesRegex(ValueError, "capture positions"):
                    self.adapter.forward_keep_with_capture(self.model, self.model_input, positions)
        for positions in ((1, 1, 3), (-1, 3), (1, 5), (True, 3), (1,), (0, 3)):
            with self.subTest(mask_positions=positions):
                with self.assertRaisesRegex(ValueError, "mask positions"):
                    self.adapter.forward_keep_with_capture(
                        self.model, replace(self.model_input, mask_positions=positions), (0,)
                    )
        self.assert_hooks_removed()

    def test_forward_conditions_and_attention_mask_are_enforced(self):
        for attention_mask in ((1, 0, 1, 1, 1), (1, 1)):
            with self.subTest(attention_mask=attention_mask):
                with self.assertRaisesRegex(ValueError, "all-ones attention"):
                    self.adapter.forward_keep_with_capture(
                        self.model, replace(self.model_input, attention_mask=attention_mask), (0,)
                    )
        self.model.train()
        with self.assertRaisesRegex(ValueError, "model.eval"):
            self.adapter.forward_keep_with_capture(self.model, self.model_input, (0,))
        self.model.eval()
        self.model.config.max_sequence_length = 4
        with self.assertRaisesRegex(ValueError, "max_sequence_length"):
            self.adapter.forward_keep_with_capture(self.model, self.model_input, (0,))
        self.assert_hooks_removed()

    def test_forward_failure_removes_hooks(self):
        with patch.object(self.model, "forward", side_effect=RuntimeError("artificial forward failure")):
            with self.assertRaisesRegex(RuntimeError, "artificial forward failure"):
                self.adapter.forward_keep_with_capture(self.model, self.model_input, (0, 4))
        self.assert_hooks_removed()

    def test_nonfinite_logits_outside_masks_are_rejected(self):
        forward = self.model.forward

        def nonfinite(*args, **kwargs):
            result = forward(*args, **kwargs)
            result.logits[0, 0, 0] = float("nan")
            return result

        with patch.object(self.model, "forward", side_effect=nonfinite):
            with self.assertRaisesRegex(ValueError, "nonfinite preservation logits"):
                self.adapter.forward_keep_with_capture(self.model, self.model_input, (4,))
        self.assert_hooks_removed()

    def test_nonfinite_captured_input_is_rejected_even_when_logits_are_finite(self):
        with torch.no_grad():
            self.model.embedding.weight[1, 0] = float("nan")
        module = self.model.model.transformer.blocks[0].ff_out
        forward = module.forward
        with patch.object(module, "forward", side_effect=lambda value: forward(torch.nan_to_num(value))):
            with self.assertRaisesRegex(ValueError, "nonfinite target module input"):
                self.adapter.forward_keep_with_capture(self.model, self.model_input, (0,))
        self.assert_hooks_removed()

    def test_edit_interface_keeps_single_vector_and_original_mask_constraints(self):
        edit = MaskedEditInput((1, 2, 7, 7), (1, 1, 1, 1), (2, 3), 1)
        edited = self.adapter.forward_with_capture(self.model, edit)
        kept = self.adapter.forward_keep_with_capture(
            self.model, KeepModelInput(edit.input_ids, edit.attention_mask, edit.mask_positions), (1,)
        )
        for name in self.adapter.target_module_names():
            self.assertEqual(edited.module_inputs[name].ndim, 1)
            torch.testing.assert_close(edited.module_inputs[name], kept.module_inputs[name][0], atol=0, rtol=0)
        torch.testing.assert_close(edited.mask_logits, kept.mask_logits, atol=0, rtol=0)
        for invalid in (
            replace(edit, subject_last_position=2),
            replace(edit, mask_positions=()),
            replace(edit, mask_positions=(2,)),
            replace(edit, attention_mask=(1, 0, 1, 1)),
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    self.adapter.forward_with_capture(self.model, invalid)
        self.assert_hooks_removed()


if __name__ == "__main__":
    unittest.main()
