"""CPU wiring checks; native CUDA kernel and scheduler validation is separate."""
import copy
import importlib
import json
import os
import types
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModelForCausalLM

from kroha_sglang import supports_config


class Linear(nn.Linear):
    def __init__(self, input_size, *sizes, params_dtype=None, **kwargs):
        if isinstance(sizes[0], list):
            output_size = sum(sizes[0])
        else:
            dim, heads, kv = sizes
            output_size = (heads + 2 * kv) * dim
        super().__init__(input_size, output_size, bias=False, dtype=params_dtype)

    def forward(self, x):
        return super().forward(x), None


class Attention(nn.Module):
    def __init__(self, heads, dim, scaling, kv, layer_id):
        super().__init__()
        self.h, self.d, self.scale, self.kv, self.layer_id = heads, dim, scaling, kv, layer_id
        self.states = {}

    def forward(self, q, k, v, batch):
        outputs, offset = [], 0
        for request, size, start in batch.segments:
            qi = q[offset:offset + size].view(1, size, self.h, self.d).transpose(1, 2)
            ki = k[offset:offset + size].view(1, size, self.kv, self.d).transpose(1, 2)
            vi = v[offset:offset + size].view(1, size, self.kv, self.d).transpose(1, 2)
            if start:
                old_k, old_v = self.states[request]
                ki, vi = torch.cat((old_k, ki), 2), torch.cat((old_v, vi), 2)
            self.states[request] = ki, vi
            mask = torch.arange(ki.shape[2])[None, :] <= torch.arange(start, start + size)[:, None]
            out = F.scaled_dot_product_attention(qi,
                ki.repeat_interleave(self.h // self.kv, 1),
                vi.repeat_interleave(self.h // self.kv, 1), attn_mask=mask, scale=self.scale)
            outputs.append(out.transpose(1, 2).reshape(size, -1))
            offset += size
        return torch.cat(outputs)


class Delta(nn.Module):
    reference = None

    def __init__(self, slot, qh, kh, vh, qd, kd, vd, activation):
        super().__init__()
        self.layer_id, self.h, self.d = slot, vh, vd
        self.states = {}

    def forward(self, batch, mixed_qkv, a, b):
        outputs, offset = [], 0
        for request, size, start in batch.segments:
            raw = mixed_qkv[offset:offset + size].T[None]
            width = self.conv_weights.shape[-1]
            if start:
                history, state = self.states[request]
            else:
                history = raw.new_zeros(1, raw.shape[1], width - 1)
                state = None
            combined = torch.cat((history, raw), -1)
            mixed = F.silu(F.conv1d(combined, self.conv_weights[:, None], groups=raw.shape[1])).transpose(1, 2)
            q, k, v = (t.reshape(1, size, self.h, self.d) for t in mixed.chunk(3, -1))
            ai, bi = a[offset:offset + size][None], b[offset:offset + size][None]
            g = -self.A_log.float().exp() * F.softplus(ai.float() + self.dt_bias)
            out, final = type(self).reference(q, k, v, g=g, beta=bi.sigmoid(),
                initial_state=state, output_final_state=True, use_qk_l2norm_in_kernel=True)
            self.states[request] = combined[..., -(width - 1):].clone(), final
            outputs.append(out)
            offset += size
        return torch.cat(outputs, 1)


class Logits(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, ids, hidden, head, batch):
        return head(hidden)


def mocks(**overrides):
    args = types.SimpleNamespace(speculative_algorithm=None, disable_radix_cache=True,
        disable_overlap_schedule=True)
    for name, value in overrides.items():
        setattr(args, name, value)
    disabled = types.SimpleNamespace(backend="disabled")
    runtime = types.SimpleNamespace(
        get_server_args=lambda: args,
        get_exec=lambda: types.SimpleNamespace(graph=types.SimpleNamespace(
            cuda_graph_config=types.SimpleNamespace(decode=disabled, prefill=disabled))),
        get_parallel=lambda: types.SimpleNamespace(tp_size=1, pp_size=1))
    return {
        "sglang.srt.runtime_context": runtime,
        "sglang.srt.layers.linear": types.SimpleNamespace(QKVParallelLinear=Linear, MergedColumnParallelLinear=Linear),
        "sglang.srt.layers.radix_attention": types.SimpleNamespace(RadixAttention=Attention),
        "sglang.srt.layers.radix_linear_attention": types.SimpleNamespace(RadixLinearAttention=Delta),
        "sglang.srt.layers.logits_processor": types.SimpleNamespace(LogitsProcessor=Logits),
    }


@unittest.skipUnless(os.environ.get("KROHA_TEST_MODEL"), "Set KROHA_TEST_MODEL to a standalone HF export")
class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cfg = AutoConfig.from_pretrained(os.environ["KROHA_TEST_MODEL"], trust_remote_code=True)
        values = cfg.to_dict()
        values.update(vocab_size=128, hidden_size=16, head_dim=8, num_attention_heads=2,
            num_key_value_heads=1, num_core_layers=2, num_hidden_layers=6,
            layer_types=None, intermediate_size=24, num_experts=4, experts_per_token=2,
            latent_size=8, expert_intermediate_size=8, shared_intermediate_size=8,
            lora_rank=4, delta_backend="reference", expert_backend="reference",
            delta_num_query_heads=2, delta_num_value_heads=1)
        values.pop("num_hidden_layers")
        cls.config = type(cfg)(**values)
        cls.config._name_or_path = cfg._name_or_path
        from kroha_sglang.model import KrohaRoutedNormForCausalLM
        cls.adapter_class = KrohaRoutedNormForCausalLM

    def setUp(self):
        torch.manual_seed(2026)
        cfg = copy.deepcopy(self.config)
        cfg._attn_implementation = "sdpa"
        self.reference = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True).eval()
        Delta.reference = importlib.import_module(type(self.reference).__module__).recurrent_delta_reference
        with patch.dict("sys.modules", mocks()):
            self.adapter = self.adapter_class(copy.deepcopy(self.config))
            self.adapter.load_weights(self.reference.state_dict().items())

    def run_batch(self, segments):
        ids = torch.tensor([token for _, tokens, _ in segments for token in tokens])
        positions = torch.tensor([pos for _, tokens, start in segments for pos in range(start, start + len(tokens))])
        batch = types.SimpleNamespace(segments=[(req, len(tokens), start) for req, tokens, start in segments],
            forward_mode=types.SimpleNamespace(is_target_verify=lambda: False))
        return self.adapter(ids, positions, batch)

    @torch.no_grad()
    def expected(self, sequences):
        return torch.cat([self.reference(torch.tensor([ids]), use_cache=False).logits[0] for ids in sequences])

    def test_packed_prefill_and_independent_virtual_caches(self):
        sequences = [[3, 4, 5, 6, 7], [9, 10, 11]]
        actual = self.run_batch([(0, sequences[0], 0), (1, sequences[1], 0)])
        torch.testing.assert_close(actual, self.expected(sequences), rtol=2e-5, atol=2e-6)
        self.assertEqual(len(self.adapter.model.layers), 6)
        self.assertIsNot(self.adapter.model.layers[1].attn, self.adapter.model.layers[3].attn)
        self.assertIsNone(self.adapter._batch)

    def test_chunked_prefill_decode_and_request_replacement(self):
        first, other = [3, 4, 5, 6, 7], [9, 10, 11]
        self.run_batch([(0, first[:2], 0), (1, other[:1], 0)])
        actual = self.run_batch([(0, first[2:], 2), (1, other[1:], 1)])
        expected = torch.cat((self.expected([first])[2:], self.expected([other])[1:]))
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        actual = self.run_batch([(0, [12], 5), (1, [13], 3)])
        expected = torch.cat((self.expected([first + [12]])[-1:], self.expected([other + [13]])[-1:]))
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        actual = self.run_batch([(0, [21, 22], 0), (1, [14], 4)])
        expected = torch.cat((self.expected([[21, 22]]), self.expected([other + [13, 14]])[-1:]))
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)

    def test_config_metadata_serializes_and_projection_weights_alias(self):
        cfg = copy.deepcopy(self.config)
        before = cfg.to_dict()
        self.assertTrue(supports_config(cfg))
        self.assertEqual(before, cfg.to_dict())
        json.loads(cfg.to_json_string())
        self.assertEqual(cfg.full_attention_layer_ids, [0, 2, 4, 5])
        self.assertEqual(cfg.linear_layer_ids, [1, 3])
        params = cfg.mamba2_cache_params
        self.assertEqual(params.layers, [1, 3])
        self.assertEqual(params.shape.temporal, (2, 8, 8))
        self.assertEqual(params.shape.conv, [(48, 3)])
        for module in self.adapter._physical:
            packed = self.adapter.qkv[module._kroha_projection_key].weight
            self.assertEqual(module.q_proj.weight.untyped_storage().data_ptr(), packed.untyped_storage().data_ptr())
        self.assertIs(self.adapter.hf.model.embed_tokens.weight, self.adapter.hf.lm_head.weight)

    def test_rejects_incomplete_weights_and_unsupported_modes(self):
        with self.assertRaisesRegex(ValueError, "Missing Kroha"):
            self.adapter.load_weights([])
        with patch.dict("sys.modules", mocks(disable_radix_cache=False)):
            with self.assertRaisesRegex(ValueError, "disable-radix-cache"):
                self.adapter_class(copy.deepcopy(self.config))
        cfg = copy.deepcopy(self.config)
        cfg.gdn_version = 2
        with self.assertRaisesRegex(ValueError, "GDN-1"):
            supports_config(cfg)


if __name__ == "__main__":
    unittest.main()
