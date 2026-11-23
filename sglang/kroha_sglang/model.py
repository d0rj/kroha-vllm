"""Native RadixAttention/GDN state, with the learned HF Kroha blocks preserved."""
import types

import torch
from torch import nn
from transformers import AutoModelForCausalLM
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from . import supports_config


def expand_heads(value, heads, target, dim):
    return value.reshape(*value.shape[:-1], heads, dim).repeat_interleave(
        target // heads, -2).flatten(-2)


def expanded_conv(module):
    heads = (module.num_query_heads, module.num_kv_heads, module.num_value_heads)
    parts = module.conv1d.weight.split([h * module.head_dim for h in heads], 0)
    return torch.cat([
        weight.reshape(h, module.head_dim, *weight.shape[1:]).repeat_interleave(
            module.num_heads // h, 0).flatten(0, 1)
        for weight, h in zip(parts, heads)
    ], 0).squeeze(1)


class KrohaForCausalLM(nn.Module):
    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        from sglang.srt.layers.linear import QKVParallelLinear, MergedColumnParallelLinear
        from sglang.srt.layers.logits_processor import LogitsProcessor
        from sglang.srt.layers.radix_attention import RadixAttention
        from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
        from sglang.srt.runtime_context import get_exec, get_parallel, get_server_args

        if not supports_config(config):
            raise ValueError("A standalone Kroha HF artifact is required")
        parallel, args = get_parallel(), get_server_args()
        if parallel.tp_size != 1 or parallel.pp_size != 1:
            raise ValueError("Kroha currently supports a single GPU")
        if quant_config is not None or args.speculative_algorithm is not None:
            raise ValueError("Kroha quantization and speculative decoding are not validated")
        if not args.disable_radix_cache or not args.disable_overlap_schedule:
            raise ValueError("Use --disable-radix-cache --disable-overlap-schedule")
        graph = get_exec().graph.cuda_graph_config
        if graph.decode.backend != "disabled" or graph.prefill.backend != "disabled":
            raise ValueError("Use --cuda-graph-backend-decode disabled --cuda-graph-backend-prefill disabled")

        self.config = config
        config._attn_implementation = "sdpa"
        # HF otherwise honors the checkpoint's FP32 dtype over SGLang's BF16 context.
        self.hf = AutoModelForCausalLM.from_config(
            config, trust_remote_code=True, dtype=torch.get_default_dtype()).eval()
        for module in self.hf.modules():
            if hasattr(module, "router") and isinstance(module.router, nn.Linear):
                module.router.float()
            for name in ("A_log", "dt_bias"):
                parameter = getattr(module, name, None)
                if isinstance(parameter, nn.Parameter):
                    parameter.data = parameter.data.float()
        ALL_MASK_ATTENTION_FUNCTIONS.register("kroha_sglang", lambda **kwargs: None)
        config._attn_implementation = "kroha_sglang"
        self.hf.config._attn_implementation = "kroha_sglang"
        self.model = nn.Module()
        self.model.embed_tokens = self.hf.model.embed_tokens
        self.model.layers = nn.ModuleList()
        self.lm_head = self.hf.lm_head
        self.logits_processor = LogitsProcessor(config)
        self.start_layer, self.end_layer = 0, config.num_hidden_layers
        self._batch = None
        for slot, kind in enumerate(config.layer_types):
            if kind == "full_attention":
                op = RadixAttention(config.num_attention_heads, config.head_dim,
                    config.head_dim ** -.5, config.num_key_value_heads, slot)
            else:
                op = RadixLinearAttention(slot, config.num_attention_heads,
                    config.num_attention_heads, config.num_attention_heads,
                    config.head_dim, config.head_dim, config.head_dim,
                    activation="silu")
            layer = nn.Module()
            layer.attn = op
            self.model.layers.append(layer)

        self.qkv, self.baz = nn.ModuleDict(), nn.ModuleDict()
        self._physical = []
        for module in list(self.hf.model.modules()):
            if hasattr(module, "gated_norm") and hasattr(module, "A_log"):
                module._kroha_projection_key = str(len(self._physical))
                self._physical.append(module)
                module.forward = types.MethodType(self._delta_forward, module)
            elif all(hasattr(module, n) for n in ("q_proj", "k_proj", "v_proj", "o_proj", "head_gate")):
                module._kroha_projection_key = str(len(self._physical))
                self._physical.append(module)
                module.forward = types.MethodType(self._attention_forward, module)
        self._qkv_class, self._baz_class = QKVParallelLinear, MergedColumnParallelLinear

    def _project(self, module, x):
        key = module._kroha_projection_key
        if key in self.qkv:
            return self.qkv[key](x)[0].split(tuple(m.out_features for m in (
                module.q_proj, module.k_proj, module.v_proj)), -1)
        return module.q_proj(x), module.k_proj(x), module.v_proj(x)

    def _attention_forward(self, module, hidden_states, position_embeddings,
            attention_mask=None, past_key_values=None, cache_layer_idx=None,
            q_adapter=None, v_adapter=None, **kwargs):
        if past_key_values is not None:
            raise ValueError("SGLang owns request caches")
        q, k, v = self._project(module, hidden_states)
        if q_adapter is not None:
            q, v = q + q_adapter(hidden_states), v + v_adapter(hidden_states)
        shape = (*hidden_states.shape[:-1], -1, module.head_dim)
        q, k, v = (t.view(shape).transpose(1, 2) for t in (q, k, v))
        if module.q_norm is not None:
            q, k = module.q_norm(q), module.k_norm(k)
        if module.use_rope:
            q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
        flatten = lambda t: t.transpose(1, 2).reshape(-1, t.shape[1] * t.shape[-1])
        slot = module.layer_idx if cache_layer_idx is None else cache_layer_idx
        out = self.model.layers[slot].attn(flatten(q), flatten(k), flatten(v), self._batch)
        out = out.reshape(*hidden_states.shape[:-1], module.num_heads, module.head_dim)
        out = module.o_proj(module.gate_heads(out, hidden_states).flatten(-2))
        return module.output_gate(out, hidden_states) if module.output_gate is not None else out

    def _delta_forward(self, module, x, position_embeddings=None, attention_mask=None,
            past_key_values=None, cache_layer_idx=None, q_adapter=None, v_adapter=None,
            document_ids=None, **kwargs):
        if past_key_values is not None or document_ids is not None:
            raise ValueError("SGLang owns request caches and isolation")
        q, k, v = self._project(module, x)
        if q_adapter is not None:
            q, v = q + q_adapter(x), v + v_adapter(x)
        h, d = module.num_heads, module.head_dim
        mixed = torch.cat([
            expand_heads(q, module.num_query_heads, h, d),
            expand_heads(k, module.num_kv_heads, h, d),
            expand_heads(v, module.num_value_heads, h, d),
        ], -1).reshape(-1, 3 * h * d)
        key = module._kroha_projection_key
        if key in self.baz:
            b, a, z = self.baz[key](x)[0].split((h, h, h * d), -1)
        else:
            b, a, z = module.b_proj(x), module.a_proj(x), module.z_proj(x)
        out = self.model.layers[cache_layer_idx].attn(
            self._batch, mixed_qkv=mixed, a=a.reshape(-1, h), b=b.reshape(-1, h))
        out = out.reshape(*x.shape[:2], h, d)
        out = module.gated_norm(out) * z.reshape_as(out).float().sigmoid().to(out.dtype)
        out = module.o_proj(out.flatten(-2))
        return module.output_gate(out, x) if module.output_gate is not None else out

    @torch.no_grad()
    def forward(self, input_ids, positions, forward_batch, inputs_embeds=None, **kwargs):
        if forward_batch.forward_mode.is_target_verify():
            raise ValueError("Kroha speculative decoding is not supported")
        self._batch = forward_batch
        try:
            hidden = self.hf.model(
                input_ids=input_ids[None] if input_ids is not None else None,
                inputs_embeds=inputs_embeds[None] if inputs_embeds is not None else None,
                position_ids=positions[None], use_cache=False).last_hidden_state[0]
            return self.logits_processor(input_ids, hidden, self.lm_head, forward_batch)
        finally:
            self._batch = None

    def get_input_embeddings(self):
        return self.hf.model.embed_tokens

    def get_embed_and_head(self):
        return self.hf.model.embed_tokens.weight, self.hf.lm_head.weight

    def load_weights(self, weights):
        targets = dict(self.hf.named_parameters(remove_duplicate=False))
        targets.update(self.hf.named_buffers())
        loaded = set()
        for name, value in weights:
            if name not in targets or targets[name].shape != value.shape:
                raise ValueError("Unknown or incompatible Kroha tensor: " + name)
            target = targets[name]
            target.data.copy_(value.to(device=target.device, dtype=target.dtype))
            loaded.add(id(target))
        missing = [name for name, p in self.hf.named_parameters() if id(p) not in loaded]
        if missing:
            raise ValueError("Missing Kroha weights: " + str(missing))
        for slot in self.config.linear_layer_ids:
            source = self.hf.model.layers[(slot - 1) % self.config.num_core_layers].self_attn
            op = self.model.layers[slot].attn
            op.conv_weights = expanded_conv(source)
            op.A_log, op.dt_bias = source.A_log, source.dt_bias
        if not self.qkv:
            for module in self._physical:
                key = module._kroha_projection_key
                fused = self._qkv_class(self.config.hidden_size, module.head_dim,
                    module.num_heads, module.num_kv_heads, bias=False,
                    params_dtype=module.q_proj.weight.dtype)
                fused.weight.data = torch.cat([m.weight for m in (
                    module.q_proj, module.k_proj, module.v_proj)])
                offset = 0
                for original in (module.q_proj, module.k_proj, module.v_proj):
                    rows = original.weight.shape[0]
                    original.weight.data = fused.weight.data[offset:offset + rows]
                    offset += rows
                self.qkv[key] = fused
                if hasattr(module, "z_proj"):
                    h, d = module.num_heads, module.head_dim
                    fused = self._baz_class(self.config.hidden_size, [h, h, h * d],
                        bias=False, params_dtype=module.b_proj.weight.dtype)
                    fused.weight.data = torch.cat([m.weight for m in (
                        module.b_proj, module.a_proj, module.z_proj)])
                    offset = 0
                    for original in (module.b_proj, module.a_proj, module.z_proj):
                        rows = original.weight.shape[0]
                        original.weight.data = fused.weight.data[offset:offset + rows]
                        offset += rows
                    self.baz[key] = fused
        return set(targets)


class KrohaRoutedNormForCausalLM(KrohaForCausalLM):
    pass


EntryClass = [KrohaForCausalLM, KrohaRoutedNormForCausalLM]
