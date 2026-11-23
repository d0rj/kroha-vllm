"""SGLang external model package; HF artifacts retain their original two files."""
from importlib.metadata import version


def supports_config(config):
    if getattr(config, "model_type", None) not in ("kroha", "kroha_routed_norm"):
        return False
    if getattr(config, "gdn_version", None) != 1:
        raise ValueError("The Kroha SGLang adapter requires GDN-1")
    expected = ["full_attention"] + [
        "linear_attention" if i % 2 == 0 else "full_attention"
        for _ in range(config.num_passes) for i in range(config.num_core_layers)
    ] + ["full_attention"]
    if list(config.layer_types) != expected or config.num_hidden_layers != len(expected):
        raise ValueError("Kroha cache slots must describe every virtual application")
    if config.model_type == "kroha_routed_norm" and not config.routed_output_norm:
        raise ValueError("KrohaRoutedNorm requires routed-output RMSNorm")
    # Runtime-only descriptors do not enter config.json or change AutoClass dispatch.
    cls = type(config)
    cls.full_attention_layer_ids = property(
        lambda c: [i for i, kind in enumerate(c.layer_types) if kind == "full_attention"])
    cls.linear_layer_ids = property(
        lambda c: [i for i, kind in enumerate(c.layer_types) if kind == "linear_attention"])
    cls.mamba2_cache_params = property(cache_params)
    return True


def cache_params(config):
    from sglang.srt.configs.mamba_utils import (
        Mamba2CacheParams, Mamba2StateShape, mamba2_state_dtype,
    )
    heads, dim = config.num_attention_heads, config.head_dim
    shape = Mamba2StateShape.create(
        tp_world_size=1, intermediate_size=heads * dim, n_groups=heads,
        num_heads=heads, head_dim=dim, state_size=dim,
        conv_kernel=config.delta_conv_kernel, conv_shard_groups=[heads * dim] * 3,
    )
    return Mamba2CacheParams(
        shape=shape, layers=config.linear_layer_ids, dtype=mamba2_state_dtype(config))


def register():
    if version("sglang") != "0.5.23":
        raise RuntimeError("kroha-sglang currently targets SGLang 0.5.23")
    from transformers import PretrainedConfig
    from sglang.srt.configs.linear_attn_model_registry import (
        LinearAttnModelSpec, get_linear_attn_spec_by_arch, register_linear_attn_model,
    )
    if get_linear_attn_spec_by_arch("KrohaRoutedNormForCausalLM") is None:
        register_linear_attn_model(LinearAttnModelSpec(
            config_class=PretrainedConfig,
            config_predicate=supports_config,
            backend_class_name="sglang.srt.layers.attention.linear.gdn_backend.GDNAttnBackend",
            arch_names=["KrohaForCausalLM", "KrohaRoutedNormForCausalLM"],
            uses_mamba_radix_cache=True, support_mamba_cache=True,
        ))


register()
