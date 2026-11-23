# Kroha native SGLang adapter

External model package for **SGLang 0.5.23**, distributed in the `sglang/`
directory of [d0rj/kroha-vllm](https://github.com/d0rj/kroha-vllm)
(currently private). It supports the GDN-1 `KrohaForCausalLM` and
`KrohaRoutedNormForCausalLM` architectures without modifying SGLang core or the
two Python files of the standalone HF model.

Native GPU checks now cover the completed 36B Kroha GDN-1 checkpoint on
RTX 5070 Ti, BF16, with a 16,384-token window. They include mixed-length text
and explicit DNA6 IDs, 512-token chunked prefill, cached decode, fixed-prefix
repeats and request-slot reuse. BF16 results are **not bit-identical to HF**:
nearby token probabilities can exchange rank. See the
[GPU validation receipt](https://github.com/d0rj/kroha-vllm/blob/main/validation/kroha-adapters-gpu-20261123.json).

The first GPU run exposed an FP32/BF16 loading error. The adapter now explicitly
uses SGLang's runtime dtype while retaining FP32 routers and decay parameters.
The regression is covered by the CPU tests. These bounded checks do not certify
other runtimes, hardware, model revisions or unsupported serving modes.

## Install

Use a dedicated Linux / WSL2 environment. SGLang 0.5.23 requires a different
runtime from the vLLM 0.32.x adapter; install the two adapters in separate
environments. For a fresh Python 3.12 environment:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install --prerelease=allow --torch-backend cu130 'sglang==0.5.23'
# From the root of the authenticated private GitHub checkout:
uv pip install --no-deps ./sglang
```

In a `tiny_llm` source checkout, the package path instead is
`integrations/sglang-kroha`. Pin the adapter commit and preserve the runtime
package versions for reproducible deployment.

## Native engine integration

SGLang supplies `RadixAttention`, `RadixLinearAttention`, its hybrid GDN cache
pool, Triton GDN kernels, packed linear projections and `LogitsProcessor`.
Kroha's HF model supplies all learned blocks, including latent MoE, routed
RMSNorm, SharedGate, HeadGate, Block AttnRes and the second-pass Q/V adapters.
Every virtual application has its own native cache slot. Shared weights remain
shared. Packed projections alias the original projection weights after loading.

The external linear-attention registry describes the hybrid cache layout.
Runtime-only descriptors supply cache metadata without serializing extra
objects into `config.json` or changing the HF model's AutoClass mapping.

The initial implementation supports one GPU and ordinary causal generation.
Prefix reuse, speculative decoding, quantization, overlapping scheduling and
CUDA graphs are rejected until they have their own validation. It uses the
native GDN kernels. Their BF16 rounding differs from the HF path, so exact
greedy sequences can differ at low-margin token choices.

## Serve the validated checkpoint

```bash
export SGLANG_EXTERNAL_MODEL_PACKAGE=kroha_sglang
export SGLANG_MAMBA_SSM_DTYPE=float32
export SGLANG_MAMBA_CONV_DTYPE=bfloat16
python -m sglang.launch_server \
  --model-path /path/to/standalone/kroha --trust-remote-code \
  --served-model-name Kroha --dtype bfloat16 --host 127.0.0.1 \
  --context-length 16384 --attention-backend triton \
  --linear-attn-decode-backend triton --linear-attn-prefill-backend triton \
  --disable-radix-cache --disable-overlap-schedule \
  --cuda-graph-backend-decode disabled --cuda-graph-backend-prefill disabled \
  --max-running-requests 16 --max-total-tokens 32768 \
  --mem-fraction-static 0.6 --chunked-prefill-size 512
```

The model path must contain a complete HF export. A longer model training
context does not establish native serving correctness at that context length.
Text/chat/DNA tokenization remains the responsibility of the HF tokenizer and
its explicit DNA6 codec. This plugin does not implement MTP drafting.

## CPU wiring checks

Point `KROHA_TEST_MODEL` at any current standalone HF Kroha export. The tests
derive tiny, randomly initialized models from its configuration and load only
the exported Python code; they do not load or modify its trained weights.

```bash
CUDA_VISIBLE_DEVICES='' SGLANG_USE_CPU_ENGINE=1 \
  KROHA_TEST_MODEL=/path/to/standalone/kroha \
  python -m unittest discover -s sglang/tests -v
```

Use `integrations/sglang-kroha/tests` in the training project checkout.
GPU validation receipts record the exact checkpoint, runtime versions, adapter
hashes and comparison scope. They cover ordinary causal serving without prefix
reuse or graphs. A separate workload benchmark is required for throughput claims.
