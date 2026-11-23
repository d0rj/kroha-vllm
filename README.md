# Kroha native vLLM adapter

Standalone Python plugin for serving Kroha GDN-1 models with **vLLM 0.32.x**.
The canonical repository is [d0rj/kroha-vllm](https://github.com/d0rj/kroha-vllm)
(currently private). The package does not import `tiny_llm`.

The plugin registers `KrohaForCausalLM` and `KrohaRoutedNormForCausalLM` through
the standard `vllm.general_plugins` entry point. The latter is the architecture
of the planned [Kroha-base](https://huggingface.co/d0rj/Kroha-base) release.
It loads the complete HF implementation from the model artifact, including
artifacts with only `configuration_kroha.py` and `modeling_kroha.py`.

HF hosts the weights, tokenizer and model implementation. This package supplies
the native engine integration; copying its Python file into an HF model
repository does not install or register a vLLM plugin.

The same GitHub repository also contains an independently packaged
[SGLang adapter](https://github.com/d0rj/kroha-vllm/tree/main/sglang).
It targets SGLang 0.5.23 and is **experimental pending native GPU validation**.
Its configuration, weight transfer and cache wiring passed CPU checks. Install
it in a separate environment, following its own README; this root package
continues to install only the vLLM adapter.

## Installation

Use a separate **Linux / WSL2** serving environment with Python 3.12 or later.
The existing validation used vLLM 0.32.x, PyTorch 2.13.0+cu130,
Transformers 5.17.0 and FLA 0.5.2 on RTX 5070 Ti. Other runtime versions
require validation.

With that compatible runtime already installed and GitHub access to this
private repository:

```bash
gh repo clone d0rj/kroha-vllm
cd kroha-vllm
python -m pip install --no-deps .
python -m pip check
```

`--no-deps` preserves the existing serving runtime. It requires the dependencies
in `pyproject.toml` to be installed beforehand. For reproducible deployments,
check out the chosen commit before installing and record `git rev-parse HEAD`.
The plugin is discovered automatically in each vLLM process. If `VLLM_PLUGINS`
is set, its comma-separated list must include `kroha`.

## Serve

Use a trained standalone HF export, or replace its path with a released HF model
ID once the weights are available. The conservative baseline is:

```bash
export VLLM_GDN_DECODE_KERNEL=triton
export VLLM_WORKER_MULTIPROC_METHOD=spawn

vllm serve /path/to/kroha --served-model-name Kroha \
  --trust-remote-code --dtype bfloat16 --host 127.0.0.1 \
  --enforce-eager --no-enable-prefix-caching --enable-chunked-prefill \
  --no-async-scheduling --mamba-ssm-cache-dtype float32 \
  --max-model-len 2048 --max-num-seqs 128 --max-num-batched-tokens 8192 \
  --kv-cache-memory-bytes 1073741824 \
  --additional-config '{"gdn_prefill_backend":"triton","kroha_fused_projections":true,"kroha_compile_regions":false}'
```

The documented serving checks cover **2,048-token context**. This command
deliberately uses that tested limit even for a model trained with a longer
context. Native serving validation of the final 16K checkpoint is pending.

## Implementation

The adapter retains the trained recursion, softplus/sqrt router, clipped SwiGLU,
shared expert, HeadGate, Block AttnRes and routed-output RMSNorm where configured.
vLLM supplies paged GQA, continuous batching, hybrid cache allocation,
convolution kernels and packed QKV/BAZ linear layers. FLA 0.5.2 supplies the
original GDN-1 recurrence, including Kroha's output gate and rounding boundaries.
Each virtual layer application has its own request state despite shared weights.
Fused projection weights alias the HF tensors instead of retaining duplicate
weights.

Supported: one GPU, BF16, mixed-length requests and chunked prefill. Prefix
caching, quantization, speculative decoding and whole-model CUDA graphs are
rejected explicitly. Regional compilation is experimental and disabled by
default; enable it only after checking generation parity for the intended model
and workload.

Earlier validation of the trained 32k HeadGate reference matched HF on all
72 greedy tokens across six prompts in the uncompiled baseline. Same-batch
repeats and replacing a neighbouring request preserved the tested output.
BF16 results can differ across batch shapes: one low-margin token changed when
batch size changed. The compiled path matched 70/72 tokens and still requires
an additional fixed-prefix cache check. These checks do not establish model
quality or certify a new checkpoint or runtime.

## Tokenization

Text and chat use the model artifact's HF tokenizer and template. For DNA6,
load `DNACodec` from the artifact's `modeling_kroha` module and pass its explicit
IDs through vLLM's `prompt_token_ids`, or an integer `prompt` to the
OpenAI-compatible completions endpoint. The tokenizer decodes DNA IDs.
A plain string with DNA tags does not enable DNA6 encoding automatically.
The plugin does not add chat or DNA abilities to model weights.

## Source layout

`kroha_vllm.py` is the complete engine adapter; `pyproject.toml` packages it and
declares the registration entry point. Model weights and training state belong
in their HF repositories and are not included here.

The `tiny_llm` training project retains a compatibility copy under
`integrations/vllm-kroha` for its existing local launchers. Those launchers
continue to use their local source without downloading changes from GitHub.
