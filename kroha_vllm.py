"""Out-of-tree Kroha adapter; no project/model-zoo imports or custom GPU kernels.

The HF artifact supplies every learned block. vLLM owns paged GQA, continuous
scheduling, convolution and per-application recurrent cache. FLA supplies GDN-1
with the original HF rounding boundaries. Optional QKV/BAZ fusion uses standard
vLLM linear layers and aliases the original weights, without duplicating them.
"""
import types


def register():
    # Keep the plugin entrypoint light: inspection/worker processes import the
    # implementation lazily through the registry's fully qualified class name.
    from vllm import ModelRegistry
    ModelRegistry.register_model('KrohaForCausalLM','kroha_vllm:KrohaVLLMForCausalLM')
    ModelRegistry.register_model('KrohaRoutedNormForCausalLM','kroha_vllm:KrohaVLLMForCausalLM')


def __getattr__(name):
    if name!='KrohaVLLMForCausalLM':raise AttributeError(name)
    cls=_implementation()
    globals()[name]=cls
    return cls


def _implementation():
    import torch
    from torch import nn
    from torch.nn import functional as F
    from transformers import AutoModelForCausalLM
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    from vllm.forward_context import get_forward_context
    from vllm.model_executor.layers.attention import Attention
    from vllm.model_executor.layers.linear import QKVParallelLinear,MergedColumnParallelLinear
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention
    from vllm.model_executor.layers.mamba.mamba_utils import (
        MambaStateDtypeCalculator,MambaStateShapeCalculator,MambaStateCopyFuncCalculator,
        is_conv_state_dim_first)
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn,causal_conv1d_update
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule,fused_recurrent_gated_delta_rule

    def expand(value,source_heads,target_heads,dim):
        return value.reshape(*value.shape[:-1],source_heads,dim).repeat_interleave(
            target_heads//source_heads,-2).flatten(-2)

    def conv_weights(module):
        counts=(module.num_query_heads,module.num_kv_heads,module.num_value_heads)
        parts=module.conv1d.weight.split([h*module.head_dim for h in counts],0)
        return torch.cat([w.reshape(h,module.head_dim,*w.shape[1:]).repeat_interleave(
            module.num_heads//h,0).flatten(0,1)for w,h in zip(parts,counts)],0)

    def delta_core(op,mixed,b,a,output,positions):
        raw=get_forward_context().attn_metadata
        meta=raw.get(op.prefix)if isinstance(raw,dict)else None
        if meta is None:
            if not getattr(op,'_kroha_warmed',False):
                op._kroha_warmed=True
                q=torch.zeros((1,64,op.num_k_heads,op.head_k_dim),device=mixed.device,dtype=mixed.dtype)
                chunk_gated_delta_rule(q,q,q,g=torch.full(q.shape[:-1],-.1,device=q.device,dtype=torch.float32),
                    beta=torch.full(q.shape[:-1],.5,device=q.device,dtype=q.dtype),use_qk_l2norm_in_kernel=True)
            output.zero_();return
        if meta.spec_sequence_masks is not None:raise ValueError('Kroha speculative decoding is not supported')
        count=meta.num_actual_tokens
        mixed,a,b=mixed[:count],a[:count],b[:count]
        conv=op.kv_cache[0]if is_conv_state_dim_first()else op.kv_cache[0].transpose(-1,-2)
        state=op.kv_cache[1]
        indices=meta.non_spec_state_indices_tensor
        cu=meta.non_spec_query_start_loc
        fresh=positions[cu[:-1].long()]==0
        conv[indices]=torch.where(fresh[:,None,None],0,conv[indices])
        weight=op.conv1d.weight.squeeze(1)
        if meta.num_prefills:
            mixed=causal_conv1d_fn(mixed.transpose(0,1),weight,None,activation=None,
                conv_states=conv,has_initial_state=meta.has_initial_state,cache_indices=indices,
                query_start_loc=cu,metadata=meta).transpose(0,1)
        else:
            mixed=causal_conv1d_update(mixed,conv,weight,None,None,
                conv_state_indices=indices[:count],validate_data=True)
        mixed=F.silu(mixed).contiguous()
        q,k,v=op.rearrange_mixed_qkv(mixed)
        g=-op.A_log.float().exp()*F.softplus(a.float()+op.dt_bias)
        beta=b.sigmoid()
        g,beta=g.unsqueeze(0),beta.unsqueeze(0)
        initial=state[indices].transpose(-1,-2).contiguous()
        initial=torch.where(fresh[:,None,None,None],0,initial)
        d,t=meta.num_decodes,meta.num_decode_tokens
        for begin,end,first,last,kernel in (
            (0,t,0,d,fused_recurrent_gated_delta_rule),
            (t,count,d,len(indices),chunk_gated_delta_rule)):
            if end==begin:continue
            take=lambda x:x[:,begin:end].contiguous()
            out,final=kernel(q=take(q),k=take(k),v=take(v),g=take(g),beta=take(beta),
                initial_state=initial[first:last].contiguous(),output_final_state=True,
                cu_seqlens=cu[first:last+1]-begin,use_qk_l2norm_in_kernel=True)
            state[indices[first:last]]=final.transpose(-1,-2).to(state.dtype)
            output[begin:end].copy_(out.squeeze(0))

    class KrohaVLLMForCausalLM(nn.Module):
        is_hybrid=True
        has_inner_state=True

        def __init__(self,*,vllm_config,prefix=''):
            super().__init__()
            c=vllm_config.model_config.hf_config
            if c.model_type not in ('kroha', 'kroha_routed_norm'):
                raise ValueError('A standalone Kroha HF artifact is required')
            if c.model_type == 'kroha_routed_norm' and not c.routed_output_norm:
                raise ValueError('The routed RMSNorm configuration is required')
            if getattr(c,'gdn_version',None)!=1 or not hasattr(c,'linear_num_key_heads'):
                raise ValueError('Re-export Kroha with the current standalone code and GDN-1 cache metadata')
            if vllm_config.parallel_config.tensor_parallel_size!=1 or vllm_config.parallel_config.pipeline_parallel_size!=1:
                raise ValueError('Kroha currently supports a single GPU')
            if vllm_config.speculative_config is not None:raise ValueError('Kroha speculative decoding is not supported')
            if vllm_config.quant_config is not None:raise ValueError('Packed-expert quantization needs a separate validated backend')
            if vllm_config.cache_config.enable_prefix_caching:
                raise ValueError('Use --no-enable-prefix-caching; recurrent prefix checkpoints are not yet certified')
            if not vllm_config.model_config.enforce_eager:
                raise ValueError('Use --enforce-eager; dynamic routed dispatch is not a whole-model CUDA graph')
            self.config=c
            options=vllm_config.additional_config or {}
            self.fuse_projections=options.get('kroha_fused_projections',True)
            self.compile_regions=options.get('kroha_compile_regions',False)
            c._attn_implementation='sdpa'
            self.hf=AutoModelForCausalLM.from_config(c,trust_remote_code=True,
                dtype=vllm_config.model_config.dtype).eval()
            for module in self.hf.modules():
                if hasattr(module,'router')and isinstance(module.router,nn.Linear):module.router.float()
                for name in ('A_log','dt_bias'):
                    parameter=getattr(module,name,None)
                    if isinstance(parameter,nn.Parameter):parameter.data=parameter.data.float()
            # Official HF mask registry: the native vLLM backends already own
            # sequence visibility, so no dense flattened T*T mask is allocated.
            ALL_MASK_ATTENTION_FUNCTIONS.register('kroha_vllm',lambda **kwargs:None)
            c._attn_implementation='kroha_vllm'
            self.hf.config._attn_implementation='kroha_vllm'
            self.paged_attention=nn.ModuleDict()
            self.delta_attention=nn.ModuleDict()
            self.qkv=nn.ModuleDict();self.baz=nn.ModuleDict()
            for slot,kind in enumerate(c.layer_types):
                name=f'{prefix}virtual_layers.{slot}.attn'
                if kind=='full_attention':
                    self.paged_attention[str(slot)]=Attention(c.num_attention_heads,c.head_dim,c.head_dim**-.5,
                        num_kv_heads=c.num_key_value_heads,cache_config=vllm_config.cache_config,
                        quant_config=None,prefix=name)
                elif kind=='linear_attention':
                    op=QwenGatedDeltaNetAttention(c,vllm_config,prefix=name)
                    for attr in ('in_proj_qkvz','in_proj_ba','norm','out_proj'):
                        if hasattr(op,attr):delattr(op,attr)
                    self.delta_attention[str(slot)]=op
                else:raise ValueError('Unsupported Kroha layer type: '+kind)
            modules=list(self.hf.model.modules())
            self._physical=[]
            for module in modules:
                if hasattr(module,'gated_norm')and hasattr(module,'A_log'):
                    module._kroha_projection_key=str(len(self._physical));self._physical.append(module)
                    module.forward=types.MethodType(self._delta_forward,module)
                elif all(hasattr(module,n)for n in ('q_proj','k_proj','v_proj','o_proj','head_gate')):
                    module._kroha_projection_key=str(len(self._physical));self._physical.append(module)
                    module.forward=types.MethodType(self._attention_forward,module)

        @classmethod
        def get_mamba_state_dtype_from_config(cls,vllm_config):
            c=vllm_config.cache_config
            return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
                vllm_config.model_config.dtype,c.mamba_cache_dtype,c.mamba_ssm_cache_dtype)

        @classmethod
        def get_mamba_state_shape_from_config(cls,vllm_config):
            c=vllm_config.model_config.hf_config
            return MambaStateShapeCalculator.gated_delta_net_state_shape(1,c.linear_num_key_heads,
                c.linear_num_value_heads,c.linear_key_head_dim,c.linear_value_head_dim,c.linear_conv_kernel_dim,0)

        @classmethod
        def get_mamba_state_copy_func(cls):return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()

        def _project(self,module,x):
            key=module._kroha_projection_key
            if key in self.qkv:
                result=self.qkv[key](x)[0]
                return result.split((module.q_proj.out_features,module.k_proj.out_features,module.v_proj.out_features),-1)
            return module.q_proj(x),module.k_proj(x),module.v_proj(x)

        def _attention_forward(self,module,hidden_states,position_embeddings,attention_mask=None,
                past_key_values=None,cache_layer_idx=None,q_adapter=None,v_adapter=None,**kwargs):
            if past_key_values is not None:raise ValueError('vLLM owns request caches')
            q,k,v=self._project(module,hidden_states)
            if q_adapter is not None:q,v=q+q_adapter(hidden_states),v+v_adapter(hidden_states)
            shape=(*hidden_states.shape[:-1],-1,module.head_dim)
            q,k,v=(t.view(shape).transpose(1,2)for t in(q,k,v))
            if module.q_norm is not None:q,k=module.q_norm(q),module.k_norm(k)
            if module.use_rope:q,k=apply_rotary_pos_emb(q,k,*position_embeddings)
            flatten=lambda t:t.transpose(1,2).reshape(-1,t.shape[1]*t.shape[-1])
            slot=module.layer_idx if cache_layer_idx is None else cache_layer_idx
            out=self.paged_attention[str(slot)](flatten(q),flatten(k),flatten(v))
            out=out.reshape(*hidden_states.shape[:-1],module.num_heads,module.head_dim)
            out=module.gate_heads(out,hidden_states)
            out=module.o_proj(out.flatten(-2))
            return module.output_gate(out,hidden_states)if module.output_gate is not None else out

        def _delta_forward(self,module,x,position_embeddings=None,attention_mask=None,past_key_values=None,
                cache_layer_idx=None,q_adapter=None,v_adapter=None,document_ids=None,**kwargs):
            if past_key_values is not None or document_ids is not None:raise ValueError('vLLM owns request caches/isolation')
            q,k,v=self._project(module,x)
            if q_adapter is not None:q,v=q+q_adapter(x),v+v_adapter(x)
            h,d=module.num_heads,module.head_dim
            mixed=torch.cat([expand(q,module.num_query_heads,h,d),expand(k,module.num_kv_heads,h,d),
                             expand(v,module.num_value_heads,h,d)],-1).reshape(-1,3*h*d)
            key=module._kroha_projection_key
            if key in self.baz:
                b,a,z=self.baz[key](x)[0].split((h,h,h*d),-1)
            else:b,a,z=module.b_proj(x),module.a_proj(x),module.z_proj(x)
            out=torch.empty((mixed.shape[0],h,d),device=x.device,dtype=x.dtype)
            delta_core(self.delta_attention[str(cache_layer_idx)],mixed,b.reshape(-1,h),a.reshape(-1,h),out,self._positions)
            out=out.reshape(*x.shape[:2],h,d)
            out=module.gated_norm(out)*z.reshape_as(out).float().sigmoid().to(out.dtype)
            out=module.o_proj(out.flatten(-2))
            return module.output_gate(out,x)if module.output_gate is not None else out

        def embed_input_ids(self,input_ids):return self.hf.model.embed_tokens(input_ids)

        def forward(self,input_ids,positions,intermediate_tensors=None,inputs_embeds=None):
            if intermediate_tensors is not None:raise ValueError('Kroha supports one pipeline stage')
            self._positions=positions
            return self.hf.model(input_ids=input_ids[None]if input_ids is not None else None,
                inputs_embeds=inputs_embeds[None]if inputs_embeds is not None else None,
                position_ids=positions[None],use_cache=False).last_hidden_state[0]

        def compute_logits(self,hidden_states):return self.hf.lm_head(hidden_states)

        def _fuse(self):
            if not self.fuse_projections:return
            for module in self._physical:
                key=module._kroha_projection_key
                h=module.num_heads;d=module.head_dim
                kv=module.num_kv_heads
                fused=QKVParallelLinear(self.config.hidden_size,d,h,kv,bias=False,
                    params_dtype=module.q_proj.weight.dtype,quant_config=None,prefix='kroha_qkv_'+key).to(module.q_proj.weight.device)
                fused.weight.data.copy_(torch.cat([m.weight for m in(module.q_proj,module.k_proj,module.v_proj)]))
                offset=0
                for original in (module.q_proj,module.k_proj,module.v_proj):
                    rows=original.weight.shape[0]
                    original.weight.data=fused.weight.data[offset:offset+rows];offset+=rows
                self.qkv[key]=fused
                if hasattr(module,'z_proj'):
                    fused=MergedColumnParallelLinear(self.config.hidden_size,[h,h,h*d],bias=False,
                        params_dtype=module.b_proj.weight.dtype,quant_config=None,prefix='kroha_baz_'+key).to(module.b_proj.weight.device)
                    fused.weight.data.copy_(torch.cat([m.weight for m in(module.b_proj,module.a_proj,module.z_proj)]))
                    offset=0
                    for original in(module.b_proj,module.a_proj,module.z_proj):
                        rows=original.weight.shape[0]
                        original.weight.data=fused.weight.data[offset:offset+rows];offset+=rows
                    self.baz[key]=fused

        def load_weights(self,weights):
            targets=dict(self.hf.named_parameters(remove_duplicate=False));targets.update(self.hf.named_buffers())
            loaded=set()
            for name,value in weights:
                if name not in targets:raise ValueError('Unknown Kroha tensor: '+name)
                target=targets[name]
                if target.shape!=value.shape:raise ValueError('Kroha checkpoint shape differs: '+name)
                target.data.copy_(value.to(device=target.device,dtype=target.dtype));loaded.add(id(target))
            missing=[name for name,p in self.hf.named_parameters()if id(p)not in loaded]
            if missing:raise ValueError('Missing Kroha weights: '+str(missing))
            for slot_text,op in self.delta_attention.items():
                source=self.hf.model.layers[(int(slot_text)-1)%self.config.num_core_layers].self_attn
                op.conv1d.weight.data.copy_(conv_weights(source))
                op.A_log=source.A_log;op.dt_bias=source.dt_bias
            self._fuse()
            self.compiled_module_count=self.hf.enable_optimizations()if self.compile_regions else 0
            return {'hf.'+name for name in self.hf.state_dict()if name in targets}

    KrohaVLLMForCausalLM.__module__=__name__
    KrohaVLLMForCausalLM.__qualname__='KrohaVLLMForCausalLM'
    return KrohaVLLMForCausalLM
