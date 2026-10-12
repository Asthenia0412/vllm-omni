"""Default-off native BF16 self-Q/K adapter for an isolated frozen candidate.

It uses the exact combined implementation screened in native BF16 QK v3.
RMSNorm reduces all5120 features before40x128 head reshape; V is a view of
the unchanged projection. Coefficients are expanded once per model forward,
then shared by40 layers. No device/global JIT policy is changed here.
"""
from collections.abc import Mapping
from dataclasses import dataclass
import math
import os


def _flag(value):
    if type(value) is bool:return value
    if type(value) is str and value in ('0','1'):return value=='1'
    raise ValueError('lingbot_npu_native_bf16_qk must be bool (environment0/1)')


def _option_text(value):
    return value.decode('ascii','backslashreplace') if isinstance(value,bytes) else value


def read_jit_policy(backend):
    """Read both official raw and effective state after existing NPU init."""
    if not backend.npu.is_initialized():
        raise RuntimeError('Read native QK JIT policy only after existing NPU initialization')
    raw=backend._C._npu_getOption('jitCompile')
    public=backend.npu.is_jit_compile_false()
    if type(public) is not bool:
        raise RuntimeError('Official effective JIT getter returned an unknown state')
    # Raw empty/unset is preserved; never substitute an assumed raw disable.
    return dict(jit_compile_raw=_option_text(raw),jit_compile_effective=not public,
                is_jit_compile_false=public,
                raw_getter='torch_npu._C._npu_getOption(jitCompile)',
                effective_getter='torch_npu.npu.is_jit_compile_false()',
                effective_getter_source='npu_config.py:147 -> _C._npu_is_jit_compile_false()',
                jit_policy_mutated=False)


@dataclass
class SharedCoefficients:
    """One immutable coefficient snapshot per transformer forward, bounded1."""
    cosine: object=None
    sine: object=None
    expanded_cosine: object=None
    expanded_sine: object=None
    generation: int=0

    def prepare(self,cosine,sine):
        if str(cosine.dtype)!='torch.bfloat16' or sine.dtype!=cosine.dtype:
            raise ValueError('Native BF16 QK requires original BF16 coefficients; no downcast')
        if cosine.ndim!=2 or cosine.shape[-1]!=64 or sine.shape!=cosine.shape:
            raise ValueError('Require original local [tokens,64] interleaved coefficients')
        if sine.device!=cosine.device:
            raise ValueError('Rotary coefficient devices differ')
        # Fresh each forward: protects even inference tensors without version
        # counters, reused mutable SP buffers, and reset/session/image changes.
        expanded_cosine=cosine.repeat_interleave(2,dim=-1).unsqueeze(0).unsqueeze(2)
        expanded_sine=sine.repeat_interleave(2,dim=-1).unsqueeze(0).unsqueeze(2)
        self.cosine,self.sine=cosine,sine
        self.expanded_cosine,self.expanded_sine=expanded_cosine,expanded_sine
        self.generation+=1

    def get(self,cosine,sine):
        if cosine is self.cosine and sine is self.sine:
            return self.expanded_cosine,self.expanded_sine
        # Direct leaf tests/calls cannot reuse a stale transformer snapshot.
        return (cosine.repeat_interleave(2,dim=-1).unsqueeze(0).unsqueeze(2),
                sine.repeat_interleave(2,dim=-1).unsqueeze(0).unsqueeze(2))


def configure_native_bf16_qk(transformer,model_config=None,*,device,environ=None,backend=None,logger=None):
    values=os.environ if environ is None else environ
    enabled=_flag(values.get('LINGBOT_NPU_NATIVE_BF16_QK','0'))
    if isinstance(model_config,Mapping) and 'lingbot_npu_native_bf16_qk' in model_config:
        value=model_config['lingbot_npu_native_bf16_qk']
        if type(value) is not bool:raise ValueError('Model native BF16 QK flag must be bool')
        enabled=value
    on_npu=getattr(device,'type',None)=='npu'
    policy=dict(enabled=enabled,active_on_this_platform=enabled and on_npu,default_enabled=False,
                variant='rms_rope_bf16',attention_role='self',tp_required=1,
                dtype='bfloat16',feature_width=5120,heads=40,head_dim=128,
                coefficient_lifetime='fresh per transformer forward; shared across layers; capacity1',
                jit_policy_mutated=False)
    if on_npu:
        if backend is None:
            import torch_npu as backend
        policy.update(read_jit_policy(backend))
        if logger is not None:logger.info('LingBot native BF16 QK runtime policy: %s',policy)
        if enabled and not policy['is_jit_compile_false']:
            raise RuntimeError('Native BF16 interleaved RoPE requires effective JIT=False; existing policy retained. Restart with an explicitly supported serving policy.')
    shared=SharedCoefficients()
    transformer._lingbot_native_bf16_qk_enabled=enabled
    transformer._lingbot_native_bf16_qk_coefficients=shared
    count=0
    for block in transformer.blocks:
        module=block.self_attn
        module._lingbot_native_bf16_qk_enabled=enabled
        module._lingbot_native_bf16_qk_coefficients=shared
        count+=1
    policy['configured_self_attention_layers']=count
    return policy


def prepare_native_bf16_qk(transformer,rotary_emb):
    if not getattr(transformer,'_lingbot_native_bf16_qk_enabled',False):return
    cosine,sine=rotary_emb
    if cosine.device.type!='npu':return
    transformer._lingbot_native_bf16_qk_coefficients.prepare(cosine,sine)


def native_bf16_qk_enabled(module,qkv,rotary_emb):
    if not getattr(module,'_lingbot_native_bf16_qk_enabled',False) or qkv.device.type!='npu':return False
    if rotary_emb is None:raise ValueError('Explicit native BF16 QK requires original rotary coefficients')
    return True


def validate_inputs(qkv,query_weight,key_weight,cosine,sine,*,heads,eps):
    if (qkv.ndim!=3 or qkv.shape[0]!=1 or qkv.shape[1]<=0 or qkv.shape[2]!=15360 or heads!=40):
        raise ValueError('Native BF16 QK requires B1 x tokens x15360, TP1/full5120 norm and40 heads')
    operands=(qkv,query_weight,key_weight,cosine,sine)
    if any(str(value.dtype)!='torch.bfloat16' for value in operands):
        raise ValueError('Native BF16 QK requires original BF16 inputs/weights/coefs; no downcast or fallback')
    if any(value.device!=qkv.device for value in operands) or qkv.device.type!='npu':
        raise ValueError('All native QK operands must share one NPU device')
    if query_weight.shape!=(5120,) or key_weight.shape!=(5120,):
        raise ValueError('Q/K norm weights must span all5120 features before head reshape')
    if cosine.shape!=(qkv.shape[1],64) or sine.shape!=cosine.shape:
        raise ValueError('Local interleaved coefficient shape differs from QKV token positions')
    if not math.isfinite(eps) or eps<=0:raise ValueError('Require finite positive norm epsilon')


def native_bf16_qk(module,qkv,query_weight,key_weight,cosine,sine,*,heads,eps,backend=None):
    validate_inputs(qkv,query_weight,key_weight,cosine,sine,heads=heads,eps=eps)
    if backend is None:
        import torch_npu as backend
    query,key,value=qkv.split((5120,5120,5120),dim=-1)
    query=backend.npu_rms_norm(query,query_weight,epsilon=eps)[0]
    key=backend.npu_rms_norm(key,key_weight,epsilon=eps)[0]
    if str(query.dtype)!='torch.bfloat16' or str(key.dtype)!='torch.bfloat16':
        raise RuntimeError('Native norm did not preserve the required BF16 boundary')
    query,key,value=(item.unflatten(2,(40,128)) for item in (query,key,value))
    cache=module._lingbot_native_bf16_qk_coefficients
    cosine_expanded,sine_expanded=cache.get(cosine,sine)
    query=backend.npu_rotary_mul(query,cosine_expanded,sine_expanded,rotary_mode='interleave')
    key=backend.npu_rotary_mul(key,cosine_expanded,sine_expanded,rotary_mode='interleave')
    if str(query.dtype)!='torch.bfloat16' or str(key.dtype)!='torch.bfloat16':
        raise RuntimeError('Native rotary did not preserve BF16 output; restart required')
    return query,key,value
