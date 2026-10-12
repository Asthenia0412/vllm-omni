# SPDX-License-Identifier: Apache-2.0
"""Default-off SP8 observation. Diagnostics are never clean timing runs.

Imports only the standard library until explicitly enabled. Device descriptors
precede readback; a CPU snapshot never establishes an NPU storage layout.
"""
from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path

MODE = os.environ.get('LINGBOT_SP8_OBSERVE_MODE', '')
ENABLED = MODE in ('profile', 'capture')
_state = None
_sessions = []
_profiler = None
_chunk_range = None
_send = None
_recv = None
_descriptor_loaded = False
FIELDS = ('npu_format', 'origin_format', 'base_sizes', 'base_strides', 'physical_sizes',
          'shape', 'stride', 'storage_offset', 'storage_bytes', 'storage_data_ptr', 'data_ptr')


def _save(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def _event(name, **extra):
    if _state is None:
        return
    with (_state['directory']/'events.jsonl').open('a') as handle:
        handle.write(json.dumps(dict(event=name, chunk=_state['chunk'], probe=_state['probe'],
                                     rank=_state['rank'], **extra))+'\n')


def prepare(pipeline, request):
    global _state, _profiler, _chunk_range
    if not ENABLED:
        return
    import torch
    import torch_npu
    if request.request_id not in _sessions:
        _sessions.append(request.request_id)
    ordinal = int(os.environ.get('LINGBOT_SP8_OBSERVE_SESSION', '2'))
    if _sessions.index(request.request_id)+1 != ordinal:
        return
    if _state is None:
        if not torch.distributed.is_initialized() or torch.distributed.get_world_size() != 8:
            raise ValueError('SP8 observation requires exactly eight workers')
        root = Path(os.environ['LINGBOT_SP8_OBSERVE_DIR']).resolve()
        if not root.is_relative_to('/mnt/data'):
            raise ValueError('Persistent SP8 observation directory required')
        rank = torch.distributed.get_rank()
        directory = root/f'rank{rank}'
        directory.mkdir(parents=True, exist_ok=False)
        trace_directory = directory/'traces'
        spool = os.environ.get('LINGBOT_SP8_PROFILE_PAYLOAD_SPOOL_DIR')
        if spool:
            spool_root = Path(spool).resolve()
            if MODE != 'profile' or not str(spool_root).startswith('/tmp/lingbot-generated-'):
                raise ValueError('Raw profiler spool requires profile mode and generated temporary storage')
            trace_directory = spool_root/f'rank{rank}'/'traces'
            trace_directory.mkdir(parents=True, exist_ok=False)
            _save(directory/'profiling_storage.json', dict(path=str(trace_directory),
                  persistent=False, metadata_persistent=True, model_weights_stored=False))
        _state = dict(directory=directory, trace_directory=trace_directory, rank=rank, session_id=request.request_id,
                      chunk=-1, probe=-1, pipeline=pipeline,
                      layers={id(b.self_attn):i for i,b in enumerate(pipeline.transformer.blocks)},
                      vae={id(m):n for n,m in pipeline.vae.named_modules()}, records=0,
                      payloads=0, vae_seen=set(), finished=False)
    chunk = int(request.chunk_index)
    if _chunk_range is not None:
        _chunk_range.__exit__(None, None, None)
        _chunk_range = None
    _state.update(chunk=chunk, probe=-1)
    if MODE == 'profile':
        start = int(os.environ.get('LINGBOT_SP8_PROFILE_START', '11'))
        end = int(os.environ.get('LINGBOT_SP8_PROFILE_END', '14'))
        if chunk == start:
            from torch_npu.profiler import (profile, ProfilerActivity, _ExperimentalConfig,
                ExportType, ProfilerLevel, AiCMetrics, tensorboard_trace_handler)
            torch_npu.npu.synchronize()
            metric = os.environ.get('LINGBOT_SP8_PROFILE_METRIC', 'PipeUtilization')
            _profiler = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
                record_shapes=True, with_stack=False, profile_memory=False,
                experimental_config=_ExperimentalConfig(export_type=ExportType.Text,
                    profiler_level=ProfilerLevel.Level1, aic_metrics=getattr(AiCMetrics, metric),
                    data_simplification=True, msprof_tx=False, record_op_args=False),
                on_trace_ready=tensorboard_trace_handler(str(_state['trace_directory']), analyse_flag=False))
            _profiler.start()
            # Module scopes expose projection/FFN/attention attribution without
            # changing operators, precision, or dispatch in clean timings.
            hooks = []
            for name, module in pipeline.transformer.named_modules():
                if (name.endswith(('qkv', 'to_q', 'to_k', 'to_v', 'to_out', 'ffn.0', 'ffn.2',
                                   'self_attn', 'cross_attn')) or name == 'head'):
                    frames = []
                    def before(mod, args, label=name, stack=frames):
                        region = torch.profiler.record_function('lingbot.sp8.module.'+label)
                        region.__enter__()
                        stack.append(region)
                    def after(mod, args, result, stack=frames):
                        if stack:
                            stack.pop().__exit__(None, None, None)
                    hooks.extend((module.register_forward_pre_hook(before),
                                  module.register_forward_hook(after, always_call=True)))
            _state['profile_hooks'] = hooks
            _event('profile_start', metric=metric, before_prepare_chunk=chunk)
        elif chunk == end and _profiler is not None:
            torch_npu.npu.synchronize()
            _profiler.stop()
            _profiler = None
            for hook in _state.get('profile_hooks', []):
                hook.remove()
            _state['finished'] = True
            _event('profile_stop', before_prepare_chunk=chunk, complete_chunks=list(range(start,end)))
        if _profiler is not None:
            _chunk_range = torch.profiler.record_function(f'lingbot.sp8.chunk.{chunk}')
            _chunk_range.__enter__()
    _event('prepare')


def probe(chunk, index):
    if _state is not None:
        _state.update(chunk=int(chunk), probe=int(index))


def active():
    return (_state is not None and MODE == 'capture' and not _state['finished']
            and _state['chunk'] in (0,1,2,3,4,5,11) and _state['probe'] in range(4))


def describe(value):
    global _descriptor_loaded
    import torch
    if not _descriptor_loaded:
        library = Path(os.environ['LINGBOT_SP8_DESCRIPTOR_LIBRARY'])
        if hashlib.sha256(library.read_bytes()).hexdigest() != os.environ['LINGBOT_SP8_DESCRIPTOR_SHA256']:
            raise ValueError('Read-only storage descriptor library changed')
        torch.ops.load_library(str(library))
        _descriptor_loaded = True
    desc = dict(zip(FIELDS, torch.ops.lingbot_storage_descriptor.describe(value)))
    desc.update(dtype=str(value.dtype), device=str(value.device), logical_bytes=value.numel()*value.element_size(),
                element_bytes=value.element_size(), is_contiguous=value.is_contiguous())
    if (desc['shape'] != list(value.shape) or desc['stride'] != list(value.stride())
            or desc['storage_offset'] != value.storage_offset()
            or desc['storage_bytes'] != value.untyped_storage().nbytes()
            or desc['data_ptr'] != value.data_ptr()):
        raise ValueError('C++ original descriptor disagrees with tensor')
    return desc


def _cpu(values):
    before = {name:describe(value) for name,value in values.items() if value is not None}
    cpu = {name:value.detach().cpu().contiguous().clone() for name,value in values.items() if value is not None}
    after = {name:describe(value) for name,value in values.items() if value is not None}
    if before != after:
        raise ValueError('Readback changed original descriptors')
    return before, cpu


def _write(name, record, tensors=None):
    record.update(rank=_state['rank'], chunk=_state['chunk'], probe=_state['probe'],
                  session_id=_state['session_id'], timing_eligible=False)
    _save(_state['directory']/(name+'.json'), record)
    _state['records'] += 1
    if tensors is not None:
        import torch
        path = _state['directory']/(name+'.pt')
        torch.save(dict(**record, tensors=tensors), path)
        _state['payloads'] += 1


def send_pre(attention, query, key, value, backing):
    global _send
    if not active():
        return
    layer = _state['layers'][id(attention)]
    values = dict(q=query, k=key, v=value, qkv_backing=backing)
    desc = {n:describe(t) for n,t in values.items()}
    tensors = _cpu(values)[1] if layer == 0 and _state['chunk'] == 11 else None
    _send = dict(kind='sp8_qkv_send', layer=layer, descriptors=desc, tensors=tensors)


def native_send(value):
    if _send is not None:
        _send['descriptors']['native_send'] = describe(value)
        if _send['tensors'] is not None:
            _send['tensors']['native_send'] = _cpu(dict(native_send=value))[1]['native_send']


def receive(attention, cache, receive, query, key, value):
    global _send, _recv
    if not active():
        return
    layer = _state['layers'][id(attention)]
    if _send is None or _send['layer'] != layer:
        raise ValueError('Missing native SP8 send observation')
    tensors = _send.pop('tensors')
    _write(f'c{_state["chunk"]:03d}_p{_state["probe"]}_l{layer:02d}_send', _send, tensors)
    _send = None
    if not hasattr(cache, 'key_pool'):
        return
    values = dict(receive=receive, query=query[0], key=key[0], value=value[0],
                  kp=cache.key_pool, vp=cache.value_pool, ks=cache.stage_key, vs=cache.stage_value,
                  slots=cache.video_slots, table=cache.block_table)
    desc = {n:describe(t) for n,t in values.items() if t is not None}
    record = dict(kind='sp8_qkv_recv', layer=layer, descriptors=desc,
                  first=cache.stage_first_block, kv_len=cache.host_kv_len, reuse_history=cache.reuse_history,
                  block_size=cache.block_size)
    ar = _state['pipeline']._ar_diffusion_kv_state
    manager = ar.kv_cache
    ctx = ar._paged_pending['main']
    if (cache.key_pool is not manager._k_pools[layer]
            or cache.value_pool is not manager._v_pools[layer]
            or ctx is None):
        raise ValueError('Capture does not refer to original manager pools')
    record['manager'] = dict(managed_num_blocks=manager.managed_num_blocks,
        scratch_num_blocks=manager.scratch_num_blocks, num_blocks_total=manager.num_blocks_total,
        block_size=manager.block_size, history_staging_tokens=manager.history_staging_tokens,
        num_kv_heads=manager.num_kv_heads, head_size=manager.head_size, null_block_id=manager.null_block_id,
        history_staging_layer_indices=list(manager.history_staging_layer_indices),
        current_video_block_ids=list(ctx.current_video_block_ids), history_block_ids=list(ctx.history_block_ids),
        commit_current=ctx.commit_current, kv_branch=ctx.kv_branch,
        scratch_block_ids=manager.scratch_block_ids(ctx.kv_branch, 0, 3),
        manager_identity=id(manager), adapter_identity=id(ctx.adapter),
        forward_owner_identity=id(ctx._history_stage_owner))
    record['ctx'] = dict(commit_current=ctx.commit_current, kv_branch=ctx.kv_branch,
        history_block_ids=list(ctx.history_block_ids), current_video_block_ids=list(ctx.current_video_block_ids),
        num_computed_tokens=int(ctx.adapter.num_computed_tokens),
        absolute_num_computed_tokens=int(ctx.adapter.absolute_num_computed_tokens),
        max_video_tokens=ctx.max_video_tokens)
    payload = None
    if layer == 0 and _state['chunk'] == 11:
        payload = _cpu(values)[1]
    _recv = (record, payload)


def attention_done(attention, cache, output):
    global _recv
    if not active() or _recv is None:
        return
    record, tensors = _recv
    layer = record['layer']
    if tensors is not None:
        _, after = _cpu(dict(kp_after=cache.key_pool, vp_after=cache.value_pool,
                             ks_after=cache.stage_key, vs_after=cache.stage_value, output=output))
        tensors.update(after)
    # Actual FIA input: dense Q and the successfully staged visible K/V.
    if layer == 0:
        length = cache.host_kv_len
        _, inputs = _cpu(dict(query=_last_query,
                              key=cache.stage_key[:length].unsqueeze(0),
                              value=cache.stage_value[:length].unsqueeze(0)))
        inputs['native_output'] = output.detach().cpu().contiguous().clone()
        _write(f'c{_state["chunk"]:03d}_p{_state["probe"]}_attention',
               dict(kind='sp8_attention', layer=layer, head_num=5, scale=128**-0.5), inputs)
    _write(f'c{_state["chunk"]:03d}_p{_state["probe"]}_l{layer:02d}_recv', record, tensors)
    _recv = None


_last_query = None


def query_for_attention(value):
    global _last_query
    if active():
        _last_query = value


def vae_before(module, stage, base, dst_a, src_a, dst_b, src_b):
    if _state is None or MODE != 'capture' or _state['chunk'] != 11:
        return None
    name = _state['vae'].get(id(module), type(module).__name__)
    key = (name, stage, tuple(base.shape))
    if key in _state['vae_seen']:
        return None
    _state['vae_seen'].add(key)
    values = dict(base=base, dst_a=dst_a, src_a=src_a, dst_b=dst_b, src_b=src_b)
    desc, cpu = _cpu(values)
    return dict(kind='sp8_vae_pack', module_name=name, stage=stage, descriptors=desc,
                tensors=dict(base_before=cpu['base'], src_a=cpu['src_a'], src_b=cpu['src_b']),
                name=f'vae_{len(_state["vae_seen"]):03d}_{stage}')


def vae_after(context, base):
    if context is None:
        return
    context['tensors']['base_after'] = _cpu(dict(base=base))[1]['base']
    tensors = context.pop('tensors')
    name = context.pop('name')
    _write(name, context, tensors)


def region(name):
    if _profiler is not None:
        import torch
        return torch.profiler.record_function('lingbot.sp8.'+name)
    return nullcontext()
