"""Explicit correctness-only real QK/cross fixtures, disabled by default.

Installed only through the existing output-capture install/bind hooks.
No device imports/calls occur when this module is imported for CPU review/tests.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import threading
import types

LENGTHS = (4680, 9360, 14040, 18720, 23400, 28080)
FIELDS = ('npu_format', 'origin_format', 'base_sizes', 'base_strides', 'physical_sizes',
          'shape', 'stride', 'storage_offset', 'storage_bytes', 'storage_data_ptr', 'data_ptr')


def _need(value, reason):
    if not value:
        raise RuntimeError(reason)


def _json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def _pin(path):
    path = Path(path).resolve(strict=True)
    data = path.read_bytes()
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def view_extent(desc):
    """Validate complete unblocked-owner access without interpreting payload."""
    _need(desc['npu_format'] in (0, 2, 30), 'Fixture requires independently admitted unblocked format')
    _need(desc['origin_format'] == desc['npu_format'], 'Origin/physical format mismatch')
    _need(desc['base_sizes'] == desc['physical_sizes'], 'Blocked/hidden physical extent not supported')
    sizes, strides = desc['base_sizes'], desc['base_strides']
    _need(len(sizes) == len(strides) and len(sizes) > 0, 'Invalid owner rank')
    elements = 1
    for size, stride in reversed(list(zip(sizes, strides))):
        _need(size > 0 and stride == elements, 'Dense physical owner required')
        elements *= size
    _need(elements * desc['element_bytes'] == desc['storage_bytes'], 'Unaccounted allocation bytes')
    _need(len(desc['shape']) == len(desc['stride']), 'Logical descriptor rank mismatch')
    high = desc['storage_offset']
    _need(high >= 0, 'Negative view offset')
    for size, stride in zip(desc['shape'], desc['stride']):
        _need(size > 0 and stride >= 0, 'Invalid view shape/stride')
        high += (size - 1) * stride
    _need(high < elements, 'View exceeds full owner')
    _need(desc['data_ptr'] == desc['storage_data_ptr'] + desc['storage_offset'] * desc['element_bytes'],
          'View pointer/offset mismatch')
    return elements


class Coverage:
    """First real occurrence per length/probe, never inferred from chunk number."""
    def __init__(self, rank, layers=40, payload_layers=(0,)):
        _need(rank in range(8), 'Exactly eight rank identities required')
        _need(layers > 0 and payload_layers and len(set(payload_layers)) == len(payload_layers)
              and all(layer in range(layers) for layer in payload_layers), 'Invalid layer selection')
        self.rank, self.layers, self.payload_layers = rank, layers, tuple(payload_layers)
        self.locations = {}
        self.descriptors, self.payloads = set(), set()

    def choose(self, request_id, chunk, probe, kv_len):
        _need(probe in range(4), 'Actual DMD probe index required')
        if kv_len not in LENGTHS:
            return False
        key = (kv_len, probe)
        location = (request_id, chunk)
        self.locations.setdefault(key, location)
        return self.locations[key] == location

    def record(self, kind, kv_len, probe, layer, *, payload):
        _need(kind in ('qk', 'cross') and kv_len in LENGTHS and probe in range(4)
              and layer in range(self.layers), 'Invalid fixture coverage key')
        key = (kind, kv_len, probe, layer)
        _need(key not in self.descriptors, 'Duplicate descriptor case')
        self.descriptors.add(key)
        if payload:
            _need(layer in self.payload_layers, 'Payload outside declared layer selection')
            self.payloads.add(key)

    def report(self):
        expected = {(kind, length, probe, layer) for kind in ('qk', 'cross')
                    for length in LENGTHS for probe in range(4) for layer in range(self.layers)}
        wanted_payload = {k for k in expected if k[3] in self.payload_layers}
        return dict(rank=self.rank, expected_payload_cases_per_role=24*len(self.payload_layers),
                    descriptor_count=len(self.descriptors), payload_count=len(self.payloads),
                    descriptor_missing=[list(k) for k in sorted(expected-self.descriptors)],
                    payload_missing=[list(k) for k in sorted(wanted_payload-self.payloads)],
                    complete=self.descriptors == expected and self.payloads == wanted_payload,
                    actual192_global_proven=False, all40_payloads=False,
                    locations=[dict(kv_len=k[0], probe=k[1], request_id=v[0], chunk=v[1])
                               for k,v in sorted(self.locations.items())])


class Recorder:
    def __init__(self, pipeline, root, request_ordinal=2):
        import torch
        from vllm_omni.diffusion.layers.lingbot_recovery import load
        _need(os.environ.get('ROOT_LINGBOT_SP8_EXPERIMENT_MODE') == 'correctness',
              'Fixture capture requires a separate correctness service')
        _need(torch.distributed.is_initialized() and torch.distributed.get_world_size() == 8,
              'Eight-rank service required')
        self.torch, self.pipeline, self.provider = torch, pipeline, load()
        source_pins = [_pin(__file__), _pin(Path(__file__).with_name('lingbot_output_capture.py'))]
        _need(all(pin in self.provider.manifest['runtime_sources'] for pin in source_pins),
              'Build manifest must pin the actual fixture helper and install hook')
        self.source_pins = source_pins
        self.rank = torch.distributed.get_rank()
        _need(len(pipeline.transformer.blocks) == 40, 'Original 40-layer model required')
        root = Path(root).resolve()
        _need(root.is_relative_to('/mnt/sfs_turbo'), 'Persistent fixture root required')
        self.directory = root / f'rank{self.rank}'
        self.directory.mkdir(parents=True, exist_ok=False)
        (self.directory/'owners').mkdir()
        self.request_ordinal, self.sessions, self.context = request_ordinal, [], None
        self.coverage = Coverage(self.rank)
        self.thread = threading.get_ident()
        self.active = False
        self.failed = False
        self.pending = {}
        self.layer_context = {}
        self.expected_dispatch = None
        self.keep_on_failure = []
        self.saved_cases = []
        self.layers = {id(block.self_attn): index for index, block in enumerate(pipeline.transformer.blocks)}
        self._publish()

    def _publish(self):
        _json(self.directory/'coverage.json', dict(schema='lingbot_real_qk_cross_capture_v1',
            **self.coverage.report(), source='fresh real model activations', failed=self.failed,
            timing_eligible=False, performance_validated=False, model_admitted=False,
            native_reference='current multiple-dispatch QK and plain cross BSND, not historical binary',
            source_pins=self.source_pins,
            build_manifest=dict(path=self.provider.policy[0], sha256=self.provider.policy[1]),
            request_ordinal=self.request_ordinal, request_ids=list(self.sessions),
            cases=self.saved_cases))

    def poison(self, error, *owners):
        first = not self.failed
        self.failed = True
        held = (self, error, owners, dict(self.pending), self.expected_dispatch)
        self.keep_on_failure.append(held)
        if first:
            self.provider.fatal.report_failure('real_fixture_capture', repr(error), held)
            self._publish()

    def bind(self, state):
        _need(state is not None and self.active, 'Actual denoise state binding required')
        _need(not self.pending and self.expected_dispatch is None, 'Previous capture boundary incomplete')
        self.layer_context.clear()
        request = str(state.request_id)
        if request not in self.sessions:
            self.sessions.append(request)
        self.context = (request, int(state.chunk_index), int(state.step_in_chunk))

    def selected(self):
        _need(threading.get_ident() == self.thread, 'Single model producer thread required')
        _need(not self.failed, 'Fixture collector poisoned; restart all workers')
        return (self.active and self.context is not None
                and self.sessions.index(self.context[0])+1 == self.request_ordinal)

    def describe(self, value):
        fields = self.provider.describe(value)
        _need(len(fields) == len(FIELDS), 'Exact eleven-field descriptor provider required')
        desc = dict(zip(FIELDS, fields))
        for key in ('base_sizes','base_strides','physical_sizes','shape','stride'):
            desc[key] = list(desc[key])
        desc.update(dtype=str(value.dtype), element_bytes=value.element_size(),
                    device=str(value.device), logical_bytes=value.numel()*value.element_size())
        _need(desc['shape'] == list(value.shape) and desc['stride'] == list(value.stride())
              and desc['storage_bytes'] == value.untyped_storage().nbytes()
              and desc['storage_data_ptr'] == value.untyped_storage().data_ptr()
              and desc['data_ptr'] == value.data_ptr(), 'Descriptor disagrees with live tensor')
        view_extent(desc)
        try:
            desc['version_counter'] = value._version
        except RuntimeError:
            desc['version_counter'] = None
        return desc

    def snapshot(self, tensors, *, payload):
        # Caller holds strong references; full-owner reads use admitted ND views,
        # never mutate physical format tags and never replace logical source views.
        descriptors, owners, aliases = {}, {}, {}
        owner_map = {}
        held = tuple(tensors.values())
        for name, tensor in tensors.items():
            desc = self.describe(tensor)
            descriptors[name] = desc
            key = (desc['device'], desc['storage_data_ptr'], desc['storage_bytes'])
            if key not in owner_map:
                oid = f'owner{len(owner_map):03d}'
                owner_map[key] = oid
                item = dict(storage_bytes=desc['storage_bytes'], base_sizes=desc['base_sizes'],
                            base_strides=desc['base_strides'], npu_format=desc['npu_format'],
                            raw_sha256=None, relative_path=None)
                if payload:
                    full = tensor.as_strided(desc['base_sizes'], desc['base_strides'], 0)
                    _need(full.data_ptr() == desc['storage_data_ptr'], 'Full owner read must start at allocation origin')
                    cpu = full.detach().cpu().contiguous()
                    data = cpu.view(self.torch.uint8).numpy().tobytes()
                    _need(len(data) == desc['storage_bytes'], 'Incomplete full-owner readback')
                    digest = hashlib.sha256(data).hexdigest()
                    path = self.directory/'owners'/f'{digest}.u8'
                    if not path.exists():
                        temp=path.with_suffix('.tmp');temp.write_bytes(data);temp.replace(path)
                    item.update(raw_sha256=digest, relative_path=str(path.relative_to(self.directory)))
                owners[oid] = item
            aliases[name] = owner_map[key]
        after = {name:self.describe(t) for name,t in tensors.items()}
        _need(after == descriptors, 'Readback changed original descriptor/available version')
        return dict(descriptors=descriptors, owners=owners, view_owner_ids=aliases)

    def expect_dispatch(self, name, tensors):
        _need(self.expected_dispatch is None, 'Nested fixture backend boundary')
        self.expected_dispatch = dict(name=name, tensors=tuple(tensors), seen=0, result=None)

    def observed_call(self, original, name, *args):
        expected = self.expected_dispatch
        matched = expected is not None and name in ('qk_send', 'cross')
        if matched:
            _need(name == expected['name'] and len(args) >= len(expected['tensors'])
                  and all(a is b for a, b in zip(args, expected['tensors'])),
                  'Actual provider dispatch differs from captured operands')
            _need(expected['seen'] == 0, 'Duplicate native backend dispatch')
        result = original(name, *args)
        if matched:
            expected['seen'] += 1
            expected['result'] = result
        return result

    def finish_dispatch(self, result):
        expected = self.expected_dispatch
        _need(expected is not None and expected['seen'] == 1 and expected['result'] is result,
              'Requested backend did not produce captured output; no fallback qualification')
        self.expected_dispatch = None
        return dict(provider_operation=expected['name'], dispatch_count=1,
                    operands_identical=True, output_identical=True,
                    build_manifest_sha256=self.provider.policy[1])

    def before_layer(self, module, args, kwargs):
        if not self.selected():
            return
        layer = self.layers[id(module)]
        cache = kwargs.get('cache')
        _need(cache is not None and hasattr(cache, 'host_kv_len'), 'Authoritative prepared KV metadata required')
        request,chunk,probe = self.context
        length = int(cache.host_kv_len)
        if not self.coverage.choose(request,chunk,probe,length):
            self.layer_context.pop(layer,None);return
        ar = self.pipeline._ar_diffusion_kv_state
        manager,ctx = ar.kv_cache, ar._paged_pending['main']
        _need(ctx is not None, 'Original pending manager context required')
        metadata = dict(request_id=request, chunk=chunk, probe=probe, kv_len=length, layer=layer,
            branch=ctx.kv_branch, commit_current=bool(ctx.commit_current),
            current_video_block_ids=list(ctx.current_video_block_ids),
            history_block_ids=list(ctx.history_block_ids), manager_identity=id(manager),
            forward_owner_identity=id(ctx._history_stage_owner), adapter_identity=id(ctx.adapter),
            pool_rows=int(manager.num_blocks_total*manager.block_size),
            staging_rows=int(manager.history_staging_tokens), num_kv_heads=manager.num_kv_heads,
            head_size=manager.head_size, block_size=manager.block_size,
            reuse_history=bool(cache.reuse_history))
        self.layer_context[layer] = metadata

    def capture_qk(self, original, module, raw, rotary, ring_size):
        layer = self.layers.get(id(module))
        if not self.selected() or layer not in self.layer_context:
            return original(module,raw,rotary,ring_size)
        meta=self.layer_context[layer];key=('qk',meta['kv_len'],meta['probe'],layer)
        if key in self.coverage.descriptors:
            return original(module,raw,rotary,ring_size)
        coefficients=module._lingbot_native_bf16_qk_coefficients
        _need(coefficients.cosine is rotary[0] and coefficients.sine is rotary[1],
              'Capture must use original shared coefficient generation; no extra expansion')
        tensors=dict(qkv=raw, query_weight=module.norm_q.weight, key_weight=module.norm_k.weight,
                     cosine=rotary[0], sine=rotary[1], expanded_cosine=coefficients.expanded_cosine,
                     expanded_sine=coefficients.expanded_sine)
        meta=dict(meta, coefficient_generation=int(coefficients.generation), epsilon=float(module.norm_q.eps),
                  head_num=int(module.num_local_heads), ring_size=ring_size)
        payload=layer in self.coverage.payload_layers
        before=self.snapshot(tensors,payload=payload)
        self.expect_dispatch('qk_send', (raw, module.norm_q.weight, module.norm_k.weight,
                                       coefficients.expanded_cosine, coefficients.expanded_sine))
        result=original(module,raw,rotary,ring_size)
        _need(result is not None and hasattr(result,'send'), 'Capture baseline prepared-send backend actually required')
        receipt=self.finish_dispatch(result.send)
        after=self.snapshot(tensors,payload=payload)
        _need(before == after, 'QK native call mutated source owner or descriptor')
        outputs=self.snapshot({'native_send':result.send},payload=payload)
        self.finish('qk',meta,before,outputs,payload,
                    actual_backend='lingbot_recovery_v1::qk_send', dispatch_receipt=receipt)
        return result

    def cross_before(self, layer, module, args, kwargs):
        if not self.selected() or layer not in self.layer_context:
            return
        meta=self.layer_context[layer];key=('cross',meta['kv_len'],meta['probe'],layer)
        if key in self.coverage.descriptors:
            return
        _need(len(args)>=3, 'Actual cross Q/K/V call operands required')
        q,k,v=args[:3]
        _need(tuple(q.shape)==(1,585,40,128) and tuple(k.shape)==tuple(v.shape)
              and tuple(k.shape)==(1,512,40,128), 'Unexpected cross native fixture geometry')
        payload=layer in self.coverage.payload_layers
        tensors=dict(query=q,key=k,value=v)
        self.pending[layer]=(dict(meta,text_kv_len=int(k.shape[1]),scale=128**-0.5),tensors,
                             self.snapshot(tensors,payload=payload),payload)
        self.expect_dispatch('cross', (q,k,v))

    def cross_after(self, layer, module, args, kwargs, result):
        pending=self.pending.pop(layer,None)
        if pending is None:return
        meta,tensors,before,payload=pending
        receipt=self.finish_dispatch(result)
        after=self.snapshot(tensors,payload=payload)
        _need(before == after, 'Cross native call mutated input/cache owner or descriptor')
        outputs=self.snapshot({'native_output':result},payload=payload)
        self.finish('cross',meta,before,outputs,payload,
                    actual_backend='lingbot_recovery_v1::cross', dispatch_receipt=receipt)

    def finish(self, kind, metadata, inputs, outputs, payload, **extra):
        layer,length,probe=metadata['layer'],metadata['kv_len'],metadata['probe']
        path=self.directory/f'{kind}_kv{length}_p{probe}_l{layer:02d}.json'
        _json(path,dict(schema='lingbot_real_fixture_case_v1',kind=kind,rank=self.rank,**metadata,**extra,
            payload_present=payload,inputs=inputs,outputs=outputs,
            owner_id_scope='case plus inputs/outputs section; identity is live storage pointer, not content hash',
            full_sources_unchanged=True if payload else None,
            source_descriptors_unchanged=True,
            destination_canary_verified=False,
            source_mutation_check='full owner bytes before/after for payload layer; descriptor/version only otherwise',
            timing_eligible=False,performance_validated=False,model_admitted=False))
        self.coverage.record(kind,length,probe,layer,payload=payload)
        self.saved_cases.append(path.name)
        # A bounded summary write at the completed probe boundary; payloads and
        # individual case descriptors are already durable. Failure publishes
        # the partial coverage before the service is retired.
        if kind == 'cross' and layer == self.coverage.layers - 1:
            self._publish()


def install(pipeline):
    """Explicit diagnostic installation through output_capture.install only."""
    _need(os.environ.get('LINGBOT_REAL_FIXTURE_CAPTURE')=='1','Explicit fixture diagnostic switch required')
    _need(not hasattr(pipeline,'_lingbot_output_capture'), 'One diagnostic owner per pipeline')
    recorder=Recorder(pipeline,os.environ['LINGBOT_REAL_FIXTURE_DIR'])
    values=pipeline.od_config.model_config or {}
    _need(values.get('lingbot_npu_recovery_qk') is True
          and values.get('lingbot_npu_recovery_cross') is True
          and not values.get('lingbot_npu_recovery_cross_combined',False),
          'Capture native QK multiple dispatch + plain cross BSND reference, not candidate')
    _need(not pipeline._lingbot_output_mc2_enabled, 'Capture original output dependency path')
    # Current Capture.bind hook in pipeline is reused; phase wrapping excludes
    # condition prefill, clean commits and unrelated transformer invocations.
    pipeline._lingbot_output_capture=recorder
    original_step=pipeline.denoise_step
    def step(self,*args,**kwargs):
        _need(not recorder.active,'Nested model probe is unsupported')
        recorder.active=True
        try:return original_step(*args,**kwargs)
        except BaseException as exc:
            recorder.poison(exc)
            raise
        finally:
            recorder.active=False
            recorder.layer_context.clear()
    pipeline.denoise_step=types.MethodType(step,pipeline)
    from vllm_omni.diffusion.layers import lingbot_recovery
    original_send=lingbot_recovery.prepare_send
    def send(module,raw,rotary,ring_size):
        try:return recorder.capture_qk(original_send,module,raw,rotary,ring_size)
        except BaseException as exc:
            recorder.poison(exc,module,raw,rotary)
            raise
    lingbot_recovery.prepare_send=send
    original_call=recorder.provider.call
    def call(self,name,*args):
        return recorder.observed_call(original_call,name,*args)
    recorder.provider.call=types.MethodType(call,recorder.provider)
    hooks=[]
    for layer,block in enumerate(pipeline.transformer.blocks):
        hooks.append(block.self_attn.register_forward_pre_hook(recorder.before_layer,with_kwargs=True))
        def before(module,args,kwargs,layer=layer):return recorder.cross_before(layer,module,args,kwargs)
        def after(module,args,kwargs,result,layer=layer):return recorder.cross_after(layer,module,args,kwargs,result)
        hooks.append(block.cross_attn.attn.register_forward_pre_hook(before,with_kwargs=True))
        hooks.append(block.cross_attn.attn.register_forward_hook(after,with_kwargs=True))
    recorder.hooks=hooks
    recorder.original_step,recorder.original_send=original_step,original_send
    recorder.original_call=original_call
    return recorder
