"""Correctness-only output projection evidence; never enable in timing runs."""
from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path


class Capture:
    def __init__(self, transformer):
        import torch
        self.rank = torch.distributed.get_rank()
        self.directory = Path(os.environ['LINGBOT_OUTPUT_CAPTURE_DIR'])/f'rank{self.rank}'
        self.directory.mkdir(parents=True,exist_ok=False)
        descriptor = os.environ['LINGBOT_OUTPUT_CAPTURE_DESCRIPTOR_LIBRARY']
        torch.ops.load_library(descriptor)
        self.describe = torch.ops.lingbot_storage_descriptor.describe
        self.context = None
        self.sequence = 0
        self.pending = {}
        self.done = set()
        self.weights = {}
        self.transformer = transformer

    def bind(self, state):
        if state is not None:
            context = (state.request_id,int(state.chunk_index),int(state.step_in_chunk))
            if self.context is None or context[0] != self.context[0]:
                self.sequence += 1
            self.context = context

    def selected(self):
        return (self.sequence==2 and self.context is not None and 0<=self.context[1]<=5)

    def before(self, module, value):
        if self.selected():
            self.pending[id(module)] = value

    def after(self, module, value, result, role='self'):
        import torch
        if not self.selected():
            return
        key = (*self.context, role)
        if key in self.done:
            raise RuntimeError('Duplicate output projection capture context')
        self.done.add(key)
        attention = self.pending.pop(id(module)) if role=='self' else None
        layer = module.o if role=='self' else module
        operands = dict(projection_input=value, weight=layer.weight,
                        bias=layer.bias, native_output=result)
        if attention is not None:
            operands['attention_output'] = attention
        fields = ('npu_format','origin_format','base_sizes','base_strides','physical_sizes',
                  'shape','stride','storage_offset','storage_bytes')
        descriptors = {name:dict(zip(fields,self.describe(tensor))) for name,tensor in operands.items()}
        # These readbacks and payload writes deliberately belong only to the
        # separate correctness service, after the complete native call.
        cpu = {name:tensor.detach().cpu().contiguous() for name,tensor in operands.items()
               if name not in ('weight','bias')}
        weight_path = self.directory/(role+'_output_weight.pt')
        if not weight_path.exists():
            torch.save({name:operands[name].detach().cpu().contiguous() for name in ('weight','bias')},weight_path)
            self.weights[role] = hashlib.sha256(weight_path.read_bytes()).hexdigest()
        session, chunk, probe = self.context
        prefix = '' if role=='self' else 'cross_'
        path = self.directory/f'{prefix}c{chunk:03d}_p{probe}.pt'
        torch.save(cpu,path)
        raw_sha = lambda tensor:hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()
        metadata = dict(rank=self.rank,layer=0,role=role,session=session,chunk=chunk,probe=probe,
            timing_eligible=False,descriptors=descriptors,
            tensors={name:dict(shape=list(tensor.shape),dtype=str(tensor.dtype),sha256=raw_sha(tensor))
                     for name,tensor in cpu.items()},payload=str(path),
            payload_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            weight=str(weight_path),weight_sha256=self.weights[role],
            layer_class=type(layer).__module__+'.'+type(layer).__qualname__,
            method_class=type(layer.quant_method).__module__+'.'+type(layer.quant_method).__qualname__,
            native_implementation_pins={inspect.getsourcefile(kind):hashlib.sha256(
                Path(inspect.getsourcefile(kind)).read_bytes()).hexdigest()
                for kind in (type(layer),type(layer.quant_method))})
        path.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')


def install(pipeline):
    if os.environ.get('ROOT_LINGBOT_SP8_EXPERIMENT_MODE')!='correctness':
        raise ValueError('Output capture is allowed only in a separate correctness service')
    if pipeline._lingbot_output_mc2_enabled:
        raise ValueError('Capture the complete native output path with MC2 disabled')
    recorder = Capture(pipeline.transformer)
    pipeline._lingbot_output_capture = recorder
    pipeline.transformer.blocks[0].self_attn._lingbot_output_capture = recorder
    cross_o = pipeline.transformer.blocks[0].cross_attn.o
    cross_o.register_forward_hook(lambda module,args,result:
                                  recorder.after(module,args[0],result,role='cross'))
