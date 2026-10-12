"""Opt-in, token-striped reverse Ulysses and BF16 output projection.

Two communication slots belong to each invocation. HCCL completion waits are
inserted on the consuming compute stream; producers use a separate stream.
No input is format-cast and no device-wide synchronization occurs in forward.
"""
from __future__ import annotations

import threading


def token_tiles(tokens: int, count: int) -> tuple[tuple[int, int], ...]:
    if count not in (2, 3) or tokens < count:
        raise ValueError("Output overlap supports two or three nonempty token tiles")
    size, extra = divmod(tokens, count)
    start = 0
    result = []
    for index in range(count):
        end = start + size + (index < extra)
        result.append((start, end))
        start = end
    return tuple(result)


def pack_tile(value, start: int, end: int, destination):
    # Original sequence is peer-major. Preserve each peer's local token index.
    destination.copy_(value.reshape(8, 585, 5, 128)[:, start:end])


def dense_tile(received):
    # Receive peer index is the head-group index, not a token index.
    return received.permute(1, 0, 2, 3).contiguous().reshape(1, received.shape[1], 5120)


def aiv_group(original, device):
    import torch
    import torch_npu

    ranks = torch.distributed.get_process_group_ranks(original)
    if len(ranks) != 8:
        raise ValueError("AIV candidate requires the original eight-rank Ulysses membership")
    options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
    # ProcessGroupHCCL translates this snake-case option into the C API field
    # hcclOpExpansionMode. Unknown camel-case keys are silently ignored.
    options.hccl_config = {"hccl_op_expansion_mode": 3}
    return torch.distributed.new_group(ranks=ranks, backend="hccl", pg_options=options)


def configure_group_vector_limit(group, device, cores):
    """Limit only this domain's collective stream, before its first operator."""
    import ctypes
    import os
    from pathlib import Path

    import torch

    if type(cores) is not int or cores not in (2, 4, 8):
        raise ValueError("AIV stream candidate supports 2/4/8 Vector cores")
    backend = group._get_backend(device)
    stream_id = backend.get_coll_stream_id(device)
    current = torch.npu.current_stream(device)
    communication = torch.npu.Stream(stream_id=stream_id, device_index=current.device_index,
                                      device_type=current.device_type)
    library = ctypes.CDLL(str(Path(os.environ["ASCEND_HOME_PATH"]) / "lib64/libascendcl.so"))
    getter, setter = library.aclrtGetStreamResLimit, library.aclrtSetStreamResLimit
    getter.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_uint32)]
    setter.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32]
    getter.restype = setter.restype = ctypes.c_int

    def query(stream):
        value = ctypes.c_uint32()
        code = getter(ctypes.c_void_p(stream.npu_stream), 1, ctypes.byref(value))
        if code:
            raise ValueError(f"Communication stream Vector limits unsupported: ACL {code}")
        return value.value

    compute_before, communication_before = query(current), query(communication)
    code = setter(ctypes.c_void_p(communication.npu_stream), 1, cores)
    if code:
        raise ValueError(f"Communication stream Vector limit rejected: ACL {code}")
    if query(communication) != cores or query(current) != compute_before:
        raise RuntimeError("Communication limit did not preserve the compute stream")
    return dict(stream_id=stream_id, communication_before=communication_before,
                communication_after=cores, compute_before=compute_before,
                compute_after=query(current), device_limit_changed=False)


def install_ulysses_aiv(transformer, vector_cores=None):
    """One independent domain for both Ulysses exchanges; VAE group is retained."""
    first = transformer.blocks[0].self_attn
    group = aiv_group(first.ulysses_group, first.o.weight.device)
    if vector_cores is not None:
        transformer._lingbot_ulysses_stream_resources = configure_group_vector_limit(
            group, first.o.weight.device, vector_cores)
    for block in transformer.blocks:
        if block.self_attn.ulysses_world_size != 8:
            raise ValueError("Independent AIV domain requires SP8")
        block.self_attn.ulysses_group = group
        block.self_attn._lingbot_requested_ulysses_backend = "aiv"
    transformer._lingbot_ulysses_aiv_group = group


class OutputOverlap:
    def __init__(self, module, *, tiles=2, group=None):
        import torch

        layer = module.o
        if (module.ulysses_world_size != 8 or layer.tp_size != 1 or layer.return_bias
                or layer.skip_bias_add or getattr(layer, "_lingbot_quant_mode", None)):
            raise ValueError("Output overlap requires SP8/TP1 and native fused-bias projection")
        if tuple(layer.weight.shape) != (5120, 5120) or layer.weight.dtype != torch.bfloat16:
            raise ValueError("Output overlap requires the original BF16 5120x5120 weight")
        self.tiles = token_tiles(585, tiles)
        self.layer = layer
        self.group = group if group is not None else module.ulysses_group
        if torch.distributed.get_world_size(self.group) != 8:
            raise ValueError("Output overlap requires eight communication ranks")
        self.weight, self.bias = layer.weight, layer.bias
        self.version = (self.weight._version, self.bias._version if self.bias is not None else None)
        self.stream = torch.npu.Stream(device=self.weight.device)
        self.lock = threading.Lock()
        self.inflight = None

    def try_forward(self, value):
        import torch
        import torch.distributed as dist

        if tuple(value.shape) != (1, 4680, 5, 128) or value.dtype != torch.bfloat16:
            return None
        if (value.device != self.weight.device or self.layer.weight is not self.weight
                or self.layer.bias is not self.bias or self.version != (
                    self.weight._version, self.bias._version if self.bias is not None else None)):
            self._fatal("Output overlap owner changed", (value, self))
        if not self.lock.acquire(blocking=False):
            self._fatal("Concurrent output overlap calls are unsupported", (value, self))
        compute = torch.npu.current_stream(value.device)
        owners = [value, self.weight, self.bias, self.group]
        self.inflight = owners
        try:
            ready = torch.npu.Event()
            ready.record(compute)
            self.stream.wait_event(ready)
            value.record_stream(self.stream)
            capacity = max(end-start for start, end in self.tiles) * 8 * 5 * 128
            slots = []
            for _ in range(2):
                send = torch.empty(capacity, dtype=value.dtype, device=value.device)
                receive = torch.empty_like(send)
                send.record_stream(self.stream)
                receive.record_stream(self.stream)
                slots.append((send, receive))
                owners.extend((send, receive))
            result = torch.empty((1, 585, 5120), dtype=value.dtype, device=value.device)
            owners.append(result)
            pending = {}

            def enqueue(index, released=None):
                start, end = self.tiles[index]
                length = end-start
                send, receive = slots[index % 2]
                count = 8 * length * 5 * 128
                send_view = send[:count].view(8, length, 5, 128)
                receive_view = receive[:count].view(8, length, 5, 128)
                with torch.npu.stream(self.stream):
                    if released is not None:
                        self.stream.wait_event(released)
                    pack_tile(value, start, end, send_view)
                    work = dist.all_to_all_single(receive_view, send_view, group=self.group, async_op=True)
                pending[index] = (work, receive_view)
                owners.extend((work, send_view, receive_view))

            enqueue(0)
            enqueue(1)
            for index, (start, end) in enumerate(self.tiles):
                work, received = pending.pop(index)
                # Work.wait inserts the HCCL end-event on this consumer stream.
                work.wait()
                dense = dense_tile(received)
                projected = self.layer(dense)
                result[:, start:end].copy_(projected)
                released = torch.npu.Event()
                released.record(compute)
                owners.extend((dense, projected, released))
                if index+2 < len(self.tiles):
                    enqueue(index+2, released)
            self.inflight = None
            return result
        except BaseException as error:
            self._fatal(str(error), tuple(owners)+(self,))
        finally:
            self.lock.release()

    @staticmethod
    def _fatal(message, owners):
        from vllm_omni.diffusion.layers.lingbot_sp8_fatal import report_failure

        report_failure("self_output_overlap", message, owners)
        raise RuntimeError(message)


def install(transformer, *, tiles=2, communication="native"):
    from vllm_omni.diffusion.layers.lingbot_sp8_fatal import ensure_protocol

    ensure_protocol()
    if communication not in ("native", "aiv"):
        raise ValueError("Unknown output overlap communication backend")
    first = transformer.blocks[0].self_attn
    group = aiv_group(first.ulysses_group, first.o.weight.device) if communication == "aiv" else None
    for block in transformer.blocks:
        block.self_attn._lingbot_output_overlap = OutputOverlap(block.self_attn, tiles=tiles, group=group)
