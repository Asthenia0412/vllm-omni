"""Default-off A3 BF16 reverse-Ulysses/output-projection candidate.

The installed eager operator does not expose transposeX2. Prepare an owned,
contiguous transposed weight once after loading. No weight conversion belongs
in the timed forward path. Admission is separate from implementation.
"""
from __future__ import annotations

import threading

_CALL_LOCK = threading.Lock()


def validate_configuration(enabled, quant_config):
    if not enabled:
        return
    from vllm_omni.diffusion.models.lingbot_world.npu_quantization import selected_linear
    if quant_config.mode != "off" and any(selected_linear(
        f"blocks.{index}.self_attn.o", quant_config.scope,
        projections=quant_config.projections, allowlist=quant_config.allowlist,
        exclude=quant_config.exclude,
    ) for index in range(40)):
        raise ValueError("SP8 BF16 MC2 and self_out quantization are mutually exclusive")


class OutputMC2:
    def __init__(self, module):
        import torch
        import torch_npu
        layer = module.o
        if (module.ulysses_world_size != 8 or layer.tp_size != 1 or layer.return_bias
                or layer.skip_bias_add or getattr(layer, "_lingbot_quant_mode", None)):
            raise ValueError("Output MC2 requires SP8/TP1 and an unquantized fused-bias projection")
        weight = layer.weight
        if weight.dtype != torch.bfloat16 or weight.device.type != "npu" or tuple(weight.shape) != (5120, 5120):
            raise ValueError("Output MC2 requires the actual BF16 5120x5120 output weight")
        if torch.distributed.get_world_size(module.ulysses_group) != 8:
            raise ValueError("Output MC2 communication group must contain the original eight ranks")
        self.group = module.ulysses_group
        self.rank = torch.distributed.get_rank(self.group)
        self.hcom = self.group._get_backend(weight.device).get_hccl_comm_name(self.rank)
        # Preparation allocation is owned for the entire loaded-model lifetime.
        self.weight = weight.transpose(0, 1).contiguous()
        if torch_npu.get_npu_format(self.weight) != 2:
            self.weight = torch_npu.npu_format_cast(self.weight, 2)
        self.bias = layer.bias.float().contiguous() if layer.bias is not None else None
        self.original_weight = weight
        self.original_version = weight._version
        self.original_pointer = weight.data_ptr()
        self.layer = layer
        self.original_bias = layer.bias
        self.bias_version = layer.bias._version if layer.bias is not None else None
        self.lock = _CALL_LOCK
        self.inflight = None
        self.op = torch_npu.npu_all_to_all_matmul

    def try_forward(self, output):
        import torch
        import torch_npu
        # Ordinary unsupported geometry chooses the existing path before any
        # collective or destination write. The model's token/head geometry is
        # uniform across the original SP group.
        if (tuple(output.shape) != (1, 4680, 5, 128) or output.dtype != torch.bfloat16
                ):
            return None
        def invalid_owner(message):
            from vllm_omni.diffusion.layers.lingbot_sp8_fatal import report_failure
            report_failure("self_output_mc2_metadata", message,
                           (output,self.original_weight,self.original_bias,self))
            raise RuntimeError(message)
        if output.device != self.weight.device:
            invalid_owner("MC2 activation and immutable weight owners use different devices")
        if (self.layer.weight is not self.original_weight or self.layer.bias is not self.original_bias
                or self.original_weight.data_ptr()!=self.original_pointer
                or self.original_weight._version != self.original_version
                or (self.original_bias is not None and self.original_bias._version != self.bias_version)):
            invalid_owner("MC2 immutable weight owner changed; reload the model")
        if not self.lock.acquire(blocking=False):
            invalid_owner("Concurrent calls to the SP8 MC2 candidate are forbidden")
        # Keep every owner strongly referenced across an admitted-call error;
        # the supervisor reclaims all workers, never retries the native path.
        self.inflight = (output, self.weight, self.bias, self.group)
        try:
            # PTA may re-label a shared base-format storage descriptor when
            # casting NCHW0 to ND2. Own the normalization arena so the caller's
            # original physical descriptor survives every invocation.
            normalized = output.clone(memory_format=torch.contiguous_format)
            if torch_npu.get_npu_format(normalized)!=2:
                normalized = torch_npu.npu_format_cast(normalized,2)
            self.inflight += (normalized,)
            result, _ = self.op(normalized.reshape(4680, 640), self.weight, self.hcom, 8,
                                bias=self.bias, all2all_axes=[-2, -1], all2all_out_flag=False)
            if tuple(result.shape) != (585, 5120) or result.dtype != output.dtype:
                raise RuntimeError("MC2 returned an invalid output descriptor after launch")
            self.inflight = None  # Existing eager allocator owns asynchronous dependencies.
            return result.unsqueeze(0)
        except BaseException as error:
            from vllm_omni.diffusion.layers.lingbot_sp8_fatal import report_failure
            report_failure("self_output_mc2", str(error), self.inflight + (self,))
            raise
        finally:
            self.lock.release()


def install(transformer):
    from vllm_omni.diffusion.layers.lingbot_sp8_fatal import ensure_protocol
    ensure_protocol()
    for block in transformer.blocks:
        block.self_attn._lingbot_output_mc2 = OutputMC2(block.self_attn)
