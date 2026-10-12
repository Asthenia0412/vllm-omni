"""Default-off tensor-only QKV and FFN graphs, excluding all cache writes."""
from __future__ import annotations

from types import MethodType


def install(transformer):
    from vllm_omni.platforms.npu.graph_tools import NPUExactGraphRunner

    for index, block in enumerate(transformer.blocks):
        for region, module, tuple_result in (("qkv", block.self_attn.qkv, True), ("ffn", block.ffn, False)):
            if hasattr(module, "_lingbot_region_runner"):
                raise ValueError("Regional graph is already installed")
            original = module.forward
            parameters = tuple(module.parameters())+tuple(module.buffers())
            identities = tuple((id(p), p.data_ptr(), p._version) for p in parameters)
            runner = NPUExactGraphRunner(max_graphs=4, component_name=f"LingBot layer {index} {region}",
                                        disable_config_hint="lingbot_npu_regional_graph=False")

            def forward(self, value, *, original=original, parameters=parameters, identities=identities,
                        runner=runner, tuple_result=tuple_result, index=index, region=region):
                current = tuple(self.parameters())+tuple(self.buffers())
                if tuple((id(p), p.data_ptr(), p._version) for p in current) != identities:
                    raise RuntimeError("Regional graph parameter owners changed; reload the model")
                if value.device.type != "npu":
                    return original(value)

                def compute(tensor):
                    output = original(tensor)
                    if tuple_result:
                        if output[1] is not None:
                            raise ValueError("QKV graph requires the existing fused-bias contract")
                        output = output[0]
                    return (output,)

                # Weight ownership belongs to this graph instance. Pin the
                # identities in constants as well as validating them above.
                try:
                    output, = runner.run(region, (value,), (index, identities), compute)
                except BaseException as error:
                    from vllm_omni.diffusion.layers.lingbot_sp8_fatal import report_failure

                    report_failure("regional_graph", str(error), (self, value, parameters, runner))
                    raise
                return (output, None) if tuple_result else output

            module._lingbot_region_runner = runner
            module.forward = MethodType(forward, module)
