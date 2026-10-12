# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from functools import wraps
import os
from typing import Any

import torch
import torch.nn as nn
from vllm.logger import init_logger

logger = init_logger(__name__)


class NPURegionalGraphCallable:
    """Bounded exact-signature compilation of a *pure* tensor region.

    This is deliberately separate from Inductor and whole DiT compilation.
    A region must not contain RNG, sampler updates, KV writes or session state.
    Unknown signatures after the limit execute eagerly, without evicting graph
    storage that a backend replay may still own. Capture/runtime failures are
    fatal: continuing with eager code can use poisoned allocator/capture state.
    """

    def __init__(self, forward: Any, *, max_graphs: int = 6, name: str = "region", graph_pool: Any = None) -> None:
        if isinstance(max_graphs, bool) or not isinstance(max_graphs, int) or not 1 <= max_graphs <= 6:
            raise ValueError("NPU regional max_graphs must be an integer in [1, 6]")
        self.forward = forward
        # Dynamo may share an FX/backend object for structurally identical
        # bound methods from different layers. npugraph_ex then sees40weight
        # addresses as captures in one object's six-entry cache and disables
        # that entire object. A user-file entry avoids Dynamo wrapping Torch
        # source methods with its own shared wrap_inline entry. Give that pure
        # entry a distinct code key and retain the original bound weight owner.
        self.capture_forward = self._isolated_forward(forward, name=name, owner=id(self))
        self.max_graphs = max_graphs
        self.name = name
        self.graphs: dict[tuple[Any, ...], Any] = {}
        self.failed = False
        self.hits = 0
        self.eager_misses = 0
        self.graph_pool = graph_pool

    @staticmethod
    def _isolated_forward(forward: Any, *, name: str, owner: int) -> Any:
        from types import FunctionType, MethodType

        bound = isinstance(forward, MethodType)
        function = forward.__func__ if bound else forward
        if not isinstance(function, FunctionType):
            raise TypeError("NPU pure region requires a Python function or bound method for independent cache ownership")
        # This must originate in a user file. Cloning nn.Sequential.forward
        # itself is insufficient: Dynamo's trace rules wrap Torch-source
        # functions in shared external_utils.wrap_inline code before tracing.
        # The entry has no hooks, RNG, mutation or session state.
        def entry(*args: Any, **kwargs: Any) -> Any:
            return forward(*args, **kwargs)

        label = name.replace(".", "_") + "_" + format(owner, "x")
        code = entry.__code__.replace(
            co_name=entry.__code__.co_name + "__lingbot_" + label,
            co_qualname=entry.__code__.co_qualname + "__lingbot_" + label,
        )
        # co_filename remains the real source path for stack/source attribution.
        # co_name+co_qualname make code inequality explicit, even for equality-
        # based Python cache keys; an address-only identity is insufficient.
        isolated = FunctionType(code, entry.__globals__, code.co_name,
                                entry.__defaults__, entry.__closure__)
        isolated.__kwdefaults__ = entry.__kwdefaults__
        isolated.__annotations__ = dict(entry.__annotations__)
        isolated.__module__ = entry.__module__
        isolated.__qualname__ = code.co_qualname
        isolated.__doc__ = function.__doc__
        return isolated

    @staticmethod
    def _signature(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return ("tensor", tuple(value.shape), tuple(value.stride()), value.storage_offset(),
                    value.dtype, value.device, value.requires_grad)
        if isinstance(value, (tuple, list)):
            return (type(value), tuple(NPURegionalGraphCallable._signature(item) for item in value))
        if isinstance(value, dict):
            return (dict, tuple((key, NPURegionalGraphCallable._signature(item)) for key, item in sorted(value.items())))
        if value is None or isinstance(value, (bool, int, float, str)):
            return (type(value), value)
        raise TypeError(f"NPU pure regions accept tensors and scalar constants, got {type(value).__name__}")

    def _profile_scope(self, phase: str) -> Any:
        # The scope surrounds compilation/replay and owned output copies in
        # the eager caller. It is never part of the captured tensor region.
        # No events, clocks or device synchronization are introduced.
        from contextlib import nullcontext

        if os.environ.get("LINGBOT_PROFILE_STAGE_RANGES", "0") != "1":
            return nullcontext()
        return torch.profiler.record_function(f"lingbot.npu_region.{self.name}.{phase}")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.failed:
            raise RuntimeError(f"NPU region {self.name} cannot continue after capture failure; restart the stage process")
        key = (self._signature(args), self._signature(kwargs))
        compiled = self.graphs.get(key)
        if compiled is None:
            if len(self.graphs) >= self.max_graphs:
                self.eager_misses += 1
                return self.forward(*args, **kwargs)
            try:
                # The backend otherwise returns a persistent tensor overwritten by
                # a later replay of this same region, even with a shared pool.
                # Leaves do not alias their inputs; return owned output clones.
                # Fullgraph rejects stateful Python/graph breaks. No CUDA
                # reduce-overhead mode or dynamic shape padding is requested.
                options = {"capture_limit": self.max_graphs, "pattern_fusion_pass": False,
                           "clone_input": True, "clone_output": True}
                if self.graph_pool is not None:
                    options["use_graph_pool"] = self.graph_pool
                compiled = torch.compile(
                    self.capture_forward, backend="npugraph_ex", fullgraph=True, dynamic=False,
                    options=options,
                )
                with self._profile_scope("capture"):
                    result = compiled(*args, **kwargs)
            except Exception as exc:
                self.failed = True
                raise RuntimeError(
                    f"NPU region {self.name} capture failed; restart the stage process and "
                    "remove model_config.lingbot_npu_graph_backend before retrying"
                ) from exc
            self.graphs[key] = compiled
            return result
        self.hits += 1
        try:
            with self._profile_scope("replay"):
                return compiled(*args, **kwargs)
        except Exception as exc:
            self.failed = True
            raise RuntimeError(f"NPU region {self.name} replay failed; restart the stage process") from exc


def regionally_compile_npu_pure(
    model: nn.Module, *, regions: tuple[str, ...] = ("ffn", "qkv"), max_graphs: int = 6, graph_pool: Any = None,
) -> nn.Module:
    """Opt-in LingBot FFN/QKV leaves; attention and cache management stay eager.

    TP collectives are not admitted by this initial backend. The normal SP
    all-to-all lives outside both selected regions and keeps its eager order.
    """
    if not regions or set(regions) - {"ffn", "qkv"}:
        raise ValueError("NPU graph regions must be a nonempty subset of ('ffn', 'qkv')")
    if "ffn" in regions and os.environ.get("LINGBOT_CAPTURE_ACTIVATIONS", "0") == "1":
        raise ValueError("FFN regional graphs require LINGBOT_CAPTURE_ACTIVATIONS=0; diagnostic child hooks are stateful")
    selected = []
    for name, submod in model.named_modules():
        parts = name.split(".")
        if len(parts) < 3 or parts[0] != "blocks" or not parts[1].isdigit():
            continue
        if not (("ffn" in regions and parts[2:] == ["ffn"])
                or ("qkv" in regions and parts[2:] == ["self_attn", "qkv"])):
            continue
        if any(int(getattr(child, "tp_size", 1)) != 1 for child in submod.modules()):
            raise ValueError(f"NPU graph region {name} requires tensor_parallel_size=1")
        if getattr(submod, "_omni_npu_region", None) is not None:
            raise ValueError(f"NPU graph region {name} is already installed")
        if getattr(submod, "_hook_registry", None) is not None:
            raise ValueError(f"NPU graph region {name} does not support layer offload hooks")
        runner = NPURegionalGraphCallable(submod.forward, max_graphs=max_graphs, name=name, graph_pool=graph_pool)

        @wraps(submod.forward)
        def compiled_forward(*args: Any, _runner: Any = runner, **kwargs: Any) -> Any:
            return _runner(*args, **kwargs)

        selected.append((submod, runner, compiled_forward))
    if not selected:
        raise ValueError("NPU pure regional graphs found no LingBot blocks.ffn/self_attn.qkv regions")
    for submod, runner, forward in selected:
        submod._omni_npu_region = runner
        submod.forward = forward
    logger.info("Configured %d pure NPU regions with npugraph_ex (max %d signatures/region)",
                len(selected), max_graphs)
    return model


def _matches_repeated_block(
    name: str,
    module: nn.Module,
    repeated_blocks: list[str],
    repeated_block_attrs: list[str],
) -> bool:
    class_name = module.__class__.__name__
    if class_name in repeated_blocks:
        return True

    for attr in ("_fsdp_wrapped_module", "module", "_orig_mod"):
        wrapped = getattr(module, attr, None)
        if wrapped is not None and wrapped.__class__.__name__ in repeated_blocks:
            return True

    parts = name.split(".")
    return len(parts) >= 2 and parts[-2] in repeated_block_attrs and parts[-1].isdigit()


def regionally_compile(
    model: nn.Module,
    *compile_args: Any,
    **compile_kwargs: Any,
) -> nn.Module:
    """
    Apply regional compilation to a PyTorch model.

    Args:
        model: The PyTorch model instance to compile
        *compile_args: Positional arguments forwarded to torch.compile
        **compile_kwargs: Keyword arguments forwarded to torch.compile

    Returns:
        The same model instance (modified in-place)
    """
    # Get the list of repeated blocks from the model
    repeated_blocks = getattr(model, "_repeated_blocks", None)

    if not repeated_blocks:
        logger.warning("Regional compilation skipped because the model does not define `_repeated_blocks`.")
        return model

    repeated_block_attrs = getattr(model, "_layerwise_offload_blocks_attrs", [])

    # Build all compiled callables before mutating the model. This keeps setup
    # failures atomic: callers can safely continue with the uncompiled model if
    # torch.compile raises synchronously for any repeated block.
    # Keep the pre-hook callable separately.  DLO and the ordinary layerwise
    # offloader install a HookRegistry wrapper in ``module.forward``.  That
    # wrapper performs stream/event synchronization and storage rebinding, so
    # compiling it pulls offload control flow into the graph.  Compile the
    # original block compute instead and leave the wrapper outside the graph.
    compiled_forwards: list[tuple[nn.Module, Any | None, Any]] = []
    for name, submod in model.named_modules():
        if _matches_repeated_block(name, submod, repeated_blocks, repeated_block_attrs):
            # Compile the block compute while keeping nn.Module.__call__ hooks
            # outside the compiled graph. If a HookRegistry is already
            # installed, ``submod.forward`` is the hook dispatcher and the
            # original callable is stored in ``_omni_original_forward``.
            # NOTE: anything that wraps this callable must stay
            # signature-transparent — cache-dit's BlockAdapter matches blocks
            # by inspecting forward's parameter names and return annotation,
            # which torch.compile preserves.
            original_forward = getattr(submod, "_omni_original_forward", None)
            forward_to_compile = original_forward if original_forward is not None else submod.forward
            compiled_forwards.append(
                (
                    submod,
                    original_forward,
                    torch.compile(forward_to_compile, *compile_args, **compile_kwargs),
                )
            )

    if not compiled_forwards:
        logger.warning(f"Regional compilation skipped because {repeated_blocks} classes are not found in the model.")
    else:
        for submod, original_forward, compiled_forward in compiled_forwards:
            if original_forward is None:
                submod.forward = compiled_forward
                continue

            # Keep HookRegistry's dispatcher as ``forward`` and replace only
            # the callable it dispatches to.  Hooks such as MagCache retain a
            # reference to the original callable as well, so update those
            # references when they still point at the pre-compile function.
            submod._omni_original_forward = compiled_forward
            registry = getattr(submod, "_hook_registry", None)
            for hook in getattr(registry, "_hooks", {}).values():
                fn_ref = getattr(hook, "fn_ref", None)
                if fn_ref is not None and fn_ref.original_forward is original_forward:
                    fn_ref.original_forward = compiled_forward
        logger.info(
            "Regional compilation applied to %d module(s) for repeated blocks %s.",
            len(compiled_forwards),
            repeated_blocks,
        )

    return model
