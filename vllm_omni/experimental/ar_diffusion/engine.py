# SPDX-License-Identifier: Apache-2.0
"""The AR-Diffusion Engine (AR-Diffusion)."""

from __future__ import annotations

from vllm.logger import init_logger

from vllm_omni.diffusion.diffusion_engine import DiffusionEngine

logger = init_logger(__name__)

#: Import path of the runner the AR-Diffusion engine routes its workers to.
AR_DIFFUSION_MODEL_RUNNER_CLS = "vllm_omni.experimental.ar_diffusion.runner.ARDiffusionModelRunner"


class ARDiffusionEngine(DiffusionEngine):
    """AR-Diffusion engine with engine-level KV cache management.

    AR-Diffusion serves autoregressive / chunked blockwise-causal diffusion models
    (world models, AR-DiT) that materialize persistent attention KV. It reuses
    vLLM's paged KV stack (``KVCacheManager`` / ``BlockPool`` / ``BlockTables``)
    as a library, driven from the engine rather than hand-rolled inside each
    model.

    It is selected per model via ``OmniDiffusionConfig.engine_backend = "ar_diffusion"``
    (resolved by :meth:`DiffusionEngine.make_engine`), so models that do not opt
    in keep using the base ``DiffusionEngine`` unchanged.

    Architecture note: in the multiproc setup the KV cache lives in the worker /
    runner process (GPU side), co-located with the model and KV tensors — so the
    actual KV *body* is :class:`~vllm_omni.experimental.ar_diffusion.kv_cache.manager.ARDiffusionKVCache`, owned
    by :class:`~vllm_omni.experimental.ar_diffusion.runner.ARDiffusionModelRunner`. ``ARDiffusionEngine`` itself is the
    thin selection / injection seam; it wires the AR-Diffusion executor → worker → runner
    so capable pipelines run against runner-owned KV session state.
    """

    #: Workers of this engine build the AR-Diffusion runner (unless the config
    #: sets an explicit ``diffusion_model_runner_cls`` override). Declared on
    #: the class so routing needs no od_config mutation at engine init.
    default_diffusion_model_runner_cls = AR_DIFFUSION_MODEL_RUNNER_CLS

    def _abort_requests(self, request_ids) -> None:
        request_ids = [request_ids] if isinstance(request_ids, str) else list(request_ids)
        request_ids = list(dict.fromkeys(request_ids))
        super()._abort_requests(request_ids)
        if request_ids and getattr(self.od_config, "step_execution", False):
            # The busy loop drains aborts after the current executor wave has
            # returned. An idle scheduler sends no further execute_stepwise
            # wave, so explicitly retire the worker-owned AR sessions now.
            self.executor.collective_rpc("release_ar_diffusion_requests", args=(request_ids,))
