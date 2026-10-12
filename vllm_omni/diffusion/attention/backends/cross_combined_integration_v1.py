"""Explicit experiment-only canonical SDPA integration; default is native.

This module imports only stdlib until a new configuration and environment both
enable it. The frozen adapter performs the unchanged route/runtime/owner and
poststart guards. Its C++ operation uses the normal public native dispatcher.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading

ENABLE = 'ROOT_LINGBOT_CROSS_COMBINED_INTEGRATION_V1'
CONFIG_KEY = 'lingbot_npu_cross_combined_guard_cpp_v1'
MANIFEST = 'ROOT_LINGBOT_CROSS_COMBINED_INTEGRATION_MANIFEST'
MANIFEST_SHA = MANIFEST + '_SHA256'
ADAPTER_ENABLE = 'LINGBOT_CROSS_COMBINED_CPP_ENABLE'
_LOCK = threading.RLock()
_BOUND = None
_QUARANTINE = []


def _need(value, message):
    if not value:
        raise RuntimeError(message)


def _pin(path):
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


class _Route:
    def __init__(self, adapter, fatal, manifest, policy, model_config):
        self._module = adapter
        self._adapter = adapter.CrossBSNDV1(enabled=True)
        self._fatal = fatal
        self._manifest = manifest
        self._policy = policy
        self._model_config = model_config
        self._poisoned = False

    def try_forward(self, query, key, value, **kwargs):
        _need(not self._poisoned, 'Combined cross worker poisoned; restart all eight workers')
        try:
            _need(all(os.environ.get(k) == v for k, v in self._policy.items())
                  and self._model_config.get(CONFIG_KEY) is True,
                  'Combined cross experiment policy changed after startup')
            # None is only a supported prestart refusal. The canonical SDPA
            # caller then executes its unchanged original path.
            return self._adapter.try_forward(query, key, value, **kwargs)
        except BaseException as error:
            self._poisoned = True
            owners = (query, key, value, self, self._module,
                      self._module._BINDING, error, sys.exc_info())
            with _LOCK:
                _QUARANTINE.append(owners)
            # Preserve the original exception and strong owners even if the
            # existing supervisor channel cannot write its failure record.
            self._fatal.report_failure('cross_combined_cpp_v1', str(error), owners)
            raise


def selected_adapter(*, role):
    """Configure once per fresh worker; old configurations never opt in."""
    global _BOUND
    if role != 'cross':
        return None
    enabled = os.environ.get(ENABLE, '0')
    _need(enabled in ('0', '1'), ENABLE + ' must be 0 or 1')
    if enabled == '0':
        return None
    from vllm_omni.diffusion.config import get_current_diffusion_config_or_none
    config = get_current_diffusion_config_or_none()
    model_config = getattr(config, 'model_config', None)
    _need(isinstance(model_config, dict) and model_config.get(CONFIG_KEY) is True,
          'New explicit bool model configuration required; old configurations cannot enable this route')
    _need(getattr(config, 'model_class_name', None) == 'LingBotWorldCausalDMDPipeline',
          'This integration is limited to the original LingBot causal model')
    policy = {k: os.environ.get(k) for k in
              (ENABLE, MANIFEST, MANIFEST_SHA, ADAPTER_ENABLE,
               'LINGBOT_NPU_SLOT_GRANTED', 'LINGBOT_ROOT_NPU_SLOT_GRANTED',
               'ROOT_LINGBOT_SP8_FATAL_DIR', 'ROOT_LINGBOT_SP8_FATAL_NONCE',
               'ROOT_LINGBOT_SP8_EXPERIMENT_MODE')}
    _need(policy[ADAPTER_ENABLE] == policy['LINGBOT_NPU_SLOT_GRANTED']
          == policy['LINGBOT_ROOT_NPU_SLOT_GRANTED'] == '1', 'Exact candidate enable and both root grants required')
    _need(policy['ROOT_LINGBOT_SP8_EXPERIMENT_MODE'] in ('correctness', 'timing', 'profile'),
          'Root-owned isolated experiment controller required')
    with _LOCK:
        if _BOUND is not None:
            _need(_BOUND._policy == policy and _BOUND._model_config is model_config
                  and not _BOUND._poisoned, 'Immutable configured cross provider/policy required')
            return _BOUND
        path = Path(policy[MANIFEST] or '')
        _need(path.is_absolute() and _pin(path)['sha256'] == policy[MANIFEST_SHA],
              'Exact new integration manifest required')
        manifest = json.loads(path.read_text())
        _need(manifest.get('schema') == 'lingbot_cross_combined_production_integration_CPU_contract_v1'
              and manifest.get('source_contract_verified') is True
              and manifest.get('model_admitted') is False,
              'New source-only experiment manifest required; no inherited model admission')
        _need(_pin(__file__)['sha256'] == manifest['helper']['sha256']
              and _pin(Path(__file__).with_name('sdpa.py'))['sha256'] == manifest['patched_SDPA_sha256'],
              'Actual canonical SDPA/helper overlay differs')
        adapter_pin = manifest['adapter']
        _need(_pin(adapter_pin['path']) == adapter_pin, 'Frozen combined adapter changed')
        spec = importlib.util.spec_from_file_location('lingbot_production_combined_cross_adapter_v1', adapter_pin['path'])
        adapter = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = adapter
        spec.loader.exec_module(adapter)
        from vllm_omni.diffusion.layers import lingbot_sp8_fatal as fatal
        _need(_pin(fatal.__file__)['sha256'] == manifest['fatal_source']['sha256']
              and fatal.ensure_protocol() is True, 'Original all-eight-worker fatal channel required before candidate start')
        host = manifest['host_provider']
        adapter.configure(library=host['library']['path'], library_sha256=host['library']['sha256'],
                          build_receipt=host['build']['path'], build_receipt_sha256=host['build']['sha256'])
        _BOUND = _Route(adapter, fatal, manifest, policy, model_config)
        return _BOUND
