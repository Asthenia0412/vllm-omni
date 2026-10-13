# SPDX-License-Identifier: Apache-2.0
"""Construction-time selection of mutually exclusive, default-off experiments."""
from . import lingbot_fixed_modulation_stageB as original

PROVIDER = "_lingbot_fixed_modulation_provider"
CPP_FLAG = "lingbot_npu_fixed_modulation_cpp_v2"
CPP_STATE = "_lingbot_fixed_modulation_cpp_v2_state"
CPP_REQUESTED = "_lingbot_fixed_modulation_cpp_v2_requested"


def configure(pipeline, config):
    old, new = config.get(original.FLAG, False), config.get(CPP_FLAG, False)
    if type(old) is not bool or type(new) is not bool:
        raise ValueError("StageB modes must be explicit booleans")
    if old and new:
        raise ValueError("Python and CPP StageB modes are mutually exclusive")
    provider = original
    if new:
        from . import lingbot_fixed_modulation_stageB_cpp
        provider = lingbot_fixed_modulation_stageB_cpp
    provider.configure(pipeline, config)
    setattr(pipeline, PROVIDER, provider)
    setattr(pipeline.transformer, PROVIDER, provider)


def selected(pipeline):
    provider = for_transformer(pipeline.transformer)
    if getattr(pipeline, PROVIDER, None) is not provider:
        _fail(provider, pipeline.transformer, "StageB pipeline provider ownership changed", pipeline)
    return provider


def _fail(provider, transformer, message, *owners):
    error = RuntimeError(message)
    state = transformer.__dict__.get(provider.STATE)
    provider._poison(state, error, transformer, *owners)
    raise error


def for_transformer(transformer):
    actual = getattr(transformer, PROVIDER, None)
    if transformer.__dict__.get(CPP_REQUESTED, False) or transformer.__dict__.get(CPP_STATE) is not None:
        from . import lingbot_fixed_modulation_stageB_cpp
        if actual is not lingbot_fixed_modulation_stageB_cpp:
            _fail(lingbot_fixed_modulation_stageB_cpp, transformer, "CPP StageB provider ownership changed", actual)
        return actual
    if transformer.__dict__.get(original.REQUESTED, False) or transformer.__dict__.get(original.STATE) is not None:
        if actual is not original:
            _fail(original, transformer, "Python StageB provider ownership changed", actual)
        return actual
    # Independent, unconfigured transformers retain their valid native path.
    if actual is None or actual is original:
        return original
    _fail(original, transformer, "Disabled StageB retains an unexpected provider owner", actual)
