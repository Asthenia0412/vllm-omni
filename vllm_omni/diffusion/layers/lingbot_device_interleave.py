"""Default-off experiment: use the same owned clone.out region in every call.

No CANN launch or format mutation is implemented here. The registered clone.out
and blocking CPU transfer retain the framework's queue and stream behavior.
Internal Transpose/TensorMove work is allowed and must be charged by the screen.
"""

from dataclasses import dataclass
import traceback


REDZONE_BYTES = 64
_POISONED = False
_QUARANTINE = []
_FAILURES = []


class InvalidOwnerError(ValueError):
    pass


class InterleaveStartedError(RuntimeError):
    pass


@dataclass
class OwnedRegion:
    source: object
    permuted_source: object
    owner: object
    target: object
    source_descriptor: dict
    owner_descriptor: dict
    target_descriptor: dict
    started: bool = False


def _descriptor(tensor, physical_format):
    storage = tensor.untyped_storage()
    return dict(shape=list(tensor.shape), stride=list(tensor.stride()),
                storage_offset=int(tensor.storage_offset()),
                storage_pointer=int(storage.data_ptr()),
                storage_bytes=int(storage.nbytes()),
                data_pointer=int(tensor.data_ptr()),
                dtype=str(tensor.dtype), device=str(tensor.device),
                physical_format=int(physical_format))


def _check_poison():
    if _POISONED:
        raise InterleaveStartedError(
            "Owned interleave is poisoned; terminate the request and restart all eight workers")


def prepare_region(planar, *, guarded_target=None, guarded_owner=None):
    """Return a validated owner or None for unsupported input, before writing.

    The caller may select its original interleave route only on None. Invalid
    owner metadata raises. A provided diagnostic target has exactly the same
    geometry as the ordinary allocation; only its prefilled contents differ.
    """
    _check_poison()
    import torch

    if (planar.device.type not in ("npu", "privateuseone")
            or planar.dtype != torch.uint8 or planar.ndim != 4):
        return None
    channels, frames, height, width = tuple(planar.shape)
    if (channels, height, width) != (3, 480, 832) or frames not in (9, 12):
        return None
    expected_stride = (frames * height * width, height * width, width, 1)
    if (tuple(planar.stride()) != expected_stride or planar.storage_offset() != 0
            or not planar.is_contiguous()):
        return None
    import torch_npu

    # NCHW is the actual captured dense producer format. Never relabel storage.
    source_format = int(torch_npu.get_npu_format(planar))
    if source_format != 0:
        return None
    count = 3 * frames * height * width
    if planar.element_size() != 1 or planar.untyped_storage().nbytes() < count:
        raise InvalidOwnerError("Planar source exceeds its actual uint8 allocation")
    shape = (frames, height, width, 3)
    if guarded_target is None:
        if guarded_owner is not None:
            raise InvalidOwnerError("A supplied owner requires its exact target view")
        owner = torch.empty((count + 2 * REDZONE_BYTES,), dtype=torch.uint8,
                            device=planar.device)
        target = owner.narrow(0, REDZONE_BYTES, count).view(shape)
    else:
        target = guarded_target
        owner = guarded_owner if guarded_owner is not None else getattr(target, "_base", None)
        if owner is None:
            raise InvalidOwnerError("Diagnostic output must retain its complete arena owner")
    owner_storage = owner.untyped_storage()
    target_storage = target.untyped_storage()
    expected_target_stride = (height * width * 3, width * 3, 3, 1)
    if (owner.device != planar.device or target.device != planar.device
            or owner.dtype != torch.uint8 or target.dtype != torch.uint8
            or owner.ndim != 1 or tuple(owner.shape) != (count + 2 * REDZONE_BYTES,)
            or owner.storage_offset() != 0 or not owner.is_contiguous()
            or owner.element_size() != 1 or target.element_size() != 1
            or int(owner_storage.nbytes()) != count + 2 * REDZONE_BYTES
            or tuple(target.shape) != shape or tuple(target.stride()) != expected_target_stride
            or not target.is_contiguous() or target.storage_offset() != REDZONE_BYTES
            or int(target_storage.data_ptr()) != int(owner_storage.data_ptr())
            or int(target_storage.nbytes()) != int(owner_storage.nbytes())
            or int(target.data_ptr()) != int(owner.data_ptr()) + REDZONE_BYTES
            or int(owner_storage.data_ptr()) == int(planar.untyped_storage().data_ptr())):
        raise InvalidOwnerError("Invalid independent uint8[N+128] owner/FHWC offset64 region")
    owner_format = int(torch_npu.get_npu_format(owner))
    target_format = int(torch_npu.get_npu_format(target))
    if owner_format != 2 or target_format != 2:
        raise InvalidOwnerError("Owned output must preserve its raw ND storage format")
    return OwnedRegion(planar, planar.permute(1, 2, 3, 0), owner, target,
                       _descriptor(planar, source_format), _descriptor(owner, owner_format),
                       _descriptor(target, target_format))


def _quarantine(region, result, host, error):
    global _POISONED
    _POISONED = True
    # Keep every actual tensor/storage owner alive until the worker exits. This
    # includes the alias source used by clone.out and a possibly produced host.
    _QUARANTINE.append((region, result, host))
    _FAILURES.append(dict(source=region.source_descriptor, owner=region.owner_descriptor,
                          target=region.target_descriptor, started=region.started,
                          error=repr(error), traceback=traceback.format_exc(),
                          action="terminate request and restart all eight workers"))


def execute_to_host(region):
    """Write the exact owned region, then retain it through blocking D2H."""
    _check_poison()
    import torch
    import torch_npu

    if not isinstance(region, OwnedRegion) or region.started:
        raise InvalidOwnerError("An owned interleave region is valid for one submission only")
    for tensor, expected in ((region.source, region.source_descriptor),
                             (region.owner, region.owner_descriptor),
                             (region.target, region.target_descriptor)):
        if _descriptor(tensor, torch_npu.get_npu_format(tensor)) != expected:
            raise InvalidOwnerError("An owned interleave descriptor changed before submission")
    result = host = None
    try:
        region.started = True
        result = torch.ops.aten.clone.out(region.permuted_source,
                                         memory_format=torch.contiguous_format,
                                         out=region.target)
        if result is not region.target:
            raise InvalidOwnerError("clone.out replaced its validated output owner")
        for tensor, expected in ((region.owner, region.owner_descriptor),
                                 (region.target, region.target_descriptor)):
            if _descriptor(tensor, torch_npu.get_npu_format(tensor)) != expected:
                raise InvalidOwnerError("clone.out changed its owned output descriptor")
        host = result.cpu()
        # Standard blocking .cpu() has completed before region can be released.
        return host.numpy()
    except BaseException as error:
        _quarantine(region, result, host, error)
        raise InterleaveStartedError(
            "Owned interleave failed after clone.out submission; retained owners, "
            "terminate request and restart all eight workers") from error


def device_interleave_to_host(planar, *, guarded_target=None, guarded_owner=None):
    region = prepare_region(planar, guarded_target=guarded_target,
                            guarded_owner=guarded_owner)
    if region is None:
        return None
    return execute_to_host(region)


def failure_state():
    """Small diagnostic receipt; actual quarantined owners are intentionally held."""
    return dict(poisoned=_POISONED, quarantined_regions=len(_QUARANTINE),
                failures=list(_FAILURES))
