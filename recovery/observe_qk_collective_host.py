"""Diagnostic-only host boundaries around the ORIGINAL prepared-send collective.

No torch import, device operation, runtime source edit, or synchronization. The
caller supplies the already imported, pinned all_to_all_5D. Only that function's
module-global torch/dist bindings are temporarily delegated during its call;
the actual modules and their public functions are never patched. Use only in an
exclusive diagnostic process. Any failure is terminal: the caller must follow
its existing owner-quarantine/worker-retirement policy, never retry or fallback.
"""
from __future__ import annotations

from array import array
import hashlib
import json
from pathlib import Path
import resource
import struct
import sys
import threading
import time
import types

SCHEMA = "lingbot_prepared_collective_host_boundaries_v2"
CODE_FINGERPRINT_METHOD = "python_typed_code_fields_v1"
CODE_FIELDS = ("co_argcount", "co_posonlyargcount", "co_kwonlyargcount", "co_nlocals",
               "co_stacksize", "co_flags", "co_code", "co_consts", "co_names",
               "co_varnames", "co_filename", "co_name", "co_qualname", "co_firstlineno",
               "co_linetable", "co_exceptiontable", "co_freevars", "co_cellvars")
MARKS = ("call_begin", "allocation_begin", "allocation_end", "dist_begin",
         "dist_end", "call_end")
COLUMNS = ("wall_ns", "thread_cpu_ns", "thread_voluntary_context_switches",
           "thread_involuntary_context_switches")
INTERVALS = (("guard_before_allocation", 0, 1), ("receive_allocation", 1, 2),
             ("allocation_to_dist_bridge", 2, 3), ("public_dist_call", 3, 4),
             ("output_layout_and_return", 4, 5))
STATUS = {"call_raised": 1, "unexpected_boundary": 2, "foreign_thread": 4,
          "concurrent_invocation": 8, "globals_conflict": 16,
          "observation_failed": 32}


class ObservationError(RuntimeError):
    pass


def _canonical_json(value):
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), allow_nan=False)


def _typed_code_value(value):
    """Value semantics, never marshal's refcount/intern/reference graph flags."""
    kind = type(value)
    if value is None:
        return ["none"]
    if value is Ellipsis:
        return ["ellipsis"]
    if kind is bool:
        return ["bool", value]
    if kind is int:
        return ["int_hex", format(value, "x")]
    if kind is float:
        return ["float_ieee754", struct.pack(">d", value).hex()]
    if kind is complex:
        return ["complex_ieee754", struct.pack(">dd", value.real, value.imag).hex()]
    if kind is str:
        return ["str", value]
    if kind is bytes:
        return ["bytes_hex", value.hex()]
    if kind is tuple:
        return ["tuple", [_typed_code_value(item) for item in value]]
    if kind is frozenset:
        items = [_typed_code_value(item) for item in value]
        return ["frozenset", sorted(items, key=_canonical_json)]
    if kind is types.CodeType:
        return ["code", [[name, _typed_code_value(getattr(value, name))] for name in CODE_FIELDS]]
    raise ObservationError("unsupported code constant type: " + kind.__name__)


def code_fingerprint(code):
    """Cold-path, deterministic, typed digest of every selected code field.

    Includes nested code, literal types/bits, filename, line/exception tables,
    flags and interpreter cache tag. Source SHA remains a separate requirement.
    co_lnotab is derived/deprecated and excluded; no executable field is inferred
    from a raw marshal serialization. Unknown constant types fail closed.
    """
    if type(code) is not types.CodeType:
        raise ObservationError("a Python code object is required")
    payload = [CODE_FINGERPRINT_METHOD, sys.implementation.cache_tag, _typed_code_value(code)]
    return hashlib.sha256(_canonical_json(payload).encode("ascii")).hexdigest()


def fingerprint(collective):
    """Cold-path pins; record provenance separately from any trusted expected pins.

    Filename and Python cache tag are intentionally bound. A relocation requires
    a freshly derived expected pin from its trusted source, never accepting the
    loaded function's observed digest as its own expected value.
    """
    code = collective.__code__
    path = Path(code.co_filename).resolve()
    return {"source_path": str(path),
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "code_fingerprint_method": CODE_FINGERPRINT_METHOD,
            "code_sha256": code_fingerprint(code)}


class _Delegate:
    __slots__ = ("_observer", "_original", "_intercept_name", "_intercept")

    def __init__(self, observer, original, name, intercept):
        self._observer, self._original = observer, original
        self._intercept_name, self._intercept = name, intercept

    def __getattr__(self, name):
        self._observer._check_thread()
        if name == self._intercept_name:
            return self._intercept
        return getattr(self._original, name)


class PreparedCollectiveObserver:
    """One caller thread, one prepared-send invocation per preallocated row.

    All source/code hashing occurs at construction, before any observed calls.
    On each call, code and global binding identities must remain unchanged.
    Args/results live only in ordinary call frames; no activation, exception,
    traceback or frame is stored on the observer. IDs in state are integers.
    """

    def __init__(self, collective, rows, *, expected_source_sha256,
                 expected_code_sha256):
        if not isinstance(rows, int) or isinstance(rows, bool) or rows <= 0:
            raise ValueError("rows must be a positive integer")
        if not hasattr(resource, "RUSAGE_THREAD"):
            raise ObservationError("RUSAGE_THREAD is required")
        pins = fingerprint(collective)
        if pins["source_sha256"] != expected_source_sha256:
            raise ObservationError("collective source SHA mismatch")
        if pins["code_sha256"] != expected_code_sha256:
            raise ObservationError("collective code SHA mismatch")
        namespace = collective.__globals__
        if namespace.get(collective.__name__) is not collective:
            raise ObservationError("collective is not its original module binding")
        self.collective, self._code, self._namespace = collective, collective.__code__, namespace
        self._torch, self._dist = namespace["torch"], namespace["dist"]
        if isinstance(self._torch, _Delegate) or isinstance(self._dist, _Delegate):
            raise ObservationError("another observer is already installed")
        self._empty_original = self._torch.empty_like
        self._dist_original = self._dist.all_to_all_single
        self._torch_proxy = _Delegate(self, self._torch, "empty_like", self._empty_like)
        self._dist_proxy = _Delegate(self, self._dist, "all_to_all_single", self._all_to_all_single)
        self.rows, self.pins = rows, pins
        self.data = array("q", [-1]) * (rows * len(MARKS) * len(COLUMNS))
        self.status = array("q", [0]) * rows
        self.visited = array("q", [0]) * rows
        self.restored = array("q", [-1]) * rows
        self._lock = threading.Lock()
        self._thread_ident = threading.get_ident()
        self.native_thread_id = threading.get_native_id()
        self._active = False
        self._failed = False
        self._row = -1
        self._next = 0
        self._prepared_id = self._output_id = self._group_id = 0

    def _invalid(self, flag, message):
        self._failed = True
        if self._active:
            self.status[self._row] |= STATUS[flag]
        raise ObservationError(message)

    def _check_thread(self):
        if threading.get_ident() != self._thread_ident:
            self._invalid("foreign_thread", "foreign thread entered observed module")
        if not self._active:
            raise ObservationError("observer proxy used outside its invocation")

    def _mark(self, point):
        self._check_thread()
        if point != self._next:
            self._invalid("unexpected_boundary", "unexpected/repeated collective boundary")
        base = (self._row * len(MARKS) + point) * len(COLUMNS)
        self.data[base] = time.perf_counter_ns()
        self.data[base + 1] = time.thread_time_ns()
        usage = resource.getrusage(resource.RUSAGE_THREAD)
        self.data[base + 2] = usage.ru_nvcsw
        self.data[base + 3] = usage.ru_nivcsw
        self._next += 1

    def _empty_like(self, *args, **kwargs):
        self._check_thread()
        if len(args) != 1 or kwargs or id(args[0]) != self._prepared_id:
            self._invalid("unexpected_boundary", "unexpected empty_like operands")
        self._mark(1)
        result = self._empty_original(*args, **kwargs)
        self._output_id = id(result)
        self._mark(2)
        return result

    def _all_to_all_single(self, *args, **kwargs):
        self._check_thread()
        if (len(args) != 2 or id(args[0]) != self._output_id
                or id(args[1]) != self._prepared_id or set(kwargs) != {"group"}
                or id(kwargs["group"]) != self._group_id):
            self._invalid("unexpected_boundary", "unexpected all_to_all_single operands")
        self._mark(3)
        result = self._dist_original(*args, **kwargs)
        self._mark(4)
        return result

    def __call__(self, row, metadata, *, scatter_idx=3, gather_idx=1, group=None,
                 _prepared_send=None, _qk_padded_owner=None):
        # A rejecting observer never substitutes a fallback collective.
        if self._failed:
            raise ObservationError("observer already failed; restart the diagnostic worker")
        if threading.get_ident() != self._thread_ident:
            self._invalid("foreign_thread", "foreign thread invoked observer")
        if not self._lock.acquire(blocking=False):
            self._invalid("concurrent_invocation", "concurrent/reentrant observer invocation")
        try:
            if (not isinstance(row, int) or isinstance(row, bool)
                    or not 0 <= row < self.rows or self.visited[row]):
                raise ObservationError("invalid/already used observation row")
            if _prepared_send is None or (scatter_idx, gather_idx) != (3, 1):
                raise ObservationError("only explicit prepared-send head-to-sequence calls are supported")
            if (self.collective.__code__ is not self._code
                    or self._namespace.get(self.collective.__name__) is not self.collective
                    or self._namespace.get("torch") is not self._torch
                    or self._namespace.get("dist") is not self._dist
                    or self._torch.empty_like is not self._empty_original
                    or self._dist.all_to_all_single is not self._dist_original):
                raise ObservationError("original function/global/API identity changed")
            self._row, self._next = row, 0
            self._prepared_id, self._group_id = id(_prepared_send), id(group)
            self._output_id = 0
            self.visited[row] = 1
            self._active = True
            succeeded = False
            self._namespace["torch"], self._namespace["dist"] = self._torch_proxy, self._dist_proxy
            try:
                self._mark(0)
                if _qk_padded_owner is None:
                    result = self.collective(metadata, scatter_idx=scatter_idx, gather_idx=gather_idx,
                        group=group, _prepared_send=_prepared_send)
                else:
                    result = self.collective(metadata, scatter_idx=scatter_idx, gather_idx=gather_idx,
                        group=group, _prepared_send=_prepared_send, _qk_padded_owner=_qk_padded_owner)
                self._mark(5)
                succeeded = True
                return result
            except BaseException:
                self.status[row] |= STATUS["call_raised"]
                raise  # Preserve the original exception; no saved traceback, retry or fallback.
            finally:
                restored = True
                for name, proxy, original in (("torch", self._torch_proxy, self._torch),
                                               ("dist", self._dist_proxy, self._dist)):
                    if self._namespace.get(name) is proxy:
                        self._namespace[name] = original
                    else:
                        restored = False  # Never overwrite an unrelated replacement.
                        self.status[row] |= STATUS["globals_conflict"]
                self.restored[row] = int(restored)
                self._active = False
                self._prepared_id = self._output_id = self._group_id = 0
                if succeeded and self.status[row]:
                    raise ObservationError("collective observation invalid; terminal diagnostic failure")
        except BaseException:
            self._failed = True
            raise
        finally:
            self._lock.release()

    def export(self):
        """Cold-path JSON-compatible output; row completeness is explicit."""
        if self._active:
            raise ObservationError("cannot export an active observation")
        complete = [bool(self.visited[r] and not self.status[r] and self.restored[r] == 1
                         and all(self.data[(r * len(MARKS) + m) * len(COLUMNS)] >= 0
                                 for m in range(len(MARKS)))) for r in range(self.rows)]
        return {"schema": SCHEMA, "diagnostic_only": True, "performance_gate": False,
                "pins": dict(self.pins), "row_count": self.rows, "marks": MARKS,
                "columns": COLUMNS, "intervals": INTERVALS, "data": list(self.data),
                "status_bits": STATUS, "status": list(self.status),
                "visited": list(self.visited), "restored_original_bindings": list(self.restored),
                "complete_rows": complete, "all_rows_complete": all(complete) and not self._failed,
                "terminal_failure": self._failed,
                "native_thread_id": self.native_thread_id,
                "original_binding_ids": {"function": id(self.collective), "code": id(self._code),
                    "torch_module": id(self._torch), "dist_module": id(self._dist),
                    "empty_like": id(self._empty_original), "all_to_all_single": id(self._dist_original)},
                "observation_order": "wall, thread CPU, RUSAGE_THREAD; sequential, not atomic",
                "scope": "unchanged prepared-send all_to_all_5D; original use_sync default false",
                "interpretation": "disjoint host intervals including observer overhead; dist return is not device completion"}
