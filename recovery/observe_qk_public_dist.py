"""Diagnostic v3: thread-local C-call boundaries inside the unmodified public API.

No torch import/device calls. C-call spans include argument/return conversion and
GIL release/reacquisition; they are not C++ execution or network measurements.
The observer never reads f_locals or retains the Work, tensors or Python frames.
"""
from array import array
import dis
import hashlib
from pathlib import Path
import resource
import sys
import sysconfig
import threading
import time
import types

import observe_qk_collective_host as base

SCHEMA = "lingbot_prepared_collective_public_dist_v3"
PUBLIC_SCHEMA = "lingbot_public_dist_cprofile_v3"
METHOD = "thread_local_exact_body_c_call_boundaries_v1"
CODE_FINGERPRINT_METHOD = base.CODE_FINGERPRINT_METHOD
MARKS, COLUMNS, INTERVALS, STATUS = base.MARKS, base.COLUMNS, base.INTERVALS, base.STATUS
code_fingerprint, fingerprint = base.code_fingerprint, base.fingerprint
ObservationError = base.ObservationError
PUBLIC_MARKS = ("body_call", "alltoall_c_call", "alltoall_c_return",
                "wait_c_call", "wait_c_return", "body_return")
PUBLIC_INTERVALS = (("python_before_alltoall", 0, 1), ("alltoall_binding", 1, 2),
                    ("python_between_bindings", 2, 3), ("wait_binding", 3, 4),
                    ("python_after_wait", 4, 5))


def _file_pin(path):
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _named_code(parent, name):
    found = [item for item in parent.co_consts if isinstance(item, types.CodeType) and item.co_name == name]
    if len(found) != 1:
        raise ObservationError("one pinned code object required: " + name)
    return found[0]


def callsites(code):
    """Only the exact single CALL following each named LOAD_ATTR is supported."""
    instructions = list(dis.get_instructions(code))
    result = {}
    for target in ("alltoall_base", "wait"):
        found = []
        for index, item in enumerate(instructions):
            if item.opname not in ("LOAD_ATTR", "LOAD_METHOD") or item.argval != target:
                continue
            for following in instructions[index + 1:]:
                if following.opname == "CALL":
                    found.append(following.offset)
                    break
                if following.opname.startswith("RETURN"):
                    break
        if len(found) != 1:
            raise ObservationError("ambiguous public C-call site: " + target)
        result[target] = found[0]
    return result


def expected_public_api():
    """Cold source-derived pins, without importing torch or executing its code."""
    site = Path(sysconfig.get_paths()["purelib"])
    source = site / "torch/distributed/distributed_c10d.py"
    logger = site / "torch/distributed/c10d_logger.py"
    code = compile(source.read_bytes(), str(source), "exec", dont_inherit=True, optimize=sys.flags.optimize)
    body = _named_code(code, "all_to_all_single")
    log_code = compile(logger.read_bytes(), str(logger), "exec", dont_inherit=True, optimize=sys.flags.optimize)
    wrapper = _named_code(_named_code(log_code, "_exception_logger"), "wrapper")
    return {"method": METHOD, "python": sys.version, "cache_tag": sys.implementation.cache_tag,
            "optimize": sys.flags.optimize, "source": _file_pin(source), "logger": _file_pin(logger),
            "torch_version": _file_pin(site / "torch/version.py"),
            "torch_npu_version": _file_pin(site / "torch_npu/version.py"),
            "body_code_sha256": code_fingerprint(body), "wrapper_code_sha256": code_fingerprint(wrapper),
            "callsites": callsites(body), "code_fingerprint_method": CODE_FINGERPRINT_METHOD}


def canonical_c_function(value):
    # Bound group methods use MethodType; the actual unbound Work.wait uses
    # PyInstanceMethod_Type, which also exposes __func__. Never admit by type
    # name or method name: require an actual builtin function, then compare
    # that SAME object against the C-event argument at the pinned call site.
    result = value if type(value) is types.BuiltinFunctionType else getattr(value, "__func__", None)
    if type(result) is not types.BuiltinFunctionType:
        raise ObservationError("expected a pybind/builtin C function; unsupported before submission")
    return result


class PublicCallProfile:
    """Integer-only per-row state. No f_locals, bound Work method, or frame storage."""
    def __init__(self, public, wait_function, rows, pins):
        body = getattr(public, "__wrapped__", None)
        if (type(public) is not types.FunctionType or type(body) is not types.FunctionType
                or code_fingerprint(body.__code__) != pins["body_code_sha256"]
                or code_fingerprint(public.__code__) != pins["wrapper_code_sha256"]
                or callsites(body.__code__) != pins["callsites"]):
            raise ObservationError("loaded public wrapper/body differs from pinned source")
        self.public, self.body, self.code = public, body, body.__code__
        self.wrapper_code = public.__code__
        self.wait_function = canonical_c_function(wait_function)
        self.pins, self.rows = pins, rows
        self.data = array("q", [-1]) * (rows * 6 * 4)
        self.events = array("q", [0]) * rows
        self.matches = array("q", [0]) * rows
        self.restored = array("q", [-1]) * rows
        self.group_ids = array("q", [0]) * rows
        self.frame_ids = array("q", [0]) * rows
        self.submit_ids = array("q", [0]) * rows
        self.status = array("q", [0]) * rows
        self.thread = threading.get_ident()
        self.native_thread_id = threading.get_native_id()
        self.row, self.next = -1, 0
        self.failed = self.active = False
        self.submit_function = None
        self.hook = self.callback

    def _bad(self, message):
        self.failed = True
        if self.row >= 0:
            self.status[self.row] = 1
        raise ObservationError(message)

    def _mark(self, point):
        if point != self.next:
            self._bad("unexpected public API C-event sequence")
        offset = (self.row * 6 + point) * 4
        self.data[offset] = time.perf_counter_ns()
        self.data[offset + 1] = time.thread_time_ns()
        usage = resource.getrusage(resource.RUSAGE_THREAD)
        self.data[offset + 2], self.data[offset + 3] = usage.ru_nvcsw, usage.ru_nivcsw
        self.next += 1
        self.matches[self.row] += 1

    def callback(self, frame, event, arg):
        self.events[self.row] += 1
        if self.failed:
            return  # Original C/Python exception must propagate unchanged.
        if frame.f_code is not self.code:
            return
        if not self.active or threading.get_ident() != self.thread:
            self._bad("public API profile thread/active identity mismatch")
        if event == "call":
            if self.frame_ids[self.row]:
                self._bad("public API body reentered")
            self.frame_ids[self.row] = id(frame)
            self._mark(0)
            return
        if id(frame) != self.frame_ids[self.row]:
            self._bad("public API frame identity changed")
        if event == "return":
            if arg is not None:
                self._bad("sync public API unexpectedly returned Work/value")
            self._mark(5)
            return
        if event not in ("c_call", "c_return", "c_exception"):
            return
        offset = frame.f_lasti
        if offset == self.pins["callsites"]["alltoall_base"]:
            if arg is not self.submit_function:
                self._bad("alltoall pybind callable identity differs")
            point = 1 if event == "c_call" else 2
        elif offset == self.pins["callsites"]["wait"]:
            if arg is not self.wait_function:
                self._bad("wait pybind callable identity differs")
            point = 3 if event == "c_call" else 4
        else:
            return
        if event == "c_exception":
            # Retain original raised exception: mark failure without replacing it.
            self.status[self.row] = 1
            self.failed = True
            return
        self._mark(point)

    def call(self, row, explicit_group, *args, **kwargs):
        if self.failed or self.active or not 0 <= row < self.rows or self.events[row]:
            raise ObservationError("profile already used/failed/active")
        if threading.get_ident() != self.thread or sys.getprofile() is not None:
            raise ObservationError("thread differs or existing profiler; refuse before public API")
        if (self.public.__code__ is not self.wrapper_code or self.public.__wrapped__ is not self.body
                or self.body.__code__ is not self.code
                or kwargs.get("group") is not explicit_group or set(kwargs) != {"group"}):
            raise ObservationError("public API identity/arguments changed")
        submit = canonical_c_function(explicit_group.alltoall_base)
        if getattr(submit, "__self__", None) is explicit_group:
            raise ObservationError("unsupported bound builtin retains group; expected pybind function")
        self.row, self.next, self.active = row, 0, True
        self.submit_function = submit
        self.group_ids[row], self.submit_ids[row] = id(explicit_group), id(submit)
        succeeded = False
        try:
            sys.setprofile(self.hook)
            result = self.public(*args, **kwargs)
            succeeded = True
            return result
        except BaseException:
            self.failed = True
            self.status[row] = 1
            raise
        finally:
            current = sys.getprofile()
            if current is self.hook:
                sys.setprofile(None)
                self.restored[row] = 1
            elif current is None and self.failed:
                # CPython disables a hook if the callback itself raises.
                self.restored[row] = 1
            else:
                self.status[row] = 1
                self.failed = True
                self.restored[row] = 0
            self.active = False
            self.submit_function = None
            if succeeded and (self.failed or self.next != 6 or self.restored[row] != 1):
                self.failed = True
                self.status[row] = 1
                raise ObservationError("incomplete/replaced public C-profile; terminal failure")

    def export(self):
        if self.active:
            raise ObservationError("cannot export active public profile")
        complete = [self.matches[r] == 6 and self.restored[r] == 1 and self.status[r] == 0
                    for r in range(self.rows)]
        return {"schema": PUBLIC_SCHEMA, "method": METHOD, "pins": self.pins,
                "row_count": self.rows, "marks": PUBLIC_MARKS, "columns": COLUMNS,
                "intervals": PUBLIC_INTERVALS, "data": list(self.data),
                "profile_events": list(self.events), "matched_events": list(self.matches),
                "restored_original_none": list(self.restored), "status": list(self.status),
                "group_ids": list(self.group_ids), "body_frame_ids": list(self.frame_ids),
                "submit_callable_ids": list(self.submit_ids), "wait_callable_id": id(self.wait_function),
                "native_thread_id": self.native_thread_id, "complete_rows": complete,
                "all_rows_complete": all(complete) and not self.failed, "terminal_failure": self.failed,
                "identity_scope": "explicit group ID + exact body frame/code/CALL offsets + canonical C functions; Work identity follows pinned work=result;work.wait dataflow, not an observed Work ID",
                "no_f_locals_or_work_tensor_frame_retention": True,
                "interpretation": "host binding spans include profile callback work, GIL release/reacquisition and conversion; not backend execution/network time; sys.setprofile sees all events while active"}


class PreparedCollectiveObserver(base.PreparedCollectiveObserver):
    def __init__(self, collective, rows, *, expected_source_sha256, expected_code_sha256, expected_public_pin):
        super().__init__(collective, rows, expected_source_sha256=expected_source_sha256,
                         expected_code_sha256=expected_code_sha256)
        if expected_public_api() != expected_public_pin:
            raise ObservationError("public API source pin changed")
        self.public_profile = PublicCallProfile(self._dist_original, self._dist.Work.wait,
                                               rows, expected_public_pin)

    def _all_to_all_single(self, *args, **kwargs):
        self._check_thread()
        if (len(args) != 2 or id(args[0]) != self._output_id or id(args[1]) != self._prepared_id
                or set(kwargs) != {"group"} or id(kwargs["group"]) != self._group_id):
            self._invalid("unexpected_boundary", "unexpected all_to_all_single operands")
        self._mark(3)
        result = self.public_profile.call(self._row, kwargs["group"], *args, **kwargs)
        self._mark(4)
        return result

    def export(self):
        result = super().export()
        result["schema"] = SCHEMA
        result["public_dist"] = self.public_profile.export()
        result["all_rows_complete"] = result["all_rows_complete"] and result["public_dist"]["all_rows_complete"]
        result["terminal_failure"] = result["terminal_failure"] or result["public_dist"]["terminal_failure"]
        return result
