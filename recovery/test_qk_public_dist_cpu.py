"""Stdlib CPU contracts only; no torch import/device execution."""
import ast
import builtins
import copy
import functools
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest
import weakref

import observe_qk_public_dist as api
import diagnose_qk_public_dist as wrapper


class Tensor:
    def is_complex(self):
        return False


class Options:
    pass


class Work:
    def __init__(self, fail=False):
        self.wait = types.MethodType(builtins.next, iter([] if fail else [True]))

    def __lt__(self, other):
        return False


def make_public():
    source = Path(api.expected_public_api()["source"]["path"])
    tree = ast.parse(source.read_bytes(), filename=str(source))
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "all_to_all_single")
    fn.decorator_list = []
    namespace = {"has_torch_function": lambda args: False, "_rank_not_in_group": lambda group: False,
                 "AllToAllOptions": Options, "_check_single_tensor": lambda *args: None,
                 "_ensure_all_tensors_same_dtype": lambda *args: None}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), namespace)
    body = namespace["all_to_all_single"]
    @functools.wraps(body)
    def public(*args, **kwargs):
        return body(*args, **kwargs)
    pins = {"body_code_sha256": api.code_fingerprint(body.__code__),
            "wrapper_code_sha256": api.code_fingerprint(public.__code__), "callsites": api.callsites(body.__code__)}
    return public, pins


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.public, self.pins = make_public()

    def observer(self, rows=1):
        return api.PublicCallProfile(self.public, builtins.next, rows, self.pins)

    def group(self, fail=False):
        work = Work(fail)
        group = types.SimpleNamespace(alltoall_base=types.MethodType(builtins.max, work))
        return group, weakref.ref(work)

    def test_exact_body_two_c_calls_and_no_owner_retention(self):
        observer = self.observer()
        group, ref = self.group()
        result = observer.call(0, group, Tensor(), Tensor(), group=group)
        self.assertIsNone(result)
        data = observer.export()
        self.assertTrue(data["all_rows_complete"])
        self.assertEqual(data["matched_events"], [6])
        self.assertEqual(data["restored_original_none"], [1])
        self.assertEqual(data["group_ids"], [id(group)])
        self.assertGreaterEqual(data["profile_events"][0], 6)
        stamps = data["data"][::4]
        self.assertEqual(stamps, sorted(stamps))
        self.assertEqual(sum(stamps[b] - stamps[a] for _, a, b in api.PUBLIC_INTERVALS), stamps[-1] - stamps[0])
        del group
        self.assertIsNone(ref())
        self.assertIsNone(sys.getprofile())
        self.assertNotIn("torch", sys.modules)

    def test_original_exception_propagates_and_hook_restores_without_retry(self):
        observer = self.observer(2)
        group, unused = self.group(True)
        with self.assertRaises(StopIteration):
            observer.call(0, group, Tensor(), Tensor(), group=group)
        self.assertIsNone(sys.getprofile())
        self.assertFalse(observer.export()["all_rows_complete"])
        with self.assertRaisesRegex(api.ObservationError, "failed"):
            observer.call(1, group, Tensor(), Tensor(), group=group)

    def test_existing_profiler_refused_unchanged_before_public_call(self):
        observer = self.observer()
        group, unused = self.group()
        old = lambda *args: None
        sys.setprofile(old)
        try:
            with self.assertRaisesRegex(api.ObservationError, "existing profiler"):
                observer.call(0, group, Tensor(), Tensor(), group=group)
            self.assertIs(sys.getprofile(), old)
            self.assertEqual(observer.events.tolist(), [0])
        finally:
            sys.setprofile(None)

    def test_foreign_thread_refused_without_global_hook_change(self):
        observer = self.observer()
        group, unused = self.group()
        results = []
        def task():
            try:
                observer.call(0, group, Tensor(), Tensor(), group=group)
            except api.ObservationError:
                results.append(sys.getprofile())
        thread = threading.Thread(target=task)
        thread.start()
        thread.join()
        self.assertEqual(results, [None])
        self.assertEqual(observer.events.tolist(), [0])

    def test_callable_mismatch_terminates_and_does_not_mask_original(self):
        observer = api.PublicCallProfile(self.public, builtins.len, 1, self.pins)
        group, unused = self.group()
        with self.assertRaisesRegex(api.ObservationError, "wait pybind callable identity"):
            observer.call(0, group, Tensor(), Tensor(), group=group)
        self.assertIsNone(sys.getprofile())
        self.assertTrue(observer.failed)

    def test_reused_row_and_code_mutation_rejected_before_public_call(self):
        observer = self.observer()
        group, unused = self.group()
        observer.call(0, group, Tensor(), Tensor(), group=group)
        with self.assertRaises(api.ObservationError):
            observer.call(0, group, Tensor(), Tensor(), group=group)
        observer = self.observer()
        original = self.public.__wrapped__.__code__
        self.public.__wrapped__.__code__ = (lambda *args, **kwargs: None).__code__
        try:
            with self.assertRaisesRegex(api.ObservationError, "identity/arguments"):
                observer.call(0, group, Tensor(), Tensor(), group=group)
            self.assertEqual(observer.events.tolist(), [0])
        finally:
            self.public.__wrapped__.__code__ = original

    def test_source_api_pins_are_deterministic_no_torch_import(self):
        self.assertEqual(api.expected_public_api(), api.expected_public_api())
        self.assertNotIn("torch", sys.modules)

    def test_foreign_replacement_is_not_overwritten_and_is_terminal(self):
        observer = self.observer()
        foreign = lambda *args: None
        class ReplacingWork(Work):
            def __lt__(self, other):
                sys.setprofile(foreign)
                return False
        work = ReplacingWork()
        group = types.SimpleNamespace(alltoall_base=types.MethodType(builtins.max, work))
        try:
            with self.assertRaisesRegex(api.ObservationError, "incomplete/replaced"):
                observer.call(0, group, Tensor(), Tensor(), group=group)
            self.assertIs(sys.getprofile(), foreign)
            self.assertEqual(observer.export()["restored_original_none"], [0])
        finally:
            sys.setprofile(None)

    def test_group_identity_mismatch_rejected_before_call(self):
        observer = self.observer()
        group, unused = self.group()
        other, unused2 = self.group()
        with self.assertRaisesRegex(api.ObservationError, "identity/arguments"):
            observer.call(0, group, Tensor(), Tensor(), group=other)
        self.assertEqual(observer.events.tolist(), [0])


class WrapperTests(unittest.TestCase):
    def test_frozen_call_bodies_and_private_constructor_only(self):
        for name in ("run_arm", "instrumented_correctness"):
            original, adapted = getattr(wrapper.frozen, name), getattr(wrapper, name)
            self.assertIs(original.__code__, adapted.__code__)
            different = {key for key in original.__globals__ if original.__globals__[key] is not adapted.__globals__[key]}
            self.assertEqual(different, {"InstrumentedCall"})
        self.assertIs(wrapper.InstrumentedCall.__call__, wrapper.frozen.InstrumentedCall.__call__)

    def test_rows_and_old_v2_rejected(self):
        trace = wrapper.BoundaryIntegerTrace("correctness")
        indices = []
        for case in range(24):
            trace.case_index = case
            for call in range(4):
                trace.row = call
                indices.append(trace.boundary_row())
        self.assertEqual(indices, list(range(96)))
        self.assertEqual(wrapper.BoundaryIntegerTrace("profile").rows, 5760)
        import json
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "rank0.json").write_text(json.dumps({
                "schema": "lingbot_qk_collective_boundaries_diagnostic_v2", "passed": True}))
            with self.assertRaisesRegex(RuntimeError, "new diagnostic correctness binding differs"):
                wrapper.verify_diagnostic_correctness(directory, {"sha256": "mock"}, {}, {}, {}, {})

    def test_public_validator_requires_fresh_six_marks_inside_same_thread_outer(self):
        public, pins = make_public()
        obs = api.PublicCallProfile(public, builtins.next, 1, pins)
        work = Work()
        group = types.SimpleNamespace(alltoall_base=types.MethodType(builtins.max, work))
        obs.call(0, group, Tensor(), Tensor(), group=group)
        import json
        data = json.loads(json.dumps(obs.export()))
        outer = {"native_thread_id": data["native_thread_id"], "data": [0] * 24}
        outer["data"][12:16] = data["data"][0:4]
        outer["data"][16:20] = data["data"][20:24]
        wrapper.validate_public_dist(data, rows=1, pins=pins, outer=outer)
        for key, value in (("schema", "old-v2"), ("matched_events", [5]),
                           ("restored_original_none", [0]), ("native_thread_id", -1)):
            bad = copy.deepcopy(data)
            bad[key] = value
            with self.assertRaises(RuntimeError):
                wrapper.validate_public_dist(bad, rows=1, pins=pins, outer=outer)
        outer["data"][16] -= 1
        with self.assertRaisesRegex(RuntimeError, "outside"):
            wrapper.validate_public_dist(data, rows=1, pins=pins, outer=outer)


if __name__ == "__main__":
    unittest.main()
