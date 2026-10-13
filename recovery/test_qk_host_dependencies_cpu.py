"""CPU-only lifecycle/observer contracts, not numerical or device evidence."""
import ast
import gc
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest
import weakref

import diagnose_qk_host_dependencies as diagnostic


class Dying:
    def __init__(self, kind, serial, deaths):
        self.kind, self.serial, self.deaths = kind, serial, deaths

    def __del__(self):
        self.deaths.append((self.kind, self.serial, time.perf_counter_ns()))


class Lease(Dying):
    def __init__(self, serial, deaths, arm):
        super().__init__("lease", serial, deaths)
        self.owner = Dying("owner", serial, deaths)
        self.metadata, self.send = object(), object()
        self.arm = arm

    def exchange(self, collective, group):
        if self.arm == "combined":
            return collective(self.metadata, scatter_idx=3, gather_idx=1, group=group,
                              _prepared_send=self.send, _qk_padded_owner=self)
        return collective(self.metadata, scatter_idx=3, gather_idx=1, group=group,
                          _prepared_send=self.send)


class Replay:
    def __init__(self, fail_on=None):
        self.deaths, self.calls, self.syncs, self.prepared = [], [], [], []
        self.serial, self.fail_on = 0, fail_on
        self.group = object()
        self.original = (object(), object())
        self.shared = SimpleNamespace(generation=0)
        def prepare(*args):
            if args != self.original:
                raise RuntimeError("producer arguments changed")
            self.shared.generation += 1
        self.shared.prepare = prepare
        self.torch = SimpleNamespace(npu=SimpleNamespace(synchronize=lambda: self.syncs.append(time.perf_counter_ns())))

    def prepare(self, arm):
        serial = self.serial
        self.serial += 1
        lease = Lease(serial, self.deaths, arm)
        self.prepared.append(weakref.ref(lease))
        return lease

    def collective(self, metadata, *, scatter_idx, gather_idx, group, _prepared_send, **kwargs):
        if (scatter_idx, gather_idx) != (3, 1) or group is not self.group:
            raise RuntimeError("collective order/group changed")
        serial = self.serial - 1
        self.calls.append((serial, tuple(kwargs)))
        if serial == self.fail_on:
            raise RuntimeError("injected collective failure")
        return Dying("result", serial, self.deaths)

    def call(self, arm, producer):
        if producer:
            self.shared.prepare(*self.original)
        lease = self.prepare(arm)
        return lease, lease.exchange(self.collective, self.group)


class DiagnosticContractTest(unittest.TestCase):
    def tearDown(self):
        diagnostic._QUARANTINE.clear()

    def test_import_and_guards_are_cpu_only_under_optimized_python(self):
        self.assertNotIn("torch", sys.modules)
        with self.assertRaisesRegex(RuntimeError, "deliberately false"):
            diagnostic.need(False, "deliberately false")
        tree = ast.parse(Path(diagnostic.__file__).read_text())
        self.assertFalse(any(isinstance(item, ast.Assert) for item in ast.walk(tree)))
        forbidden = ("disable", "enable", "collect", "set_threshold", "set_debug")
        self.assertFalse(any(isinstance(item, ast.Call) and isinstance(item.func, ast.Attribute)
                             and isinstance(item.func.value, ast.Name) and item.func.value.id == "gc"
                             and item.func.attr in forbidden for item in ast.walk(tree)))

    def test_old_lease_owner_result_die_inside_replacement_span(self):
        replay = Replay()
        trace = diagnostic.IntegerTrace(3)
        phases = []
        result = diagnostic.run_arm(replay, "combined", True, trace, 0, phases.append, calls=3, warmup=2)
        self.assertEqual(phases, ["warm20", "diagnostic_calls"])
        self.assertEqual(len(replay.syncs), 6)  # No per-stage NPU sync.
        self.assertEqual(len(replay.calls), 5)
        self.assertEqual(list(trace.alive), [0] * 6)
        self.assertEqual(result["generation_end"], 6)  # initial + warm2 + sample3.
        for row in range(3):
            start = trace.data[(row * len(diagnostic.MARKS) + diagnostic.BEFORE_REPLACE) * 4]
            end = trace.data[(row * len(diagnostic.MARKS) + diagnostic.AFTER_REPLACE) * 4]
            old_serial = row + 1  # Last warm output, then previous diagnostics.
            deaths = [item for item in replay.deaths if item[1] == old_serial]
            self.assertEqual({item[0] for item in deaths}, {"lease", "owner", "result"})
            self.assertTrue(all(start <= item[2] <= end for item in deaths))
        final_sync_end = trace.data[((2 * len(diagnostic.MARKS)) + diagnostic.POST_END) * 4]
        self.assertTrue(all(item[2] >= final_sync_end for item in replay.deaths if item[1] == 4))
        self.assertTrue(all(ref() is None for ref in replay.prepared))
        self.assertEqual(diagnostic._QUARANTINE, [])

    def test_both_arms_keep_exact_collective_arguments_and_no_proxy_owners(self):
        for arm in diagnostic.ARMS:
            replay = Replay()
            trace = diagnostic.IntegerTrace(1)
            result = diagnostic.run_arm(replay, arm, False, trace, 0, lambda _: None, calls=1, warmup=1)
            self.assertEqual(result["generation_start"], result["generation_end"])
            expected = ("_qk_padded_owner",) if arm == "combined" else ()
            self.assertEqual(replay.calls, [(0, expected), (1, expected)])
            for point in (diagnostic.PRODUCER_BEGIN, diagnostic.PRODUCER_END):
                self.assertEqual(trace.data[point * 4], -1)
            self.assertTrue(all(ref() is None for ref in replay.prepared))

    def test_failure_retains_previous_and_pending_owners_without_fallback(self):
        replay = Replay(fail_on=1)
        trace = diagnostic.IntegerTrace(2)
        with self.assertRaisesRegex(RuntimeError, "injected collective failure"):
            diagnostic.run_arm(replay, "combined", False, trace, 0, lambda _: None, calls=2, warmup=1)
        self.assertEqual(len(replay.calls), 2)
        self.assertEqual(len(replay.syncs), 1)
        self.assertTrue(all(ref() is not None for ref in replay.prepared))
        self.assertGreaterEqual(len(diagnostic._QUARANTINE), 2)

    def test_gc_callback_restores_only_its_own_identity_even_on_failure(self):
        before_enabled, before_thresholds = gc.isenabled(), gc.get_threshold()
        first = lambda *_: None
        later = lambda *_: None
        gc.callbacks.append(first)
        recorder = diagnostic.GCTrace(8)
        try:
            with self.assertRaisesRegex(RuntimeError, "stop"):
                with recorder:
                    gc.callbacks.remove(first)
                    gc.callbacks.append(later)
                    self.assertTrue(any(item is recorder.callback for item in gc.callbacks))
                    raise RuntimeError("stop")
            self.assertFalse(any(item is recorder.callback for item in gc.callbacks))
            self.assertFalse(any(item is first for item in gc.callbacks))
            self.assertTrue(any(item is later for item in gc.callbacks))
            self.assertEqual(gc.isenabled(), before_enabled)
            self.assertEqual(gc.get_threshold(), before_thresholds)
        finally:
            for callback in (first, later):
                for index in range(len(gc.callbacks) - 1, -1, -1):
                    if gc.callbacks[index] is callback:
                        del gc.callbacks[index]

    def test_gc_is_bounded_thread_attributed_and_marks_incomplete(self):
        recorder = diagnostic.GCTrace(2)
        recorder.callback("start", {"generation": 1})
        recorder.callback("stop", {"generation": 1})
        recorder.callback("start", {"generation": 0})
        report = recorder.export()
        self.assertEqual((report["count"], report["dropped"], report["errors"]), (2, 1, 0))
        self.assertFalse(report["complete"])
        self.assertEqual(report["data"][1], diagnostic.threading.get_native_id())
        self.assertEqual(report["data"][2::4], [0, 1])
        broken = diagnostic.GCTrace(2)
        broken.callback("start", {"generation": object()})
        self.assertEqual(broken.errors, 1)
        self.assertFalse(broken.export()["complete"])

    def test_profile_never_has_per_sample_rank_barrier_or_stage_sync(self):
        tree = ast.parse(Path(diagnostic.__file__).read_text())
        run = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == "run_arm")
        sample = next(item for item in ast.walk(run) if isinstance(item, ast.For) and ast.unparse(item.iter) == "range(calls)")
        sync = [item for item in ast.walk(sample) if isinstance(item, ast.Call)
                and isinstance(item.func, ast.Attribute) and item.func.attr == "synchronize"]
        self.assertEqual(len(sync), 2)
        self.assertFalse(any(isinstance(item, ast.Call) and isinstance(item.func, ast.Attribute)
                             and item.func.attr == "barrier" for item in ast.walk(sample)))
        for cls_name in ("InstrumentedCall", "CollectiveProbe"):
            cls = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == cls_name)
            self.assertFalse(any(isinstance(item, ast.Call) and isinstance(item.func, ast.Attribute)
                                 and item.func.attr in ("synchronize", "barrier") for item in ast.walk(cls)))


if __name__ == "__main__":
    unittest.main()
