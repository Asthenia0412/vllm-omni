"""CPU-only contract tests; no torch import and no device execution."""
import gc
import marshal
from pathlib import Path
import py_compile
import tempfile
import threading
import types
import unittest
import weakref

from observe_qk_collective_host import ObservationError, PreparedCollectiveObserver, code_fingerprint, fingerprint


SOURCE = '''def all_to_all_5D(metadata, *, scatter_idx=3, gather_idx=1, group=None,
                  _prepared_send=None, _qk_padded_owner=None):
    assert (scatter_idx, gather_idx) == (3, 1)
    dist.get_world_size(group)
    torch.guard(metadata, _qk_padded_owner)
    output = torch.empty_like(_prepared_send)
    dist.all_to_all_single(output, _prepared_send, group=group)
    return torch.layout(output)
'''


class Box:
    pass


class ObserverTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        path = Path(self.tmp.name) / "fixture_comm.py"
        path.write_text(SOURCE)
        self.events = []
        self.output = Box()
        # Plain namespace callables have stable identity like module functions.
        self.torch = types.SimpleNamespace(
            guard=lambda metadata, owner: self.events.append(("guard", metadata, owner)),
            empty_like=lambda send: self._allocate(send),
            layout=lambda output: self._layout(output))
        self.dist = types.SimpleNamespace(
            get_world_size=lambda group: self.events.append(("world", group)) or 8,
            all_to_all_single=lambda output, send, **kwargs: self._dist(output, send, kwargs))
        self.ns = {"torch": self.torch, "dist": self.dist}
        exec(compile(SOURCE, str(path), "exec"), self.ns)
        self.function = self.ns["all_to_all_5D"]
        self.send, self.metadata, self.group, self.owner = Box(), Box(), Box(), Box()

    def _allocate(self, send):
        self.events.append(("allocate", send))
        return self.output

    def _layout(self, output):
        self.events.append(("layout", output))
        return output

    def _dist(self, output, send, kwargs):
        self.events.append(("dist", output, send, kwargs))

    def observer(self, rows=1):
        pins = fingerprint(self.function)
        return PreparedCollectiveObserver(self.function, rows,
            expected_source_sha256=pins["source_sha256"], expected_code_sha256=pins["code_sha256"])

    def call(self, observer, row=0, **extra):
        return observer(row, self.metadata, group=self.group, _prepared_send=self.send, **extra)

    def assert_restored(self):
        self.assertIs(self.ns["torch"], self.torch)
        self.assertIs(self.ns["dist"], self.dist)

    def test_exact_forwarding_baseline_and_padded_with_disjoint_intervals(self):
        observer = self.observer(2)
        original_code = self.function.__code__
        original_empty, original_dist = self.torch.empty_like, self.dist.all_to_all_single
        self.assertIs(self.call(observer), self.output)
        self.assertIs(self.call(observer, 1, _qk_padded_owner=self.owner), self.output)
        self.assertEqual([x[0] for x in self.events], ["world", "guard", "allocate", "dist", "layout"] * 2)
        self.assertIs(self.events[1][1], self.metadata)
        self.assertIsNone(self.events[1][2])
        self.assertIs(self.events[6][2], self.owner)
        for event in (self.events[3], self.events[8]):
            self.assertIs(event[1], self.output)
            self.assertIs(event[2], self.send)
            self.assertEqual(event[3], {"group": self.group})
        self.assertIs(self.function.__code__, original_code)
        self.assertIs(self.torch.empty_like, original_empty)
        self.assertIs(self.dist.all_to_all_single, original_dist)
        self.assert_restored()
        report = observer.export()
        self.assertTrue(report["all_rows_complete"])
        for row in range(2):
            stamps = [report["data"][(row * 6 + m) * 4] for m in range(6)]
            self.assertEqual(stamps, sorted(stamps))
            self.assertEqual(sum(stamps[b] - stamps[a] for _, a, b in report["intervals"]), stamps[-1] - stamps[0])

    def test_original_exception_once_and_restoration(self):
        error = ValueError("original failure")
        calls = []
        def fail(*args, **kwargs):
            calls.append(1)
            raise error
        self.dist.all_to_all_single = fail
        observer = self.observer(2)
        with self.assertRaises(ValueError) as caught:
            self.call(observer)
        self.assertIs(caught.exception, error)
        self.assertEqual(calls, [1])
        with self.assertRaisesRegex(ObservationError, "already failed"):
            self.call(observer, 1)
        self.assertEqual(calls, [1])
        self.assert_restored()
        self.assertFalse(observer.export()["all_rows_complete"])

    def test_globals_conflict_is_not_clobbered(self):
        foreign = object()
        self.torch.layout = lambda output: self.ns.__setitem__("torch", foreign) or output
        observer = self.observer()
        with self.assertRaises(ObservationError):
            self.call(observer)
        self.assertIs(self.ns["torch"], foreign)
        self.assertIs(self.ns["dist"], self.dist)
        self.assertEqual(observer.export()["restored_original_bindings"], [0])

    def test_foreign_thread_proxy_rejected_and_invalidates_main_receipt(self):
        errors = []
        def guard(*unused):
            def foreign():
                try:
                    self.ns["torch"].empty_like(self.send)
                except BaseException as error:
                    errors.append(type(error))
            thread = threading.Thread(target=foreign)
            thread.start()
            thread.join()
        self.torch.guard = guard
        observer = self.observer()
        with self.assertRaises(ObservationError):
            self.call(observer)
        self.assertEqual(errors, [ObservationError])
        self.assertEqual(sum(event[0] == "allocate" for event in self.events), 1)
        self.assert_restored()
        self.assertFalse(observer.export()["all_rows_complete"])

    def test_reentry_is_terminal_even_if_original_catches_it(self):
        observer = self.observer()
        errors = []
        def guard(*unused):
            try:
                self.call(observer)
            except ObservationError as error:
                errors.append(type(error))
        self.torch.guard = guard
        with self.assertRaises(ObservationError):
            self.call(observer)
        self.assertEqual(errors, [ObservationError])
        self.assert_restored()

    def test_pin_identity_and_reused_row_rejections_before_call(self):
        pins = fingerprint(self.function)
        for source, code in (("0" * 64, pins["code_sha256"]), (pins["source_sha256"], "0" * 64)):
            with self.assertRaises(ObservationError):
                PreparedCollectiveObserver(self.function, 1, expected_source_sha256=source, expected_code_sha256=code)
        observer = self.observer()
        self.call(observer)
        count = len(self.events)
        with self.assertRaises(ObservationError):
            self.call(observer)
        self.assertEqual(len(self.events), count)

    def test_code_and_public_api_replacement_rejected_before_dispatch(self):
        observer = self.observer()
        original = self.function.__code__
        self.function.__code__ = (lambda: None).__code__
        try:
            with self.assertRaisesRegex(ObservationError, "identity changed"):
                self.call(observer)
        finally:
            self.function.__code__ = original
        self.assertEqual(self.events, [])
        self.assertTrue(observer.export()["terminal_failure"])
        other = self.observer()
        self.dist.all_to_all_single = lambda *args, **kwargs: None
        with self.assertRaisesRegex(ObservationError, "identity changed"):
            self.call(other)
        self.assertEqual(self.events, [])
        other = self.observer()
        self.ns["dist"] = object()
        with self.assertRaises(ObservationError):
            self.call(other)
        self.assertEqual(self.events, [])

    def test_no_activation_retained_by_observer(self):
        observer = self.observer()
        result = self.call(observer, _qk_padded_owner=self.owner)
        refs = [weakref.ref(x) for x in (self.metadata, self.send, self.owner, self.output, self.group)]
        self.events.clear()
        self.metadata = self.send = self.owner = self.output = self.group = None
        del result
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))
        self.assertTrue(observer.export()["all_rows_complete"])

    def test_fresh_compile_and_real_pyc_semantic_fingerprints_match(self):
        source = Path(self.function.__code__.co_filename)
        compiled = Path(self.tmp.name) / "sample.pyc"
        py_compile.compile(str(source), cfile=str(compiled), doraise=True)
        cached_module = marshal.loads(compiled.read_bytes()[16:])
        cached = next(item for item in cached_module.co_consts
                      if isinstance(item, types.CodeType) and item.co_name == "all_to_all_5D")
        self.assertEqual(code_fingerprint(cached), code_fingerprint(self.function.__code__))
        expected = fingerprint(self.function)
        self.function.__code__ = cached
        observer = PreparedCollectiveObserver(self.function, 1,
            expected_source_sha256=expected["source_sha256"], expected_code_sha256=expected["code_sha256"])
        self.assertIs(self.call(observer), self.output)

    def test_fingerprint_ignores_reference_graph_but_preserves_literal_types(self):
        original = self.function.__code__
        shared = "".join(("noninterned", " ", "literal"))
        independent = "".join(("noninterned", " ", "literal"))
        self.assertIsNot(shared, independent)
        a = original.replace(co_consts=(shared, shared))
        b = original.replace(co_consts=(shared, independent))
        self.assertEqual(code_fingerprint(a), code_fingerprint(b))
        for left, right in ((True, 1), (0.0, -0.0), (0.0, 0j), (b"text", "text")):
            self.assertNotEqual(code_fingerprint(original.replace(co_consts=(left,))),
                                code_fingerprint(original.replace(co_consts=(right,))))
        left = original.replace(co_consts=(frozenset((1, "x", b"y")),))
        right = original.replace(co_consts=(frozenset((b"y", "x", 1)),))
        self.assertEqual(code_fingerprint(left), code_fingerprint(right))

    def test_genuine_code_field_mutation_rejected_by_expected_pin(self):
        original = self.function.__code__
        expected = fingerprint(self.function)
        changes = ({"co_consts": original.co_consts + (987654321,)},
                   {"co_firstlineno": original.co_firstlineno + 1},
                   {"co_flags": original.co_flags ^ 0x1000000},
                   {"co_exceptiontable": original.co_exceptiontable + b"\0"},
                   {"co_linetable": original.co_linetable + b"\0"})
        for change in changes:
            self.function.__code__ = original.replace(**change)
            try:
                with self.assertRaisesRegex(ObservationError, "code SHA mismatch"):
                    PreparedCollectiveObserver(self.function, 1,
                        expected_source_sha256=expected["source_sha256"],
                        expected_code_sha256=expected["code_sha256"])
            finally:
                self.function.__code__ = original
        with self.assertRaisesRegex(ObservationError, "unsupported code constant"):
            code_fingerprint(original.replace(co_consts=(object(),)))

    def test_nested_code_mutation_changes_fingerprint(self):
        nested = compile("def outer():\n    def inner():\n        return 1\n    return inner\n",
                         "nested.py", "exec")
        outer = next(item for item in nested.co_consts if isinstance(item, types.CodeType))
        constants = list(outer.co_consts)
        index = next(i for i, item in enumerate(constants) if isinstance(item, types.CodeType))
        inner = constants[index]
        constants[index] = inner.replace(co_consts=tuple(2 if item == 1 else item for item in inner.co_consts))
        self.assertNotEqual(code_fingerprint(outer), code_fingerprint(outer.replace(co_consts=tuple(constants))))


if __name__ == "__main__":
    unittest.main()
