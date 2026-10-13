"""CPU contract tests for the isolated proposal; never imports torch or uses NPU.

Run after timing closes with QK_RECORD_ONCE_SOURCE set to the proposed/new-worktree
module. Fake tensors exercise only consumer registration/exception ownership;
they provide no numerical, allocator or real192 device admission.
"""
import importlib.util
import os
from pathlib import Path
import unittest
from types import SimpleNamespace


class Tensor:
    def __init__(self, label, calls, error=None):
        self.label, self.calls, self.error = label, calls, error

    def record_stream(self, stream):
        self.calls.append((self.label, stream))
        if self.error:
            raise self.error


class RecordOnceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(os.environ["QK_RECORD_ONCE_SOURCE"]).resolve(strict=True)
        spec = importlib.util.spec_from_file_location("qk_record_once_cpu_subject", path)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def lease(self, result_error=None):
        calls, failures, stream = [], [], object()
        base = SimpleNamespace(poisoned=False)
        def fail(error, owners):
            base.poisoned = True
            failures.append((error, owners))
        provider = SimpleNamespace(base=base, fail=fail)
        lease = object.__new__(self.module.Lease)
        lease.provider, lease.stream = provider, stream
        lease.used = lease.entering = False
        lease.args = tuple(Tensor(name, calls) for name in ("raw", "qw", "kw", "cos", "sin"))
        lease.metadata = Tensor("metadata", calls)
        lease.send, lease.owner = Tensor("send", calls), Tensor("owner", calls)
        lease.owners = (*lease.args, lease.metadata, lease.send, lease.owner)
        result = Tensor("result", calls, result_error)
        def collective(metadata, *, scatter_idx, gather_idx, group, _prepared_send, _qk_padded_owner):
            self.assertIs(metadata, lease.metadata)
            self.assertIs(_prepared_send, lease.send)
            self.assertIs(_qk_padded_owner, lease)
            self.assertEqual((scatter_idx, gather_idx), (3, 1))
            self.assertTrue(lease.entering)
            lease.used = True  # Models successful unchanged consumer admission.
            return result
        return lease, collective, result, calls, failures

    def test_input_and_target_owners_remain_but_only_result_is_registered(self):
        lease, collective, result, calls, failures = self.lease()
        owners_before = lease.owners
        self.assertIs(lease.exchange(collective, "group"), result)
        self.assertEqual(calls, [("result", lease.stream)])
        self.assertIs(lease.owners, owners_before)
        self.assertTrue(lease.used)
        self.assertFalse(lease.entering)
        self.assertEqual(failures, [])

    def test_result_registration_failure_still_poison_retains_lease(self):
        error = RuntimeError("result registration failed")
        lease, collective, _, calls, failures = self.lease(error)
        with self.assertRaisesRegex(RuntimeError, "result registration failed"):
            lease.exchange(collective, "group")
        self.assertEqual(calls, [("result", lease.stream)])
        self.assertTrue(lease.provider.base.poisoned)
        self.assertIs(failures[0][0], error)
        self.assertIs(failures[0][1][0], lease)
        self.assertFalse(lease.entering)

    def test_reuse_fails_before_collective_and_retains_original_lease(self):
        lease, collective, _, calls, failures = self.lease()
        lease.exchange(collective, "group")
        with self.assertRaisesRegex(RuntimeError, "lease reused"):
            lease.exchange(collective, "group")
        self.assertEqual(calls, [("result", lease.stream)])
        self.assertTrue(lease.provider.base.poisoned)
        self.assertIs(failures[0][1][0], lease)


if __name__ == "__main__":
    unittest.main()
