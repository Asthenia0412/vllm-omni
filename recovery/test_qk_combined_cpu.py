"""No accelerator operations: refusal and post-start shared poison contracts."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
NAME = "vllm_omni.diffusion.layers.lingbot_qk_combined"
spec = importlib.util.spec_from_file_location(NAME, ROOT / "vllm_omni/diffusion/layers/lingbot_qk_combined.py")
qk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qk)


class CombinedContracts(unittest.TestCase):
    def test_snapshot_miss_refuses_without_producer_or_dispatch(self):
        from vllm_omni.diffusion.layers.lingbot_native_bf16_qk import SharedCoefficients
        shared = SharedCoefficients()
        values = {qk.CONFIG_KEY: True}
        module = SimpleNamespace(ulysses_world_size=8, num_local_heads=40, head_dim=128,
            norm_q=SimpleNamespace(eps=1e-6), norm_k=SimpleNamespace(eps=1e-6),
            _lingbot_native_bf16_qk_coefficients=shared)
        route = qk.Route(module, SimpleNamespace(model_config=values), values)
        route._provider = SimpleNamespace(check_policy=lambda: None)
        self.assertIsNone(route.prepare(SimpleNamespace(shape=(1,585,15360)), (object(),object()), 1))
        self.assertEqual(shared.generation, 0)
        self.assertIsNone(shared.expanded_cosine)
        self.assertIsNone(shared.expanded_sine)

    def test_consumer_failure_poison_is_shared_across_new_leases(self):
        reports = []
        base = SimpleNamespace(poisoned=False, torch=None, backend=None,
            fatal=SimpleNamespace(report_failure=lambda *args: reports.append(args)))
        provider = qk.Provider(base, None, {}, ())
        def lease():
            result = object.__new__(qk.Lease)
            result.provider = provider
            result.used = result.entering = False
            result.metadata = result.send = result.owner = object()
            return result
        first, second = lease(), lease()
        def fail(*args, **kwargs):
            raise RuntimeError("injected started consumer failure")
        with self.assertRaisesRegex(RuntimeError, "injected started"):
            first.exchange(fail, object())
        self.assertTrue(base.poisoned)
        self.assertFalse(first.entering)
        self.assertEqual(len(reports), 1)
        with self.assertRaisesRegex(RuntimeError, "provider poisoned"):
            second.exchange(fail, object())
        self.assertEqual(len(reports), 2)

    def test_invalid_lease_checks_are_not_python_asserts(self):
        with self.assertRaisesRegex(RuntimeError, "registered QK lease"):
            qk.validate_padded_send(object(), object(), object(), 8)
        tree = __import__("ast").parse((ROOT / "vllm_omni/diffusion/layers/lingbot_qk_combined.py").read_text())
        self.assertFalse(any(isinstance(n, __import__("ast").Assert) for n in __import__("ast").walk(tree)))


if __name__ == "__main__":
    unittest.main()
