"""CPU-only ownership, cancellation and default-OFF tests. No Torch imports."""

import gc
import importlib.util
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "text_residency",
    Path(__file__).resolve().parents[3] / "vllm_omni/diffusion/offloader/lingbot_text_block_residency.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Stream:
    def __init__(self, log):
        self.log, self.fail = log, False

    def synchronize(self):
        self.log.append("drain")
        if self.fail:
            raise RuntimeError("drain failed")


class Runtime:
    def __init__(self):
        self.log = []
        self.stream = Stream(self.log)

    def current_stream(self):
        return self.stream

    def clear_blocks(self, hooks):
        self.log.append("clear")

    def disable(self, encoder, restore_weights):
        assert restore_weights is False
        self.log.append("detach_without_full_reload")

    def publish(self, owner):
        self.log.append("publish")


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        module._QUARANTINE.clear()

    def tearDown(self):
        module._QUARANTINE.clear()

    def owner(self, fn=lambda x: x):
        runtime = Runtime()
        encoder = SimpleNamespace(forward=fn)
        hooks = [SimpleNamespace(copy_stream=None, cpu_master=object()) for _ in range(24)]
        owner = module.TextBlockLifecycle(runtime, encoder, hooks)
        encoder.forward = owner.forward
        return owner, runtime, encoder, hooks

    def test_default_off_no_framework_or_model_access(self):
        self.assertIsNone(
            module.install_lingbot_text_block_residency(SimpleNamespace(model_config={}), None, None, None)
        )

    def test_false_returns_existing_backend_unchanged(self):
        backend = object()
        self.assertIs(
            module.install_lingbot_text_block_residency(
                SimpleNamespace(model_config={"lingbot_npu_text_block_offload": False}), None, None, backend
            ),
            backend,
        )

    def test_invalid_boolean_fails_closed(self):
        with self.assertRaises(ValueError):
            module._enabled("yes")

    def test_same_stream_all_hooks_and_output_identity(self):
        owner, runtime, encoder, hooks = self.owner()
        x = object()
        self.assertIs(encoder.forward(x), x)
        self.assertTrue(all(h.copy_stream is runtime.stream for h in hooks))
        self.assertEqual(runtime.log, ["drain", "clear", "publish"])
        self.assertEqual((owner.encoder_calls, owner.encoder_drains), (1, 1))

    def test_independent_encoder_invocation_selects_new_stream(self):
        owner, runtime, encoder, hooks = self.owner()
        encoder.forward(1)
        runtime.stream = Stream(runtime.log)
        encoder.forward(2)
        self.assertTrue(all(h.copy_stream is runtime.stream for h in hooks))

    def test_forward_exception_drains_then_poison_rejects(self):
        def broken(x):
            raise ValueError("original")

        owner, runtime, encoder, hooks = self.owner(broken)
        with self.assertRaisesRegex(ValueError, "original"):
            encoder.forward(1)
        self.assertEqual(runtime.log[:2], ["drain", "clear"])
        self.assertTrue(owner.poisoned)
        self.assertFalse(owner.active)
        with self.assertRaisesRegex(RuntimeError, "poisoned"):
            encoder.forward(2)
        self.assertEqual(owner.encoder_calls, 1)

    def test_cancellation_is_base_exception_and_retains_masters(self):
        def cancelled(x):
            raise KeyboardInterrupt()

        owner, runtime, encoder, hooks = self.owner(cancelled)
        with self.assertRaises(KeyboardInterrupt):
            encoder.forward(1)
        self.assertIs(owner.hooks[0], hooks[0])
        self.assertTrue(owner.poisoned)

    def test_failed_drain_does_not_clear_or_detach_masters(self):
        owner, runtime, encoder, hooks = self.owner()
        runtime.stream.fail = True
        with self.assertRaisesRegex(RuntimeError, "drain failed"):
            encoder.forward(1)
        self.assertEqual(runtime.log, ["drain"])
        self.assertTrue(owner.active)
        self.assertTrue(owner.poisoned)
        self.assertEqual(len(owner.hooks), 24)
        with self.assertRaises(RuntimeError):
            owner.shutdown()
        self.assertEqual(len(owner.hooks), 24)

    def test_shutdown_retries_failed_fence_then_releases(self):
        owner, runtime, encoder, hooks = self.owner()
        runtime.stream.fail = True
        with self.assertRaises(RuntimeError):
            encoder.forward(1)
        runtime.stream.fail = False
        owner.shutdown()
        self.assertEqual(runtime.log[-4:], ["drain", "clear", "detach_without_full_reload", "publish"])
        self.assertEqual(owner.hooks, ())
        self.assertTrue(owner.closed)

    def test_shutdown_active_nonpoisoned_rejects_detachment(self):
        owner, runtime, encoder, hooks = self.owner()
        owner.before_encode()
        with self.assertRaisesRegex(RuntimeError, "finish"):
            owner.shutdown()
        self.assertEqual(runtime.log, [])
        self.assertEqual(len(owner.hooks), 24)

    def test_shutdown_is_idempotent_and_never_full_reload(self):
        owner, runtime, encoder, hooks = self.owner()
        encoder.forward(1)
        owner.shutdown()
        previous = list(runtime.log)
        owner.shutdown()
        self.assertEqual(runtime.log, previous)
        self.assertEqual(runtime.log.count("detach_without_full_reload"), 1)
        self.assertIs(encoder.forward, owner.original_forward)

    def test_reentrancy_fails_before_second_operation(self):
        owner, runtime, encoder, hooks = self.owner()
        owner.before_encode()
        with self.assertRaisesRegex(RuntimeError, "concurrently"):
            owner.before_encode()
        self.assertEqual(owner.encoder_calls, 1)

    def test_failed_fence_survives_dropped_runner_and_gc(self):
        owner, runtime, encoder, hooks = self.owner()
        runtime.stream.fail = True
        reference = weakref.ref(owner)
        with self.assertRaises(RuntimeError):
            encoder.forward(1)
        with self.assertRaises(RuntimeError):
            owner.shutdown()
        del owner, encoder, hooks, runtime
        gc.collect()
        self.assertIsNotNone(reference())
        self.assertEqual(len(reference().hooks), 24)
        self.assertEqual(len(module._QUARANTINE), 1)

    def test_quarantine_is_idempotent(self):
        owner, runtime, encoder, hooks = self.owner()
        module._quarantine(owner)
        module._quarantine(owner)
        self.assertTrue(owner.poisoned)
        self.assertEqual(module._QUARANTINE, [owner])

    def test_repeated_shutdown_cannot_bypass_running_invocation(self):
        owner, runtime, encoder, hooks = self.owner()
        owner.before_encode()
        with self.assertRaisesRegex(RuntimeError, "finish"):
            owner.shutdown()
        self.assertTrue(owner.poisoned)
        with self.assertRaisesRegex(RuntimeError, "finish"):
            owner.shutdown()
        self.assertEqual(runtime.log, [])
        self.assertEqual(len(owner.hooks), 24)
        self.assertTrue(owner.invocation_running)

    def test_restart_guard_also_rejects_default_off(self):
        module._QUARANTINE.append(object())
        with self.assertRaisesRegex(RuntimeError, "restart"):
            module.install_lingbot_text_block_residency(None, None, None, None)

    def test_restart_guard_also_rejects_capture_only(self):
        module._QUARANTINE.append(object())
        with patch.dict(module.os.environ, {"LINGBOT_TEXT_BLOCK_CAPTURE_DIR": "/mnt/data/capture"}):
            with self.assertRaisesRegex(RuntimeError, "restart"):
                module.install_lingbot_text_block_residency(SimpleNamespace(model_config={}), None, None, None)


if __name__ == "__main__":
    unittest.main()
