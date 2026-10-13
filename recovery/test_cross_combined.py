"""CPU contracts for the opt-in combined cross host path."""
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]


def load_module():
    spec = importlib.util.spec_from_file_location(
        "combined_cross_recovery_test", ROOT / "vllm_omni/diffusion/layers/lingbot_recovery.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CrossCombinedTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()
        self.provider = object.__new__(self.module.Provider)
        self.provider.policy = (os.environ.get("LINGBOT_RECOVERY_MANIFEST"),
                                os.environ.get("LINGBOT_RECOVERY_MANIFEST_SHA256"))
        self.provider.poisoned = False
        self.provider.fatal = SimpleNamespace(report_failure=Mock())
        self.operation = Mock(return_value=object())
        self.provider.ops = SimpleNamespace(try_cross_combined=self.operation)
        self.kwargs = dict(role="cross", scale=128**-.5, attn_mask=None,
                           dropout_p=0, is_causal=False, enable_gqa=False)

    def test_provider_bound_once_and_tensor_metadata_stays_in_cpp(self):
        with patch.object(self.module, "load", return_value=self.provider) as load:
            adapter = self.module.CrossCombined()
            inputs = (object(), object(), object())
            for _ in range(4):
                self.assertIs(adapter.try_forward(*inputs, **self.kwargs), self.operation.return_value)
            load.assert_called_once_with()
            self.assertEqual(self.operation.call_count, 4)
            self.operation.assert_called_with(*inputs, 128**-.5, True)

    def test_refusal_is_none_without_poisoning(self):
        self.operation.return_value = None
        with patch.object(self.module, "load", return_value=self.provider):
            result = self.module.CrossCombined().try_forward(
                object(), object(), object(), **{**self.kwargs, "role": "self"})
        self.assertIsNone(result)
        self.assertFalse(self.operation.call_args.args[-1])
        self.assertFalse(self.provider.poisoned)
        self.provider.fatal.report_failure.assert_not_called()

    def test_native_error_quarantines_owners_and_forbids_retry(self):
        error = RuntimeError("native failure")
        self.operation.side_effect = error
        inputs = (object(), object(), object())
        with patch.object(self.module, "load", return_value=self.provider):
            adapter = self.module.CrossCombined()
            with self.assertRaisesRegex(RuntimeError, "native failure"):
                adapter.try_forward(*inputs, **self.kwargs)
            with self.assertRaisesRegex(RuntimeError, "poisoned"):
                adapter.try_forward(*inputs, **self.kwargs)
        self.operation.assert_called_once()
        self.provider.fatal.report_failure.assert_called_once()
        self.assertEqual(self.module._QUARANTINE[0][1][:3], inputs)
        self.assertIs(self.module._QUARANTINE[0][2], error)

    def test_policy_change_is_fatal_before_dispatch(self):
        self.provider.policy = ("changed", None)
        with patch.object(self.module, "load", return_value=self.provider):
            with self.assertRaisesRegex(RuntimeError, "manifest changed"):
                self.module.CrossCombined().try_forward(object(), object(), object(), **self.kwargs)
        self.operation.assert_not_called()
        self.assertTrue(self.provider.poisoned)
        self.provider.fatal.report_failure.assert_called_once()
        self.assertIs(self.module._QUARANTINE[0][1], self.provider)

    def config(self, **values):
        config = SimpleNamespace(model_config=values,
            model_class_name="LingBotWorldCausalDMDPipeline", enforce_eager=True,
            parallel_config=SimpleNamespace(tensor_parallel_size=1, sequence_parallel_size=8,
                                            ulysses_degree=8, ring_degree=1))
        return patch.dict("sys.modules", {"vllm_omni.diffusion.config": SimpleNamespace(
            get_current_diffusion_config_or_none=lambda: config)})

    def test_default_off_and_explicit_combined_only(self):
        with self.config():
            self.assertFalse(self.module.configured("cross_combined"))
        with self.config(lingbot_npu_recovery_cross_combined=True):
            self.assertTrue(self.module.configured("cross_combined"))
            self.assertFalse(self.module.configured("cross"))

    def test_conflicting_or_nonboolean_modes_fail_before_loading(self):
        with self.config(lingbot_npu_recovery_cross=True, lingbot_npu_recovery_cross_combined=True):
            with self.assertRaisesRegex(RuntimeError, "mutually exclusive"):
                self.module.configured("cross_combined")
        with self.config(lingbot_npu_recovery_cross_combined="1"):
            with self.assertRaisesRegex(RuntimeError, "must be bool"):
                self.module.configured("cross")
        self.assertIsNone(self.module._BOUND)


if __name__ == "__main__":
    unittest.main()
