"""CPU checks for restaging, full-region scope and diagnostic-only selection."""
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "vllm_omni/diffusion/layers/lingbot_fixed_modulation_screen.py"
spec = importlib.util.spec_from_file_location("stageB_screen_cpu", SOURCE)
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)


class ScreenTests(unittest.TestCase):
    def test_disabled_configuration_needs_no_runtime(self):
        pipeline = NS(transformer=NS())
        with patch.dict(os.environ, {}, clear=True):
            s.configure(pipeline, {})
        self.assertIsNone(pipeline.transformer._lingbot_stageB_operator_screen)

    def test_rejects_timing_service_before_device_work(self):
        with patch.dict(os.environ, {"LINGBOT_STAGEB_OPERATOR_SCREEN_DIR": "/not-used",
                                    "ROOT_LINGBOT_SP8_EXPERIMENT_MODE": "timing"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "diagnostic service"):
                s.configure(NS(transformer=NS()), {"lingbot_npu_fixed_modulation_stageB": True})

    def test_rejects_unrelated_metadata_combination(self):
        with patch.dict(os.environ, {"LINGBOT_STAGEB_OPERATOR_SCREEN_DIR": "/not-used",
                                    "ROOT_LINGBOT_SP8_EXPERIMENT_MODE": "profile",
                                    "VLLM_LINGBOT_NPU_ASYNC_METADATA": "1"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "metadata remains OFF"):
                s.configure(NS(transformer=NS()), {"lingbot_npu_fixed_modulation_stageB": True})

    def fixture(self):
        import torch
        generator = torch.Generator().manual_seed(19)
        values = torch.randn(1,1,6,5120, generator=generator).to(torch.bfloat16)
        original = torch.full_like(values, float("nan"))
        blocks = [NS(modulation=torch.randn(1,6,5120, generator=generator).to(torch.bfloat16)) for _ in range(40)]
        transformer = NS(blocks=blocks)
        args = (transformer, object(), object(), object(), original, True)
        return torch, transformer, dict(args=args, snapshot=values)

    def test_native_region_restages_then_preserves_all40_fp32_expressions(self):
        torch, transformer, case = self.fixture()
        result = s.Screen.invoke("native", transformer, None, case)
        self.assertEqual(len(result), 40)
        self.assertTrue(torch.equal(case["args"][4], case["snapshot"]))
        for block, row in zip(transformer.blocks, result):
            expected = (block.modulation.unsqueeze(1) + case["snapshot"].float()).chunk(6, dim=2)
            self.assertEqual(len(row), 6)
            for a, b in zip(row, expected):
                self.assertEqual(a.dtype, torch.float32)
                self.assertTrue(torch.equal(a.view(torch.int32), b.view(torch.int32)))

    def test_candidate_scope_includes_actual_prepare_and40_ordered_lookups(self):
        torch, transformer, case = self.fixture()
        expected = s.Screen.invoke("native", transformer, None, case)
        seen = []
        forward = NS(get=lambda index, block, projection: seen.append((index, block, projection)) or expected[index])
        observed = []
        core = NS(prepare_forward=lambda *args: observed.append(args) or forward,
                  bank_published=lambda value: value is forward)
        case["args"][4].zero_()
        result = s.Screen.invoke("stageB", transformer, core, case)
        self.assertEqual(observed, [case["args"]])
        self.assertEqual([x[0] for x in seen], list(range(40)))
        self.assertTrue(all(x[1] is transformer.blocks[i] and x[2] is case["args"][4] for i,x in enumerate(seen)))
        self.assertTrue(torch.equal(case["args"][4], case["snapshot"]))
        self.assertEqual(tuple(id(x) for x in result), tuple(id(x) for x in expected))

    def test_unpublished_bank_cannot_be_reported_as_cached_timing(self):
        _, transformer, case = self.fixture()
        core = NS(prepare_forward=lambda *args: NS(), bank_published=lambda value: False)
        with self.assertRaisesRegex(RuntimeError, "warm actual StageB bank refused"):
            s.Screen.invoke("stageB", transformer, core, case)


if __name__ == "__main__":
    unittest.main()
