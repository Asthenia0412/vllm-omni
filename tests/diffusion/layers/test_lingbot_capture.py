"""Quality artifacts preserve unencoded values and expose incomplete sequences."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

SOURCE = Path(__file__).resolve().parents[3] / "vllm_omni/diffusion/layers/lingbot_capture.py"
spec = importlib.util.spec_from_file_location("capture_under_test", SOURCE)
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)


def test_raw_rgb_and_latents_round_trip_without_changing_inputs(tmp_path):
    writer = capture.DiagnosticCapture(tmp_path, "paired-reference")
    latents = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 2, 2).to(torch.bfloat16)
    rgb = np.arange(36, dtype=np.uint8).reshape(3, 2, 2, 3)
    expected_latents, expected_rgb = latents.clone(), rgb.copy()
    writer.output(0, latents, rgb)
    torch.testing.assert_close(torch.load(writer.directory / "chunk0000.latents.pt", weights_only=True), latents,
                               atol=0, rtol=0)
    np.testing.assert_array_equal(np.load(writer.directory / "chunk0000.rgb.npy"), rgb)
    torch.testing.assert_close(latents, expected_latents, atol=0, rtol=0)
    np.testing.assert_array_equal(rgb, expected_rgb)
    manifest = json.loads((writer.directory / "capture.json").read_text())
    assert manifest["timing_eligible"] is False
    assert manifest["chunks"][0]["latent_dtype"] == "torch.bfloat16"


def test_out_of_order_chunk_is_rejected_without_overwriting(tmp_path):
    writer = capture.DiagnosticCapture(tmp_path, "request")
    writer.output(0, torch.zeros(1), None)
    previous = (writer.directory / "chunk0000.latents.pt").read_bytes()
    with pytest.raises(RuntimeError, match="ordering mismatch"):
        writer.output(0, torch.ones(1), None)
    assert (writer.directory / "chunk0000.latents.pt").read_bytes() == previous


def test_bad_pixel_contract_rejected(tmp_path):
    writer = capture.DiagnosticCapture(tmp_path, "request")
    with pytest.raises(ValueError, match="uint8 FHWC"):
        writer.output(0, torch.zeros(1), np.zeros((1, 2, 2, 3), dtype=np.float32))


def test_session_name_cannot_escape_destination(tmp_path):
    writer = capture.DiagnosticCapture(tmp_path, "../../bad/session")
    assert writer.directory.parent == tmp_path
