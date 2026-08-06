from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "paper_idea1b" / "scripts" / "prepare_gate_a0.py"
SPEC = importlib.util.spec_from_file_location("prepare_gate_a0", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_kodak_s33_rate_is_one_over_24() -> None:
    config = yaml.safe_load(
        (ROOT / "paper_idea1b/configs/gate_a0_benchmark_setup.yaml").read_text()
    )
    source = {
        "dataset": "kodak",
        "sample_id": "kodak/kodim01.png",
        "width": 768,
        "height": 512,
        "source_real_dimensions": 3 * 768 * 512,
    }
    rate_rows, tile_rows, sgd_rows = MODULE.build_processing_manifests(
        [source], config
    )
    s33 = next(row for row in rate_rows if row["method"] == "s33_strong")
    assert len(tile_rows) == 6
    assert s33["main_real_symbols"] == 98_304
    assert s33["actual_complex_channel_uses"] == 49_152
    assert abs(s33["actual_cbr"] - 1 / 24) < 1e-15
    assert len(sgd_rows) == 24


def test_diffjscc_kodak_keeps_whole_frame() -> None:
    assert MODULE.diffjscc_internal_size(768, 512, 512, 64) == (
        768,
        512,
        768,
        512,
    )


def test_sgd_positions_match_released_divisible_case() -> None:
    positions = MODULE.sgd_positions(512, 768, 128)
    assert len(positions) == 24
    assert len(set(positions)) == 24
    assert positions[0] == (0, 0)
    assert positions[-1] == (384, 640)
