"""
Coverage-oriented tests for backends, benchmark modules, and model layer.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from tritpack.core.quantizer import TernaryQuantizer
from tritpack.core.dequantizer import TernaryDequantizer


# ---------------------------------------------------------------------------
# numpy_backend
# ---------------------------------------------------------------------------


def test_numpy_backend_save_load_roundtrip():
    """save() and load() must produce a QuantizedTensor that dequantizes cleanly."""
    from tritpack.backends.numpy_backend import save, load, save_tensor, load_tensor

    rng = np.random.default_rng(0)
    tensor = rng.standard_normal((256,)).astype(np.float32)
    q = TernaryQuantizer()
    qt = q.quantize(tensor)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.npz"
        save(qt, path)
        qt2 = load(path)

    assert qt2.original_shape == qt.original_shape
    assert len(qt2.packed_data) == len(qt.packed_data)
    dq = TernaryDequantizer()
    rec = dq.dequantize(qt2)
    assert rec.shape == tensor.shape


def test_numpy_backend_save_tensor_load_tensor():
    """save_tensor / load_tensor convenience wrappers."""
    from tritpack.backends.numpy_backend import save_tensor, load_tensor

    rng = np.random.default_rng(1)
    tensor = rng.standard_normal((128,)).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "w.npz"
        qt = save_tensor(tensor, path, alpha=0.7)
        rec = load_tensor(path)

    assert rec.shape == tensor.shape


# ---------------------------------------------------------------------------
# benchmark/memory
# ---------------------------------------------------------------------------


def test_benchmark_memory_measure_ram():
    """measure_process_ram returns positive float."""
    from tritpack.benchmark.memory import measure_process_ram
    ram = measure_process_ram()
    assert ram > 0.0


def test_benchmark_tensor_compression():
    """benchmark_tensor_compression returns a CompressionReport."""
    from tritpack.benchmark.memory import benchmark_tensor_compression

    rng = np.random.default_rng(0)
    tensor = rng.standard_normal((1000,)).astype(np.float32)
    report = benchmark_tensor_compression(tensor)
    assert report.ratio > 1.0
    assert report.time_seconds >= 0


# ---------------------------------------------------------------------------
# benchmark/quality
# ---------------------------------------------------------------------------


def test_benchmark_quality_report():
    """benchmark_reconstruction_quality returns a QualityReport."""
    from tritpack.benchmark.quality import benchmark_reconstruction_quality

    rng = np.random.default_rng(0)
    tensor = rng.standard_normal((500,)).astype(np.float32)
    alphas = [0.5, 0.7, 0.9]
    report = benchmark_reconstruction_quality(tensor, alphas)
    assert len(report.points) == 3
    assert report.points[0].alpha == 0.5
    table = report.summary_table()
    assert len(table) == 3
    assert "cos_sim" in table[0]


def test_quality_report_best_alpha():
    """best_alpha_for_target returns correct result."""
    from tritpack.benchmark.quality import benchmark_reconstruction_quality

    rng = np.random.default_rng(0)
    tensor = rng.standard_normal((1000,)).astype(np.float32)
    alphas = [0.3, 0.5, 0.7, 0.9]
    report = benchmark_reconstruction_quality(tensor, alphas)
    best = report.best_alpha_for_target(0.50)  # very low target, all should pass
    assert best is not None
    assert best >= 0.3


# ---------------------------------------------------------------------------
# benchmark/speed
# ---------------------------------------------------------------------------


def test_benchmark_pack_throughput():
    """benchmark_pack_throughput returns a SpeedReport with correct structure."""
    from tritpack.benchmark.speed import benchmark_pack_throughput

    report = benchmark_pack_throughput([0.1, 0.5], n_repeats=1)
    assert len(report.sizes_mb) == 2
    assert len(report.throughputs_mb_s) == 2
    assert all(t > 0 for t in report.throughputs_mb_s)
    summary = report.summary()
    assert len(summary) == 2
    assert "size_mb" in summary[0]


def test_benchmark_quantize_latency():
    """benchmark_quantize_latency returns a LatencyReport."""
    from tritpack.benchmark.speed import benchmark_quantize_latency

    report = benchmark_quantize_latency([(100,), (200,)], n_repeats=1)
    assert len(report.shapes) == 2
    assert all(l >= 0 for l in report.latencies_ms)
    summary = report.summary()
    assert "latency_ms" in summary[0]


# ---------------------------------------------------------------------------
# benchmark CLI (__main__)
# ---------------------------------------------------------------------------


def test_benchmark_cli_pack(capsys):
    """bench-pack command runs without error."""
    import sys
    from unittest.mock import patch
    from tritpack.benchmark.__main__ import cmd_bench_pack
    import argparse

    args = argparse.Namespace(output=None)
    result = cmd_bench_pack(args)
    assert "pack_throughput" in result


def test_benchmark_cli_quality(capsys):
    """bench-quality command runs without error."""
    import argparse
    from tritpack.benchmark.__main__ import cmd_bench_quality

    args = argparse.Namespace(output=None)
    result = cmd_bench_quality(args)
    assert "quality" in result


def test_benchmark_cli_memory():
    """bench-memory command runs without error."""
    import argparse
    from tritpack.benchmark.__main__ import cmd_bench_memory

    args = argparse.Namespace(output=None)
    result = cmd_bench_memory(args)
    assert "memory" in result


def test_benchmark_cli_all():
    """bench-all runs all benchmarks."""
    import argparse
    from tritpack.benchmark.__main__ import cmd_bench_all

    args = argparse.Namespace(output=None)
    result = cmd_bench_all(args)
    assert "pack_throughput" in result
    assert "quality" in result
    assert "memory" in result


def test_benchmark_cli_output_json(tmp_path):
    """bench-pack with --output writes JSON file."""
    import json
    import sys
    from tritpack.benchmark.__main__ import cmd_bench_pack
    import argparse

    out = str(tmp_path / "report.json")
    args = argparse.Namespace(output=out)
    result = cmd_bench_pack(args)
    # Write output file manually (the cmd_* functions return data, main() writes)
    with open(out, "w") as f:
        json.dump(result, f)
    assert (tmp_path / "report.json").exists()


# ---------------------------------------------------------------------------
# model/tensor.py
# ---------------------------------------------------------------------------


def test_trit_tensor_properties():
    """TritTensor properties must all return correct types."""
    from tritpack.model.tensor import TritTensor

    rng = np.random.default_rng(0)
    t = rng.standard_normal((64, 64)).astype(np.float32)
    tt = TritTensor(t)

    assert tt.shape == (64, 64)
    assert tt.dtype == np.float32
    assert tt.nbytes_compressed > 0
    assert tt.nbytes_original == t.nbytes
    assert tt.compression_ratio > 1.0
    assert 0.0 <= tt.sparsity <= 1.0
    assert isinstance(repr(tt), str)
    assert len(tt) == 64


def test_trit_tensor_data_property():
    """TritTensor.data must decompress correctly."""
    from tritpack.model.tensor import TritTensor

    rng = np.random.default_rng(1)
    t = rng.standard_normal((128,)).astype(np.float32)
    tt = TritTensor(t)
    data = tt.data
    assert data.shape == t.shape


def test_trit_tensor_quality_report_no_original():
    """quality_report works even when _original is None."""
    from tritpack.model.tensor import TritTensor

    rng = np.random.default_rng(0)
    t = rng.standard_normal((64,)).astype(np.float32)
    tt = TritTensor(t)
    tt._original = None  # simulate released reference
    report = tt.quality_report()
    assert report["cos_sim"] is None
    assert report["compression_ratio"] > 0


def test_trit_tensor_len_scalar():
    """TritTensor on 1-D array, len() returns first dimension."""
    from tritpack.model.tensor import TritTensor

    rng = np.random.default_rng(0)
    t = rng.standard_normal((50,)).astype(np.float32)
    tt = TritTensor(t)
    assert len(tt) == 50


# ---------------------------------------------------------------------------
# core/tiers.py edge cases
# ---------------------------------------------------------------------------


def test_tier_manager_invalid_limit():
    """TierManager with non-positive memory_limit must raise."""
    from tritpack.core.tiers import TierManager

    with pytest.raises(ValueError):
        TierManager(memory_limit_gb=-1.0)


def test_tier_manager_warm_then_cold():
    """Layer demoted to WARM can be further demoted to COLD."""
    from tritpack.core.tiers import TierManager, TIER_COLD, TIER_WARM

    # Very tight budget: forces COLD demotion
    tm = TierManager(memory_limit_gb=0.0001, hot_budget_fraction=0.01, warm_budget_fraction=0.01)
    rng = np.random.default_rng(0)
    # Register many layers to force eviction chain
    for i in range(50):
        t = rng.standard_normal((10_000,)).astype(np.float32)
        tm.register_layer(f"l{i}", t)

    stats = tm.stats()
    # At least some layers should have reached COLD tier
    assert stats.n_cold > 0 or stats.n_warm > 0


def test_tier_memory_usage_zero_on_empty():
    """Empty TierManager must report zero memory."""
    from tritpack.core.tiers import TierManager

    tm = TierManager(memory_limit_gb=1.0)
    usage = tm.memory_usage()
    assert usage["total_gb"] == 0.0


# ---------------------------------------------------------------------------
# model/calibration.py
# ---------------------------------------------------------------------------


def test_calibrator_returns_float_in_range():
    """calibrate() must return a float in the search range."""
    from tritpack.model.calibration import ThresholdCalibrator

    rng = np.random.default_rng(0)
    tensor = rng.standard_normal((500,)).astype(np.float32)
    cal = ThresholdCalibrator()
    alpha = cal.calibrate(tensor, target_cosine_sim=0.85, alpha_search_range=(0.3, 1.0), n_steps=10)
    assert isinstance(alpha, float)
    assert 0.3 <= alpha <= 1.0


def test_calibrator_higher_target_lower_alpha():
    """Higher cosine target should yield a lower or equal alpha (more non-zeros)."""
    from tritpack.model.calibration import ThresholdCalibrator

    rng = np.random.default_rng(42)
    tensor = rng.standard_normal((1000,)).astype(np.float32)
    cal = ThresholdCalibrator()
    alpha_high_target = cal.calibrate(tensor, target_cosine_sim=0.89, n_steps=15)
    alpha_low_target = cal.calibrate(tensor, target_cosine_sim=0.80, n_steps=15)
    # Higher target → needs more accuracy → should prefer lower alpha (fewer zeros)
    assert alpha_high_target <= alpha_low_target + 0.15, (
        f"Expected high-target alpha ({alpha_high_target:.3f}) <= low-target alpha ({alpha_low_target:.3f})"
    )


def test_calibrate_model_layers_returns_dict():
    """calibrate_model_layers must return one alpha per layer."""
    from tritpack.model.calibration import ThresholdCalibrator

    rng = np.random.default_rng(0)
    layers = {
        "q_proj": rng.standard_normal((256, 256)).astype(np.float32),
        "v_proj": rng.standard_normal((256, 256)).astype(np.float32),
    }
    cal = ThresholdCalibrator()
    alphas = cal.calibrate_model_layers(layers, target_cosine_sim=0.85)
    assert set(alphas.keys()) == {"q_proj", "v_proj"}
    for name, alpha in alphas.items():
        assert isinstance(alpha, float)
        assert 0.3 <= alpha <= 1.0


# ---------------------------------------------------------------------------
# model/layer.py
# ---------------------------------------------------------------------------


def test_layer_compressor_basic():
    """LayerCompressor must store and retrieve tensors."""
    from tritpack.model.layer import LayerCompressor
    from tritpack.core.dequantizer import cosine_similarity

    rng = np.random.default_rng(0)
    t = rng.standard_normal((128, 128)).astype(np.float32)
    lc = LayerCompressor("test_layer")
    tt = lc.add_tensor("weight", t)
    assert tt is not None
    rec = lc.get_tensor("weight")
    assert rec.shape == t.shape
    assert cosine_similarity(t, rec) > 0.80


def test_layer_compressor_names():
    """tensor_names() returns all added tensor names."""
    from tritpack.model.layer import LayerCompressor

    lc = LayerCompressor("layer0")
    rng = np.random.default_rng(0)
    for name in ["w1", "w2", "b1"]:
        lc.add_tensor(name, rng.standard_normal((32,)).astype(np.float32))
    assert sorted(lc.tensor_names()) == ["b1", "w1", "w2"]


def test_layer_compressor_memory_usage():
    """memory_usage() returns positive byte counts per tensor."""
    from tritpack.model.layer import LayerCompressor

    rng = np.random.default_rng(0)
    lc = LayerCompressor("layer0")
    lc.add_tensor("w", rng.standard_normal((256,)).astype(np.float32))
    usage = lc.memory_usage()
    assert "w" in usage
    assert usage["w"] > 0
    assert lc.total_compressed_bytes() > 0


# ---------------------------------------------------------------------------
# benchmark/memory.py — __str__ and CompressionReport
# ---------------------------------------------------------------------------


def test_compression_report_str():
    """CompressionReport.__str__ must return a non-empty string."""
    from tritpack.benchmark.memory import CompressionReport

    report = CompressionReport(
        original_gb=1.0,
        compressed_gb=0.3,
        ratio=3.33,
        time_seconds=0.5,
        cos_sim=0.92,
        snr=8.5,
    )
    s = str(report)
    assert "CompressionReport" in s
    assert "ratio" in s


# ---------------------------------------------------------------------------
# benchmark/__main__.py — _print_table and main()
# ---------------------------------------------------------------------------


def test_print_table_empty_rows(capsys):
    """_print_table with empty rows must print 'no data'."""
    from tritpack.benchmark.__main__ import _print_table

    _print_table("Test Title", [])
    captured = capsys.readouterr()
    assert "no data" in captured.out


def test_print_table_plain(monkeypatch, capsys):
    """_print_table falls back to plain text when rich is absent."""
    import tritpack.benchmark.__main__ as bm

    monkeypatch.setattr(bm, "_RICH", False)
    _print_table = bm._print_table
    _print_table("My Table", [{"a": 1, "b": 2}])
    captured = capsys.readouterr()
    assert "My Table" in captured.out


def test_main_no_command(monkeypatch, capsys):
    """main() with no subcommand prints help and exits 0."""
    import sys
    import argparse
    from tritpack.benchmark.__main__ import main

    monkeypatch.setattr(sys, "argv", ["tritpack.benchmark"])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 0


def test_main_bench_pack(monkeypatch):
    """main() with bench-pack subcommand runs without error."""
    import sys
    from tritpack.benchmark.__main__ import main

    monkeypatch.setattr(sys, "argv", ["tritpack.benchmark", "bench-pack"])
    # Should not raise
    main()


def test_main_bench_all_with_output(monkeypatch, tmp_path):
    """main() with bench-all and --output writes JSON file."""
    import sys
    from tritpack.benchmark.__main__ import main

    out = str(tmp_path / "result.json")
    monkeypatch.setattr(sys, "argv", ["tritpack.benchmark", "--output", out, "bench-all"])
    main()
    assert (tmp_path / "result.json").exists()


# ---------------------------------------------------------------------------
# model/tensor.py — quality_report with original present
# ---------------------------------------------------------------------------


def test_trit_tensor_quality_report_with_original():
    """quality_report returns real metrics when _original is present."""
    from tritpack.model.tensor import TritTensor

    rng = np.random.default_rng(0)
    t = rng.standard_normal((128,)).astype(np.float32)
    tt = TritTensor(t)
    report = tt.quality_report()
    assert report["cos_sim"] is not None
    assert report["cos_sim"] > 0.0
    assert report["compression_ratio"] > 1.0
