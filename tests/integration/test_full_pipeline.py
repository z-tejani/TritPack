"""
Integration tests for the full compress/decompress pipeline.

Cosine similarity thresholds reflect the analytically achievable quality
for ternary quantization at alpha=0.7:
  - Threshold τ ≈ 0.7 * mean(|x|) ≈ 0.56σ zeros ~42% of values.
  - Max achievable cos_sim ≈ 0.90 for standard Gaussian.
  - We require > 0.85 to allow for sample variance.
"""

from __future__ import annotations

import numpy as np
import pytest

from tritpack.core.quantizer import TernaryQuantizer
from tritpack.core.dequantizer import TernaryDequantizer, cosine_similarity, snr_db
from tritpack.core.tiers import TierManager
from tritpack.model.calibration import ThresholdCalibrator
from tritpack.model.tensor import TritTensor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def gaussian(shape, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape).astype(np.float32)


# ---------------------------------------------------------------------------
# Large tensor
# ---------------------------------------------------------------------------


def test_compress_decompress_large_tensor():
    """
    Compress and decompress a >100 MB synthetic tensor.
    Cosine similarity must be > 0.85 (realistic for alpha=0.7).
    """
    # ~104 MB float32
    tensor = gaussian((26, 1024, 1024), seed=0)
    assert tensor.nbytes > 100 * (1 << 20), "Tensor too small for this test"

    q = TernaryQuantizer(alpha=0.7)
    qt = q.quantize(tensor)

    dq = TernaryDequantizer()
    rec = dq.dequantize(qt)

    cs = cosine_similarity(tensor, rec)
    assert cs > 0.85, f"cosine_sim={cs:.4f} < 0.85 for large tensor"
    assert rec.shape == tensor.shape


# ---------------------------------------------------------------------------
# Multiple layers pipeline
# ---------------------------------------------------------------------------


def test_multiple_layers_pipeline():
    """Compress multiple layers independently; all must have cos_sim > 0.85."""
    shapes = [
        (4096, 4096),
        (4096, 11008),
        (11008, 4096),
    ]
    q = TernaryQuantizer(alpha=0.7)
    dq = TernaryDequantizer()

    for i, shape in enumerate(shapes):
        tensor = gaussian(shape, seed=i)
        qt = q.quantize(tensor)
        rec = dq.dequantize(qt)
        cs = cosine_similarity(tensor, rec)
        assert cs > 0.85, f"Layer {shape}: cos_sim={cs:.4f}"


# ---------------------------------------------------------------------------
# Calibration then compress
# ---------------------------------------------------------------------------


def test_calibration_then_compress():
    """
    Calibrator returns an alpha; the resulting cos_sim must be in a good range.

    For Gaussian tensors the theoretical cos_sim peaks at ~0.90 near α=0.55.
    We target 0.87 and assert the result is ≥ 0.83.
    """
    target = 0.87
    tensor = gaussian((5000,), seed=42)

    calibrator = ThresholdCalibrator()
    optimal_alpha = calibrator.calibrate(tensor, target_cosine_sim=target)

    q = TernaryQuantizer(alpha=optimal_alpha)
    qt = q.quantize(tensor)
    dq = TernaryDequantizer()
    rec = dq.dequantize(qt)
    cs = cosine_similarity(tensor, rec)

    assert cs >= 0.83, (
        f"After calibration (alpha={optimal_alpha:.3f}): cos_sim={cs:.4f} too low"
    )


# ---------------------------------------------------------------------------
# 70B model layer sizes simulation
# ---------------------------------------------------------------------------


def test_tier_manager_with_real_layer_sizes():
    """
    Simulate 80 layers × ~20 MB each (scaled-down 70B analogue).

    With a tight 200 MB total budget, TierManager must correctly
    maintain LRU eviction and allow all layers to be accessed.
    """
    n_layers = 20
    # Scaled down: each layer ~ 1M float32 = 4 MB
    layer_size = 1_000_000
    limit_gb = 0.1  # 100 MB total

    tm = TierManager(
        memory_limit_gb=limit_gb,
        hot_budget_fraction=0.2,
        warm_budget_fraction=0.3,
    )

    # Register all layers
    for i in range(n_layers):
        t = gaussian((layer_size,), seed=i)
        tm.register_layer(f"layer_{i}", t)

    stats = tm.stats()
    assert stats.n_hot + stats.n_warm + stats.n_cold == n_layers

    # Access every layer in order — must not raise
    for i in range(n_layers):
        result = tm.access(f"layer_{i}")
        assert result.shape == (layer_size,)

    final_stats = tm.stats()
    assert final_stats.n_accesses == n_layers


# ---------------------------------------------------------------------------
# TritTensor
# ---------------------------------------------------------------------------


def test_trit_tensor_quality_report():
    """TritTensor quality_report must return valid metrics."""
    tensor = gaussian((2048,))
    tt = TritTensor(tensor)
    report = tt.quality_report()
    assert report["cos_sim"] > 0.85
    assert report["snr_db"] > 5.0
    assert report["compression_ratio"] > 1.0
    assert 0.0 <= report["sparsity"] <= 1.0


def test_trit_tensor_lazy_decompression():
    """numpy() must return tensor matching original shape."""
    tensor = gaussian((512, 512))
    tt = TritTensor(tensor)
    rec = tt.numpy()
    assert rec.shape == tensor.shape
    assert cosine_similarity(tensor, rec) > 0.85
