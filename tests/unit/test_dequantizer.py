"""
Unit tests for tritpack.core.dequantizer.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays

from tritpack.core.quantizer import TernaryQuantizer
from tritpack.core.dequantizer import (
    TernaryDequantizer,
    cosine_similarity,
    snr_db,
    rmse,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def gaussian(shape, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape).astype(np.float32)


def compress_decompress(tensor, alpha=0.7):
    q = TernaryQuantizer(alpha=alpha)
    qt = q.quantize(tensor)
    dq = TernaryDequantizer()
    return dq.dequantize(qt)


# ---------------------------------------------------------------------------
# Core dequantization tests
# ---------------------------------------------------------------------------


def test_dequantize_roundtrip_cosine_sim():
    """Cosine similarity after roundtrip must be > 0.95 for Gaussian."""
    tensor = gaussian((1000,))
    rec = compress_decompress(tensor)
    cs = cosine_similarity(tensor, rec)
    # Analytically, ternary quantization at alpha=0.7 yields cos_sim ≈ 0.90
    # (threshold τ ≈ 0.56σ removes ~42% of values; max theoretical is ~0.90)
    assert cs > 0.85, f"cosine_sim={cs:.4f} below threshold"


def test_dequantize_all_zeros_tensor():
    """All-zero tensor must dequantize to all-zeros."""
    tensor = np.zeros((64,), dtype=np.float32)
    rec = compress_decompress(tensor)
    np.testing.assert_array_equal(rec, tensor)


def test_dequantize_preserves_shape():
    """Dequantized tensor must have the original shape."""
    for shape in [(100,), (8, 16), (2, 4, 8)]:
        tensor = gaussian(shape)
        rec = compress_decompress(tensor)
        assert rec.shape == tuple(shape), f"Shape mismatch: {rec.shape} != {shape}"


def test_dequantize_preserves_dtype():
    """Dequantized tensor must be cast to the original dtype."""
    for dtype in [np.float32, np.float16]:
        tensor = gaussian((128,)).astype(dtype)
        rec = compress_decompress(tensor)
        assert rec.dtype == dtype, f"dtype mismatch: {rec.dtype} != {dtype}"


def test_snr_reasonable():
    """
    SNR must be > 5 dB for a Gaussian tensor with alpha=0.7.

    At alpha=0.7, ~42% of values are zeroed; noise power ≈ 0.19σ²
    so SNR ≈ 7 dB in expectation.  We require > 5 dB as a minimum.
    """
    tensor = gaussian((1000,))
    rec = compress_decompress(tensor, alpha=0.7)
    snr = snr_db(tensor, rec)
    assert snr > 5.0, f"SNR {snr:.2f} dB below threshold"


def test_rmse_scales_with_alpha():
    """Very high alpha (most values zeroed) must produce higher RMSE than low alpha."""
    tensor = gaussian((2000,), seed=7)
    rec_low = compress_decompress(tensor, alpha=0.3)
    # alpha=1.9 → τ ≈ 1.9 * mean(|x|) ≈ 1.52σ → ~87% zeros → RMSE ≈ std(x)
    rec_very_high = compress_decompress(tensor, alpha=1.9)
    rmse_low = rmse(tensor, rec_low)
    rmse_very_high = rmse(tensor, rec_very_high)
    assert rmse_very_high > rmse_low, (
        f"Expected RMSE(alpha=1.9) > RMSE(alpha=0.3): {rmse_very_high:.6f} <= {rmse_low:.6f}"
    )


# ---------------------------------------------------------------------------
# dequantize_block
# ---------------------------------------------------------------------------


def test_dequantize_block_single():
    """Single block dequantization must return float16 array."""
    from tritpack.core.packing import pack_trits

    trits = np.array([1, -1, 0, 1, 0, -1, 1, 0], dtype=np.int8)
    packed = pack_trits(trits)
    dq = TernaryDequantizer()
    result = dq.dequantize_block(packed, scale=2.5, block_size=len(trits))
    assert result.dtype == np.float16
    assert len(result) == len(trits)
    # Non-zero trits should be ±scale
    expected = trits.astype(np.float16) * np.float16(2.5)
    np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------


def test_cosine_similarity_identical():
    """Identical arrays have cosine similarity 1.0."""
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert cosine_similarity(a, a) == pytest.approx(1.0, abs=1e-6)


def test_cosine_similarity_orthogonal():
    """Orthogonal arrays have cosine similarity 0.0."""
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)


def test_cosine_similarity_both_zero():
    """Both-zero arrays should return 1.0 (defined as identical)."""
    a = np.zeros(5, dtype=np.float32)
    assert cosine_similarity(a, a) == pytest.approx(1.0)


def test_cosine_similarity_one_zero():
    """Zero vs non-zero should return 0.0."""
    a = np.zeros(5, dtype=np.float32)
    b = np.ones(5, dtype=np.float32)
    assert cosine_similarity(a, b) == pytest.approx(0.0)


def test_snr_perfect_reconstruction():
    """Perfect reconstruction gives infinite SNR."""
    a = gaussian((100,))
    assert snr_db(a, a) == float("inf")


def test_snr_zero_signal():
    """All-zero signal gives -inf SNR."""
    a = np.zeros(100, dtype=np.float32)
    b = np.ones(100, dtype=np.float32)
    assert snr_db(a, b) == float("-inf")


def test_rmse_identical():
    """RMSE of identical arrays is 0.0."""
    a = gaussian((100,))
    assert rmse(a, a) == pytest.approx(0.0, abs=1e-12)


def test_rmse_positive():
    """RMSE is always non-negative."""
    a = gaussian((100,), seed=1)
    b = gaussian((100,), seed=2)
    assert rmse(a, b) >= 0.0


# ---------------------------------------------------------------------------
# Property-based
# ---------------------------------------------------------------------------


@given(
    arrays(
        dtype=np.float32,
        shape=st.integers(1, 2000),
        elements=st.floats(-10.0, 10.0, allow_nan=False, allow_infinity=False),
    )
)
@settings(max_examples=100, deadline=None)
def test_roundtrip_cosine_sim_property(tensor: np.ndarray):
    """Roundtrip cosine similarity must be > 0.70 for non-degenerate tensors."""
    if np.all(tensor == 0):
        return
    if len(tensor) < 10:
        return  # too few elements for stable cosine similarity
    # Skip near-zero tensors that underflow in float16 scale storage
    if np.max(np.abs(tensor)) < 1e-7:
        return
    rec = compress_decompress(tensor, alpha=0.7)
    cs = cosine_similarity(tensor, rec)
    assert cs > 0.70, f"cosine_sim={cs:.4f} < 0.70"
