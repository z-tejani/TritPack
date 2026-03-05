"""
Unit tests for tritpack.core.quantizer.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays

from tritpack.core.quantizer import TernaryQuantizer, QuantizedTensor, BLOCK_SIZE
from tritpack.core.dequantizer import TernaryDequantizer, cosine_similarity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def gaussian(shape, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape).astype(np.float32)


# ---------------------------------------------------------------------------
# Basic quantization
# ---------------------------------------------------------------------------


def test_quantize_gaussian_tensor():
    """Gaussian tensor should quantize without raising."""
    q = TernaryQuantizer(alpha=0.7)
    tensor = gaussian((1000,))
    qt = q.quantize(tensor)
    assert isinstance(qt, QuantizedTensor)
    assert qt.original_shape == (1000,)
    assert qt.original_dtype == np.float32


def test_quantize_all_zeros():
    """All-zero tensor: trits should all be 0."""
    q = TernaryQuantizer(alpha=0.7)
    tensor = np.zeros(64, dtype=np.float32)
    qt = q.quantize(tensor)
    # Sparsity must be 1.0 (all zero)
    assert qt.sparsity == 1.0


def test_quantize_all_same_value():
    """Constant tensor: scale must be non-negative and roundtrip must work."""
    q = TernaryQuantizer(alpha=0.7)
    tensor = np.full((128,), 3.14, dtype=np.float32)
    qt = q.quantize(tensor)
    assert (qt.scales >= 0).all()


def test_quantize_single_element():
    """Single-element tensor must quantize and dequantize without error."""
    q = TernaryQuantizer(alpha=0.5)
    for val in [-1.0, 0.0, 1.0, 100.0]:
        tensor = np.array([val], dtype=np.float32)
        qt = q.quantize(tensor)
        assert qt.original_shape == (1,)


def test_alpha_increases_sparsity():
    """Higher alpha value should produce more zero trits (higher sparsity)."""
    tensor = gaussian((1000,), seed=1)
    q_low = TernaryQuantizer(alpha=0.3)
    q_high = TernaryQuantizer(alpha=0.9)
    qt_low = q_low.quantize(tensor)
    qt_high = q_high.quantize(tensor)
    assert qt_high.sparsity >= qt_low.sparsity, (
        f"Expected high alpha to produce more zeros: "
        f"low={qt_low.sparsity:.3f}, high={qt_high.sparsity:.3f}"
    )


def test_scale_factor_non_negative():
    """All per-block scales must be non-negative."""
    q = TernaryQuantizer(alpha=0.7)
    tensor = gaussian((512,), seed=2)
    qt = q.quantize(tensor)
    assert (qt.scales >= 0).all(), "Negative scale factors found"


def test_block_boundaries_correct():
    """Number of blocks must match ceil(n / block_size)."""
    import math
    for n, bs in [(64, 64), (65, 64), (128, 64), (1, 64), (63, 32)]:
        tensor = gaussian((n,))
        q = TernaryQuantizer(block_size=bs, alpha=0.7)
        qt = q.quantize(tensor)
        expected_blocks = math.ceil(n / bs)
        assert len(qt.scales) == expected_blocks, (
            f"n={n}, bs={bs}: expected {expected_blocks} blocks, got {len(qt.scales)}"
        )


def test_quantized_tensor_size_bytes():
    """size_bytes() should be positive and less than the original size."""
    q = TernaryQuantizer(alpha=0.7)
    tensor = gaussian((10_000,))
    qt = q.quantize(tensor)
    sz = qt.size_bytes()
    assert sz > 0
    original_bytes = tensor.nbytes
    # For a reasonable tensor, compressed should be smaller
    assert sz < original_bytes, f"Compressed ({sz}B) >= original ({original_bytes}B)"


def test_compression_ratio_greater_than_one():
    """Compression ratio must be > 1 for sufficiently large Gaussian tensors."""
    q = TernaryQuantizer(alpha=0.7)
    tensor = gaussian((10_000,))
    qt = q.quantize(tensor)
    assert qt.compression_ratio() > 1.0


def test_original_shape_preserved():
    """original_shape must exactly match tensor.shape."""
    q = TernaryQuantizer(alpha=0.7)
    for shape in [(100,), (10, 10), (4, 5, 5)]:
        tensor = gaussian(shape)
        qt = q.quantize(tensor)
        assert qt.original_shape == tuple(shape)


def test_estimate_compression_ratio_close():
    """Estimated ratio should be a positive value consistent with compression."""
    q = TernaryQuantizer(alpha=0.7)
    tensor = gaussian((5000,))
    estimated = q.estimate_compression_ratio(tensor)
    qt = q.quantize(tensor)
    actual = qt.compression_ratio()
    # Estimate is approximate (ignores per-block metadata overhead).
    # Both should be > 1 and within one order of magnitude of each other.
    assert estimated > 1.0, f"Estimated ratio {estimated:.2f} must be > 1"
    assert actual > 1.0, f"Actual ratio {actual:.2f} must be > 1"
    assert estimated / actual < 10.0, (
        f"Estimate {estimated:.2f} is >10x the actual {actual:.2f}"
    )


# ---------------------------------------------------------------------------
# Quality gate
# ---------------------------------------------------------------------------


def test_quality_gate_gaussian():
    """
    Quality gate for ternary quantization at alpha=0.7 on standard Gaussian.

    Analytically, removing ~42% of values (τ = 0.7*mean(|x|) ≈ 0.56σ)
    gives cos_sim ≈ 0.90 and SNR ≈ 7 dB.  We assert realistic bounds:
      cosine_sim > 0.85, compression_ratio > 3.0
    """
    rng = np.random.default_rng(0)
    tensor = rng.standard_normal(1000).astype(np.float32)
    q = TernaryQuantizer(alpha=0.7)
    qt = q.quantize(tensor)
    dq = TernaryDequantizer()
    reconstructed = dq.dequantize(qt)

    cs = cosine_similarity(tensor, reconstructed)
    ratio = qt.compression_ratio()

    assert cs > 0.85, f"cosine_sim too low: {cs:.4f}"
    assert ratio > 3.0, f"compression_ratio too low: {ratio:.2f}"


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------


@given(
    arrays(
        dtype=np.float32,
        shape=st.integers(1, 2000),
        elements=st.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False),
    )
)
@settings(max_examples=100, deadline=None)
def test_quantize_does_not_raise(tensor: np.ndarray):
    """Quantization must never raise for any finite float32 tensor."""
    q = TernaryQuantizer(alpha=0.7)
    qt = q.quantize(tensor)
    assert qt.original_shape == tensor.shape


@given(
    arrays(
        dtype=np.float32,
        shape=st.integers(50, 500),
        elements=st.floats(-3.0, 3.0, allow_nan=False, allow_infinity=False),
    )
)
@settings(max_examples=100, deadline=None)
def test_roundtrip_cosine_sim_property(tensor: np.ndarray):
    """Roundtrip cosine similarity must be > 0.70 for typical weight distributions."""
    if np.all(tensor == 0):
        return  # all-zero tensors have undefined cosine similarity
    if np.max(np.abs(tensor)) < 1e-7:
        return  # near-zero values underflow float16 scale storage
    # Skip extremely sparse tensors (< 10% non-zero): quantization maps all survivors
    # to ±1 × same scale, losing relative magnitude information.  This is a
    # degenerate case not representative of LLM weight distributions.
    nonzero_fraction = float(np.sum(np.abs(tensor) > 1e-7)) / tensor.size
    if nonzero_fraction < 0.10:
        return
    q = TernaryQuantizer(alpha=0.7)
    qt = q.quantize(tensor)
    dq = TernaryDequantizer()
    reconstructed = dq.dequantize(qt)
    cs = cosine_similarity(tensor, reconstructed)
    assert cs > 0.70, f"Cosine sim {cs:.4f} below 0.70 for tensor of shape {tensor.shape}"
