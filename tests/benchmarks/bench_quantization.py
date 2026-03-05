"""
pytest-benchmark tests for quantization performance.
"""

from __future__ import annotations

import numpy as np
import pytest

from tritpack.core.quantizer import TernaryQuantizer
from tritpack.core.dequantizer import TernaryDequantizer


def gaussian(shape, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape).astype(np.float32)


def test_bench_quantize_small(benchmark):
    """Quantize a 1K-element tensor."""
    t = gaussian((1000,))
    q = TernaryQuantizer()
    benchmark(q.quantize, t)


def test_bench_quantize_medium(benchmark):
    """Quantize a 1M-element tensor (4 MB)."""
    t = gaussian((1_000_000,))
    q = TernaryQuantizer()
    benchmark(q.quantize, t)


def test_bench_quantize_large(benchmark):
    """Quantize a 4096×4096 tensor (64 MB)."""
    t = gaussian((4096, 4096))
    q = TernaryQuantizer()
    benchmark(q.quantize, t)


def test_bench_dequantize_medium(benchmark):
    """Dequantize a 1M-element tensor."""
    t = gaussian((1_000_000,))
    q = TernaryQuantizer()
    qt = q.quantize(t)
    dq = TernaryDequantizer()
    benchmark(dq.dequantize, qt)
