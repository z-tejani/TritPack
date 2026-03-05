"""
pytest-benchmark tests for trit packing performance.
"""

from __future__ import annotations

import numpy as np
import pytest

from tritpack.core.packing import pack_trits, pack_trits_vectorized, unpack_trits


def _trits(n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(-1, 2, size=n, dtype=np.int8)


# ---------------------------------------------------------------------------
# Pack throughput
# ---------------------------------------------------------------------------


def test_bench_pack_1mb(benchmark):
    """Benchmark packing ~1 MB of trits."""
    trits = _trits(1 << 20)
    benchmark(pack_trits, trits)


def test_bench_pack_10mb(benchmark):
    """Benchmark packing ~10 MB of trits."""
    trits = _trits(10 * (1 << 20))
    benchmark(pack_trits, trits)


def test_bench_pack_100mb(benchmark):
    """Benchmark packing ~100 MB of trits."""
    trits = _trits(100 * (1 << 20))
    benchmark(pack_trits, trits)


# ---------------------------------------------------------------------------
# Vectorized vs reference
# ---------------------------------------------------------------------------


def test_bench_pack_vec_1mb(benchmark):
    """Vectorised packing ~1 MB."""
    trits = _trits(1 << 20)
    benchmark(pack_trits_vectorized, trits)


def test_bench_pack_vec_10mb(benchmark):
    """Vectorised packing ~10 MB."""
    trits = _trits(10 * (1 << 20))
    benchmark(pack_trits_vectorized, trits)
