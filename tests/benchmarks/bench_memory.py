"""
Memory-focused benchmark: measure RAM during compression.
"""

from __future__ import annotations

import numpy as np
import pytest

from tritpack.benchmark.memory import measure_process_ram, benchmark_tensor_compression


def test_bench_memory_compression(benchmark):
    """Benchmark full compress/decompress cycle tracking RAM."""
    rng = np.random.default_rng(0)
    tensor = rng.standard_normal((4096, 4096)).astype(np.float32)  # 64 MB
    benchmark(benchmark_tensor_compression, tensor)


def test_measure_process_ram():
    """measure_process_ram() must return a positive float."""
    ram = measure_process_ram()
    assert ram > 0
