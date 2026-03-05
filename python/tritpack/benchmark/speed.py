"""
TritPack speed benchmark — packing throughput and quantization latency.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from tritpack.core.packing import pack_trits, pack_trits_vectorized
from tritpack.core.quantizer import TernaryQuantizer


@dataclass
class SpeedReport:
    """Pack throughput measurements."""

    sizes_mb: list[float]
    throughputs_mb_s: list[float]       # reference (loop)
    throughputs_vec_mb_s: list[float]   # vectorized

    def summary(self) -> list[dict]:
        return [
            {
                "size_mb": s,
                "loop_MB_s": round(t, 1),
                "vec_MB_s": round(tv, 1),
                "speedup": round(tv / t, 2) if t > 0 else None,
            }
            for s, t, tv in zip(self.sizes_mb, self.throughputs_mb_s, self.throughputs_vec_mb_s)
        ]


@dataclass
class LatencyReport:
    """Quantization latency measurements."""

    shapes: list[tuple]
    latencies_ms: list[float]

    def summary(self) -> list[dict]:
        return [
            {"shape": s, "latency_ms": round(lat, 3)}
            for s, lat in zip(self.shapes, self.latencies_ms)
        ]


def _random_trits(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(-1, 2, size=n, dtype=np.int8)


def benchmark_pack_throughput(
    sizes_mb: list[float],
    n_repeats: int = 3,
    seed: int = 42,
) -> SpeedReport:
    """
    Measure trit packing throughput for different data sizes.

    Parameters
    ----------
    sizes_mb : list[float]
        Sizes in MB to benchmark.
    n_repeats : int
        Number of timing repetitions (best of n_repeats is reported).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    SpeedReport
    """
    rng = np.random.default_rng(seed)
    throughputs_loop: list[float] = []
    throughputs_vec: list[float] = []

    for size_mb in sizes_mb:
        n = int(size_mb * (1 << 20))  # treat 1 trit ≈ 1 byte for sizing
        trits = _random_trits(n, rng)

        # Reference (loop) implementation
        best_loop = float("inf")
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            pack_trits(trits)
            best_loop = min(best_loop, time.perf_counter() - t0)
        throughputs_loop.append(size_mb / best_loop if best_loop > 0 else 0.0)

        # Vectorized implementation
        best_vec = float("inf")
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            pack_trits_vectorized(trits)
            best_vec = min(best_vec, time.perf_counter() - t0)
        throughputs_vec.append(size_mb / best_vec if best_vec > 0 else 0.0)

    return SpeedReport(
        sizes_mb=sizes_mb,
        throughputs_mb_s=throughputs_loop,
        throughputs_vec_mb_s=throughputs_vec,
    )


def benchmark_quantize_latency(
    tensor_sizes: list[tuple],
    alpha: float = 0.7,
    n_repeats: int = 3,
    seed: int = 42,
) -> LatencyReport:
    """
    Measure quantization latency for tensors of different shapes.

    Parameters
    ----------
    tensor_sizes : list[tuple]
        List of shapes to benchmark.
    alpha : float
        Quantization threshold.
    n_repeats : int
        Timing repetitions.
    seed : int
        Random seed.

    Returns
    -------
    LatencyReport
    """
    rng = np.random.default_rng(seed)
    q = TernaryQuantizer(alpha=alpha)
    latencies: list[float] = []

    for shape in tensor_sizes:
        tensor = rng.standard_normal(shape).astype(np.float32)
        best = float("inf")
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            q.quantize(tensor)
            best = min(best, time.perf_counter() - t0)
        latencies.append(best * 1000)  # convert to ms

    return LatencyReport(shapes=tensor_sizes, latencies_ms=latencies)
