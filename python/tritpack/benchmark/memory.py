"""
TritPack memory benchmark — measure RAM usage during compression.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import psutil


@dataclass
class CompressionReport:
    """Results from a model compression benchmark."""

    original_gb: float
    compressed_gb: float
    ratio: float
    time_seconds: float
    cos_sim: float
    snr: float

    def __str__(self) -> str:
        return (
            f"CompressionReport("
            f"original={self.original_gb:.3f} GB, "
            f"compressed={self.compressed_gb:.3f} GB, "
            f"ratio={self.ratio:.2f}x, "
            f"time={self.time_seconds:.2f}s, "
            f"cos_sim={self.cos_sim:.4f}, "
            f"snr={self.snr:.1f} dB)"
        )


def measure_process_ram() -> float:
    """
    Return current process RSS (resident set size) in MB.

    Returns
    -------
    float
        Memory in megabytes.
    """
    proc = psutil.Process()
    return proc.memory_info().rss / (1 << 20)


def benchmark_model_compression(
    gguf_path: str,
    alpha: float = 0.7,
) -> CompressionReport:
    """
    Compress every tensor in a GGUF file and report overall statistics.

    Parameters
    ----------
    gguf_path : str
        Path to a ``.gguf`` model file.
    alpha : float
        Quantization threshold ratio.

    Returns
    -------
    CompressionReport
    """
    from tritpack.backends.gguf_backend import GGUFBackend
    from tritpack.core.quantizer import TernaryQuantizer
    from tritpack.core.dequantizer import TernaryDequantizer, cosine_similarity, snr_db

    backend = GGUFBackend(gguf_path)
    q = TernaryQuantizer(alpha=alpha)
    dq = TernaryDequantizer()

    original_bytes = 0
    compressed_bytes = 0
    cos_sims: list[float] = []
    snrs: list[float] = []

    t0 = time.perf_counter()
    for name, tensor in backend.iter_tensors():
        original_bytes += tensor.nbytes
        qt = q.quantize(tensor)
        compressed_bytes += qt.size_bytes()
        reconstructed = dq.dequantize(qt)
        cos_sims.append(cosine_similarity(tensor, reconstructed))
        snrs.append(snr_db(tensor, reconstructed))

    elapsed = time.perf_counter() - t0
    gb = 1 << 30

    return CompressionReport(
        original_gb=original_bytes / gb,
        compressed_gb=compressed_bytes / gb,
        ratio=original_bytes / compressed_bytes if compressed_bytes else 1.0,
        time_seconds=elapsed,
        cos_sim=float(np.mean(cos_sims)) if cos_sims else 0.0,
        snr=float(np.mean(snrs)) if snrs else 0.0,
    )


def benchmark_tensor_compression(
    tensor: np.ndarray,
    alpha: float = 0.7,
) -> CompressionReport:
    """
    Compress a single tensor and return a :class:`CompressionReport`.
    """
    from tritpack.core.quantizer import TernaryQuantizer
    from tritpack.core.dequantizer import TernaryDequantizer, cosine_similarity, snr_db

    q = TernaryQuantizer(alpha=alpha)
    dq = TernaryDequantizer()

    t0 = time.perf_counter()
    qt = q.quantize(tensor)
    reconstructed = dq.dequantize(qt)
    elapsed = time.perf_counter() - t0

    gb = 1 << 30
    return CompressionReport(
        original_gb=tensor.nbytes / gb,
        compressed_gb=qt.size_bytes() / gb,
        ratio=tensor.nbytes / qt.size_bytes() if qt.size_bytes() else 1.0,
        time_seconds=elapsed,
        cos_sim=cosine_similarity(tensor, reconstructed),
        snr=snr_db(tensor, reconstructed),
    )
