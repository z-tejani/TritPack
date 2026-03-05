"""
TritPack quality benchmark — reconstruction quality across alpha values.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tritpack.core.quantizer import TernaryQuantizer
from tritpack.core.dequantizer import TernaryDequantizer, cosine_similarity, snr_db, rmse


@dataclass
class QualityPoint:
    """Quality metrics for a single alpha value."""

    alpha: float
    cos_sim: float
    snr_db: float
    rmse: float
    sparsity: float
    compression_ratio: float


@dataclass
class QualityReport:
    """Quality sweep across multiple alpha values."""

    points: list[QualityPoint] = field(default_factory=list)

    def best_alpha_for_target(self, target_cos_sim: float = 0.97) -> float | None:
        """Return the highest alpha achieving *target_cos_sim*."""
        candidates = [p for p in self.points if p.cos_sim >= target_cos_sim]
        return max((p.alpha for p in candidates), default=None)

    def summary_table(self) -> list[dict]:
        return [
            {
                "alpha": p.alpha,
                "cos_sim": round(p.cos_sim, 4),
                "snr_db": round(p.snr_db, 2),
                "rmse": round(p.rmse, 6),
                "sparsity_%": round(p.sparsity * 100, 1),
                "ratio": round(p.compression_ratio, 2),
            }
            for p in self.points
        ]


def benchmark_reconstruction_quality(
    tensor: np.ndarray,
    alphas: list[float],
    block_size: int = 64,
) -> QualityReport:
    """
    Evaluate reconstruction quality across a range of *alphas*.

    Parameters
    ----------
    tensor : np.ndarray
        Input floating-point tensor.
    alphas : list[float]
        Alpha values to evaluate.
    block_size : int
        Block size for quantization.

    Returns
    -------
    QualityReport
    """
    dq = TernaryDequantizer()
    points: list[QualityPoint] = []

    for alpha in alphas:
        q = TernaryQuantizer(block_size=block_size, alpha=alpha)
        qt = q.quantize(tensor)
        rec = dq.dequantize(qt)

        points.append(
            QualityPoint(
                alpha=alpha,
                cos_sim=cosine_similarity(tensor, rec),
                snr_db=snr_db(tensor, rec),
                rmse=rmse(tensor, rec),
                sparsity=qt.sparsity,
                compression_ratio=qt.compression_ratio(),
            )
        )

    return QualityReport(points=points)
