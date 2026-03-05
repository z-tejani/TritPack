"""
TritPack calibration — find optimal quantization threshold per tensor.

The calibrator binary-searches over the ``alpha`` parameter to find the
highest compression threshold that keeps cosine similarity above a target.
"""

from __future__ import annotations

import numpy as np

from tritpack.core.quantizer import TernaryQuantizer
from tritpack.core.dequantizer import TernaryDequantizer, cosine_similarity


class ThresholdCalibrator:
    """
    Finds the optimal ``alpha`` (threshold ratio) for each tensor so that
    compression is maximised while cosine similarity stays above a target.

    Parameters
    ----------
    block_size : int
        Block size forwarded to :class:`TernaryQuantizer`.
    """

    def __init__(self, block_size: int = 64) -> None:
        self.block_size = block_size
        self._dequantizer = TernaryDequantizer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calibrate(
        self,
        tensor: np.ndarray,
        target_cosine_sim: float = 0.97,
        alpha_search_range: tuple[float, float] = (0.3, 1.0),
        n_steps: int = 20,
    ) -> float:
        """
        Find the highest ``alpha`` that keeps cosine similarity ≥ *target*.

        Uses a linear scan over *n_steps* evenly-spaced alpha values (rather
        than a pure binary search) so that the full quality/compression
        trade-off curve is evaluated.

        Parameters
        ----------
        tensor : np.ndarray
            Input tensor.
        target_cosine_sim : float
            Minimum acceptable cosine similarity (default 0.97).
        alpha_search_range : tuple[float, float]
            ``(low, high)`` range to search over.
        n_steps : int
            Number of candidate alpha values to evaluate.

        Returns
        -------
        float
            Highest *alpha* for which cos_sim ≥ *target*, or *low* if no
            candidate satisfies the target.
        """
        low, high = alpha_search_range
        alphas = np.linspace(low, high, n_steps)
        best_alpha = float(low)

        for alpha in alphas:
            q = TernaryQuantizer(block_size=self.block_size, alpha=float(alpha))
            qt = q.quantize(tensor)
            reconstructed = self._dequantizer.dequantize(qt)
            sim = cosine_similarity(tensor, reconstructed)
            if sim >= target_cosine_sim:
                best_alpha = float(alpha)
            else:
                # Quality dropped below target; stop searching higher alphas
                break

        return best_alpha

    def calibrate_model_layers(
        self,
        layers: dict[str, np.ndarray],
        target_cosine_sim: float = 0.97,
    ) -> dict[str, float]:
        """
        Calibrate each layer independently.

        Parameters
        ----------
        layers : dict[str, np.ndarray]
            Mapping of layer identifier to weight tensor.
        target_cosine_sim : float
            Target cosine similarity for all layers.

        Returns
        -------
        dict[str, float]
            ``{layer_id: optimal_alpha}`` for each layer.
        """
        return {
            name: self.calibrate(tensor, target_cosine_sim=target_cosine_sim)
            for name, tensor in layers.items()
        }
