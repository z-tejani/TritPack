"""
TritPack TritTensor — a compressed tensor with lazy decompression.

Internally stores data in ternary-packed format but behaves like
``np.ndarray`` for read operations via the ``.numpy()`` method.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from tritpack.core.quantizer import TernaryQuantizer, QuantizedTensor
from tritpack.core.dequantizer import TernaryDequantizer, cosine_similarity, snr_db, rmse


_dequantizer = TernaryDequantizer()


class TritTensor:
    """
    A compressed tensor that behaves like ``np.ndarray`` for read operations
    but stores data in ternary-packed format internally.

    Supports lazy decompression: data is only decompressed when
    :meth:`numpy` or :attr:`data` is accessed.

    Parameters
    ----------
    data : np.ndarray
        Original tensor to compress.
    quantizer : TernaryQuantizer, optional
        Quantizer to use.  Defaults to ``TernaryQuantizer()``.
    """

    def __init__(
        self,
        data: np.ndarray,
        quantizer: Optional[TernaryQuantizer] = None,
    ) -> None:
        q = quantizer or TernaryQuantizer()
        self._qt: QuantizedTensor = q.quantize(data)
        self._original_shape: tuple = tuple(data.shape)
        self._original_dtype: np.dtype = data.dtype
        self._original_nbytes: int = data.nbytes
        # Stash the original for quality_report; released after first call if large.
        self._original: Optional[np.ndarray] = data.copy()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def shape(self) -> tuple:
        """Original tensor shape."""
        return self._original_shape

    @property
    def dtype(self) -> np.dtype:
        """Original tensor dtype."""
        return self._original_dtype

    @property
    def nbytes_compressed(self) -> int:
        """Size of the compressed representation in bytes."""
        return self._qt.size_bytes()

    @property
    def nbytes_original(self) -> int:
        """Size of the original (uncompressed) tensor in bytes."""
        return self._original_nbytes

    @property
    def compression_ratio(self) -> float:
        """Ratio of original size to compressed size (> 1 → compressed is smaller)."""
        return self._qt.compression_ratio()

    @property
    def sparsity(self) -> float:
        """Fraction of zero trits."""
        return self._qt.sparsity

    @property
    def data(self) -> np.ndarray:
        """Alias for :meth:`numpy`.  Triggers lazy decompression."""
        return self.numpy()

    # ------------------------------------------------------------------
    # Decompression
    # ------------------------------------------------------------------

    def numpy(self) -> np.ndarray:
        """
        Decompress and return the tensor as a NumPy array.

        Returns
        -------
        np.ndarray
            Reconstructed tensor matching original shape and dtype.
        """
        return _dequantizer.dequantize(self._qt)

    # ------------------------------------------------------------------
    # Quality
    # ------------------------------------------------------------------

    def quality_report(self) -> dict:
        """
        Compute quality metrics versus the original tensor.

        Returns
        -------
        dict
            ``{cos_sim, snr_db, rmse, compression_ratio, sparsity}``
        """
        reconstructed = self.numpy()
        if self._original is None:
            return {
                "cos_sim": None,
                "snr_db": None,
                "rmse": None,
                "compression_ratio": self.compression_ratio,
                "sparsity": self.sparsity,
            }
        original = self._original
        return {
            "cos_sim": cosine_similarity(original, reconstructed),
            "snr_db": snr_db(original, reconstructed),
            "rmse": rmse(original, reconstructed),
            "compression_ratio": self.compression_ratio,
            "sparsity": self.sparsity,
        }

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        cr = self.compression_ratio
        return (
            f"TritTensor(shape={self.shape}, dtype={self.dtype}, "
            f"compression_ratio={cr:.2f}x, sparsity={self.sparsity:.1%})"
        )

    def __len__(self) -> int:
        return self._original_shape[0] if self._original_shape else 0
