"""
TritPack dequantizer — ternary to float reconstruction.

Each block of packed trits is unpacked and multiplied by its scale factor
to produce a float approximation of the original values.
"""

from __future__ import annotations

import numpy as np

from tritpack.core.packing import unpack_trits
from tritpack.core.quantizer import QuantizedTensor


class TernaryDequantizer:
    """
    Reconstruct floating-point tensors from :class:`QuantizedTensor` objects.
    """

    def dequantize(self, qt: QuantizedTensor) -> np.ndarray:
        """
        Reconstruct the original tensor from its ternary-quantized form.

        Parameters
        ----------
        qt : QuantizedTensor
            Compressed tensor produced by :class:`TernaryQuantizer`.

        Returns
        -------
        np.ndarray
            Reconstructed array with *qt.original_shape* and
            *qt.original_dtype*.
        """
        n = int(np.prod(qt.original_shape))
        result = np.zeros(n, dtype=np.float16)

        for b, (offset, blen) in enumerate(zip(qt.block_offsets, qt.block_lengths)):
            end_offset = (
                qt.block_offsets[b + 1] if b + 1 < len(qt.block_offsets) else len(qt.packed_data)
            )
            block_bytes = qt.packed_data[offset:end_offset]
            trits = unpack_trits(block_bytes, blen)
            scale = float(qt.scales[b])
            start = b * qt.block_size
            result[start : start + blen] = trits.astype(np.float16) * np.float16(scale)

        return result.reshape(qt.original_shape).astype(qt.original_dtype)

    def dequantize_block(self, packed: bytes, scale: float, block_size: int) -> np.ndarray:
        """
        Dequantize a single block.

        Used for JIT (just-in-time) decompression where only the
        currently-needed block is expanded into memory.

        Parameters
        ----------
        packed : bytes
            Packed bytes for a single block (with 8-byte header).
        scale : float
            Scale factor for this block.
        block_size : int
            Number of trits in this block.

        Returns
        -------
        np.ndarray
            float16 array of length *block_size*.
        """
        trits = unpack_trits(packed, block_size)
        return trits.astype(np.float16) * np.float16(scale)


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two flat arrays.

    Parameters
    ----------
    a, b : np.ndarray
        Input arrays.  Must have the same total number of elements.

    Returns
    -------
    float
        Cosine similarity in [-1, 1]; 1.0 means identical direction.
    """
    a_flat = a.ravel().astype(np.float64)
    b_flat = b.ravel().astype(np.float64)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a == 0 and norm_b == 0:
        return 1.0
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a_flat, b_flat) / (norm_a * norm_b))


def snr_db(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Signal-to-noise ratio in decibels.

    SNR = 10 * log10(signal_power / noise_power)

    Parameters
    ----------
    original : np.ndarray
        Original (reference) array.
    reconstructed : np.ndarray
        Reconstructed (noisy) array.

    Returns
    -------
    float
        SNR in dB.  Higher is better.  Returns ``inf`` for perfect
        reconstruction and ``-inf`` for zero-signal tensors.
    """
    sig = original.ravel().astype(np.float64)
    rec = reconstructed.ravel().astype(np.float64)
    signal_power = float(np.mean(sig ** 2))
    noise_power = float(np.mean((sig - rec) ** 2))
    if noise_power == 0.0:
        return float("inf")
    if signal_power == 0.0:
        return float("-inf")
    return 10.0 * np.log10(signal_power / noise_power)


def rmse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Root mean square error between *original* and *reconstructed*.

    Parameters
    ----------
    original : np.ndarray
        Reference array.
    reconstructed : np.ndarray
        Reconstructed array.

    Returns
    -------
    float
        RMSE (lower is better).
    """
    diff = original.ravel().astype(np.float64) - reconstructed.ravel().astype(np.float64)
    return float(np.sqrt(np.mean(diff ** 2)))
