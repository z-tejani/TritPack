"""
TritPack quantizer — per-block ternary quantization of float tensors.

Each block of BLOCK_SIZE elements is independently quantized to ternary
values {-1, 0, +1} and a single float16 scale factor, then trit-packed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from tritpack.core.packing import pack_trits, unpack_trits

BLOCK_SIZE: int = 64  # default block size (configurable)


@dataclass
class QuantizedTensor:
    """
    Container for a ternary-quantized tensor.

    Attributes
    ----------
    packed_data : bytes
        Trit-packed data for *all* blocks concatenated.
    scales : np.ndarray
        float16 scale factor per block, shape ``(n_blocks,)``.
    block_offsets : list[int]
        Byte offset in *packed_data* where each block starts.
    block_lengths : list[int]
        Number of trits (elements) in each block.
    original_shape : tuple
        Shape of the original tensor.
    original_dtype : np.dtype
        dtype of the original tensor.
    block_size : int
        Nominal number of elements per block.
    alpha : float
        Threshold ratio used during quantization.
    sparsity : float
        Fraction of zero trits across all blocks.
    """

    packed_data: bytes
    scales: np.ndarray          # float16, one per block
    block_offsets: list[int]
    block_lengths: list[int]
    original_shape: tuple
    original_dtype: np.dtype
    block_size: int
    alpha: float
    sparsity: float

    def size_bytes(self) -> int:
        """Total size of the compressed representation in bytes."""
        return (
            len(self.packed_data)
            + self.scales.nbytes
            + len(self.block_offsets) * 8   # 8 bytes each (int64)
            + len(self.block_lengths) * 8
            + 64                             # metadata overhead
        )

    def compression_ratio(self) -> float:
        """
        Ratio of original tensor bytes to compressed representation bytes.

        Values > 1 mean the compressed form is smaller.
        """
        original_bytes = int(np.prod(self.original_shape)) * np.dtype(self.original_dtype).itemsize
        compressed = self.size_bytes()
        if compressed == 0:
            return 1.0
        return original_bytes / compressed


class TernaryQuantizer:
    """
    Per-block ternary quantizer for floating-point tensors.

    Parameters
    ----------
    block_size : int
        Number of elements in each quantization block.
    alpha : float
        Threshold ratio: ``τ = alpha * mean(|block|)``.
        Higher *alpha* → more zeros → higher compression, lower quality.
    """

    def __init__(self, block_size: int = BLOCK_SIZE, alpha: float = 0.7) -> None:
        if block_size < 1:
            raise ValueError(f"block_size must be ≥ 1, got {block_size}")
        if not (0.0 < alpha <= 2.0):
            raise ValueError(f"alpha must be in (0, 2], got {alpha}")
        self.block_size = block_size
        self.alpha = alpha

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def quantize(self, tensor: np.ndarray) -> QuantizedTensor:
        """
        Quantize *tensor* to ternary representation.

        Parameters
        ----------
        tensor : np.ndarray
            Input floating-point tensor of any shape.

        Returns
        -------
        QuantizedTensor
            Compressed representation with packed trits and per-block scales.
        """
        flat = tensor.ravel().astype(np.float32)
        n = len(flat)

        n_blocks = max(1, math.ceil(n / self.block_size))
        scales = np.zeros(n_blocks, dtype=np.float16)
        all_packed: list[bytes] = []
        block_offsets: list[int] = []
        block_lengths: list[int] = []
        zero_count = 0
        total_count = 0
        byte_cursor = 0

        for b in range(n_blocks):
            start = b * self.block_size
            end = min(start + self.block_size, n)
            block = flat[start:end]
            blen = len(block)

            trits, scale = self._quantize_block(block)

            zero_count += int(np.sum(trits == 0))
            total_count += blen

            scales[b] = np.float16(scale)
            packed_block = pack_trits(trits)

            block_offsets.append(byte_cursor)
            block_lengths.append(blen)
            all_packed.append(packed_block)
            byte_cursor += len(packed_block)

        packed_data = b"".join(all_packed)
        sparsity = zero_count / total_count if total_count > 0 else 0.0

        return QuantizedTensor(
            packed_data=packed_data,
            scales=scales,
            block_offsets=block_offsets,
            block_lengths=block_lengths,
            original_shape=tuple(tensor.shape),
            original_dtype=tensor.dtype,
            block_size=self.block_size,
            alpha=self.alpha,
            sparsity=sparsity,
        )

    def estimate_compression_ratio(self, tensor: np.ndarray) -> float:
        """
        Fast estimate of the compression ratio without full quantization.

        Uses a sample of at most 1000 blocks to predict overall sparsity,
        then derives the expected ratio analytically.

        Parameters
        ----------
        tensor : np.ndarray
            Input tensor.

        Returns
        -------
        float
            Estimated compression ratio (original bytes / compressed bytes).
        """
        flat = tensor.ravel().astype(np.float32)
        n = len(flat)
        n_blocks = max(1, math.ceil(n / self.block_size))

        # Sample at most 1000 blocks uniformly
        sample_size = min(n_blocks, 1000)
        sample_indices = np.random.choice(n_blocks, size=sample_size, replace=False)

        zero_count = 0
        total_sampled = 0
        for b in sample_indices:
            start = b * self.block_size
            end = min(start + self.block_size, n)
            block = flat[start:end]
            trits, _ = self._quantize_block(block)
            zero_count += int(np.sum(trits == 0))
            total_sampled += len(block)

        sparsity = zero_count / total_sampled if total_sampled > 0 else 0.0

        # Packed: ~n/5 bytes for trits + n_blocks * 2 bytes for float16 scales
        packed_bytes = math.ceil(n / 5) + n_blocks * 2
        original_bytes = n * np.dtype(tensor.dtype).itemsize
        if packed_bytes == 0:
            return 1.0
        return original_bytes / packed_bytes

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _quantize_block(self, block: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Quantize a single block.

        Parameters
        ----------
        block : np.ndarray
            1-D float32 array.

        Returns
        -------
        tuple[np.ndarray, float]
            ``(trits, scale)`` where *trits* is int8 with values in
            {-1, 0, +1} and *scale* is the mean absolute value of
            non-zero elements (float32).
        """
        abs_block = np.abs(block)
        mean_abs = float(np.mean(abs_block))
        tau = self.alpha * mean_abs

        trits = np.zeros(len(block), dtype=np.int8)
        trits[block > tau] = 1
        trits[block < -tau] = -1

        nonzero_mask = trits != 0
        if nonzero_mask.any():
            scale = float(np.mean(abs_block[nonzero_mask]))
        else:
            scale = 1.0

        return trits, scale
