"""
TritPack NumPy backend — save/load compressed tensors to disk.

Provides a simple file format for persisting :class:`QuantizedTensor` objects
as ``.tritpack`` files using NumPy's ``npz`` format.
"""

from __future__ import annotations

import io
import struct
from pathlib import Path

import numpy as np

from tritpack.core.quantizer import QuantizedTensor


_MAGIC = b"TPCK"
_VERSION = 1


def save(qt: QuantizedTensor, path: str | Path) -> None:
    """
    Save a :class:`QuantizedTensor` to *path* (``.tritpack`` file).

    Parameters
    ----------
    qt : QuantizedTensor
    path : str | Path
    """
    path = Path(path)
    np.savez_compressed(
        path,
        packed_data=np.frombuffer(qt.packed_data, dtype=np.uint8),
        scales=qt.scales,
        block_offsets=np.array(qt.block_offsets, dtype=np.int64),
        block_lengths=np.array(qt.block_lengths, dtype=np.int64),
        original_shape=np.array(qt.original_shape, dtype=np.int64),
        # Metadata encoded as a 1-element object array for easy retrieval
        meta=np.array(
            [qt.original_dtype.str, qt.block_size, qt.alpha, qt.sparsity],
            dtype=object,
        ),
    )


def load(path: str | Path) -> QuantizedTensor:
    """
    Load a :class:`QuantizedTensor` from *path*.

    Parameters
    ----------
    path : str | Path

    Returns
    -------
    QuantizedTensor
    """
    path = Path(path)
    npz = np.load(path, allow_pickle=True)
    meta = npz["meta"]
    return QuantizedTensor(
        packed_data=bytes(npz["packed_data"].tobytes()),
        scales=npz["scales"],
        block_offsets=list(npz["block_offsets"]),
        block_lengths=list(npz["block_lengths"]),
        original_shape=tuple(int(x) for x in npz["original_shape"]),
        original_dtype=np.dtype(meta[0]),
        block_size=int(meta[1]),
        alpha=float(meta[2]),
        sparsity=float(meta[3]),
    )


def save_tensor(tensor: np.ndarray, path: str | Path, alpha: float = 0.7) -> QuantizedTensor:
    """
    Convenience: quantize *tensor* and save to *path*.

    Returns the :class:`QuantizedTensor` for inspection.
    """
    from tritpack.core.quantizer import TernaryQuantizer

    qt = TernaryQuantizer(alpha=alpha).quantize(tensor)
    save(qt, path)
    return qt


def load_tensor(path: str | Path) -> np.ndarray:
    """
    Convenience: load and dequantize a saved tensor.
    """
    from tritpack.core.dequantizer import TernaryDequantizer

    qt = load(path)
    return TernaryDequantizer().dequantize(qt)
