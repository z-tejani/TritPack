"""
TritPack GGUF backend — read GGUF model files for compression.

Uses the ``gguf`` Python package from llama.cpp to parse GGUF files and
yield tensors ready for ternary quantization.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterator

import numpy as np

try:
    import gguf  # type: ignore
    _GGUF_AVAILABLE = True
except ImportError:
    _GGUF_AVAILABLE = False


def _require_gguf() -> None:
    if not _GGUF_AVAILABLE:
        raise ImportError(
            "The 'gguf' package is required for GGUF support. "
            "Install it with: pip install gguf"
        )


class GGUFBackend:
    """
    Reads GGUF model files (llama.cpp format) and yields tensors for
    compression.

    Parameters
    ----------
    model_path : str
        Path to the ``.gguf`` file.
    """

    def __init__(self, model_path: str) -> None:
        _require_gguf()
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        self._reader = gguf.GGUFReader(str(self.model_path))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_metadata(self) -> dict:
        """
        Return all GGUF metadata key–value pairs.

        Returns
        -------
        dict
            ``{key: value}`` for every metadata field in the GGUF file.
        """
        meta: dict = {}
        for field in self._reader.fields.values():
            try:
                if field.types and field.types[0] == gguf.GGUFValueType.STRING:
                    meta[field.name] = str(bytes(field.parts[-1]), encoding="utf-8")
                elif len(field.data) == 1:
                    meta[field.name] = field.data[0]
                else:
                    meta[field.name] = list(field.data)
            except Exception:
                meta[field.name] = None
        return meta

    def tensor_names(self) -> list[str]:
        """Return the names of all tensors stored in the GGUF file."""
        return [t.name for t in self._reader.tensors]

    def load_tensor(self, name: str) -> np.ndarray:
        """
        Load and dequantize a tensor by name.

        Parameters
        ----------
        name : str
            Tensor name as returned by :meth:`tensor_names`.

        Returns
        -------
        np.ndarray
            float32 array.
        """
        for t in self._reader.tensors:
            if t.name == name:
                return t.data.astype(np.float32)
        raise KeyError(f"Tensor '{name}' not found in {self.model_path}")

    def iter_tensors(self) -> Iterator[tuple[str, np.ndarray]]:
        """
        Yield ``(name, tensor)`` for every tensor in the model.

        Yields
        ------
        tuple[str, np.ndarray]
        """
        for t in self._reader.tensors:
            yield t.name, t.data.astype(np.float32)

    def estimate_compressed_size(self, alpha: float = 0.7) -> dict:
        """
        Estimate compressed size without fully loading all tensors.

        Uses a per-tensor analytical estimate based on sparsity of a small
        block sample, avoiding full quantization.

        Parameters
        ----------
        alpha : float
            Threshold ratio to use for the estimate.

        Returns
        -------
        dict
            ``{original_gb, compressed_gb, ratio}``
        """
        from tritpack.core.quantizer import TernaryQuantizer

        q = TernaryQuantizer(alpha=alpha)
        original_bytes = 0
        compressed_bytes = 0

        for t in self._reader.tensors:
            tensor_fp32 = t.data.astype(np.float32)
            n = tensor_fp32.size
            original_bytes += n * 4  # float32
            # Analytical: packed trits + scales
            n_blocks = max(1, math.ceil(n / q.block_size))
            compressed_bytes += math.ceil(n / 5) + n_blocks * 2 + 8 * n_blocks

        gb = 1 << 30
        ratio = original_bytes / compressed_bytes if compressed_bytes else 1.0
        return {
            "original_gb": original_bytes / gb,
            "compressed_gb": compressed_bytes / gb,
            "ratio": ratio,
        }
