"""Layer-level compression manager for TritPack."""

from __future__ import annotations

import numpy as np

from tritpack.core.quantizer import TernaryQuantizer, QuantizedTensor
from tritpack.core.dequantizer import TernaryDequantizer
from tritpack.model.tensor import TritTensor


class LayerCompressor:
    """
    Manages compression/decompression of all tensors in a model layer.

    Attributes
    ----------
    layer_id : str
        Unique identifier for this layer.
    quantizer : TernaryQuantizer
        Quantizer used to compress tensors.
    """

    def __init__(self, layer_id: str, quantizer: TernaryQuantizer | None = None) -> None:
        self.layer_id = layer_id
        self.quantizer = quantizer or TernaryQuantizer()
        self._tensors: dict[str, TritTensor] = {}

    def add_tensor(self, name: str, tensor: np.ndarray) -> TritTensor:
        """Compress and store a tensor under *name*."""
        tt = TritTensor(tensor, self.quantizer)
        self._tensors[name] = tt
        return tt

    def get_tensor(self, name: str) -> np.ndarray:
        """Return decompressed tensor for *name*."""
        return self._tensors[name].numpy()

    def tensor_names(self) -> list[str]:
        return list(self._tensors.keys())

    def memory_usage(self) -> dict[str, int]:
        """Return {name: compressed_bytes} for every stored tensor."""
        return {name: tt.nbytes_compressed for name, tt in self._tensors.items()}

    def total_compressed_bytes(self) -> int:
        return sum(self.memory_usage().values())
