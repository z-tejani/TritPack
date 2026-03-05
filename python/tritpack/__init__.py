"""
TritPack: Balanced ternary memory compression for LLMs.

Achieves ~3x RAM reduction by compressing model weights and KV cache
to ternary values {-1, 0, +1}.
"""

from tritpack.core.packing import pack_trits, unpack_trits, pack_trits_batch, unpack_trits_batch
from tritpack.core.quantizer import TernaryQuantizer, QuantizedTensor
from tritpack.core.dequantizer import TernaryDequantizer
from tritpack.core.tiers import TierManager
from tritpack.model.tensor import TritTensor

__version__ = "0.1.0"
__all__ = [
    "pack_trits",
    "unpack_trits",
    "pack_trits_batch",
    "unpack_trits_batch",
    "TernaryQuantizer",
    "QuantizedTensor",
    "TernaryDequantizer",
    "TierManager",
    "TritTensor",
]
