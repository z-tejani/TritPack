"""
TritPack tiered memory manager — mirrors CPU cache hierarchy.

Three tiers manage LLM layer weights by compression level:

  HOT  (Tier 0): Raw FP16/FP32.  Active computation.  No overhead.
  WARM (Tier 1): INT8 quantized.  ~2x compression.  <1 ms decompress.
  COLD (Tier 2): Ternary packed.  ~3x compression.  <5 ms decompress.

Layers are promoted on access and demoted via LRU when memory pressure
exceeds configured budgets.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from tritpack.core.quantizer import TernaryQuantizer
from tritpack.core.dequantizer import TernaryDequantizer


# ---------------------------------------------------------------------------
# Tier constants
# ---------------------------------------------------------------------------

TIER_HOT = 0
TIER_WARM = 1
TIER_COLD = 2

_BYTES_PER_GB = 1 << 30


# ---------------------------------------------------------------------------
# Internal layer record
# ---------------------------------------------------------------------------


@dataclass
class _LayerRecord:
    layer_id: str
    # Per-tier storage (only one tier is active at a time)
    hot_data: Optional[np.ndarray] = None       # FP16/FP32 original
    warm_data: Optional[np.ndarray] = None      # int8 quantized values
    warm_scale: Optional[np.ndarray] = None     # per-block scale for int8
    cold_qt: object = None                       # QuantizedTensor (ternary)

    original_shape: tuple = field(default_factory=tuple)
    original_dtype: np.dtype = field(default_factory=lambda: np.dtype("float32"))
    tier: int = TIER_HOT
    last_access: float = field(default_factory=time.monotonic)

    def hot_bytes(self) -> int:
        return self.hot_data.nbytes if self.hot_data is not None else 0

    def warm_bytes(self) -> int:
        if self.warm_data is None:
            return 0
        return self.warm_data.nbytes + (self.warm_scale.nbytes if self.warm_scale is not None else 0)

    def cold_bytes(self) -> int:
        if self.cold_qt is None:
            return 0
        return self.cold_qt.size_bytes()


# ---------------------------------------------------------------------------
# TierStats
# ---------------------------------------------------------------------------


@dataclass
class TierStats:
    n_hot: int
    n_warm: int
    n_cold: int
    hot_gb: float
    warm_gb: float
    cold_gb: float
    total_gb: float
    n_promotions: int
    n_demotions: int
    n_accesses: int


# ---------------------------------------------------------------------------
# TierManager
# ---------------------------------------------------------------------------

_WARM_BLOCK_SIZE = 64


class TierManager:
    """
    Manages three compression tiers for LLM layer weights.

    Parameters
    ----------
    memory_limit_gb : float
        Total RAM budget in gigabytes.
    hot_budget_fraction : float
        Fraction of *memory_limit_gb* allocated to HOT tier.
    warm_budget_fraction : float
        Fraction of *memory_limit_gb* allocated to WARM tier.
    """

    def __init__(
        self,
        memory_limit_gb: float,
        hot_budget_fraction: float = 0.15,
        warm_budget_fraction: float = 0.25,
    ) -> None:
        if memory_limit_gb <= 0:
            raise ValueError("memory_limit_gb must be positive")
        self.memory_limit_bytes = int(memory_limit_gb * _BYTES_PER_GB)
        self.hot_budget = int(hot_budget_fraction * self.memory_limit_bytes)
        self.warm_budget = int(warm_budget_fraction * self.memory_limit_bytes)

        self._layers: dict[str, _LayerRecord] = {}
        self._quantizer = TernaryQuantizer()
        self._dequantizer = TernaryDequantizer()

        self._n_promotions = 0
        self._n_demotions = 0
        self._n_accesses = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_layer(self, layer_id: str, tensor: np.ndarray) -> None:
        """
        Register a new layer, storing it in HOT tier initially.

        Parameters
        ----------
        layer_id : str
            Unique identifier for this layer.
        tensor : np.ndarray
            Original weight tensor (any floating-point dtype).
        """
        record = _LayerRecord(
            layer_id=layer_id,
            hot_data=tensor.copy(),
            original_shape=tuple(tensor.shape),
            original_dtype=tensor.dtype,
            tier=TIER_HOT,
            last_access=time.monotonic(),
        )
        self._layers[layer_id] = record
        self._enforce_budgets()

    def access(self, layer_id: str) -> np.ndarray:
        """
        Access a layer, decompressing from whichever tier it resides in.

        On access the layer is promoted to HOT and budgets are enforced.

        Parameters
        ----------
        layer_id : str
            Layer to access.

        Returns
        -------
        np.ndarray
            Decompressed float tensor matching the original shape/dtype.
        """
        if layer_id not in self._layers:
            raise KeyError(f"Layer '{layer_id}' not registered")

        self._n_accesses += 1
        record = self._layers[layer_id]
        record.last_access = time.monotonic()

        # Decompress if needed, then promote to HOT
        if record.tier == TIER_HOT:
            return record.hot_data.copy()

        if record.tier == TIER_WARM:
            data = self._warm_to_fp(record)
        else:  # TIER_COLD
            data = self._cold_to_fp(record)

        # Promote to HOT
        self._promote_to_hot(record, data)
        self._enforce_budgets()
        return data.copy()

    def evict_lru(self) -> None:
        """
        Evict the least-recently-used HOT layer to WARM,
        or the LRU WARM layer to COLD.

        Called automatically by :meth:`_enforce_budgets`.
        """
        # Try to demote a HOT layer to WARM
        hot_layers = [r for r in self._layers.values() if r.tier == TIER_HOT]
        if hot_layers:
            lru = min(hot_layers, key=lambda r: r.last_access)
            self._demote_to_warm(lru)
            return

        # Otherwise demote a WARM layer to COLD
        warm_layers = [r for r in self._layers.values() if r.tier == TIER_WARM]
        if warm_layers:
            lru = min(warm_layers, key=lambda r: r.last_access)
            self._demote_to_cold(lru)

    def memory_usage(self) -> dict[str, float]:
        """
        Return current memory usage in GB per tier.

        Returns
        -------
        dict[str, float]
            Keys: ``hot_gb``, ``warm_gb``, ``cold_gb``, ``total_gb``.
        """
        hot_b = sum(r.hot_bytes() for r in self._layers.values())
        warm_b = sum(r.warm_bytes() for r in self._layers.values())
        cold_b = sum(r.cold_bytes() for r in self._layers.values())
        total_b = hot_b + warm_b + cold_b
        gb = _BYTES_PER_GB
        return {
            "hot_gb": hot_b / gb,
            "warm_gb": warm_b / gb,
            "cold_gb": cold_b / gb,
            "total_gb": total_b / gb,
        }

    def stats(self) -> TierStats:
        """Return a :class:`TierStats` snapshot."""
        usage = self.memory_usage()
        return TierStats(
            n_hot=sum(1 for r in self._layers.values() if r.tier == TIER_HOT),
            n_warm=sum(1 for r in self._layers.values() if r.tier == TIER_WARM),
            n_cold=sum(1 for r in self._layers.values() if r.tier == TIER_COLD),
            hot_gb=usage["hot_gb"],
            warm_gb=usage["warm_gb"],
            cold_gb=usage["cold_gb"],
            total_gb=usage["total_gb"],
            n_promotions=self._n_promotions,
            n_demotions=self._n_demotions,
            n_accesses=self._n_accesses,
        )

    # ------------------------------------------------------------------
    # Internal: budget enforcement
    # ------------------------------------------------------------------

    def _enforce_budgets(self) -> None:
        """Evict layers until all tier budgets are satisfied."""
        # Enforce HOT budget
        while self._hot_bytes() > self.hot_budget and self._has_demotable_hot():
            hot_layers = [r for r in self._layers.values() if r.tier == TIER_HOT]
            lru = min(hot_layers, key=lambda r: r.last_access)
            self._demote_to_warm(lru)

        # Enforce HOT + WARM combined budget
        combined_budget = self.hot_budget + self.warm_budget
        while self._hot_bytes() + self._warm_bytes() > combined_budget and self._has_demotable_warm():
            warm_layers = [r for r in self._layers.values() if r.tier == TIER_WARM]
            lru = min(warm_layers, key=lambda r: r.last_access)
            self._demote_to_cold(lru)

    def _has_demotable_hot(self) -> bool:
        return any(r.tier == TIER_HOT for r in self._layers.values())

    def _has_demotable_warm(self) -> bool:
        return any(r.tier == TIER_WARM for r in self._layers.values())

    def _hot_bytes(self) -> int:
        return sum(r.hot_bytes() for r in self._layers.values())

    def _warm_bytes(self) -> int:
        return sum(r.warm_bytes() for r in self._layers.values())

    # ------------------------------------------------------------------
    # Internal: tier transitions
    # ------------------------------------------------------------------

    def _promote_to_hot(self, record: _LayerRecord, data: np.ndarray) -> None:
        """Promote *record* to HOT tier, clearing lower-tier data."""
        record.hot_data = data
        record.warm_data = None
        record.warm_scale = None
        record.cold_qt = None
        record.tier = TIER_HOT
        self._n_promotions += 1

    def _demote_to_warm(self, record: _LayerRecord) -> None:
        """Demote *record* from HOT to WARM (int8 quantization)."""
        data = record.hot_data.astype(np.float32)
        # Per-block int8 quantization: scale = max(|block|), quant = round(data/scale * 127)
        flat = data.ravel()
        n = len(flat)
        n_blocks = max(1, (n + _WARM_BLOCK_SIZE - 1) // _WARM_BLOCK_SIZE)
        quant = np.zeros(n, dtype=np.int8)
        scale = np.zeros(n_blocks, dtype=np.float16)
        for b in range(n_blocks):
            start = b * _WARM_BLOCK_SIZE
            end = min(start + _WARM_BLOCK_SIZE, n)
            block = flat[start:end]
            max_abs = float(np.max(np.abs(block)))
            if max_abs == 0:
                max_abs = 1.0
            scale[b] = np.float16(max_abs / 127.0)
            quant[start:end] = np.clip(np.round(block / float(scale[b])), -127, 127).astype(np.int8)

        record.warm_data = quant.reshape(record.original_shape)
        record.warm_scale = scale
        record.hot_data = None
        record.tier = TIER_WARM
        self._n_demotions += 1

    def _demote_to_cold(self, record: _LayerRecord) -> None:
        """Demote *record* from WARM to COLD (ternary packing)."""
        data = self._warm_to_fp(record)
        qt = self._quantizer.quantize(data)
        record.cold_qt = qt
        record.warm_data = None
        record.warm_scale = None
        record.tier = TIER_COLD
        self._n_demotions += 1

    # ------------------------------------------------------------------
    # Internal: decompression helpers
    # ------------------------------------------------------------------

    def _warm_to_fp(self, record: _LayerRecord) -> np.ndarray:
        """Reconstruct float tensor from WARM (int8) tier."""
        flat = record.warm_data.ravel().astype(np.float32)
        n = len(flat)
        result = np.zeros(n, dtype=np.float32)
        for b, s in enumerate(record.warm_scale):
            start = b * _WARM_BLOCK_SIZE
            end = min(start + _WARM_BLOCK_SIZE, n)
            result[start:end] = flat[start:end] * float(s)
        return result.reshape(record.original_shape).astype(record.original_dtype)

    def _cold_to_fp(self, record: _LayerRecord) -> np.ndarray:
        """Reconstruct float tensor from COLD (ternary) tier."""
        return self._dequantizer.dequantize(record.cold_qt)
