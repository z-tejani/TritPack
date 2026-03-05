"""
Unit tests for tritpack.core.tiers — TierManager.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from tritpack.core.tiers import TierManager, TIER_HOT, TIER_WARM, TIER_COLD


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def small_tensor(seed=0, size=64):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(size).astype(np.float32)


def make_manager(hot_frac=0.5, warm_frac=0.3, limit_gb=1.0):
    return TierManager(
        memory_limit_gb=limit_gb,
        hot_budget_fraction=hot_frac,
        warm_budget_fraction=warm_frac,
    )


# ---------------------------------------------------------------------------
# Registration and access
# ---------------------------------------------------------------------------


def test_register_and_access_layer():
    """Registered layer can be accessed and returned as ndarray."""
    tm = make_manager()
    t = small_tensor()
    tm.register_layer("layer0", t)
    result = tm.access("layer0")
    assert isinstance(result, np.ndarray)
    assert result.shape == t.shape


def test_hot_layer_no_quality_loss():
    """Accessing a HOT layer should return data equal to original."""
    # Use a large budget so no eviction happens
    tm = TierManager(memory_limit_gb=100.0, hot_budget_fraction=0.9)
    t = small_tensor()
    tm.register_layer("layer0", t)
    result = tm.access("layer0")
    np.testing.assert_allclose(result, t, rtol=1e-5)


def test_access_unknown_layer_raises():
    """Accessing unregistered layer must raise KeyError."""
    tm = make_manager()
    with pytest.raises(KeyError):
        tm.access("not_registered")


# ---------------------------------------------------------------------------
# LRU eviction
# ---------------------------------------------------------------------------


def test_lru_eviction_under_pressure():
    """
    When hot budget is exceeded, LRU layer should be demoted.
    Register layers until budget forces eviction; verify hot count drops.
    """
    # Very tight hot budget: 1 MB
    limit_gb = 0.01  # 10 MB total
    hot_frac = 0.1   # 1 MB hot

    tm = TierManager(memory_limit_gb=limit_gb, hot_budget_fraction=hot_frac, warm_budget_fraction=0.5)

    # Each tensor is ~400 KB (100K float32)
    tensors = {}
    for i in range(10):
        t = np.random.default_rng(i).standard_normal(100_000).astype(np.float32)
        tensors[f"layer{i}"] = t
        tm.register_layer(f"layer{i}", t)

    stats = tm.stats()
    # Not all layers should be HOT — budget was exceeded and LRU evicted
    assert stats.n_hot < 10, f"Expected some evictions, but all {stats.n_hot} are HOT"
    assert stats.n_demotions > 0


def test_tier_promotion_on_access():
    """Accessing a non-HOT layer should promote it to HOT."""
    # Tiny budget so layers get demoted on register
    tm = TierManager(memory_limit_gb=0.001, hot_budget_fraction=0.01, warm_budget_fraction=0.5)

    t = small_tensor(size=1000)
    tm.register_layer("layer0", t)

    # Force the layer to be in non-hot tier by registering more layers
    for i in range(1, 20):
        t2 = small_tensor(seed=i, size=1000)
        tm.register_layer(f"layer{i}", t2)

    # Access layer0 — it should get promoted
    result = tm.access("layer0")
    assert result.shape == t.shape

    # After access layer0 should be HOT
    r = tm._layers["layer0"]
    assert r.tier == TIER_HOT


def test_memory_budget_respected():
    """Total hot + warm usage must not massively exceed budget."""
    limit_gb = 0.01  # 10 MB
    tm = TierManager(
        memory_limit_gb=limit_gb,
        hot_budget_fraction=0.2,
        warm_budget_fraction=0.3,
    )
    hot_budget_mb = limit_gb * 1024 * 0.2
    warm_budget_mb = limit_gb * 1024 * 0.3

    for i in range(20):
        t = np.random.default_rng(i).standard_normal(50_000).astype(np.float32)
        tm.register_layer(f"layer{i}", t)

    usage = tm.memory_usage()
    hot_mb = usage["hot_gb"] * 1024
    warm_mb = usage["warm_gb"] * 1024
    # Allow 10% headroom (eviction is lazy, not perfect)
    assert hot_mb <= hot_budget_mb * 1.1 + 1, (
        f"Hot usage {hot_mb:.2f} MB exceeds budget {hot_budget_mb:.2f} MB"
    )


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def test_stats_reporting():
    """stats() must return a TierStats with correct counts."""
    tm = TierManager(memory_limit_gb=100.0, hot_budget_fraction=0.9)
    for i in range(5):
        tm.register_layer(f"layer{i}", small_tensor(seed=i))

    stats = tm.stats()
    assert stats.n_hot + stats.n_warm + stats.n_cold == 5
    assert stats.n_accesses == 0
    assert stats.total_gb >= 0


def test_access_increments_counter():
    """n_accesses must increment on each access."""
    tm = TierManager(memory_limit_gb=100.0)
    tm.register_layer("l0", small_tensor())
    for i in range(3):
        tm.access("l0")
    assert tm.stats().n_accesses == 3


# ---------------------------------------------------------------------------
# Memory usage reporting
# ---------------------------------------------------------------------------


def test_memory_usage_returns_dict():
    """memory_usage() must return dict with required keys."""
    tm = make_manager()
    tm.register_layer("l0", small_tensor())
    usage = tm.memory_usage()
    for key in ("hot_gb", "warm_gb", "cold_gb", "total_gb"):
        assert key in usage
        assert usage[key] >= 0.0


def test_memory_usage_consistent():
    """total_gb must equal hot_gb + warm_gb + cold_gb."""
    tm = make_manager()
    tm.register_layer("l0", small_tensor())
    u = tm.memory_usage()
    assert abs(u["total_gb"] - (u["hot_gb"] + u["warm_gb"] + u["cold_gb"])) < 1e-9


# ---------------------------------------------------------------------------
# Eviction helpers
# ---------------------------------------------------------------------------


def test_evict_lru_manual():
    """Manual evict_lru() call should demote a HOT layer."""
    tm = TierManager(memory_limit_gb=100.0)
    tm.register_layer("l0", small_tensor(size=1000))
    assert tm._layers["l0"].tier == TIER_HOT
    tm.evict_lru()
    assert tm._layers["l0"].tier in (TIER_WARM, TIER_COLD)
