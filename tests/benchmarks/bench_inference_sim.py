"""
Simulated inference benchmark: tier access latency.
"""

from __future__ import annotations

import numpy as np
import pytest

from tritpack.core.tiers import TierManager


def gaussian(shape, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape).astype(np.float32)


@pytest.fixture
def tier_manager_hot():
    """TierManager with layers in HOT tier."""
    tm = TierManager(memory_limit_gb=100.0, hot_budget_fraction=0.99)
    tm.register_layer("l0", gaussian((10_000,)))
    return tm


@pytest.fixture
def tier_manager_cold():
    """TierManager with layers forced to COLD tier."""
    tm = TierManager(memory_limit_gb=0.0001, hot_budget_fraction=0.01, warm_budget_fraction=0.01)
    tm.register_layer("l0", gaussian((10_000,)))
    # Register many more to force eviction to COLD
    for i in range(1, 30):
        tm.register_layer(f"l{i}", gaussian((10_000,), seed=i))
    return tm


def test_bench_tier_access_hot(benchmark, tier_manager_hot):
    """Access latency for a HOT-tier layer."""
    benchmark(tier_manager_hot.access, "l0")


def test_bench_tier_access_cold(benchmark, tier_manager_cold):
    """Access latency for a COLD-tier layer."""
    # First ensure l0 is cold by accessing newer layers
    for i in range(1, 30):
        tier_manager_cold.access(f"l{i}")
    benchmark(tier_manager_cold.access, "l0")
