"""
Unit tests for tritpack.core.packing — 100% coverage target.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays

from tritpack.core.packing import (
    TRITS_PER_BYTE,
    pack_trits,
    pack_trits_batch,
    pack_trits_vectorized,
    unpack_trits,
    unpack_trits_batch,
    unpack_trits_vectorized,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def random_trits(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(-1, 2, size=n, dtype=np.int8)


def assert_valid_trits(arr: np.ndarray) -> None:
    assert set(np.unique(arr)).issubset({-1, 0, 1}), f"Invalid trit values: {np.unique(arr)}"


# ---------------------------------------------------------------------------
# Roundtrip tests
# ---------------------------------------------------------------------------


def test_pack_unpack_roundtrip_small():
    """Small array roundtrip must be exact."""
    trits = np.array([-1, 0, 1, 0, -1], dtype=np.int8)
    packed = pack_trits(trits)
    recovered = unpack_trits(packed, len(trits))
    np.testing.assert_array_equal(recovered, trits)


def test_pack_unpack_roundtrip_large():
    """Roundtrip on 1M trits must be exact."""
    trits = random_trits(1_000_000)
    packed = pack_trits(trits)
    recovered = unpack_trits(packed, len(trits))
    np.testing.assert_array_equal(recovered, trits)


def test_pack_padding_correctness_non_multiple_of_5():
    """Non-multiples of 5 must roundtrip without corruption."""
    for n in [1, 2, 3, 4, 6, 7, 8, 9, 11, 99, 1001]:
        trits = random_trits(n, seed=n)
        packed = pack_trits(trits)
        recovered = unpack_trits(packed, n)
        np.testing.assert_array_equal(recovered, trits, err_msg=f"Failed for n={n}")


def test_all_trit_values_in_all_positions():
    """Every trit value (-1, 0, +1) in every position within a 5-trit group."""
    for val in (-1, 0, 1):
        for pos in range(TRITS_PER_BYTE):
            trits = np.zeros(TRITS_PER_BYTE, dtype=np.int8)
            trits[pos] = val
            packed = pack_trits(trits)
            recovered = unpack_trits(packed, TRITS_PER_BYTE)
            np.testing.assert_array_equal(recovered, trits)


def test_pack_empty_array():
    """Packing an empty array must return an 8-byte header with zero payload."""
    trits = np.array([], dtype=np.int8)
    packed = pack_trits(trits)
    # Header encodes original length = 0
    length = struct.unpack("<Q", packed[:8])[0]
    assert length == 0
    recovered = unpack_trits(packed, 0)
    assert recovered.shape == (0,)


def test_pack_single_trit():
    """Single trit must survive a roundtrip."""
    for val in (-1, 0, 1):
        trits = np.array([val], dtype=np.int8)
        packed = pack_trits(trits)
        recovered = unpack_trits(packed, 1)
        np.testing.assert_array_equal(recovered, trits)


# ---------------------------------------------------------------------------
# Vectorized vs reference
# ---------------------------------------------------------------------------


def test_vectorized_matches_reference():
    """NumPy vectorised implementation must match reference loop."""
    trits = random_trits(10_000, seed=7)
    ref_packed = pack_trits(trits)
    vec_packed = pack_trits_vectorized(trits)

    # The vectorised version returns a uint8 array (no header).
    # Compare payloads only (skip 8-byte header of reference).
    ref_payload = np.frombuffer(ref_packed[8:], dtype=np.uint8)
    np.testing.assert_array_equal(ref_payload, vec_packed)


def test_vectorized_unpack_matches_reference():
    """Vectorised unpack must produce identical result to reference unpack."""
    trits = random_trits(997, seed=13)
    vec_packed = pack_trits_vectorized(trits)
    ref_packed = pack_trits(trits)

    ref_result = unpack_trits(ref_packed, len(trits))
    vec_result = unpack_trits_vectorized(vec_packed, len(trits))
    np.testing.assert_array_equal(ref_result, vec_result)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_deterministic():
    """Same input must always produce same output."""
    trits = random_trits(500, seed=99)
    packed_a = pack_trits(trits)
    packed_b = pack_trits(trits)
    assert packed_a == packed_b


# ---------------------------------------------------------------------------
# Batch API
# ---------------------------------------------------------------------------


def test_pack_trits_batch_roundtrip():
    """pack_trits_batch / unpack_trits_batch must be inverses."""
    tensor = random_trits(1000, seed=3).reshape(10, 100)
    packed, count = pack_trits_batch(tensor)
    recovered = unpack_trits_batch(packed, tensor.shape, tensor.dtype)
    np.testing.assert_array_equal(recovered, tensor)


def test_batch_preserves_shape():
    """unpack_trits_batch must restore original shape."""
    for shape in [(5,), (4, 5), (2, 3, 7)]:
        n = int(np.prod(shape))
        tensor = random_trits(n, seed=0).reshape(shape)
        packed, count = pack_trits_batch(tensor)
        recovered = unpack_trits_batch(packed, shape, np.int8)
        assert recovered.shape == shape


# ---------------------------------------------------------------------------
# Header correctness
# ---------------------------------------------------------------------------


def test_header_encodes_original_length():
    """The 8-byte header must encode the exact original array length."""
    for n in [0, 1, 5, 6, 100, 999, 100_000]:
        trits = random_trits(n, seed=n)
        packed = pack_trits(trits)
        stored_len = struct.unpack("<Q", packed[:8])[0]
        assert stored_len == n, f"Header mismatch for n={n}: got {stored_len}"


# ---------------------------------------------------------------------------
# Byte range
# ---------------------------------------------------------------------------


def test_packed_bytes_in_valid_range():
    """All packed bytes must be in [0, 242] (3^5 - 1 = 242)."""
    trits = random_trits(1000, seed=5)
    packed = pack_trits(trits)
    payload = np.frombuffer(packed[8:], dtype=np.uint8)
    assert int(payload.max()) <= 242, "Packed byte out of base-3 range"


# ---------------------------------------------------------------------------
# Property-based tests (Hypothesis)
# ---------------------------------------------------------------------------

_trit_arrays = arrays(
    dtype=np.int8,
    shape=st.integers(0, 500),
    elements=st.integers(-1, 1),
)


@given(_trit_arrays)
@settings(max_examples=300, deadline=None)
def test_pack_unpack_exact_inverse_property(trits: np.ndarray):
    """pack then unpack must be the exact identity for any trit array."""
    packed = pack_trits(trits)
    recovered = unpack_trits(packed, len(trits))
    np.testing.assert_array_equal(recovered, trits)


@given(_trit_arrays)
@settings(max_examples=200, deadline=None)
def test_vectorized_matches_reference_property(trits: np.ndarray):
    """Vectorised and reference packing must produce identical payloads."""
    ref = pack_trits(trits)
    vec = pack_trits_vectorized(trits)
    ref_payload = np.frombuffer(ref[8:], dtype=np.uint8)
    np.testing.assert_array_equal(ref_payload, vec)
