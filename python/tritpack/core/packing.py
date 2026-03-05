"""
TritPack packing engine — base-3 encoding of ternary values.

Encodes 5 trits per byte (3^5 = 243 ≤ 256), giving ~1.6 bits per trit
versus 8 bits for int8.  The encoding is exact and invertible.

Encoding convention
-------------------
Trit values {-1, 0, +1} are shifted to digits {0, 1, 2} before packing:

    byte = d[0]*81 + d[1]*27 + d[2]*9 + d[3]*3 + d[4]

where d[i] = trit[i] + 1.

Decoding reverses the process: extract base-3 digits, subtract 1.
"""

from __future__ import annotations

import struct
from typing import Tuple

import numpy as np

# How many trits are packed per byte.
TRITS_PER_BYTE: int = 5
# Powers of 3 for positions [0..4]
_POW3: Tuple[int, ...] = (81, 27, 9, 3, 1)


# ---------------------------------------------------------------------------
# Reference (loop-based) implementation
# ---------------------------------------------------------------------------


def pack_trits(trits: np.ndarray) -> bytes:
    """
    Pack an array of trits {-1, 0, +1} into bytes using base-3 encoding.

    Parameters
    ----------
    trits : np.ndarray
        Integer array with values in {-1, 0, +1} and any shape.
        Flattened internally before packing.

    Returns
    -------
    bytes
        Packed byte string containing: 8-byte little-endian uint64 original
        length, followed by ceil(n/5) packed bytes.

    Notes
    -----
    5 trits fit into one byte because 3^5 = 243 ≤ 256.
    Padding trits (value 0) are appended when ``len(trits)`` is not a
    multiple of 5; the original length is stored so :func:`unpack_trits`
    can strip padding exactly.
    """
    flat = np.asarray(trits, dtype=np.int8).ravel()
    n = len(flat)

    # Pad to multiple of 5
    remainder = n % TRITS_PER_BYTE
    if remainder:
        pad = TRITS_PER_BYTE - remainder
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.int8)])

    n_bytes = len(flat) // TRITS_PER_BYTE
    packed = bytearray(n_bytes)

    for i in range(n_bytes):
        offset = i * TRITS_PER_BYTE
        byte_val = 0
        for j in range(TRITS_PER_BYTE):
            digit = int(flat[offset + j]) + 1  # shift -1→0, 0→1, 1→2
            byte_val += digit * _POW3[j]
        packed[i] = byte_val

    # Prepend original length as little-endian uint64
    header = struct.pack("<Q", n)
    return header + bytes(packed)


def unpack_trits(data: bytes, original_length: int) -> np.ndarray:
    """
    Unpack bytes produced by :func:`pack_trits` into a trit array.

    Parameters
    ----------
    data : bytes
        Raw bytes as produced by :func:`pack_trits` (header + payload).
    original_length : int
        Number of trits in the original array (used to strip padding).

    Returns
    -------
    np.ndarray
        ``int8`` ndarray of shape ``(original_length,)`` with values in
        {-1, 0, +1}.
    """
    # Skip header (8 bytes) — caller already knows original_length
    payload = data[8:]
    n_bytes = len(payload)
    total_trits = n_bytes * TRITS_PER_BYTE
    result = np.empty(total_trits, dtype=np.int8)

    for i in range(n_bytes):
        byte_val = payload[i]
        for j in range(TRITS_PER_BYTE - 1, -1, -1):
            # j counts down 4,3,2,1,0 extracting d[4],d[3],...,d[0].
            # We want d[j] at position j, so index directly by j.
            result[i * TRITS_PER_BYTE + j] = (byte_val % 3) - 1
            byte_val //= 3

    return result[:original_length]


# ---------------------------------------------------------------------------
# Vectorized NumPy implementation (faster for large arrays)
# ---------------------------------------------------------------------------


def pack_trits_vectorized(trits: np.ndarray) -> np.ndarray:
    """
    Pack trits to bytes using vectorised NumPy operations.

    Returns a ``uint8`` ndarray (no length header).  Use this for
    benchmarking against the reference loop implementation.

    Parameters
    ----------
    trits : np.ndarray
        Integer array with values in {-1, 0, +1}.

    Returns
    -------
    np.ndarray
        ``uint8`` array of length ``ceil(n/5)``.
    """
    flat = np.asarray(trits, dtype=np.int8).ravel()
    n = len(flat)
    remainder = n % TRITS_PER_BYTE
    if remainder:
        pad = TRITS_PER_BYTE - remainder
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.int8)])

    # digits shape: (n_bytes, 5)
    digits = (flat.reshape(-1, TRITS_PER_BYTE) + 1).astype(np.uint8)
    weights = np.array(_POW3, dtype=np.uint8)
    packed = (digits * weights).sum(axis=1).astype(np.uint8)
    return packed


def unpack_trits_vectorized(packed: np.ndarray, original_length: int) -> np.ndarray:
    """
    Vectorised reverse of :func:`pack_trits_vectorized`.

    Parameters
    ----------
    packed : np.ndarray
        ``uint8`` array produced by :func:`pack_trits_vectorized`.
    original_length : int
        Original number of trits.

    Returns
    -------
    np.ndarray
        ``int8`` array of shape ``(original_length,)``.
    """
    n_bytes = len(packed)
    # Decompose each byte into 5 base-3 digits using integer division
    # Work from least-significant digit to most-significant, then reverse
    digits = np.zeros((n_bytes, TRITS_PER_BYTE), dtype=np.int8)
    remainder = packed.copy().astype(np.uint16)
    for j in range(TRITS_PER_BYTE - 1, -1, -1):
        digits[:, j] = (remainder % 3).astype(np.int8) - 1
        remainder //= 3

    return digits.ravel()[:original_length]


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------


def pack_trits_batch(tensor: np.ndarray) -> tuple[bytes, int]:
    """
    Flatten *tensor*, pack all trits, and return the packed bytes plus count.

    Parameters
    ----------
    tensor : np.ndarray
        Array of trit values {-1, 0, +1} in any shape.

    Returns
    -------
    tuple[bytes, int]
        ``(packed_bytes, original_count)`` where *original_count* is the
        total number of elements before packing.
    """
    flat = np.asarray(tensor, dtype=np.int8).ravel()
    packed = pack_trits(flat)
    return packed, len(flat)


def unpack_trits_batch(data: bytes, shape: tuple, dtype: np.dtype) -> np.ndarray:
    """
    Unpack bytes and reshape to the original tensor shape.

    Parameters
    ----------
    data : bytes
        Bytes produced by :func:`pack_trits_batch`.
    shape : tuple
        Target shape for the reconstructed tensor.
    dtype : np.dtype
        Target dtype (e.g. ``np.int8``).

    Returns
    -------
    np.ndarray
        Reconstructed trit array with the given shape and dtype.
    """
    original_length = int(np.prod(shape))
    flat = unpack_trits(data, original_length)
    return flat.reshape(shape).astype(dtype)
