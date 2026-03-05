"""
TritPack Rust/Python parity test suite.

Verifies that the Rust and Python implementations produce bit-identical
output for packing and numerically identical output for quantization.

The Rust extension module (``tritpack_native``) must be compiled and
importable; if not, all tests are skipped with an informative message.

Compatibility notes
-------------------
The Python ``pack_trits`` function includes an 8-byte length header in its
output; the Rust ``pack_trits`` function does NOT (it returns raw packed
bytes).  The parity tests compare only the payload bytes (stripping the
Python header).  The quantizer parity test compares trit arrays derived
from unpacking both payloads and checks that scales match within float16
precision.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from tritpack.core.packing import pack_trits as py_pack, unpack_trits as py_unpack
from tritpack.core.quantizer import TernaryQuantizer
from tritpack.core.dequantizer import TernaryDequantizer

# Try to import the compiled Rust extension.
try:
    import tritpack_native as _rust  # type: ignore
    _RUST_AVAILABLE = True
except ImportError:
    _RUST_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _RUST_AVAILABLE,
    reason="tritpack_native Rust extension not compiled (run `maturin develop` in rust/)",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _py_payload(trits: np.ndarray) -> bytes:
    """Return the payload bytes from Python pack_trits (strips 8-byte header)."""
    return py_pack(trits)[8:]


def _rs_payload(trits: np.ndarray) -> bytes:
    """Return packed bytes from Rust pack_trits (no header)."""
    return bytes(_rust.pack_trits(trits.tolist()))


def _make_trits(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(-1, 2, size=n, dtype=np.int8)


def _make_float32(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n).astype(np.float32)


# ---------------------------------------------------------------------------
# Packing parity tests
# ---------------------------------------------------------------------------


def test_pack_parity_small():
    """Small array: Rust and Python packing must be bit-identical."""
    trits = np.array([-1, 0, 1, 0, -1, 1], dtype=np.int8)
    py_bytes = _py_payload(trits)
    rs_bytes = _rs_payload(trits)
    assert py_bytes == rs_bytes, (
        f"Packing mismatch:\n  Python:  {py_bytes.hex()}\n  Rust:    {rs_bytes.hex()}"
    )


def test_pack_parity_1000_random():
    """1000 random trit arrays of varied sizes must match exactly."""
    mismatches = []
    for seed in range(1000):
        n = (seed % 200) + 1
        trits = _make_trits(n, seed)
        py_bytes = _py_payload(trits)
        rs_bytes = _rs_payload(trits)
        if py_bytes != rs_bytes:
            mismatches.append(
                f"seed={seed}, n={n}: py={py_bytes[:4].hex()}... rs={rs_bytes[:4].hex()}..."
            )

    assert not mismatches, (
        f"Rust/Python packing mismatch on {len(mismatches)} cases:\n"
        + "\n".join(mismatches[:5])
    )


def test_unpack_parity():
    """Unpacking Rust-packed data with Python must recover original trits."""
    for seed in range(200):
        n = (seed % 100) + 1
        trits = _make_trits(n, seed)
        rs_packed = _rs_payload(trits)
        # Wrap with Python header so we can call py_unpack
        header = struct.pack("<Q", n)
        recovered = py_unpack(header + rs_packed, n)
        np.testing.assert_array_equal(
            recovered, trits,
            err_msg=f"Unpack mismatch at seed={seed}, n={n}",
        )


# ---------------------------------------------------------------------------
# Quantizer parity tests
# ---------------------------------------------------------------------------


def test_quantize_parity_trits():
    """Rust and Python trit patterns must match across varied tensors."""
    q_py = TernaryQuantizer(block_size=64, alpha=0.7)
    dq = TernaryDequantizer()
    mismatches = []

    for seed in range(100):
        n = 64 * (seed % 10 + 1)  # always a multiple of block_size for simplicity
        values = _make_float32(n, seed)

        # Python quantization
        py_qt = q_py.quantize(values)

        # Rust quantization (returns packed_data, scales, offsets, lengths, sparsity)
        rs_result = _rust.quantize_tensor(values.tolist(), 64, 0.7)
        rs_packed_data, rs_scales, rs_offsets, rs_lengths, rs_sparsity = rs_result

        # Unpack both and compare trit arrays per block
        for b, (blen, py_offset, rs_offset) in enumerate(
            zip(py_qt.block_lengths, py_qt.block_offsets, rs_offsets)
        ):
            py_end = py_qt.block_offsets[b + 1] if b + 1 < len(py_qt.block_offsets) else len(py_qt.packed_data)
            rs_end = rs_offsets[b + 1] if b + 1 < len(rs_offsets) else len(rs_packed_data)

            py_block_bytes = py_qt.packed_data[py_offset:py_end]
            rs_block_bytes = bytes(rs_packed_data)[rs_offset:rs_end]

            # Python packs with 8-byte header; Rust does not.  Strip Python header.
            py_payload = py_block_bytes[8:]

            if py_payload != rs_block_bytes:
                mismatches.append(
                    f"seed={seed}, block={b}: py={py_payload[:4].hex()} rs={rs_block_bytes[:4].hex()}"
                )

    assert not mismatches, (
        f"Trit mismatch in {len(mismatches)} blocks:\n" + "\n".join(mismatches[:5])
    )


def test_quantize_parity_scales():
    """Rust and Python scale factors must agree within float16 precision."""
    q_py = TernaryQuantizer(block_size=64, alpha=0.7)
    mismatches = []

    for seed in range(100):
        n = 64 * (seed % 10 + 1)
        values = _make_float32(n, seed)

        py_qt = q_py.quantize(values)
        rs_result = _rust.quantize_tensor(values.tolist(), 64, 0.7)
        _, rs_scales_raw, _, _, _ = rs_result

        py_scales_f32 = py_qt.scales.astype(np.float32)
        rs_scales = np.array(rs_scales_raw, dtype=np.float32)

        # Rust uses f32 scales; Python stores f16.  Allow rounding to f16.
        if not np.allclose(py_scales_f32, rs_scales, atol=1e-3):
            mismatches.append(
                f"seed={seed}: max_diff={np.max(np.abs(py_scales_f32 - rs_scales)):.6f}"
            )

    assert not mismatches, (
        f"Scale mismatch in {len(mismatches)} tensors:\n" + "\n".join(mismatches[:5])
    )
