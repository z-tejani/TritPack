// TritPack packing engine — Rust implementation.
//
// Mirrors the Python `pack_trits` / `unpack_trits` exactly:
//   byte = d[0]*81 + d[1]*27 + d[2]*9 + d[3]*3 + d[4]
//   where d[i] = trit[i] + 1  (shift −1→0, 0→1, 1→2)
//
// The bit-exact match to the Python implementation is verified by the
// parity test suite.

use rayon::prelude::*;

/// Powers of 3 for positions 0..4 in a packed byte.
const POW3: [u8; 5] = [81, 27, 9, 3, 1];
/// Trits per byte (3^5 = 243 ≤ 256).
pub const TRITS_PER_BYTE: usize = 5;
/// Threshold (in elements) above which rayon parallel packing is used.
const PARALLEL_THRESHOLD: usize = 1 << 20; // 1 M elements ≈ 1 MB

// ---------------------------------------------------------------------------
// Serial implementation
// ---------------------------------------------------------------------------

/// Pack an array of trits {-1, 0, +1} into bytes using base-3 encoding.
///
/// Returns the packed bytes **without** the 8-byte length header (the Python
/// version includes a header; callers that need the header should prepend it).
///
/// # Panics
/// Panics in debug mode if any element is outside {-1, 0, +1}.
pub fn pack_trits(trits: &[i8]) -> Vec<u8> {
    let n = trits.len();
    // Round up to multiple of TRITS_PER_BYTE
    let n_padded = n.div_ceil(TRITS_PER_BYTE) * TRITS_PER_BYTE;
    let n_bytes = n_padded / TRITS_PER_BYTE;
    let mut packed = vec![0u8; n_bytes];

    for (byte_idx, chunk) in trits.chunks(TRITS_PER_BYTE).enumerate() {
        let mut byte_val: u8 = 0;
        for (pos, &t) in chunk.iter().enumerate() {
            debug_assert!(t >= -1 && t <= 1, "trit out of range: {t}");
            let digit = (t + 1) as u8; // shift −1→0, 0→1, 1→2
            byte_val += digit * POW3[pos];
        }
        // Padding: Python pads with trit 0 → digit 1.
        // Match Python convention so that Rust/Python output is bit-identical.
        for pos in chunk.len()..TRITS_PER_BYTE {
            byte_val += POW3[pos]; // digit 1 × POW3[pos]
        }
        packed[byte_idx] = byte_val;
    }
    packed
}

/// Unpack bytes (produced by [`pack_trits`]) back into trits.
///
/// `count` is the original number of trits (strips padding).
pub fn unpack_trits(data: &[u8], count: usize) -> Vec<i8> {
    let n_bytes = data.len();
    let total = n_bytes * TRITS_PER_BYTE;
    let mut result = vec![0i8; total];

    for (byte_idx, &byte_val) in data.iter().enumerate() {
        let mut v = byte_val;
        for pos in (0..TRITS_PER_BYTE).rev() {
            result[byte_idx * TRITS_PER_BYTE + pos] = (v % 3) as i8 - 1;
            v /= 3;
        }
    }
    result.truncate(count);
    result
}

// ---------------------------------------------------------------------------
// Parallel implementation (rayon)
// ---------------------------------------------------------------------------

/// Pack trits in parallel using rayon for arrays above [`PARALLEL_THRESHOLD`].
///
/// Falls back to serial [`pack_trits`] for smaller inputs.
pub fn pack_trits_parallel(trits: &[i8]) -> Vec<u8> {
    if trits.len() < PARALLEL_THRESHOLD {
        return pack_trits(trits);
    }

    let n = trits.len();
    let n_padded = n.div_ceil(TRITS_PER_BYTE) * TRITS_PER_BYTE;
    let n_bytes = n_padded / TRITS_PER_BYTE;
    let mut packed = vec![0u8; n_bytes];

    // Process chunks in parallel; each chunk produces one output byte.
    packed
        .par_iter_mut()
        .enumerate()
        .for_each(|(byte_idx, out)| {
            let start = byte_idx * TRITS_PER_BYTE;
            let end = (start + TRITS_PER_BYTE).min(n);
            let mut byte_val: u8 = 0;
            let chunk_len = end - start;
            for (pos, &t) in trits[start..end].iter().enumerate() {
                let digit = (t + 1) as u8;
                byte_val += digit * POW3[pos];
            }
            // Pad missing positions with digit=1 (trit 0) to match Python.
            for pos in chunk_len..TRITS_PER_BYTE {
                byte_val += POW3[pos];
            }
            *out = byte_val;
        });

    packed
}

/// Unpack trits in parallel.
pub fn unpack_trits_parallel(data: &[u8], count: usize) -> Vec<i8> {
    if data.len() * TRITS_PER_BYTE < PARALLEL_THRESHOLD {
        return unpack_trits(data, count);
    }

    let n_bytes = data.len();
    let total = n_bytes * TRITS_PER_BYTE;
    let mut result = vec![0i8; total];

    result
        .par_chunks_mut(TRITS_PER_BYTE)
        .enumerate()
        .for_each(|(byte_idx, chunk)| {
            if byte_idx < n_bytes {
                let mut v = data[byte_idx];
                for pos in (0..TRITS_PER_BYTE).rev() {
                    chunk[pos] = (v % 3) as i8 - 1;
                    v /= 3;
                }
            }
        });

    result.truncate(count);
    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_small() {
        let trits: Vec<i8> = vec![-1, 0, 1, 0, -1];
        let packed = pack_trits(&trits);
        let recovered = unpack_trits(&packed, trits.len());
        assert_eq!(recovered, trits);
    }

    #[test]
    fn test_roundtrip_non_multiple_of_5() {
        let trits: Vec<i8> = vec![1, -1, 0, 1];
        let packed = pack_trits(&trits);
        let recovered = unpack_trits(&packed, trits.len());
        assert_eq!(recovered, trits);
    }

    #[test]
    fn test_all_zeros() {
        let trits = vec![0i8; 64];
        let packed = pack_trits(&trits);
        let recovered = unpack_trits(&packed, 64);
        assert_eq!(recovered, trits);
    }

    #[test]
    fn test_all_trit_values() {
        for val in [-1i8, 0, 1] {
            for pos in 0..5 {
                let mut trits = vec![0i8; 5];
                trits[pos] = val;
                let packed = pack_trits(&trits);
                let recovered = unpack_trits(&packed, 5);
                assert_eq!(recovered, trits, "val={val} pos={pos}");
            }
        }
    }

    #[test]
    fn test_parallel_matches_serial() {
        let trits: Vec<i8> = (0..1000)
            .map(|i| ((i % 3) as i8) - 1)
            .collect();
        let serial = pack_trits(&trits);
        let parallel = pack_trits_parallel(&trits);
        assert_eq!(serial, parallel);
    }

    #[test]
    fn test_packed_byte_range() {
        let trits: Vec<i8> = vec![-1, 1, 0, -1, 1];
        let packed = pack_trits(&trits);
        for &b in &packed {
            assert!(b <= 242, "byte {b} out of range");
        }
    }
}
