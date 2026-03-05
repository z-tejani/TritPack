// TritPack quantizer — Rust implementation.
//
// Mirrors the Python TernaryQuantizer exactly so that the parity tests pass.

use crate::packing::{pack_trits, unpack_trits};

/// A quantized tensor (Rust-side representation).
pub struct QuantizedTensor {
    /// Concatenated packed bytes for all blocks.
    pub packed_data: Vec<u8>,
    /// Per-block scale factors (f32 for precision; Python stores f16).
    pub scales: Vec<f32>,
    /// Byte offset of each block in `packed_data`.
    pub block_offsets: Vec<usize>,
    /// Number of trits in each block.
    pub block_lengths: Vec<usize>,
    /// Original element count.
    pub original_count: usize,
    /// Nominal block size.
    pub block_size: usize,
    /// Threshold ratio.
    pub alpha: f32,
    /// Fraction of zero trits.
    pub sparsity: f32,
}

// ---------------------------------------------------------------------------
// Per-block helpers
// ---------------------------------------------------------------------------

/// Quantize a single block into trits and a scale factor.
///
/// Returns `(trits, scale)` where trits ∈ {-1, 0, +1}.
pub fn quantize_block(values: &[f32], alpha: f32) -> (Vec<i8>, f32) {
    let abs_vals: Vec<f32> = values.iter().map(|x| x.abs()).collect();
    let mean_abs = abs_vals.iter().sum::<f32>() / abs_vals.len() as f32;
    let tau = alpha * mean_abs;

    let mut trits = vec![0i8; values.len()];
    let mut sum_nonzero = 0.0f32;
    let mut n_nonzero: usize = 0;

    for (i, (&v, &a)) in values.iter().zip(abs_vals.iter()).enumerate() {
        if v > tau {
            trits[i] = 1;
            sum_nonzero += a;
            n_nonzero += 1;
        } else if v < -tau {
            trits[i] = -1;
            sum_nonzero += a;
            n_nonzero += 1;
        }
    }

    let scale = if n_nonzero > 0 {
        sum_nonzero / n_nonzero as f32
    } else {
        1.0
    };

    (trits, scale)
}

/// Dequantize a single block: `reconstructed[i] = trit[i] * scale`.
pub fn dequantize_block(trits: &[i8], scale: f32) -> Vec<f32> {
    trits.iter().map(|&t| t as f32 * scale).collect()
}

// ---------------------------------------------------------------------------
// Full-tensor quantization
// ---------------------------------------------------------------------------

/// Quantize a flat float32 slice into a [`QuantizedTensor`].
pub fn quantize_tensor(values: &[f32], block_size: usize, alpha: f32) -> QuantizedTensor {
    let n = values.len();
    let n_blocks = n.div_ceil(block_size);

    let mut packed_data = Vec::new();
    let mut scales = Vec::with_capacity(n_blocks);
    let mut block_offsets = Vec::with_capacity(n_blocks);
    let mut block_lengths = Vec::with_capacity(n_blocks);
    let mut zero_count = 0usize;
    let mut total_count = 0usize;

    for b in 0..n_blocks {
        let start = b * block_size;
        let end = (start + block_size).min(n);
        let block = &values[start..end];
        let blen = block.len();

        let (trits, scale) = quantize_block(block, alpha);
        zero_count += trits.iter().filter(|&&t| t == 0).count();
        total_count += blen;

        let packed_block = pack_trits(&trits);
        block_offsets.push(packed_data.len());
        block_lengths.push(blen);
        scales.push(scale);
        packed_data.extend_from_slice(&packed_block);
    }

    let sparsity = if total_count > 0 {
        zero_count as f32 / total_count as f32
    } else {
        0.0
    };

    QuantizedTensor {
        packed_data,
        scales,
        block_offsets,
        block_lengths,
        original_count: n,
        block_size,
        alpha,
        sparsity,
    }
}

/// Dequantize a [`QuantizedTensor`] back to float32.
pub fn dequantize_tensor(qt: &QuantizedTensor) -> Vec<f32> {
    let mut result = vec![0.0f32; qt.original_count];

    for (b, (&offset, &blen)) in qt
        .block_offsets
        .iter()
        .zip(qt.block_lengths.iter())
        .enumerate()
    {
        let end_offset = if b + 1 < qt.block_offsets.len() {
            qt.block_offsets[b + 1]
        } else {
            qt.packed_data.len()
        };

        let block_bytes = &qt.packed_data[offset..end_offset];
        let trits = unpack_trits(block_bytes, blen);
        let scale = qt.scales[b];
        let start = b * qt.block_size;
        for (i, &t) in trits.iter().enumerate() {
            result[start + i] = t as f32 * scale;
        }
    }

    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_block_all_zeros() {
        let block = vec![0.0f32; 64];
        let (trits, scale) = quantize_block(&block, 0.7);
        assert!(trits.iter().all(|&t| t == 0));
        assert_eq!(scale, 1.0);
    }

    #[test]
    fn test_quantize_block_positive() {
        let block = vec![1.0f32; 64];
        let (trits, scale) = quantize_block(&block, 0.5);
        // τ = 0.5 * 1.0 = 0.5; all values > 0.5, so all trits = +1
        assert!(trits.iter().all(|&t| t == 1));
        assert!((scale - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_quantize_dequantize_roundtrip() {
        let values: Vec<f32> = (0..1000).map(|i| (i as f32 - 500.0) / 100.0).collect();
        let qt = quantize_tensor(&values, 64, 0.7);
        let reconstructed = dequantize_tensor(&qt);
        assert_eq!(reconstructed.len(), values.len());
    }

    #[test]
    fn test_scale_non_negative() {
        let values: Vec<f32> = (0..1000).map(|i| (i as f32).sin()).collect();
        let qt = quantize_tensor(&values, 64, 0.7);
        assert!(qt.scales.iter().all(|&s| s >= 0.0));
    }
}
