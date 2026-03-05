// TritPack Rust library.
//
// Exposes:
//  - `packing`   — trit pack/unpack engine
//  - `quantizer` — ternary quantization
//  - `ffi`       — C-ABI exports for ctypes/cffi
//  - PyO3 Python extension module `tritpack_native`

pub mod ffi;
pub mod packing;
pub mod quantizer;

use pyo3::prelude::*;
use pyo3::types::PyBytes;

// ---------------------------------------------------------------------------
// PyO3 bindings
// ---------------------------------------------------------------------------

/// Pack a list/bytes of trit values {-1, 0, +1} into bytes.
///
/// Parameters
/// ----------
/// trits : list[int]  or  bytes  (interpreted as signed i8)
///
/// Returns
/// -------
/// bytes  (no header; length = ceil(len(trits) / 5))
#[pyfunction]
fn pack_trits(py: Python<'_>, trits: Vec<i8>) -> PyResult<PyObject> {
    let packed = packing::pack_trits(&trits);
    Ok(PyBytes::new_bound(py, &packed).into())
}

/// Unpack bytes produced by `pack_trits` back to a list of trits.
///
/// Parameters
/// ----------
/// data : bytes
/// count : int   — number of trits to recover (strips padding)
///
/// Returns
/// -------
/// list[int]
#[pyfunction]
fn unpack_trits(_py: Python<'_>, data: &[u8], count: usize) -> PyResult<Vec<i8>> {
    Ok(packing::unpack_trits(data, count))
}

/// Pack trits in parallel (uses rayon for large inputs).
#[pyfunction]
fn pack_trits_parallel(py: Python<'_>, trits: Vec<i8>) -> PyResult<PyObject> {
    let packed = packing::pack_trits_parallel(&trits);
    Ok(PyBytes::new_bound(py, &packed).into())
}

/// Quantize a float32 tensor using per-block ternary quantization.
///
/// Parameters
/// ----------
/// values : list[float]
/// block_size : int
/// alpha : float
///
/// Returns
/// -------
/// tuple[bytes, list[float], list[int], list[int], float]
///     (packed_data, scales, block_offsets, block_lengths, sparsity)
#[pyfunction]
fn quantize_tensor(
    py: Python<'_>,
    values: Vec<f32>,
    block_size: usize,
    alpha: f32,
) -> PyResult<PyObject> {
    let qt = quantizer::quantize_tensor(&values, block_size, alpha);
    let packed_bytes = PyBytes::new_bound(py, &qt.packed_data);
    let result = (
        packed_bytes,
        qt.scales,
        qt.block_offsets,
        qt.block_lengths,
        qt.sparsity,
    );
    Ok(result.to_object(py))
}

/// Dequantize a ternary-quantized tensor back to float32.
///
/// Parameters
/// ----------
/// packed_data : bytes
/// scales : list[float]
/// block_offsets : list[int]
/// block_lengths : list[int]
/// original_count : int
/// block_size : int
///
/// Returns
/// -------
/// list[float]
#[pyfunction]
fn dequantize_tensor(
    _py: Python<'_>,
    packed_data: &[u8],
    scales: Vec<f32>,
    block_offsets: Vec<usize>,
    block_lengths: Vec<usize>,
    original_count: usize,
    block_size: usize,
) -> PyResult<Vec<f32>> {
    let qt = quantizer::QuantizedTensor {
        packed_data: packed_data.to_vec(),
        scales,
        block_offsets,
        block_lengths,
        original_count,
        block_size,
        alpha: 0.0, // not needed for dequantization
        sparsity: 0.0,
    };
    Ok(quantizer::dequantize_tensor(&qt))
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------

#[pymodule]
fn tritpack_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pack_trits, m)?)?;
    m.add_function(wrap_pyfunction!(unpack_trits, m)?)?;
    m.add_function(wrap_pyfunction!(pack_trits_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(quantize_tensor, m)?)?;
    m.add_function(wrap_pyfunction!(dequantize_tensor, m)?)?;
    Ok(())
}
