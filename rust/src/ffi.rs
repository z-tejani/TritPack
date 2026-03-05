// TritPack C-compatible FFI exports.
//
// These symbols can be called from Python via ctypes or cffi without
// the PyO3 extension module.

use std::slice;

use crate::packing::{pack_trits, unpack_trits};
use crate::quantizer::quantize_tensor;

/// Pack `count` trits from `trits_ptr` into the caller-allocated `out_ptr`.
///
/// Returns the number of bytes written, or 0 on failure.
///
/// # Safety
/// * `trits_ptr` must point to `count` valid `i8` values.
/// * `out_ptr` must point to at least `ceil(count / 5)` bytes of writable memory.
#[no_mangle]
pub unsafe extern "C" fn tritpack_pack(
    trits_ptr: *const i8,
    count: usize,
    out_ptr: *mut u8,
    out_capacity: usize,
) -> usize {
    if trits_ptr.is_null() || out_ptr.is_null() {
        return 0;
    }
    let trits = unsafe { slice::from_raw_parts(trits_ptr, count) };
    let packed = pack_trits(trits);
    if packed.len() > out_capacity {
        return 0;
    }
    let out = unsafe { slice::from_raw_parts_mut(out_ptr, packed.len()) };
    out.copy_from_slice(&packed);
    packed.len()
}

/// Unpack `byte_count` packed bytes from `data_ptr` into `out_ptr`.
///
/// Returns the number of trits written (== `original_count`), or 0 on failure.
///
/// # Safety
/// * `data_ptr` must point to `byte_count` valid bytes.
/// * `out_ptr` must point to at least `original_count` writable `i8` cells.
#[no_mangle]
pub unsafe extern "C" fn tritpack_unpack(
    data_ptr: *const u8,
    byte_count: usize,
    original_count: usize,
    out_ptr: *mut i8,
    out_capacity: usize,
) -> usize {
    if data_ptr.is_null() || out_ptr.is_null() {
        return 0;
    }
    if out_capacity < original_count {
        return 0;
    }
    let data = unsafe { slice::from_raw_parts(data_ptr, byte_count) };
    let trits = unpack_trits(data, original_count);
    let out = unsafe { slice::from_raw_parts_mut(out_ptr, original_count) };
    out.copy_from_slice(&trits);
    original_count
}

/// Quantize `count` float32 values from `values_ptr` using ternary encoding.
///
/// Writes packed bytes to `packed_out` (must hold `ceil(count/5)` bytes),
/// and per-block f32 scales to `scales_out` (must hold `ceil(count/block_size)` f32 values).
///
/// Returns the number of blocks produced, or 0 on failure.
///
/// # Safety
/// * Pointers must be valid for the given sizes.
#[no_mangle]
pub unsafe extern "C" fn tritpack_quantize(
    values_ptr: *const f32,
    count: usize,
    block_size: usize,
    alpha: f32,
    packed_out: *mut u8,
    packed_capacity: usize,
    scales_out: *mut f32,
    scales_capacity: usize,
) -> usize {
    if values_ptr.is_null() || packed_out.is_null() || scales_out.is_null() {
        return 0;
    }
    let values = unsafe { slice::from_raw_parts(values_ptr, count) };
    let qt = quantize_tensor(values, block_size, alpha);

    if qt.packed_data.len() > packed_capacity || qt.scales.len() > scales_capacity {
        return 0;
    }

    let packed_slice = unsafe { slice::from_raw_parts_mut(packed_out, qt.packed_data.len()) };
    packed_slice.copy_from_slice(&qt.packed_data);

    let scales_slice = unsafe { slice::from_raw_parts_mut(scales_out, qt.scales.len()) };
    scales_slice.copy_from_slice(&qt.scales);

    qt.scales.len() // number of blocks
}
