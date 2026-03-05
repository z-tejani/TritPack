// TritPack Criterion benchmarks.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use tritpack::packing::{pack_trits, pack_trits_parallel, unpack_trits};
use tritpack::quantizer::{dequantize_tensor, quantize_tensor};

fn make_trits(n: usize) -> Vec<i8> {
    (0..n).map(|i| ((i % 3) as i8) - 1).collect()
}

fn make_floats(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i as f32).sin()).collect()
}

// ---------------------------------------------------------------------------
// Packing benchmarks
// ---------------------------------------------------------------------------

fn bench_pack_1mb(c: &mut Criterion) {
    let trits = make_trits(1 << 20);
    c.bench_function("pack_1mb_serial", |b| {
        b.iter(|| pack_trits(black_box(&trits)))
    });
}

fn bench_pack_100mb(c: &mut Criterion) {
    let trits = make_trits(100 * (1 << 20));
    c.bench_function("pack_100mb_serial", |b| {
        b.iter(|| pack_trits(black_box(&trits)))
    });
}

fn bench_pack_parallel_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("pack_parallel");
    for mb in [1u64, 10, 100] {
        let trits = make_trits((mb as usize) * (1 << 20));
        group.bench_with_input(BenchmarkId::new("rayon", mb), &mb, |b, _| {
            b.iter(|| pack_trits_parallel(black_box(&trits)))
        });
        group.bench_with_input(BenchmarkId::new("serial", mb), &mb, |b, _| {
            b.iter(|| pack_trits(black_box(&trits)))
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Quantization benchmarks
// ---------------------------------------------------------------------------

fn bench_quantize_1mb(c: &mut Criterion) {
    let values = make_floats(1 << 20);
    c.bench_function("quantize_1mb", |b| {
        b.iter(|| quantize_tensor(black_box(&values), 64, 0.7))
    });
}

fn bench_quantize_100mb(c: &mut Criterion) {
    let values = make_floats(100 * (1 << 18)); // 100M f32 = 400 MB
    c.bench_function("quantize_100mb", |b| {
        b.iter(|| quantize_tensor(black_box(&values), 64, 0.7))
    });
}

fn bench_dequantize_1mb(c: &mut Criterion) {
    let values = make_floats(1 << 20);
    let qt = quantize_tensor(&values, 64, 0.7);
    c.bench_function("dequantize_1mb", |b| {
        b.iter(|| dequantize_tensor(black_box(&qt)))
    });
}

// ---------------------------------------------------------------------------
// Criterion groups
// ---------------------------------------------------------------------------

criterion_group!(
    benches,
    bench_pack_1mb,
    bench_pack_100mb,
    bench_pack_parallel_sizes,
    bench_quantize_1mb,
    bench_quantize_100mb,
    bench_dequantize_1mb,
);
criterion_main!(benches);
