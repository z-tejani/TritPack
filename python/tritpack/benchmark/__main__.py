"""
TritPack benchmark CLI.

Usage
-----
    python -m tritpack.benchmark --help
    python -m tritpack.benchmark bench-pack
    python -m tritpack.benchmark bench-quality
    python -m tritpack.benchmark bench-memory [--model model.gguf]
    python -m tritpack.benchmark bench-all
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

try:
    from rich.console import Console
    from rich.table import Table

    _RICH = True
except ImportError:
    _RICH = False


def _print_table(title: str, rows: list[dict]) -> None:
    if not rows:
        print(f"\n{title}: no data")
        return
    if _RICH:
        console = Console()
        table = Table(title=title, show_lines=True)
        for key in rows[0]:
            table.add_column(str(key))
        for row in rows:
            table.add_row(*[str(v) for v in row.values()])
        console.print(table)
    else:
        print(f"\n=== {title} ===")
        keys = list(rows[0].keys())
        print("\t".join(keys))
        for row in rows:
            print("\t".join(str(row[k]) for k in keys))


def cmd_bench_pack(args: argparse.Namespace) -> dict:
    from tritpack.benchmark.speed import benchmark_pack_throughput

    sizes = [1.0, 10.0, 100.0]
    report = benchmark_pack_throughput(sizes)
    summary = report.summary()
    _print_table("Pack Throughput", summary)
    return {"pack_throughput": summary}


def cmd_bench_quality(args: argparse.Namespace) -> dict:
    from tritpack.benchmark.quality import benchmark_reconstruction_quality

    rng = np.random.default_rng(42)
    tensor = rng.standard_normal((1000,)).astype(np.float32)
    alphas = [0.3, 0.5, 0.7, 0.9, 1.0]
    report = benchmark_reconstruction_quality(tensor, alphas)
    summary = report.summary_table()
    _print_table("Reconstruction Quality (1000-element Gaussian)", summary)
    return {"quality": summary}


def cmd_bench_memory(args: argparse.Namespace) -> dict:
    from tritpack.benchmark.memory import benchmark_tensor_compression, measure_process_ram

    ram_before = measure_process_ram()
    rng = np.random.default_rng(42)
    # Simulate a ~100 MB layer (fp32)
    tensor = rng.standard_normal((25, 1024, 1024)).astype(np.float32)
    report = benchmark_tensor_compression(tensor)
    ram_after = measure_process_ram()

    result = {
        "original_gb": round(report.original_gb, 4),
        "compressed_gb": round(report.compressed_gb, 4),
        "ratio": round(report.ratio, 2),
        "time_s": round(report.time_seconds, 3),
        "cos_sim": round(report.cos_sim, 4),
        "snr_db": round(report.snr, 1),
        "ram_delta_mb": round(ram_after - ram_before, 1),
    }
    _print_table("Memory Compression", [result])
    return {"memory": result}


def cmd_bench_all(args: argparse.Namespace) -> dict:
    results = {}
    results.update(cmd_bench_pack(args))
    results.update(cmd_bench_quality(args))
    results.update(cmd_bench_memory(args))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m tritpack.benchmark",
        description="TritPack benchmark CLI",
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        help="Write results to JSON file",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("bench-pack", help="Benchmark packing throughput")
    sub.add_parser("bench-quality", help="Benchmark reconstruction quality")
    sub.add_parser("bench-memory", help="Benchmark memory compression")
    sub.add_parser("bench-all", help="Run all benchmarks")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(0)

    dispatch = {
        "bench-pack": cmd_bench_pack,
        "bench-quality": cmd_bench_quality,
        "bench-memory": cmd_bench_memory,
        "bench-all": cmd_bench_all,
    }
    results = dispatch[args.command](args)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
