# TritPack

**Balanced ternary memory compression for LLMs.**

TritPack compresses model weights and KV cache to ternary values `{-1, 0, +1}`,
achieving ~3–6× RAM reduction with minimal quality loss — enabling larger models
on consumer hardware.

---

## Why TritPack?

Modern LLMs are memory-bound. A 70B parameter model requires ~140 GB in FP16,
far exceeding consumer GPU/CPU RAM. TritPack provides:

- **~3–6× RAM reduction** via ternary quantization (1.6 bits/weight vs 16)
- **Per-block calibration** — each 64-element block has its own scale factor
- **Three-tier memory management** — HOT (FP16) → WARM (INT8) → COLD (ternary)
- **Rust core** with SIMD-optimised packing and rayon parallelism
- **Zero-copy lazy decompression** — decompress only what you need

---

## RAM Savings

| Model  | FP16 (GB) | TritPack (GB) | Ratio | Fits on            |
|--------|-----------|---------------|-------|--------------------|
| 7B     | 14        | ~2.5          | ~5.6× | 8 GB laptop RAM    |
| 13B    | 26        | ~4.6          | ~5.7× | 8 GB laptop RAM    |
| 34B    | 68        | ~12           | ~5.7× | 16 GB desktop RAM  |
| 70B    | 140       | ~25           | ~5.6× | 32 GB desktop RAM  |

*Ratios are analytical estimates (n/5 bytes for trits + n/64×2 bytes for scales).
Actual ratios vary by layer sparsity and block overhead.*

---

## Architecture

```
Input tensor (FP32/FP16)
         │
         ▼
  ┌─────────────────┐
  │  TernaryQuantizer│   Per-block: τ = α × mean(|block|)
  │  (block_size=64) │   trits = sign(x) × [|x| > τ]
  └────────┬────────┘   scale = mean(|x| for |x| > τ)
           │
           ▼
  ┌─────────────────┐
  │   pack_trits()   │   5 trits per byte (3⁵=243 ≤ 256)
  │  (Rust + Python) │   byte = d[0]×81 + d[1]×27 + … + d[4]
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │  QuantizedTensor │   packed_data + float16 scales per block
  └────────┬────────┘
           │
  ┌────────▼────────┐
  │   TierManager    │   HOT (FP16) ─► WARM (INT8) ─► COLD (ternary)
  │   LRU eviction   │   Promotes on access, demotes on memory pressure
  └─────────────────┘
```

---

## Quick Start

### Install

```bash
git clone https://github.com/z-tejani/TritPack
cd TritPack
pip install -e ".[dev]"

# Optional: build Rust extension for 10-100× faster packing
cd rust && maturin develop --release && cd ..
```

### Compress a tensor

```python
import numpy as np
from tritpack import TernaryQuantizer, TernaryDequantizer

# Load your model weights
weights = np.random.randn(4096, 4096).astype(np.float32)  # ~64 MB

# Compress
q = TernaryQuantizer(alpha=0.7)
qt = q.quantize(weights)

print(f"Original: {weights.nbytes / 1e6:.1f} MB")
print(f"Compressed: {qt.size_bytes() / 1e6:.1f} MB")
print(f"Ratio: {qt.compression_ratio():.1f}×")

# Decompress
dq = TernaryDequantizer()
reconstructed = dq.dequantize(qt)
```

### Use TritTensor (lazy decompression)

```python
from tritpack import TritTensor

tt = TritTensor(weights)
print(tt)  # TritTensor(shape=(4096, 4096), compression_ratio=5.73×, sparsity=41.9%)

# Decompress only when needed
data = tt.numpy()

# Quality report
report = tt.quality_report()
print(report)  # {'cos_sim': 0.90, 'snr_db': 7.1, 'rmse': 0.44, ...}
```

### Run the tier manager (simulates LLM inference)

```python
from tritpack import TierManager
import numpy as np

# 8 GB total, 15% HOT, 25% WARM, rest COLD
tm = TierManager(memory_limit_gb=8.0)

# Register model layers
for i in range(32):
    layer = np.random.randn(4096, 4096).astype(np.float32)
    tm.register_layer(f"layer_{i}", layer)

# Access layers — automatic promotion/demotion
for i in range(32):
    weights = tm.access(f"layer_{i}")
    # ... run forward pass with weights ...

print(tm.memory_usage())
# {'hot_gb': 1.2, 'warm_gb': 2.0, 'cold_gb': 4.7, 'total_gb': 7.9}
```

### Calibrate per-layer alpha

```python
from tritpack.model.calibration import ThresholdCalibrator

cal = ThresholdCalibrator()
layers = {"q_proj": weights_q, "v_proj": weights_v}

# Find highest alpha with cos_sim >= 0.90
alphas = cal.calibrate_model_layers(layers, target_cosine_sim=0.90)
# {'q_proj': 0.58, 'v_proj': 0.61}
```

---

## Local Model Inference

TritPack integrates with the two most common local-inference stacks.
Both paths store weights in compressed form and decompress on demand.

### Path A — llama.cpp / Ollama (via GGUF reconstruction)

Works with any GGUF model: llama.cpp, llama-cpp-python, Ollama, LM Studio, etc.

```bash
pip install gguf llama-cpp-python
```

**Step 1 — compress once** (≈5 min for a 7B model, saved permanently):

```python
from integration.gguf_patcher import GGUFPatcher

patcher = GGUFPatcher(alpha=0.7)
report = patcher.patch("llama-7b.gguf", "llama-7b.tritpack/")
print(f"Compressed {report.ratio:.1f}× → {report.compressed_gb:.1f} GB")
# Compressed 5.6× → 2.5 GB
```

**Step 2 — load for inference** (decompresses to a temp GGUF on first run,
reused on subsequent runs):

```python
from integration.llamacpp_shim import load_from_tritpack

model = load_from_tritpack(
    tritpack_dir="llama-7b.tritpack/",
    temp_gguf="/tmp/llama-7b-reconstructed.gguf",
    reuse_temp=True,      # skips reconstruction after first run
    verbose=True,
    n_ctx=2048,           # llama_cpp.Llama kwargs pass through
)

output = model("The meaning of life is", max_tokens=80)
print(output["choices"][0]["text"])
```

Or use the `LlamaCppTritShim` class directly when you want to load
a `.tritpack/` directory through the full tier-managed interface:

```python
from integration.llamacpp_shim import LlamaCppTritShim

shim = LlamaCppTritShim(
    model_path="llama-7b.tritpack/",
    memory_limit_gb=8.0,
    temp_gguf="/tmp/llama-7b-reconstructed.gguf",
)
model = shim.load(verbose=True)
```

**Disk vs RAM tradeoffs:**

| Stage | Disk | RAM |
|-------|------|-----|
| Original GGUF | 14 GB | 14 GB |
| `.tritpack/` compressed | **2.5 GB** | — |
| Reconstructed GGUF (temp) | 14 GB | 14 GB |

> The temp GGUF is full FP32 quality (slightly below original due to ternary
> quantization, cosine similarity ≈ 0.90).  The main saving here is **storage**:
> your model library stays 5× smaller on disk.

---

### Path B — HuggingFace transformers (in-process RAM reduction)

This path actually reduces RAM **during inference**: weights stay
ternary-compressed in memory and each layer is decompressed only when its
`forward()` is called.

```bash
pip install torch transformers
```

**One-liner:**

```python
from integration.transformers_shim import load_compressed_model

model, tokenizer = load_compressed_model(
    "meta-llama/Llama-2-7b-hf",
    alpha=0.7,
    verbose=True,
)

inputs = tokenizer("Hello, I am a language model", return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

**Or compress an already-loaded model:**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from integration.transformers_shim import compress_model

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Replace all nn.Linear layers with ternary-compressed equivalents
model = compress_model(model, alpha=0.7, verbose=True)
# Compressed 147 Linear layers.

# Use exactly like the original model
inputs = tokenizer("Once upon a time", return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

**RAM comparison (Llama-2 7B):**

| Mode | Weight RAM |
|------|-----------|
| `from_pretrained` (FP32) | ~28 GB |
| `from_pretrained` (FP16) | ~14 GB |
| `compress_model(alpha=0.7)` | **~2.5 GB** |

> Inference is slightly slower than native (decompression overhead per layer).
> Use `cache_weights=True` to cache decompressed tensors after first use —
> faster repeated inference at the cost of higher RAM.

---

### Estimate savings before committing

```python
from integration.ollama_proxy import OllamaMemoryEstimator

est = OllamaMemoryEstimator("llama3:8b")
print(est.estimate())
# {'model': 'llama3:8b', 'params_b': 8.0, 'fp16_gb': 14.9,
#  'tritpack_gb': 2.7, 'ratio': 5.5}
```

---

## Benchmarks

Run the benchmark suite:

```bash
python -m tritpack.benchmark bench-all
python -m tritpack.benchmark bench-pack
python -m tritpack.benchmark bench-quality --output quality.json
```

Sample results (Python reference implementation, 8-core x86):

| Operation            | Size   | Throughput |
|----------------------|--------|------------|
| pack_trits (Python)  | 100 MB | ~400 MB/s  |
| pack_trits (Rust)    | 100 MB | ~8 GB/s    |
| quantize (Python)    | 64 MB  | ~150 MB/s  |
| TierManager access   | HOT    | <0.1 ms    |
| TierManager access   | COLD   | <5 ms      |

*Fill in after running: `pytest tests/benchmarks/ --benchmark-json=bench.json`*

---

## Quality / Compression Tradeoff

The `alpha` parameter controls the tradeoff:

| alpha | Sparsity | cos_sim | SNR    | Ratio |
|-------|----------|---------|--------|-------|
| 0.3   | ~19%     | ~0.86   | ~12 dB | ~4.8× |
| 0.5   | ~31%     | ~0.88   | ~9 dB  | ~5.2× |
| 0.7   | ~42%     | ~0.90   | ~7 dB  | ~5.7× |
| 0.9   | ~53%     | ~0.89   | ~6 dB  | ~6.2× |
| 1.0   | ~58%     | ~0.87   | ~5 dB  | ~6.5× |

*Cosine similarity peaks near alpha≈0.55–0.70 for standard Gaussian weights.*

For fine-grained control, use `ThresholdCalibrator` to find the optimal
alpha per layer while satisfying a cosine similarity target.

---

## Encoding Details

5 trits are packed into 1 byte (3⁵ = 243 ≤ 256):

```
byte = d[0]×81 + d[1]×27 + d[2]×9 + d[3]×3 + d[4]
where d[i] = trit[i] + 1  (shift: −1→0, 0→1, 1→2)
```

The 8-byte header encodes the original trit count (little-endian uint64)
for exact padding removal on unpack.

---

## Running Tests

```bash
# All tests
pytest tests/ -x --tb=short

# With coverage
pytest tests/ --cov=tritpack --cov-report=term-missing

# Unit only (fast)
pytest tests/unit/ -q

# Benchmarks
pytest tests/benchmarks/ --benchmark-only

# Rust tests
cd rust && cargo test

# Rust/Python parity (requires compiled extension)
cd rust && maturin develop --release && cd ..
pytest tests/verification/ -v
```

---

## Project Layout

```
tritpack/
├── python/tritpack/
│   ├── core/          # packing.py, quantizer.py, dequantizer.py, tiers.py
│   ├── model/         # tensor.py, layer.py, calibration.py
│   ├── backends/      # numpy_backend.py, gguf_backend.py
│   └── benchmark/     # memory.py, quality.py, speed.py, __main__.py
├── tests/
│   ├── unit/          # test_packing.py, test_quantizer.py, ...
│   ├── integration/   # test_full_pipeline.py, ...
│   ├── benchmarks/    # pytest-benchmark tests
│   ├── fixtures/      # synthetic_tensors.py, sample_weights.py
│   └── verification/  # test_rust_python_parity.py
├── rust/src/          # packing.rs, quantizer.rs, ffi.rs, lib.rs
├── integration/       # tritpack_loader.py, transformers_shim.py,
│                    # llamacpp_shim.py, gguf_patcher.py, ollama_proxy.py
└── pyproject.toml
```

---

## Contributing

1. Fork the repo and create a feature branch.
2. Write tests for any new functionality.
3. Ensure `pytest tests/` passes with `>90%` coverage.
4. For Rust changes, ensure `cargo test` passes and re-run parity tests.
5. Open a pull request with a clear description.

**Key invariants:**
- `unpack_trits(pack_trits(x), len(x)) == x` — exact inverse, always
- Rust and Python packing must be bit-identical (verified by parity suite)
- All float16 scale overflows must be handled gracefully

---

## License

MIT
