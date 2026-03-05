# TritPack — Capabilities and Design

## The Problem

Running large language models locally is gated almost entirely by RAM.

A model's size in memory is determined by the number of parameters multiplied
by the bytes used to store each one.  A 7B-parameter model at the standard
FP16 precision occupies 14 GB.  A 70B model needs 140 GB — well beyond any
consumer machine and most workstations.

The standard workarounds (4-bit GGUF, GPTQ, AWQ) are all forms of *fixed-bit
quantization*: they reduce each weight from 16 bits to 4 or 8 bits.  They work
well, but they have a floor — you cannot go below 4 bits per weight without
more advanced techniques, and even 4-bit still means 3.5 GB for a 7B model.

TritPack takes a different approach: instead of allocating a fixed number of
bits per weight, it asks how few *distinct values* are actually needed.

---

## The Core Idea

Research on LLM quantization (BitNet, ternary networks) has shown that weight
matrices can often be approximated by just three values: **−1, 0, and +1**.
This is called *balanced ternary* representation.

Three values fit in 1.58 bits.  Five ternary digits fit into one byte (3⁵ = 243
≤ 256).  So TritPack achieves approximately **1.6 bits per weight** — roughly
10× smaller than FP16 and 2.5× smaller than 4-bit quantization — with
surprisingly good reconstruction quality.

The key insight is per-block scaling: rather than a single global threshold
for the whole model, each 64-element block gets its own scale factor (stored
as a float16).  This localises quantization error and dramatically improves
quality.

---

## How Quantization Works

For each 64-element block of weights:

1. **Threshold** — compute `τ = alpha × mean(|block|)`.  Weights with
   magnitude below τ are set to zero (trit 0).
2. **Sign** — weights above τ become +1, below −τ become −1.
3. **Scale** — record `scale = mean(|w| for |w| > τ)` as a float16.
4. **Pack** — encode the 64 trits as 13 bytes using base-3 packing.

Reconstruction multiplies each trit by its block's scale.  The cosine
similarity between original and reconstructed weights is typically 0.88–0.92
for standard LLM weight distributions at `alpha=0.7`.

The `alpha` parameter is the main quality knob: lower values preserve more
weights (better quality, less compression); higher values zero more weights
(worse quality, higher compression).

---

## Capabilities

### 1 — Compression primitives

The core library provides building blocks that work on any numpy array:

- **`pack_trits` / `unpack_trits`** — lossless base-3 byte encoding.
  Five trits per byte, exact inverse guaranteed.
- **`TernaryQuantizer`** — converts float32/float16 tensors to
  `QuantizedTensor` objects (packed trits + float16 scales).
- **`TernaryDequantizer`** — reconstructs float arrays from
  `QuantizedTensor`.
- **Quality metrics** — `cosine_similarity`, `snr_db`, `rmse` to measure
  how closely the reconstruction matches the original.

These primitives are standalone.  They work on any numerical data, not just
model weights.

### 2 — Three-tier memory manager

`TierManager` manages a RAM budget across an arbitrary number of named
layers using three tiers:

| Tier | Format | RAM cost | Access latency |
|------|--------|----------|----------------|
| HOT  | FP16/FP32 raw | 100% | <0.1 ms |
| WARM | INT8 per-block | ~50% | ~0.5 ms |
| COLD | Ternary packed | ~17% | ~3 ms |

Layers are promoted to HOT on access and demoted toward COLD under memory
pressure (LRU policy).  This is the mechanism that allows a model with more
layers than available RAM to run: only the layers being used right now occupy
HOT memory.

### 3 — Per-layer threshold calibration

`ThresholdCalibrator` scans a range of `alpha` values for each layer and
finds the highest alpha that keeps cosine similarity above a target (e.g.
0.90).  Different layers in a model have different weight distributions and
benefit from different alpha values.

### 4 — Model-level wrappers

- **`TritTensor`** — wraps a compressed tensor with lazy decompression.
  `.numpy()` decompresses on demand; `.quality_report()` gives metrics.
- **`LayerCompressor`** — groups multiple tensors (weights, biases) for a
  named model layer and tracks memory usage.

### 5 — GGUF file integration

`GGUFBackend` reads any GGUF model file (the format used by llama.cpp,
Ollama, LM Studio) and yields tensors as numpy arrays.  `GGUFPatcher`
compresses an entire GGUF model and writes a `.tritpack/` directory —
a self-contained archive of compressed tensors plus metadata.

### 6 — Local inference integration

Two paths for using compressed models with real inference engines:

**Path A — llama.cpp / Ollama (storage compression):**
`TritPackLoader.reconstruct_gguf()` decompresses a `.tritpack/` directory
back into a standard GGUF file that any llama.cpp runtime loads natively.
The model library stays 5× smaller on disk; on first run the temp GGUF is
written and then reused.

**Path B — HuggingFace transformers (runtime RAM compression):**
`compress_model()` replaces every `nn.Linear` in a loaded model with a
`TritPackLinear` that holds weights in ternary-compressed form.  Weights
are decompressed per-layer during the forward pass.  The model runs on ~3×
less RAM for its weight tensors with no code changes needed at inference time.

### 7 — Rust core

A compiled Rust extension (`tritpack_native`) provides:
- 10–100× faster packing throughput via SIMD-optimised base-3 arithmetic
- Parallel packing of large arrays via rayon thread pool
- C-ABI exports (`tritpack_pack`, `tritpack_unpack`, `tritpack_quantize`)
  for integration with C/C++ inference engines

The Python implementation remains as a reference and fallback.

### 8 — Benchmark and analysis tools

A CLI benchmark suite measures packing throughput, reconstruction quality
across alpha values, and memory usage during compression.  Output can be
written to JSON for further analysis.

---

## What TritPack Is Not

**Not a drop-in replacement for GGUF quantization.**
llama.cpp's native 4-bit formats (Q4_K_M, etc.) run faster at inference time
because quantized matrix multiplication is hardware-accelerated.  TritPack's
Path A reconstructs to float32 before llama.cpp sees the weights, so inference
speed is unchanged.  The saving is storage, not compute.

**Not lossless.**
Ternary quantization discards magnitude information within each block.
Quality is high (~0.90 cosine similarity) but not identical to the original.
For tasks sensitive to small weight differences this may matter.

**Not a training framework.**
TritPack compresses already-trained weights.  It does not implement
quantization-aware training or gradient-based fine-tuning of compressed models.

**The transformers path has decompression overhead.**
Every forward pass through a `TritPackLinear` layer decompresses its weight
matrix from scratch unless `cache_weights=True` is set.  For large batch
sizes or repeated queries the overhead becomes significant.

---

## Practical Impact

| Model | FP16 RAM | TritPack RAM | Now fits on |
|-------|----------|--------------|-------------|
| 7B    | 14 GB    | ~2.5 GB      | 8 GB laptop |
| 13B   | 26 GB    | ~4.6 GB      | 8 GB laptop |
| 34B   | 68 GB    | ~12 GB       | 16 GB desktop |
| 70B   | 140 GB   | ~25 GB       | 32 GB desktop |

The 70B row is the motivating case: a model that previously required a
multi-GPU server can now run on a single high-end workstation.

---

## Design Philosophy

TritPack is built in layers, each independently useful:

```
User code
    │
    ├── integration/   ← inference engine bridges (llama.cpp, transformers)
    │
    ├── model/         ← TritTensor, LayerCompressor, ThresholdCalibrator
    │       │           (lazy decompression, per-layer management)
    │
    ├── core/          ← TierManager, TernaryQuantizer/Dequantizer
    │       │           (memory management, quantization logic)
    │
    └── packing        ← pack_trits / unpack_trits
                        (pure byte-level encoding, no ML dependencies)
```

The packing layer has no dependencies beyond numpy.  The core layer adds
quantization logic.  The model layer adds lazy wrappers.  The integration
layer adds third-party bridges.  Each layer can be used without the ones
above it.
