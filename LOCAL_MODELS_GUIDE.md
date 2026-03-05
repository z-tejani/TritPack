# Using TritPack with Local Models

This guide covers the two most common ways to run compressed models locally:

- **Ollama / LM Studio / llama.cpp** — via GGUF file reconstruction (Path A)
- **HuggingFace transformers** — via in-process weight compression (Path B)

---

## Prerequisites

```bash
pip install tritpack

# For GGUF support (required for Path A)
pip install gguf

# For Path B (HuggingFace)
pip install torch transformers

# For llama-cpp-python integration
pip install llama-cpp-python
```

---

## Path A — Ollama and LM Studio (via GGUF)

Both Ollama and LM Studio use GGUF files under the hood.  TritPack compresses
a GGUF file into a `.tritpack/` directory (about 5× smaller) and reconstructs
it back to a GGUF when you want to run it.

### Step 1 — Get your GGUF file

**From Ollama** — find where Ollama stores its blobs:

```bash
# Linux / macOS
ls ~/.ollama/models/blobs/

# The blob files are the actual GGUF data.
# Find the one for your model:
ollama show llama3.2 --modelfile
# Look for the FROM line — that path is the GGUF (or a symlink to it)
```

**From LM Studio** — models are stored in:

```
~/LM Studio/models/<author>/<model-name>/<file>.gguf
# Example:
~/LM Studio/models/lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
```

**From HuggingFace directly** — download a GGUF-format repo:

```bash
pip install huggingface-hub
huggingface-cli download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF \
    --include "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf" \
    --local-dir ./models/llama3.1-8b
```

> **Note:** TritPack is most useful for **F16 or F32 GGUFs** (unquantized).
> Applying ternary compression on top of an already-quantised Q4 file gives
> diminishing returns — the weights are already coarsely bucketed.  Download
> the `F16` variant when available.

### Step 2 — Compress the GGUF

```python
from integration.gguf_patcher import GGUFPatcher

patcher = GGUFPatcher(alpha=0.7, block_size=64)

report = patcher.patch(
    input_path="./models/llama3.1-8b/Meta-Llama-3.1-8B-Instruct-F16.gguf",
    output_path="./models/llama3.1-8b-tritpack/",   # becomes a directory
)

print(f"Original:   {report.original_gb:.2f} GB")
print(f"Compressed: {report.compressed_gb:.2f} GB")
print(f"Ratio:      {report.ratio:.1f}×")
```

This writes a `./models/llama3.1-8b-tritpack/` directory containing:
- `metadata.json` — model info and compression config
- one `.npz` file per tensor (packed trits + float16 scales)

Compression takes a few minutes.  CPU-only, no GPU required.

**Tuning `alpha`:**

| alpha | Quality (cos sim) | Compression |
|-------|------------------|-------------|
| 0.5   | ~0.94            | ~4×         |
| 0.7   | ~0.90            | ~5-6×       |
| 0.9   | ~0.85            | ~7-8×       |

Lower alpha = better quality, less compression.  `0.7` is a good default.

**Verifying quality after compression:**

```python
report = patcher.verify(
    original_path="./models/llama3.1-8b/Meta-Llama-3.1-8B-Instruct-F16.gguf",
    patched_path="./models/llama3.1-8b-tritpack/",
)
# report contains per-layer cos_sim, snr_db, rmse
```

### Step 3 — Reconstruct and run with Ollama or LM Studio

TritPack cannot inject weights directly into Ollama or LM Studio at runtime —
both treat models as opaque GGUF files.  The workflow is:

1. Decompress the `.tritpack/` directory back to a GGUF
2. Point Ollama or LM Studio at that GGUF

```python
from integration.tritpack_loader import TritPackLoader

loader = TritPackLoader("./models/llama3.1-8b-tritpack/")

loader.reconstruct_gguf(
    output_path="./models/llama3.1-8b-reconstructed.gguf"
)
```

The reconstructed GGUF is a standard float32 GGUF that any llama.cpp-based
runtime accepts.

**Ollama — create a custom model from the reconstructed GGUF:**

```bash
# Create a Modelfile
cat > Modelfile <<'EOF'
FROM ./models/llama3.1-8b-reconstructed.gguf
PARAMETER num_ctx 4096
EOF

ollama create llama3.1-8b-tritpack -f Modelfile
ollama run llama3.1-8b-tritpack
```

**LM Studio — load the reconstructed GGUF directly:**

Open LM Studio → "My Models" tab → click the folder icon → navigate to
`llama3.1-8b-reconstructed.gguf` and select it.  It will appear in the model
list immediately.

**Disk vs. RAM trade-off:**

The `.tritpack/` directory is the compressed archive (5-8× smaller than the
original GGUF).  Keep it as your long-term storage.  Generate the reconstructed
GGUF only when you want to run the model — it is the full float32 size.

```
./models/
├── llama3.1-8b-tritpack/          ← keep this (~2.5 GB for 8B-F16)
│   ├── metadata.json
│   └── *.npz
└── llama3.1-8b-reconstructed.gguf ← generate when needed (~14 GB)
```

### Optional — llama-cpp-python (no Ollama/LM Studio needed)

If you use llama-cpp-python directly you can skip the manual reconstruction:

```python
from integration.llamacpp_shim import load_from_tritpack

llm = load_from_tritpack(
    tritpack_dir="./models/llama3.1-8b-tritpack/",
    temp_gguf="./models/llama3.1-8b-temp.gguf",
    reuse_temp=True,        # skip reconstruction on second run
    verbose=True,
    n_ctx=4096,
    n_gpu_layers=0,
)

output = llm("What is the capital of France?", max_tokens=64)
print(output["choices"][0]["text"])
```

With `reuse_temp=True` the GGUF is only written once; subsequent calls load
it directly.

**With memory-tier management** (useful when RAM is tight):

```python
from integration.llamacpp_shim import LlamaCppTritShim

shim = LlamaCppTritShim(
    model_path="./models/llama3.1-8b-tritpack/",
    memory_limit_gb=8.0,        # your available RAM
    hot_budget_fraction=0.15,   # 1.2 GB for active layers
    warm_budget_fraction=0.25,  # 2.0 GB for recently used
    temp_gguf="./models/llama3.1-8b-temp.gguf",
    reuse_temp=True,
)

llm = shim.load(verbose=True, n_ctx=4096)
print(shim.tier_stats())        # shows HOT/WARM/COLD layer counts
```

---

## Path B — HuggingFace Models (in-process)

This path loads a model with the `transformers` library and replaces every
linear layer with a compressed `TritPackLinear`.  Weights stay compressed in
RAM and are decompressed per-layer during inference.

No GGUF conversion is needed.  Works with any `AutoModelForCausalLM`-compatible
model.

### Step 1 — Download a model from HuggingFace

```bash
huggingface-cli download meta-llama/Llama-3.2-3B-Instruct \
    --local-dir ./models/llama3.2-3b
```

Or let `transformers` download on first use (see below).

> **Gated models** (Llama, Mistral, Gemma): you need a HuggingFace account
> and must accept the model's licence at hf.co/settings/tokens before
> downloading.  Run `huggingface-cli login` once to authenticate.

### Step 2 — Load and compress in one call

```python
from integration.transformers_shim import load_compressed_model

model, tokenizer = load_compressed_model(
    "meta-llama/Llama-3.2-3B-Instruct",
    alpha=0.7,
    block_size=64,
    cache_weights=False,    # True = faster inference, more RAM
    verbose=True,
    torch_dtype="auto",     # pass any from_pretrained kwargs here
)
```

`load_compressed_model` is a thin wrapper that calls
`AutoModelForCausalLM.from_pretrained()` then replaces all `nn.Linear` layers
in-place.

**If you already have a loaded model:**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from integration.transformers_shim import compress_model

model = AutoModelForCausalLM.from_pretrained(
    "./models/llama3.2-3b",
    torch_dtype="auto",
)
tokenizer = AutoTokenizer.from_pretrained("./models/llama3.2-3b")

compress_model(model, alpha=0.7, block_size=64, verbose=True)
# model is modified in-place; also returned for convenience
```

### Step 3 — Run inference normally

After compression the model's API is unchanged:

```python
inputs = tokenizer("The capital of France is", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

Each forward pass decompresses only the layers it touches — the trit-packed
weights stay in RAM between calls.

**`cache_weights=True`** — decompressed weights are cached after the first
forward pass, trading RAM for speed.  Use this if you make many inference
calls and have enough headroom:

```python
compress_model(model, alpha=0.7, cache_weights=True)
```

### RAM usage comparison

| Model    | FP16 (uncompressed) | TritPack Path B |
|----------|---------------------|-----------------|
| 3B       | ~6 GB               | ~2 GB           |
| 8B       | ~16 GB              | ~5.5 GB         |
| 13B      | ~26 GB              | ~9 GB           |
| 70B      | ~140 GB             | ~48 GB*         |

*70B at Path B is slower than Path A because decompression happens per-layer,
per-forward-pass.  Use Path A (reconstructed GGUF) for 70B+ models.

---

## Choosing Between Path A and Path B

| | Path A (GGUF → Ollama / LM Studio) | Path B (HuggingFace in-process) |
|---|---|---|
| **Frontend** | Ollama, LM Studio, any llama.cpp runner | Python only |
| **Setup** | Compress once, reconstruct when needed | Load and compress once per session |
| **Inference speed** | Native llama.cpp (GPU-optimised) | PyTorch (slower without GPU) |
| **RAM at inference** | Full float32 reconstructed model | ~3× compressed weights |
| **Best for** | Running models through a UI or chat app | Python pipelines, fine-tuning prep, research |
| **GPU acceleration** | Yes (llama.cpp offloads layers to GPU) | Yes (standard torch CUDA) |

---

## Benchmarking Your Compression

```bash
# Check packing throughput
python -m tritpack.benchmark bench-pack

# Check reconstruction quality across alpha values
python -m tritpack.benchmark bench-quality

# Full report saved to JSON
python -m tritpack.benchmark bench-all --output results.json
```

Or from Python against your specific model tensors:

```python
from integration.gguf_patcher import GGUFPatcher

patcher = GGUFPatcher()
report = patcher.verify(
    original_path="./models/mymodel.gguf",
    patched_path="./models/mymodel-tritpack/",
)
# inspect report.original_gb, report.compressed_gb, per-layer cos_sim
```

---

## Per-Layer Alpha Calibration

If default quality is insufficient for your use case, calibrate `alpha`
per layer to maximise compression while keeping cosine similarity above a
threshold:

```python
import numpy as np
from python.tritpack.backends.gguf_backend import GGUFBackend
from python.tritpack.model.calibration import ThresholdCalibrator

backend = GGUFBackend("./models/mymodel.gguf")
layers = {name: tensor for name, tensor in backend.iter_tensors()}

calibrator = ThresholdCalibrator(block_size=64)
optimal_alphas = calibrator.calibrate_model_layers(
    layers,
    target_cosine_sim=0.92,   # raise for better quality, lower for smaller size
)

# optimal_alphas is {layer_name: alpha_float}
# Pass these per-layer into a custom compression loop
print(optimal_alphas)
```

---

## Troubleshooting

**`ImportError: No module named 'gguf'`**
Install the optional GGUF package: `pip install gguf`

**`ImportError: No module named 'llama_cpp'`**
Install llama-cpp-python: `pip install llama-cpp-python`
For GPU support follow the [llama-cpp-python install guide](https://github.com/abetlen/llama-cpp-python#installation-with-specific-hardware-acceleration).

**Ollama can't find the reconstructed GGUF**
Make sure the path in the `Modelfile` is an absolute path, or run
`ollama create` from the directory containing the GGUF.

**Quality is lower than expected**
Try a lower `alpha` (e.g. `0.5`).  You can also use `ThresholdCalibrator`
to find the optimal per-layer alpha for your target cosine similarity.

**Compression is slow**
The pure Python implementation is single-threaded.  Build the Rust extension
for 10–100× faster throughput:
```bash
pip install maturin
cd rust/
maturin develop --release
```

**OllamaMemoryEstimator — why is there no live integration?**
Ollama's REST API does not expose per-layer weights.  `ollama_proxy.py`
provides RAM savings *estimates* only.  Use Path A (GGUF compression +
Modelfile) for real integration with Ollama.
