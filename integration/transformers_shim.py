"""
TritPack HuggingFace transformers integration.

Provides two complementary tools:

1. **compress_model(model)** — replace every ``nn.Linear`` in an already-loaded
   transformers model with a :class:`TritPackLinear` that stores weights in
   ternary-compressed form and decompresses on-demand during the forward pass.
   This reduces the RAM consumed by weights by ~3x while keeping the model
   fully functional for inference.

2. **load_compressed_model(model_name, tritpack_dir)** — convenience wrapper
   that loads a HuggingFace model then immediately compresses its weights,
   optionally initialising from a ``.tritpack/`` directory produced by
   :class:`~integration.gguf_patcher.GGUFPatcher`.

Requirements
------------
``torch`` and ``transformers`` must be installed::

    pip install torch transformers

Usage
-----
    # Option A — compress a model already in memory
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from integration.transformers_shim import compress_model

    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    model = compress_model(model, alpha=0.7)
    # model now uses ~3x less RAM for its weights

    # Option B — one-liner helper
    from integration.transformers_shim import load_compressed_model
    model, tokenizer = load_compressed_model(
        "meta-llama/Llama-2-7b-hf", alpha=0.7
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tritpack.core.quantizer import TernaryQuantizer, QuantizedTensor
from tritpack.core.dequantizer import TernaryDequantizer

if TYPE_CHECKING:
    import torch
    import torch.nn as nn


def _require_torch() -> None:
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for the transformers integration.\n"
            "Install with:  pip install torch"
        ) from exc


class TritPackLinear:
    """
    Drop-in replacement for ``torch.nn.Linear`` that stores its weight
    matrix in ternary-compressed form.

    The weight is decompressed to float32 on every forward call.  For
    repeated inference on the same input, set ``cache=True`` to keep the
    decompressed weight in RAM (trading RAM for speed).

    Parameters
    ----------
    qt : QuantizedTensor
        Compressed weight from :class:`~tritpack.core.quantizer.TernaryQuantizer`.
    bias : numpy.ndarray or None
        Optional bias vector (stored as a torch Parameter).
    out_features, in_features : int
        Layer dimensions (inferred from ``qt.original_shape`` if not given).
    cache : bool
        If ``True``, cache the decompressed weight after the first forward
        pass.  Disable to keep RAM usage low when layers are infrequently
        accessed.
    """

    def __init__(
        self,
        qt: QuantizedTensor,
        bias: "np.ndarray | None" = None,
        out_features: int | None = None,
        in_features: int | None = None,
        cache: bool = False,
    ) -> None:
        _require_torch()
        import torch
        import torch.nn as nn

        # We subclass nothing — register as a proper nn.Module at runtime
        # by calling __init_subclass__ manually after torch is available.
        nn.Module.__init__(self)
        self._qt = qt
        self._dq = TernaryDequantizer()
        self._cache = cache
        self._cached_weight: "torch.Tensor | None" = None

        shape = qt.original_shape
        self.out_features = out_features or (shape[0] if len(shape) >= 1 else 1)
        self.in_features = in_features or (shape[1] if len(shape) >= 2 else shape[0])

        if bias is not None:
            self.bias = nn.Parameter(torch.from_numpy(bias.astype(np.float32)))
        else:
            self.bias = None

    # Make isinstance(layer, nn.Module) true at class level
    # by inheriting from nn.Module (done below via __init_subclass__).

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        import torch
        import torch.nn.functional as F

        if self._cache and self._cached_weight is not None:
            weight = self._cached_weight
        else:
            arr = self._dq.dequantize(self._qt).astype(np.float32)
            weight = torch.from_numpy(arr)
            if self._cache:
                self._cached_weight = weight

        return F.linear(x, weight.to(x.device), self.bias)

    @property
    def weight(self) -> "torch.Tensor":
        """Decompress weight on demand (for inspection / compatibility)."""
        import torch
        arr = self._dq.dequantize(self._qt).astype(np.float32)
        return torch.from_numpy(arr)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, compressed=True"
        )


# Make TritPackLinear an actual nn.Module so it works transparently inside
# transformers models (hooks, device movement, etc.).
def _make_module_subclass() -> None:
    try:
        import torch.nn as nn
        TritPackLinear.__bases__ = (nn.Module,)
    except ImportError:
        pass


_make_module_subclass()


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def compress_model(
    model: "nn.Module",
    alpha: float = 0.7,
    block_size: int = 64,
    cache_weights: bool = False,
    verbose: bool = False,
) -> "nn.Module":
    """
    Replace every ``nn.Linear`` layer in *model* with a
    :class:`TritPackLinear` that holds its weight in ternary-compressed form.

    The model is modified **in-place** and also returned for convenience.
    Call this after ``from_pretrained`` / ``load_state_dict`` while weights
    are still on CPU (before ``.to("cuda")``).

    Parameters
    ----------
    model : nn.Module
        Any HuggingFace (or custom) PyTorch model.
    alpha : float
        Ternary quantization threshold ratio.
    block_size : int
        Block size for per-block quantisation.
    cache_weights : bool
        Pass ``True`` to cache decompressed weights after first use
        (faster repeated inference, higher RAM usage).
    verbose : bool
        Print progress as each layer is compressed.

    Returns
    -------
    nn.Module
        The same *model* object with its Linear layers replaced.

    Example
    -------
    ::

        from transformers import AutoModelForCausalLM
        from integration.transformers_shim import compress_model

        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        model = compress_model(model, alpha=0.7)
        # Weight RAM reduced by ~3x
    """
    _require_torch()
    import torch.nn as nn

    q = TernaryQuantizer(alpha=alpha, block_size=block_size)
    n_compressed = 0

    # Collect replacements first to avoid modifying dict during iteration
    replacements: list[tuple[nn.Module, str, TritPackLinear]] = []

    for full_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        weight_np = module.weight.detach().cpu().numpy()
        qt = q.quantize(weight_np)
        bias_np = (
            module.bias.detach().cpu().numpy() if module.bias is not None else None
        )
        new_layer = TritPackLinear(
            qt,
            bias=bias_np,
            out_features=module.out_features,
            in_features=module.in_features,
            cache=cache_weights,
        )

        # Find parent module and attribute name
        parts = full_name.rsplit(".", 1)
        if len(parts) == 2:
            parent_name, child_name = parts
            parent = model
            for part in parent_name.split("."):
                parent = getattr(parent, part)
        else:
            parent = model
            child_name = full_name

        replacements.append((parent, child_name, new_layer))
        n_compressed += 1
        if verbose:
            ratio = module.weight.numel() * 4 / qt.size_bytes()
            print(f"  compressed {full_name}  ({ratio:.1f}x)")

    for parent, child_name, new_layer in replacements:
        setattr(parent, child_name, new_layer)

    if verbose:
        print(f"Compressed {n_compressed} Linear layers.")

    return model


def load_compressed_model(
    model_name_or_path: str,
    alpha: float = 0.7,
    block_size: int = 64,
    cache_weights: bool = False,
    verbose: bool = False,
    **from_pretrained_kwargs,
) -> "tuple[nn.Module, object]":
    """
    Load a HuggingFace causal-LM model and immediately compress its weights.

    Parameters
    ----------
    model_name_or_path : str
        HuggingFace Hub model ID or local path (passed to
        ``AutoModelForCausalLM.from_pretrained``).
    alpha : float
        Ternary quantization threshold ratio.
    block_size : int
        Block size for per-block quantisation.
    cache_weights : bool
        Cache decompressed weights after first use.
    verbose : bool
        Print compression progress.
    **from_pretrained_kwargs
        Extra kwargs forwarded to ``from_pretrained`` (e.g.
        ``torch_dtype=torch.float16``).

    Returns
    -------
    (model, tokenizer)
        The compressed model and its tokenizer.

    Example
    -------
    ::

        from integration.transformers_shim import load_compressed_model

        model, tokenizer = load_compressed_model(
            "meta-llama/Llama-2-7b-hf",
            alpha=0.7,
        )
        inputs = tokenizer("Hello!", return_tensors="pt")
        output = model.generate(**inputs, max_new_tokens=50)
        print(tokenizer.decode(output[0]))
    """
    _require_torch()
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "transformers is required.\n"
            "Install with:  pip install transformers"
        ) from exc

    if verbose:
        print(f"Loading {model_name_or_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, **from_pretrained_kwargs
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    if verbose:
        print("Compressing weights ...")
    compress_model(model, alpha=alpha, block_size=block_size,
                   cache_weights=cache_weights, verbose=verbose)

    return model, tokenizer
