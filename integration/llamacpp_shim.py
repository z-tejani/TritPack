"""
TritPack llama.cpp integration.

Two usage modes:

1. **From a .tritpack/ directory** (recommended):
   Pre-compress a GGUF file with :class:`~integration.gguf_patcher.GGUFPatcher`,
   then call :func:`load_from_tritpack` to decompress weights at load time and
   pass them to llama.cpp via a temporary reconstructed GGUF.  The temporary
   file is written once and reused on subsequent loads if ``reuse_temp`` is set.

2. **Direct GGUF** (memory monitoring only):
   Pass a plain ``.gguf`` file to :class:`LlamaCppTritShim`.  The model loads
   normally and a :class:`~tritpack.core.tiers.TierManager` is attached for
   future layer-level memory management.

Dependencies
------------
``llama-cpp-python`` must be installed::

    pip install llama-cpp-python

Usage
-----
    # Recommended — compress once, load fast afterwards
    from integration.llamacpp_shim import load_from_tritpack

    model = load_from_tritpack(
        tritpack_dir="llama-7b.tritpack/",
        temp_gguf="/tmp/llama-7b-reconstructed.gguf",
        reuse_temp=True,   # skip reconstruction on second run
    )
    output = model("Hello, I am", max_tokens=50)
    print(output["choices"][0]["text"])
"""

from __future__ import annotations

import os
import warnings
from typing import Any

try:
    import llama_cpp  # type: ignore
    _LLAMA_CPP_AVAILABLE = True
except ImportError:
    _LLAMA_CPP_AVAILABLE = False

from tritpack.core.tiers import TierManager


def _require_llama_cpp() -> None:
    if not _LLAMA_CPP_AVAILABLE:
        raise ImportError(
            "llama-cpp-python is required.\n"
            "Install with:  pip install llama-cpp-python"
        )


def load_from_tritpack(
    tritpack_dir: str,
    temp_gguf: str,
    reuse_temp: bool = True,
    verbose: bool = False,
    **llama_kwargs,
) -> Any:
    """
    Load a llama.cpp model from a ``.tritpack/`` compressed directory.

    On first call the weights are decompressed and written to *temp_gguf*
    (a standard GGUF file).  On subsequent calls the existing file is reused
    when ``reuse_temp=True``, so startup is fast after the first run.

    Parameters
    ----------
    tritpack_dir : str
        Path to the directory produced by
        :class:`~integration.gguf_patcher.GGUFPatcher`.
    temp_gguf : str
        Path where the reconstructed GGUF will be written.
    reuse_temp : bool
        If ``True`` and *temp_gguf* already exists, skip reconstruction.
    verbose : bool
        Print progress messages.
    **llama_kwargs
        Extra kwargs forwarded to ``llama_cpp.Llama()``.

    Returns
    -------
    llama_cpp.Llama
        A fully functional Llama instance.

    Example
    -------
    ::

        model = load_from_tritpack(
            "llama-7b.tritpack/",
            "/tmp/llama-7b-reconstructed.gguf",
        )
        output = model("The capital of France is", max_tokens=10)
        print(output["choices"][0]["text"])
    """
    _require_llama_cpp()
    from integration.tritpack_loader import TritPackLoader

    if reuse_temp and os.path.exists(temp_gguf):
        if verbose:
            print(f"Reusing existing reconstructed GGUF: {temp_gguf}")
    else:
        if verbose:
            print(f"Reconstructing GGUF from {tritpack_dir} → {temp_gguf} ...")
        loader = TritPackLoader(tritpack_dir)
        loader.reconstruct_gguf(temp_gguf)
        if verbose:
            ratio = loader.compression_ratio
            print(f"Done.  Compression ratio was {ratio:.1f}x.")

    if verbose:
        print("Loading model with llama.cpp ...")
    return llama_cpp.Llama(model_path=temp_gguf, **llama_kwargs)


class LlamaCppTritShim:
    """
    Wrapper around ``llama_cpp.Llama`` with TritPack memory management.

    For full weight compression, prefer :func:`load_from_tritpack`.
    This class is useful when you want to attach TritPack's tier manager
    to an existing llama.cpp model for monitoring or future layer control.

    Parameters
    ----------
    model_path : str
        Path to a ``.gguf`` model file **or** a ``.tritpack/`` directory.
        When a ``.tritpack/`` directory is given, *temp_gguf* is required.
    memory_limit_gb : float
        Total RAM budget in gigabytes for tier management.
    temp_gguf : str or None
        Where to write the reconstructed GGUF when *model_path* is a
        ``.tritpack/`` directory.
    reuse_temp : bool
        Reuse the reconstructed GGUF if it already exists.
    """

    def __init__(
        self,
        model_path: str,
        memory_limit_gb: float,
        hot_budget_fraction: float = 0.15,
        warm_budget_fraction: float = 0.25,
        temp_gguf: str | None = None,
        reuse_temp: bool = True,
    ) -> None:
        _require_llama_cpp()
        self.model_path = model_path
        self.temp_gguf = temp_gguf
        self.reuse_temp = reuse_temp
        self._tier_manager = TierManager(
            memory_limit_gb=memory_limit_gb,
            hot_budget_fraction=hot_budget_fraction,
            warm_budget_fraction=warm_budget_fraction,
        )

    def load(self, verbose: bool = False, **llama_kwargs) -> Any:
        """
        Load the model and attach TritPack tier management.

        If *model_path* is a ``.tritpack/`` directory the weights are
        decompressed first via :func:`load_from_tritpack`.

        Returns
        -------
        llama_cpp.Llama
            A Llama instance with ``._tritpack_tiers`` attribute attached.
        """
        path = self.model_path

        if os.path.isdir(path):
            # .tritpack/ directory — decompress to temp GGUF first
            if self.temp_gguf is None:
                raise ValueError(
                    "temp_gguf must be specified when model_path is a "
                    ".tritpack/ directory."
                )
            model = load_from_tritpack(
                path,
                self.temp_gguf,
                reuse_temp=self.reuse_temp,
                verbose=verbose,
                **llama_kwargs,
            )
        else:
            # Plain GGUF — load normally
            warnings.warn(
                "Loading a plain GGUF without pre-compression.  "
                "For memory savings, run GGUFPatcher first.",
                stacklevel=2,
            )
            model = llama_cpp.Llama(model_path=path, **llama_kwargs)

        model._tritpack_tiers = self._tier_manager
        return model

    def tier_stats(self) -> dict:
        """Return current TierManager statistics."""
        stats = self._tier_manager.stats()
        return {
            "n_hot": stats.n_hot,
            "n_warm": stats.n_warm,
            "n_cold": stats.n_cold,
            "hot_gb": stats.hot_gb,
            "warm_gb": stats.warm_gb,
            "cold_gb": stats.cold_gb,
            "total_gb": stats.total_gb,
        }
