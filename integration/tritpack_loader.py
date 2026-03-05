"""
TritPack model loader — load .tritpack/ directories for inference.

A ``.tritpack/`` directory is produced by :class:`~integration.gguf_patcher.GGUFPatcher`.
It contains ``metadata.json`` and one compressed ``<layer>.npz`` file per tensor.

Two primary inference paths are supported:

1. **GGUF reconstruction** (llama.cpp / Ollama):
   Decompress all tensors and write a new ``.gguf`` file that any llama.cpp
   runtime can load natively.  Slightly lower quality than the original but
   fully format-compatible.

2. **Transformers in-process patching**:
   Load a HuggingFace model, then call :func:`patch_transformers_model` to
   replace its weight tensors with decompressed arrays in-place — no extra
   files needed.

Usage
-----
    # Path A — llama.cpp via reconstructed GGUF
    from integration.tritpack_loader import TritPackLoader

    loader = TritPackLoader("llama-7b.tritpack/")
    loader.reconstruct_gguf("/tmp/llama-7b-reconstructed.gguf")

    from llama_cpp import Llama
    model = Llama("/tmp/llama-7b-reconstructed.gguf")

    # Path B — HuggingFace transformers (see transformers_shim for RAM savings)
    from integration.tritpack_loader import TritPackLoader
    loader = TritPackLoader("llama-7b.tritpack/")
    weights = loader.load_weights()   # {gguf_name: np.ndarray}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np

from tritpack.backends.numpy_backend import load as _np_load
from tritpack.core.dequantizer import TernaryDequantizer


class TritPackLoader:
    """
    Load and decompress weights from a ``.tritpack/`` directory.

    Parameters
    ----------
    tritpack_dir : str
        Path to the directory produced by :class:`~integration.gguf_patcher.GGUFPatcher`.
    """

    def __init__(self, tritpack_dir: str) -> None:
        self.dir = Path(tritpack_dir)
        meta_path = self.dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"metadata.json not found in '{tritpack_dir}'. "
                "Run GGUFPatcher.patch() to create a .tritpack/ directory first."
            )
        with open(meta_path) as f:
            self.metadata: dict = json.load(f)
        self._dq = TernaryDequantizer()
        # Build name → npz path index from stored layer list
        self._name_index: dict[str, Path] = {
            layer["name"]: self.dir / f"{self._safe(layer['name'])}.npz"
            for layer in self.metadata.get("layers", [])
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _safe(name: str) -> str:
        """Reproduce the safe filename used by GGUFPatcher."""
        return name.replace("/", "__").replace(".", "_")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tensor_names(self) -> list[str]:
        """Return original GGUF tensor names in storage order."""
        return list(self._name_index)

    @property
    def n_tensors(self) -> int:
        return len(self._name_index)

    @property
    def compression_ratio(self) -> float:
        return float(self.metadata.get("ratio", 1.0))

    def load_tensor(self, name: str) -> np.ndarray:
        """Decompress and return a single tensor by its original GGUF name."""
        path = self._name_index.get(name)
        if path is None:
            raise KeyError(f"Tensor '{name}' not found in .tritpack/ directory.")
        return self._dq.dequantize(_np_load(path))

    def iter_weights(self) -> Iterator[tuple[str, np.ndarray]]:
        """
        Lazily yield ``(name, array)`` pairs, decompressing one at a time.

        Keeps peak RAM to ~1 decompressed tensor + all compressed tensors,
        rather than all tensors decompressed simultaneously.
        """
        for name, path in self._name_index.items():
            yield name, self._dq.dequantize(_np_load(path))

    def load_weights(self) -> dict[str, np.ndarray]:
        """
        Decompress **all** tensors into a ``{name: array}`` dict.

        For large models prefer :meth:`iter_weights` to reduce peak RAM.
        """
        return dict(self.iter_weights())

    def reconstruct_gguf(self, output_path: str) -> None:
        """
        Write a GGUF file with decompressed float32 weights.

        The resulting file is fully compatible with llama.cpp, llama-cpp-python,
        and Ollama.  Weights are slightly lossy versus the original (ternary
        quantization error, cosine similarity ~0.90 with alpha=0.7).

        Parameters
        ----------
        output_path : str
            Destination path for the ``.gguf`` file.

        Raises
        ------
        ImportError
            If the ``gguf`` package is not installed
            (``pip install gguf``).
        FileNotFoundError
            If the .tritpack/ directory has no tensors.
        """
        try:
            import gguf  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "The 'gguf' package is required to reconstruct GGUF files.\n"
                "Install with:  pip install gguf"
            ) from exc

        if not self._name_index:
            raise FileNotFoundError("No tensors found in .tritpack/ directory.")

        model_meta = self.metadata.get("model_meta", {})
        arch = model_meta.get("general.architecture", "llama")
        writer = gguf.GGUFWriter(output_path, arch=arch)

        # Restore key-value metadata from the original GGUF
        for key, value in model_meta.items():
            if key == "general.name":
                writer.add_name(value)
            elif key in ("general.architecture", "general.file_type"):
                pass  # handled by GGUFWriter constructor / auto-set
            else:
                try:
                    writer.add_string(key, str(value))
                except Exception:
                    pass  # skip unrecognised keys

        # Write tensors (decompressed on-the-fly to stay memory-efficient)
        for name, array in self.iter_weights():
            writer.add_tensor(name, array.astype(np.float32))

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
