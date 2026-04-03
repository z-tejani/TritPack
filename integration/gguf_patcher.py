"""
TritPack GGUF patcher — pre-process GGUF files for ternary compression.

Converts a standard GGUF model file to a TritPack-aware format by
quantizing all weight tensors to ternary representation and saving
metadata for fast loading.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from tritpack.core.quantizer import TernaryQuantizer
from tritpack.core.dequantizer import TernaryDequantizer, cosine_similarity, snr_db, rmse
from tritpack.benchmark.quality import QualityReport, QualityPoint


class QualityPatchReport:
    """Report from verifying a patched GGUF file."""

    def __init__(self) -> None:
        self.layers: list[dict] = []
        self.patch_time_s: float = 0.0
        self.original_gb: float = 0.0
        self.compressed_gb: float = 0.0

    @property
    def ratio(self) -> float:
        if self.compressed_gb == 0:
            return 1.0
        return self.original_gb / self.compressed_gb

    @property
    def mean_cos_sim(self) -> float:
        if not self.layers:
            return 0.0
        return float(np.mean([l["cos_sim"] for l in self.layers]))

    def summary(self) -> dict:
        return {
            "n_layers": len(self.layers),
            "original_gb": round(self.original_gb, 4),
            "compressed_gb": round(self.compressed_gb, 4),
            "ratio": round(self.ratio, 2),
            "mean_cos_sim": round(self.mean_cos_sim, 4),
            "patch_time_s": round(self.patch_time_s, 2),
        }


class GGUFPatcher:
    """
    Pre-processes a GGUF file: converts weights to ternary-packed format
    and saves as a ``.tritpack`` directory for use with TritPack-aware loaders.

    Parameters
    ----------
    alpha : float
        Ternary quantization threshold ratio.
    block_size : int
        Block size for per-block quantization.
    """

    def __init__(self, alpha: float = 0.7, block_size: int = 64) -> None:
        self.alpha = alpha
        self.block_size = block_size
        self._quantizer = TernaryQuantizer(block_size=block_size, alpha=alpha)
        self._dequantizer = TernaryDequantizer()

    def patch(
        self,
        input_path: str,
        output_path: str,
        alpha: float | None = None,
        progress_callback=None,
    ) -> QualityPatchReport:
        """
        Quantize all tensors in *input_path* and save to *output_path*.

        The output is a directory containing:
        - ``metadata.json``    — model metadata and compression config
        - ``<layer>.npz``      — one compressed file per tensor

        Parameters
        ----------
        input_path : str
            Path to input ``.gguf`` file.
        output_path : str
            Path to output directory (``.tritpack`` by convention).
        alpha : float, optional
            Override the instance *alpha* for this call.

        Returns
        -------
        QualityPatchReport
        """
        try:
            from tritpack.backends.gguf_backend import GGUFBackend
        except ImportError as e:
            raise ImportError("gguf package required: pip install gguf") from e

        from tritpack.backends.numpy_backend import save as np_save

        alpha = alpha if alpha is not None else self.alpha
        q = TernaryQuantizer(block_size=self.block_size, alpha=alpha)

        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        backend = GGUFBackend(input_path)
        meta = backend.load_metadata()

        report = QualityPatchReport()
        t0 = time.perf_counter()
        gb = 1 << 30

        tensor_names = backend.tensor_names()
        total_tensors = max(1, len(tensor_names))

        for index, (name, tensor) in enumerate(backend.iter_tensors()):
            qt = q.quantize(tensor)
            safe_name = name.replace("/", "__").replace(".", "_")
            np_save(qt, output_dir / f"{safe_name}.npz")

            report.original_gb += tensor.nbytes / gb
            report.compressed_gb += qt.size_bytes() / gb
            report.layers.append({"name": name, "shape": list(tensor.shape)})
            if progress_callback is not None:
                progress_callback(index + 1, total_tensors, name)

        report.patch_time_s = time.perf_counter() - t0

        # Write metadata
        patch_meta = {
            "source": input_path,
            "alpha": alpha,
            "block_size": self.block_size,
            "n_tensors": len(report.layers),
            "original_gb": report.original_gb,
            "compressed_gb": report.compressed_gb,
            "ratio": report.ratio,
            # Store layer names so TritPackLoader can recover original GGUF names.
            "layers": [{"name": l["name"], "shape": l["shape"]} for l in report.layers],
            "model_meta": {k: str(v) for k, v in meta.items()},
        }
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(patch_meta, f, indent=2)

        return report

    def verify(self, original_path: str, patched_path: str) -> QualityPatchReport:
        """
        Verify a patched directory by comparing reconstructed tensors
        with the originals.

        Parameters
        ----------
        original_path : str
            Path to the original ``.gguf`` file.
        patched_path : str
            Path to the ``.tritpack`` output directory.

        Returns
        -------
        QualityPatchReport
        """
        try:
            from tritpack.backends.gguf_backend import GGUFBackend
        except ImportError as e:
            raise ImportError("gguf package required") from e

        from tritpack.backends.numpy_backend import load as np_load

        backend = GGUFBackend(original_path)
        patched_dir = Path(patched_path)
        report = QualityPatchReport()
        gb = 1 << 30

        for name, tensor in backend.iter_tensors():
            safe_name = name.replace("/", "__").replace(".", "_")
            npz_path = patched_dir / f"{safe_name}.npz"
            if not npz_path.exists():
                continue

            qt = np_load(npz_path)
            rec = self._dequantizer.dequantize(qt)

            report.original_gb += tensor.nbytes / gb
            report.compressed_gb += qt.size_bytes() / gb
            report.layers.append({
                "name": name,
                "shape": list(tensor.shape),
                "cos_sim": cosine_similarity(tensor, rec),
                "snr_db": snr_db(tensor, rec),
                "rmse": rmse(tensor, rec),
            })

        return report
