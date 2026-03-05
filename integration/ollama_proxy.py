"""
TritPack Ollama proxy — experimental memory-compressed inference bridge.

Wraps the Ollama Python client to transparently compress model weights
loaded by a local Ollama server using TritPack's tier manager.

This is a proof-of-concept; Ollama does not expose per-layer weight
tensors via its REST API, so direct weight interception is not possible
without patching Ollama's internals.  This module provides:

  1. A utility to estimate memory savings for an Ollama model.
  2. A placeholder for future integration once Ollama exposes plugin hooks.

Usage
-----
    from integration.ollama_proxy import OllamaMemoryEstimator
    est = OllamaMemoryEstimator("llama3:8b")
    print(est.estimate())
"""

from __future__ import annotations

import warnings


class OllamaMemoryEstimator:
    """
    Estimate TritPack memory savings for an Ollama model.

    Parameters
    ----------
    model_name : str
        Ollama model identifier (e.g. ``"llama3:8b"``).
    alpha : float
        Assumed ternary quantization threshold.
    """

    # Approximate parameter counts for common model families (in billions)
    _PARAM_COUNTS = {
        "7b": 7e9,
        "8b": 8e9,
        "13b": 13e9,
        "34b": 34e9,
        "70b": 70e9,
    }

    def __init__(self, model_name: str, alpha: float = 0.7) -> None:
        self.model_name = model_name
        self.alpha = alpha

    def estimate(self) -> dict:
        """
        Return estimated memory usage before and after TritPack compression.

        Returns
        -------
        dict
            ``{model, params_b, fp16_gb, tritpack_gb, ratio}``
        """
        params = self._guess_params()
        fp16_gb = params * 2 / (1 << 30)  # 2 bytes per FP16 param
        # Ternary: ~0.2 bytes/trit + per-block scales (~0.03 bytes/param)
        tritpack_gb = params * (1 / 5 + 1 / 64 * 2) / (1 << 30)
        ratio = fp16_gb / tritpack_gb if tritpack_gb > 0 else 1.0

        return {
            "model": self.model_name,
            "params_b": round(params / 1e9, 1),
            "fp16_gb": round(fp16_gb, 1),
            "tritpack_gb": round(tritpack_gb, 1),
            "ratio": round(ratio, 1),
        }

    def _guess_params(self) -> float:
        name_lower = self.model_name.lower()
        for key, count in self._PARAM_COUNTS.items():
            if key in name_lower:
                return count
        warnings.warn(
            f"Unknown model size in '{self.model_name}'; assuming 7B params.",
            stacklevel=2,
        )
        return 7e9
