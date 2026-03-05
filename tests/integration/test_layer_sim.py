"""
Integration tests: layer-level simulation for LLM compression pipelines.
"""

from __future__ import annotations

import numpy as np
import pytest

from tritpack.core.dequantizer import cosine_similarity
from tritpack.model.calibration import ThresholdCalibrator
from tritpack.model.layer import LayerCompressor


def gaussian(shape, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape).astype(np.float32)


def test_layer_compressor_stores_and_retrieves():
    """LayerCompressor must store and reconstruct tensors with cos_sim > 0.85."""
    lc = LayerCompressor("transformer.layer.0")
    weight = gaussian((4096, 4096), seed=0)
    lc.add_tensor("weight", weight)
    rec = lc.get_tensor("weight")
    cs = cosine_similarity(weight, rec)
    assert cs > 0.85


def test_layer_compressor_multiple_tensors():
    """Multiple tensors can be stored in one LayerCompressor."""
    lc = LayerCompressor("layer0")
    names = ["q_proj", "k_proj", "v_proj", "o_proj"]
    for i, name in enumerate(names):
        lc.add_tensor(name, gaussian((512, 512), seed=i))
    assert sorted(lc.tensor_names()) == sorted(names)
    assert lc.total_compressed_bytes() > 0


def test_calibrate_model_layers():
    """calibrate_model_layers must return an alpha per layer."""
    layers = {
        "q_proj": gaussian((1024, 1024), seed=0),
        "v_proj": gaussian((1024, 1024), seed=1),
    }
    cal = ThresholdCalibrator()
    alphas = cal.calibrate_model_layers(layers, target_cosine_sim=0.95)
    assert set(alphas.keys()) == set(layers.keys())
    for name, alpha in alphas.items():
        assert 0.3 <= alpha <= 1.0, f"alpha={alpha} out of expected range for {name}"
