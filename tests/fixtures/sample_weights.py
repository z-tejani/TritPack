"""
Sample weight tensors simulating real LLM layer dimensions.
"""

from __future__ import annotations

import numpy as np


# Approximate weight shapes from common LLM architectures

LLAMA_7B_SHAPES = {
    "model.embed_tokens.weight": (32000, 4096),
    "model.layers.0.self_attn.q_proj.weight": (4096, 4096),
    "model.layers.0.self_attn.k_proj.weight": (4096, 4096),
    "model.layers.0.self_attn.v_proj.weight": (4096, 4096),
    "model.layers.0.self_attn.o_proj.weight": (4096, 4096),
    "model.layers.0.mlp.gate_proj.weight": (11008, 4096),
    "model.layers.0.mlp.up_proj.weight": (11008, 4096),
    "model.layers.0.mlp.down_proj.weight": (4096, 11008),
}

LLAMA_70B_SHAPES = {
    "model.layers.0.self_attn.q_proj.weight": (8192, 8192),
    "model.layers.0.self_attn.k_proj.weight": (1024, 8192),
    "model.layers.0.self_attn.v_proj.weight": (1024, 8192),
    "model.layers.0.self_attn.o_proj.weight": (8192, 8192),
    "model.layers.0.mlp.gate_proj.weight": (28672, 8192),
    "model.layers.0.mlp.up_proj.weight": (28672, 8192),
    "model.layers.0.mlp.down_proj.weight": (8192, 28672),
}


def make_sample_layer(shape: tuple, seed: int = 0) -> np.ndarray:
    """Return a Gaussian float32 tensor with *shape* and *seed*."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape).astype(np.float32)


def make_7b_layers(seed: int = 0) -> dict[str, np.ndarray]:
    """Generate one set of 7B-scale weight tensors."""
    return {
        name: make_sample_layer(shape, seed=seed + i)
        for i, (name, shape) in enumerate(LLAMA_7B_SHAPES.items())
    }
