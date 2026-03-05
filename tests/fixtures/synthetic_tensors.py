"""
Synthetic tensor generators for TritPack tests.

All generators accept a *seed* parameter for reproducibility.
"""

from __future__ import annotations

import numpy as np


def gaussian_tensor(shape: tuple, seed: int = 0) -> np.ndarray:
    """
    Standard-normal (Gaussian) tensor — typical LLM weight distribution.

    Parameters
    ----------
    shape : tuple
    seed : int

    Returns
    -------
    np.ndarray  (float32)
    """
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape).astype(np.float32)


def uniform_tensor(shape: tuple, seed: int = 0, low: float = -1.0, high: float = 1.0) -> np.ndarray:
    """
    Uniform distribution tensor.

    Parameters
    ----------
    shape : tuple
    seed : int
    low : float
    high : float

    Returns
    -------
    np.ndarray  (float32)
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size=shape).astype(np.float32)


def sparse_tensor(shape: tuple, sparsity: float = 0.8, seed: int = 0) -> np.ndarray:
    """
    Already-sparse tensor — mimics pruned weight layers.

    Parameters
    ----------
    shape : tuple
    sparsity : float
        Fraction of elements set to exactly 0.
    seed : int

    Returns
    -------
    np.ndarray  (float32)
    """
    rng = np.random.default_rng(seed)
    tensor = rng.standard_normal(shape).astype(np.float32)
    mask = rng.random(shape) < sparsity
    tensor[mask] = 0.0
    return tensor


def adversarial_tensor(shape: tuple) -> list[np.ndarray]:
    """
    Edge-case tensors that stress-test the quantizer.

    Returns a list of adversarial tensors:
    1. All-zero tensor
    2. All-same-value tensor (non-zero)
    3. Huge variance (very large + very small values)
    4. Single-element tensor

    Parameters
    ----------
    shape : tuple

    Returns
    -------
    list[np.ndarray]
    """
    tensors = [
        np.zeros(shape, dtype=np.float32),
        np.full(shape, 3.14, dtype=np.float32),
        np.concatenate([
            np.full((max(1, int(np.prod(shape)) // 2),), 1e6, dtype=np.float32),
            np.full((int(np.prod(shape)) - max(1, int(np.prod(shape)) // 2),), 1e-6, dtype=np.float32),
        ]).reshape(shape),
    ]
    return tensors


def attention_tensor(seq_len: int = 512, d_model: int = 768, seed: int = 0) -> np.ndarray:
    """
    Realistic attention weight tensor (QKV projection shape).

    Parameters
    ----------
    seq_len : int
    d_model : int
    seed : int

    Returns
    -------
    np.ndarray  (float32)  shape (d_model, d_model)
    """
    rng = np.random.default_rng(seed)
    # Attention weights are typically initialised with small Gaussian values
    scale = 1.0 / (d_model ** 0.5)
    return (rng.standard_normal((d_model, d_model)) * scale).astype(np.float32)
