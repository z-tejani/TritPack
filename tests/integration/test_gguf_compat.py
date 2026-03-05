"""
Integration tests for GGUF backend compatibility.

These tests are skipped when the `gguf` package is not installed or no
GGUF model file is available.
"""

from __future__ import annotations

import pytest

try:
    import gguf  # noqa: F401
    _GGUF_AVAILABLE = True
except ImportError:
    _GGUF_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not _GGUF_AVAILABLE, reason="gguf package not installed"
)


def test_gguf_backend_import():
    """GGUFBackend can be imported without error."""
    from tritpack.backends.gguf_backend import GGUFBackend  # noqa: F401


def test_gguf_backend_missing_file():
    """GGUFBackend must raise FileNotFoundError for missing files."""
    from tritpack.backends.gguf_backend import GGUFBackend

    with pytest.raises(FileNotFoundError):
        GGUFBackend("/nonexistent/model.gguf")
