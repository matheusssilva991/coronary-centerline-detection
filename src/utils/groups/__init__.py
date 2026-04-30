"""Higher-level grouped utility namespaces.

This package provides domain-oriented groupings while preserving the existing
flat modules in ``utils`` for backward compatibility.
"""

from importlib import import_module

from . import io, processing, segmentation

__all__ = ["io", "processing", "segmentation", "visualization"]


def __getattr__(name):
    if name == "visualization":
        module = import_module(f"{__name__}.visualization")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
