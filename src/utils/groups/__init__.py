"""Higher-level grouped utility namespaces.

This package provides domain-oriented groupings while preserving the existing
flat modules in ``utils`` for backward compatibility.
"""

from . import io, processing, segmentation, visualization

__all__ = ["io", "processing", "segmentation", "visualization"]
