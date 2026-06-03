"""Compatibilidade legada para etapas reutilizáveis do pipeline.

Código novo deve importar diretamente de:
- ``pipeline_preprocessing``
- ``pipeline_detection``
- ``pipeline_arteries``
"""

from .pipeline_arteries import segment_arteries_from_ostia
from .pipeline_detection import (
    detect_and_evaluate_ostia,
    get_or_detect_aorta_circles,
    get_or_segment_aorta,
)
from .pipeline_preprocessing import get_or_compute_vesselness, load_and_preprocess_image

__all__ = [
    "detect_and_evaluate_ostia",
    "get_or_compute_vesselness",
    "get_or_detect_aorta_circles",
    "get_or_segment_aorta",
    "load_and_preprocess_image",
    "segment_arteries_from_ostia",
]
