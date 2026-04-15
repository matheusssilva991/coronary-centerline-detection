"""Utilitários de segmentação grosseira de tecidos baseada em HU."""

import numpy as np


def segment_by_hu(img_3d, include_labels=None):
    """Segmenta volume de CT em classes de tecido com base em faixas de HU."""
    hu_ranges = {
        1: {"name": "Ar", "range": (-1050, -950), "color": [0, 0, 0]},
        2: {"name": "Pulmão", "range": (-950, -500), "color": [194, 220, 232]},
        3: {"name": "Gordura", "range": (-190, -30), "color": [255, 255, 150]},
        4: {"name": "Água/Fluidos", "range": (-30, 30), "color": [170, 250, 250]},
        5: {"name": "Tecidos Moles", "range": (30, 100), "color": [230, 170, 170]},
        6: {"name": "Osso Esponjoso", "range": (100, 400), "color": [255, 180, 100]},
        7: {
            "name": "Osso Cortical/Denso",
            "range": (400, 3000),
            "color": [255, 255, 255],
        },
        8: {"name": "Metal/Implantes", "range": (3000, 4000), "color": [220, 220, 50]},
    }

    if include_labels:
        hu_ranges = {k: v for k, v in hu_ranges.items() if k in include_labels}

    segmented = np.zeros_like(img_3d, dtype=np.uint8)
    for label, props in hu_ranges.items():
        min_hu, max_hu = props["range"]
        mask = (img_3d >= min_hu) & (img_3d <= max_hu)
        segmented[mask] = label

    return segmented, hu_ranges


__all__ = ["segment_by_hu"]
