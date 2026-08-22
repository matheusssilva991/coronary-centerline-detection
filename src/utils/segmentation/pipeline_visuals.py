"""Persistência opcional das visualizações 3D produzidas pelo pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

from ..visualization.volume import visualize_aorta_ostia_artery

logger = logging.getLogger(__name__)


def save_segmentation_visual(
    output_dir: str | Path,
    img_id: int | str,
    *,
    aorta_mask: Any,
    ostia_left: Sequence[float] | None,
    ostia_right: Sequence[float] | None,
    artery_mask: Any | None,
    label_artery: Any,
    spacing: Sequence[float],
) -> Path | None:
    """Salva aorta, óstios, predição e referência em um HTML 3D interativo.

    A visualização é um artefato auxiliar. Qualquer erro de geração é registrado
    no log e não invalida os resultados numéricos da imagem.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    html_path = output_path / f"img_{int(img_id)}_aorta_ostia_artery.html"

    try:
        visualize_aorta_ostia_artery(
            aorta_mask,
            ostia_left,
            ostia_right,
            artery_mask=artery_mask,
            label_artery=label_artery,
            spacing=spacing,
            use_physical_coords=True,
            save_html_path=str(html_path),
            display_plot=False,
            plot_name=f"IMG {int(img_id)} | Aorta, óstios e artérias",
        )
    except Exception:
        logger.exception(
            "Não foi possível salvar a visualização 3D da imagem %s.",
            img_id,
        )
        return None

    logger.info("Visualização 3D salva em: %s", html_path)
    return html_path


__all__ = ["save_segmentation_visual"]
