"""Auxiliares de visualização e exportação de figuras axiais e calcificações do OrCaScore."""

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from utils.project.datasets.ccta import load_ccta_volume
from utils.project.datasets.orcascore import ORCASCORE_CALCIFICATION_LABEL_SPECS
from utils.visualization.images.hu_threshold import hu_axial_mip

__all__ = [
    "export_axial_image",
    "export_axial_mip",
    "export_central_axial_slice",
    "save_print_figure",
    "show_calcification_view",
]


def save_print_figure(
    fig: Figure,
    stem: str,
    metadata: dict[str, Any],
    *,
    output_dir: Path,
    export_dpi: int = 600,
) -> None:
    """Exporta uma figura em alta resolução (PNG, TIFF, PDF) junto com seu JSON de proveniência.

    Args:
        fig (Figure): Figura Matplotlib a ser salva.
        stem (str): Prefixo do nome dos arquivos exportados sem extensão.
        metadata (dict[str, Any]): Metadados de proveniência da imagem exportada.
        output_dir (Path): Diretório de destino para os arquivos gerados.
        export_dpi (int): Resolução em DPI utilizada na exportação para impressão.
    """
    # Garante que o diretório de saída exista antes de exportar os formatos gráficos.
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "tiff", "pdf"):
        path = output_dir / f"{stem}.{suffix}"
        options = (
            {"pil_kwargs": {"compression": "tiff_lzw"}} if suffix == "tiff" else {}
        )
        fig.savefig(
            path,
            dpi=export_dpi,
            facecolor="black",
            edgecolor="black",
            bbox_inches=None,
            pad_inches=0,
            **options,
        )
        print(path)

    # Registra os metadados de proveniência em JSON ao lado das figuras.
    (output_dir / f"{stem}.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def export_axial_image(
    acquisition: str,
    mode: str,
    *,
    pair: Mapping[str, Any],
    subset: str,
    exam_id: str,
    common_fov_mm: float,
    window_level_hu: float,
    window_width_hu: float,
    figure_width_mm: float,
    preview_dpi: int,
    export_dpi: int,
    output_dir: Path,
    save_figure: Callable[[Figure, str, dict[str, Any]], None] | None = None,
) -> None:
    """Exibe e exporta a projeção MIP axial ou a fatia axial central sem reorientar o volume.

    Args:
        acquisition (str): Tipo de aquisição selecionada (`contrast` ou `noncontrast`).
        mode (str): Modo de projeção axial (`mip` ou `central_slice`).
        pair (Mapping[str, Any]): Dicionário com os registros das aquisições do exame.
        subset (str): Subconjunto do exame no dataset OrCaScore.
        exam_id (str): Identificador do exame analisado.
        common_fov_mm (float): Campo de visão físico quadrado compartilhado em milímetros.
        window_level_hu (float): Nível da janela de exibição em unidades Hounsfield.
        window_width_hu (float): Largura da janela de exibição em unidades Hounsfield.
        figure_width_mm (float): Largura física da figura impressa em milímetros.
        preview_dpi (int): Resolução em DPI para pré-visualização no notebook.
        export_dpi (int): Resolução em DPI para exportação dos arquivos gráficos.
        output_dir (Path): Diretório de saída para exportação das figuras e metadados.
        save_figure (Callable[[Figure, str, dict[str, Any]], None] | None): Callback opcional de salvamento.
    """
    # Carrega o volume da aquisição e extrai o MIP axial ou a fatia central nativa.
    record = pair[acquisition]
    volume = load_ccta_volume(record)
    if mode == "mip":
        slice_index = None
        projection = hu_axial_mip(volume)
        image_suffix = "axial_mip"
    elif mode == "central_slice":
        slice_index = volume.shape[2] // 2
        projection = volume[:, :, slice_index].copy()
        image_suffix = f"axial_central_slice_z{slice_index:04d}"
    else:
        raise ValueError("mode deve ser 'mip' ou 'central_slice'.")
    del volume

    # Calcula a extensão física nativa e os limites da janela HU.
    sx, sy = float(record["spacing_x_mm"]), float(record["spacing_y_mm"])
    nx, ny = projection.shape
    extent = (-ny * sy / 2, ny * sy / 2, -nx * sx / 2, nx * sx / 2)
    half_fov = common_fov_mm / 2
    lower = window_level_hu - window_width_hu / 2
    upper = window_level_hu + window_width_hu / 2
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"orcascore_{subset}_{exam_id}_{acquisition}_{image_suffix}"

    # Renderiza a projeção axial centralizada no campo de visão comum.
    with plt.rc_context({"savefig.facecolor": "black", "savefig.edgecolor": "black"}):
        width_inches = figure_width_mm / 25.4
        fig = plt.figure(
            figsize=(width_inches, width_inches), dpi=preview_dpi, facecolor="black"
        )
        ax = fig.add_axes((0.0, 0.0, 1.0, 1.0), facecolor="black")
        ax.imshow(
            projection,
            cmap="gray",
            origin="upper",
            extent=extent,
            vmin=lower,
            vmax=upper,
            interpolation="nearest",
            aspect="equal",
        )
        ax.set(xlim=(-half_fov, half_fov), ylim=(-half_fov, half_fov))
        ax.set_axis_off()
        plt.show()
        plt.close(fig)

    # Monta o dicionário de metadados e exporta a figura gerada.
    metadata = {
        "exam_id": exam_id,
        "subset": subset,
        "acquisition": acquisition,
        "source_path": str(record["path"]),
        "view": "axial",
        "projection": "maximum_full_volume" if mode == "mip" else None,
        "image_type": mode,
        "slice_index_zero_based": slice_index,
        "flipped_axes": [],
        "native_shape_xyz": [
            int(record["size_x"]),
            int(record["size_y"]),
            int(record["slice_count"]),
        ],
        "spacing_xyz_mm": [float(record[f"spacing_{axis}_mm"]) for axis in "xyz"],
        "common_fov_mm": common_fov_mm,
        "centered_fov": True,
        "registered": False,
        "rotation_clockwise_degrees": 0,
        "window_level_hu": window_level_hu,
        "window_width_hu": window_width_hu,
        "figure_width_mm": figure_width_mm,
        "export_dpi": export_dpi,
    }
    if save_figure is not None:
        save_figure(fig, stem, metadata)
    else:
        save_print_figure(
            fig,
            stem,
            metadata,
            output_dir=output_dir,
            export_dpi=export_dpi,
        )
    if slice_index is not None:
        print(f"Fatia axial central: z={slice_index} (índice começando em zero).")


def export_axial_mip(
    acquisition: str,
    *,
    pair: Mapping[str, Any],
    subset: str,
    exam_id: str,
    common_fov_mm: float,
    window_level_hu: float,
    window_width_hu: float,
    figure_width_mm: float,
    preview_dpi: int,
    export_dpi: int,
    output_dir: Path,
    save_figure: Callable[[Figure, str, dict[str, Any]], None] | None = None,
) -> None:
    """Exporta a projeção de intensidade máxima (MIP) axial da aquisição informada.

    Args:
        acquisition (str): Tipo de aquisição (`contrast` ou `noncontrast`).
        pair (Mapping[str, Any]): Dicionário com os registros das aquisições do exame.
        subset (str): Subconjunto do exame no dataset OrCaScore.
        exam_id (str): Identificador do exame analisado.
        common_fov_mm (float): Campo de visão físico quadrado compartilhado em milímetros.
        window_level_hu (float): Nível da janela de exibição em unidades Hounsfield.
        window_width_hu (float): Largura da janela de exibição em unidades Hounsfield.
        figure_width_mm (float): Largura física da figura impressa em milímetros.
        preview_dpi (int): Resolução em DPI para pré-visualização no notebook.
        export_dpi (int): Resolução em DPI para exportação dos arquivos gráficos.
        output_dir (Path): Diretório de saída para exportação das figuras e metadados.
        save_figure (Callable[[Figure, str, dict[str, Any]], None] | None): Callback opcional de salvamento.
    """
    # Delega a renderização para export_axial_image no modo MIP.
    export_axial_image(
        acquisition,
        "mip",
        pair=pair,
        subset=subset,
        exam_id=exam_id,
        common_fov_mm=common_fov_mm,
        window_level_hu=window_level_hu,
        window_width_hu=window_width_hu,
        figure_width_mm=figure_width_mm,
        preview_dpi=preview_dpi,
        export_dpi=export_dpi,
        output_dir=output_dir,
        save_figure=save_figure,
    )


def export_central_axial_slice(
    acquisition: str,
    *,
    pair: Mapping[str, Any],
    subset: str,
    exam_id: str,
    common_fov_mm: float,
    window_level_hu: float,
    window_width_hu: float,
    figure_width_mm: float,
    preview_dpi: int,
    export_dpi: int,
    output_dir: Path,
    save_figure: Callable[[Figure, str, dict[str, Any]], None] | None = None,
) -> None:
    """Exporta a fatia axial central (`n_z // 2`) da aquisição informada.

    Args:
        acquisition (str): Tipo de aquisição (`contrast` ou `noncontrast`).
        pair (Mapping[str, Any]): Dicionário com os registros das aquisições do exame.
        subset (str): Subconjunto do exame no dataset OrCaScore.
        exam_id (str): Identificador do exame analisado.
        common_fov_mm (float): Campo de visão físico quadrado compartilhado em milímetros.
        window_level_hu (float): Nível da janela de exibição em unidades Hounsfield.
        window_width_hu (float): Largura da janela de exibição em unidades Hounsfield.
        figure_width_mm (float): Largura física da figura impressa em milímetros.
        preview_dpi (int): Resolução em DPI para pré-visualização no notebook.
        export_dpi (int): Resolução em DPI para exportação dos arquivos gráficos.
        output_dir (Path): Diretório de saída para exportação das figuras e metadados.
        save_figure (Callable[[Figure, str, dict[str, Any]], None] | None): Callback opcional de salvamento.
    """
    # Delega a renderização para export_axial_image no modo de fatia central.
    export_axial_image(
        acquisition,
        "central_slice",
        pair=pair,
        subset=subset,
        exam_id=exam_id,
        common_fov_mm=common_fov_mm,
        window_level_hu=window_level_hu,
        window_width_hu=window_width_hu,
        figure_width_mm=figure_width_mm,
        preview_dpi=preview_dpi,
        export_dpi=export_dpi,
        output_dir=output_dir,
        save_figure=save_figure,
    )


def show_calcification_view(
    view: str = "comparison",
    *,
    calcification_available: bool,
    calcification_ct_slice: np.ndarray | None,
    calcification_label_slice: np.ndarray | None,
    calcification_label: np.ndarray | None,
    calcification_crop: Sequence[slice] | None,
    calcification_z: int | None,
    spacing_xyz: Sequence[float] | None,
    figure_width_mm: float,
    preview_dpi: int,
    export_dpi: int,
    window_level_hu: float,
    window_width_hu: float,
    calcification_alpha: float,
    calcification_margin_mm: float,
    calcification_slice_index: int | None,
    calcification_reference_path: Path,
    exam_id: str,
    subset: str,
    pair: Mapping[str, Any],
    output_dir: Path | None = None,
    save_figure: Callable[[Figure, str, dict[str, Any]], None] | None = None,
) -> None:
    """Exibe e exporta a referência de calcificação na CTI com cores fixas por artéria.

    Args:
        view (str): Modo de visualização (`comparison`, `ct`, `label` ou `overlay`).
        calcification_available (bool): Indica se a máscara de referência existe para o exame.
        calcification_ct_slice (np.ndarray | None): Fatia 2D recortada da CT sem contraste.
        calcification_label_slice (np.ndarray | None): Fatia 2D recortada dos rótulos de cálcio.
        calcification_label (np.ndarray | None): Volume 3D completo de rótulos de calcificação.
        calcification_crop (Sequence[slice] | None): Fatias `(x, y, z)` do recorte em torno do cálcio.
        calcification_z (int | None): Índice axial `z` da fatia selecionada.
        spacing_xyz (Sequence[float] | None): Espaçamento físico `(sx, sy, sz)` em milímetros.
        figure_width_mm (float): Largura física de cada painel em milímetros.
        preview_dpi (int): Resolução em DPI para pré-visualização no notebook.
        export_dpi (int): Resolução em DPI para exportação dos arquivos gráficos.
        window_level_hu (float): Nível da janela HU da CT sem contraste.
        window_width_hu (float): Largura da janela HU da CT sem contraste.
        calcification_alpha (float): Opacidade da máscara sobreposta no modo `overlay`.
        calcification_margin_mm (float): Margem física em milímetros aplicada ao bounding box.
        calcification_slice_index (int | None): Índice axial explícito ou `None` para seleção automática.
        calcification_reference_path (Path): Caminho da máscara de referência carregada.
        exam_id (str): Identificador do exame OrCaScore.
        subset (str): Subconjunto do exame (`train` ou `test`).
        pair (Mapping[str, Any]): Registro das aquisições do exame contendo `noncontrast`.
        output_dir (Path | None): Diretório de saída usado quando `save_figure` não é fornecido.
        save_figure (Callable[[Figure, str, dict[str, Any]], None] | None): Callback opcional de salvamento.
    """
    # Verifica disponibilidade da referência e validade do modo solicitado.
    if (
        not calcification_available
        or calcification_ct_slice is None
        or calcification_label_slice is None
        or calcification_label is None
        or calcification_crop is None
        or calcification_z is None
        or spacing_xyz is None
    ):
        print(
            "Visualização indisponível: não há referência de calcificação para este exame."
        )
        return
    if view not in {"comparison", "ct", "label", "overlay"}:
        raise ValueError("Visualização de calcificação desconhecida.")

    # Prepara o mapa de cores discreto por artéria com fundo transparente fora das lesões.
    colors = [
        "#000000",
        *[f"#{color:06x}" for _, color in ORCASCORE_CALCIFICATION_LABEL_SPECS.values()],
    ]
    label_cmap = ListedColormap(colors)
    label_cmap.set_bad(alpha=0)
    visible = np.ma.masked_equal(calcification_label_slice, 0)
    nx, ny = calcification_ct_slice.shape
    sx, sy = float(spacing_xyz[0]), float(spacing_xyz[1])
    extent = (-ny * sy / 2, ny * sy / 2, -nx * sx / 2, nx * sx / 2)
    half_fov = max(nx * sx, ny * sy) / 2
    views = ("ct", "label", "overlay") if view == "comparison" else (view,)
    width_inches = figure_width_mm / 25.4

    # Renderiza os painéis solicitados mantendo os eixos nativos sem rotação ou espelhamento.
    fig, axes = plt.subplots(
        1,
        len(views),
        squeeze=False,
        figsize=(
            width_inches * len(views),
            width_inches + (0.45 if view == "comparison" else 0),
        ),
        dpi=preview_dpi,
        facecolor="black",
    )
    for ax, kind in zip(axes.ravel(), views, strict=True):
        ax.set_facecolor("black")
        if kind != "label":
            ax.imshow(
                calcification_ct_slice,
                cmap="gray",
                origin="upper",
                extent=extent,
                vmin=window_level_hu - window_width_hu / 2,
                vmax=window_level_hu + window_width_hu / 2,
                interpolation="nearest",
                aspect="equal",
            )
        if kind != "ct":
            ax.imshow(
                visible,
                cmap=label_cmap,
                vmin=0,
                vmax=3,
                origin="upper",
                extent=extent,
                alpha=calcification_alpha if kind == "overlay" else 1,
                interpolation="nearest",
                aspect="equal",
            )
        ax.set(xlim=(-half_fov, half_fov), ylim=(-half_fov, half_fov))
        ax.set_axis_off()
        if view == "comparison":
            ax.set_title(
                {
                    "ct": "CT sem contraste",
                    "label": "Calcificações de referência",
                    "overlay": "Sobreposição",
                }[kind],
                color="white",
            )

    # Adiciona legenda das classes anatômicas no modo comparativo e ajusta margens.
    if view == "comparison":
        fig.legend(
            handles=[
                Patch(facecolor=f"#{color:06x}", label=f"{value}: {name}")
                for value, (name, color) in ORCASCORE_CALCIFICATION_LABEL_SPECS.items()
            ],
            loc="lower center",
            ncol=3,
            frameon=False,
            labelcolor="white",
        )
        fig.subplots_adjust(left=0.015, right=0.985, bottom=0.16, top=0.88, wspace=0.06)
    else:
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    # Compõe metadados de proveniência da fatia e exporta a figura.
    stem = f"orcascore_{subset}_{exam_id}_calcification_{view}_z{calcification_z:04d}"
    metadata = {
        "exam_id": exam_id,
        "subset": subset,
        "acquisition": "noncontrast",
        "image_path": str(pair["noncontrast"]["path"]),
        "reference_path": str(calcification_reference_path),
        "view": view,
        "slice_index_zero_based": int(calcification_z),
        "slice_selection": "explicit"
        if calcification_slice_index is not None
        else "max_label_area"
        if np.any(calcification_label)
        else "volume_center_no_calcifications",
        "crop_slices_xyz": [[s.start, s.stop] for s in calcification_crop],
        "spacing_xyz_mm": [float(value) for value in spacing_xyz],
        "margin_mm": calcification_margin_mm,
        "flipped_axes": [],
        "rotation_clockwise_degrees": 0,
        "registered": False,
        "window_level_hu": window_level_hu,
        "window_width_hu": window_width_hu,
        "alpha": calcification_alpha,
        "export_dpi": export_dpi,
        "label_specs": {
            str(value): name
            for value, (name, _) in ORCASCORE_CALCIFICATION_LABEL_SPECS.items()
        },
    }
    if save_figure is not None:
        save_figure(fig, stem, metadata)
    elif output_dir is not None:
        save_print_figure(
            fig,
            stem,
            metadata,
            output_dir=output_dir,
            export_dpi=export_dpi,
        )
    plt.show()
    plt.close(fig)
