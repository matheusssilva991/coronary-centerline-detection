import numpy as np
import k3d
from k3d.factory import plot as create_plot
from skimage import measure
from pathlib import Path
from typing import Any, Optional, Sequence, cast
from numpy.typing import NDArray


def visualize_3d_k3d(
    mask_3d: NDArray[Any],
    spacing: Sequence[float] = (1, 1, 1),
    color: int = 0xFF0000,
    opacity: float = 0.5,
    use_physical_coords: bool = True,
    save_html_path: Optional[str] = None,
    display_plot: bool = True,
) -> Any:
    """Renderiza uma máscara 3D em k3d usando marching cubes."""
    # Define unidade de coordenada (mm ou pixel).
    if use_physical_coords:
        dy, dx, dz = tuple(float(s) for s in spacing)
        axes_labels = ["Y (mm)", "X (mm)", "Z (mm)"]
    else:
        dy, dx, dz = 1.0, 1.0, 1.0
        axes_labels = ["Y (pixels)", "X (pixels)", "Z (pixels)"]

    # Mensagem curta para indicar a geração da malha.
    print("Gerando mesh 3D...")

    # Extrai malha de superfície da máscara.
    verts, faces, normals, values = measure.marching_cubes(
        mask_3d.astype(float), level=0.5, spacing=(dy, dx, dz)
    )

    # Inicializa cena 3D.
    plot = cast(
        Any,
        create_plot(
            name="Segmentação 3D", height=800, grid_visible=True, axes=axes_labels
        ),
    )

    mesh = k3d.mesh(
        verts.astype(np.float32),
        faces.astype(np.uint32),
        color=color,
        opacity=opacity,
        name="Máscara 3D",
    )

    # Adiciona malha e renderiza.
    plot += mesh
    if save_html_path is not None:
        with open(save_html_path, "w", encoding="utf-8") as html_file:
            html_file.write(plot.get_snapshot())

    if display_plot:
        plot.display()
    return plot


def visualize_aorta_with_ostia(
    aorta_mask: NDArray[Any],
    ostia_left: Sequence[float],
    ostia_right: Sequence[float],
    spacing: Sequence[float] = (1, 1, 1),
    label_mask: Optional[NDArray[Any]] = None,
    use_physical_coords: bool = True,
    save_html_path: Optional[str] = None,
    display_plot: bool = True,
) -> Any:
    """Renderiza a aorta 3D com marcação dos óstios e rótulo opcional."""

    # Define unidade de coordenada (mm ou pixel).
    if use_physical_coords:
        dy, dx, dz = tuple(float(s) for s in spacing)
        axes_labels = ["Y (mm)", "X (mm)", "Z (mm)"]
    else:
        dy, dx, dz = 1.0, 1.0, 1.0
        axes_labels = ["Y (pixels)", "X (pixels)", "Z (pixels)"]

    # Inicializa cena com aorta, label opcional e óstios.
    plot = cast(
        Any,
        create_plot(
            name="Aorta + Óstios + Label",
            height=800,
            grid_visible=True,
            axes=axes_labels,
        ),
    )

    # Extrai malha da aorta predita.
    verts, faces, _, _ = measure.marching_cubes(
        aorta_mask.astype(float), level=0.5, spacing=(dy, dx, dz)
    )

    mesh_pred = k3d.mesh(
        verts.astype(np.float32),
        faces.astype(np.uint32),
        color=0xFF5555,
        opacity=0.3,
        name="Aorta Predita",
    )
    # Camada principal da cena.
    plot += mesh_pred

    if label_mask is not None:
        # Se houver label, adiciona malha de referência.
        verts_label, faces_label, _, _ = measure.marching_cubes(
            label_mask.astype(float), level=0.5, spacing=(dy, dx, dz)
        )

        mesh_label = k3d.mesh(
            verts_label.astype(np.float32),
            faces_label.astype(np.uint32),
            color=0x55FF55,
            opacity=0.3,
            name="Ground Truth",
        )
        # Cor distinta para comparar com a predição.
        plot += mesh_label

    pos_left = np.array(
        [
            [
                float(ostia_left[0] * dy),
                float(ostia_left[1] * dx),
                float(ostia_left[2] * dz),
            ]
        ],
        dtype=np.float32,
    )
    # Converte coordenada do óstio esquerdo.
    point_left = k3d.points(
        positions=pos_left,
        point_size=12.0,
        color=0xFFFF00,
        name=(
            f"Óstio Esquerdo\ny={ostia_left[0]}, x={ostia_left[1]}, z={ostia_left[2]}"
        ),
    )
    # Marca o óstio esquerdo.
    plot += point_left

    pos_right = np.array(
        [
            [
                float(ostia_right[0] * dy),
                float(ostia_right[1] * dx),
                float(ostia_right[2] * dz),
            ]
        ],
        dtype=np.float32,
    )
    # Converte coordenada do óstio direito.
    point_right = k3d.points(
        positions=pos_right,
        point_size=12.0,
        color=0x00FFFF,
        name=(
            f"Óstio Direito\ny={ostia_right[0]}, x={ostia_right[1]}, z={ostia_right[2]}"
        ),
    )
    # Marca o óstio direito.
    plot += point_right

    if save_html_path is not None:
        with open(save_html_path, "w", encoding="utf-8") as html_file:
            html_file.write(plot.get_snapshot())

    if display_plot:
        plot.display()
    return plot


def _add_mask_mesh(
    plot: Any,
    mask: NDArray[Any],
    *,
    spacing: Sequence[float],
    color: int,
    opacity: float,
    name: str,
) -> None:
    """Adiciona uma máscara 3D ao plot quando ela contém voxels."""
    mask_bool = np.asarray(mask) > 0
    if not np.any(mask_bool):
        print(f"Máscara vazia ignorada: {name}")
        return

    verts, faces, _, _ = measure.marching_cubes(
        mask_bool.astype(float), level=0.5, spacing=spacing
    )
    plot += k3d.mesh(
        verts.astype(np.float32),
        faces.astype(np.uint32),
        color=color,
        opacity=opacity,
        name=name,
    )


def _add_ostium_point(
    plot: Any,
    ostium: Optional[Sequence[float]],
    *,
    spacing: Sequence[float],
    color: int,
    name: str,
) -> None:
    """Adiciona um ponto de óstio ao plot quando a coordenada existe."""
    if ostium is None:
        return

    dy, dx, dz = tuple(float(s) for s in spacing)
    position = np.array(
        [[float(ostium[0] * dy), float(ostium[1] * dx), float(ostium[2] * dz)]],
        dtype=np.float32,
    )
    plot += k3d.points(
        positions=position,
        point_size=12.0,
        color=color,
        name=f"{name}\ny={ostium[0]}, x={ostium[1]}, z={ostium[2]}",
    )


def visualize_aorta_ostia_artery(
    aorta_mask: NDArray[Any],
    ostia_left: Optional[Sequence[float]],
    ostia_right: Optional[Sequence[float]],
    artery_mask: Optional[NDArray[Any]] = None,
    label_artery: Optional[NDArray[Any]] = None,
    spacing: Sequence[float] = (1, 1, 1),
    use_physical_coords: bool = True,
    save_html_path: Optional[str] = None,
    display_plot: bool = True,
    plot_name: str = "Aorta + óstios + artéria",
) -> Any:
    """Renderiza aorta, óstios, artéria predita e referência no mesmo plot 3D."""
    if use_physical_coords:
        dy, dx, dz = tuple(float(s) for s in spacing)
        mesh_spacing = (dy, dx, dz)
        axes_labels = ["Y (mm)", "X (mm)", "Z (mm)"]
    else:
        mesh_spacing = (1.0, 1.0, 1.0)
        axes_labels = ["Y (pixels)", "X (pixels)", "Z (pixels)"]

    plot = cast(
        Any,
        create_plot(name=plot_name, height=800, grid_visible=True, axes=axes_labels),
    )
    _add_mask_mesh(
        plot,
        aorta_mask,
        spacing=mesh_spacing,
        color=0xFF5555,
        opacity=0.22,
        name="Aorta",
    )
    if label_artery is not None:
        _add_mask_mesh(
            plot,
            label_artery,
            spacing=mesh_spacing,
            color=0x39B54A,
            opacity=0.34,
            name="Artéria GT",
        )
    if artery_mask is not None:
        _add_mask_mesh(
            plot,
            artery_mask,
            spacing=mesh_spacing,
            color=0x2F80ED,
            opacity=0.44,
            name="Artéria predita",
        )

    _add_ostium_point(
        plot,
        ostia_left,
        spacing=mesh_spacing,
        color=0xFFFF00,
        name="Óstio esquerdo",
    )
    _add_ostium_point(
        plot,
        ostia_right,
        spacing=mesh_spacing,
        color=0x00FFFF,
        name="Óstio direito",
    )

    if save_html_path is not None:
        with open(save_html_path, "w", encoding="utf-8") as html_file:
            html_file.write(plot.get_snapshot())

    if display_plot:
        plot.display()
    return plot


def visualize_arteries_comparison(
    label_mask: NDArray[Any],
    predicted_mask: NDArray[Any],
    spacing: Sequence[float] = (1, 1, 1),
    use_physical_coords: bool = True,
    save_html_path: Optional[str] = None,
    display_plot: bool = True,
    plot_name: str = "Artéria original vs predita",
    label_color: int = 0x00FF00,
    predicted_color: int = 0xFF0000,
    label_opacity: float = 0.35,
    predicted_opacity: float = 0.35,
) -> Any:
    """Renderiza a artéria de referência e a predita no mesmo gráfico 3D."""

    if use_physical_coords:
        dy, dx, dz = tuple(float(s) for s in spacing)
        axes_labels = ["Y (mm)", "X (mm)", "Z (mm)"]
    else:
        dy, dx, dz = 1.0, 1.0, 1.0
        axes_labels = ["Y (pixels)", "X (pixels)", "Z (pixels)"]

    plot = cast(
        Any,
        create_plot(name=plot_name, height=800, grid_visible=True, axes=axes_labels),
    )

    verts_label, faces_label, _, _ = measure.marching_cubes(
        label_mask.astype(float), level=0.5, spacing=(dy, dx, dz)
    )
    mesh_label = k3d.mesh(
        verts_label.astype(np.float32),
        faces_label.astype(np.uint32),
        color=label_color,
        opacity=label_opacity,
        name="Label",
    )
    plot += mesh_label

    verts_pred, faces_pred, _, _ = measure.marching_cubes(
        predicted_mask.astype(float), level=0.5, spacing=(dy, dx, dz)
    )
    mesh_pred = k3d.mesh(
        verts_pred.astype(np.float32),
        faces_pred.astype(np.uint32),
        color=predicted_color,
        opacity=predicted_opacity,
        name="Predita",
    )
    plot += mesh_pred

    if save_html_path is not None:
        with open(save_html_path, "w", encoding="utf-8") as html_file:
            html_file.write(plot.get_snapshot())

    if display_plot:
        plot.display()
    return plot


def save_k3d_plot_html(plot: Any, html_path: str) -> None:
    output_path = Path(html_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(plot.get_snapshot(), encoding="utf-8")
    print(f"HTML salvo em: {output_path}")
