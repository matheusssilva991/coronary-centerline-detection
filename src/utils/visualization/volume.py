import numpy as np
import k3d
from skimage import measure


def visualize_3d_k3d(
    mask_3d, spacing=(1, 1, 1), color=0xFF0000, opacity=0.5, use_physical_coords=True
):
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
    plot = k3d.plot(
        name="Segmentação 3D", height=800, grid_visible=True, axes=axes_labels
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
    plot.display()
    return plot


def visualize_aorta_with_ostia(
    aorta_mask,
    ostia_left,
    ostia_right,
    spacing=(1, 1, 1),
    label_mask=None,
    use_physical_coords=True,
):
    """Renderiza a aorta 3D com marcação dos óstios e rótulo opcional."""

    # Define unidade de coordenada (mm ou pixel).
    if use_physical_coords:
        dy, dx, dz = tuple(float(s) for s in spacing)
        axes_labels = ["Y (mm)", "X (mm)", "Z (mm)"]
    else:
        dy, dx, dz = 1.0, 1.0, 1.0
        axes_labels = ["Y (pixels)", "X (pixels)", "Z (pixels)"]

    # Inicializa cena com aorta, label opcional e óstios.
    plot = k3d.plot(
        name="Aorta + Óstios + Label", height=800, grid_visible=True, axes=axes_labels
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

    plot.display()
    return plot
