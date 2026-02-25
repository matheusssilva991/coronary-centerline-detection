"""
Módulo de funções de visualização para volumes médicos 3D.

Fornece ferramentas para visualizar volumes médicos usando técnicas 2D
(projeções MIP, slices) e 3D (renderização de malhas com K3D), incluindo
visualização de círculos detectados e marcação de pontos anatômicos (óstios).
"""

from typing import Literal, Optional, Sequence

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from skimage import measure
import k3d


# =============================================================================
# Visualizações 2D - Projeções e Slices
# =============================================================================


def plot_mip_projection(
    image_volume: NDArray,
    title: str = "Maximum Intensity Projections (MIP)",
    cmap: str = "gray",
    views: Sequence[Literal["axial", "coronal", "sagital"]] = (
        "axial",
        "coronal",
        "sagital",
    ),
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    projection: Literal["max", "min", "mean"] = "max",
    window_level: Optional[float] = None,
    window_width: Optional[float] = None,
    return_fig: bool = False,
):
    """
    Plota projeções de intensidade (MIP/MinIP/MeanIP) em múltiplas orientações.

    Colapsa o volume 3D em imagens 2D usando max/min/mean ao longo de cada eixo
    anatômico. Útil para visualização rápida de estruturas vasculares e ósseas.

    Args:
        image_volume (NDArray): Volume 3D com shape (H, W, D)
        title (str): Título principal da figura. Default: "Maximum Intensity Projections (MIP)"
        cmap (str): Colormap do Matplotlib. Default: "gray"
        views (Sequence): Views a exibir dentre "axial", "coronal", "sagital".
            Default: todas as três views
        vmin (float, optional): Valor mínimo para escala de intensidade.
            Se None, usa mínimo da imagem. Default: None
        vmax (float, optional): Valor máximo para escala de intensidade.
            Se None, usa máximo da imagem. Default: None
        projection (str): Tipo de projeção: "max" (MIP), "min" (MinIP), "mean".
            Default: "max"
        window_level (float, optional): Nível central da janela HU para windowing.
            Requer window_width. Default: None
        window_width (float, optional): Largura da janela HU para windowing.
            Requer window_level. Default: None
        return_fig (bool): Se True, retorna (fig, axes) sem exibir.
            Se False, exibe com plt.show(). Default: False

    Returns:
        tuple or None: Se return_fig=True, retorna (fig, axes), caso contrário None

    Raises:
        ValueError: Se image_volume não for 3D ou se view for inválida

    Example:
        >>> ccta = load_volume()  # Shape: (512, 512, 300)
        >>> # MIP padrão (max projection em todas as views)
        >>> plot_mip_projection(ccta, vmin=-200, vmax=600)

        >>> # MinIP com windowing para pulmão
        >>> plot_mip_projection(
        ...     ccta,
        ...     projection="min",
        ...     window_level=-600,
        ...     window_width=1200,
        ...     views=["axial", "coronal"]
        ... )

        >>> # Salvar figura
        >>> fig, axes = plot_mip_projection(ccta, return_fig=True)
        >>> fig.savefig('mip.png', dpi=150)

    Note:
        - MIP (max) realça estruturas densas (vasos com contraste, ossos)
        - MinIP (min) realça estruturas de baixa densidade (ar, pulmão)
        - MeanIP útil para visão geral sem realce de extremos
        - Windowing aplica clip antes da projeção (útil para focar tecidos específicos)
        - Mapeamento de eixos: axial=Z, coronal=Y, sagital=X
    """
    if image_volume.ndim != 3:
        raise ValueError("A imagem deve ser 3D.")

    # Windowing (opcional)
    if window_level is not None and window_width is not None:
        lower = window_level - window_width // 2
        upper = window_level + window_width // 2
        image_volume = np.clip(image_volume, lower, upper)

    # Mapear views para eixos
    axis_map = {"axial": 2, "coronal": 1, "sagital": 0}

    # Selecionar função de projeção
    proj_func = {"max": np.max, "min": np.min, "mean": np.mean}[projection]

    n_views = len(views)
    fig, axes = plt.subplots(1, n_views, figsize=(6 * n_views, 6))
    if n_views == 1:
        axes = [axes]

    imshow_kwargs = {"cmap": cmap}
    if vmin is not None:
        imshow_kwargs["vmin"] = vmin
    if vmax is not None:
        imshow_kwargs["vmax"] = vmax

    for i, view in enumerate(views):
        if view not in axis_map:
            raise ValueError(
                f"View inválida: {view}. Use 'axial', 'coronal' ou 'sagital'."
            )
        mip = proj_func(image_volume, axis=axis_map[view])
        axes[i].imshow(mip, **imshow_kwargs)
        axes[i].set_title(f"MIP {view.capitalize()} ({projection}) - shape {mip.shape}")
        axes[i].axis("off")

    plt.suptitle(title, fontsize=16, y=0.96)
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    if return_fig:
        return fig, axes
    else:
        plt.show()


def plot_slices(img, slices_indices, cmap="gray", title=None, vmin=None, vmax=None):
    """
    Plota fatias específicas de um volume 3D em layout de grade.

    Exibe slices selecionados do volume em uma grade organizada, útil para
    inspeção visual de resultados de segmentação ou detecção em múltiplos níveis.

    Args:
        img (np.ndarray): Volume 3D com shape (H, W, D) onde D é o eixo de slices
        slices_indices (list): Lista de índices de slices a exibir (coordenadas Z)
        cmap (str): Colormap do Matplotlib. Default: "gray"
        title (str, optional): Título principal da figura.
            Se None, não exibe título. Default: None
        vmin (float, optional): Valor mínimo para escala de cores.
            Se None, usa mínimo de cada slice. Default: None
        vmax (float, optional): Valor máximo para escala de cores.
            Se None, usa máximo de cada slice. Default: None

    Returns:
        None: Exibe a figura usando plt.show()

    Example:
        >>> volume = load_volume()  # Shape: (512, 512, 300)
        >>> # Visualizar 10 slices uniformemente espaçados
        >>> indices = np.linspace(0, 299, 10, dtype=int)
        >>> plot_slices(volume, indices, title="Aorta Segmentation", vmin=-200, vmax=600)

        >>> # Visualizar slices específicos onde círculos foram detectados
        >>> detected_slices = [45, 67, 89, 120, 145]
        >>> plot_slices(volume, detected_slices, cmap="viridis")

    Note:
        - Layout automático: até 5 colunas, linhas adicionais conforme necessário
        - Subplots vazios são removidos automaticamente
        - Cada slice mostra seu índice Z no título
        - Para grande número de slices, considere amostragem uniforme
        - vmin/vmax consistentes permitem comparação visual entre slices
    """
    # Número de fatias a plotar
    n_slices = len(slices_indices)

    # Calcular número de linhas e colunas para subplots
    n_cols = min(5, n_slices)
    n_rows = (n_slices + n_cols - 1) // n_cols

    # Criar figura
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))

    # Garantir que axes seja sempre um array 2D
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = np.array([axes]).reshape(n_rows, n_cols)

    # Plotar cada fatia
    for i, slice_idx in enumerate(slices_indices):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col]
        ax.imshow(img[:, :, slice_idx], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"Fatia {slice_idx}")
        ax.axis("off")

    # Remover subplots vazios
    for i in range(n_slices, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        fig.delaxes(axes[row, col])

    # Adicionar título principal se fornecido
    if title:
        plt.suptitle(title, fontsize=16, y=0.98)
        plt.subplots_adjust(top=0.90)  # Ajusta o espaço para o título

    plt.tight_layout()
    plt.show()


def visualize_circles_on_slices(
    image, detected_circles, num_samples=6, vmin=None, vmax=None
):
    """
    Visualiza círculos detectados sobrepostos em slices amostrados uniformemente.

    Exibe círculos detectados pela Transformada de Hough sobrepostos em slices,
    com marcadores de centro e borda. Útil para validação visual de detecção de aorta.

    Args:
        image (np.ndarray): Volume 3D com shape (H, W, D)
        detected_circles (list): Lista de dicts com círculos detectados.
            Cada dict deve conter: 'slice_index' (int), 'center_x' (float),
            'center_y' (float), 'radius' (float)
        num_samples (int): Número máximo de slices a visualizar.
            Slices são amostrados uniformemente. Default: 6
        vmin (float, optional): Valor mínimo para escala de intensidade.
            Se None, usa mínimo da imagem. Default: None
        vmax (float, optional): Valor máximo para escala de intensidade.
            Se None, usa máximo da imagem. Default: None

    Returns:
        None: Exibe a figura usando plt.show()

    Example:
        >>> volume = load_volume()
        >>> circles = detect_aorta_circles(volume)
        >>> # circles = [
        >>> #     {'slice_index': 45, 'center_x': 256, 'center_y': 230, 'radius': 25},
        >>> #     {'slice_index': 67, 'center_x': 255, 'center_y': 228, 'radius': 26},
        >>> #     ...
        >>> # ]
        >>> visualize_circles_on_slices(volume, circles, num_samples=8, vmin=-200, vmax=600)

    Note:
        - Layout fixo: 2 linhas × 3 colunas (máximo 6 slices)
        - Círculos desenhados com borda vermelha (linewidth=2)
        - Centro marcado com cruz vermelha (+)
        - Amostragem uniforme: se houver 30 slices e num_samples=6, mostra 1 a cada 5
        - Múltiplos círculos no mesmo slice são todos exibidos
        - Slices sem círculos são ignorados
    """
    # Selecionar fatias uniformemente distribuídas
    slice_indices = sorted(set([c["slice_index"] for c in detected_circles]))
    step = max(1, len(slice_indices) // num_samples)
    selected_slices = slice_indices[::step][:num_samples]

    # Criar subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, slice_idx in enumerate(selected_slices):
        if i >= len(axes):
            break

        # Encontrar círculos nesta fatia
        circles_in_slice = [
            c for c in detected_circles if c["slice_index"] == slice_idx
        ]

        # Plotar fatia
        axes[i].imshow(image[:, :, slice_idx], cmap="gray", vmin=vmin, vmax=vmax)

        # Adicionar círculos
        for circle in circles_in_slice:
            circle_patch = patches.Circle(
                (circle["center_x"], circle["center_y"]),
                circle["radius"],
                fill=False,
                edgecolor="red",
                linewidth=2,
            )
            axes[i].add_patch(circle_patch)

            # Marcar centro
            axes[i].plot(circle["center_x"], circle["center_y"], "r+", markersize=10)

        axes[i].set_title(f"Fatia {slice_idx}")
        axes[i].axis("off")

    # Remover subplots extras
    for i in range(len(selected_slices), len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


# =============================================================================
# Visualizações 3D - K3D e Marching Cubes
# =============================================================================


def visualize_3d_k3d(
    mask_3d, spacing=(1, 1, 1), color=0xFF0000, opacity=0.5, use_physical_coords=True
):
    """
    Renderiza máscara binária 3D como malha interativa usando K3D e Marching Cubes.

    Converte máscara segmentada em malha triangular 3D e exibe em viewer interativo
    do Jupyter. Útil para inspeção visual de qualidade de segmentação.

    Args:
        mask_3d (np.ndarray): Máscara binária 3D com shape (Y, X, Z).
            Valores > 0 são considerados parte da estrutura
        spacing (tuple): Espaçamento de voxels (dy, dx, dz) em mm,
            correspondendo aos eixos (Y, X, Z). Default: (1, 1, 1)
        color (int): Cor da malha em formato hexadecimal (0xRRGGBB).
            Default: 0xff0000 (vermelho)
        opacity (float): Opacidade da malha no intervalo [0, 1].
            0 = transparente, 1 = opaco. Default: 0.5
        use_physical_coords (bool): Se True, usa coordenadas físicas (mm) com spacing.
            Se False, usa coordenadas de pixels. Default: True

    Returns:
        k3d.Plot: Objeto plot do K3D (pode ser reutilizado para adicionar mais objetos)

    Example:
        >>> aorta_mask = segment_aorta(volume)
        >>> # Visualizar com spacing real do CCTA
        >>> plot = visualize_3d_k3d(
        ...     aorta_mask,
        ...     spacing=(0.5, 0.5, 0.625),  # mm por voxel
        ...     color=0xff0000,
        ...     opacity=0.7
        ... )

        >>> # Visualizar em coordenadas de pixels
        >>> plot = visualize_3d_k3d(aorta_mask, use_physical_coords=False)

        >>> # Múltiplas cores
        >>> visualize_3d_k3d(left_artery, color=0xffff00, opacity=0.5)  # Amarelo

    Note:
        - Usa Marching Cubes (level=0.5) para gerar malha a partir de máscara
        - Spacing deve vir de DICOM header ou NIfTI affine para escala correta
        - K3D requer ambiente Jupyter (notebook/lab)
        - Ordem de eixos: (Y, X, Z) seguindo convenção de arrays numpy
        - Para performance, considere downsampling de máscaras muito grandes
        - Cores úteis: 0xff0000 (vermelho), 0x00ff00 (verde), 0x0000ff (azul),
          0xffff00 (amarelo), 0x00ffff (ciano)
    """
    if use_physical_coords:
        dy, dx, dz = tuple(float(s) for s in spacing)
        axes_labels = ["Y (mm)", "X (mm)", "Z (mm)"]
    else:
        dy, dx, dz = 1.0, 1.0, 1.0
        axes_labels = ["Y (pixels)", "X (pixels)", "Z (pixels)"]

    print("Gerando mesh 3D...")

    # marching_cubes: passar (y, x, z) com spacing (dy, dx, dz)
    verts, faces, normals, values = measure.marching_cubes(
        mask_3d.astype(float), level=0.5, spacing=(dy, dx, dz)
    )

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
    """
    Renderiza aorta segmentada com óstios coronários marcados, opcionalmente com ground truth.

    Visualização 3D interativa mostrando: (1) malha da aorta predita, (2) óstios
    esquerdo e direito como pontos coloridos, (3) opcionalmente ground truth para comparação.

    Args:
        aorta_mask (np.ndarray): Máscara binária 3D da aorta segmentada com shape (Y, X, Z)
        ostia_left (tuple): Coordenadas (y, x, z) do óstio coronário esquerdo em pixels
        ostia_right (tuple): Coordenadas (y, x, z) do óstio coronário direito em pixels
        spacing (tuple): Espaçamento de voxels (dy, dx, dz) em mm.
            Default: (1, 1, 1)
        label_mask (np.ndarray, optional): Máscara binária 3D do ground truth com shape (Y, X, Z).
            Se fornecida, exibida em verde translúcido. Default: None
        use_physical_coords (bool): Se True, converte coordenadas para mm usando spacing.
            Se False, usa coordenadas de pixels. Default: True

    Returns:
        k3d.Plot: Objeto plot do K3D com todos os elementos renderizados

    Example:
        >>> # Carregar dados
        >>> aorta_mask = level_set_segmentation(volume, circles)
        >>> ostia_left, ostia_right = find_ostia(aorta_mask, circles)
        >>> gt_mask = load_ground_truth()

        >>> # Visualizar com ground truth
        >>> plot = visualize_aorta_with_ostia(
        ...     aorta_mask,
        ...     ostia_left=(120, 256, 45),
        ...     ostia_right=(150, 290, 45),
        ...     spacing=(0.5, 0.5, 0.625),
        ...     label_mask=gt_mask
        ... )

        >>> # Visualizar apenas predição
        >>> plot = visualize_aorta_with_ostia(
        ...     aorta_mask,
        ...     ostia_left=(120, 256, 45),
        ...     ostia_right=(150, 290, 45)
        ... )

    Esquema de cores:
        - Aorta predita: vermelho translúcido (0xff5555, opacity=0.3)
        - Ground truth: verde translúcido (0x55ff55, opacity=0.3)
        - Óstio esquerdo: amarelo (0xffff00, point_size=12)
        - Óstio direito: ciano (0x00ffff, point_size=12)

    Note:
        - Coordenadas de óstios são em pixels e convertidas para mm internamente
        - Útil para validar detecção de óstios e qualidade de segmentação
        - Label overlay permite comparação visual entre predição e ground truth
        - Tooltips nos pontos mostram coordenadas (y, x, z) em pixels
        - K3D permite rotação, zoom e pan interativos
        - Requer ambiente Jupyter para exibição
    """
    if use_physical_coords:
        dy, dx, dz = tuple(float(s) for s in spacing)
        axes_labels = ["Y (mm)", "X (mm)", "Z (mm)"]
    else:
        dy, dx, dz = 1.0, 1.0, 1.0
        axes_labels = ["Y (pixels)", "X (pixels)", "Z (pixels)"]

    plot = k3d.plot(
        name="Aorta + Óstios + Label", height=800, grid_visible=True, axes=axes_labels
    )

    # 1. Mesh da aorta predita (vermelho translúcido)
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
    plot += mesh_pred

    # 2. Adicionar label (verde translúcido) se fornecido
    if label_mask is not None:
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
        plot += mesh_label

    # 3. Óstio esquerdo (amarelo)
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

    point_left = k3d.points(
        positions=pos_left,
        point_size=12.0,
        color=0xFFFF00,
        name=f"Óstio Esquerdo\ny={ostia_left[0]}, x={ostia_left[1]}, z={ostia_left[2]}",
    )
    plot += point_left

    # 4. Óstio direito (ciano)
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

    point_right = k3d.points(
        positions=pos_right,
        point_size=12.0,
        color=0x00FFFF,
        name=f"Óstio Direito\ny={ostia_right[0]}, x={ostia_right[1]}, z={ostia_right[2]}",
    )
    plot += point_right

    plot.display()
    return plot
