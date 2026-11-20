"""Visualização interativa 3D em K3D de máscaras, óstios e mapas de rótulos."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional, Sequence, cast

import k3d
import numpy as np
from k3d.factory import plot as create_plot
from numpy.typing import NDArray
from skimage import measure


def visualize_3d_k3d(
    mask_3d: NDArray[Any],
    spacing: Sequence[float] = (1, 1, 1),
    color: int = 0xFF0000,
    opacity: float = 0.5,
    use_physical_coords: bool = True,
    save_html_path: Optional[str] = None,
    display_plot: bool = True,
) -> Any:
    """Renderiza uma máscara 3D em K3D usando o algoritmo marching cubes.

    Args:
        mask_3d (NDArray[Any]): Volume binário 3D ``(Y, X, Z)`` a renderizar.
        spacing (Sequence[float]): Espaçamento dos voxels ``(dy, dx, dz)`` em
            mm usado quando ``use_physical_coords`` é ``True``.
        color (int): Cor hexadecimal RGB da malha 3D (por exemplo, ``0xFF0000``).
        opacity (float): Opacidade da malha no intervalo ``[0.0, 1.0]``.
        use_physical_coords (bool): Se ``True``, escala a cena em milímetros
            usando ``spacing``; caso contrário usa coordenadas em pixels.
        save_html_path (str | None): Caminho opcional para salvar o snapshot
            HTML; ``None`` não grava arquivo em disco.
        display_plot (bool): Se ``True``, exibe o widget interativo do K3D.

    Returns:
        Any: Objeto ``k3d.Plot`` contendo a malha gerada.
    """
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

    # Adiciona malha, exporta HTML opcional e renderiza.
    plot += mesh
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
    smoothing_iterations: int = 0,
) -> None:
    """Adiciona a malha de uma máscara 3D ao plot K3D quando ela contém voxels.

    Args:
        plot (Any): Instância ``k3d.Plot`` que receberá a nova malha.
        mask (NDArray[Any]): Volume 3D cujos voxels positivos definem a
            superfície.
        spacing (Sequence[float]): Espaçamento entre voxels aplicado na extração
            da malha.
        color (int): Cor hexadecimal RGB da malha.
        opacity (float): Opacidade da superfície no intervalo ``[0.0, 1.0]``.
        name (str): Nome exibido na legenda de objetos da cena K3D.
        smoothing_iterations (int): Número de passos de suavização Laplaciana
            aplicados aos vértices; ``0`` preserva a malha bruta.
    """
    # Converte para máscara booleana e evita gerar malha para volumes vazios.
    mask_bool = np.asarray(mask) > 0
    if not np.any(mask_bool):
        print(f"Máscara vazia ignorada: {name}")
        return

    # Extrai a superfície isovalor com marching cubes no espaçamento informado.
    verts, faces, _, _ = measure.marching_cubes(
        mask_bool.astype(float), level=0.5, spacing=spacing
    )
    # Suaviza somente os vértices; a máscara de entrada permanece intacta.
    if smoothing_iterations:
        edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
        edges = np.unique(np.sort(edges, axis=1), axis=0)
        source = np.concatenate((edges[:, 0], edges[:, 1]))
        target = np.concatenate((edges[:, 1], edges[:, 0]))
        degree = np.bincount(source, minlength=len(verts))
        for _ in range(smoothing_iterations):
            sums = np.zeros_like(verts)
            np.add.at(sums, source, verts[target])
            means = sums / np.maximum(degree[:, None], 1)
            verts = verts + 0.15 * (means - verts)
    # Adiciona a malha triangulada à cena K3D.
    plot += k3d.mesh(
        verts.astype(np.float32),
        faces.astype(np.uint32),
        color=color,
        opacity=opacity,
        name=name,
    )


def visualize_label_map_3d(
    label_map: NDArray[Any],
    label_specs: Mapping[int, tuple[str, int]],
    *,
    spacing: Sequence[float] = (1, 1, 1),
    opacity: float = 0.72,
    use_physical_coords: bool = True,
    save_html_path: str | Path | None = None,
    display_plot: bool = True,
    plot_name: str = "Estruturas rotuladas",
) -> Any:
    """Renderiza em uma cena K3D os rótulos presentes em um mapa 3D.

    Args:
        label_map (NDArray[Any]): Volume 3D de rótulos inteiros por voxel.
        label_specs (Mapping[int, tuple[str, int]]): Mapeamento de código de
            rótulo para ``(nome, cor_hex)``.
        spacing (Sequence[float]): Espaçamento dos voxels em mm usado quando
            ``use_physical_coords`` é ``True``.
        opacity (float): Opacidade compartilhada por todas as malhas no
            intervalo ``[0.0, 1.0]``.
        use_physical_coords (bool): Se ``True``, aplica ``spacing`` em
            milímetros; caso contrário usa coordenadas unitárias em pixels.
        save_html_path (str | Path | None): Caminho opcional para salvar o
            gráfico em HTML; ``None`` ignora a exportação.
        display_plot (bool): Se ``True``, exibe a cena K3D após montar as
            malhas.
        plot_name (str): Título identificador da cena K3D.

    Returns:
        Any: Objeto ``k3d.Plot`` populado com as malhas dos rótulos encontrados.
    """
    # Valida dimensionalidade do volume, espaçamento positivo e faixa de opacidade.
    label_array = np.asarray(label_map)
    if label_array.ndim != 3:
        raise ValueError("O mapa de rótulos deve ser tridimensional.")

    spacing_values = np.asarray(tuple(spacing), dtype=float)
    if spacing_values.shape != (3,):
        raise ValueError("O espaçamento deve conter exatamente três valores.")
    if not np.all(np.isfinite(spacing_values)) or np.any(spacing_values <= 0):
        raise ValueError("O espaçamento deve conter valores finitos e positivos.")

    resolved_opacity = float(opacity)
    if not np.isfinite(resolved_opacity) or not 0 <= resolved_opacity <= 1:
        raise ValueError("A opacidade deve ser um valor finito entre 0 e 1.")

    # Filtra apenas os rótulos configurados que efetivamente ocorrem no volume.
    present_specs = [
        (int(label_value), name, int(color))
        for label_value, (name, color) in label_specs.items()
        if np.any(label_array == label_value)
    ]
    if not present_specs:
        raise ValueError("Nenhum dos rótulos selecionados está presente no volume.")

    # Define escala física ou em pixels e cria a cena K3D compartilhada.
    if use_physical_coords:
        mesh_spacing = tuple(float(value) for value in spacing_values)
        axes_labels = ["X (mm)", "Y (mm)", "Z (mm)"]
    else:
        mesh_spacing = (1.0, 1.0, 1.0)
        axes_labels = ["X (pixels)", "Y (pixels)", "Z (pixels)"]

    plot = cast(
        Any,
        create_plot(
            name=plot_name,
            height=800,
            grid_visible=True,
            axes=axes_labels,
        ),
    )
    # Gera e anexa uma malha separada para cada rótulo presente.
    for label_value, name, color in present_specs:
        _add_mask_mesh(
            plot,
            label_array == label_value,
            spacing=mesh_spacing,
            color=color,
            opacity=resolved_opacity,
            name=name,
        )

    # Exporta snapshot HTML e exibe o plot conforme parâmetros.
    if save_html_path is not None:
        save_k3d_plot_html(plot, save_html_path)
    if display_plot:
        plot.display()
    return plot


def _add_ostium_point(
    plot: Any,
    ostium: Optional[Sequence[float]],
    *,
    spacing: Sequence[float],
    color: int,
    name: str,
) -> None:
    """Adiciona um marcador pontual de óstio ao plot K3D quando informado.

    Args:
        plot (Any): Instância ``k3d.Plot`` onde o ponto será inserido.
        ostium (Sequence[float] | None): Coordenada ``(y, x, z)`` em voxels do
            óstio; ``None`` não adiciona ponto à cena.
        spacing (Sequence[float]): Fatores de escala ``(dy, dx, dz)`` aplicados
            à coordenada do óstio.
        color (int): Cor hexadecimal RGB do marcador.
        name (str): Prefixo do nome exibido no painel de objetos do K3D.
    """
    if ostium is None:
        return

    # Converte a coordenada do óstio para o sistema da cena e adiciona marcador.
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
    """Renderiza aorta, óstios, artéria predita e referência em um mesmo plot 3D.

    Args:
        aorta_mask (NDArray[Any]): Máscara binária 3D ``(Y, X, Z)`` da aorta.
        ostia_left (Sequence[float] | None): Coordenada ``(y, x, z)`` do óstio
            esquerdo em voxels, ou ``None`` se ausente.
        ostia_right (Sequence[float] | None): Coordenada ``(y, x, z)`` do óstio
            direito em voxels, ou ``None`` se ausente.
        artery_mask (NDArray[Any] | None): Máscara 3D predita das artérias
            coronárias; ``None`` omite a camada predita.
        label_artery (NDArray[Any] | None): Máscara 3D de referência
            (ground truth) das artérias; ``None`` omite a camada de referência.
        spacing (Sequence[float]): Espaçamento dos voxels ``(dy, dx, dz)`` em
            mm.
        use_physical_coords (bool): Se ``True``, renderiza em milímetros; caso
            contrário usa coordenadas em pixels.
        save_html_path (str | None): Caminho opcional para gravar a cena como
            HTML.
        display_plot (bool): Se ``True``, exibe o gráfico K3D ao final.
        plot_name (str): Nome atribuído à cena K3D.

    Returns:
        Any: Objeto ``k3d.Plot`` com as camadas anatômicas adicionadas.
    """
    # Seleciona escala física (mm) ou em pixels e inicializa a cena K3D.
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
    # Adiciona malhas da aorta e, quando disponíveis, referência e predição arterial.
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

    # Sobrepõe os marcadores dos óstios esquerdo e direito.
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

    # Salva o snapshot HTML e exibe a cena interativa se solicitado.
    if save_html_path is not None:
        with open(save_html_path, "w", encoding="utf-8") as html_file:
            html_file.write(plot.get_snapshot())

    if display_plot:
        plot.display()
    return plot


def visualize_binary_masks_comparison(
    reference_mask: NDArray[Any],
    predicted_mask: NDArray[Any],
    spacing: Sequence[float] = (1, 1, 1),
    use_physical_coords: bool = True,
    save_html_path: str | Path | None = None,
    display_plot: bool = True,
    plot_name: str = "Referência vs predição",
    reference_name: str = "Referência",
    predicted_name: str = "Predita",
    reference_color: int = 0x00FF00,
    predicted_color: int = 0xFF0000,
    reference_opacity: float = 0.35,
    predicted_opacity: float = 0.35,
    mesh_smoothing_iterations: int = 0,
    camera: Sequence[float] | None = None,
) -> Any:
    """Renderiza duas máscaras binárias comparáveis no mesmo gráfico 3D.

    Args:
        reference_mask (NDArray[Any]): Máscara binária 3D ``(Y, X, Z)`` de
            referência.
        predicted_mask (NDArray[Any]): Máscara binária 3D ``(Y, X, Z)`` predita,
            com o mesmo shape de ``reference_mask``.
        spacing (Sequence[float]): Espaçamento dos voxels ``(dy, dx, dz)`` em
            mm.
        use_physical_coords (bool): Se ``True``, escala as malhas em
            milímetros; caso contrário usa coordenadas em pixels.
        save_html_path (str | Path | None): Caminho opcional para exportar a
            cena em HTML; ``None`` não grava arquivo.
        display_plot (bool): Se ``True``, exibe o plot K3D interativo.
        plot_name (str): Título identificador da cena K3D.
        reference_name (str): Rótulo da máscara de referência no painel do K3D.
        predicted_name (str): Rótulo da máscara predita no painel do K3D.
        reference_color (int): Cor hexadecimal RGB da malha de referência.
        predicted_color (int): Cor hexadecimal RGB da malha predita.
        reference_opacity (float): Opacidade da malha de referência em
            ``[0.0, 1.0]``.
        predicted_opacity (float): Opacidade da malha predita em ``[0.0, 1.0]``.
        mesh_smoothing_iterations (int): Número de iterações de suavização
            aplicadas às superfícies; ``0`` mantém a malha sem suavização.
        camera (Sequence[float] | None): Vetor de 9 valores finitos para
            posicionar a câmera K3D; ``None`` mantém ajuste automático.

    Returns:
        Any: Objeto ``k3d.Plot`` contendo ambas as máscaras sobrepostas.
    """
    # Valida suavização, câmera e consistência geométrica entre as duas máscaras.
    if (
        mesh_smoothing_iterations < 0
        or int(mesh_smoothing_iterations) != mesh_smoothing_iterations
    ):
        raise ValueError("mesh_smoothing_iterations deve ser um inteiro não negativo.")
    if camera is not None and (len(camera) != 9 or not np.isfinite(camera).all()):
        raise ValueError("camera deve conter nove valores finitos.")
    reference = np.asarray(reference_mask) > 0
    prediction = np.asarray(predicted_mask) > 0
    if reference.ndim != 3 or prediction.ndim != 3:
        raise ValueError("As máscaras de comparação devem ser tridimensionais.")
    if reference.shape != prediction.shape:
        raise ValueError("As máscaras de comparação devem possuir o mesmo shape.")
    if not np.any(reference) or not np.any(prediction):
        raise ValueError("As máscaras de comparação não podem estar vazias.")

    # Define escala em milímetros ou pixels e cria a cena com câmera opcional.
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

    if camera is not None:
        plot.camera_auto_fit = False
        plot.camera = list(camera)

    # Constrói e adiciona as malhas de referência e predição à cena.
    _add_mask_mesh(
        plot,
        reference,
        spacing=(dy, dx, dz),
        color=reference_color,
        opacity=reference_opacity,
        name=reference_name,
        smoothing_iterations=int(mesh_smoothing_iterations),
    )
    _add_mask_mesh(
        plot,
        prediction,
        spacing=(dy, dx, dz),
        color=predicted_color,
        opacity=predicted_opacity,
        name=predicted_name,
        smoothing_iterations=int(mesh_smoothing_iterations),
    )

    # Exporta o HTML interativo e exibe a cena quando configurado.
    if save_html_path is not None:
        save_k3d_plot_html(plot, save_html_path)

    if display_plot:
        plot.display()
    return plot


def save_k3d_plot_html(plot: Any, html_path: str | Path) -> None:
    """Salva um gráfico K3D como arquivo HTML independente.

    Args:
        plot (Any): Instância ``k3d.Plot`` cujo snapshot HTML será exportado.
        html_path (str | Path): Caminho de destino do arquivo HTML.
    """
    output_path = Path(html_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(plot.get_snapshot(), encoding="utf-8")
    print(f"HTML salvo em: {output_path}")
