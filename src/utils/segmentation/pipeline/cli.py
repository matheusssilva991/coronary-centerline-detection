"""Argumentos de linha de comando para o pipeline de segmentação."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from utils.project.config import (
    LEGACY_LABEL_PROTOCOL,
    load_config_json,
    scale_config_to_resolution,
)
from utils.project.evaluation.artery_distances import (
    add_distance_metric_arguments,
    apply_distance_metric_options,
)
from utils.project.results.artery_masks import add_artery_mask_export_argument
from utils.project.runtime.resource_monitor import (
    add_resource_monitoring_arguments,
    apply_resource_monitoring_options,
)


def _parse_rg_comparison_window(value: str) -> int:
    """Converte ``ALL`` ou um tamanho inteiro para a representação interna.

    Args:
        value (str): Valor textual informado na flag ``--rg-comparison-window``.

    Returns:
        int: ``-1`` para ``ALL``/``-1`` ou inteiro positivo da janela deslizante.
    """
    normalized = value.strip().upper()
    if normalized == "ALL":
        return -1
    try:
        window = int(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "use ALL, -1 ou um número inteiro positivo"
        ) from exc

    if window == 0 or window < -1:
        raise argparse.ArgumentTypeError("use ALL, -1 ou um número inteiro positivo")
    return window


PIPELINE_EPILOG = """
Exemplos de uso:
  # Processar o dataset completo como um único conjunto
  python segmentation_pipeline.py

  # Processar apenas treino
  python segmentation_pipeline.py --split train

  # Processar validação
  python segmentation_pipeline.py --split val

  # Usar OpenCV para downscaling com interpolação AREA
  python segmentation_pipeline.py --downscale-method opencv

  # Usar OpenCV com interpolação LINEAR
  python segmentation_pipeline.py --downscale-method opencv --opencv-interpolation linear

  # Usar resolução alta (sem downscaling)
  python segmentation_pipeline.py --resolution high

  # Usar resolução média (downscale 2x)
  python segmentation_pipeline.py --resolution mid --split val

  # Escolher método de segmentação arterial
  python segmentation_pipeline.py --split val --artery-method fc

  # Comparar candidatos do region growing com a média acumulada da região
  python segmentation_pipeline.py --split train --rg-comparison-window ALL

  # Usar fuzzy threshold
  python segmentation_pipeline.py --split val --threshold-method fuzzy --lower-threshold-percentile 10.5

  # Testar limiar inferior adaptativo mantendo threshold normal + RG
  python segmentation_pipeline.py --split train --threshold-method normal --artery-method rg --lower-threshold-method percentile --lower-threshold-percentile 10.75

  # Sobrescrever apenas o percentil superior do threshold
  python segmentation_pipeline.py --split test --upper-threshold-percentile 99.9

  # Salvar um HTML 3D por imagem com aorta, óstios e artérias
  python segmentation_pipeline.py --split train --save-segmentation-visuals

  # Salvar os HTMLs em um disco externo, mantendo CSVs e logs no repositório
  python segmentation_pipeline.py --split train --save-segmentation-visuals --visual-output-dir /run/media/matheus/HD/Results_dataset_ccta/imagecas

  # PROCESSAMENTO EM LOTES (salvamento incremental):
    # Processar em 10 lotes (divide as imagens entre 10 blocos)
    python segmentation_pipeline.py --num-batches 10

    # Processar teste em 5 lotes
    python segmentation_pipeline.py --split test --num-batches 5

  # RETOMADA DE LOTES (em caso de falha):
  # Primeira execução - cria novo diretório
    python segmentation_pipeline.py --split test --num-batches 70
  # Saída: output/segmentation/runs/mid_res/2026-03-14_10-30-00/

  # Se falhar no lote 3, retomar no MESMO diretório:
    python segmentation_pipeline.py --split test --num-batches 70 --resume-batch 3 --resume-dir output/segmentation/runs/mid_res/2026-03-14_10-30-00

  # Versão curta (se no mesmo diretório):
    python segmentation_pipeline.py --split test --num-batches 70 --resume-batch 3 --resume-dir ./output/segmentation/runs/mid_res/2026-03-14_10-30-00

  # Apenas consolidar lotes já processados, sem reprocessar imagens:
    python segmentation_pipeline.py --merge-only --split test --resume-dir output/segmentation/runs/mid_res/2026-03-14_10-30-00

  # Sobrescrever o caminho do dataset pela CLI:
    python segmentation_pipeline.py --split test --base-path /dados/ImageCAS/1-1000

Arquivos de saída:
  - numeric/results_{split}.csv: Resultados consolidados por imagem
  - numeric/results_{split}_lote_1.csv, numeric/results_{split}_lote_2.csv, etc: Resultados de cada lote
  - numeric/metadata_{split}.json: Identidade e rótulos principais da configuração
  - numeric/batch_timings_{split}.csv: Tempos de cada lote
  - config/effective_pipeline_config.json: Configuração efetiva completa do run
  - config/split_ids.json: IDs processados na coorte selecionada
  - logs/pipeline.log: Log da execução
  - visual/*.html: Visualizações 3D; pode ser redirecionado com --visual-output-dir
"""


def build_parser(default_base_path, default_output_dir):
    """Cria o parser ``argparse`` completo do pipeline de segmentação.

    Args:
        default_base_path (Path | str): Caminho padrão usado como default de
            ``--base-path``.
        default_output_dir (Path | str): Diretório padrão usado como default de
            ``--output-dir``.

    Returns:
        argparse.ArgumentParser: Instância configurada do parser da CLI.
    """
    parser = argparse.ArgumentParser(
        description="Pipeline de segmentação coronária",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=PIPELINE_EPILOG,
    )

    # Dataset, resolução e dispositivo de processamento.
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "full"],
        default="full",
        help=(
            "Conjunto único para processar. 'full' usa todos os exames como "
            "uma única coorte, sem dividi-los em train/val/test (padrão: full)."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=str,
        choices=["mid", "high"],
        default="mid",
        help="Resolução da imagem: 'mid' (downscale 2x) ou 'high' (sem downscale) (padrão: mid)",
    )
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument(
        "--gpu",
        dest="use_gpu",
        action="store_true",
        default=None,
        help="Força uso de GPU nas etapas compatíveis, se disponível.",
    )
    gpu_group.add_argument(
        "--no-gpu",
        dest="use_gpu",
        action="store_false",
        help="Força CPU nas etapas compatíveis, mesmo se houver GPU disponível.",
    )
    # Método arterial e parâmetros geométricos da aorta (level set / Hough).
    parser.add_argument(
        "--artery-segmentation-method",
        "--artery-method",
        choices=["rg", "fc", "region_growing", "fuzzy_connectedness"],
        default=None,
        help=(
            "Método de segmentação arterial: 'rg'/'region_growing' ou "
            "'fc'/'fuzzy_connectedness'."
        ),
    )
    parser.add_argument(
        "--aorta-level-set-iterations",
        type=int,
        default=None,
        help="Número de iterações da evolução nominal do level set da aorta.",
    )
    parser.add_argument(
        "--aorta-level-set-radius-reduction-factor",
        type=float,
        default=None,
        help=(
            "Fração do raio de cada círculo usada para inicializar o level set. "
            "Por exemplo, 0.15 cria sementes com 15%% do raio detectado."
        ),
    )
    parser.add_argument(
        "--aorta-level-set-balloon",
        type=float,
        default=None,
        help=(
            "Força balloon da evolução nominal da aorta. Valores menores "
            "tornam a expansão mais conservadora."
        ),
    )
    parser.add_argument(
        "--aorta-level-set-alpha",
        type=float,
        default=None,
        help=(
            "Sensibilidade do mapa de bordas do level set da aorta. Valores "
            "maiores reforçam a influência das bordas."
        ),
    )
    parser.add_argument(
        "--aorta-opening-radius",
        type=int,
        default=None,
        help=(
            "Raio, em voxels, da abertura morfológica aplicada à máscara da "
            "aorta após o level set. Use 0 para desativar a abertura."
        ),
    )
    parser.add_argument(
        "--aorta-trajectory-radius-factor",
        type=float,
        default=None,
        help=(
            "Restringe a máscara pós-level set ao envelope dos círculos com "
            "raio k*r. Use somente para experimentos; valores sugeridos: "
            "1.75, 2.0 e 2.25."
        ),
    )
    parser.add_argument(
        "--aorta-trajectory-axial-margin-slices",
        type=int,
        default=None,
        help=(
            "Prolonga o envelope da trajetória antes do primeiro e depois do "
            "último círculo. Use com --aorta-trajectory-radius-factor; valor "
            "experimental sugerido: 5 fatias."
        ),
    )
    # Parâmetros da transformada de Hough e filtro da trajetória dos círculos.
    parser.add_argument(
        "--aorta-hough-radii-start-px",
        type=int,
        default=None,
        help="Menor raio, em pixels, incluído na busca circular da aorta.",
    )
    parser.add_argument(
        "--aorta-hough-radii-end-px",
        type=int,
        default=None,
        help=(
            "Limite superior exclusivo dos raios da Hough. Por exemplo, 30 "
            "testa raios até 29 pixels."
        ),
    )
    parser.add_argument(
        "--aorta-circle-filter",
        choices=["none", "robust"],
        default=None,
        help=(
            "Filtro experimental da trajetória da Hough. 'robust' remove uma "
            "cauda geometricamente incompatível e pode extrapolar uma continuação "
            "curta a partir da última região estável; "
            "'none' preserva todos os círculos detectados."
        ),
    )
    parser.add_argument(
        "--aorta-circle-filter-min-coverage",
        type=float,
        default=None,
        help=(
            "Cobertura mínima da trajetória para permitir remoção de cauda. "
            "Use 0.4 para reproduzir o filtro agressivo experimental."
        ),
    )
    parser.add_argument(
        "--aorta-circle-filter-max-trim-fraction",
        type=float,
        default=None,
        help=(
            "Fração máxima da trajetória original que o filtro robusto pode "
            "remover. Use 0.4 para rejeitar cortes axiais maiores que 40%%."
        ),
    )
    parser.add_argument(
        "--aorta-circle-filter-synthetic-tail-slices",
        type=int,
        default=None,
        help=(
            "Número de fatias sintéticas extrapoladas da última região estável "
            "depois que uma cauda incompatível é removida."
        ),
    )
    parser.add_argument(
        "--rg-comparison-window",
        type=_parse_rg_comparison_window,
        default=None,
        help=(
            "Referência de comparação do region growing: 1 compara com o voxel "
            "atual; ALL (ou -1) usa a média acumulada; valores >1 usam a média dos "
            "últimos N voxels aceitos."
        ),
    )
    # Thresholding inicial (normal/fuzzy) e interpolação do downscaling.
    parser.add_argument(
        "--threshold-method",
        choices=["normal", "fuzzy"],
        default=None,
        help=(
            "Threshold inicial: 'normal' usa o piso configurado até percentil; "
            "'fuzzy' mantém voxels cuja maior pertinência é objeto."
        ),
    )
    parser.add_argument(
        "--lower-threshold-method",
        choices=["fixed", "percentile"],
        default=None,
        help=(
            "Método do limiar inferior HU: fixed usa MIN_THRESHOLD; "
            "percentile usa percentil baixo dos voxels válidos."
        ),
    )
    parser.add_argument(
        "--lower-threshold-percentile",
        type=float,
        default=None,
        help="Percentil baixo usado nos métodos adaptativos de limiar inferior.",
    )
    parser.add_argument(
        "--upper-threshold-percentile",
        type=float,
        default=None,
        help="Percentil superior usado pelo threshold normal.",
    )
    parser.add_argument(
        "--lower-threshold-clip-min",
        type=float,
        default=None,
        help="HU mínimo da faixa considerada no cálculo adaptativo do piso.",
    )
    parser.add_argument(
        "--lower-threshold-clip-max",
        type=float,
        default=None,
        help="HU máximo da faixa considerada no cálculo adaptativo do piso.",
    )
    parser.add_argument(
        "--downscale-method",
        type=str,
        choices=["scipy", "opencv"],
        default=None,
        help="Método de downscaling: scipy (ndi.zoom) ou opencv (cv2.resize)",
    )
    parser.add_argument(
        "--opencv-interpolation",
        type=str,
        choices=["nearest", "linear", "cubic", "area", "lanczos4"],
        default=None,
        help="Método de interpolação do OpenCV (usado apenas se --downscale-method=opencv)",
    )
    # Diretórios de saída, retomada em lotes, logs e monitoramento opcional.
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(default_output_dir),
        help=f"Diretório de saída (padrão: {default_output_dir})",
    )
    parser.add_argument(
        "--run-group",
        type=str,
        default=None,
        help=(
            "Subpasta relativa dentro de runs/<resolução>_res para organizar "
            "a execução, por exemplo aorta_segmentation_experiments/train/variant."
        ),
    )
    parser.add_argument(
        "--base-path",
        type=str,
        default=str(default_base_path),
        help=(
            "Diretório base do dataset ImageCAS "
            f"(padrão: {default_base_path}; se indisponível, o pipeline tenta o fallback configurado)"
        ),
    )
    # Configuração externa via JSON, seleção de splits/IDs e retomada em lotes.
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Apenas mescla CSVs de lotes existentes e atualiza metadados. Use com --resume-dir.",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default=None,
        help="Arquivo JSON com configurações para sobrescrever valores padrão",
    )
    parser.add_argument(
        "--config-replace",
        action="store_true",
        help="Usa --config-file como configuração completa, sem mesclar com os padrões atuais.",
    )
    parser.add_argument(
        "--split-config",
        type=str,
        default=None,
        help=(
            "Arquivo JSON alternativo com splits train/val/test. "
            "Útil para testar tamanhos diferentes de treino sem alterar config/imagecas_splits.json."
        ),
    )
    parser.add_argument(
        "--image-ids",
        type=str,
        default=None,
        help=(
            "Processa somente os IDs informados, separados por vírgula, dentro "
            "do split selecionado. Exemplo: --image-ids 44,330,603."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Habilitar logging detalhado (DEBUG)",
    )
    parser.add_argument(
        "--save-segmentation-visuals",
        action="store_true",
        help=(
            "Salva um HTML 3D interativo por imagem com aorta, óstios, "
            "artéria predita e ground truth."
        ),
    )
    parser.add_argument(
        "--visual-output-dir",
        type=str,
        default=None,
        help=(
            "Raiz alternativa para os HTMLs 3D. A estrutura relativa do run "
            "é preservada nessa raiz; CSVs, configs e logs continuam em --output-dir."
        ),
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=5,
        help="Número de lotes para dividir o conjunto de imagens (ex: 5 divide as 700 imagens em 5 lotes)",
    )
    parser.add_argument(
        "--resume-batch",
        type=int,
        default=0,
        help="Número do lote para retomar, em numeração humana/arquivo (ex: 11 retoma salvando lote_11). Padrão: 0 = começar do início.",
    )
    parser.add_argument(
        "--resume-dir",
        type=str,
        default=None,
        help="Diretório anterior para retomar (ex: output/segmentation/runs/mid_res/2026-03-14_10-30-00). Obrigatório quando a retomada começa de um lote > 0.",
    )
    parser.add_argument(
        "--notify",
        action="store_true",
        help="Avisa no desktop e toca um som ao terminar o run.",
    )
    add_resource_monitoring_arguments(parser)
    add_distance_metric_arguments(parser)
    add_artery_mask_export_argument(parser)
    return parser


def parse_pipeline_args(default_base_path, default_output_dir):
    """Faz o parsing, valida faixas e normaliza os argumentos da CLI do pipeline.

    Args:
        default_base_path (Path | str): Caminho padrão repassado ao parser para
            ``--base-path``.
        default_output_dir (Path | str): Diretório padrão repassado ao parser para
            ``--output-dir``.

    Returns:
        argparse.Namespace: Namespace validado com ``image_ids`` deduplicado e
            caminhos expandidos como ``Path``.
    """
    parser = build_parser(default_base_path, default_output_dir)
    args = parser.parse_args()

    # Valida faixas numéricas dos hiperparâmetros antes de inicializar diretórios/logs.
    if args.resume_batch < 0:
        parser.error("--resume-batch deve ser 0 ou maior")
    if args.num_batches <= 0:
        parser.error("--num-batches deve ser maior que 0")
    if (
        args.aorta_trajectory_radius_factor is not None
        and args.aorta_trajectory_radius_factor <= 0
    ):
        parser.error("--aorta-trajectory-radius-factor deve ser maior que 0")
    if (
        args.aorta_level_set_iterations is not None
        and args.aorta_level_set_iterations <= 0
    ):
        parser.error("--aorta-level-set-iterations deve ser maior que 0")
    if (
        args.aorta_level_set_radius_reduction_factor is not None
        and args.aorta_level_set_radius_reduction_factor <= 0
    ):
        parser.error("--aorta-level-set-radius-reduction-factor deve ser maior que 0")
    if args.aorta_level_set_alpha is not None and args.aorta_level_set_alpha <= 0:
        parser.error("--aorta-level-set-alpha deve ser maior que 0")
    if args.aorta_opening_radius is not None and args.aorta_opening_radius < 0:
        parser.error("--aorta-opening-radius deve ser zero ou maior")
    if (
        args.aorta_trajectory_axial_margin_slices is not None
        and args.aorta_trajectory_axial_margin_slices < 0
    ):
        parser.error("--aorta-trajectory-axial-margin-slices deve ser zero ou maior")
    # Converte e deduplica --image-ids mantendo a ordem informada pelo usuário.
    if args.image_ids:
        try:
            args.image_ids = [
                int(value.strip())
                for value in args.image_ids.split(",")
                if value.strip()
            ]
        except ValueError:
            parser.error(
                "--image-ids deve conter somente inteiros separados por vírgula"
            )
        if not args.image_ids:
            parser.error("--image-ids não pode ser vazio")
        args.image_ids = list(dict.fromkeys(args.image_ids))
    # Verifica incompatibilidades entre modos de retomada, consolidação e substituição de config.
    if args.merge_only and not args.resume_dir:
        parser.error("--merge-only requer --resume-dir com a pasta de saída existente")
    if args.config_replace and not args.config_file:
        parser.error("--config-replace requer --config-file")
    if args.merge_only and args.image_ids:
        parser.error("--image-ids não pode ser usado com --merge-only")
    if args.run_group:
        run_group = Path(args.run_group)
        if run_group.is_absolute() or ".." in run_group.parts:
            parser.error("--run-group deve ser um caminho relativo sem '..'")

    if args.split == "full" and args.split_config:
        parser.error("--split-config não se aplica a --split full")

    # Expande caminhos relativos/~ de entrada e saída para objetos Path.
    args.base_path = Path(args.base_path).expanduser()
    args.output_dir = Path(args.output_dir).expanduser()
    args.visual_output_dir = (
        Path(args.visual_output_dir).expanduser() if args.visual_output_dir else None
    )
    args.resume_dir = Path(args.resume_dir).expanduser() if args.resume_dir else None
    args.split_config = (
        Path(args.split_config).expanduser() if args.split_config else None
    )
    args.resume_requested = args.resume_batch > 0

    return args


def select_resolution_config(args, mid_config, high_config):
    """Seleciona a configuração base para mid ou high resolution.

    Args:
        args (argparse.Namespace): Namespace CLI contendo ``resolution`` (``"mid"`` ou
            ``"high"``).
        mid_config (dict[str, Any]): Configuração base para mid resolution
            (com downscale).
        high_config (dict[str, Any]): Dicionário base para high resolution (sem
            downscale).

    Returns:
        dict[str, Any]: Dicionário de configuração correspondente à resolução
            selecionada.
    """
    if args.resolution == "high":
        print(
            "🔍 Resolução: HIGH "
            "(sem downscaling, DOWNSCALE_FACTORS = "
            f"{high_config['DOWNSCALE_FACTORS']})"
        )
        return high_config

    print(
        "🔍 Resolução: MID "
        f"(downscale 2x, DOWNSCALE_FACTORS = {mid_config['DOWNSCALE_FACTORS']})"
    )
    return mid_config


def _apply_execution_overrides(config, args):
    """Aplica opções operacionais da CLI à configuração efetiva.

    Args:
        config (dict[str, Any]): Dicionário de configuração atualizado in-place.
        args (argparse.Namespace): Argumentos já validados da linha de comando.
    """
    apply_resource_monitoring_options(config, args)
    apply_distance_metric_options(config, args)
    # Configurações antigas podem conter opções de cache já removidas.
    config.pop("SAVE_CACHE", None)
    config.pop("LOAD_CACHE", None)

    # Overrides operacionais e flags de visualização 3D.
    direct_overrides = {
        "DOWNSCALE_METHOD": args.downscale_method,
        "OPENCV_INTERPOLATION": args.opencv_interpolation,
        "USE_GPU": args.use_gpu,
    }
    for config_key, value in direct_overrides.items():
        if value is not None:
            config[config_key] = value
    save_visuals = bool(getattr(args, "save_segmentation_visuals", False))
    if (
        not getattr(args, "config_replace", False)
        or "SAVE_SEGMENTATION_VISUALS" in config
        or save_visuals
    ):
        config["SAVE_SEGMENTATION_VISUALS"] = save_visuals
    visual_output_dir = getattr(args, "visual_output_dir", None)
    if (
        not getattr(args, "config_replace", False)
        or "VISUAL_OUTPUT_DIR" in config
        or visual_output_dir is not None
    ):
        config["VISUAL_OUTPUT_DIR"] = (
            Path(visual_output_dir).as_posix()
            if visual_output_dir is not None
            else None
        )

    # Overrides da segmentação arterial e do level set da aorta.
    nested_overrides = (
        ("ARTERY_SEGMENTATION", "method", args.artery_segmentation_method),
        ("REGION_GROWING", "comparison_window", args.rg_comparison_window),
    )
    for section, config_key, value in nested_overrides:
        if value is not None:
            config.setdefault(section, {})[config_key] = value

    # Sobrescreve os hiperparâmetros do level set da aorta quando informados na CLI.
    level_set_config = config.setdefault("LEVEL_SET", {})
    level_set_overrides = (
        ("aorta_level_set_iterations", "num_iter", None),
        ("aorta_level_set_radius_reduction_factor", "radius_reduction_factor", None),
        ("aorta_level_set_balloon", "balloon", None),
        ("aorta_level_set_alpha", "alpha", None),
        ("aorta_opening_radius", "leak_removal_radius", None),
        ("aorta_trajectory_radius_factor", "trajectory_radius_factor", float),
        ("aorta_trajectory_axial_margin_slices", "trajectory_axial_margin_slices", int),
    )
    for arg_name, config_key, converter in level_set_overrides:
        value = getattr(args, arg_name, None)
        if value is not None:
            level_set_config[config_key] = converter(value) if converter else value

    # Overrides da detecção por Hough e do filtro de trajetória dos círculos.
    circle_detection_config = config.setdefault("CIRCLE_DETECTION", {})
    for arg_name, config_key in (
        ("aorta_hough_radii_start_px", "radii_start_px"),
        ("aorta_hough_radii_end_px", "radii_end_px"),
    ):
        value = getattr(args, arg_name, None)
        if value is not None:
            circle_detection_config[config_key] = int(value)
    if (
        "radii_start_px" in circle_detection_config
        and "radii_end_px" in circle_detection_config
        and float(circle_detection_config["radii_start_px"])
        >= float(circle_detection_config["radii_end_px"])
    ):
        raise ValueError(
            "CIRCLE_DETECTION.radii_start_px deve ser menor que radii_end_px"
        )

    # Atualiza parâmetros do filtro mantendo compatibilidade com configs antigas.
    circle_filter_config = circle_detection_config.get("trajectory_filter") or {}
    circle_filter_overrides = (
        ("aorta_circle_filter", "method", None),
        ("aorta_circle_filter_min_coverage", "min_tail_coverage", None),
        ("aorta_circle_filter_max_trim_fraction", "max_tail_trim_fraction", float),
        ("aorta_circle_filter_synthetic_tail_slices", "synthetic_tail_slices", int),
    )
    for arg_name, config_key, converter in circle_filter_overrides:
        value = getattr(args, arg_name, None)
        if value is not None:
            circle_filter_config[config_key] = converter(value) if converter else value
    if circle_filter_config:
        circle_detection_config["trajectory_filter"] = circle_filter_config


def _apply_threshold_overrides(config, args):
    """Aplica opções de threshold normal ou fuzzy e piso inferior.

    Args:
        config (dict[str, Any]): Dicionário de configuração atualizado in-place.
        args (argparse.Namespace): Argumentos validados contendo opções de
            limiarização.
    """
    # Atualiza o método de threshold inicial e o percentil superior quando fornecidos.
    thresholding_config = config.setdefault("THRESHOLDING", {})
    if args.threshold_method is not None:
        thresholding_config["method"] = args.threshold_method
    if args.upper_threshold_percentile is not None:
        config["MAX_THRESHOLD_PERCENTILE"] = args.upper_threshold_percentile

    # Aplica sobrescritas do limiar inferior e sincroniza o percentil no modo fuzzy.
    lower_threshold_config = config.setdefault("LOWER_THRESHOLD", {})
    lower_overrides = {
        "method": args.lower_threshold_method,
        "percentile": args.lower_threshold_percentile,
        "clip_min_hu": args.lower_threshold_clip_min,
        "clip_max_hu": args.lower_threshold_clip_max,
    }
    for config_key, value in lower_overrides.items():
        if value is not None:
            lower_threshold_config[config_key] = value

    if (
        args.lower_threshold_percentile is not None
        and thresholding_config.get("method") == "fuzzy"
    ):
        thresholding_config.setdefault("fuzzy", {})["lower_percentile"] = (
            args.lower_threshold_percentile
        )


def build_effective_config(args, mid_config, high_config):
    """Monta e escala a configuração efetiva para a resolução selecionada.

    Args:
        args (argparse.Namespace): Namespace CLI validado contendo flags de resolução,
            sobrescritas e retomada.
        mid_config (dict[str, Any]): Configuração base para mid resolution.
        high_config (dict[str, Any]): Configuração base para high resolution.

    Returns:
        dict[str, Any]: Dicionário de configuração final normalizado e já escalado
            para a resolução escolhida.
    """
    # Parte da configuração da resolução selecionada e aplica eventual JSON externo.
    effective_config = copy.deepcopy(
        select_resolution_config(args, mid_config, high_config)
    )

    if args.config_file:
        base_config = {} if getattr(args, "config_replace", False) else effective_config
        effective_config = load_config_json(args.config_file, base_config)
        print(f"⚙️  Configuração carregada de: {args.config_file}")

    # Preserva o protocolo dos lotes anteriores, inclusive em snapshots históricos.
    resume_dir = getattr(args, "resume_dir", None)
    requested_mask_export = getattr(args, "save_artery_masks", None)
    resuming = bool(resume_dir) and (
        getattr(args, "resume_requested", False) or getattr(args, "merge_only", False)
    )
    saved_mask_export = False
    merge_protocol = (
        {"version": "unrecorded"} if getattr(args, "merge_only", False) else None
    )
    if resume_dir:
        resume_root = Path(resume_dir)
        if resume_root.name == "numeric":
            resume_root = resume_root.parent
        snapshot_path = resume_root / "config/effective_pipeline_config.json"
        if snapshot_path.is_file():
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
            saved_mask_export = bool(snapshot.get("SAVE_ARTERY_MASKS", False))
            effective_config["LABEL_RESAMPLING"] = snapshot.get(
                "LABEL_RESAMPLING", LEGACY_LABEL_PROTOCOL
            )
            if getattr(args, "merge_only", False):
                merge_protocol = (snapshot.get("EVALUATION") or {}).get(
                    "distance_metrics_protocol", {"version": "unrecorded"}
                )
            if not args.config_file:
                effective_config["EVALUATION"] = copy.deepcopy(
                    snapshot.get("EVALUATION") or {"distance_metrics": False}
                )
        elif getattr(args, "resume_requested", False):
            effective_config["LABEL_RESAMPLING"] = LEGACY_LABEL_PROTOCOL

    if (
        resuming
        and requested_mask_export is not None
        and requested_mask_export != saved_mask_export
    ):
        raise ValueError(
            "--save-artery-masks diverge da opção salva. Crie um novo run para "
            "alterar a exportação arterial."
        )
    mask_export_enabled = saved_mask_export if resuming else bool(requested_mask_export)
    if mask_export_enabled:
        effective_config["SAVE_ARTERY_MASKS"] = True
    else:
        effective_config.pop("SAVE_ARTERY_MASKS", None)

    # A resolução escolhida na CLI prevalece sobre fatores salvos em snapshots
    # de configuração produzidos originalmente para mid resolution.
    if args.resolution == "high":
        effective_config["DOWNSCALE_FACTORS"] = [1, 1, 1]

    # Aplica flags da CLI por último e escala parâmetros espaciais para a resolução.
    _apply_execution_overrides(effective_config, args)
    if merge_protocol is not None:
        effective_config.setdefault("EVALUATION", {})["distance_metrics_protocol"] = (
            merge_protocol
        )
    _apply_threshold_overrides(effective_config, args)

    effective_config["NUM_BATCHES"] = args.num_batches
    return scale_config_to_resolution(effective_config)
